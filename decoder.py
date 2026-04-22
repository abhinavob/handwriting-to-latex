"""
Transformer Decoder for Handwriting-to-LaTeX pipeline.
Person 2 — Transformer Architecture Lead

Receives:
    From P1 (Encoder):  encoder_output (B, src_len, 512), source_lengths (B,)
    From P4 (Data):     vocab.json, DataLoader yielding (image_tensor, token_ids)

Exposes to P3 (Beam Search):
    model            — this module
    decode_step()    — one-token-at-a-time inference interface
    attention_weights returned from decode_step for confidence/ambiguity use
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Custom Decoder Layer — exposes cross-attention weights
#     PyTorch's nn.Transformer  Layer does NOT return attention weights,
#     so we build our own thin wrapper around MultiheadAttention.
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Single transformer decoder block with accessible attention weights."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Masked self-attention (decoder attends to its own past tokens)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (decoder queries the encoder feature map)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        # Position-wise feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,             # (B, tgt_len, d_model)
        encoder_output: torch.Tensor,   # (B, src_len, d_model)
        tgt_mask: torch.Tensor = None,           # (tgt_len, tgt_len) causal mask
        tgt_key_padding_mask: torch.Tensor = None,  # (B, tgt_len)
        memory_key_padding_mask: torch.Tensor = None # (B, src_len)
    ):
        """
        Returns
        -------
        out              : (B, tgt_len, d_model)
        cross_attn_weights : (B, tgt_len, src_len)  — what P3 needs
        """
        # --- masked self-attention ---
        residual = tgt
        x, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(residual + self.dropout(x))

        # --- cross-attention over encoder feature map ---
        residual = x
        x, cross_attn_weights = self.cross_attn(
            query=x,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,               # <-- this is the key line
            average_attn_weights=True,        # average across heads → (B, tgt_len, src_len)
        )
        x = self.norm2(residual + self.dropout(x))

        # --- feed-forward ---
        residual = x
        x = self.norm3(residual + self.dropout(self.ff(x)))

        return x, cross_attn_weights


# ---------------------------------------------------------------------------
# 2.  Full Decoder Model
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """
    Full decoder: token embeddings → N × DecoderLayer → linear → logits.

    Parameters
    ----------
    vocab_size   : int   — number of tokens (from vocab.json)
    d_model      : int   — must match encoder output dim (512 per spec)
    n_heads      : int   — number of attention heads
    n_layers     : int   — number of stacked decoder blocks
    d_ff         : int   — hidden size of feed-forward sub-layer
    max_seq_len  : int   — maximum LaTeX output sequence length
    dropout      : float
    pad_id       : int   — token ID for <pad> (used to build padding masks)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding + learned 1-D positional embedding
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Decoder layer stack
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final projection to vocabulary logits
        self.output_proj = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)  # embedding scale factor

        self._init_weights()

    # ---- weight initialisation (Xavier) ----
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ---- mask helpers ----

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask → future positions are -inf."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def _make_pad_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """True where token == PAD. Shape (B, seq_len)."""
        return ids == self.pad_id

    @staticmethod
    def _make_memory_pad_mask(source_lengths: torch.Tensor, max_src_len: int) -> torch.Tensor:
        """
        True where encoder position is padding.
        source_lengths: (B,) — number of valid encoder positions per sample.
        """
        batch_size = source_lengths.size(0)
        # (B, max_src_len): True at positions >= valid length
        indices = torch.arange(max_src_len, device=source_lengths.device).unsqueeze(0)
        return indices >= source_lengths.unsqueeze(1)

    # ---- full-sequence forward (training with teacher forcing) ----

    def forward(
        self,
        tgt_ids: torch.Tensor,          # (B, tgt_len) — ground-truth token IDs
        encoder_output: torch.Tensor,    # (B, src_len, d_model)
        source_lengths: torch.Tensor,    # (B,)
    ):
        """
        Teacher-forced forward pass for training.

        Returns
        -------
        logits : (B, tgt_len, vocab_size)
        """
        B, tgt_len = tgt_ids.shape
        device = tgt_ids.device

        # Embeddings
        positions = torch.arange(tgt_len, device=device).unsqueeze(0)  # (1, tgt_len)
        x = self.dropout(
            self.token_emb(tgt_ids) * self.scale + self.pos_emb(positions)
        )

        # Masks
        causal_mask = self._make_causal_mask(tgt_len, device)               # (tgt_len, tgt_len)
        tgt_pad_mask = self._make_pad_mask(tgt_ids)                          # (B, tgt_len)
        mem_pad_mask = self._make_memory_pad_mask(source_lengths,
                                                  encoder_output.size(1))    # (B, src_len)

        # Run through all decoder layers (we don't need attn weights at train time)
        for layer in self.layers:
            x, _ = layer(x, encoder_output, causal_mask, tgt_pad_mask, mem_pad_mask)

        logits = self.output_proj(x)   # (B, tgt_len, vocab_size)
        return logits

    # ---- single-step decode (inference — what P3 calls) ----

    def decode_step(
        self,
        ys: torch.Tensor,              # (B, step) — tokens generated so far (starts with SOS)
        encoder_output: torch.Tensor,   # (B, src_len, d_model)
        source_lengths: torch.Tensor,   # (B,)
    ):
        """
        Decode one step for beam search / greedy inference.

        Parameters
        ----------
        ys              : (B, step) token IDs generated so far (first token is SOS)
        encoder_output  : (B, src_len, d_model) from P1 encoder
        source_lengths  : (B,) valid lengths in encoder output

        Returns
        -------
        logits          : (B, vocab_size)  — scores for the NEXT token only
        attn_weights    : (B, src_len)     — cross-attention over encoder for this step
                          (P3 uses this for confidence flagging)
        """
        B, step = ys.shape
        device = ys.device

        # Embeddings for all tokens so far
        positions = torch.arange(step, device=device).unsqueeze(0)
        x = self.dropout(
            self.token_emb(ys) * self.scale + self.pos_emb(positions)
        )

        # Masks
        causal_mask = self._make_causal_mask(step, device)
        mem_pad_mask = self._make_memory_pad_mask(source_lengths,
                                                  encoder_output.size(1))

        # Run through layers, keep last layer's cross-attention weights
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(x, encoder_output, causal_mask, None, mem_pad_mask)

        # Only the last time-step matters for next-token prediction
        logits = self.output_proj(x[:, -1, :])   # (B, vocab_size)
        attn_weights = attn_weights[:, -1, :]     # (B, src_len) — last step's attention

        return logits, attn_weights


# ---------------------------------------------------------------------------
# 3.  Helper: load vocab and build model
# ---------------------------------------------------------------------------

def build_decoder(vocab_path: str = "vocab.json", device: str = "cpu", **kwargs):
    """
    Convenience factory that reads vocab.json and returns a ready model.

    vocab.json expected format:
        { "<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3, "\\alpha": 4, ... }
    """
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    pad_id = vocab.get("<pad>", 0)
    vocab_size = len(vocab)

    model = TransformerDecoder(
        vocab_size=vocab_size,
        pad_id=pad_id,
        **kwargs,
    ).to(device)

    print(f"[Decoder] vocab_size={vocab_size}  pad_id={pad_id}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return model, vocab


class DecoderInterface:
    """
    Convenience wrapper for beam search.

    IMPORTANT — SOS convention:
        The `step(tokens)` method expects `tokens` to be the tokens generated
        AFTER SOS. It prepends SOS internally. So at the start of decoding,
        pass an empty list `[]` to get the distribution over the first token.

    Vocab constants (exposed as attributes):
        sos, eos, pad, unk
        open_brace_id, close_brace_id  (useful for P3's bracket-balance check)
        source_len  (49 for 7x7 ResNet18 feature map)
    """

    def __init__(self, vocab_path: str = "vocab.json", device: str = None, source_len: int = 49):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model, self.vocab = build_decoder(vocab_path, device=device)
        self.model.eval()

        # Vocab constants
        self.sos = self.vocab["<start>"]
        self.eos = self.vocab["<end>"]
        self.pad = self.vocab["<pad>"]
        self.unk = self.vocab.get("<unk>", None)
        self.open_brace_id = self.vocab.get("{", None)
        self.close_brace_id = self.vocab.get("}", None)
        self.source_len = source_len

        # Per-image state
        self._enc_out = None
        self._src_lens = None
        self._last_attn = None

    def load_weights(self, checkpoint_path: str):
        """Load trained decoder weights."""
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def set_encoder_output(self, enc_out: torch.Tensor, src_lengths: torch.Tensor):
        """Call once per image, before running beam search."""
        self._enc_out = enc_out.to(self.device)
        self._src_lens = src_lengths.to(self.device)

    def step(self, tokens_after_sos):
        """
        Run one decode step.

        Parameters
        ----------
        tokens_after_sos : list[int]
            Tokens generated so far, NOT including SOS. Can be [] for the first step.

        Returns
        -------
        log_probs : torch.Tensor of shape (vocab_size,)
            Log-probabilities over the vocabulary for the next token.
        """
        if self._enc_out is None:
            raise RuntimeError("Call set_encoder_output() before step().")

        ys = torch.tensor([[self.sos] + list(tokens_after_sos)], device=self.device)
        with torch.no_grad():
            logits, attn = self.model.decode_step(ys, self._enc_out, self._src_lens)

        self._last_attn = attn[0].detach().cpu()          # (src_len,)
        return torch.log_softmax(logits[0], dim=-1)       # (vocab_size,)

    def last_attention(self):
        """Attention weights from the most recent step() call. Shape: (src_len,)."""
        return self._last_attn