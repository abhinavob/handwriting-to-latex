import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import difflib
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


from dataset import HMEDataset
from tokenizer import Tokenizer
from dataloader import collate_fn
from encoder import Encoder
from beam_search import beam_decode

from decoder import TransformerDecoder, build_decoder


# ------------------------------------------------------------
# TRAIN ONE EPOCH
# ------------------------------------------------------------

def train_one_epoch(
    model,
    encoder,
    dataloader,
    optimizer,
    criterion,
    device,
    grad_clip=1.0,
    pad_id=0,
):
    model.train()
    #changes 1 - pranava
    encoder.train()  # encoder is trainable

    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (images, token_ids) in enumerate(dataloader):

        images = images.to(device)
        token_ids = token_ids.to(device)

        # ---- RUN ENCODER HERE ----
        #with torch.no_grad(): removed as per gemini's orders
        encoder_output, source_lengths = encoder(images)

        # ---- TEACHER FORCING ----
        tgt_input = token_ids[:, :-1]
        tgt_target = token_ids[:, 1:]

        logits = model(tgt_input, encoder_output, source_lengths)

        logits_flat = logits.reshape(-1, model.vocab_size)
        target_flat = tgt_target.reshape(-1)

        loss = criterion(logits_flat, target_flat)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        non_pad = (tgt_target != pad_id).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

        if (batch_idx + 1) % 50 == 0:
            avg = total_loss / total_tokens if total_tokens > 0 else 0
            print(f"batch {batch_idx+1} | loss: {avg:.4f}")

    return total_loss / total_tokens if total_tokens > 0 else 0


# ------------------------------------------------------------
# GREEDY DECODE (uses decode_step — matches what P3's beam search calls)
# ------------------------------------------------------------

def greedy_decode(model, encoder, image, tokenizer, device, max_len=150):
    model.eval()
    encoder.eval()

    image = image.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        encoder_output, source_lengths = encoder(image)

    start_id = tokenizer.token_to_id["<start>"]
    end_id = tokenizer.token_to_id["<end>"]

    generated = torch.tensor([[start_id]], device=device)

    for _ in range(max_len):
        with torch.no_grad():
            # Use decode_step — same interface P3's beam search uses.
            # Returns logits for the next token only + attention weights.
            logits, _ = model.decode_step(generated, encoder_output, source_lengths)

        next_token = logits.argmax(dim=-1)  # [1]

        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        if next_token.item() == end_id:
            break

    token_ids = generated.squeeze(0).tolist()
    token_ids = token_ids[1:]  # strip <start>

    if end_id in token_ids:
        token_ids = token_ids[:token_ids.index(end_id)]

    tokens = [tokenizer.id_to_token[i] for i in token_ids]
    return " ".join(tokens)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data_size", type=int, default=100,
                        help="Number of samples to actually train on. Keep small for fast iteration.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    device = args.device

    # ---- PATHS ----
    DATA_ROOT = "../hme100k"
    TRAIN_FILE = "../hme100k/train.txt"
    VOCAB_PATH = "vocab.json"

    # ---- TOKENIZER: build vocab from FULL train.txt (text only, fast) ----
    # This ensures the vocab covers every possible token, independent of
    # how many samples we train on. If vocab already exists, reuse it.
    tokenizer = Tokenizer()
    if os.path.exists(VOCAB_PATH):
        print(f"[Tokenizer] Loading existing vocab from {VOCAB_PATH}")
        tokenizer.load_vocab(VOCAB_PATH)
    else:
        tokenizer.build_vocab_from_label_file(TRAIN_FILE)
        tokenizer.save_vocab(VOCAB_PATH)

    """# ---- DATASET (limited to data_size for fast iteration) ----
    dataset = HMEDataset(DATA_ROOT, TRAIN_FILE, tokenizer, args.data_size)

    # ---- SPLIT ----
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size

    # Fixed seed so train/test split is reproducible across runs
    generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
"""
    full_train = HMEDataset(DATA_ROOT, TRAIN_FILE, tokenizer, args.data_size, train=True)
    full_eval  = HMEDataset(DATA_ROOT, TRAIN_FILE, tokenizer, args.data_size, train=False)

    n = len(full_train)
    train_size = int(0.9 * n)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx, test_idx = perm[:train_size], perm[train_size:]

    from torch.utils.data import Subset
    train_dataset = Subset(full_train, train_idx)
    test_dataset  = Subset(full_eval,  test_idx)   # ← clean, no augmentation
    # ---- LOADERS ----
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ---- MODELS ----
    encoder = Encoder().to(device)
# stopping from freezing encoder
    #for param in encoder.parameters():
       # param.requires_grad = False

    decoder, vocab = build_decoder(
        vocab_path=VOCAB_PATH,
        device=device
    )

    pad_id = vocab.get("<pad>", 0)

    # ---- LOSS + OPT ----
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    
    # <-- CHANGED: Give both models to the optimizer with different Learning Rates
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': args.lr * 0.1}, 
        {'params': decoder.parameters(), 'lr': args.lr}
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    best_em = 0.0
    # ---- TRAIN ----
    for epoch in range(1, args.epochs + 1):

        t0 = time.time()

        loss = train_one_epoch(
            decoder,
            encoder,
            train_loader,
            optimizer,
            criterion,
            device,
            pad_id=pad_id,
        )

        scheduler.step()

        print(f"\nEpoch {epoch} | Loss: {loss:.4f} | Time: {time.time() - t0:.1f}s")

        # Quick sanity check — show a TRAIN sample (memorization check) and a TEST sample
        # so we can see whether the model is just memorizing or actually learning.
        if len(train_dataset) > 0:
            image, label = train_dataset[0]
            pred = greedy_decode(decoder, encoder, image, tokenizer, device)
            gt_tokens = [tokenizer.id_to_token[i] for i in label]
            print("TRAIN GT  :", " ".join(gt_tokens))
            print("TRAIN PRED:", pred)

        if len(test_dataset) > 0:
            image, label = test_dataset[0]
            pred = greedy_decode(decoder, encoder, image, tokenizer, device)
            gt_tokens = [tokenizer.id_to_token[i] for i in label]
            print("TEST  GT  :", " ".join(gt_tokens))
            print("TEST  PRED:", pred)

        """if loss < best_loss:
            best_loss = loss
            print(f"🔥 New best training loss ({best_loss:.4f})! Saving weights to 'best_im2latex_model.pth'...")
            
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }
            torch.save(checkpoint, 'best_im2latex_model.pth')
            """

        decoder.eval(); encoder.eval()
        hits = 0
        n_eval = min(200, len(test_dataset))
        with torch.no_grad():
            for j in range(n_eval):
                img, lbl = test_dataset[j]
                pred = greedy_decode(decoder, encoder, img, tokenizer, device, max_len=150)
                gt = " ".join(tokenizer.id_to_token[t] for t in lbl
                                if t not in (0, tokenizer.token_to_id["<start>"],
                                    tokenizer.token_to_id["<end>"]))
                if pred.strip() == gt.strip(): hits += 1
        val_em = hits / n_eval
        print(f"val EM: {val_em:.4f}")
        if val_em > best_em:
            print(f"--new val EM: {val_em:.4f} is the best")
            best_em = val_em
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_em': best_em
            }
            torch.save(checkpoint, 'best_im2latex_model.pth') 
        decoder.train()
        encoder.train()   
        
        


    print("\nTraining complete")

    # ---- FULL TEST EVALUATION ----
    print("\n--- TESTING ON UNSEEN DATA ---")

    correct_exprs = 0
    total_exprs = 0
    total_similarity = 0.0
    total_bleu = 0.0  # <-- NEW: Track total BLEU score

    # A smoothing function prevents the BLEU score from crashing to 0 
    # if a sequence is too short
    chencherry = SmoothingFunction()

    for i, (images, tokens) in enumerate(test_loader):

        """image = images[0]
        gt_ids = tokens[0].tolist()

        pred = beam_decode(decoder, encoder, image, tokenizer, device)"""

        image = images.to(device)
        gt_ids = tokens[0].tolist()

        pred = beam_decode(decoder, encoder, image, tokenizer, device)

        gt_tokens = [
            tokenizer.id_to_token[t]
            for t in gt_ids
            if t not in (tokenizer.token_to_id["<pad>"],
                         tokenizer.token_to_id["<start>"],
                         tokenizer.token_to_id["<end>"])
        ]
        gt_str = " ".join(gt_tokens)

        total_exprs += 1
        
        # 1. Structural Similarity (difflib)
        similarity = difflib.SequenceMatcher(None, gt_str.strip(), pred.strip()).ratio()
        total_similarity += similarity

        # 2. Raw BLEU Score (nltk)
        reference = [gt_str.split()] 
        candidate = pred.split()
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)

        # --- NEW: The Short-Sequence Fallback ---
        # reference[0] is the list of Ground Truth tokens. 
        if len(reference[0]) < 4:
            # If it's too short for BLEU, use the Similarity Score instead
            final_sample_score = similarity
        else:
            # Otherwise, trust the BLEU score
            final_sample_score = bleu_score
            
        total_bleu += final_sample_score

        if pred.strip() == gt_str.strip():
            correct_exprs += 1

        if i < 10:  
            print(f"\nSample {i}")
            print(f"GT   : {gt_str}")
            print(f"PRED : {pred}")
            print(f"Match: {similarity * 100:.1f}% | Hybrid Score: {final_sample_score:.4f}")

    acc = correct_exprs / total_exprs if total_exprs > 0 else 0
    avg_sim = total_similarity / total_exprs if total_exprs > 0 else 0
    avg_bleu = total_bleu / total_exprs if total_exprs > 0 else 0
    
    print(f"\nExact Match Accuracy: {acc:.4f}  ({correct_exprs}/{total_exprs})")
    print(f"Structural Similarity Score: {avg_sim * 100:.2f}%") 
    print(f"Average BLEU Score: {avg_bleu:.4f}") # <-- NEW: The academic score!


if __name__ == "__main__":
    main()
