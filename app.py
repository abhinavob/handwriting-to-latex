import streamlit as st
import torch
import tempfile
import os
import json
import time
from PIL import Image
from torchvision import transforms

st.set_page_config(
    page_title="MathScript — Handwriting to LaTeX",
    page_icon="∑",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #09090f;
    color: #e8e4dc;
}
.stApp { background: #09090f; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1100px; }

.hero { text-align: center; padding: 2rem 0 1.5rem; }
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.4rem;
    font-style: italic;
    background: linear-gradient(135deg, #f5e6c8 0%, #e8c97a 50%, #c49a3c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5a5245;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2e2c24, transparent);
    margin: 1.5rem 0;
}
.card {
    background: #111018;
    border: 1px solid #1e1d24;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.card-header {
    padding: 0.8rem 1.2rem;
    border-bottom: 1px solid #1e1d24;
    background: #0d0c13;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c49a3c;
}
.card-body { padding: 1.2rem; }
.latex-box {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #a89b7a;
    background: #0a0a0f;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    border: 1px solid #1a1918;
    word-break: break-all;
    line-height: 1.8;
}
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
}
.badge-gold { background: #231e0e; color: #c49a3c; border: 1px solid #3a3010; }
.badge-green { background: #0a1f12; color: #4caf7d; border: 1px solid #1a3d28; }
.badge-blue { background: #0a0f20; color: #5b8cff; border: 1px solid #1a2a50; }
.metrics-row {
    display: flex;
    gap: 1rem;
    margin-top: 0.8rem;
    flex-wrap: wrap;
}
.metric {
    flex: 1;
    min-width: 120px;
    background: #0d0c13;
    border: 1px solid #1e1d24;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #c49a3c;
    line-height: 1;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #4a4640;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.stButton > button {
    background: linear-gradient(135deg, #c49a3c, #a07828) !important;
    color: #09090f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    width: 100%;
    padding: 0.6rem !important;
}
.stSlider > div > div > div > div { background: #c49a3c !important; }
.stTabs [data-baseweb="tab-list"] { background: #0d0c13; border-radius: 10px; padding: 0.3rem; }
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    color: #5a5245 !important;
}
.stTabs [aria-selected="true"] {
    background: #1e1c2a !important;
    color: #c49a3c !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model once ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    from encoder import Encoder
    from decoder import build_decoder
    from tokenizer import Tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer()
    tokenizer.load_vocab("vocab.json")
    encoder = Encoder().to(device)
    decoder, _ = build_decoder(vocab_path="vocab.json", device=str(device))
    checkpoint = torch.load("best_im2latex_model.pth", map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, tokenizer, device, checkpoint


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def beam_decode_with_width(decoder, encoder, image, tokenizer, device, beam_width=7):
    """Beam search with tuned parameters — best config: beam_width=7, a=0.5"""
    from math import log

    def max_len(length, a=150): return length < a
    def get_repetition_penalty(tokens, current_id, penalty=-0.7):
        if len(tokens) > 1 and tokens[-1] == current_id and tokens[-2] == current_id:
            return penalty
        return 0.0
    def lenp(length, k=6, a=0.5): return ((k + length) / (k + 1)) ** a
    def conv_penalty(conv, convpenalty=0.7, penalty_sum=0):
        temp_const = 1e-6
        for i in range(len(conv)):
            penalty_sum += log(min(conv[i], 1.0) + temp_const)
        return penalty_sum * convpenalty
    def bracket_penalty(new_token, ob, cb, penalty=-10):
        s_open = s_close = 0
        if ob == -1 or cb == -1: return 0
        for i in new_token:
            if i == ob: s_open += 1
            elif i == cb: s_close += 1
            if s_open - s_close < 0: return penalty
        return penalty if s_open != s_close else 0
    def normalised_beam_score(bs, length, conv, new_token, ob, cb):
        return bs / lenp(length) + conv_penalty(conv) + bracket_penalty(new_token, ob, cb)

    eos = tokenizer.token_to_id.get("<end>")
    start_id = tokenizer.token_to_id.get("<start>")
    ob = tokenizer.token_to_id.get("{", -1)
    cb = tokenizer.token_to_id.get("}", -1)
    min_score_token = -11
    min_length = 4

    with torch.no_grad():
        enc_out, src_lengths = encoder(image)

    src_len = enc_out.shape[1]
    completed = []
    active_beams = [([start_id], 0.0, [0.0] * src_len)]

    while active_beams:
        all_candidates = []
        for b_tokens, b_score, b_conv in active_beams:
            tgt_tensor = torch.tensor([b_tokens], dtype=torch.long).to(device)
            with torch.no_grad():
                logits, attn = decoder.decode_step(tgt_tensor, enc_out, src_lengths)
                log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1)
                topb, indices = torch.topk(log_probs, beam_width)
                best_score = topb[0].item()

            attn_weights = attn[0].tolist()

            for score, tok_id in zip(topb, indices):
                t_id = tok_id.item()
                if (best_score - score.item()) > 3.2: break
                if score.item() < min_score_token: continue
                new_token = b_tokens + [t_id]
                if new_token.count(cb) > new_token.count(ob): continue
                rep_penalty = get_repetition_penalty(b_tokens, t_id)
                new_score = b_score + score.item() + rep_penalty
                new_coverage = b_conv.copy()
                for i in range(src_len):
                    new_coverage[i] += attn_weights[i]
                if not max_len(len(new_token)): continue
                if t_id == eos and len(new_token) > min_length:
                    ns = normalised_beam_score(new_score, len(new_token), new_coverage, new_token, ob, cb)
                    completed.append({'tokens': new_token, 'score': ns})
                    continue
                all_candidates.append((new_token, new_score, new_coverage))

        all_candidates.sort(key=lambda x: x[1] / lenp(len(x[0])), reverse=True)
        active_beams = all_candidates[:beam_width]
        if not active_beams or len(active_beams[0][0]) > 150: break

    if not completed:
        if active_beams:
            completed = [{'tokens': b[0], 'score': b[1]} for b in active_beams]
        else:
            return ""

    best = max(completed, key=lambda x: x['score'])
    best_tokens = best['tokens']
    if best_tokens and best_tokens[0] == start_id: best_tokens = best_tokens[1:]
    if best_tokens and best_tokens[-1] == eos: best_tokens = best_tokens[:-1]

    result = tokenizer.decode(best_tokens)
    # strip any remaining special tokens
    result = result.replace('<end>', '').replace('<start>', '').replace('<pad>', '').replace('<unk>', '').strip()
    return result


def run_prediction(pil_image, beam_width_val):
    encoder, decoder, tokenizer, device, _ = load_models()
    image_tensor = transform(pil_image).unsqueeze(0).to(device)
    t0 = time.time()
    result = beam_decode_with_width(decoder, encoder, image_tensor, tokenizer, device, beam_width_val)
    elapsed = time.time() - t0
    return result, elapsed


def latex_to_readable(latex_str):
    replacements = {
        '\\frac': '÷', '\\times': '×', '\\div': '÷',
        '\\sqrt': '√', '\\sum': 'Σ', '\\int': '∫',
        '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ',
        '\\delta': 'δ', '\\theta': 'θ', '\\pi': 'π',
        '\\mu': 'μ', '\\sigma': 'σ', '\\omega': 'ω',
        '\\infty': '∞', '\\leq': '≤', '\\geq': '≥',
        '\\neq': '≠', '\\approx': '≈', '\\equiv': '≡',
        '\\in': '∈', '\\rightarrow': '→', '\\leftarrow': '←',
        '\\uparrow': '↑', '\\downarrow': '↓',
        '\\pm': '±', '\\cdot': '·', '\\cdots': '···',
        '{': '', '}': '',
    }
    result = latex_str
    for k, v in replacements.items():
        result = result.replace(k, v)
    return result.strip()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">MathScript</div>
    <div class="hero-sub">Handwritten Mathematics → LaTeX · BITS Pilani Project 6</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; 
    color:#c49a3c; font-style:italic; margin-bottom:1rem;">Controls</div>
    """, unsafe_allow_html=True)

    beam_width_val = st.slider(
        "Beam Width", min_value=1, max_value=15, value=7, step=1,
        help="Higher = more accurate but slower. 7 is optimal."
    )

    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace; font-size:0.68rem; 
    color:#4a4640; margin-top:-0.5rem; margin-bottom:1rem;">
    k={beam_width_val} · {'Greedy' if beam_width_val==1 else 'Beam Search'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.68rem; 
    color:#4a4640; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;">
    Model Info
    </div>
    """, unsafe_allow_html=True)

    try:
        _, _, _, device, ckpt = load_models()
        epoch = ckpt.get('epoch', '?')
        best_em = ckpt.get('best_val_em', 0)
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace; font-size:0.72rem; color:#6b6557; line-height:2;">
        Epoch: <span style="color:#c49a3c">{epoch}</span><br>
        Val EM: <span style="color:#c49a3c">{best_em:.2%}</span><br>
        Device: <span style="color:#c49a3c">{str(device).upper()}</span><br>
        Backbone: <span style="color:#c49a3c">ResNet34</span><br>
        Dataset: <span style="color:#c49a3c">HME100K</span>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.markdown('<span style="color:#ff6b6b; font-size:0.75rem;">Model not loaded</span>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Clear button in sidebar ───────────────────────────────────────────────
    if st.button("🔄 Clear / New Image"):
        st.session_state.pop('pil_image', None)
        st.session_state.pop('source_label', None)
        st.rerun()


# ── Input tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📁  Upload Image", "📷  Camera", "🔗  URL"])

pil_image = None
source_label = ""

with tab1:
    uploaded = st.file_uploader(
        "Drop a handwritten math image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
        label_visibility="collapsed"
    )
    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        source_label = uploaded.name

with tab2:
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.75rem; 
    color:#4a4640; margin-bottom:0.5rem;">
    Take a photo of your handwritten equation — fill the frame with just the equation
    </div>
    """, unsafe_allow_html=True)
    camera_img = st.camera_input("Camera", label_visibility="collapsed")
    if camera_img:
        pil_image = Image.open(camera_img).convert("RGB")
        source_label = "camera_capture.jpg"

with tab3:
    url_input = st.text_input(
        "Image URL",
        placeholder="https://example.com/math.jpg",
        label_visibility="collapsed"
    )
    if url_input and st.button("Load from URL", key="url_load"):
        try:
            import requests
            from io import BytesIO
            resp = requests.get(url_input, timeout=10)
            pil_image = Image.open(BytesIO(resp.content)).convert("RGB")
            source_label = url_input.split("/")[-1]
            st.success("Image loaded")
        except Exception as e:
            st.error(f"Could not load image: {e}")


# ── Main result area ──────────────────────────────────────────────────────────
if pil_image is not None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_img, col_out = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="card"><div class="card-header"><span class="card-label">Input Image</span><span class="badge badge-blue">∑ ready</span></div><div class="card-body">', unsafe_allow_html=True)
        st.image(pil_image, use_container_width=True)
        st.markdown(f"""
        <div style="margin-top:0.8rem; display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;">
            <span class="badge badge-gold">beam k={beam_width_val}</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.65rem; color:#3a3830;">
            {pil_image.size[0]}×{pil_image.size[1]}px · {source_label}
            </span>
        </div>
        </div></div>
        """, unsafe_allow_html=True)

    with col_out:
        with st.spinner("Running beam search..."):
            try:
                result, elapsed = run_prediction(pil_image, beam_width_val)
                success = True
            except Exception as e:
                success = False
                err_msg = str(e)

        if success and result:
            st.markdown(f"""
            <div class="card">
                <div class="card-header">
                    <span class="card-label">LaTeX Output</span>
                    <span class="badge badge-green">✓ {elapsed:.2f}s</span>
                </div>
                <div class="card-body">
                    <div class="latex-box">{result}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.code(result, language=None)

            st.markdown(f"""
            <div class="card" style="margin-top:0.8rem;">
                <div class="card-header">
                    <span class="card-label">Rendered Preview</span>
                    <span class="badge badge-gold">∑</span>
                </div>
                <div class="card-body" style="text-align:center; padding:1.5rem;">
            """, unsafe_allow_html=True)
            st.latex(result)
            st.markdown("</div></div>", unsafe_allow_html=True)

            token_count = len(result.split())
            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric">
                    <div class="metric-val">{token_count}</div>
                    <div class="metric-label">Tokens</div>
                </div>
                <div class="metric">
                    <div class="metric-val">{beam_width_val}</div>
                    <div class="metric-label">Beam Width</div>
                </div>
                <div class="metric">
                    <div class="metric-val">{elapsed:.2f}s</div>
                    <div class="metric-label">Inference</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif success and not result:
            st.warning("Model returned empty output. Try a clearer image.")
        else:
            st.error(f"Prediction failed: {err_msg}")

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#2e2c24;">
        <div style="font-size:3rem; margin-bottom:1rem;">∫</div>
        <div style="font-family:'DM Mono',monospace; font-size:0.8rem; 
        letter-spacing:0.15em; text-transform:uppercase; color:#3a3830;">
        Upload an image to begin
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="divider" style="margin-top:3rem;"></div>
<div style="text-align:center; font-family:'DM Mono',monospace; font-size:0.62rem; 
color:#2e2c24; letter-spacing:0.12em; padding-bottom:1rem;">
HME100K · ResNet34 Encoder · 6-Layer Transformer Decoder · Beam Search k=7 · BITS Pilani CS F407
</div>
""", unsafe_allow_html=True)
