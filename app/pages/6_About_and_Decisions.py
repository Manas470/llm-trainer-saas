"""
Page 6 — About & Architecture Decisions
Shows the "why" behind every technology choice directly in the app.
"""
import streamlit as st
from app.utils.state import init_session

st.set_page_config(
    page_title="About & Decisions · LLM Trainer",
    page_icon="🧠",
    layout="wide",
)
init_session()

st.markdown("""
<style>
.decision-card {
    background: #1E2130;
    border-left: 4px solid #6C63FF;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.decision-card h3 { color: #6C63FF; margin: 0 0 .4rem; font-size: 1.05rem; }
.decision-card p  { color: #ddd; margin: 0; line-height: 1.6; font-size: .9rem; }
.alt-rejected {
    background: #1a1020;
    border: 1px solid #FF444433;
    border-radius: 8px;
    padding: .6rem 1rem;
    margin: .3rem 0;
    font-size: .85rem;
    color: #ccc;
}
.alt-chosen {
    background: #0d2a1a;
    border: 1px solid #00C85366;
    border-radius: 8px;
    padding: .6rem 1rem;
    margin: .3rem 0;
    font-size: .85rem;
    color: #ccc;
}
.philosophy-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #6C63FF44;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}
.philosophy-box h2 { color: #6C63FF; margin: 0 0 .8rem; font-size: 1.5rem; }
.philosophy-box p  { color: #ddd; font-size: 1rem; line-height: 1.7; margin: 0; }
.philosophy-box .quote { font-size: 1.3rem; font-style: italic; color: #fff;
                          margin: .8rem 0; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 About This Project & Architecture Decisions")
st.caption("Every technology choice was deliberate. Here's the full story.")

# ── Philosophy ────────────────────────────────────────────────
st.markdown("""
<div class="philosophy-box">
  <h2>The Core Philosophy</h2>
  <div class="quote">"Own the intelligence. Rent the compute."</div>
  <p>
    The LLM Trainer SaaS was built around one constraint: <strong>completely free, no credit card required</strong>.
    This forced every design decision to be intentional. We provide the configuration UI, the training code,
    the notebook generation, and the UX. Free cloud providers supply the rest.
    The result is production-quality LLM fine-tuning at literally $0.
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Decision cards ────────────────────────────────────────────
decisions = [
    {
        "number": "01",
        "title": "Training Framework — PyTorch Lightning",
        "chosen": "PyTorch Lightning 2.x wraps raw PyTorch with clean abstractions for DataModule, LightningModule, and Trainer.",
        "why": (
            "Lightning handles gradient accumulation, mixed precision, early stopping, checkpointing, "
            "and logging automatically — correctly. Building this in raw PyTorch would be 300+ lines of "
            "boilerplate code that's been tested and hardened. Lightning also makes the generated Colab "
            "notebooks readable and educational, not opaque."
        ),
        "rejected": [
            ("HuggingFace Trainer", "Opinionated, tightly coupled, hard to customize for dynamic notebook generation"),
            ("Raw PyTorch", "Would require ~300 lines of training-loop boilerplate prone to subtle bugs"),
            ("Keras / TensorFlow", "ML community has converged on PyTorch; most PEFT tools are PyTorch-native"),
            ("Axolotl / LLaMA-Factory", "Expert-facing, config-file-heavy, black boxes — not suitable for code generation"),
        ],
    },
    {
        "number": "02",
        "title": "Cloud Compute — Google Colab",
        "chosen": "Google Colab free tier provides a T4 GPU (15 GB VRAM) — no credit card, just a Google account.",
        "why": (
            "Colab is the single best option for truly free GPU access. The T4 is sufficient for QLoRA "
            "fine-tuning of 7B models. GitHub Gist integration enables one-click 'Open in Colab' URLs. "
            "Most importantly: virtually everyone already has a Google account, so the barrier to entry is zero."
        ),
        "rejected": [
            ("Kaggle Kernels", "30h/week limit, sessions must restart weekly, less flexible notebook sharing"),
            ("Lightning AI free tier", "Only 4 GPU hours/month, requires new account, proprietary environment"),
            ("HuggingFace Spaces", "Designed for serving demos, not training runs; free tier is CPU-only"),
            ("AWS / GCP / Azure free tier", "Requires credit card, complex setup, minimal GPU on free tier"),
        ],
    },
    {
        "number": "03",
        "title": "Fine-tuning Approach — LoRA / QLoRA as Default",
        "chosen": "LoRA adds tiny adapter matrices (~0.1% of params) to frozen base weights. QLoRA adds 4-bit compression to fit 7B models in 15 GB VRAM.",
        "why": (
            "LoRA achieves 97–99% of full fine-tuning quality at a fraction of the VRAM cost. "
            "A 7B model that would need 112 GB VRAM for full fine-tuning fits in 8–10 GB with QLoRA. "
            "This is the only approach that makes 7B fine-tuning possible on a free T4 GPU. "
            "It's also the industry standard — virtually all production fine-tuning uses LoRA."
        ),
        "rejected": [
            ("Full fine-tuning", "Needs 112 GB+ VRAM for 7B models — impossible on free Colab (still offered as option for small models)"),
            ("Prefix tuning", "Less widely adopted, harder to merge with base model, slightly worse results than LoRA"),
            ("RLHF", "Requires reward model + reference model + main model simultaneously = 3× the VRAM"),
            ("Adapter layers (Houlsby)", "LoRA achieves similar quality with a cleaner, better-supported implementation"),
        ],
    },
    {
        "number": "04",
        "title": "Frontend — Streamlit",
        "chosen": "Streamlit renders the entire multi-page SaaS UI in pure Python, hosted free on Streamlit Community Cloud.",
        "why": (
            "Streamlit requires zero frontend knowledge — the entire app is Python. "
            "Multi-page support, file upload, dataframes, Plotly charts, session state, and download buttons "
            "are all built-in. Deployment is a GitHub push away with no server management. "
            "It's the right tool: the users of this platform (ML practitioners, data scientists) already know it."
        ),
        "rejected": [
            ("Next.js + FastAPI", "Right for a startup, wrong here: needs React skills, separate backend, paid hosting, $50-500/month"),
            ("Gradio", "Good for demos but multi-page apps are harder, state management is weaker"),
            ("Dash (Plotly)", "Excellent dashboards but complex callback mental model, not designed for ML workflows"),
            ("Flask / Django", "Requires HTML/CSS/JS, no built-in ML components, no reactive updates"),
        ],
    },
    {
        "number": "05",
        "title": "Notebook Sharing — GitHub Gist",
        "chosen": "Generated .ipynb notebooks are uploaded to GitHub Gist, enabling colab.research.google.com/gist/... one-click URLs.",
        "why": (
            "GitHub Gists are free, widely available, and Colab has native Gist support — "
            "a specific URL pattern opens the notebook directly in Colab with zero manual steps. "
            "Only a 'gist' scope token is needed — we never ask for repo or org access. "
            "Direct download is always available as a fallback when no token is provided."
        ),
        "rejected": [
            ("Google Drive upload", "Requires OAuth2 — users rightly distrust apps with Drive access"),
            ("Our own cloud storage (S3/GCS)", "Costs money, requires infrastructure — violates the free constraint"),
            ("nbviewer", "Can view but not open in Colab directly"),
            ("Direct download only", "Still supported as fallback, but requires 2 extra manual steps in Colab"),
        ],
    },
    {
        "number": "06",
        "title": "Model Registry — HuggingFace Hub",
        "chosen": "Trained models are saved to HuggingFace Hub (free for public models), enabling immediate API access and sharing.",
        "why": (
            "HuggingFace is the de-facto standard for open-source model hosting. "
            "model.push_to_hub() and from_pretrained() work seamlessly with transformers. "
            "The Inference API lets users query their model immediately via a simple HTTP call at no cost. "
            "PEFT/LoRA adapter saves are tiny (often just a few MB), so even the free tier is more than sufficient."
        ),
        "rejected": [
            ("Google Drive", "Not designed for model serving, no API access, no versioning"),
            ("AWS S3 / GCS", "Costs money for storage and egress, complex access management"),
            ("Weights & Biases Artifacts", "Great for tracking, tight free-tier limits for model weights"),
            ("Local-only storage", "Doesn't survive Colab session end without Drive mount, can't be shared or served"),
        ],
    },
]

for dec in decisions:
    with st.expander(f"**Decision {dec['number']}:** {dec['title']}", expanded=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"""
            <div class="alt-chosen">
            ✅ <strong>What we chose:</strong> {dec['chosen']}<br><br>
            💡 <strong>Why:</strong> {dec['why']}
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("**❌ Alternatives rejected:**")
            for alt, reason in dec["rejected"]:
                st.markdown(f"""
                <div class="alt-rejected">
                <strong>{alt}</strong><br>
                <span style="color:#aaa">{reason}</span>
                </div>
                """, unsafe_allow_html=True)

st.divider()

# ── Architecture diagram ──────────────────────────────────────
st.subheader("🏗️ System Architecture")
st.code("""
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM Trainer SaaS                             │
│                   (Streamlit · free hosting)                        │
│                                                                     │
│  ┌────────────┐  ┌──────────────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 0. Getting │  │ 1. Upload Data   │  │ 2. Config│  │ 3. Jobs  │  │
│  │    Started │  │ (CSV/JSONL/TXT)  │  │ Training │  │ Dashboard│  │
│  └────────────┘  └──────────────────┘  └────┬─────┘  └──────────┘  │
│                                             │                       │
│  ┌──────────────────────────────────────────▼────────────────────┐  │
│  │              Notebook Generator (nbformat)                     │  │
│  │   • LoRA/QLoRA notebook   • Full FT notebook                  │  │
│  │   • Prompt notebook       • All user settings embedded        │  │
│  └──────────────────────────────────┬──────────────────────────── ┘  │
│                                     │                               │
│  ┌────────────────┐       ┌─────────▼──────────┐                   │
│  │ 4. Test Model  │       │  GitHub Gist (free) │                   │
│  │ (HF Inference) │       │  → Open in Colab URL│                   │
│  └────────────────┘       └─────────┬───────────┘                  │
└─────────────────────────────────────┼───────────────────────────────┘
                                      │ one click
               ┌──────────────────────▼────────────────────┐
               │          Google Colab — Free T4 GPU        │
               │                                            │
               │  1. Install deps (pytorch-lightning, peft) │
               │  2. Upload dataset from your computer      │
               │  3. Download base model from HuggingFace   │
               │  4. PyTorch Lightning training loop        │
               │     • LoRA/QLoRA adapters                  │
               │     • Cosine LR schedule + warmup          │
               │     • Early stopping + checkpointing       │
               │  5. Save model weights                     │
               └───────────────────┬────────────────────────┘
                                   │
               ┌───────────────────▼────────────────────────┐
               │        HuggingFace Hub (free public)        │
               │   model.push_to_hub("yourname/my-model")   │
               │   → Inference API ready immediately        │
               └────────────────────────────────────────────┘
""", language="text")

st.divider()

# ── What we'd add next ────────────────────────────────────────
st.subheader("🗺️ What We'd Add Next")

roadmap = [
    ("🎯 High Priority", [
        ("Weights & Biases integration", "Real-time loss curves, experiment comparison, hyperparameter tracking"),
        ("Dataset augmentation via LLM", "Use GPT-4 / Claude to synthetically expand small datasets before training"),
        ("Post-training evaluation", "MMLU, HellaSwag, HumanEval benchmarks run automatically after training"),
        ("Multi-user support", "User auth + job queue (Celery + Redis) for shared deployment"),
    ]),
    ("🔧 Medium Priority", [
        ("DPO fine-tuning", "Direct Preference Optimization for alignment, requires preference pairs"),
        ("LoRA adapter merging", "Merge LoRA weights back into base model for single-file deployment"),
        ("Inference optimization", "GGUF quantization, llama.cpp export for local CPU inference"),
        ("Multi-GPU training", "Lightning DDP strategy, for users with Colab Pro or cloud GPUs"),
    ]),
    ("💡 Future Ideas", [
        ("Real-time training metrics", "WebSocket callback from Colab to update the app during training"),
        ("Dataset marketplace", "Share and discover community fine-tuning datasets"),
        ("Automatic hyperparameter tuning", "Optuna integration to find optimal LR, rank, etc."),
        ("Model merging (SLERP)", "Combine multiple LoRA adapters or merge specialist models"),
    ]),
]

for priority, items in roadmap:
    with st.expander(priority):
        for name, desc in items:
            st.markdown(f"**{name}** — {desc}")

st.divider()

# ── Links ─────────────────────────────────────────────────────
st.subheader("📚 Further Reading")
c1, c2, c3, c4 = st.columns(4)
c1.link_button("QLoRA Paper (Dettmers et al.)",  "https://arxiv.org/abs/2305.14314",              use_container_width=True)
c2.link_button("LoRA Paper (Hu et al.)",          "https://arxiv.org/abs/2106.09685",              use_container_width=True)
c3.link_button("PyTorch Lightning Docs",           "https://lightning.ai/docs/pytorch/stable/",    use_container_width=True)
c4.link_button("PEFT Documentation",               "https://huggingface.co/docs/peft",             use_container_width=True)

st.caption("🤖 LLM Trainer SaaS · MIT License · Built with ❤️ using PyTorch Lightning, HuggingFace, Streamlit, Google Colab")
