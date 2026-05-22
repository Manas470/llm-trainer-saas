"""
LLM Trainer SaaS — Main Streamlit entry-point
Run: streamlit run app/main.py
"""
import streamlit as st

st.set_page_config(
    page_title  = "LLM Trainer SaaS",
    page_icon   = "🤖",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

from app.utils.state import init_session, list_jobs, DONE, RUNNING, PENDING

init_session()

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg,#6C63FF 0%,#3ECFCF 100%);
    padding: 2.5rem 2rem; border-radius: 16px; color: #fff; margin-bottom: 2rem;
}
.hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0 0 .5rem; }
.hero p  { font-size: 1.1rem; margin: 0; opacity: .9; }
.hero .badges { margin-top: 1rem; display: flex; gap: .5rem; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.2); border-radius: 20px;
    padding: .25rem .75rem; font-size: .8rem; font-weight: 600;
}
.stat-card {
    background: #1E2130; border: 1px solid #2E3250;
    border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center;
}
.stat-card .num   { font-size: 2.2rem; font-weight: 800; color: #6C63FF; }
.stat-card .label { font-size: .85rem; color: #aaa; margin-top: .3rem; }
.step-card {
    background: #1E2130; border: 1px solid #2E3250; border-radius: 12px;
    padding: 1rem 1.2rem; margin-bottom: .8rem; display: flex; gap: 1rem; align-items: flex-start;
}
.step-num {
    background: #6C63FF; color: #fff; border-radius: 50%;
    width: 32px; height: 32px; display: flex; align-items: center;
    justify-content: center; font-weight: 700; flex-shrink: 0; font-size: 1rem;
}
.model-tag {
    background: #252840; border: 1px solid #3C3F6E; border-radius: 8px;
    padding: .2rem .6rem; font-size: .78rem; color: #9B9EFF; display: inline-block;
    margin: .15rem;
}
.free-badge {
    background: #0a3d1f; border: 1px solid #00C85366;
    border-radius: 8px; padding: .5rem 1rem; color: #00C853;
    font-weight: 600; font-size: .9rem; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🤖 LLM Trainer SaaS</h1>
  <p>Fine-tune any language model on your own data — completely free, no credit card required.<br>
     Upload your dataset · Configure training · Get a Colab notebook · Train on free GPU.</p>
  <div class="badges">
    <span class="badge">⚡ PyTorch Lightning</span>
    <span class="badge">🤗 HuggingFace</span>
    <span class="badge">☁️ Google Colab</span>
    <span class="badge">🆓 100% Free</span>
    <span class="badge">🔧 LoRA / QLoRA</span>
    <span class="badge">📦 No GPU needed locally</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stats row ─────────────────────────────────────────────────
all_jobs = list_jobs()
col1, col2, col3, col4, col5 = st.columns(5)
metrics = [
    (len(all_jobs), "Total Jobs"),
    (sum(1 for j in all_jobs if j["status"] == DONE),    "Completed"),
    (sum(1 for j in all_jobs if j["status"] == RUNNING), "Running"),
    (sum(1 for j in all_jobs if j["status"] == PENDING), "Pending"),
    (1 if st.session_state.get("dataset_info") else 0,   "Dataset Loaded"),
]
for col, (num, label) in zip([col1,col2,col3,col4,col5], metrics):
    col.markdown(
        f'<div class="stat-card"><div class="num">{num}</div><div class="label">{label}</div></div>',
        unsafe_allow_html=True
    )

st.divider()

# ── Quick start CTAs ──────────────────────────────────────────
st.subheader("🚀 Quick Start")
cta1, cta2, cta3, cta4 = st.columns(4)
with cta1:
    if st.button("🚀 Getting Started Guide", use_container_width=True, type="primary"):
        st.switch_page("pages/0_Getting_Started.py")
with cta2:
    if st.button("📂 Upload Data", use_container_width=True):
        st.switch_page("pages/1_Upload_Data.py")
with cta3:
    if st.button("⚙️ Configure & Train", use_container_width=True):
        st.switch_page("pages/2_Configure_Training.py")
with cta4:
    if st.button("💬 Test a Model", use_container_width=True):
        st.switch_page("pages/4_Test_Model.py")

st.divider()

# ── How it works ──────────────────────────────────────────────
col_flow, col_free = st.columns([3, 1])

with col_flow:
    st.subheader("🗺️ How It Works — 5 Steps")
    steps = [
        ("1", "📂 Upload Your Data",
         "Drop a <strong>CSV, JSONL, TXT, or Parquet</strong> file — or load a built-in sample dataset instantly. "
         "The app auto-detects instruction-tuning vs. text-generation format."),
        ("2", "⚙️ Configure Training",
         "Choose a <strong>base model</strong> (GPT-2 to Mistral 7B), select your <strong>training mode</strong> "
         "(LoRA / QLoRA / Full / Prompt-based), and tune hyperparameters with helpful defaults."),
        ("3", "📓 Generate Colab Notebook",
         "Click one button. We embed ALL your settings into a ready-to-run <strong>.ipynb notebook</strong>. "
         "Share via GitHub Gist for a one-click Colab link, or download and upload manually."),
        ("4", "☁️ Train on Free GPU",
         "Open in <strong>Google Colab</strong>, switch to T4 GPU runtime (free, no card), "
         "click <em>Run All</em>, upload your dataset when prompted. Training runs in the cloud."),
        ("5", "🎉 Use Your Model",
         "Download your trained model or push to <strong>HuggingFace Hub</strong> (free). "
         "Test it immediately in the <em>Test Model</em> page with a built-in chat interface."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
          <div class="step-num">{num}</div>
          <div><strong>{title}</strong><br><small style="color:#ccc">{desc}</small></div>
        </div>
        """, unsafe_allow_html=True)

with col_free:
    st.subheader("🆓 Free Tier Summary")
    st.markdown("""
    <div class="free-badge" style="margin-bottom:.6rem">✅ $0 cost</div>
    """, unsafe_allow_html=True)
    free_items = [
        ("☁️ Compute",    "Google Colab T4 GPU · 15 GB VRAM"),
        ("🤗 Model Hub",  "HuggingFace Hub (public models)"),
        ("💾 Storage",    "Google Drive 15 GB or HF Hub"),
        ("🖥️ UI Hosting", "Streamlit Community Cloud"),
        ("📓 Notebooks",  "GitHub Gist (free private gists)"),
        ("⏱️ GPU time",   "~4–12h per Colab session"),
    ]
    for icon_label, desc in free_items:
        st.markdown(f"""
        <div style='background:#1E2130;border-radius:8px;padding:.5rem .8rem;margin:.3rem 0;'>
          <strong style='color:#00C853'>{icon_label}</strong>
          <br><small style='color:#aaa'>{desc}</small>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Supported models ──────────────────────────────────────────
st.subheader("🤖 Supported Models")
models = [
    ("GPT-2 124M",         "✅ CPU / T4",  "Fastest, great for learning"),
    ("DistilGPT-2 82M",    "✅ CPU / T4",  "Lightest model"),
    ("GPT-2 Medium 345M",  "✅ T4",        "Good balance"),
    ("GPT-2 Large 774M",   "✅ T4",        "Better quality"),
    ("TinyLlama 1.1B",     "✅ T4",        "Modern, instruction-tunable"),
    ("Phi-2 2.7B",         "✅ T4 QLoRA",  "Excellent quality/size ratio"),
    ("OPT 1.3B",           "✅ T4",        "Facebook research model"),
    ("Gemma 2B",           "✅ T4 QLoRA",  "Google's efficient model"),
    ("Falcon 7B Instruct", "⚡ T4 QLoRA",  "Needs QLoRA on free tier"),
    ("Llama-3.2 1B",       "✅ T4",        "Meta's latest compact model"),
    ("Llama-3.2 3B",       "✅ T4 QLoRA",  "Meta's latest 3B model"),
    ("Mistral 7B Instruct","⚡ T4 QLoRA",  "Best quality on free tier"),
]
model_cols = st.columns(4)
for i, (name, compat, note) in enumerate(models):
    with model_cols[i % 4]:
        color = "#00C853" if "✅" in compat else "#FFA500"
        st.markdown(f"""
        <div style='background:#1E2130;border:1px solid #2E3250;border-radius:10px;
                    padding:.7rem;margin-bottom:.5rem;'>
          <div style='font-weight:700;font-size:.9rem;color:#fff'>{name}</div>
          <div style='color:{color};font-size:.78rem'>{compat}</div>
          <div style='color:#888;font-size:.75rem'>{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Training mode comparison ──────────────────────────────────
st.subheader("🧠 Training Mode Comparison")
mc1, mc2, mc3 = st.columns(3)
modes = [
    (mc1, "🔧 LoRA / QLoRA", "#6C63FF",
     "✅ Recommended for 99% of users",
     ["Works on free T4 GPU", "Any model up to 7B", "Fast (minutes–hours)", "~1% of params trained", "4-bit with QLoRA"]),
    (mc2, "💪 Full Fine-tuning", "#FFA500",
     "⚠️ Advanced — needs big GPU",
     ["All weights updated", "Best for small models (GPT-2)", "Needs 40GB+ for 7B", "Slower, more memory", "Maximum quality ceiling"]),
    (mc3, "💬 Prompt-based", "#3ECFCF",
     "🟡 No training — instant",
     ["No GPU needed", "Zero cost", "Use as few-shot examples", "Great for quick tests", "Data stays as context"]),
]
for col, title, color, badge, points in modes:
    with col:
        st.markdown(f"""
        <div style='background:#1E2130;border:1px solid {color}44;border-radius:12px;padding:1.2rem;height:100%'>
          <div style='font-size:1.1rem;font-weight:800;color:{color};margin-bottom:.4rem'>{title}</div>
          <div style='background:{color}22;border-radius:6px;padding:.3rem .6rem;font-size:.8rem;
                      color:{color};margin-bottom:.8rem'>{badge}</div>
          {''.join(f"<div style='color:#ccc;font-size:.85rem;padding:.2rem 0'>✓ {p}</div>" for p in points)}
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Footer ────────────────────────────────────────────────────
f1, f2, f3 = st.columns(3)
with f1:
    st.caption("🤖 **LLM Trainer SaaS** · Open source · MIT License")
with f2:
    st.caption("⚡ Built with PyTorch Lightning · HuggingFace · Streamlit")
with f3:
    st.caption("☁️ Free cloud training via Google Colab · No credit card needed")
