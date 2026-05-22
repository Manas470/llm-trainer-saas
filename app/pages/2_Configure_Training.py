"""Page 2 — Configure hyperparameters & generate Colab notebook."""
import json
import tempfile
from pathlib import Path

import streamlit as st

from app.utils.config import BASE_MODELS, TRAINING_MODES, TrainingConfig, JOBS_DIR
from app.utils.state import init_session, create_job
from app.components.notebook_generator import (
    generate_notebook,
    save_notebook_locally,
    share_via_gist,
)

st.set_page_config(page_title="Configure Training · LLM Trainer", page_icon="⚙️", layout="wide")
init_session()

st.title("⚙️ Configure Training")

# ── Guard: dataset required ───────────────────────────────────
ds_info = st.session_state.get("dataset_info")
if not ds_info:
    st.warning("⚠️ Please upload a dataset first.")
    if st.button("Go to Upload →"):
        st.switch_page("pages/1_Upload_Data.py")
    st.stop()

st.success(f"📂 Dataset: **{ds_info['filename']}** · {ds_info['rows']:,} rows")
st.divider()

cfg = st.session_state["config"]

# ─────────────────────────────────────────────────────────────
# 1 — Training mode
# ─────────────────────────────────────────────────────────────
st.subheader("1️⃣  Training Mode")
mode_labels = list(TRAINING_MODES.values())
mode_keys   = list(TRAINING_MODES.keys())
current_idx = mode_keys.index(cfg.get("training_mode", "lora"))
chosen_label = st.radio("Select training approach", mode_labels, index=current_idx, horizontal=False)
cfg["training_mode"] = mode_keys[mode_labels.index(chosen_label)]

mode_help = {
    "lora":   "🟢 **Recommended.** Adds small adapter layers (LoRA). Trains in minutes on a free T4 GPU. Works with any model.",
    "full":   "🔴 Updates **all** model weights. Needs a large GPU (A100). Use small models (GPT-2) on free Colab.",
    "prompt": "🟡 **No training.** Your data is injected as few-shot examples into the prompt. Instant, no GPU needed.",
}
st.info(mode_help[cfg["training_mode"]])

st.divider()

# ─────────────────────────────────────────────────────────────
# 2 — Base model
# ─────────────────────────────────────────────────────────────
st.subheader("2️⃣  Base Model")
model_labels  = list(BASE_MODELS.keys())
model_values  = list(BASE_MODELS.values())
current_model = cfg.get("base_model", "gpt2")
try:
    mi = model_values.index(current_model)
except ValueError:
    mi = 0
chosen_model = st.selectbox("Choose a base model", model_labels, index=mi)
cfg["base_model"] = BASE_MODELS[chosen_model]

col_a, col_b = st.columns(2)
with col_a:
    cfg["hf_repo_id"] = st.text_input(
        "HuggingFace repo to push model (optional)",
        value=cfg.get("hf_repo_id", ""),
        placeholder="yourname/my-finetuned-model",
    )
with col_b:
    cfg["push_to_hub"] = st.checkbox("Push model to HuggingFace Hub after training",
                                     value=cfg.get("push_to_hub", False))

st.divider()

# ─────────────────────────────────────────────────────────────
# 3 — Data settings
# ─────────────────────────────────────────────────────────────
st.subheader("3️⃣  Data Settings")
cols = ds_info.get("columns", ["text"])

c1, c2, c3 = st.columns(3)
with c1:
    tc = cols.index(cfg.get("text_column","text")) if cfg.get("text_column","text") in cols else 0
    cfg["text_column"] = st.selectbox("Text column", cols, index=tc)
with c2:
    pc = cols.index(cfg.get("prompt_column","prompt")) if cfg.get("prompt_column","prompt") in cols else 0
    cfg["prompt_column"] = st.selectbox("Prompt column (instruction tuning)", cols, index=pc)
with c3:
    rc = cols.index(cfg.get("response_column","response")) if cfg.get("response_column","response") in cols else 0
    cfg["response_column"] = st.selectbox("Response column (instruction tuning)", cols, index=rc)

c4, c5 = st.columns(2)
with c4:
    cfg["max_seq_length"] = st.select_slider("Max sequence length (tokens)",
                                              options=[128,256,512,1024,2048,4096],
                                              value=cfg.get("max_seq_length",512))
with c5:
    cfg["val_split"] = st.slider("Validation split", 0.05, 0.3, cfg.get("val_split",0.1), 0.05)

st.divider()

# ─────────────────────────────────────────────────────────────
# 4 — Hyperparameters
# ─────────────────────────────────────────────────────────────
st.subheader("4️⃣  Training Hyperparameters")

with st.expander("📐 Core hyperparameters", expanded=True):
    h1, h2, h3 = st.columns(3)
    with h1:
        cfg["num_epochs"] = st.number_input("Epochs", 1, 50, cfg.get("num_epochs",3))
        cfg["batch_size"] = st.number_input("Batch size", 1, 64, cfg.get("batch_size",4))
    with h2:
        cfg["grad_accum"]    = st.number_input("Gradient accumulation steps", 1, 64, cfg.get("grad_accum",4))
        cfg["learning_rate"] = st.number_input("Learning rate", 1e-6, 1e-1,
                                                cfg.get("learning_rate",2e-4),
                                                format="%.2e", step=1e-5)
    with h3:
        cfg["warmup_steps"]  = st.number_input("Warmup steps", 0, 1000, cfg.get("warmup_steps",100))
        cfg["weight_decay"]  = st.number_input("Weight decay", 0.0, 0.5, cfg.get("weight_decay",0.01), 0.001)
    cfg["fp16"] = st.checkbox("Mixed precision (fp16) — faster on GPU", value=cfg.get("fp16",True))

# LoRA-specific
if cfg["training_mode"] in ("lora", "qlora"):
    with st.expander("🔧 LoRA / QLoRA settings", expanded=True):
        l1, l2, l3 = st.columns(3)
        with l1:
            cfg["lora_r"]       = st.number_input("LoRA rank (r)", 4, 128, cfg.get("lora_r",16), step=4)
            cfg["use_qlora"]    = st.checkbox("Use QLoRA (4-bit quantisation)", value=cfg.get("use_qlora",True))
        with l2:
            cfg["lora_alpha"]   = st.number_input("LoRA alpha", 8, 256, cfg.get("lora_alpha",32), step=8)
        with l3:
            cfg["lora_dropout"] = st.slider("LoRA dropout", 0.0, 0.5, cfg.get("lora_dropout",0.05), 0.01)

        st.caption(
            "**LoRA rank** controls adapter size (higher = more params, better quality, more VRAM).  \n"
            "**QLoRA** loads the base model in 4-bit, reducing VRAM by ~75% with minimal accuracy loss."
        )

st.divider()

# ─────────────────────────────────────────────────────────────
# 5 — Generate notebook
# ─────────────────────────────────────────────────────────────
st.subheader("5️⃣  Generate Your Colab Notebook")
st.caption("We embed all your settings into a ready-to-run `.ipynb` file.")

cfg["dataset_path"] = ds_info["path"]   # embed path into notebook

github_token = st.text_input(
    "GitHub token (optional — for one-click Colab sharing via Gist)",
    type="password",
    placeholder="ghp_xxxx  ·  scope: gist  ·  get one at github.com/settings/tokens",
    help="If left blank, you'll download the notebook and upload it to Colab manually.",
)

if st.button("🚀 Generate Notebook & Create Job", type="primary", use_container_width=True):
    with st.spinner("Generating notebook …"):
        import uuid
        job_id = str(uuid.uuid4())[:8]

        nb, nb_json = generate_notebook(cfg)

        # Save locally
        nb_path = save_notebook_locally(nb_json, job_id, JOBS_DIR)

        # Try GitHub Gist
        gist_id, colab_url = share_via_gist(nb_json, job_id, github_token)

        # Create job record
        create_job(cfg.copy(), colab_url or "", gist_id)

    st.success(f"✅ Job **{job_id}** created!")

    # Download button
    st.download_button(
        label    = "⬇️ Download Notebook (.ipynb)",
        data     = nb_json,
        file_name = f"llm_trainer_{job_id}.ipynb",
        mime     = "application/json",
    )

    if colab_url:
        st.link_button("🚀 Open in Google Colab ↗", colab_url, use_container_width=True)
        st.info(
            "**Next steps:**\n"
            "1. Click **Open in Google Colab** above.\n"
            "2. Go to **Runtime → Change runtime type → T4 GPU** (free).\n"
            "3. Click **Runtime → Run all**.\n"
            "4. Upload your dataset when prompted."
        )
    else:
        st.info(
            "**No GitHub token provided.**\n\n"
            "1. Download the notebook above.\n"
            "2. Go to [colab.research.google.com](https://colab.research.google.com).\n"
            "3. Click **File → Upload notebook** and select the downloaded file.\n"
            "4. Switch to T4 GPU runtime and click **Run all**."
        )

    if st.button("📋 View All Jobs →", use_container_width=True):
        st.switch_page("pages/3_Training_Jobs.py")

# ── Config preview ────────────────────────────────────────────
with st.expander("🔍 Current config JSON"):
    st.json(cfg)
