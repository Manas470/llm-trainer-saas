"""Page 1 — Upload & preview dataset (with built-in sample datasets)."""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

from app.utils.config import UPLOAD_DIR
from app.utils.state import init_session
from app.components.sample_datasets import (
    SAMPLE_DATASETS, get_sample_dataframe, get_sample_info, list_sample_datasets
)
from app.components.data_processor import detect_dataset_type, compute_stats, estimate_training_time

st.set_page_config(page_title="Upload Data · LLM Trainer", page_icon="📂", layout="wide")
init_session()

st.title("📂 Upload Your Dataset")

# ── Source selector ───────────────────────────────────────────
src = st.radio(
    "Choose data source",
    ["📁 Upload my own file", "🎓 Use a built-in sample dataset"],
    horizontal=True,
    help="New here? Try a sample dataset first to see how the platform works end-to-end.",
)
st.divider()

df        = None
filename  = None
is_sample = False

# ════════════════════════════════════════════════════════════
# PATH A — built-in sample datasets
# ════════════════════════════════════════════════════════════
if "sample" in src:
    is_sample = True
    st.subheader("🎓 Built-in Sample Datasets")
    st.caption("Pick one to load instantly and try the full training flow without preparing any data.")

    cols = st.columns(len(SAMPLE_DATASETS))
    chosen = None
    for i, (name, info) in enumerate(SAMPLE_DATASETS.items()):
        with cols[i]:
            st.markdown(f"""
            <div style='background:#1E2130;border:1px solid #2E3250;border-radius:12px;
                        padding:1rem;text-align:center;min-height:160px;'>
              <div style='font-size:1.1rem;font-weight:700;color:#6C63FF;margin-bottom:.4rem;'>{name}</div>
              <div style='font-size:.8rem;color:#aaa;margin-bottom:.6rem;'>{info['description']}</div>
              <div style='font-size:.75rem;color:#888;'>{info['rows']} rows · {info['format']}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Load →", key=f"load_{i}", use_container_width=True):
                chosen = name

    # persist selection in session
    if chosen:
        st.session_state["_chosen_sample"] = chosen
    chosen = st.session_state.get("_chosen_sample")

    if chosen:
        df       = get_sample_dataframe(chosen)
        filename = f"sample_{chosen.lower().replace(' ','_')}.csv"
        info     = get_sample_info(chosen)

        st.success(f"✅ Loaded: **{chosen}** — {info['rows']} rows · {info['format']}")
        st.info(f"💡 **Use case:** {info['use_case']}")

# ════════════════════════════════════════════════════════════
# PATH B — user upload
# ════════════════════════════════════════════════════════════
else:
    st.caption("Supported: CSV · JSONL · JSON · TXT · Parquet  ·  Max 200 MB")
    uploaded = st.file_uploader(
        "Drop your file here or click to browse",
        type=["csv","jsonl","json","txt","parquet"],
        help="Your data stays in this session — it is NOT sent to any external server.",
    )
    if uploaded is None:
        st.info("👆 Upload a file above, or switch to **Use a built-in sample dataset** to try without your own data.")
        st.stop()

    filename = uploaded.name
    ext      = Path(filename).suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(uploaded)
        elif ext in (".jsonl", ".json"):
            df = pd.read_json(uploaded, lines=(ext == ".jsonl"))
        elif ext == ".parquet":
            df = pd.read_parquet(uploaded)
        elif ext == ".txt":
            lines = uploaded.read().decode("utf-8").splitlines()
            df    = pd.DataFrame({"text": [l for l in lines if l.strip()]})
        else:
            st.error(f"Unsupported format: {ext}"); st.stop()
    except Exception as e:
        st.error(f"Failed to parse file: {e}"); st.stop()

    save_path = UPLOAD_DIR / filename
    save_path.write_bytes(uploaded.getvalue())

if df is None:
    st.stop()

# ════════════════════════════════════════════════════════════
# Common: dataset processing & display
# ════════════════════════════════════════════════════════════

# Detect type
ds_type = detect_dataset_type(df)

# Save path
save_path = UPLOAD_DIR / (filename or "dataset.csv")
df.to_csv(save_path, index=False)

# ── Infer column names ────────────────────────────────────────
has_prompt   = "prompt"   in df.columns
has_response = "response" in df.columns
has_text     = "text"     in df.columns

dataset_info = {
    "filename":              filename,
    "path":                  str(save_path),
    "rows":                  len(df),
    "columns":               list(df.columns),
    "size_kb":               round(df.memory_usage(deep=True).sum() / 1024, 1),
    "format":                Path(filename).suffix.lower(),
    "has_instruction_cols":  has_prompt and has_response,
    "dataset_type":          ds_type,
    "is_sample":             is_sample,
}
st.session_state["dataset_info"] = dataset_info

cfg = st.session_state["config"]
if has_text:     cfg["text_column"]     = "text"
if has_prompt:   cfg["prompt_column"]   = "prompt"
if has_response: cfg["response_column"] = "response"

# ── Metrics ───────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows",    f"{len(df):,}")
m2.metric("Columns", len(df.columns))
m3.metric("Memory",  f"{dataset_info['size_kb']} KB")
m4.metric("Type",    ds_type.capitalize())
est = estimate_training_time(len(df), "lora", 3)
m5.metric("Est. training time*", est, help="*LoRA, 3 epochs on free T4 GPU. Very rough estimate.")

st.divider()

# ── Dataset type badge ────────────────────────────────────────
if ds_type == "instruction":
    st.info(
        "🎓 **Instruction-tuning dataset** detected — `prompt` + `response` columns found.  \n"
        "The model will learn to follow instructions using the `### Instruction / ### Response` template.  \n"
        "This is the **best format** for building chatbots and task-specific assistants."
    )
elif ds_type == "text":
    st.info(
        "📝 **Text completion dataset** detected — `text` column found.  \n"
        "The model will be trained for next-token prediction (style transfer, text generation)."
    )
else:
    st.warning(
        "⚠️ Column names not recognised.  \n"
        "Tip: rename your columns to `prompt` + `response` (instruction tuning) or `text` (text generation).  \n"
        "Or use the **Column mapping** section below to map your columns manually."
    )

# ── Column mapping ────────────────────────────────────────────
with st.expander("🗂️ Column mapping (click to customise)", expanded=(ds_type == "unknown")):
    cols_list = list(df.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        idx = cols_list.index(cfg.get("text_column","")) if cfg.get("text_column","") in cols_list else 0
        cfg["text_column"] = st.selectbox("Text column (for text datasets)", cols_list, index=idx)
    with c2:
        idx = cols_list.index(cfg.get("prompt_column","")) if cfg.get("prompt_column","") in cols_list else 0
        cfg["prompt_column"] = st.selectbox("Prompt column (for instruction datasets)", cols_list, index=idx)
    with c3:
        idx = cols_list.index(cfg.get("response_column","")) if cfg.get("response_column","") in cols_list else 0
        cfg["response_column"] = st.selectbox("Response column (for instruction datasets)", cols_list, index=idx)

st.divider()

# ── Data preview ──────────────────────────────────────────────
st.subheader("👀 Data Preview")

tab_prev, tab_stats, tab_nulls = st.tabs(["📋 Preview", "📊 Statistics", "🔍 Missing Values"])
with tab_prev:
    st.dataframe(df.head(10), use_container_width=True)
with tab_stats:
    st.dataframe(df.describe(include="all").T, use_container_width=True)
with tab_nulls:
    null_df = df.isnull().sum().reset_index()
    null_df.columns = ["Column", "Null Count"]
    null_df["% Missing"] = (null_df["Null Count"] / len(df) * 100).round(2)
    st.dataframe(null_df, use_container_width=True)
    if null_df["Null Count"].sum() > 0:
        st.warning("⚠️ Missing values detected. The training code will drop null rows automatically.")
    else:
        st.success("✅ No missing values found!")

# ── Download sample as CSV ────────────────────────────────────
if is_sample:
    st.divider()
    st.subheader("⬇️ Download Sample Dataset")
    st.caption("Download this sample and inspect/edit it to understand the expected format for your own data.")
    st.download_button(
        "⬇️ Download as CSV",
        df.to_csv(index=False),
        filename,
        "text/csv",
        use_container_width=True,
    )

st.divider()
if st.button("Next → Configure Training ⚙️", type="primary", use_container_width=True):
    st.switch_page("pages/2_Configure_Training.py")
