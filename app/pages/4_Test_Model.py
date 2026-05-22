"""Page 4 — Test a trained model (local or from HuggingFace Hub)."""
import streamlit as st

from app.utils.state import init_session, list_jobs, DONE

st.set_page_config(page_title="Test Model · LLM Trainer", page_icon="💬", layout="wide")
init_session()

st.title("💬 Test Your Model")
st.caption("Run inference against a locally saved or HuggingFace-hosted model.")

# ── Source selector ───────────────────────────────────────────
source = st.radio("Model source", ["HuggingFace Hub (recommended)", "Local path"], horizontal=True)

if source == "HuggingFace Hub (recommended)":
    # Pre-populate from completed jobs
    done_jobs  = [j for j in list_jobs() if j["status"] == DONE and j["config"].get("hf_repo_id")]
    hf_options = [j["config"]["hf_repo_id"] for j in done_jobs]

    if hf_options:
        repo_id = st.selectbox("Select a completed job's model", hf_options)
    else:
        repo_id = st.text_input("HuggingFace repo ID",
                                 placeholder="username/my-finetuned-model")
    model_path = repo_id
else:
    model_path = st.text_input("Local model directory", placeholder="/path/to/outputs")

hf_token = st.text_input("HuggingFace token (for private/gated models)", type="password")

# ── Generation settings ───────────────────────────────────────
with st.expander("🎛️ Generation settings"):
    c1, c2, c3 = st.columns(3)
    max_new   = c1.slider("Max new tokens",   16, 1024, 200, 16)
    temp      = c2.slider("Temperature",      0.1, 2.0, 0.7, 0.05)
    top_p     = c3.slider("Top-p",            0.1, 1.0, 0.9, 0.05)
    rep_pen   = c1.slider("Repetition penalty", 1.0, 2.0, 1.1, 0.05)
    do_sample = c2.checkbox("Sampling (uncheck = greedy)", value=True)

st.divider()

# ── Load model ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model(path: str, token: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    tok   = AutoTokenizer.from_pretrained(path, trust_remote_code=True, token=token or None)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, token=token or None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tok)
    return pipe

if not model_path:
    st.info("👆 Enter a model path or repo ID above, then start chatting.")
    st.stop()

try:
    pipe = load_model(model_path, hf_token)
    st.success(f"✅ Model loaded: `{model_path}`")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.divider()

# ── Chat interface ────────────────────────────────────────────
st.subheader("💬 Chat")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if prompt := st.chat_input("Ask your model something …"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating …"):
            try:
                out = pipe(
                    prompt,
                    max_new_tokens     = max_new,
                    temperature        = temp,
                    top_p              = top_p,
                    repetition_penalty = rep_pen,
                    do_sample          = do_sample,
                )
                response = out[0]["generated_text"][len(prompt):].strip()
            except Exception as e:
                response = f"Error: {e}"

        st.write(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})

# Clear chat
if st.session_state["messages"]:
    if st.button("🗑️ Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()

# ── Batch inference ───────────────────────────────────────────
st.divider()
st.subheader("📦 Batch Inference")
st.caption("Run your model against multiple prompts at once.")

batch_input = st.text_area(
    "Enter prompts (one per line)",
    height=120,
    placeholder="What is the capital of France?\nExplain quantum computing in simple terms.\n...",
)

if st.button("▶️ Run batch inference", use_container_width=True):
    prompts = [p.strip() for p in batch_input.splitlines() if p.strip()]
    if not prompts:
        st.warning("Enter at least one prompt.")
    else:
        results = []
        progress = st.progress(0)
        for i, p in enumerate(prompts):
            try:
                out = pipe(p, max_new_tokens=max_new, temperature=temp, top_p=top_p, do_sample=do_sample)
                results.append({"Prompt": p, "Response": out[0]["generated_text"][len(p):].strip()})
            except Exception as e:
                results.append({"Prompt": p, "Response": f"ERROR: {e}"})
            progress.progress((i + 1) / len(prompts))

        import pandas as pd
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download results CSV",
            df.to_csv(index=False),
            "inference_results.csv",
            "text/csv",
        )
