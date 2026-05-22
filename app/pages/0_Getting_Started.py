"""
Page 0 — Getting Started
Interactive step-by-step guide embedded directly in the app.
Anyone landing here for the first time gets a full walkthrough
without needing to read any external documentation.
"""
import streamlit as st
from app.utils.state import init_session

st.set_page_config(
    page_title="Getting Started · LLM Trainer",
    page_icon="🚀",
    layout="wide",
)
init_session()

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
.guide-hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #6C63FF44;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
}
.guide-hero h1 { font-size: 2.2rem; font-weight: 800; color: #fff; margin: 0 0 .5rem; }
.guide-hero p  { color: #ccc; font-size: 1.05rem; margin: 0; }

.step-box {
    background: #1E2130;
    border-left: 4px solid #6C63FF;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.step-box h3 { margin: 0 0 .4rem; color: #6C63FF; font-size: 1.1rem; }
.step-box p  { margin: 0; color: #ddd; line-height: 1.6; }

.tip-box {
    background: #0d2137;
    border: 1px solid #3ECFCF44;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: .6rem 0;
}
.warn-box {
    background: #2a1a0a;
    border: 1px solid #FFA50066;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: .6rem 0;
}
.concept-card {
    background: #1E2130;
    border: 1px solid #2E3250;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    height: 100%;
}
.concept-card h4 { color: #6C63FF; margin: 0 0 .4rem; }
.concept-card p  { color: #ccc; margin: 0; font-size: .9rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────
st.markdown("""
<div class="guide-hero">
  <h1>🚀 Getting Started — Complete Beginner's Guide</h1>
  <p>From zero to a trained AI model on your own data. No credit card. No GPU needed on your computer.
     Works for absolute beginners and experienced ML engineers alike.</p>
</div>
""", unsafe_allow_html=True)

# ── Who is this for? ──────────────────────────────────────────
st.subheader("👥 Who Is This For?")
u1, u2, u3 = st.columns(3)
with u1:
    st.markdown("""
    <div class="concept-card">
      <h4>🧑‍💼 Business / Non-technical</h4>
      <p>You have a collection of documents, FAQs, or conversations and want an AI that knows your domain.
         You don't need to write a single line of code — just upload and click.</p>
    </div>
    """, unsafe_allow_html=True)
with u2:
    st.markdown("""
    <div class="concept-card">
      <h4>👩‍🔬 Data Scientist / ML Engineer</h4>
      <p>You want to fine-tune a 7B model with QLoRA, push it to HuggingFace Hub, and integrate it via the
         Inference API. All hyperparameters are exposed and the generated notebook is fully editable.</p>
    </div>
    """, unsafe_allow_html=True)
with u3:
    st.markdown("""
    <div class="concept-card">
      <h4>🧑‍🎓 Student / Researcher</h4>
      <p>You want to experiment with LLM fine-tuning without paying for cloud compute.
         This gives you free T4 GPU time via Google Colab and a clean PyTorch Lightning codebase to learn from.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Key concepts ──────────────────────────────────────────────
st.subheader("🧠 Key Concepts (Plain English)")

with st.expander("What is fine-tuning?", expanded=False):
    st.markdown("""
    A large language model (LLM) like GPT-2 or Mistral is pre-trained on billions of web pages — it knows
    a lot about language in general, but nothing about **your** specific domain.

    **Fine-tuning** is the process of continuing to train the model on **your** data so it learns your:
    - Writing style
    - Domain knowledge (medicine, law, customer support, code, etc.)
    - Q&A pairs or instructions specific to your use case

    Think of it like hiring an expert who already knows how to speak and write (the pre-trained model),
    then giving them a 2-week intensive course on your company's products and procedures (fine-tuning).
    """)

with st.expander("What is LoRA / QLoRA?", expanded=False):
    st.markdown("""
    **The problem:** A 7-billion-parameter model has 7 billion numbers. Updating all of them during training
    needs ~28 GB of GPU memory — far more than a free T4 GPU (15 GB).

    **LoRA (Low-Rank Adaptation)** is a clever trick: instead of changing the 7B original weights,
    it inserts tiny *adapter* matrices (just ~0.1% of the original size) and only trains those.
    The original model weights are frozen. Result: same quality, 100x less memory.

    **QLoRA** goes further — it compresses the base model to 4-bit precision (from 16-bit) before inserting
    the adapters. This cuts VRAM usage by another ~75%, letting you fine-tune a 7B model on a **free T4 GPU**.

    **Rule of thumb:**
    | Model size | Recommended mode | Fits on free T4? |
    |---|---|---|
    | GPT-2 (124M) | Full or LoRA | ✅ Yes |
    | TinyLlama (1.1B) | LoRA / QLoRA | ✅ Yes |
    | Phi-2 (2.7B) | QLoRA | ✅ Yes |
    | Mistral 7B | QLoRA | ✅ Yes (barely) |
    | Llama 70B | QLoRA | ❌ Needs A100 |
    """)

with st.expander("What is Google Colab and why do we use it?", expanded=False):
    st.markdown("""
    [Google Colab](https://colab.research.google.com) is a free cloud service by Google that gives you
    access to a real GPU (NVIDIA T4, 15 GB VRAM) for free — no credit card, no setup.

    **How this platform uses it:**
    1. You configure your training here in the Streamlit app.
    2. We generate a `.ipynb` notebook file with all your settings baked in.
    3. You open it in Colab with one click.
    4. Colab provides the GPU and runs your training.
    5. Your model gets saved and (optionally) uploaded to HuggingFace.

    **Free tier limits:**
    - ~4–12 hours of GPU time per session
    - T4 GPU (15 GB VRAM) — sufficient for QLoRA up to 7B models
    - Sessions disconnect if idle for ~90 minutes
    - No persistent storage (save to Google Drive or HuggingFace Hub)
    """)

with st.expander("What dataset formats are supported?", expanded=False):
    st.markdown("""
    We support four formats:

    **CSV** — spreadsheet format, easiest for non-technical users
    ```
    prompt,response
    "What is your return policy?","We offer 30-day returns on all items."
    "How do I track my order?","Visit our tracking page and enter your order ID."
    ```

    **JSONL** — one JSON object per line, common in ML datasets
    ```json
    {"prompt": "What is photosynthesis?", "response": "The process by which plants convert sunlight into energy."}
    {"prompt": "Explain gravity", "response": "Gravity is a force that attracts objects with mass toward each other."}
    ```

    **TXT** — plain text, one document per line (good for style transfer / creative writing)
    ```
    The quick brown fox jumps over the lazy dog.
    Once upon a time in a land far away...
    ```

    **Parquet** — efficient binary format, used by HuggingFace datasets.

    **Column naming conventions:**
    - For instruction tuning: name your columns `prompt` and `response` (or map them in the UI)
    - For text generation: name your column `text` (or map it in the UI)
    """)

st.divider()

# ── Step-by-step walkthrough ───────────────────────────────────
st.subheader("📋 Step-by-Step Walkthrough")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Step 1 · Prepare Data",
    "Step 2 · Upload & Preview",
    "Step 3 · Configure",
    "Step 4 · Colab Training",
    "Step 5 · Use Your Model",
])

with tab1:
    st.markdown("### 📁 Step 1 — Prepare Your Dataset")
    st.markdown("""
    Before uploading, make sure your data is in a supported format. Here are templates for each use case:
    """)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🎓 Instruction / Q&A tuning (recommended for most users)**")
        st.code("""prompt,response
"Summarize this text: {text}","Here is a summary: ..."
"Translate to French: {text}","Voici la traduction: ..."
"Answer as a doctor: {question}","Based on medical knowledge: ..."
""", language="text")
        st.caption("Save as `my_data.csv`. Minimum recommended: 100+ rows.")

    with c2:
        st.markdown("**📝 Text completion / style tuning**")
        st.code("""text
"In our company, all support tickets should be..."
"The patient presented with symptoms of..."
"def calculate_tax(income): # This function..."
""", language="text")
        st.caption("Save as `my_data.csv`. Each row = one training example.")

    st.markdown("""
    <div class="tip-box">
    💡 <strong>Tips for better results:</strong><br>
    • <strong>Quality over quantity</strong> — 500 high-quality examples beat 5,000 noisy ones<br>
    • <strong>Consistent format</strong> — keep your prompts / responses in the same style throughout<br>
    • <strong>Representative coverage</strong> — include examples of all the things you want the model to do<br>
    • <strong>Minimum viable dataset</strong> — 50 rows is enough to see results; 500+ for production quality<br>
    • <strong>Clean your data</strong> — remove HTML, fix encoding issues, deduplicate
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Don't include:</strong> personally identifiable information (PII), passwords, API keys,
    copyrighted text you don't have rights to, or harmful/illegal content.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 📂 Step 2 — Upload & Preview")
    st.markdown("""
    Go to **Upload Data** in the sidebar (or click below).

    What happens when you upload:
    1. Your file is parsed and stored temporarily in this session
    2. The app auto-detects whether it's an instruction dataset or text dataset
    3. You see a live preview of the first 10 rows
    4. Column statistics and null-value counts are shown
    5. You can remap column names if they don't match `prompt`/`response`/`text`

    **If you don't have data yet**, click the button below to load a sample dataset and explore the platform:
    """)
    if st.button("📂 Go to Upload Data →", type="primary", use_container_width=True):
        st.switch_page("pages/1_Upload_Data.py")

    st.markdown("""
    <div class="tip-box">
    💡 <strong>New to this?</strong> The Upload page has a <strong>"Try a sample dataset"</strong> button
    that loads a built-in example dataset so you can explore the full flow without preparing any data first.
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("### ⚙️ Step 3 — Configure Training")
    st.markdown("""
    Go to **Configure Training** in the sidebar. Here's what each setting means:
    """)

    st.markdown("""
    **🎯 Training Mode** — the most important choice:
    """)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="step-box">
        <h3>LoRA / QLoRA ✅ Recommended</h3>
        <p>Best for 99% of users. Fast, memory-efficient, works on free Colab T4 GPU with any model up to 7B.
        QLoRA adds 4-bit compression for even larger models.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="step-box">
        <h3>Full Fine-tuning ⚠️ Advanced</h3>
        <p>Updates every parameter. Only feasible for small models (GPT-2) on free Colab.
        Use this if you need maximum accuracy and have access to a large GPU.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="step-box">
        <h3>Prompt-based 🟡 Instant</h3>
        <p>No training at all. Your dataset is used as few-shot examples in the prompt.
        Instant results, no GPU needed. Great for quick testing.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **🤖 Model Selection guide:**

    | Your situation | Recommended model |
    |---|---|
    | First time, just exploring | **GPT-2 (124M)** — trains in 5 min, no token needed |
    | Want instruction-following on free Colab | **TinyLlama 1.1B** or **Phi-2 2.7B** |
    | Best quality on free Colab | **Mistral 7B** with QLoRA |
    | Need a small, fast model | **DistilGPT-2** |
    | Research / custom architecture | **OPT 1.3B** or **Falcon 7B** |

    **📐 Key hyperparameters:**

    | Setting | Beginner default | What it controls |
    |---|---|---|
    | Epochs | 3 | How many times the model sees your whole dataset |
    | Batch size | 4 | Samples processed at once (lower = less VRAM) |
    | Gradient accumulation | 4 | Simulates batch of 16 (4×4) without extra VRAM |
    | Learning rate | 2e-4 | How fast the model adapts (don't go above 5e-4) |
    | LoRA rank | 16 | Adapter size — higher = better quality, more VRAM |
    | Max seq length | 512 | Max tokens per example (longer = more VRAM) |
    """)

    if st.button("⚙️ Go to Configure Training →", type="primary", use_container_width=True):
        st.switch_page("pages/2_Configure_Training.py")

with tab4:
    st.markdown("### 🖥️ Step 4 — Running Training in Google Colab")
    st.markdown("After clicking **Generate Notebook**, follow these exact steps:")

    steps_colab = [
        ("Open the notebook", "Click **Open in Google Colab** (if you provided a GitHub token) OR go to [colab.research.google.com](https://colab.research.google.com), click **File → Upload notebook**, and upload the downloaded `.ipynb` file."),
        ("Switch to GPU runtime", "In Colab, click **Runtime** (top menu) → **Change runtime type** → set **Hardware accelerator** to **T4 GPU** → click **Save**. This is FREE — no credit card."),
        ("Run the GPU check cell", "Click the first code cell (`▶`) or press `Shift+Enter`. It should say `✅ GPU: Tesla T4`. If you see an error, repeat step 2."),
        ("Install dependencies", "Run the install cell. This takes 2–3 minutes. You'll see progress bars. Wait for `✅ All packages installed`."),
        ("Set your HuggingFace token", "In the **Tokens** cell, paste your HF token (get one free at huggingface.co/settings/tokens). This is needed for gated models (Llama, Gemma) and to push your trained model."),
        ("Upload your dataset", "Run the **Upload Dataset** cell. A file picker appears — select your CSV/JSONL/TXT file from your computer."),
        ("Review the config", "Glance at the CONFIG cell to confirm your settings are correct. You can edit any value here before continuing."),
        ("Run training", "Click **Runtime → Run all** OR run the training cell manually. You'll see a live progress bar with loss values."),
        ("Watch the metrics", "Training logs show `train/loss` and `val/loss`. Both should decrease over time. If val/loss stops decreasing, early stopping kicks in automatically."),
        ("Save your model", "Run the **Save & Push** cell. Your model saves to `/content/outputs/`. If you set `PUSH_TO_HUB=True`, it uploads to HuggingFace."),
        ("Download the model", "Run the **Download** cell to get a `.zip` of your model weights for local use."),
    ]
    for i, (title, desc) in enumerate(steps_colab, 1):
        st.markdown(f"""
        <div class="step-box">
        <h3>{i}. {title}</h3>
        <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
    ⚠️ <strong>Colab session limits:</strong> Free sessions disconnect after ~12 hours or ~90 min of idle time.
    For datasets larger than ~10,000 rows, mount Google Drive first (<code>from google.colab import drive; drive.mount('/content/drive')</code>)
    and save your model there to avoid losing progress.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-box">
    💡 <strong>Pro tip:</strong> Run <code>!nvidia-smi</code> at any time to see GPU memory usage.
    If you get an OOM (out-of-memory) error, reduce <code>batch_size</code> to 1 or 2,
    or reduce <code>max_seq_length</code> to 256.
    </div>
    """, unsafe_allow_html=True)

with tab5:
    st.markdown("### 🎉 Step 5 — Use Your Trained Model")
    st.markdown("""
    Once training is done, you have several options:

    **Option A — Test it right here in this app:**
    Go to **Test Model** in the sidebar. Enter your HuggingFace repo ID (e.g. `yourname/my-model`)
    and start chatting immediately. No local GPU needed — inference runs on CPU.
    """)
    if st.button("💬 Go to Test Model →", use_container_width=True):
        st.switch_page("pages/4_Test_Model.py")

    st.markdown("""
    **Option B — Use it via the HuggingFace Inference API (free):**
    ```python
    from huggingface_hub import InferenceClient
    client = InferenceClient("yourname/my-model")
    response = client.text_generation("Your prompt here", max_new_tokens=200)
    print(response)
    ```

    **Option C — Load locally:**
    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    base  = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = PeftModel.from_pretrained(base, "yourname/my-model")
    tok   = AutoTokenizer.from_pretrained("yourname/my-model")

    inputs = tok("### Instruction:\\nExplain LoRA\\n\\n### Response:\\n", return_tensors="pt")
    out    = model.generate(**inputs, max_new_tokens=200)
    print(tok.decode(out[0]))
    ```

    **Option D — Deploy with Streamlit / Gradio / FastAPI:**
    Your model folder (downloaded from Colab or pulled from HuggingFace) works with any standard
    `transformers` `pipeline()` call. Drop it into any web framework.
    """)

st.divider()

# ── Troubleshooting ───────────────────────────────────────────
st.subheader("🛠️ Troubleshooting")

faqs = [
    ("CUDA out of memory (OOM) error in Colab",
     "Reduce `batch_size` to 1 or 2. Reduce `max_seq_length` to 256. Enable QLoRA (4-bit). "
     "Make sure you selected T4 GPU runtime, not CPU."),
    ("Model not generating coherent text after training",
     "You likely need more data or more epochs. Try 3–5 epochs with at least 200 rows. "
     "Check that your dataset is clean and consistent. Lower the learning rate to 1e-4."),
    ("HuggingFace download is slow or failing",
     "Some models (Llama, Gemma) require you to accept a license on huggingface.co and provide an HF token. "
     "Set `HF_TOKEN` in the Colab notebook's token cell."),
    ("Session disconnects mid-training",
     "Mount Google Drive and save checkpoints there. Enable ModelCheckpoint (it's on by default) — "
     "the best checkpoint is saved even if the session disconnects."),
    ("File upload widget doesn't appear in Colab",
     "Make sure you're running the cell, not just clicking the play button. Try Runtime → Restart and run all."),
    ("ValueError: text column not found",
     "Your dataset column names don't match the configured column names. Go back to the Upload page and "
     "remap your columns using the Column mapping expander."),
    ("Loss is NaN from the start",
     "Your learning rate is too high. Try 1e-4 or lower. Also check your dataset for empty or malformed rows."),
    ("Training loss decreases but val loss goes up",
     "Your model is overfitting. Reduce epochs (try 1–2). Add more data. Increase LoRA dropout to 0.1."),
]

for question, answer in faqs:
    with st.expander(f"❓ {question}"):
        st.info(answer)

st.divider()

# ── Quick links ───────────────────────────────────────────────
st.subheader("🔗 Useful External Links")
c1, c2, c3, c4 = st.columns(4)
c1.link_button("🤗 HuggingFace Hub",       "https://huggingface.co",                         use_container_width=True)
c2.link_button("⚡ Google Colab",           "https://colab.research.google.com",              use_container_width=True)
c3.link_button("⚡ PyTorch Lightning Docs", "https://lightning.ai/docs/pytorch/stable/",      use_container_width=True)
c4.link_button("🔧 PEFT / LoRA Docs",       "https://huggingface.co/docs/peft/index",         use_container_width=True)

st.caption("🤖 LLM Trainer SaaS · Free cloud LLM fine-tuning · No credit card needed")
