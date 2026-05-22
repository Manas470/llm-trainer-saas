# 🤖 LLM Trainer SaaS

**Train any LLM on your own data — completely free, no credit card required.**

Built with PyTorch Lightning · Google Colab · Streamlit · HuggingFace

> **Complete beginner?** Jump straight to [▶ Start Here](#-start-here--zero-to-fine-tuned-model-for-complete-beginners). No ML background needed.

---

## What Is This?

This is a web app that lets you **teach an AI model your own knowledge** — using your own text data — without paying for a single GPU, without writing a single line of training code, and without any machine learning experience.

You provide the data. The app generates the training code. Google runs it for free.

**Real examples of what you can build:**

- 🏥 A medical Q&A assistant trained on your clinic's protocols
- 📚 A tutor that answers in your exact teaching style
- 🛒 A customer support bot trained on your product documentation
- 🌍 A model that understands a low-resource language or dialect
- 💼 A domain expert for legal, finance, or engineering queries

---

## ▶ Start Here — Zero to Fine-tuned Model (For Complete Beginners)

**Total time: ~15 minutes setup + 30–90 minutes training (Google does that part)**

### Step 1 — Get the app running on your computer

**Option A: One double-click (Mac only)**
1. Download or clone this repo
2. Double-click `launch_app.command` in Finder
3. A Terminal window opens, installs everything automatically, and launches the app
4. Your browser opens at `http://localhost:8501` — you're in!

**Option B: Manual (Mac / Windows / Linux)**
```bash
# 1. Clone the repo
git clone https://github.com/Manas470/llm-trainer-saas
cd llm-trainer-saas

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app/main.py
```
Then open `http://localhost:8501` in your browser.

> **Don't have Python?** Download it from [python.org/downloads](https://python.org/downloads) — get Python 3.10 or newer. Then re-run the steps above.

---

### Step 2 — Prepare your dataset

Your data needs to be a file where each row has a **question** (or prompt) and an **answer** (or response). The simplest format is a CSV file with two columns.

**Example — customer support dataset (save this as `my_data.csv`):**

```csv
instruction,response
"How do I reset my password?","Go to Settings → Account → Reset Password. You'll get an email within 2 minutes."
"What payment methods do you accept?","We accept Visa, Mastercard, PayPal, and bank transfers."
"How long does shipping take?","Standard shipping is 3–5 business days. Express is 1–2 days."
"Can I cancel my order?","Yes, orders can be cancelled within 24 hours of placing them via your account dashboard."
```

**Don't have data yet?** No problem. The app includes built-in sample datasets (customer support, medical Q&A, code generation, creative writing). You can run a complete training job on a sample before using your own data.

**Supported file formats:** CSV · JSONL · JSON · TXT · Parquet

**How many rows do I need?**
| Rows | What to expect |
|---|---|
| 20–50 | Basic proof of concept — model picks up style but not much knowledge |
| 100–200 | Noticeable improvement on your domain |
| 500–2,000 | Solid domain-specific model |
| 5,000+ | Production-quality fine-tune |

---

### Step 3 — Upload your data and configure training

1. Open the app and click **"Upload Data"** in the left sidebar
2. Upload your CSV/JSONL file
3. The app previews your data and flags any issues (missing columns, empty rows, etc.)
4. Click **"Configure Training"** in the sidebar
5. Choose your settings — recommended defaults for beginners:

| Setting | Beginner recommendation | Why |
|---|---|---|
| **Model** | TinyLlama 1.1B | Small, trains fast, works perfectly on free Colab |
| **Training mode** | LoRA | Uses minimal GPU memory, excellent results |
| **Epochs** | 3 | Good balance — not underfit, not overfit |
| **Batch size** | 4 | Safe for the free T4 GPU (15 GB VRAM) |
| Everything else | Leave as default | Defaults are tuned for free Colab |

6. Click **"Generate Notebook"** — the app builds a complete training notebook with your exact settings baked in

---

### Step 4 — Run training for free (all you need is a Gmail account)

1. Click the **"Open in Google Colab"** button the app gives you
2. Sign in with your **Gmail account** — that's the only account you need
3. In Colab: click **Runtime → Change runtime type → select T4 GPU → Save**
4. Click **Runtime → Run All**
5. The notebook runs automatically — training progress appears in the output cells

**That's it.** Google's free T4 GPU is now training your model. You can leave the tab open in the background or come back later — progress is saved to your Google Drive automatically.

> ⏱ **How long will it take?**
> - 100 rows, TinyLlama → ~15–25 minutes
> - 500 rows, TinyLlama → ~45–70 minutes
> - 500 rows, Mistral 7B (QLoRA) → ~90–120 minutes
> - 2,000 rows, Mistral 7B → ~4–6 hours

---

### Step 5 — Use your trained model

When training finishes, the notebook saves your model to your Google Drive automatically.

**Test it immediately:** The notebook includes a chat cell at the very end. Run it and ask your model questions — see the difference from the base model.

**Share it publicly (optional):** Add your HuggingFace token in the notebook's "Push to Hub" cell and upload your model so anyone can use it via API — free hosting on [huggingface.co](https://huggingface.co).

**Use it in code:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("path/to/your/saved/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/your/saved/model")

inputs = tokenizer("Your question here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🗺️ How It All Fits Together

```
Your data (CSV/JSONL)
        │
        ▼
┌────────────────────────────────────┐
│       LLM Trainer SaaS App         │
│  (this repo — runs on your laptop  │
│   or on Streamlit Community Cloud) │
│                                    │
│   Upload → Configure → Generate   │
└───────────────┬────────────────────┘
                │  produces a complete .ipynb notebook
                ▼
         GitHub Gist (free)
                │  one-click "Open in Colab" link
                ▼
     ┌───────────────────────┐
     │  Google Colab T4 GPU  │  ← FREE — just need Gmail
     │  PyTorch Lightning 2  │
     │  + QLoRA (4-bit PEFT) │
     └──────────┬────────────┘
                │  saves trained model
                ▼
     ┌───────────────────────┐
     │   Your Google Drive   │  ← or push to HuggingFace Hub
     └───────────────────────┘
```

**The core idea:** This app never runs training itself. It generates production-quality training code with your exact configuration, then hands it off to Google's free GPU infrastructure. You pay nothing. Google pays for the compute.

---

## 🔑 Optional: API Tokens (Unlocks Extra Features)

These are completely optional. The app works without them.

| Token | Where to get it | What it unlocks |
|---|---|---|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → New token → Read | Access gated models (Llama 3, Mistral) + push model to Hub |
| `GITHUB_TOKEN` | [github.com/settings/tokens](https://github.com/settings/tokens) → New token → tick **gist** | One-click "Open in Colab" sharing via Gist URL |

To add them, copy the example file and fill in your tokens:
```bash
cp .env.example .env
# Open .env in any text editor and paste your tokens
```

---

## 🧠 Which Training Mode Should I Use?

### LoRA / QLoRA ✅ Right for 99% of users
- Trains only ~1% of the model's parameters by adding small "adapter" layers
- QLoRA loads the base model in 4-bit — brings a 7B model from 28 GB down to ~6 GB VRAM
- Runs on the free Colab T4 (15 GB VRAM) with room to spare
- Quality: 97–99% as good as full fine-tuning for domain adaptation tasks

### Full Fine-tuning
- Updates every single weight in the model — theoretically better but rarely worth it
- A 7B model needs ~28 GB VRAM minimum. Free Colab T4 has 15 GB. Doesn't fit.
- Realistic only for tiny models like GPT-2 (124M params) on the free tier
- Use only if you have access to an A100 or similar

### Prompt-based (no training)
- Uses your data as few-shot examples fed into the prompt — no weights updated at all
- Instant results, no GPU needed, works on CPU
- Great for a 2-minute experiment before committing to real training
- Less powerful than actual fine-tuning for anything beyond a handful of examples

---

## 🛠️ Deploy Your Own Public Instance (Free)

Want to give your team or community access to this tool without asking them to clone a repo?

1. **Fork** this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Connect your GitHub account, select your fork
4. Set **Main file path** to `app/main.py`
5. Under **Advanced settings → Secrets**, optionally add:
   ```toml
   HF_TOKEN = "hf_..."
   GITHUB_TOKEN = "ghp_..."
   ```
6. Click **Deploy**

Your app is now live at a public URL — share it with anyone. Zero server management, zero cost.

---

## 📁 Project Structure (For Developers)

```
llm-trainer-saas/
├── app/
│   ├── main.py                      # Home / onboarding page
│   ├── pages/
│   │   ├── 0_Getting_Started.py     # Beginner walkthrough page
│   │   ├── 1_Upload_Data.py         # Dataset upload & validation
│   │   ├── 2_Configure_Training.py  # Hyperparameters + notebook generation
│   │   ├── 3_Training_Jobs.py       # Job tracking dashboard
│   │   ├── 4_Test_Model.py          # Chat with your trained model
│   │   ├── 5_Dashboard.py           # Analytics & charts
│   │   └── 6_About_and_Decisions.py # Why each technology was chosen
│   ├── components/
│   │   ├── notebook_generator.py    # Builds the Colab .ipynb dynamically
│   │   ├── data_processor.py        # CSV/JSONL parsing, cleaning, validation
│   │   └── sample_datasets.py       # Built-in demo datasets for first-time users
│   └── utils/
│       ├── config.py                # Constants, BASE_MODELS list, TrainingConfig
│       └── state.py                 # Session state + disk-backed job persistence
├── training/
│   ├── data_module.py               # PyTorch Lightning DataModule
│   ├── lightning_module.py          # LightningModule (LoRA + Full fine-tuning)
│   └── trainer.py                   # CLI entry-point for local/server training
├── launch_app.command               # macOS double-click launcher
├── requirements.txt                 # All Python dependencies
├── .env.example                     # Template for API tokens
├── THOUGHTS.md                      # Builder's perspective: every decision explained
└── .streamlit/config.toml           # Dark theme + usage stats off
```

---

## 💡 Tips for Best Results on Free Colab

| Tip | Details |
|---|---|
| **Always select T4 GPU** | Runtime → Change runtime type → T4 GPU → Save (before running anything) |
| **Run the Drive mount cell first** | Saves checkpoints automatically — you won't lose progress if the session drops |
| **Start with TinyLlama 1.1B** | Trains in minutes, lets you validate your dataset is working before committing hours |
| **Use QLoRA for 7B models** | 4-bit quantisation makes Mistral 7B fit in ~6 GB instead of 28 GB |
| **Keep sequence length at 512** | Longer sequences use VRAM fast; 512 tokens covers most Q&A tasks comfortably |
| **3 epochs, then evaluate** | Running more epochs on small datasets causes overfitting — the model memorises instead of generalising |
| **200+ rows for a useful model** | Under 100 rows, results are unpredictable; 500+ rows consistently produces good outputs |

---

## ❓ Frequently Asked Questions

**Q: Do I need to know Python or machine learning?**
No. The app is fully visual. You need a dataset file and a Gmail account. That's it.

**Q: Is my data uploaded to any server?**
No. Your data is processed locally in the app on your own computer. When training runs in Colab, it's entirely within your own Google account — the data never touches any third-party server.

**Q: The Colab session disconnected mid-training. Did I lose everything?**
No — if you ran the Google Drive mount cell at the top of the notebook (it's the very first cell), your checkpoints were saved automatically every epoch. Re-open the notebook and re-run — it detects the latest checkpoint and resumes from there.

**Q: How is this different from just using ChatGPT?**
ChatGPT is a general-purpose model you control with prompts. Fine-tuning creates a model that *natively* understands your domain, your terminology, your style — without any prompt engineering. It's the difference between hiring a generalist and training a specialist.

**Q: Can I use the fine-tuned model commercially?**
The code in this repo is MIT licensed — free for any use. The model you train depends on the base model's license. Check the HuggingFace model card for each model (e.g., Mistral 7B is Apache 2.0, fully commercial-friendly; Llama 3 requires agreeing to Meta's license).

**Q: My model isn't performing well. Why?**
The most common causes: dataset too small (under 200 rows), duplicate rows in the data, inconsistent formatting between instruction and response columns, or training for too many epochs. Read [`THOUGHTS.md`](THOUGHTS.md) for a detailed breakdown of dataset quality issues and how to fix them.

**Q: Can I run training locally instead of on Colab?**
Yes. The generated notebook code is standard Python. You can also use the CLI trainer directly:
```bash
python training/trainer.py --config your_config.json
```
You'll need a GPU with at least 6 GB VRAM for QLoRA on a 7B model, or 2 GB for TinyLlama.

---

## 🤝 Contributing

PRs are welcome. High-impact areas:

- **Dataset augmentation** — generate synthetic training examples using an LLM before fine-tuning
- **"Resume from checkpoint" auto-detection** — detect the latest Drive checkpoint and restart automatically
- **Weights & Biases logging** — stream training metrics to a W&B dashboard
- **Multi-modal support** — extend to image+text datasets via LLaVA-style training
- **Real-time metrics** — stream loss/accuracy from Colab back to the dashboard via webhook

---

## 📄 License

MIT — free to use, modify, and deploy for any purpose.

---

*Built by [Manas Venkata Raghupatruni](https://github.com/Manas470) · May 2026*
*Read the full design story — every alternative considered, every tradeoff made — in [THOUGHTS.md](THOUGHTS.md)*
