# 🤖 LLM Trainer SaaS

**Train any LLM on your own data — completely free, no credit card required.**

Built with PyTorch Lightning · Google Colab · Streamlit · HuggingFace

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ What It Does

Anyone can visit the app, upload their dataset, configure training, and get a **ready-to-run Google Colab notebook** that fine-tunes an LLM on their data — entirely for free.

| Feature | Details |
|---|---|
| **Training modes** | LoRA/QLoRA · Full fine-tuning · Prompt-based |
| **Supported models** | GPT-2, TinyLlama, Phi-2, Mistral 7B, Llama 3.2, Gemma, Falcon, OPT, … |
| **Dataset formats** | CSV · JSONL · JSON · TXT · Parquet |
| **Cloud platform** | Google Colab (free T4 GPU) |
| **Framework** | PyTorch Lightning 2.x |
| **Model hosting** | HuggingFace Hub (free) |
| **Cost** | $0 — no credit card needed |

---

## 🗺️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit SaaS App                       │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────┐  │
│  │ Upload   │→ │  Configure   │→ │  Jobs    │  │  Test  │  │
│  │ Data     │  │  Training    │  │ Dashboard│  │  Model │  │
│  └──────────┘  └──────┬───────┘  └──────────┘  └────────┘  │
│                       │                                     │
│                       ▼                                     │
│           ┌───────────────────────┐                         │
│           │  Notebook Generator   │  (nbformat)             │
│           │  LoRA / Full / Prompt │                         │
│           └──────────┬────────────┘                         │
└──────────────────────┼──────────────────────────────────────┘
                       │
           ┌───────────▼────────────┐
           │   GitHub Gist (free)   │  ← optional, one-click
           └───────────┬────────────┘
                       │
           ┌───────────▼────────────┐
           │   Google Colab T4 GPU  │  ← FREE, no card
           │   PyTorch Lightning    │
           │   + PEFT (LoRA/QLoRA) │
           └───────────┬────────────┘
                       │
           ┌───────────▼────────────┐
           │   HuggingFace Hub      │  ← optional push
           └────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/yourname/llm-trainer-saas
cd llm-trainer-saas
pip install -r requirements.txt
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your tokens (optional but recommended)
```

| Variable | Where to get it | Required? |
|---|---|---|
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | For gated models & Hub push |
| `GITHUB_TOKEN` | [github.com/settings/tokens](https://github.com/settings/tokens) (scope: `gist`) | For one-click Colab sharing |

### 3. Run locally
```bash
streamlit run app/main.py
```

### 4. Deploy to Streamlit Cloud (free)
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, set **Main file path** to `app/main.py`.
4. Add your secrets under **Advanced settings → Secrets**.
5. Click **Deploy** — done!

---

## 📖 User Flow

```
1. Upload Data     → CSV / JSONL / TXT / Parquet
2. Configure       → pick model, mode, hyperparameters
3. Generate        → click "Generate Notebook"
4. Open Colab      → T4 GPU runtime (free) → Run All
5. Download model  → or push to HuggingFace Hub
6. Test model      → chat with it in the "Test Model" page
```

---

## 🧠 Training Modes

### LoRA / QLoRA (Recommended)
- Adds small adapter matrices — only ~1% of params are trained
- QLoRA loads base model in **4-bit** → runs on 15 GB T4 (free Colab)
- Works with all models up to 7B on free tier
- Best quality/speed tradeoff

### Full Fine-tuning
- All model weights updated
- Needs large GPU (A100 for 7B+)
- Recommended only for small models (GPT-2, DistilGPT-2) on free Colab

### Prompt-based
- No training — your data is used as few-shot examples
- Runs on CPU — instant results
- Great for quick experimentation

---

## 📁 Project Structure

```
llm-trainer-saas/
├── app/
│   ├── main.py                    # Home page
│   ├── pages/
│   │   ├── 1_Upload_Data.py       # Dataset upload & preview
│   │   ├── 2_Configure_Training.py # Hyperparameter config + notebook gen
│   │   ├── 3_Training_Jobs.py     # Job tracking dashboard
│   │   ├── 4_Test_Model.py        # Chat with trained model
│   │   └── 5_Dashboard.py         # Analytics & charts
│   ├── components/
│   │   ├── notebook_generator.py  # Builds Colab .ipynb dynamically
│   │   └── data_processor.py      # Dataset utilities
│   └── utils/
│       ├── config.py              # Constants, BASE_MODELS, TrainingConfig
│       └── state.py               # Session state & job persistence
├── training/
│   ├── data_module.py             # PyTorch Lightning DataModule
│   ├── lightning_module.py        # LightningModule (LoRA + Full)
│   └── trainer.py                 # CLI entry-point
├── jobs/                          # Persisted job JSON files
├── uploads/                       # Uploaded datasets
├── requirements.txt
├── .env.example
└── .streamlit/config.toml
```

---

## 💡 Tips for Free Colab

| Tip | Details |
|---|---|
| **Always use T4 GPU** | Runtime → Change runtime type → T4 GPU |
| **Use QLoRA for 7B models** | Enables 4-bit quantisation — fits in 15 GB VRAM |
| **Use gradient accumulation** | Simulate larger batch sizes without OOM |
| **Save to Drive early** | Colab sessions disconnect after ~12 hours |
| **Use smaller seq length** | 512 tokens is fine for most tasks; 1024 for complex ones |

---

## 🤝 Contributing

PRs welcome! Key areas for improvement:
- Add Weights & Biases logging integration
- Add dataset augmentation utilities
- Add model merging (LoRA → base merge)
- Add multi-GPU support via Lightning's DDP
- Add Hugging Face Inference API for hosted inference

---

## 📄 License

MIT License — free to use, modify, and deploy.
