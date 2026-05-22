# Architecture Decisions — LLM Trainer SaaS

> **Why we built it this way.**
> Every technology choice in this project was made deliberately.
> This document explains what alternatives were considered, why they were rejected,
> and why the chosen approach is the right one for a free, production-quality LLM training platform.

---

## 1. Training Framework — Why PyTorch Lightning?

### What we chose
**PyTorch Lightning 2.x** as the training framework wrapping raw PyTorch.

### Alternatives considered

| Alternative | Why we rejected it |
|---|---|
| **Raw PyTorch** | Requires writing the entire training loop manually — gradient accumulation, mixed precision, checkpointing, logging, distributed training. That's 300+ lines of boilerplate that Lightning handles automatically and correctly. |
| **HuggingFace Trainer** | Excellent for standard fine-tuning, but tightly coupled to HuggingFace's opinionated workflow. Hard to customize, harder to extend with custom data pipelines or non-HF models. Less transparent about what's happening under the hood. |
| **Keras / TensorFlow** | The ML community has largely converged on PyTorch for research and fine-tuning. Most LoRA/PEFT tooling is PyTorch-native. Using TensorFlow would mean reimplementing or porting these tools. |
| **Axolotl** | Great framework but designed for experts. Config-file-heavy, minimal Python API, not suitable for a SaaS platform where we generate code dynamically. |
| **LLaMA-Factory** | Similar to Axolotl — feature-rich but a black box from the code generation perspective. Our notebook generator needs to produce readable, educational code. |
| **FastAI** | High-level wrappers are great for vision but the NLP/LLM support is thin and not actively developed. |

### Why Lightning wins
1. **Structured code without magic** — Lightning separates concerns cleanly: DataModule (data), LightningModule (model + loss + optimizer), Trainer (training loop). The generated Colab notebooks are readable and educational.
2. **Built-in production features** — mixed precision (`fp16`, `bf16`), gradient accumulation, gradient clipping, early stopping, checkpointing, logging — all configured with single parameters.
3. **Framework agnostic** — Lightning works with any PyTorch model, any optimizer, any scheduler. We can use HuggingFace models, PEFT adapters, or custom architectures without modification.
4. **Distributed training ready** — if a user wants to scale to multiple GPUs or nodes, they change one parameter (`strategy="ddp"`). No code changes needed.
5. **Active development** — Lightning AI (the company) is actively investing in the framework and the broader ecosystem.

---

## 2. Cloud Compute — Why Google Colab?

### What we chose
**Google Colab (free tier)** as the training compute platform, with notebooks generated dynamically by the SaaS app.

### Alternatives considered

| Alternative | Why we rejected it |
|---|---|
| **Kaggle Kernels** | Free GPU (T4/P100), but limited to 30 hours/week, sessions must be re-run weekly, no persistent storage, kernel output is public by default. Notebook sharing is harder (no clean "open in Kaggle" URL). |
| **Lightning AI (free tier)** | Native Lightning support is appealing, but the free tier is limited (4 GPU hours/month) and requires account creation. The "Studio" environment is proprietary and not as widely known as Colab. |
| **HuggingFace Spaces** | Excellent for hosting demos, but not designed for long training runs. Free CPU instances only; GPU Spaces require payment. Sessions are limited and Spaces are meant for serving, not training. |
| **AWS / GCP / Azure free tier** | Technically free but requires credit card for signup, has strict usage limits, complex setup (IAM roles, VPCs, etc.), and the free GPU instances (if any) are tiny. Not accessible to non-technical users. |
| **Vast.ai / Runpod** | Cheap GPU rental but requires payment. Violates the "no credit card" constraint. |
| **Local GPU** | Many users don't have a local GPU. The entire point of this platform is to work without local hardware. |
| **Paperspace Gradient** | Free tier exists but is CPU-only. GPU requires credit card. Less widely known. |

### Why Colab wins
1. **Truly free T4 GPU** — 15 GB VRAM, no credit card, available to anyone with a Google account (which most people already have).
2. **Zero setup** — no account creation beyond Google, no software installation, no configuration. Just click and run.
3. **Universal familiarity** — Colab is the most widely used free ML compute platform. Students, researchers, and professionals all know it.
4. **GitHub Gist integration** — Colab can open notebooks directly from GitHub Gists via `colab.research.google.com/gist/username/gist_id` — enabling one-click opening from our SaaS UI.
5. **Google Drive integration** — users can mount Drive for persistent storage across sessions with one cell.
6. **Sufficient for QLoRA** — the free T4 GPU (15 GB VRAM) is enough to run QLoRA fine-tuning on models up to 7B parameters. This covers the most popular open-source models.

### The notebook generation strategy
Rather than running training inside the Streamlit app (which would require our own GPU infrastructure and violate the "free" constraint), we generate a `.ipynb` notebook with all user settings baked in. The user runs it in Colab. This is the critical design insight:

- **We provide the intelligence** (smart defaults, LoRA config, data processing, Lightning training loop)
- **Colab provides the compute** (free GPU)
- **HuggingFace provides the models and hosting** (free)

The platform's value is in the configuration, code generation, and UX — not in owning compute.

---

## 3. Fine-tuning Approach — Why LoRA / QLoRA as Default?

### What we chose
**LoRA (Low-Rank Adaptation) with QLoRA (4-bit quantization)** as the recommended and default training mode.

### Alternatives considered

| Alternative | Why we rejected it as default |
|---|---|
| **Full fine-tuning** | Requires 2× the model size in GPU memory for optimizer states. A 7B model needs ~112 GB VRAM for full fine-tuning (with Adam optimizer states). Impossible on free Colab. Still supported as an option for small models. |
| **Prefix tuning** | Adds learned "soft prompts" to the input. Less widely adopted, harder to merge with the base model, slightly worse results than LoRA at the same parameter budget. |
| **Adapter layers (Houlsby et al.)** | The original adapter approach adds bottleneck layers inside transformer blocks. LoRA achieves similar or better results with a cleaner implementation. |
| **Prompt tuning / P-tuning** | Only tunes the embedding layer. Fastest and most memory-efficient, but quality is much lower than LoRA for most tasks. |
| **RLHF** | Requires a reward model, a reference model, and the main model simultaneously. Needs 3× the VRAM. Way beyond free Colab limits. |
| **DPO (Direct Preference Optimization)** | Better than RLHF for alignment but still needs paired preference data. Added as a future roadmap item. |

### Why LoRA / QLoRA wins
1. **Fits on free hardware** — QLoRA (4-bit base model + LoRA adapters) reduces VRAM usage from ~28 GB (7B float16) to ~8–10 GB. This fits comfortably on a T4 GPU.
2. **Quality very close to full fine-tuning** — multiple papers show that QLoRA fine-tuning achieves 97–99% of full fine-tuning quality on most downstream tasks.
3. **Fast training** — with LoRA, you're only computing gradients for ~0.1–1% of parameters. Training a 7B model for 1 epoch on 1,000 examples takes ~15–30 minutes on a T4 GPU.
4. **Industry standard** — PEFT + LoRA is how virtually all production fine-tuning is done in 2024–2025 for models above 1B parameters. It's the right thing to teach users.
5. **Easy to merge** — after training, LoRA adapters can be merged back into the base model with `merge_and_unload()` for a single deployable file.
6. **Supported by PEFT** — HuggingFace's PEFT library handles all the complexity. We get target module detection, adapter saving, and Hub integration for free.

---

## 4. Frontend — Why Streamlit?

### What we chose
**Streamlit** for the SaaS web interface, hosted on **Streamlit Community Cloud** (free).

### Alternatives considered

| Alternative | Why we rejected it |
|---|---|
| **Gradio** | Also Python-native and great for ML demos. However, multi-page apps are harder, the UI customization is more limited, and job management / state persistence is not as mature. HuggingFace Spaces hosting is excellent but less flexible. |
| **Next.js + FastAPI** | The right choice for a production startup, but way overkill for this project. Requires frontend engineering (React/TypeScript), a separate backend service, a database, and paid hosting. Total cost: $50–$500/month. Deployment complexity: high. |
| **Flask / Django templates** | Old-school web frameworks. Poor support for reactive UI updates, no built-in data visualization, requires frontend HTML/CSS/JS knowledge. |
| **Dash (Plotly)** | Excellent for data dashboards, but the mental model is complex (callbacks everywhere) and it's not designed for ML workflows. |
| **Panel (HoloViz)** | Powerful but niche. Smaller community, less ML-specific tooling, fewer tutorials. |
| **Anvil** | Drag-and-drop app builder with Python backend. Very limited customization, closed ecosystem, paid tiers for most real features. |

### Why Streamlit wins
1. **Zero frontend knowledge required** — the entire UI is Python. Any ML engineer or data scientist can read and modify the code.
2. **Free hosting** — Streamlit Community Cloud hosts apps directly from a GitHub repo for free. No server management.
3. **Fast iteration** — changes to `.py` files are picked up in seconds (`--server.runOnSave=true`). Building a new page takes minutes.
4. **Native data tools** — `st.dataframe()`, `st.plotly_chart()`, `st.download_button()`, `st.file_uploader()` are exactly what we need. No JavaScript integration required.
5. **Multi-page support** — Streamlit's page system (files in `pages/`) gives us clean navigation with zero routing code.
6. **Session state** — `st.session_state` provides the in-memory persistence we need for a single-user workflow (config, dataset info, job list).
7. **Large ML community** — Streamlit is the de-facto standard for ML demos and internal tools. Users are likely already familiar with it.

---

## 5. Model Sharing — Why GitHub Gist for Notebooks?

### What we chose
**GitHub Gist** for hosting generated notebooks, enabling one-click "Open in Colab" URLs.

### Alternatives considered

| Alternative | Why we rejected it |
|---|---|
| **Google Drive upload** | Would require OAuth2 authorization from the SaaS app — complex to implement and requires trusting the app with Drive access. Users are rightly cautious about this. |
| **nbviewer** | Renders notebooks but can't open them in Colab directly. |
| **Direct file download** | Simplest option (we keep this as the fallback). Users download the `.ipynb` and upload to Colab manually — an extra 2-step process. |
| **Our own notebook storage (S3/GCS)** | Requires hosting infrastructure and potentially payment. Violates the "free" constraint. |
| **Firebase Storage** | Has a generous free tier but adds complexity (Firebase SDK, authentication) and a dependency on another third-party service. |

### Why Gist wins (when available)
1. **Zero-click Colab opening** — `https://colab.research.google.com/gist/{user}/{gist_id}` opens the notebook directly. No upload required.
2. **Free and widely available** — GitHub accounts are universal in the developer community. Gists are free and private.
3. **Minimal permissions** — a GitHub token with only `gist` scope is needed. We never ask for repo access, organization access, or any other permission.
4. **Version history** — Gists are Git repos, so previous versions of the notebook are preserved automatically.
5. **Graceful fallback** — if no GitHub token is provided, we fall back to direct download + manual upload. The platform works with or without Gist integration.

---

## 6. Model Registry — Why HuggingFace Hub?

### What we chose
**HuggingFace Hub** for hosting trained models after fine-tuning.

### Alternatives considered

| Alternative | Why we rejected it |
|---|---|
| **Local-only storage** | Fine for experimentation, but doesn't enable model sharing, doesn't survive Colab session ends (without Drive), and can't be used via the Inference API. |
| **Google Drive** | Good for backups but not designed for model serving. No versioning, no API access, difficult to load directly in `transformers`. |
| **AWS S3 / GCS** | Requires payment for storage and egress. Access management is complex. |
| **Weights & Biases (W&B) Artifacts** | W&B is excellent for experiment tracking (and we'd love to add it), but artifacts storage is not the primary use case and has tight free-tier limits for model weights. |
| **MLflow** | Great for internal MLOps but requires your own server (or paid cloud). Not universally accessible. |

### Why HuggingFace Hub wins
1. **Free for public models** — HuggingFace hosts public models for free with no storage limits and generous bandwidth.
2. **First-class `transformers` integration** — `model.push_to_hub("yourname/model")` and `from_pretrained("yourname/model")` work seamlessly. No extra code.
3. **Free Inference API** — hosted models can be queried immediately via the HuggingFace Inference API at no cost (within rate limits).
4. **Community and discoverability** — models on the Hub can be shared with anyone, discovered via search, and benchmarked against others.
5. **PEFT / LoRA support** — `push_to_hub` works for both base models and LoRA adapter-only saves, which are tiny (often just a few MB vs. the base model's GB).

---

## 7. Data Handling — Design Decisions

### Decision: Support 4 formats natively
We support CSV, JSONL, JSON, TXT, and Parquet because:
- CSV is the most accessible format for non-technical users (Excel can produce it)
- JSONL is the standard ML training format (HuggingFace datasets, OpenAI fine-tuning format)
- TXT is the simplest format for text generation / style transfer
- Parquet is used by HuggingFace datasets and is efficient for large datasets

### Decision: Auto-detect instruction vs. text datasets
Rather than requiring users to specify the dataset type, we check for `prompt`+`response` columns (instruction tuning) vs. `text` column (text generation). This makes the platform more accessible to non-ML users who may not know the terminology.

### Decision: No server-side data storage
User datasets are stored temporarily in the session only (`/tmp/uploads/`). They are not sent to any external server, not stored in a database, and not retained after the session ends. This is a deliberate privacy decision — user data stays local.

---

## 8. What We Would Add Next

These features were considered but not included in the initial version to keep the scope manageable:

| Feature | Why not yet | Priority |
|---|---|---|
| **Weights & Biases integration** | Adds a dependency and requires W&B account | High |
| **DPO fine-tuning** | Needs paired preference data format | Medium |
| **LoRA → base model merging** | Useful but adds complexity to the notebook | Medium |
| **Multi-GPU training** | Lightning supports it; needs Colab Pro or cloud GPU | Medium |
| **Dataset augmentation** | LLM-based data augmentation before training | High |
| **Automatic hyperparameter tuning** | Optuna integration with Lightning | Low |
| **Evaluation benchmarks** | MMLU, HellaSwag, etc. post-training | High |
| **User authentication** | Not needed for single-user deployment; needed for SaaS | High |
| **Job queue (Celery/Redis)** | For multi-user deployment with shared Colab sessions | Medium |
| **Real-time training metrics** | WebSocket connection back from Colab | Complex |

---

## Summary

The fundamental design philosophy of this project is:

> **Own the intelligence. Rent the compute.**

We provide:
- Smart configuration UI (Streamlit)
- Production-quality training code (PyTorch Lightning + PEFT)
- Dynamic notebook generation (nbformat + GitHub Gist)
- Model management (HuggingFace Hub)

We rely on free providers for:
- GPU compute (Google Colab)
- Model weights (HuggingFace Hub)
- App hosting (Streamlit Community Cloud)
- Notebook sharing (GitHub Gist)

This stack gives users **production-quality LLM fine-tuning at literally $0**, which was the core goal from day one.
