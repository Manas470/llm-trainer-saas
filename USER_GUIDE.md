# Complete User Guide — LLM Trainer SaaS

**Train any language model on your own data. Free. No credit card. No local GPU needed.**

---

## Table of Contents

1. [Who This Is For](#who-this-is-for)
2. [What You'll Get](#what-youll-get)
3. [Before You Start — What You Need](#before-you-start)
4. [Your Data — How to Prepare It](#your-data)
5. [Step 1 — Upload Your Data](#step-1-upload)
6. [Step 2 — Configure Training](#step-2-configure)
7. [Step 3 — Generate Your Colab Notebook](#step-3-generate)
8. [Step 4 — Run Training in Google Colab](#step-4-colab)
9. [Step 5 — Download & Use Your Model](#step-5-use)
10. [Real-World Use Cases with Walkthroughs](#use-cases)
11. [Understanding Training Modes](#training-modes)
12. [Understanding Base Models](#base-models)
13. [Understanding Hyperparameters](#hyperparameters)
14. [Troubleshooting](#troubleshooting)
15. [FAQ](#faq)
16. [Glossary](#glossary)

---

## 1. Who This Is For <a name="who-this-is-for"></a>

This platform is designed for **three types of users**:

### 🧑‍💼 Non-technical Business Users
You have data — customer conversations, FAQs, product descriptions, support tickets — and you want an AI that knows your domain. You don't need to understand neural networks. Just upload a spreadsheet and follow the steps.

**What you need:** A Google account and a CSV file.

### 👩‍🔬 Data Scientists & ML Engineers
You want control: choose your base model, set LoRA rank, configure learning rate schedules, push to HuggingFace Hub. Everything is exposed. The generated Colab notebooks are clean, readable PyTorch Lightning code you can extend.

**What you need:** Basic ML knowledge, HuggingFace account, optional GitHub account for Gist sharing.

### 🧑‍🎓 Students & Researchers
You want to learn how LLM fine-tuning works without paying for cloud compute. The generated notebooks are educational — every step is explained inline. The PyTorch Lightning codebase is clean enough to learn from.

**What you need:** Google account, curiosity.

---

## 2. What You'll Get <a name="what-youll-get"></a>

After completing this guide, you will have:

- ✅ A language model fine-tuned on your specific data
- ✅ A downloadable `.zip` of model weights (usable offline or in your own applications)
- ✅ (Optionally) Your model hosted on HuggingFace Hub, accessible via API
- ✅ Training logs and loss curves showing model improvement over time
- ✅ A chat interface to test your model immediately

**Example timeline:**
| Task | Time |
|---|---|
| Prepare dataset | 10–30 min |
| Configure training in the app | 5 min |
| Generate and open notebook | 2 min |
| Install Colab dependencies | 3 min |
| Upload dataset in Colab | 1 min |
| Training (GPT-2, 500 rows, 3 epochs) | ~8 min |
| Training (Mistral 7B QLoRA, 500 rows, 1 epoch) | ~45 min |
| Download model | 2 min |
| **Total (GPT-2)** | **~30 min** |
| **Total (Mistral 7B)** | **~90 min** |

---

## 3. Before You Start — What You Need <a name="before-you-start"></a>

### Required (free, no credit card)
- **Google account** — needed for Google Colab (the free GPU platform)
- **Your dataset** — a CSV, JSONL, TXT, or Parquet file with your training data

### Recommended (free, no credit card)
- **HuggingFace account** — for accessing the full model library and pushing your trained model
  - Sign up at [huggingface.co](https://huggingface.co) (free)
  - Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - Select "Write" permissions

### Optional (for one-click Colab sharing)
- **GitHub account** — for sharing notebooks via Gist
  - Create a token at [github.com/settings/tokens](https://github.com/settings/tokens)
  - Enable only the `gist` scope
  - Without this, you'll download the notebook and upload to Colab manually (2 extra clicks)

### NOT required
- ❌ Local GPU
- ❌ Credit card
- ❌ Paid cloud subscription
- ❌ Docker / Kubernetes knowledge
- ❌ DevOps experience

---

## 4. Your Data — How to Prepare It <a name="your-data"></a>

### The Two Dataset Types

**Type A — Instruction / Chat Dataset (recommended for most users)**

This format teaches the model to respond to instructions. Best for:
- Customer support chatbots
- Q&A systems
- Task-specific assistants (summarization, translation, classification)
- Domain-specific knowledge bases

Format (CSV):
```csv
prompt,response
"What is your return policy?","We offer 30-day hassle-free returns. Items must be unused."
"How do I track my order?","Visit our tracking page and enter your order number."
"My package is late.","I'm sorry to hear that! Let me look into this for you..."
```

Format (JSONL):
```jsonl
{"prompt": "What is your return policy?", "response": "We offer 30-day hassle-free returns."}
{"prompt": "How do I track my order?", "response": "Visit our tracking page."}
```

**Type B — Text Completion Dataset**

This format teaches the model to generate text in a specific style. Best for:
- Creative writing
- Style transfer (make AI write like your brand)
- Code generation
- Document summarization

Format (CSV):
```csv
text
"In our company, all customer communications should begin with a warm greeting..."
"Product descriptions must be concise, benefit-focused, and end with a call to action..."
"When handling complaints, always acknowledge the customer's frustration first..."
```

Format (TXT — one example per line):
```
The defendant in this matter was charged under Section 42(b) of the Commercial Code...
Whereas the parties hereto, being duly authorized representatives, agree to...
In consideration of the mutual covenants contained herein, the parties agree...
```

### Data Quality Guidelines

**How much data do you need?**
| Goal | Minimum | Recommended | Expected quality |
|---|---|---|---|
| Just seeing it work | 10 rows | — | Poor (for testing only) |
| Basic task adaptation | 50 rows | 200 rows | Fair |
| Production chatbot | 200 rows | 1,000+ rows | Good |
| Domain expert model | 1,000 rows | 5,000+ rows | Excellent |

**Data quality rules:**
1. **Consistency** — Keep your format the same throughout. If prompts always end with `?`, keep that pattern.
2. **Diversity** — Cover all the topics/tasks you want the model to handle. Don't include 500 variants of the same question.
3. **Correctness** — Every response should be a gold-standard example. The model will imitate everything it sees.
4. **No duplicates** — Exact duplicates waste training time and can cause overfitting.
5. **Appropriate length** — Responses should be roughly the length you expect the model to generate.
6. **Clean encoding** — Use UTF-8, remove HTML tags, fix encoding issues before uploading.

**What NOT to include:**
- Personally identifiable information (names, emails, phone numbers, addresses)
- Passwords, API keys, secrets
- Copyright-protected text you don't have rights to
- Harmful, illegal, or unethical content
- Low-quality, auto-generated, or spammy content

### Converting Your Data to CSV

**From Excel / Google Sheets:**
- File → Download → CSV (comma-separated values)
- Check that text with commas is properly quoted

**From a database (SQL):**
```sql
COPY (SELECT prompt_column, response_column FROM your_table)
TO '/tmp/training_data.csv' WITH CSV HEADER;
```

**From a JSON API export:**
```python
import json, csv
data = json.load(open('export.json'))
with open('training.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['prompt', 'response'])
    writer.writeheader()
    for item in data:
        writer.writerow({'prompt': item['question'], 'response': item['answer']})
```

**From plain text (split into chunks):**
```python
text = open('document.txt').read()
# Split into paragraphs
chunks = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
import pandas as pd
pd.DataFrame({'text': chunks}).to_csv('training.csv', index=False)
```

---

## 5. Step 1 — Upload Your Data <a name="step-1-upload"></a>

Navigate to **Upload Data** in the sidebar.

### Option A: Upload your own file
1. Click the upload area or drag and drop your file
2. Supported formats: `.csv`, `.jsonl`, `.json`, `.txt`, `.parquet`
3. Maximum file size: 200 MB

### Option B: Try a built-in sample dataset (new users)
If you don't have data yet:
1. Click **"Use a built-in sample dataset"**
2. Choose from:
   - **Customer Support Q&A** — 40 e-commerce support examples
   - **General Knowledge Q&A** — 30 educational Q&A pairs
   - **Creative Writing Samples** — 20 short fiction passages
   - **Python Code Assistant** — 25 programming Q&A pairs
3. Click **"Load →"** to use the dataset

### What happens after upload
The app automatically:
- **Detects the dataset type** — instruction (prompt+response) or text completion
- **Shows a preview** — first 10 rows of your data
- **Computes statistics** — row count, column types, missing values
- **Estimates training time** — rough estimate based on dataset size

### Column mapping
If your columns aren't named `prompt`/`response`/`text`, use the **"Column mapping"** expander to map them:
- For instruction datasets: map your question/input column to "Prompt column" and your answer/output column to "Response column"
- For text datasets: map your content column to "Text column"

---

## 6. Step 2 — Configure Training <a name="step-2-configure"></a>

Navigate to **Configure Training** in the sidebar.

### Section 1: Training Mode

**LoRA / QLoRA (recommended)**
Choose this if:
- You're using any model larger than GPT-2
- You want training to finish within minutes-to-hours on free Colab
- You want the most efficient use of the free T4 GPU

Enable **"Use QLoRA (4-bit quantization)"** for models 2.7B and larger. This cuts VRAM usage by ~75%.

**Full Fine-tuning**
Choose this if:
- You're using GPT-2 or DistilGPT-2 (small enough for full fine-tuning on T4)
- You need maximum possible quality and have access to a large GPU
- You're running training locally on a machine with 40+ GB VRAM

**Prompt-based**
Choose this if:
- You want instant results with no training time
- You want to see how your data works as few-shot examples
- You're prototyping before committing to training

### Section 2: Base Model

| Model | Size | VRAM needed | Free Colab? | Best for |
|---|---|---|---|---|
| DistilGPT-2 | 82M | ~500 MB | ✅ Even CPU | Learning, experiments |
| GPT-2 | 124M | ~600 MB | ✅ Even CPU | Fast fine-tuning, text generation |
| GPT-2 Medium | 345M | ~1.5 GB | ✅ T4 | Better text quality |
| GPT-2 Large | 774M | ~3 GB | ✅ T4 | Good text generation |
| TinyLlama 1.1B | 1.1B | ~2.5 GB | ✅ T4 | Modern architecture, instruction following |
| Llama-3.2 1B | 1B | ~3 GB | ✅ T4 | Meta's latest small model |
| Llama-3.2 3B | 3B | ~7 GB | ✅ T4 QLoRA | Excellent quality, needs HF token |
| OPT 1.3B | 1.3B | ~3 GB | ✅ T4 | Open-source alternative |
| Phi-2 | 2.7B | ~6 GB | ✅ T4 QLoRA | Microsoft's efficient model |
| Gemma 2B | 2B | ~5 GB | ✅ T4 QLoRA | Google's efficient model, needs HF token |
| Falcon 7B | 7B | ~14 GB | ⚡ T4 QLoRA (tight) | TII's open model |
| Mistral 7B | 7B | ~14 GB | ⚡ T4 QLoRA (tight) | Best quality on free tier |

**For first-time users:** Start with **GPT-2** or **TinyLlama**. They train fastest and don't require a HuggingFace token.

**Note:** Llama, Gemma, and some other models require accepting a license on HuggingFace before you can download them. Visit the model's HuggingFace page and click "Agree and access repository".

### Section 3: Data Settings

Map your dataset columns here (pre-filled from the upload step).

**Max sequence length:** The maximum number of tokens per training example. Longer examples are truncated. Start with 512 for most use cases; use 1024 or 2048 for longer documents (more VRAM needed).

**Validation split:** The fraction of data reserved for validation (not used in training). Default 10% (0.1). Validation loss tells you if the model is improving or overfitting.

### Section 4: Training Hyperparameters

| Parameter | Default | Guidance |
|---|---|---|
| **Epochs** | 3 | Number of complete passes through your dataset. More epochs = more fitting, but risk overfitting. Start with 3. |
| **Batch size** | 4 | Samples per gradient update. Reduce to 2 or 1 if you get CUDA out-of-memory errors. |
| **Gradient accumulation** | 4 | Simulates larger batches. With batch=4 and accum=4, effective batch = 16. Keeps VRAM low while training stably. |
| **Learning rate** | 2e-4 | How fast the model adapts. Too high = unstable loss. Too low = slow learning. 2e-4 is a safe default for LoRA. |
| **Warmup steps** | 100 | Gradually increases learning rate at the start. Helps with training stability. |
| **Weight decay** | 0.01 | Regularization — slightly penalizes large weights to prevent overfitting. |
| **Mixed precision (fp16)** | ✅ On | Uses 16-bit floats during training — cuts VRAM usage roughly in half, speeds up training. Keep this on. |

### Section 5: LoRA Settings (if using LoRA mode)

| Parameter | Default | Guidance |
|---|---|---|
| **LoRA rank (r)** | 16 | Controls the size of LoRA adapters. Higher = more parameters, better quality, more VRAM. Range: 4–128. Start with 16. |
| **LoRA alpha** | 32 | Scaling factor for LoRA weights. Usually set to 2× the rank. |
| **LoRA dropout** | 0.05 | Dropout applied to LoRA layers for regularization. Increase to 0.1 if overfitting. |
| **Use QLoRA** | ✅ On (for 2B+) | Loads base model in 4-bit. Reduces VRAM by ~75%. Keeps on for all models 2B and larger. |

### Section 6: HuggingFace Hub

- **HuggingFace repo ID**: Your model destination (e.g., `yourusername/my-customer-support-bot`). Leave blank to skip Hub push.
- **Push to Hub**: Enable to automatically upload after training.

### GitHub Gist Token (optional)

Paste a GitHub token (scope: `gist`) to get a one-click "Open in Colab" link. Without it, you'll download the notebook and upload to Colab manually.

---

## 7. Step 3 — Generate Your Colab Notebook <a name="step-3-generate"></a>

Click **"🚀 Generate Notebook & Create Job"**.

The app:
1. Creates a `.ipynb` notebook with all your settings embedded
2. Uploads it to GitHub Gist (if token provided) → generates a Colab URL
3. Creates a job record so you can track progress

**You'll see two options:**

**Option A (with GitHub token):** Click **"🚀 Open in Google Colab ↗"** — the notebook opens directly in your browser, all settings pre-configured.

**Option B (without GitHub token):** Click **"⬇️ Download Notebook (.ipynb)"**, then follow the manual upload steps.

---

## 8. Step 4 — Run Training in Google Colab <a name="step-4-colab"></a>

### Initial Setup (do this once per session)

**Step 4.1 — Open the notebook in Colab**
- Option A: Click the Colab link from the app
- Option B: Go to [colab.research.google.com](https://colab.research.google.com) → File → Upload notebook → select your `.ipynb`

**Step 4.2 — Switch to GPU runtime (CRITICAL)**
```
Runtime menu → Change runtime type → Hardware accelerator → T4 GPU → Save
```
Without this, training runs on CPU and takes 10–100× longer (or fails due to memory).

**Step 4.3 — Verify GPU is available**
Run the first cell. You should see:
```
✅ GPU: Tesla T4
   VRAM: 15.0 GB
```
If you see an error, repeat Step 4.2.

### Running the Training Cells

**Cell: Install dependencies**
This installs PyTorch Lightning, Transformers, PEFT, and other required packages.
Expected time: 2–4 minutes.
Expected output: Progress bars ending with `✅ All packages installed`

> ⚠️ **Note:** If Colab asks you to restart the runtime after installation, do so, then run from the install cell again.

**Cell: Tokens**
Paste your HuggingFace token (if using gated models like Llama or Gemma, or if pushing to Hub):
```python
HF_TOKEN   = "hf_xxxxxxxxxxxxxxxxxxxx"  # ← your token here
HF_REPO_ID = "yourname/my-model"        # ← your HF username/model-name (or leave empty)
PUSH_TO_HUB = True                       # ← set to True to push after training
```

**Cell: Upload your dataset**
Run this cell and a file picker will appear. Select your CSV/JSONL/TXT file from your computer.
After upload, you'll see a preview of your data.

**Cell: Review CONFIG**
Your settings from the app are displayed here. You can edit any value before training:
```python
CONFIG = {
    "base_model":     "gpt2",
    "training_mode":  "lora",
    "num_epochs":     3,
    "batch_size":     4,
    ...
}
```

**Cell: Run training**
This is the main training cell. When you run it:
1. The tokenizer downloads (~1-5 min depending on model size)
2. The base model downloads (~1-30 min depending on model size and internet speed)
3. Training begins — you'll see a live progress bar:
   ```
   Epoch 1/3: 100%|████████| 125/125 [08:23<00:00, train/loss=2.145, val/loss=2.089]
   Epoch 2/3: 100%|████████| 125/125 [08:11<00:00, train/loss=1.876, val/loss=1.901]
   Epoch 3/3: 100%|████████| 125/125 [08:15<00:00, train/loss=1.654, val/loss=1.782]
   ✅ Training complete!
   ```

**What to watch:**
- `train/loss` should decrease over time
- `val/loss` should also decrease (if it stops while train/loss keeps falling, the model is overfitting — stop training)
- If loss is `nan` from the start, your learning rate is too high

**Cell: Save & Push**
Saves model weights to `/content/outputs/`. If `PUSH_TO_HUB=True` and you set `HF_REPO_ID`, uploads to HuggingFace Hub.

**Cell: Download**
Creates a `.zip` of your model and starts a download in your browser.

### Troubleshooting Colab Issues

**CUDA out of memory:**
```python
# In the CONFIG cell, change:
CONFIG["batch_size"] = 1       # was 4
CONFIG["max_seq_length"] = 256  # was 512
```

**Model download is slow:**
This is normal for large models (Mistral 7B is ~14 GB). Leave the tab open and wait. Use a smaller model (GPT-2, TinyLlama) for faster iteration.

**Session disconnected mid-training:**
The ModelCheckpoint callback saves the best checkpoint automatically. Look in `/content/outputs/` — your best model checkpoint is already there. Run the Save cell to download it.

**"RuntimeError: Expected all tensors to be on the same device":**
Add `device_map="auto"` to model loading (already in the generated notebooks) or restart runtime and try again.

---

## 9. Step 5 — Use Your Model <a name="step-5-use"></a>

### Option A — Test in the SaaS app
Go to **Test Model** in the sidebar. Enter your HuggingFace repo ID and start chatting.

### Option B — Use locally with Python
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# If you pushed to HuggingFace Hub:
model_id = "yourname/my-model"
tok   = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# If you downloaded locally:
model_id = "/path/to/outputs"
tok   = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Generate text
gen = pipeline("text-generation", model=model, tokenizer=tok)
result = gen(
    "### Instruction:\nWhat is your return policy?\n\n### Response:\n",
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
)
print(result[0]["generated_text"])
```

### Option C — HuggingFace Inference API (free, no local GPU)
```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="yourname/my-model", token="hf_xxxx")
response = client.text_generation(
    "### Instruction:\nWhat is your return policy?\n\n### Response:\n",
    max_new_tokens=200,
)
print(response)
```

### Option D — Deploy in a web app
```python
# Simple FastAPI endpoint
from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
gen = pipeline("text-generation", model="yourname/my-model")

@app.post("/chat")
def chat(prompt: str):
    result = gen(prompt, max_new_tokens=200)
    return {"response": result[0]["generated_text"]}
```

---

## 10. Real-World Use Cases with Walkthroughs <a name="use-cases"></a>

### Use Case 1: Customer Support Chatbot

**Goal:** Fine-tune a model on your company's historical support tickets to automate responses.

**Dataset prep:**
Export from your CRM/helpdesk (Zendesk, Intercom, Freshdesk) as CSV:
```
prompt,response
"How do I reset my password?","Click 'Forgot Password' on the login page..."
"My invoice is incorrect.","I'm sorry to hear that. Please send your invoice number to billing@company.com..."
```
Aim for 500–2,000 examples covering all common topics.

**Configuration:**
- Model: TinyLlama 1.1B or Phi-2 (good quality, runs on free T4)
- Mode: LoRA (QLoRA for Phi-2)
- Epochs: 3–5
- Batch size: 4, Gradient accum: 4

**Expected result:** A model that responds to customer queries in your company's voice and with your specific policies.

---

### Use Case 2: Domain-Specific Knowledge Base

**Goal:** Fine-tune a model to answer questions about a specific domain (medical, legal, technical).

**Dataset prep:**
Create Q&A pairs from your documentation, manuals, or textbooks:
```
prompt,response
"What is the maximum safe dosage of ibuprofen for adults?","The maximum recommended dose of ibuprofen for adults is 1,200 mg/day for OTC use..."
"What are the contraindications for beta-blockers?","Beta-blockers are contraindicated in: severe bradycardia, cardiogenic shock, decompensated heart failure..."
```

**Configuration:**
- Model: Mistral 7B Instruct with QLoRA (best accuracy for complex knowledge)
- Mode: QLoRA
- Epochs: 2–3
- Max seq length: 1024 (medical/legal answers tend to be longer)

**Important:** Always have a domain expert review model outputs. Fine-tuned models can hallucinate — never use medical/legal AI outputs without human verification.

---

### Use Case 3: Code Generation Assistant

**Goal:** Fine-tune a model to write code in your company's style, using your internal APIs and patterns.

**Dataset prep:**
```
prompt,response
"Write a function to connect to our internal UserService API","from services import UserServiceClient\n\ndef get_user(user_id: str) -> User:\n    client = UserServiceClient(endpoint=Config.USER_SERVICE_URL)\n    return client.get_user(user_id=user_id)\n"
"How do I handle errors in our framework?","from errors import AppError, ErrorCode\n\ntry:\n    result = risky_operation()\nexcept ValidationError as e:\n    raise AppError(ErrorCode.VALIDATION, str(e)) from e\n"
```

**Configuration:**
- Model: GPT-2 Large or TinyLlama (code tasks don't always need the biggest model)
- Mode: LoRA
- Max seq length: 1024–2048 (code can be long)
- Epochs: 3–5

---

### Use Case 4: Creative Writing / Brand Voice

**Goal:** Train a model to write in your brand's specific voice and style.

**Dataset prep (TXT format — simplest):**
```
Our products are designed for people who care about quality. Every detail matters.
Innovation starts with listening. That's why we spend more time understanding you than building features.
We don't just make software. We make your work feel effortless.
```

**Configuration:**
- Model: GPT-2 Medium or Large (good for text style)
- Mode: Full fine-tuning (GPT-2 is small enough)
- Epochs: 5–10 (style learning benefits from more epochs)
- Text column: your text column

---

### Use Case 5: Document Summarization

**Goal:** Fine-tune a model to summarize documents in a specific format.

**Dataset prep:**
```
prompt,response
"Summarize the following meeting notes: [full meeting notes here...]","Key decisions: 1) Launch delayed to Q3. 2) Budget increased by 15%. Action items: Alice to update timeline, Bob to revise budget."
```

**Configuration:**
- Model: Phi-2 2.7B or Mistral 7B (longer context understanding)
- Max seq length: 2048 (summaries need long input context)
- Mode: QLoRA

---

## 11. Understanding Training Modes <a name="training-modes"></a>

### LoRA (Low-Rank Adaptation)

LoRA adds small "adapter" matrices to the attention layers of the transformer. Only these adapters are trained; the original model weights are frozen.

**How it works technically:**
- A weight matrix W (e.g., 4096×4096) is approximated by W + ΔW = W + BA
- Where B is (4096×r) and A is (r×4096), with r = "LoRA rank" (e.g., 16)
- Instead of training 4096×4096 = 16M parameters, you train 2×4096×16 = 131K parameters
- That's 99.2% fewer parameters to train

**When to use LoRA:**
- Models up to 7B parameters on free T4 GPU
- When you want training to complete in under an hour
- For instruction fine-tuning (chat, Q&A, task-following)
- For most production use cases

### QLoRA (Quantized LoRA)

QLoRA combines LoRA with 4-bit quantization of the base model.

**How it works:**
- The base model weights are compressed from 16-bit (float16) to 4-bit (NF4 format)
- This reduces model VRAM from ~14 GB (Mistral 7B float16) to ~5 GB
- LoRA adapters are then added in float16 precision
- Gradients flow through the 4-bit base model to the 16-bit adapters

**Quality impact:** QLoRA models perform at 97–99% of full-precision LoRA quality on most tasks. The slight degradation is usually imperceptible.

**When to use QLoRA:**
- Models 2.7B+ on free T4 GPU (always enable for these)
- When LoRA runs out of memory, try QLoRA first

### Full Fine-tuning

All model weights are updated during training. No adapters — the entire model learns from your data.

**Pros:**
- Maximum possible quality
- Simpler deployment (no separate adapter files)

**Cons:**
- Requires 2× model VRAM (for optimizer states) = ~56 GB for 7B model
- Very slow on free Colab
- Risk of catastrophic forgetting (model loses general capabilities)

**When to use Full Fine-tuning:**
- GPT-2 or DistilGPT-2 (small enough for T4)
- When you have access to an A100 GPU
- When you need absolute maximum quality and have the hardware

### Prompt-based (no training)

Your data is formatted as few-shot examples and appended to the prompt.

**How it works:**
```
[Your examples from dataset...]
Q: How do I reset my password?
A: [model generates here using your examples as context]
```

**When to use:**
- Quick prototyping
- Small datasets (fewer than 50 examples)
- When you can't afford training time
- When you want to test your data quality before committing to training

---

## 12. Understanding Base Models <a name="base-models"></a>

### How to Choose

**For beginners:** Start with **GPT-2** — it trains in minutes, requires no HF token, and gives you immediate results to verify your workflow.

**For production instruction-following:** Use **Mistral 7B Instruct** with QLoRA — it's the best quality model available on free Colab.

**For a balance of quality and speed:** **TinyLlama 1.1B** or **Phi-2 2.7B** — modern architectures with surprisingly good quality for their size.

**For code:** **Phi-2** is specifically trained on code and performs very well.

### Model Families Explained

**GPT-2 family (OpenAI):** Classic autoregressive language models. Not instruction-tuned by default, so they generate text continuations rather than chat responses. Good for: text generation, style learning. Freely available, no token needed.

**TinyLlama (Zhang et al.):** A 1.1B model trained on 3 trillion tokens using the Llama architecture. Punches above its weight class. Good chat capability after instruction fine-tuning. Free, no token needed.

**Phi-2 (Microsoft):** 2.7B model trained on high-quality "textbook" data. Outperforms many 7B models on reasoning benchmarks. Excellent for code. Free, no token needed.

**Llama family (Meta):** Industry-standard open models. Llama-3.2 (1B, 3B) are the latest compact versions. Requires accepting Meta's license on HuggingFace. HF token needed.

**Mistral 7B (Mistral AI):** The gold standard for 7B models. The Instruct version is excellent at following instructions and chat. Requires HF token.

**Gemma (Google):** Google's efficient transformer models. The 2B version is very capable. Requires accepting Google's license + HF token.

**Falcon (TII):** Technology Innovation Institute's open models. Falcon 7B Instruct is competitive with Mistral. HF token needed.

**OPT (Meta):** Earlier Meta models, fully open without license requirements. Good for research.

---

## 13. Understanding Hyperparameters <a name="hyperparameters"></a>

### The Most Important Settings

**Learning Rate (lr)**
- Controls how fast the model adapts to your data
- Too high (>1e-3): loss jumps around, training is unstable, model may get worse
- Too low (<1e-5): training is extremely slow, model barely changes
- **Sweet spot for LoRA: 1e-4 to 3e-4** (default: 2e-4)
- For full fine-tuning, use lower: 5e-5 to 1e-4

**Epochs**
- One epoch = one complete pass through all your training data
- More epochs = model fits your data more closely = risk of overfitting
- **Rule of thumb:** 3 epochs is a good starting point
- If val/loss stops improving before 3 epochs, early stopping kicks in automatically
- If val/loss is still improving at epoch 3, try 5–7 epochs

**Batch Size × Gradient Accumulation = Effective Batch Size**
- Effective batch size = batch_size × grad_accum_steps
- Default: 4 × 4 = 16 effective batch size
- Larger effective batch = more stable gradients = often better quality
- If you get OOM errors: reduce batch_size (not grad_accum)
- The product (effective batch) stays the same

**LoRA Rank (r)**
- The "width" of LoRA adapters
- Higher rank = more parameters = better quality = more VRAM
- r=4: minimal, use when VRAM is very tight
- r=16: good default (99% of use cases)
- r=64: high quality, needs more VRAM
- r=128: used for complex, large datasets

**Max Sequence Length**
- Maximum tokens per training example (examples longer than this are truncated)
- Longer = more context = more VRAM
- 512: good for chat, Q&A, short texts
- 1024: for medium documents, code
- 2048: for long documents, papers
- 4096: large context, needs A100+

### Signs of Good Training
- `train/loss` decreases steadily over time
- `val/loss` decreases alongside `train/loss`
- No NaN values in loss
- Progress bar completes without CUDA errors

### Signs of Problems

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss is NaN from step 1 | LR too high | Reduce to 1e-4 |
| Loss oscillates wildly | LR too high or bad data | Reduce LR, clean data |
| Train loss ↓ but val loss ↑ | Overfitting | Reduce epochs, add more data |
| No loss decrease | LR too low or model too large | Increase LR to 3e-4 |
| OOM error | Batch too large or model too big | Reduce batch_size, enable QLoRA |
| Garbled/repetitive output | Overfitting or too many epochs | Reduce epochs |

---

## 14. Troubleshooting <a name="troubleshooting"></a>

### In the Streamlit App

**"Please upload a dataset first" on Configure page**
→ Go to Upload Data first and upload or load a sample dataset.

**Notebook download is empty / 0 bytes**
→ Try refreshing the page and generating the notebook again.

**GitHub Gist token not working**
→ Make sure the token has `gist` scope (not just `repo`). Create a new token at github.com/settings/tokens → "Generate new token (classic)" → check only "gist".

### In Google Colab

**"No GPU detected" error**
→ Go to Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save.

**CUDA out of memory (OOM)**
Options, in order of impact:
1. Reduce `batch_size` to 1 or 2
2. Reduce `max_seq_length` to 256 or 128
3. Enable `use_qlora = True`
4. Use a smaller base model

**"ModuleNotFoundError: No module named 'peft'"**
→ The install cell didn't complete. Restart runtime and run the install cell again. Wait for all progress bars to finish.

**"Access to model is restricted"**
→ You need to: (1) visit the model's HuggingFace page and accept the license, (2) paste your HF token in the tokens cell.

**Model downloads very slowly**
→ Large models (7B = ~14 GB) take 15–30 min to download in Colab. This is normal. Don't close the tab.

**Training completed but loss was still high**
→ Try: more epochs, more data, higher quality data, a larger model, a higher LoRA rank.

**Session disconnected mid-training**
→ The best checkpoint is saved automatically in `/content/outputs/`. When the session restarts, run just the "Download" cell to retrieve it.

**File upload widget doesn't appear**
→ Make sure you're running the cell (not just viewing it). Try Runtime → Restart runtime → Run all.

---

## 15. FAQ <a name="faq"></a>

**Q: Is this really free? What's the catch?**
A: Yes, completely free. Google gives everyone limited free T4 GPU time on Colab. HuggingFace hosts models for free. Streamlit hosts apps for free. The "catch" is that Colab sessions disconnect after ~12 hours and GPU availability isn't guaranteed (sometimes you have to wait or try at a different time).

**Q: My data is sensitive (customer PII, trade secrets). Is it safe?**
A: Your data is uploaded directly from your computer to Google Colab's runtime — it doesn't pass through our app's servers. However, it does reside on Google's servers during the Colab session. For highly sensitive data, run the training script locally instead of in Colab.

**Q: How good will my model be?**
A: It depends heavily on your data quality and quantity. With 500+ high-quality instruction pairs and a good base model (Mistral 7B), the result can be genuinely impressive for domain-specific tasks. With 50 rows, expect it to show the direction but not production quality.

**Q: Can I fine-tune models larger than 7B on free Colab?**
A: Not with the T4 GPU. You'd need Colab Pro (A100) or a paid cloud GPU. The platform generates code that works on any GPU — just run it with a better GPU elsewhere.

**Q: Can I use this for commercial purposes?**
A: The platform's code is MIT licensed. However, the base models have their own licenses — check each model's license on HuggingFace. GPT-2 is fully open. Llama, Gemma, and Mistral have licenses that permit commercial use with conditions.

**Q: Will my trained model forget everything it knew before?**
A: LoRA fine-tuning specifically aims to preserve the base model's general capabilities while adding domain knowledge. Full fine-tuning can cause "catastrophic forgetting" if trained too many epochs on too little data. With LoRA, this risk is very low.

**Q: How do I know if training worked?**
A: Run the Test Model page and ask it questions related to your training data. Compare responses before and after fine-tuning (you can test the base model too at huggingface.co/{model_name}).

**Q: My model outputs the same thing every time.**
A: Enable sampling in inference: `do_sample=True, temperature=0.7`. Without sampling, the model greedily picks the highest-probability token at each step, leading to repetitive outputs.

**Q: Can I continue training a model I already trained?**
A: Yes. Load the saved LoRA adapters instead of starting from the base model: `PeftModel.from_pretrained(base_model, "path/to/adapters")` and continue training.

---

## 16. Glossary <a name="glossary"></a>

| Term | Definition |
|---|---|
| **LLM** | Large Language Model — AI model trained on vast text data to understand and generate language |
| **Fine-tuning** | Continuing to train a pre-trained model on your specific data |
| **LoRA** | Low-Rank Adaptation — efficient fine-tuning by adding small adapter matrices |
| **QLoRA** | Quantized LoRA — LoRA with 4-bit compressed base model for lower VRAM usage |
| **VRAM** | Video RAM — memory on the GPU. More VRAM = larger models |
| **Tokenizer** | Converts text to numbers (tokens) that the model processes |
| **Token** | The basic unit of text a model processes (~0.75 words on average) |
| **Epoch** | One complete pass through the training dataset |
| **Batch size** | Number of training examples processed simultaneously |
| **Learning rate** | How fast the model updates its weights during training |
| **Gradient accumulation** | Technique to simulate larger batches without using more VRAM |
| **Overfitting** | Model memorizes training data but fails to generalize to new inputs |
| **Early stopping** | Automatically stops training when validation loss stops improving |
| **Checkpoint** | A saved snapshot of model weights at a specific training step |
| **HuggingFace Hub** | Platform for hosting and sharing ML models |
| **PEFT** | Parameter-Efficient Fine-Tuning — umbrella term for methods like LoRA |
| **fp16** | 16-bit floating point — half precision, uses half the VRAM of fp32 |
| **Inference** | Using a trained model to generate outputs (as opposed to training) |
| **Perplexity** | Measure of how well a language model predicts text — lower is better |
| **Adapter** | Small trainable module added to a frozen base model (e.g., LoRA adapter) |
| **Tokenization** | Process of converting text to tokens |
| **Context window** | Maximum number of tokens a model can process at once |
| **Instruction tuning** | Fine-tuning on pairs of instructions and desired responses |
| **Hallucination** | When an AI model confidently generates false information |
