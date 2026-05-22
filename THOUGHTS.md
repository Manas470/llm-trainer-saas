# My Thoughts on This Project — The Real Story Behind Every Decision

> *Written by the builder. This isn't documentation — it's the raw thinking, the rejected ideas,
> the "what ifs", and the honest assessment of where this could go.*

---

## Why I Built This in the First Place

The number one barrier to LLM fine-tuning isn't technical knowledge. It's money.

Every tutorial I read assumed you had an A100 somewhere, a GCP account with billing enabled,
or a Weights & Biases Pro subscription. The ML community talks about democratising AI constantly,
but the infrastructure around fine-tuning remained stubbornly expensive and inaccessible.

I wanted to build something where a student in India, a startup founder with no runway, or a
researcher at a small university could fine-tune a 7B model on their own dataset — for free,
today, with nothing but a Google account.

That single constraint — **zero dollars, zero credit card** — forced every other decision to be
intentional. And intentional constraints produce better design than open-ended freedom.

---

## The Alternatives I Actually Considered (And Why I Rejected Each One)

### Training Framework

**What I wanted:** Something that produces readable, educational notebook code. Not a black box.

The moment I looked at HuggingFace's `Trainer`, I knew it wasn't right for this use case. It's
excellent for standard fine-tuning, but the generated code looks like configuration soup —
80% of the parameters are passed through a `TrainingArguments` dataclass that hides what's
actually happening. A student running the notebook for the first time wouldn't understand *why*
gradient accumulation is set the way it is, or *what* `fp16=True` actually does.

Raw PyTorch was the opposite problem: too much to write, too many subtle bugs. The standard
training loop has eight failure modes that experts know to avoid (gradient clipping before the
step, not after; zeroing gradients at the right point; correctly handling the last partial batch)
that beginners will hit silently.

PyTorch Lightning sits in the exact middle. The `LightningModule` structure is pedagogically
clear — `training_step`, `configure_optimizers`, `validation_step` map 1:1 to the concepts
in any ML textbook. And it handles the hard parts (mixed precision, gradient accumulation,
early stopping, checkpointing) with a single line each.

I briefly considered **Axolotl** and **LLaMA-Factory**. Both are powerful. Both are config-file
driven. Neither produces notebook code I'd want a learner to read. They're tools for experts
who want to run a job, not for someone who wants to understand what they're running.

**FastAI** almost made the cut for its pedagogical philosophy, but its LLM support is thin and
not actively developed for modern transformer fine-tuning.

---

### Cloud Compute

**What I wanted:** GPU access with zero signup friction beyond what a typical person already has.

I went through every option seriously:

**Kaggle Kernels** — I used Kaggle for months before starting this project. The 30h/week GPU
limit sounds generous until you realise it resets weekly and sessions can't be resumed. More
importantly, sharing a Kaggle kernel requires the recipient to have a Kaggle account. The
friction of "sign up for Kaggle, verify your phone, accept the terms" is a real barrier. And
the "Open in Kaggle" URL pattern doesn't exist in the way Colab's does.

**Lightning AI free tier** — Ironic that I didn't use the platform made by the framework's
creators. The reason: 4 GPU hours per month. That's enough for maybe 2 fine-tuning runs.
The proprietary Studio environment also adds a mental model that I didn't want to introduce.

**HuggingFace Spaces** — This was the most tempting alternative. The integration with the HF
ecosystem is seamless. But Spaces is designed for inference, not training. The free tier is
CPU-only, and the GPU Spaces have session limits that make long training runs unreliable.

**Vast.ai / Runpod** — Extremely cheap GPU rental, sometimes $0.20/hr for a T4. But "cheap"
isn't "free". The credit card barrier is real. And neither platform has a notebook interface
that matches Colab's UX.

**Google Colab** won on every dimension that actually mattered: universally available (everyone
has Gmail), free T4 GPU, excellent notebook UX, and the killer feature — the
`colab.research.google.com/gist/{user}/{gist_id}` URL pattern that lets us generate a
one-click "Open in Colab" link. No other platform has this.

---

### The LoRA Decision

This was the easiest call, but I want to explain the numbers clearly because they're striking.

A 7B parameter model in float16 takes **14 GB just to load weights**. Adam optimiser states
(m and v for each parameter) double that to **28 GB** minimum for training. A free Colab T4
has 15 GB of VRAM. The math doesn't work — at all — for full fine-tuning of any serious model.

QLoRA changes the equation completely:
- 4-bit quantised base model: ~4 GB for 7B parameters
- LoRA adapter matrices (rank 16, ~0.1% of params): ~50 MB
- Gradient states for adapters only: ~100 MB
- Total: **under 6 GB** — fits on a T4 with room to spare

And the quality? Multiple papers (QLoRA paper, Hu et al., various benchmarks) show 97–99% of
full fine-tuning quality on downstream tasks. The 1–3% quality gap is not worth 10× the VRAM.

I also briefly considered **prefix tuning** and **prompt tuning** (P-tuning). Both are more
parameter-efficient than LoRA but produce lower quality results and are harder to merge back
into the base model. LoRA's `merge_and_unload()` is one of the most useful functions in the
entire PEFT library — it gives you a single clean deployable file.

---

### Frontend

I'll be honest: I considered building this as a Next.js + FastAPI application for about 20
minutes before deciding it was exactly the wrong call.

The users of a fine-tuning platform are ML engineers and data scientists. They know Streamlit.
They've built things in Streamlit. The moment they see `st.dataframe()` or `st.plotly_chart()`,
they understand what's happening. There's no cognitive overhead from learning a new framework.

The free hosting argument is also decisive. Streamlit Community Cloud deploys directly from a
GitHub repo with zero server management. The alternative — a Next.js frontend + FastAPI backend
— would require at minimum: a Vercel account (free tier has function limits), a Railway or
Render account for the backend (free tier sleeps after 15 minutes), and a database of some kind
for session state. Complexity multiplies, and complexity breaks.

---

## What I Think About the Edge Cases and Failure Modes

### The Token Limit Problem

Colab's free T4 has 15 GB VRAM, which is enough for a 7B model with QLoRA — but only if your
sequence length is reasonable. A dataset with very long sequences (legal documents, entire books,
long code files) can exceed VRAM even with 4-bit quantization. The current app doesn't warn
the user about this.

**The fix I'd add:** auto-compute the maximum safe sequence length given the selected model
size and available VRAM, and truncate/warn accordingly before generating the notebook.

### The Dataset Quality Trap

The app accepts any CSV or JSONL. A user who uploads a dataset with 50% duplicate rows, missing
values in the response column, or wildly inconsistent formatting will get a model that works but
performs poorly — and they won't know why.

**The fix:** much more aggressive data validation and cleaning before training. We do basic
cleaning now, but I'd want to add: exact duplicate detection, near-duplicate detection (Jaccard
similarity), length distribution analysis, and a sample of "worst quality" rows shown to the
user before they commit to training.

### The Colab Session Timeout

Colab free tier disconnects after 90 minutes of inactivity, or 12 hours total. A 7B model
training for 3 epochs on a decent dataset can take 4–6 hours. If the session disconnects
mid-run, the training state is lost unless the user has set up Google Drive checkpointing.

The generated notebook includes checkpointing via PyTorch Lightning's `ModelCheckpoint`, and
I added a Drive mount cell — but I didn't make it mandatory. That was a mistake. Drive mount
should be step 1 of every notebook.

**The fix:** make Drive mount the very first cell, always. And add a "resume from checkpoint"
code path that auto-detects the latest checkpoint in Drive if the session restarted.

### The Catastrophic Forgetting Problem

LoRA helps, but it doesn't eliminate catastrophic forgetting. A model fine-tuned heavily on
a narrow domain (say, customer support for a specific software product) can lose general
reasoning ability. If a user trains for too many epochs with a high learning rate, the LoRA
adapters overfit badly and the model degrades on anything outside the training distribution.

The current app has early stopping (patience=3 on validation loss), which helps. But the
validation set is only 10% of the data. With a 40-row dataset, that's 4 rows — statistically
meaningless.

**The fix:** require a minimum of 200 rows for LoRA training, 100 for prompt-based. Add a
"catastrophic forgetting test" post-training cell that evaluates the model on a fixed
general-knowledge Q&A set and flags if performance has dropped more than 20%.

### The "My Model Is Worse Than GPT-4" Expectation Problem

This is the most common user failure mode and it's not technical. Users fine-tune a 1.1B model
on 50 rows and expect it to outperform GPT-4 on their domain. It won't. A fine-tuned TinyLlama
will know your domain's terminology and style, but its reasoning ability is capped by its
parameter count.

**The fix:** clear expectation-setting in the UI. A "model capability estimator" that says
"with a 1.1B model and 50 rows, you can expect the model to correctly handle ~65% of domain
queries — consider using a 7B model or adding more data for better results."

---

## The Future I'm Excited About

### Synthetic Data Generation Before Training

The single biggest leverage point for small datasets is synthetic augmentation. If a user
uploads 50 Q&A pairs, you can use GPT-4 or Claude to generate 500 variations, phrasings,
and edge-case examples before training. The fine-tuned model improves dramatically.

This is already on the roadmap, but I'd build it as a first-class feature, not an afterthought.
The pipeline would be: upload 50 rows → augment to 500 with an LLM → review augmented samples
→ train. The UX cost is one extra step; the quality gain is 3–5×.

### Federated Fine-tuning for Privacy-sensitive Data

The current architecture requires users to upload their dataset to the app. For most use cases,
this is fine. But healthcare, legal, and financial users have data they legally cannot upload to
a third-party service.

The notebook generation architecture actually solves this problem already — the notebook runs
locally in Colab with the user's own Google account, and data never touches our servers. But
we could go further: support local-only mode where the notebook is generated without any data
upload, and the user uploads directly in Colab from their own Drive.

### Multi-modal Fine-tuning

LLaVA, CogVLM, and similar models can be fine-tuned with image-text pairs. The Lightning +
PEFT stack supports this — it's a matter of extending the data module to handle image inputs
and the notebook generator to include the right preprocessing pipeline.

A user wanting to fine-tune a vision model on their product catalog images would use the exact
same interface: upload a CSV with image paths and descriptions, configure training, run in Colab.

### Real-time Training Metrics via WebSocket

The biggest UX gap right now: you submit the job, open Colab, and have no visibility into
training progress from the SaaS dashboard. The loss curve, validation perplexity, and training
speed should stream back to the dashboard in real time.

This is technically possible with a lightweight WebSocket server (FastAPI + WebSockets) and a
Lightning callback that sends metrics at each step. The challenge is that Colab doesn't easily
expose outbound connections. The solution: use a webhook endpoint (ngrok or a free service like
pipedream.com) that the Colab notebook posts metrics to, which the dashboard subscribes to.

### The Broader Impact I Care About

Fine-tuning democratisation matters most in the places where it's hardest:

**Low-resource languages.** Most open-source LLMs are heavily English-biased. A user who speaks
Telugu, Swahili, or Quechua and wants a model that understands their language's nuances has
almost no accessible options today. Our platform — with a dataset of 500 sentences in their
language — could produce a usable model in under an hour.

**Domain-specific knowledge in developing markets.** A rural healthcare worker in sub-Saharan
Africa could fine-tune a model on local disease patterns, treatment protocols, and drug
availability. The base model provides reasoning; the fine-tuning provides local expertise.
No cloud budget required.

**Education.** A teacher who wants an AI tutor that follows their specific curriculum, uses
their students' vocabulary, and focuses on their school's exam format. Fine-tuning makes this
possible. The alternative — prompt engineering GPT-4 forever — is fragile and expensive.

---

## What I'd Do Differently

If I were building this from scratch with 3 months and a small team:

1. **Make Drive mount mandatory.** The session persistence problem is the biggest UX failure
   and it has a simple fix.

2. **Build data quality checks first, training second.** At least 50% of poor fine-tuning
   outcomes are caused by bad data, not wrong hyperparameters.

3. **Add a "training budget" estimator.** Show the user: "this configuration will take ~45
   minutes of Colab GPU time and produce a model of approximately X quality." Set expectations
   before they hit run.

4. **Build the augmentation pipeline before launch.** The single most impactful feature for
   users with small datasets. I'd build this before adding multi-GPU support or DPO.

5. **Add a comparison baseline.** After training, run the fine-tuned model and the base model
   side-by-side on 10 test examples from the user's dataset. Show the diff. Nothing demonstrates
   the value of fine-tuning more clearly than concrete before/after examples.

---

## The One Thing I'm Most Proud Of

The architecture insight: **generate code, don't run it**.

Every other fine-tuning platform I found either runs training on their own expensive servers
(cost: $$$) or requires users to manage their own cloud infrastructure (complexity: high).

The notebook generation approach sidesteps both problems entirely. We generate perfect,
production-quality training code — with the user's exact configuration baked in — and hand it
off to Google's free infrastructure to execute. We pay nothing. The user pays nothing. The code
is readable and educational. The user learns something by looking at it.

This is the pattern I'd apply to any future AI infrastructure product: find what's expensive,
find who provides it for free, and build the intelligence layer in between.

> *"Own the intelligence. Rent the compute."*

That's not just a tagline. It's a business model for the next decade of AI infrastructure.

---

*— Manas Venkata Raghupatruni*
*May 2026*
