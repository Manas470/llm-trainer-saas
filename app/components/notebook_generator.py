"""
Dynamically generate a ready-to-run Google Colab notebook (.ipynb)
from a TrainingConfig dict, then share it via GitHub Gist so the
user can open it in Colab with a single click.
"""
from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional

import nbformat as nbf

from app.utils.config import GITHUB_TOKEN


# ─────────────────────────────────────────────────────────────────────────────
# Cell builders
# ─────────────────────────────────────────────────────────────────────────────

def _md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def _code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


# ─────────────────────────────────────────────────────────────────────────────
# Install cell (common to all modes)
# ─────────────────────────────────────────────────────────────────────────────

def _install_cell(training_mode: str) -> nbf.NotebookNode:
    pkgs = [
        "pytorch-lightning",
        "transformers",
        "datasets",
        "huggingface-hub",
        "peft",
        "accelerate",
        "trl",
        "sentencepiece",
        "loguru",
        "rich",
    ]
    if training_mode in ("lora", "qlora"):
        pkgs.append("bitsandbytes")

    pip_line = " ".join(pkgs)
    return _code(f"""\
        # ── Install dependencies ──────────────────────────────────
        !pip install -q {pip_line}
        print("✅ All packages installed")
    """)


# ─────────────────────────────────────────────────────────────────────────────
# LoRA / QLoRA notebook
# ─────────────────────────────────────────────────────────────────────────────

def _build_lora_notebook(cfg: dict) -> nbf.NotebookNode:
    nb    = nbf.v4.new_notebook()
    cells = []

    # Header
    _base  = cfg.get('base_model', 'gpt2')
    _mode  = cfg.get('training_mode', 'lora')
    cells.append(_md(f"""\
        # 🤖 LLM Trainer — LoRA / QLoRA Fine-tuning
        **Model:** `{_base}`
        **Mode:** {"QLoRA (4-bit)" if cfg.get('use_qlora') else "LoRA (16-bit)"}
        **Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

        > Run cells top-to-bottom. GPU runtime required (Runtime → Change runtime type → T4 GPU).
    """))

    # Check GPU
    cells.append(_code("""\
        import torch
        assert torch.cuda.is_available(), "❌ No GPU detected! Go to Runtime → Change runtime type → T4 GPU"
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    """))

    # Install
    cells.append(_install_cell(_mode))

    # Secrets / tokens
    cells.append(_md("## 🔑 Tokens\nPaste your HuggingFace token below (needed for gated models & Hub push)."))
    cells.append(_code(f"""\
        HF_TOKEN   = ""   # ← paste your HF token here (https://huggingface.co/settings/tokens)
        HF_REPO_ID = "{cfg.get('hf_repo_id', '')}"   # e.g. "yourname/my-finetuned-model"  (leave empty to skip)
        PUSH_TO_HUB = {str(bool(cfg.get('push_to_hub', False)))}
    """))

    # Mount Drive & upload dataset
    cells.append(_md("## 📂 Upload Your Dataset\nRun the cell below and select your file (CSV / JSONL / TXT)."))
    cells.append(_code("""\
        from google.colab import files
        import io, os, pandas as pd

        print("Select your dataset file ↓")
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
        DATASET_PATH = f"/content/{filename}"
        print(f"✅ Uploaded: {filename}")

        # Quick preview
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(DATASET_PATH)
        elif ext in (".jsonl", ".json"):
            df = pd.read_json(DATASET_PATH, lines=(ext==".jsonl"))
        elif ext == ".txt":
            lines = open(DATASET_PATH).readlines()
            df = pd.DataFrame({"text": [l.strip() for l in lines if l.strip()]})
        else:
            df = pd.read_parquet(DATASET_PATH)

        print(f"Rows: {len(df):,}  |  Columns: {list(df.columns)}")
        df.head(3)
    """))

    # Config
    cells.append(_md("## ⚙️ Training Configuration"))
    cells.append(_code(f"""\
        CONFIG = {{
            "base_model":       "{cfg['base_model']}",
            "training_mode":    "{cfg['training_mode']}",
            "text_column":      "{cfg.get('text_column', 'text')}",
            "prompt_column":    "{cfg.get('prompt_column', 'prompt')}",
            "response_column":  "{cfg.get('response_column', 'response')}",
            "max_seq_length":   {cfg.get('max_seq_length', 512)},
            "val_split":        {cfg.get('val_split', 0.1)},
            "num_epochs":       {cfg.get('num_epochs', 3)},
            "batch_size":       {cfg.get('batch_size', 4)},
            "grad_accum":       {cfg.get('grad_accum', 4)},
            "learning_rate":    {cfg.get('learning_rate', 2e-4)},
            "warmup_steps":     {cfg.get('warmup_steps', 100)},
            "weight_decay":     {cfg.get('weight_decay', 0.01)},
            "fp16":             {cfg.get('fp16', True)},
            "lora_r":           {cfg.get('lora_r', 16)},
            "lora_alpha":       {cfg.get('lora_alpha', 32)},
            "lora_dropout":     {cfg.get('lora_dropout', 0.05)},
            "use_qlora":        {cfg.get('use_qlora', True)},
            "output_dir":       "/content/outputs",
        }}
        print("✅ Config ready")
    """))

    # Training code
    cells.append(_md("## 🚀 Run Training"))
    cells.append(_code("""\
        import torch, os
        from pathlib import Path
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
        from torch.utils.data import DataLoader, Dataset, random_split
        import pandas as pd

        # ── Dataset ──────────────────────────────────────────────
        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_len):
                self.texts, self.tok, self.max_len = texts, tokenizer, max_len
            def __len__(self): return len(self.texts)
            def __getitem__(self, i):
                enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                               padding="max_length", return_tensors="pt")
                ids = enc["input_ids"].squeeze(0)
                mask = enc["attention_mask"].squeeze(0)
                labels = ids.clone(); labels[mask==0] = -100
                return {"input_ids": ids, "attention_mask": mask, "labels": labels}

        class InstructionDataset(Dataset):
            TPL = "### Instruction:\\n{p}\\n\\n### Response:\\n{r}"
            def __init__(self, df, tokenizer, pcol, rcol, max_len):
                self.records = df[[pcol, rcol]].to_dict("records")
                self.tok, self.max_len, self.pcol, self.rcol = tokenizer, max_len, pcol, rcol
            def __len__(self): return len(self.records)
            def __getitem__(self, i):
                row = self.records[i]
                text = self.TPL.format(p=row[self.pcol], r=row[self.rcol])
                enc = self.tok(text, truncation=True, max_length=self.max_len,
                               padding="max_length", return_tensors="pt")
                ids = enc["input_ids"].squeeze(0); mask = enc["attention_mask"].squeeze(0)
                labels = ids.clone(); labels[mask==0] = -100
                return {"input_ids": ids, "attention_mask": mask, "labels": labels}

        # ── Lightning Module ──────────────────────────────────────
        class LLMModule(pl.LightningModule):
            def __init__(self, model, lr, warmup, weight_decay):
                super().__init__(); self.model=model; self.lr=lr
                self.warmup=warmup; self.wd=weight_decay
            def training_step(self, batch, _):
                loss = self.model(**batch).loss
                self.log("train/loss", loss, prog_bar=True)
                return loss
            def validation_step(self, batch, _):
                loss = self.model(**batch).loss
                self.log("val/loss", loss, prog_bar=True)
            def configure_optimizers(self):
                params = [p for p in self.model.parameters() if p.requires_grad]
                opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.wd)
                total = self.trainer.estimated_stepping_batches
                sched = get_cosine_schedule_with_warmup(opt, self.warmup, total)
                return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        # ── Build model ───────────────────────────────────────────
        print("Loading tokenizer …")
        tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True, token=HF_TOKEN or None)
        if tok.pad_token is None: tok.pad_token = tok.eos_token

        print("Loading model …")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True) if CONFIG["use_qlora"] else None
        base = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], quantization_config=bnb,
                                                     device_map="auto", trust_remote_code=True, token=HF_TOKEN or None)
        if CONFIG["use_qlora"]: base = prepare_model_for_kbit_training(base)
        lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
                               lora_dropout=CONFIG["lora_dropout"], target_modules=["q_proj","v_proj"], bias="none")
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()

        # ── Build dataset ─────────────────────────────────────────
        df = pd.read_csv(DATASET_PATH) if DATASET_PATH.endswith(".csv") else \\
             pd.read_json(DATASET_PATH, lines=DATASET_PATH.endswith(".jsonl")) if DATASET_PATH.endswith((".jsonl",".json")) else \\
             pd.DataFrame({"text": open(DATASET_PATH).read().splitlines()})

        has_instr = CONFIG["prompt_column"] in df.columns and CONFIG["response_column"] in df.columns
        ds = InstructionDataset(df, tok, CONFIG["prompt_column"], CONFIG["response_column"], CONFIG["max_seq_length"]) if has_instr \\
             else TextDataset(df[CONFIG["text_column"] if CONFIG["text_column"] in df.columns else df.columns[0]].dropna().tolist(), tok, CONFIG["max_seq_length"])

        n_val = max(1, int(len(ds)*CONFIG["val_split"]))
        train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val], generator=torch.Generator().manual_seed(42))
        train_dl = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
        val_dl   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)

        # ── Train ─────────────────────────────────────────────────
        module = LLMModule(model, CONFIG["learning_rate"], CONFIG["warmup_steps"], CONFIG["weight_decay"])
        Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
        trainer = pl.Trainer(
            max_epochs=CONFIG["num_epochs"],
            accumulate_grad_batches=CONFIG["grad_accum"],
            precision="16-mixed" if CONFIG["fp16"] else "32-true",
            callbacks=[
                ModelCheckpoint(dirpath=CONFIG["output_dir"], monitor="val/loss", mode="min", save_top_k=2),
                EarlyStopping(monitor="val/loss", patience=3, mode="min"),
                LearningRateMonitor(logging_interval="step"),
            ],
            log_every_n_steps=10,
        )
        trainer.fit(module, train_dl, val_dl)
        print("\\n✅ Training complete!")
    """))

    # Save & push
    cells.append(_md("## 💾 Save & Push to HuggingFace Hub"))
    cells.append(_code("""\
        model.save_pretrained(CONFIG["output_dir"])
        tok.save_pretrained(CONFIG["output_dir"])
        print(f"✅ Model saved to {CONFIG['output_dir']}")

        if PUSH_TO_HUB and HF_REPO_ID:
            from huggingface_hub import login
            login(token=HF_TOKEN)
            model.push_to_hub(HF_REPO_ID, private=False)
            tok.push_to_hub(HF_REPO_ID, private=False)
            print(f"🤗 Pushed to https://huggingface.co/{HF_REPO_ID}")
        else:
            print("ℹ️  Skipping Hub push (set HF_REPO_ID and PUSH_TO_HUB=True to enable)")
    """))

    # Download outputs
    cells.append(_md("## 📥 Download Your Model"))
    cells.append(_code("""\
        import shutil
        shutil.make_archive("/content/my_model", "zip", CONFIG["output_dir"])
        from google.colab import files
        files.download("/content/my_model.zip")
        print("✅ Download started!")
    """))

    nb.cells = cells
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Full fine-tune notebook (same structure, no LoRA cells)
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_notebook(cfg: dict) -> nbf.NotebookNode:
    nb    = nbf.v4.new_notebook()
    cells = []

    cells.append(_md(f"""\
        # 🤖 LLM Trainer — Full Fine-tuning
        **Model:** `{cfg['base_model']}`
        **Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

        > ⚠️ Full fine-tuning updates ALL model weights. Requires a high-VRAM GPU (A100 recommended).
        > For free Colab, use smaller models (GPT-2, DistilGPT-2) or switch to LoRA mode.
    """))

    cells.append(_code("""\
        import torch
        assert torch.cuda.is_available(), "❌ No GPU! Go to Runtime → Change runtime type → GPU"
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    """))

    cells.append(_install_cell("full"))

    cells.append(_code(f"""\
        HF_TOKEN   = ""
        HF_REPO_ID = "{cfg.get('hf_repo_id', '')}"
        PUSH_TO_HUB = {str(bool(cfg.get('push_to_hub', False)))}
    """))

    cells.append(_md("## 📂 Upload Dataset"))
    cells.append(_code("""\
        from google.colab import files
        import pandas as pd, os
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
        DATASET_PATH = f"/content/{filename}"
        print(f"✅ {filename}")
    """))

    cells.append(_code(f"""\
        CONFIG = {{
            "base_model":      "{cfg['base_model']}",
            "text_column":     "{cfg.get('text_column','text')}",
            "prompt_column":   "{cfg.get('prompt_column','prompt')}",
            "response_column": "{cfg.get('response_column','response')}",
            "max_seq_length":  {cfg.get('max_seq_length', 512)},
            "val_split":       {cfg.get('val_split', 0.1)},
            "num_epochs":      {cfg.get('num_epochs', 3)},
            "batch_size":      {cfg.get('batch_size', 2)},
            "grad_accum":      {cfg.get('grad_accum', 8)},
            "learning_rate":   {cfg.get('learning_rate', 5e-5)},
            "warmup_steps":    {cfg.get('warmup_steps', 100)},
            "weight_decay":    {cfg.get('weight_decay', 0.01)},
            "fp16":            True,
            "output_dir":      "/content/outputs",
        }}
    """))

    cells.append(_md("## 🚀 Run Full Fine-tuning"))
    cells.append(_code("""\
        import torch, pandas as pd
        from pathlib import Path
        from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
        from torch.utils.data import DataLoader, Dataset, random_split

        class TextDataset(Dataset):
            def __init__(self, texts, tok, ml):
                self.texts, self.tok, self.ml = texts, tok, ml
            def __len__(self): return len(self.texts)
            def __getitem__(self, i):
                enc = self.tok(self.texts[i], truncation=True, max_length=self.ml,
                               padding="max_length", return_tensors="pt")
                ids = enc["input_ids"].squeeze(); mask = enc["attention_mask"].squeeze()
                labels = ids.clone(); labels[mask==0] = -100
                return {"input_ids": ids, "attention_mask": mask, "labels": labels}

        class FullLLM(pl.LightningModule):
            def __init__(self, model, lr, warmup, wd):
                super().__init__(); self.model=model; self.lr=lr; self.warmup=warmup; self.wd=wd
            def training_step(self, batch, _):
                loss = self.model(**batch).loss
                self.log("train/loss", loss, prog_bar=True); return loss
            def validation_step(self, batch, _):
                self.log("val/loss", self.model(**batch).loss, prog_bar=True)
            def configure_optimizers(self):
                opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
                sched = get_cosine_schedule_with_warmup(opt, self.warmup, self.trainer.estimated_stepping_batches)
                return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], trust_remote_code=True, token=HF_TOKEN or None)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], trust_remote_code=True,
                                                      torch_dtype=torch.float16, token=HF_TOKEN or None)

        ext = DATASET_PATH.rsplit(".", 1)[-1]
        df  = pd.read_csv(DATASET_PATH) if ext=="csv" else pd.read_json(DATASET_PATH, lines=(ext=="jsonl"))
        col = CONFIG["text_column"] if CONFIG["text_column"] in df.columns else df.columns[0]
        ds  = TextDataset(df[col].dropna().tolist(), tok, CONFIG["max_seq_length"])
        nv  = max(1, int(len(ds)*CONFIG["val_split"]))
        tr, va = random_split(ds, [len(ds)-nv, nv])
        tdl = DataLoader(tr, batch_size=CONFIG["batch_size"], shuffle=True)
        vdl = DataLoader(va, batch_size=CONFIG["batch_size"])

        module = FullLLM(model, CONFIG["learning_rate"], CONFIG["warmup_steps"], CONFIG["weight_decay"])
        Path(CONFIG["output_dir"]).mkdir(exist_ok=True)
        pl.Trainer(max_epochs=CONFIG["num_epochs"], accumulate_grad_batches=CONFIG["grad_accum"],
                   precision="16-mixed", callbacks=[ModelCheckpoint(dirpath=CONFIG["output_dir"],
                   monitor="val/loss", mode="min"), EarlyStopping(monitor="val/loss",patience=3)],
                   ).fit(module, tdl, vdl)
        model.save_pretrained(CONFIG["output_dir"]); tok.save_pretrained(CONFIG["output_dir"])
        print("✅ Done!")
    """))

    cells.append(_code("""\
        import shutil
        from google.colab import files
        shutil.make_archive("/content/my_model","zip", CONFIG["output_dir"])
        files.download("/content/my_model.zip")
    """))

    nb.cells = cells
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Prompt-based notebook
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt_notebook(cfg: dict) -> nbf.NotebookNode:
    nb    = nbf.v4.new_notebook()
    cells = []

    cells.append(_md(f"""\
        # 🤖 LLM Trainer — Prompt-based Inference
        **Model:** `{cfg['base_model']}`
        **Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

        No training needed! Your dataset is used as few-shot examples injected into the system prompt.
        CPU runtime is fine.
    """))

    cells.append(_install_cell("prompt"))

    cells.append(_code(f"""\
        HF_TOKEN = ""
        BASE_MODEL = "{cfg['base_model']}"
    """))

    cells.append(_md("## 📂 Upload Dataset"))
    cells.append(_code("""\
        from google.colab import files
        import pandas as pd
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
        df = pd.read_csv(f"/content/{filename}")
        print(f"✅ {len(df)} rows loaded")
        df.head()
    """))

    cells.append(_md("## 💬 Run Prompt-based Inference"))
    cells.append(_code(f"""\
        from transformers import pipeline

        generator = pipeline("text-generation", model=BASE_MODEL,
                             token=HF_TOKEN or None, device_map="auto")

        # Build few-shot context from your dataset
        pcol = "{cfg.get('prompt_column','prompt')}"
        rcol = "{cfg.get('response_column','response')}"
        if pcol in df.columns and rcol in df.columns:
            examples = df[[pcol, rcol]].head(5).to_dict("records")
            few_shot = "\\n\\n".join(
                f"Q: {{r[pcol]}}\\nA: {{r[rcol]}}" for r in examples
            )
        else:
            few_shot = df.iloc[:, 0].head(5).str.cat(sep="\\n---\\n")

        def ask(question: str) -> str:
            prompt = f"{{few_shot}}\\n\\nQ: {{question}}\\nA:"
            out = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
            return out[0]["generated_text"].split("A:", 1)[-1].strip()

        # Try it!
        answer = ask("Tell me something interesting.")
        print(answer)
    """))

    cells.append(_md("## 🔁 Interactive Chat Loop"))
    cells.append(_code("""\
        print("Chat with your model (type 'quit' to exit)\\n")
        while True:
            q = input("You: ").strip()
            if q.lower() in ("quit","exit","q"): break
            print(f"Bot: {ask(q)}\\n")
    """))

    nb.cells = cells
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_notebook(cfg: dict) -> tuple[nbf.NotebookNode, str]:
    """
    Build the appropriate notebook for the training mode.

    Returns
    -------
    (notebook, json_string)
    """
    mode = cfg.get("training_mode", "lora")
    if mode in ("lora", "qlora"):
        nb = _build_lora_notebook(cfg)
    elif mode == "full":
        nb = _build_full_notebook(cfg)
    else:
        nb = _build_prompt_notebook(cfg)

    return nb, nbf.writes(nb)


def share_via_gist(nb_json: str, job_id: str, github_token: str = "") -> tuple[str, str]:
    """
    Upload the notebook to a GitHub Gist and return
    (gist_id, colab_open_url).

    Falls back to a local-save URL if no token is provided.
    """
    token = github_token or GITHUB_TOKEN
    filename = f"llm_trainer_{job_id}.ipynb"

    if not token:
        # No token: user must upload manually
        return "", f"https://colab.research.google.com/notebooks/welcome.ipynb"

    try:
        from github import Github
        g    = Github(token)
        user = g.get_user()
        gist = user.create_gist(
            public=False,
            files={filename: type("F", (), {"content": nb_json})()},
            description=f"LLM Trainer job {job_id}",
        )
        colab_url = f"https://colab.research.google.com/gist/{user.login}/{gist.id}"
        return gist.id, colab_url

    except Exception as e:
        print(f"[Gist] Upload failed: {e}")
        return "", ""


def save_notebook_locally(nb_json: str, job_id: str, output_dir: Path) -> Path:
    """Save the notebook to disk so the user can download it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"llm_trainer_{job_id}.ipynb"
    path.write_text(nb_json, encoding="utf-8")
    return path
