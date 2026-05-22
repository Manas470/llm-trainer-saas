"""Central configuration & constants for the LLM Trainer SaaS."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[2]
UPLOAD_DIR = ROOT_DIR / "uploads"
JOBS_DIR   = ROOT_DIR / "jobs"
for d in (UPLOAD_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Tokens ────────────────────────────────────────────────────
HF_TOKEN     = os.getenv("HF_TOKEN", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
APP_SECRET   = os.getenv("APP_SECRET_KEY", "dev-secret")

# ── Supported base models ─────────────────────────────────────
BASE_MODELS: Dict[str, str] = {
    "GPT-2 (124M) — fastest, CPU-OK":          "gpt2",
    "GPT-2 Medium (345M)":                      "gpt2-medium",
    "GPT-2 Large (774M)":                       "gpt2-large",
    "DistilGPT-2 (82M) — lightest":             "distilgpt2",
    "TinyLlama 1.1B":                           "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2 (2.7B)":                             "microsoft/phi-2",
    "Mistral 7B Instruct":                      "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama-3.2 1B Instruct":                    "meta-llama/Llama-3.2-1B-Instruct",
    "Llama-3.2 3B Instruct":                    "meta-llama/Llama-3.2-3B-Instruct",
    "Gemma 2B":                                 "google/gemma-2b-it",
    "Falcon 7B Instruct":                       "tiiuae/falcon-7b-instruct",
    "OPT 1.3B":                                 "facebook/opt-1.3b",
}

# ── Training modes ────────────────────────────────────────────
TRAINING_MODES = {
    "lora":   "LoRA / QLoRA Fine-tuning  (recommended — fast, GPU-friendly)",
    "full":   "Full Fine-tuning  (all weights — needs big GPU)",
    "prompt": "Prompt-based  (no training — data used as context/examples)",
}

# ── Dataset formats ───────────────────────────────────────────
SUPPORTED_FORMATS = [".csv", ".jsonl", ".json", ".txt", ".parquet"]

# ── Default hyperparameters ───────────────────────────────────
@dataclass
class TrainingConfig:
    # Model
    base_model: str       = "gpt2"
    training_mode: str    = "lora"

    # Data
    dataset_path: str     = ""
    text_column: str      = "text"
    prompt_column: str    = "prompt"
    response_column: str  = "response"
    max_seq_length: int   = 512
    val_split: float      = 0.1

    # Training
    num_epochs: int       = 3
    batch_size: int       = 4
    grad_accum: int       = 4
    learning_rate: float  = 2e-4
    warmup_steps: int     = 100
    weight_decay: float   = 0.01
    max_grad_norm: float  = 1.0
    fp16: bool            = True
    bf16: bool            = False

    # LoRA specific
    lora_r: int           = 16
    lora_alpha: int       = 32
    lora_dropout: float   = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    use_qlora: bool       = True   # 4-bit quantisation

    # Output
    output_dir: str       = "./outputs"
    hf_repo_id: str       = ""     # e.g. "username/my-model"
    push_to_hub: bool     = False

    # Logging
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 1

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
