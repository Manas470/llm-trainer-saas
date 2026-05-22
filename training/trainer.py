"""
Entry-point trainer: wires DataModule + LightningModule + Trainer.
Can be run directly (CLI) or imported by the Colab notebook.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger

from training.data_module import LLMDataModule
from training.lightning_module import CausalLMLightningModule, load_tokenizer


def run_training(
    # ── Model ─────────────────────────────────────────────────
    base_model:       str   = "gpt2",
    training_mode:    str   = "lora",      # lora | full | prompt
    # ── Data ──────────────────────────────────────────────────
    dataset_path:     str   = "",
    text_column:      str   = "text",
    prompt_column:    str   = "prompt",
    response_column:  str   = "response",
    max_seq_length:   int   = 512,
    val_split:        float = 0.1,
    # ── Training ──────────────────────────────────────────────
    num_epochs:       int   = 3,
    batch_size:       int   = 4,
    grad_accum:       int   = 4,
    learning_rate:    float = 2e-4,
    warmup_steps:     int   = 100,
    weight_decay:     float = 0.01,
    max_grad_norm:    float = 1.0,
    fp16:             bool  = True,
    # ── LoRA ──────────────────────────────────────────────────
    lora_r:           int   = 16,
    lora_alpha:       int   = 32,
    lora_dropout:     float = 0.05,
    use_qlora:        bool  = True,
    # ── Output ────────────────────────────────────────────────
    output_dir:       str   = "./outputs",
    hf_repo_id:       str   = "",
    push_to_hub:      bool  = False,
    hf_token:         str   = "",
    # ── Misc ──────────────────────────────────────────────────
    seed:             int   = 42,
    log_every_n:      int   = 10,
) -> None:

    pl.seed_everything(seed, workers=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Prompt-based: no training needed ──────────────────────
    if training_mode == "prompt":
        print("ℹ️  Prompt-based mode — skipping gradient training.")
        print("   Your dataset has been processed and is ready for retrieval-augmented inference.")
        return

    # ── Tokenizer ─────────────────────────────────────────────
    print(f"📥 Loading tokenizer: {base_model}")
    tokenizer = load_tokenizer(base_model, hf_token)

    # ── DataModule ────────────────────────────────────────────
    print(f"📂 Loading dataset: {dataset_path}")
    dm = LLMDataModule(
        dataset_path    = dataset_path,
        tokenizer       = tokenizer,
        text_column     = text_column,
        prompt_column   = prompt_column,
        response_column = response_column,
        max_seq_length  = max_seq_length,
        val_split       = val_split,
        batch_size      = batch_size,
        num_workers     = 2,
    )

    # ── Lightning Module ──────────────────────────────────────
    print(f"🤖 Building model [{training_mode.upper()}]: {base_model}")
    module = CausalLMLightningModule(
        model_name_or_path  = base_model,
        training_mode       = training_mode,
        learning_rate       = learning_rate,
        warmup_steps        = warmup_steps,
        weight_decay        = weight_decay,
        max_grad_norm       = max_grad_norm,
        lora_r              = lora_r,
        lora_alpha          = lora_alpha,
        lora_dropout        = lora_dropout,
        use_qlora           = use_qlora,
        push_to_hub         = push_to_hub,
        hf_repo_id          = hf_repo_id,
        hf_token            = hf_token,
    )

    # ── Callbacks ─────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath    = output_dir,
            filename   = "best-{epoch:02d}-{val/loss:.4f}",
            monitor    = "val/loss",
            mode       = "min",
            save_top_k = 2,
        ),
        EarlyStopping(
            monitor   = "val/loss",
            patience  = 3,
            mode      = "min",
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # ── Logger ────────────────────────────────────────────────
    logger = CSVLogger(save_dir=output_dir, name="logs")

    # ── Trainer ───────────────────────────────────────────────
    precision = "16-mixed" if fp16 else "32-true"
    trainer = pl.Trainer(
        max_epochs              = num_epochs,
        accumulate_grad_batches = grad_accum,
        gradient_clip_val       = max_grad_norm,
        precision               = precision,
        callbacks               = callbacks,
        logger                  = logger,
        log_every_n_steps       = log_every_n,
        deterministic           = False,   # faster with flash-attn
        enable_progress_bar     = True,
    )

    print("🚀 Starting training …")
    trainer.fit(module, datamodule=dm)

    print(f"\n✅ Training complete! Model saved → {output_dir}")
    if push_to_hub and hf_repo_id:
        print(f"🤗 Model available at: https://huggingface.co/{hf_repo_id}")


# ── CLI entry-point ───────────────────────────────────────────
if __name__ == "__main__":
    import json, argparse

    parser = argparse.ArgumentParser(description="LLM Trainer CLI")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to JSON config file")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    run_training(**cfg)
