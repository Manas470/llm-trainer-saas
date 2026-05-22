"""
PyTorch Lightning LightningModule — unified for all training modes.

Supports:
  • LoRA / QLoRA (PEFT)  — memory-efficient, Colab-friendly
  • Full fine-tuning      — all weights updated
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
class CausalLMLightningModule(pl.LightningModule):
    """
    General-purpose causal-LM Lightning module.

    Parameters
    ----------
    model_name_or_path : str
    training_mode      : "lora" | "qlora" | "full"
    learning_rate      : float
    warmup_steps       : int
    weight_decay       : float
    lora_r             : int
    lora_alpha         : int
    lora_dropout       : float
    lora_target_modules: list[str]
    use_qlora          : bool  — 4-bit quantisation
    push_to_hub        : bool
    hf_repo_id         : str
    """

    def __init__(
        self,
        model_name_or_path: str,
        training_mode:       str   = "lora",
        learning_rate:       float = 2e-4,
        warmup_steps:        int   = 100,
        weight_decay:        float = 0.01,
        max_grad_norm:       float = 1.0,
        lora_r:              int   = 16,
        lora_alpha:          int   = 32,
        lora_dropout:        float = 0.05,
        lora_target_modules: list  | None = None,
        use_qlora:           bool  = True,
        push_to_hub:         bool  = False,
        hf_repo_id:          str   = "",
        hf_token:            str   = "",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_name        = model_name_or_path
        self.training_mode     = training_mode.lower()
        self.lr                = learning_rate
        self.warmup_steps      = warmup_steps
        self.weight_decay      = weight_decay
        self.max_grad_norm     = max_grad_norm
        self.lora_r            = lora_r
        self.lora_alpha        = lora_alpha
        self.lora_dropout      = lora_dropout
        self.lora_targets      = lora_target_modules or ["q_proj", "v_proj"]
        self.use_qlora         = use_qlora
        self.push_to_hub       = push_to_hub
        self.hf_repo_id        = hf_repo_id
        self.hf_token          = hf_token

        self.model = self._build_model()

    # ── model construction ───────────────────────────────────
    def _build_model(self) -> Any:
        is_lora = self.training_mode in ("lora", "qlora")

        # 4-bit quantisation for QLoRA
        bnb_config = None
        if is_lora and self.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=self.hf_token or None,
        )

        if is_lora:
            if not _PEFT_AVAILABLE:
                raise ImportError("Install 'peft' for LoRA fine-tuning: pip install peft")
            if self.use_qlora:
                model = prepare_model_for_kbit_training(model)
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.lora_targets,
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        return model

    # ── training step ────────────────────────────────────────
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        loss    = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ── validation step ──────────────────────────────────────
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs  = self.model(**batch)
        val_loss = outputs.loss
        ppl      = torch.exp(val_loss)
        self.log("val/loss",        val_loss, prog_bar=True,  on_epoch=True)
        self.log("val/perplexity",  ppl,      prog_bar=False, on_epoch=True)

    # ── optimiser & scheduler ────────────────────────────────
    def configure_optimizers(self):
        # Only optimise trainable params (critical for LoRA)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Total steps computed after trainer sets them up
        total_steps = self.trainer.estimated_stepping_batches
        scheduler   = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # ── save / push ──────────────────────────────────────────
    def on_train_end(self) -> None:
        """Save model and optionally push to HuggingFace Hub."""
        output_dir = self.hparams.get("output_dir", "./outputs")
        self.model.save_pretrained(output_dir)

        if self.push_to_hub and self.hf_repo_id:
            self.model.push_to_hub(
                self.hf_repo_id,
                token=self.hf_token or None,
                private=False,
            )
            print(f"✅ Model pushed → https://huggingface.co/{self.hf_repo_id}")


# ─────────────────────────────────────────────────────────────
def load_tokenizer(model_name: str, hf_token: str = "") -> Any:
    tok = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token or None,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
