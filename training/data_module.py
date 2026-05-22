"""PyTorch Lightning DataModule — supports CSV, JSONL, TXT, Parquet."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizerBase


# ─────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    """Generic causal-LM dataset: tokenises and packs to max_length."""

    def __init__(
        self,
        texts: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = texts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels         = input_ids.clone()
        # mask padding tokens from loss
        labels[attention_mask == 0] = -100
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


class InstructionDataset(Dataset):
    """Chat/instruction dataset: formats prompt→response pairs."""

    PROMPT_TEMPLATE = (
        "### Instruction:\n{prompt}\n\n### Response:\n{response}"
    )

    def __init__(
        self,
        records: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        prompt_col: str  = "prompt",
        response_col: str = "response",
        max_length: int  = 512,
    ) -> None:
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        self.prompt_col   = prompt_col
        self.response_col = response_col
        self.records      = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row  = self.records[idx]
        text = self.PROMPT_TEMPLATE.format(
            prompt=row.get(self.prompt_col, ""),
            response=row.get(self.response_col, ""),
        )
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels         = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────
class LLMDataModule(pl.LightningDataModule):
    """
    Unified Lightning DataModule.

    Parameters
    ----------
    dataset_path   : str  — path to CSV / JSONL / TXT / Parquet
    tokenizer      : PreTrainedTokenizerBase
    text_column    : str  — column name for raw-text datasets
    prompt_column  : str  — column for instruction-tuning prompts
    response_column: str  — column for instruction-tuning responses
    max_seq_length : int
    val_split      : float — fraction reserved for validation
    batch_size     : int
    num_workers    : int
    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        text_column:     str   = "text",
        prompt_column:   str   = "prompt",
        response_column: str   = "response",
        max_seq_length:  int   = 512,
        val_split:       float = 0.1,
        batch_size:      int   = 4,
        num_workers:     int   = 2,
    ) -> None:
        super().__init__()
        self.dataset_path    = Path(dataset_path)
        self.tokenizer       = tokenizer
        self.text_column     = text_column
        self.prompt_column   = prompt_column
        self.response_column = response_column
        self.max_seq_length  = max_seq_length
        self.val_split       = val_split
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.train_ds: Optional[Dataset] = None
        self.val_ds:   Optional[Dataset] = None

    # ── loading ───────────────────────────────────────────────
    def _load_raw(self) -> pd.DataFrame:
        ext = self.dataset_path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(self.dataset_path)
        if ext in (".jsonl", ".json"):
            return pd.read_json(self.dataset_path, lines=(ext == ".jsonl"))
        if ext == ".parquet":
            return pd.read_parquet(self.dataset_path)
        if ext == ".txt":
            lines = self.dataset_path.read_text(encoding="utf-8").splitlines()
            return pd.DataFrame({"text": [l for l in lines if l.strip()]})
        raise ValueError(f"Unsupported file type: {ext}")

    def setup(self, stage: Optional[str] = None) -> None:
        df = self._load_raw()

        # Decide dataset type: instruction-tuning or plain-text
        has_instruction_cols = (
            self.prompt_column   in df.columns
            and self.response_column in df.columns
        )

        if has_instruction_cols:
            records = df[[self.prompt_column, self.response_column]].to_dict("records")
            full_ds = InstructionDataset(
                records,
                self.tokenizer,
                self.prompt_column,
                self.response_column,
                self.max_seq_length,
            )
        else:
            col = self.text_column if self.text_column in df.columns else df.columns[0]
            texts   = df[col].dropna().astype(str).tolist()
            full_ds = TextDataset(texts, self.tokenizer, self.max_seq_length)

        # Train / val split
        n_val   = max(1, int(len(full_ds) * self.val_split))
        n_train = len(full_ds) - n_val
        self.train_ds, self.val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    # ── loaders ───────────────────────────────────────────────
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
