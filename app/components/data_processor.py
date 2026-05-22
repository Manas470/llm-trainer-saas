"""
Data processing utilities: format detection, cleaning, stats.
Used by the Upload page and notebook generator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd


def detect_dataset_type(df: pd.DataFrame,
                        prompt_col: str = "prompt",
                        response_col: str = "response",
                        text_col: str = "text") -> str:
    """Return 'instruction', 'text', or 'unknown'."""
    if prompt_col in df.columns and response_col in df.columns:
        return "instruction"
    if text_col in df.columns:
        return "text"
    return "unknown"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully-empty rows and strip whitespace from string cols."""
    df = df.dropna(how="all")
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""]
    return df.reset_index(drop=True)


def compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a dict of basic dataset stats."""
    stats: Dict[str, Any] = {
        "rows":    len(df),
        "columns": list(df.columns),
        "dtypes":  {c: str(t) for c, t in df.dtypes.items()},
        "nulls":   df.isnull().sum().to_dict(),
    }
    for col in df.select_dtypes("object").columns:
        lengths = df[col].dropna().str.len()
        stats[f"{col}_avg_len"]  = round(lengths.mean(), 1)
        stats[f"{col}_max_len"]  = int(lengths.max())
    return stats


def estimate_training_time(n_rows: int, mode: str, epochs: int) -> str:
    """Very rough estimate of training time on a T4 GPU."""
    # ~60 samples/sec on T4 for LoRA; ~20 for full
    sps = 60 if mode in ("lora","qlora") else 20
    seconds = (n_rows * epochs) / sps
    if seconds < 60:
        return f"~{int(seconds)}s"
    if seconds < 3600:
        return f"~{int(seconds/60)}min"
    return f"~{seconds/3600:.1f}h"
