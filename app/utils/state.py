"""Session-state helpers shared across all Streamlit pages."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from app.utils.config import JOBS_DIR, TrainingConfig


# ── Job status literals ───────────────────────────────────────
PENDING   = "pending"
RUNNING   = "running"
DONE      = "done"
FAILED    = "failed"


def init_session() -> None:
    """Initialise all required keys in st.session_state."""
    defaults: Dict[str, Any] = {
        "jobs":          [],        # list[dict]  – training job records
        "active_job_id": None,      # str | None
        "dataset_info":  None,      # dict | None
        "config":        TrainingConfig().to_dict(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Job CRUD ──────────────────────────────────────────────────

def create_job(config: dict, notebook_url: str, gist_id: str = "") -> str:
    job_id = str(uuid.uuid4())[:8]
    job = {
        "id":           job_id,
        "status":       PENDING,
        "config":       config,
        "notebook_url": notebook_url,
        "gist_id":      gist_id,
        "created_at":   datetime.utcnow().isoformat(),
        "completed_at": None,
        "logs":         [],
    }
    # Guard: initialise the list if session hasn't been set up yet
    # (can happen if a page calls create_job before init_session)
    try:
        if "jobs" not in st.session_state:
            st.session_state["jobs"] = []
        st.session_state["jobs"].insert(0, job)
    except Exception:
        pass  # outside Streamlit runtime; disk persistence below is sufficient
    _persist_job(job)
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    for j in st.session_state.get("jobs", []):
        if j["id"] == job_id:
            return j
    # fall back to disk
    path = JOBS_DIR / f"{job_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def update_job(job_id: str, **kwargs) -> None:
    for j in st.session_state.get("jobs", []):
        if j["id"] == job_id:
            j.update(kwargs)
            _persist_job(j)
            return
    # Not found in session state — fall back to disk so updates survive reloads
    path = JOBS_DIR / f"{job_id}.json"
    if path.exists():
        job = json.loads(path.read_text())
        job.update(kwargs)
        _persist_job(job)


def list_jobs() -> List[dict]:
    # Merge in-memory + disk (disk wins for persistent state)
    on_disk = {
        p.stem: json.loads(p.read_text())
        for p in sorted(JOBS_DIR.glob("*.json"), reverse=True)
    }
    in_mem  = {j["id"]: j for j in st.session_state.get("jobs", [])}
    merged  = {**on_disk, **in_mem}   # in-memory takes precedence
    return sorted(merged.values(), key=lambda x: x["created_at"], reverse=True)


def _persist_job(job: dict) -> None:
    path = JOBS_DIR / f"{job['id']}.json"
    path.write_text(json.dumps(job, indent=2))
