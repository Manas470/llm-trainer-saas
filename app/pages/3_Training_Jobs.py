"""Page 3 — Training job dashboard."""
from datetime import datetime

import streamlit as st

from app.utils.state import (
    init_session, list_jobs, update_job,
    PENDING, RUNNING, DONE, FAILED,
)
from app.utils.config import JOBS_DIR

st.set_page_config(page_title="Training Jobs · LLM Trainer", page_icon="📋", layout="wide")
init_session()

st.title("📋 Training Jobs")

# ── Status colours ────────────────────────────────────────────
STATUS_COLOR = {
    PENDING: "#FFA500",
    RUNNING: "#6C63FF",
    DONE:    "#00C853",
    FAILED:  "#FF4444",
}
STATUS_ICON  = {PENDING: "⏳", RUNNING: "🔄", DONE: "✅", FAILED: "❌"}

jobs = list_jobs()
if not jobs:
    st.info("No jobs yet. Go to **Configure Training** to generate your first notebook.")
    if st.button("Go to Configure Training →", type="primary"):
        st.switch_page("pages/2_Configure_Training.py")
    st.stop()

# ── Filters ───────────────────────────────────────────────────
col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    filter_status = st.multiselect(
        "Filter by status",
        [PENDING, RUNNING, DONE, FAILED],
        default=[PENDING, RUNNING, DONE, FAILED],
    )
with col_f2:
    st.metric("Total jobs", len(jobs))

filtered = [j for j in jobs if j["status"] in filter_status]
st.divider()

# ── Job cards ─────────────────────────────────────────────────
for job in filtered:
    status  = job["status"]
    color   = STATUS_COLOR.get(status, "#888")
    icon    = STATUS_ICON.get(status, "❓")
    model   = job["config"].get("base_model", "unknown")
    mode    = job["config"].get("training_mode", "?").upper()
    created = job.get("created_at", "")[:16].replace("T", " ")

    with st.expander(
        f"{icon} Job **{job['id']}** · `{model}` · {mode} · {created}",
        expanded=(status in (RUNNING, PENDING)),
    ):
        c1, c2, c3 = st.columns([2, 2, 1])

        with c1:
            st.markdown(f"**Status:** <span style='color:{color}'>{status.upper()}</span>", unsafe_allow_html=True)
            st.markdown(f"**Model:** `{model}`")
            st.markdown(f"**Mode:** `{mode}`")
            st.markdown(f"**Created:** {created}")

        with c2:
            cfg = job["config"]
            st.markdown(f"**Epochs:** {cfg.get('num_epochs','—')}")
            st.markdown(f"**Batch size:** {cfg.get('batch_size','—')} × {cfg.get('grad_accum','—')} accum")
            st.markdown(f"**LR:** {cfg.get('learning_rate','—')}")
            if cfg.get("training_mode") in ("lora","qlora"):
                st.markdown(f"**LoRA r:** {cfg.get('lora_r','—')} · QLoRA: {cfg.get('use_qlora',False)}")

        with c3:
            # Colab link
            nb_url = job.get("notebook_url", "")
            if nb_url:
                st.link_button("🚀 Open Colab", nb_url, use_container_width=True)

            # Download notebook from disk
            nb_path = JOBS_DIR / f"llm_trainer_{job['id']}.ipynb"
            if nb_path.exists():
                st.download_button(
                    "⬇️ Download .ipynb",
                    data      = nb_path.read_bytes(),
                    file_name = nb_path.name,
                    mime      = "application/json",
                    use_container_width=True,
                )

            # Manual status update
            new_status = st.selectbox(
                "Update status",
                [PENDING, RUNNING, DONE, FAILED],
                index=[PENDING, RUNNING, DONE, FAILED].index(status),
                key=f"sel_{job['id']}",
            )
            if new_status != status:
                if st.button("Save", key=f"save_{job['id']}", use_container_width=True):
                    completed_at = datetime.utcnow().isoformat() if new_status == DONE else None
                    update_job(job["id"], status=new_status, completed_at=completed_at)
                    st.rerun()

        # HF repo link
        hf = cfg.get("hf_repo_id", "")
        if hf and status == DONE:
            st.markdown(f"🤗 **Model on HuggingFace:** [https://huggingface.co/{hf}](https://huggingface.co/{hf})")

        # Raw config
        with st.popover("🔍 Full config"):
            st.json(cfg)
