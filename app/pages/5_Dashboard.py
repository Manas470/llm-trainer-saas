"""Page 5 — Analytics dashboard."""
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from app.utils.state import init_session, list_jobs, PENDING, RUNNING, DONE, FAILED
from app.utils.config import BASE_MODELS

st.set_page_config(page_title="Dashboard · LLM Trainer", page_icon="📊", layout="wide")
init_session()

st.title("📊 Dashboard")

jobs = list_jobs()

if not jobs:
    st.info("No jobs yet. Start by uploading data and configuring training.")
    st.stop()

# ── Build dataframe ───────────────────────────────────────────
rows = []
for j in jobs:
    cfg = j.get("config", {})
    rows.append({
        "id":            j["id"],
        "status":        j["status"],
        "model":         cfg.get("base_model", "unknown"),
        "mode":          cfg.get("training_mode", "?"),
        "epochs":        cfg.get("num_epochs", 0),
        "batch":         cfg.get("batch_size", 0),
        "lr":            cfg.get("learning_rate", 0),
        "lora_r":        cfg.get("lora_r", 0),
        "qlora":         cfg.get("use_qlora", False),
        "push_hub":      cfg.get("push_to_hub", False),
        "created_at":    j.get("created_at", ""),
        "completed_at":  j.get("completed_at") or "",
    })
df = pd.DataFrame(rows)

# ── KPI row ───────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total jobs",  len(df))
k2.metric("Completed",   len(df[df.status == DONE]))
k3.metric("Running",     len(df[df.status == RUNNING]))
k4.metric("Pending",     len(df[df.status == PENDING]))
k5.metric("Failed",      len(df[df.status == FAILED]))

st.divider()

# ── Charts ────────────────────────────────────────────────────
row1_c1, row1_c2 = st.columns(2)

# Status distribution
with row1_c1:
    st.subheader("Job Status Distribution")
    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    colors = {PENDING: "#FFA500", RUNNING: "#6C63FF", DONE: "#00C853", FAILED: "#FF4444"}
    fig = px.pie(status_counts, names="Status", values="Count",
                 color="Status", color_discrete_map=colors, hole=0.4)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
    st.plotly_chart(fig, use_container_width=True)

# Training mode distribution
with row1_c2:
    st.subheader("Training Mode Usage")
    mode_counts = df["mode"].value_counts().reset_index()
    mode_counts.columns = ["Mode", "Count"]
    fig2 = px.bar(mode_counts, x="Mode", y="Count",
                  color="Mode", color_discrete_sequence=["#6C63FF","#3ECFCF","#FF6B6B"])
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#fff", showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

row2_c1, row2_c2 = st.columns(2)

# Models used
with row2_c1:
    st.subheader("Base Models Used")
    model_counts = df["model"].value_counts().reset_index()
    model_counts.columns = ["Model", "Count"]
    # Shorten model names
    model_counts["Model"] = model_counts["Model"].apply(lambda m: m.split("/")[-1])
    fig3 = px.bar(model_counts, x="Count", y="Model", orientation="h",
                  color_discrete_sequence=["#6C63FF"])
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#fff", yaxis_autorange="reversed")
    st.plotly_chart(fig3, use_container_width=True)

# Jobs over time
with row2_c2:
    st.subheader("Jobs Created Over Time")
    if df["created_at"].any():
        df["date"] = pd.to_datetime(df["created_at"]).dt.date
        timeline = df.groupby(["date","status"]).size().reset_index(name="count")
        fig4 = px.line(timeline, x="date", y="count", color="status",
                       color_discrete_map={DONE:"#00C853", PENDING:"#FFA500",
                                           RUNNING:"#6C63FF", FAILED:"#FF4444"})
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#fff")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Not enough date data yet.")

st.divider()

# ── All jobs table ────────────────────────────────────────────
st.subheader("📋 All Jobs")
display_df = df[["id","status","model","mode","epochs","batch","lr","created_at"]].copy()
display_df.columns = ["ID","Status","Model","Mode","Epochs","Batch","LR","Created"]
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Export jobs CSV",
    display_df.to_csv(index=False),
    "llm_trainer_jobs.csv",
    "text/csv",
)
