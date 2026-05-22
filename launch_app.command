#!/bin/bash
# ── LLM Trainer SaaS — one-click launcher ──────────────────────
cd "$(dirname "$0")"

echo "========================================"
echo "  LLM Trainer SaaS — Starting up..."
echo "========================================"
echo ""

# Skip Streamlit's first-run email prompt
mkdir -p ~/.streamlit
cat > ~/.streamlit/credentials.toml << 'TOML'
[general]
email = ""
TOML

# Suppress usage stats prompt too
cat > ~/.streamlit/config.toml << 'TOML'
[browser]
gatherUsageStats = false
TOML

echo "📦 Upgrading pip..."
python3 -m pip install --upgrade pip -q

echo "📦 Installing dependencies (this takes ~1 min first time)..."
python3 -m pip install --upgrade \
  "streamlit>=1.38.0" \
  "starlette>=0.40.0" \
  "pandas" "plotly" "nbformat" "PyGithub" \
  "huggingface-hub" "datasets" "python-dotenv" -q

echo ""
echo "✅ Dependencies ready!"
echo "🚀 Launching app at http://localhost:8501"
echo "   (Your browser will open automatically)"
echo ""

python3 -m streamlit run app/main.py --server.port 8501
