#!/bin/bash
# Run Cervical Posture Detection app - keep this terminal open while using the app.
cd "$(dirname "$0")"

echo "Starting Streamlit app..."
echo ""

# Prefer Anaconda streamlit if available
if [ -x "/opt/anaconda3/bin/streamlit" ]; then
    /opt/anaconda3/bin/streamlit run app.py --server.port=8501
elif [ -f ".venv/bin/streamlit" ]; then
    .venv/bin/streamlit run app.py --server.port=8501
else
    streamlit run app.py --server.port=8501
fi

echo ""
echo "App stopped. Close this window or run this script again to restart."
