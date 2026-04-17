#!/bin/bash
# Quick start for local development
# Usage: bash start.sh

set -e

if [ ! -d "venv" ]; then
    echo "No venv found. Run the setup steps in README.md first."
    exit 1
fi

source venv/bin/activate
python -m uvicorn shemes_rag:app --host 0.0.0.0 --port 8013 --workers 1 --reload --log-level info
