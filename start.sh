#!/bin/bash
# Quick start (dev / manual run)
source venv/bin/activate
uvicorn shemes_rag:app --host 0.0.0.0 --port 8010 --workers 1 --log-level info
