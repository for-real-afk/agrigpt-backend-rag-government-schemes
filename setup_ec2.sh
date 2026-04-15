#!/bin/bash
# EC2 t2.micro setup for AgriGPT RAG Backend
# Run once as: bash setup_ec2.sh

set -e

echo "=== System update ==="
sudo apt-get update -y && sudo apt-get upgrade -y

echo "=== Install Python 3.11 + tools ==="
sudo apt-get install -y python3.11 python3.11-venv python3-pip git

# ── Swap file (critical on t2.micro — BGE model needs ~450 MB RAM) ────────────
echo "=== Creating 2 GB swap file ==="
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap enabled."
else
    echo "Swap already exists, skipping."
fi

# ── Python virtual environment ────────────────────────────────────────────────
echo "=== Creating virtual environment ==="
python3.11 -m venv venv
source venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements-backend.txt

# ── Environment file ──────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo ">>> IMPORTANT: Edit .env with your API keys before starting the server."
    echo "    nano .env"
fi

# ── Pre-download the embedding model (avoids cold-start delay) ────────────────
echo "=== Pre-downloading BAAI/bge-base-en-v1.5 model ==="
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5'); print('Model cached.')"

# ── Install systemd service ───────────────────────────────────────────────────
PROJECT_DIR=$(pwd)
SERVICE_FILE=/etc/systemd/system/agrigpt-rag.service

echo "=== Installing systemd service ==="
sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=AgriGPT RAG Backend
After=network.target

[Service]
User=$USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv/bin/uvicorn shemes_rag:app --host 0.0.0.0 --port 8010 --workers 1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable agrigpt-rag
sudo systemctl start agrigpt-rag

echo ""
echo "=== Done! ==="
echo "Service status:  sudo systemctl status agrigpt-rag"
echo "Live logs:       sudo journalctl -u agrigpt-rag -f"
echo "API docs:        http://<your-ec2-public-ip>:8010/docs"
echo ""
echo "EC2 Security Group: open inbound TCP port 8010 (or 80 if you set up Nginx)."
