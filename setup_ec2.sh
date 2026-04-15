#!/bin/bash
# EC2 setup for AgriGPT RAG Backend
# Tested on Ubuntu 20.04 and 22.04
# Run once as: bash setup_ec2.sh

set -e

echo "=== System update ==="
sudo apt-get update -y && sudo apt-get upgrade -y

# ── Python: use system python3 or install 3.11 via deadsnakes ─────────────────
echo "=== Checking Python version ==="
PYTHON_BIN=$(which python3)
PY_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

echo "Found: Python $PY_VERSION"

if [ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 9 ]; then
    echo "=== Python < 3.9 detected — installing 3.11 via deadsnakes PPA ==="
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils
    PYTHON_BIN=$(which python3.11)
else
    echo "Python $PY_VERSION is sufficient — using system python3."
    sudo apt-get install -y python3-venv python3-pip
fi

sudo apt-get install -y git

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
echo "=== Creating virtual environment using $PYTHON_BIN ==="
$PYTHON_BIN -m venv venv
source venv/bin/activate

echo "=== Installing dependencies ==="
pip install --upgrade pip
# Install CPU-only torch first — avoids downloading 900 MB of CUDA libs on a GPU-less instance
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-backend.txt

# ── Environment file ──────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo ">>> IMPORTANT: Edit .env with your API keys before starting the server."
    echo "    nano .env"
fi

# ── Pre-download the embedding model (avoids cold-start OOM on first request) ─
echo "=== Pre-downloading BAAI/bge-base-en-v1.5 model ==="
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5'); print('Model cached.')"

# ── Systemd service ───────────────────────────────────────────────────────────
PROJECT_DIR=$(pwd)
VENV_UVICORN="$PROJECT_DIR/venv/bin/uvicorn"
SERVICE_FILE=/etc/systemd/system/agrigpt-rag.service
CURRENT_USER=$(whoami)

echo "=== Installing systemd service ==="
sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=AgriGPT RAG Backend
After=network.target

[Service]
User=$CURRENT_USER
WorkingDirectory=$PROJECT_DIR
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$VENV_UVICORN shemes_rag:app --host 0.0.0.0 --port 8010 --workers 1 --log-level info
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

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
echo "Restart:         sudo systemctl restart agrigpt-rag"
echo "API docs:        http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8010/docs"
echo ""
echo "EC2 Security Group: open inbound TCP port 8010 (or 80 if you set up Nginx)."
