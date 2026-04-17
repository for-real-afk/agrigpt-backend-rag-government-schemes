# AgriGPT RAG Backend — Government Schemes

A Retrieval-Augmented Generation (RAG) API that lets farmers upload agricultural PDFs and query them in natural language. Built with FastAPI, Pinecone, BGE embeddings, and Gemini / Groq as the LLM.

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Vector DB | Pinecone serverless (cosine, 768 dims) |
| Embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers) |
| LLM | Gemini 2.0 Flash (primary) · Groq llama-3.3-70b (fallback) |
| Frontend | React (served by Nginx) or Streamlit (local dev) |

---

## Prerequisites

- Python 3.9 or higher
- A [Pinecone](https://www.pinecone.io/) account — free Starter tier is enough
- At least one of:
  - [Google AI Studio](https://aistudio.google.com/) API key (Gemini)
  - [Groq](https://console.groq.com/) API key (free: 14,400 req/day)

---

## Local Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd agrigpt-backend-rag-government-schemes
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Install dependencies

On any machine, install CPU-only PyTorch first to avoid downloading 900 MB of CUDA libraries:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
PINECONE_API_KEY=your_pinecone_api_key

# Gemini — add up to 3 keys; they rotate automatically when quota is hit
GEMINI_API_KEY=your_gemini_key
GEMINI_API_KEY_2=your_second_key      # optional
GEMINI_API_KEY_3=your_third_key       # optional
GEMINI_MODEL=gemini-2.0-flash

# Groq fallback — used when all Gemini keys are exhausted
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.3-70b-versatile
```

You only need **one** of Gemini or Groq. Having both gives automatic fallback.

### 5. Start the backend

```bash
bash start.sh
```

Or directly:

```bash
python -m uvicorn shemes_rag:app --host 0.0.0.0 --port 8013 --reload
```

API is now running at `http://localhost:8013`  
Interactive docs: `http://localhost:8013/docs`

### 6. (Optional) Run the Streamlit frontend

In a separate terminal:

```bash
source venv/bin/activate
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Set `API_URL=http://localhost:8013` in the Streamlit sidebar if it is not already the default.

---

## Environment Variables Reference

| Variable | Required | Description |
|---|---|---|
| `PINECONE_API_KEY` | Yes | Pinecone project API key |
| `GEMINI_API_KEY` | One of Gemini/Groq | Primary LLM key |
| `GEMINI_API_KEY_2` | No | Rotated when key 1 hits quota |
| `GEMINI_API_KEY_3` | No | Rotated when key 2 hits quota |
| `GEMINI_MODEL` | No | Default: `gemini-2.0-flash` |
| `GROQ_API_KEY` | One of Gemini/Groq | Fallback LLM (free tier) |
| `GROQ_MODEL` | No | Default: `llama-3.3-70b-versatile` |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check + Pinecone connection |
| `GET` | `/stats` | Total vectors, index dimensions |
| `POST` | `/upload` | Upload a `.pdf`, `.txt`, or `.docx` |
| `GET` | `/files` | List all uploaded filenames |
| `DELETE` | `/files/{filename}` | Delete a file and its embeddings |
| `DELETE` | `/clear` | Wipe the entire index |
| `POST` | `/query` | Ask a question; returns answer + sources |
| `POST` | `/eval/generate-silver` | Auto-generate Q&A pairs from a file |
| `GET` | `/eval/pairs` | List silver / golden eval pairs |
| `POST` | `/eval/promote` | Promote a silver pair to golden |
| `DELETE` | `/eval/pairs/{id}` | Delete an eval pair |
| `POST` | `/eval/run` | Run evaluation with LLM judge scoring |

Full schema with request/response bodies: `http://localhost:8013/docs`

---

## Connecting with a Frontend

The backend has CORS enabled for all origins (`*`). For a React or Next.js frontend, point your API calls to the backend base URL.

**Local development**

```js
const API_BASE = "http://localhost:8013";
```

**Production (behind Nginx)**

The included `setup_ec2.sh` configures Nginx to proxy `/api/fastapi/` to the backend. Your frontend should use:

```js
const API_BASE = "/api/fastapi";    // relative — works on the same domain
```

Example fetch:

```js
// Upload a PDF
const formData = new FormData();
formData.append("file", file);
await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });

// Ask a question
const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: "What are the benefits of the PM-KISAN scheme?", top_k: 5 }),
});
const data = await res.json();
console.log(data.answer, data.sources);
```

---

## EC2 Deployment (Ubuntu 20.04 / 22.04, t2.micro)

### One-time setup

SSH into the instance, clone the repo, then run:

```bash
git clone <repo-url> && cd agrigpt-backend-rag-government-schemes
bash setup_ec2.sh
```

The script handles:
- Python version detection (installs 3.11 via deadsnakes PPA if system Python < 3.9)
- 2 GB swap file (required — BGE model uses ~450 MB RAM)
- CPU-only PyTorch install (saves ~877 MB vs CUDA build)
- Virtualenv + dependency install
- BGE model pre-download (avoids cold-start OOM)
- `systemd` service (`agrigpt-rag`) — starts on boot, restarts on failure
- Nginx reverse proxy on port 80 → backend on `127.0.0.1:8013`

After the script finishes, fill in your keys:

```bash
nano .env
sudo systemctl restart agrigpt-rag
```

Open inbound TCP **port 80** in the EC2 Security Group. The app will be at `http://<EC2_PUBLIC_IP>`.

### Service management

```bash
sudo systemctl status agrigpt-rag      # check status
sudo journalctl -u agrigpt-rag -f      # live logs
sudo systemctl restart agrigpt-rag     # restart after code change
sudo systemctl stop agrigpt-rag        # stop
```

### Deploying code updates (manual)

```bash
cd /home/ubuntu/agrigpt-backend-rag-government-schemes
git pull
source venv/bin/activate
pip install -r requirements-backend.txt --quiet
sudo systemctl restart agrigpt-rag
```

### CI/CD via GitHub Actions

`.github/workflows/deploy.yml` automates the steps above on every push to `main`.

Add these secrets in **GitHub → Settings → Secrets → Actions**:

| Secret | Value |
|---|---|
| `EC2_HOST` | EC2 public IP or hostname |
| `EC2_USER` | `ubuntu` |
| `EC2_SSH_KEY` | Full contents of your `.pem` private key |

Push to `main` → GitHub SSHs into EC2 → pulls code → restarts the service.

---

## Re-ingesting Documents After Config Changes

You only need to re-upload a file when the chunking parameters change (`CHUNK_SIZE` or `CHUNK_OVERLAP` in `shemes_rag.py`). New uploads always use the current settings automatically.

To re-ingest a specific file:
1. `DELETE /files/{filename}` — removes old embeddings
2. `POST /upload` — re-chunks and re-embeds with current settings

---

## Troubleshooting

See [DEPLOYMENT_ISSUES.md](DEPLOYMENT_ISSUES.md) for documented errors with root causes, before/after fixes, and performance impact numbers.
