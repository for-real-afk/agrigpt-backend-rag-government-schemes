# AgriGPT RAG — Deployment Issues & Resolutions

Platform: EC2 t2.micro (1 vCPU, 1 GB RAM) · Ubuntu 22.04 / 20.04  
Stack: FastAPI · Uvicorn · Pinecone (serverless, cosine) · Gemini 2.5 Flash · BGE-base-en-v1.5 · Streamlit

---

## Issue 1 — Streamlit sidebar crash on every page load

### Error
```
requests.exceptions.ReadTimeout
File "streamlit_app.py", line 61, in <module>
    resp = api_get("/stats", timeout=10)
```

### Root Cause
`/stats` calls Pinecone's `describe_index_stats()` on every Streamlit re-render (which happens
on every user interaction). Cold Pinecone connections regularly exceed 10 s, triggering the
timeout. The exception was not caught, crashing the entire app.

### Changes Made

**Before — `streamlit_app.py`**
```python
def api_get(path, timeout=10):
    try:
        return requests.get(f"{API_URL}{path}", timeout=timeout)
    except requests.exceptions.ConnectionError:
        return None

# Called unconditionally at module level on every render:
resp = api_get("/stats")
st.metric("Total Vectors", resp.json()["total_vectors"])
```

**After — `streamlit_app.py`**
```python
def api_get(path, timeout=30):                          # timeout raised: 10s → 30s
    try:
        return requests.get(f"{API_URL}{path}", timeout=timeout)
    except (requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout):            # ReadTimeout now caught
        return None

# Stats fetched only when user clicks the button (lazy load):
if st.button("Refresh Stats"):
    resp = api_get("/stats", timeout=30)
    ...
```

### Impact
| Metric | Before | After |
|---|---|---|
| App crash on page load | Every load | Never |
| Stats fetch timeout | 10 s (too low) | 30 s |
| Pinecone calls per render | 1 (unconditional) | 0 (button-triggered) |

---

## Issue 2 — `generate-silver` and `delete_file` returning 404 for valid files

### Error
```
HTTP 404: No chunks found for '3.pdf'
```
Files 2.pdf, 3.pdf, 4.pdf all failed. The same files appeared correctly in `/files`.

### Root Cause
Pinecone uses **cosine similarity** as its distance metric. Cosine similarity is defined as
`dot(a, b) / (|a| * |b|)`. When the query vector is all zeros (`[0.0] * 768`), the denominator
`|a|` is zero → division by zero → undefined similarity for every candidate vector.

Without a metadata filter, Pinecone still returns all vectors (falls back to an arbitrary
ordering). With a metadata filter applied, the scoring step returns no results, producing an
empty `matches` list.

### Changes Made

**Before — `shemes_rag.py`**
```python
# generate_silver and delete_file both used:
dummy_vector = [0.0] * 768
results = index.query(
    vector=dummy_vector,
    filter={"filename": {"$eq": filename}},
    top_k=request.max_chunks,
    include_metadata=True
)
```

**After — `shemes_rag.py`**
```python
# generate_silver — semantically meaningful query finds best chunks:
query_emb = get_embedding("main topics key findings important information")
results = index.query(
    vector=query_emb,
    filter={"filename": {"$eq": filename}},
    top_k=request.max_chunks,
    include_metadata=True
)

# delete_file — any real embedding works; top_k=10000 with filter returns all:
query_emb = get_embedding("document text content")
results = index.query(
    vector=query_emb,
    filter={"filename": {"$eq": filename}},
    top_k=10000,
    include_metadata=False
)
```

### Impact
| Endpoint | Before | After |
|---|---|---|
| `POST /eval/generate-silver` | 404 for all files | Correctly returns chunks |
| `DELETE /files/{filename}` | 404 / 0 deletions | Deletes all chunks |
| Extra embedding call per request | 0 | 1 (~15–30 ms on CPU) |

---

## Issue 3 — Streamlit crash on non-JSON error responses

### Error
```
File "streamlit_app.py", line 426, in <module>
    show_error(f"Eval failed: {resp.json().get('detail', resp.text)}")
requests.exceptions.JSONDecodeError: ...
```

### Root Cause
When the backend returned a 500 with a plain-text Python traceback (not JSON), calling
`resp.json()` raised `JSONDecodeError`. All error display paths used this pattern, so any
backend exception crashed the Streamlit frontend too.

### Changes Made

**Before — `streamlit_app.py`**
```python
show_error(f"Eval failed: {resp.json().get('detail', resp.text)}")
# Repeated at every error path — 6 locations
```

**After — `streamlit_app.py`**
```python
def resp_detail(resp) -> str:
    """Return error detail, safely handling non-JSON bodies."""
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text

show_error(f"Eval failed: {resp_detail(resp)}")
# All 6 error paths replaced with resp_detail(resp)
```

### Impact
| Scenario | Before | After |
|---|---|---|
| Backend returns JSON error | Works | Works |
| Backend returns plain-text traceback | Streamlit crashes | Error shown in UI |
| Backend returns HTML (e.g. gateway 502) | Streamlit crashes | Error shown in UI |

---

## Issue 4 — `python3.11` not found on EC2 (Ubuntu 20.04)

### Error
```
E: Unable to locate package python3.11
E: Couldn't find any package by glob 'python3.11'
E: Unable to locate package python3.11-venv
```

### Root Cause
`setup_ec2.sh` hardcoded `python3.11`. Ubuntu 20.04 (Focal) ships Python 3.8 by default and
does not include Python 3.11 in its standard apt repositories. Python 3.11 requires the
third-party `deadsnakes` PPA on that OS version.

### Changes Made

**Before — `setup_ec2.sh`**
```bash
sudo apt-get install -y python3.11 python3.11-venv python3-pip git
python3.11 -m venv venv
```

**After — `setup_ec2.sh`**
```bash
PYTHON_BIN=$(which python3)
PY_VERSION=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MINOR" -lt 9 ]; then
    # Ubuntu 20.04 path — add deadsnakes PPA for 3.11
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -y
    sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils
    PYTHON_BIN=$(which python3.11)
else
    # Ubuntu 22.04+ already has Python 3.10+
    sudo apt-get install -y python3-venv python3-pip
fi

$PYTHON_BIN -m venv venv
```

### Impact
| Ubuntu Version | Before | After |
|---|---|---|
| 20.04 (Python 3.8) | Setup fails immediately | Installs 3.11 via PPA |
| 22.04 (Python 3.10) | Unnecessary failure | Uses system Python directly |

---

## Issue 5 — `ModuleNotFoundError: No module named 'fastapi'`

### Error
```
File "/home/ubuntu/agrigpt-backend-rag-government-schemes/shemes_rag.py", line 6, in <module>
    from fastapi import FastAPI, ...
ModuleNotFoundError: No module named 'fastapi'
```
`which uvicorn` returned `/usr/bin/uvicorn` even with venv activated.

### Root Cause
The venv was created but `pip install -r requirements-backend.txt` had previously failed
(due to disk space — Issue 6 below). The venv therefore had no packages, including no
`uvicorn`. When `uvicorn` was typed in the shell, PATH resolved to `/usr/bin/uvicorn`
(the system package) which runs under the system Python, not the venv Python. The system
Python has no knowledge of the venv's site-packages.

### Fix
```bash
# Wrong — resolves to /usr/bin/uvicorn regardless of venv:
uvicorn shemes_rag:app ...

# Correct — always uses the active Python's uvicorn:
python -m uvicorn shemes_rag:app --host 0.0.0.0 --port 8013 --workers 1
```
`python -m uvicorn` bypasses PATH and uses the module installed in the currently active
Python interpreter, making it venv-safe even if the venv's `bin/` directory is shadowed.

### Impact
| Command | Python used | fastapi visible |
|---|---|---|
| `uvicorn` | `/usr/bin/python3` (system) | No |
| `python -m uvicorn` | `venv/bin/python` (venv) | Yes |

---

## Issue 6 — `[Errno 28] No space left on device` during pip install

### Error
```
Downloading nvidia_cusparselt_cu13-0.8.0...  167.8/169.9 MB
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

### Root Cause
`sentence-transformers` depends on `torch`. Without explicit constraints, pip selects the
CUDA-enabled build from PyPI — even on a GPU-less machine like t2.micro.

Packages downloaded before failure:
| Package | Size |
|---|---|
| `torch-2.11.0` (CUDA) | 530.7 MB |
| `nvidia_cudnn_cu13-9.19.0.56` | 366.1 MB |
| `nvidia_cusparselt_cu13-0.8.0` | 169.9 MB |
| **Total** | **1,066.7 MB** |

t2.micro default root EBS volume: **8 GB**. After OS (~2.5 GB) and existing packages, there
was insufficient space for the CUDA packages.

### Fix

**Step 1 — free space**
```bash
pip cache purge          # removes pip's download cache
sudo apt-get clean       # removes apt package cache
```

**Step 2 — install CPU-only torch first**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-backend.txt
```
By pre-installing the CPU build, pip satisfies the `torch` dependency without downloading
GPU packages when it processes `sentence-transformers`.

**Step 3 — persisted in `setup_ec2.sh`**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-backend.txt
```

### Impact
| Metric | CUDA build | CPU build | Saving |
|---|---|---|---|
| torch package size | 530.7 MB | ~190 MB | ~341 MB |
| CUDA companion libs | ~536 MB | 0 MB | ~536 MB |
| **Total install size** | **~1,067 MB** | **~190 MB** | **~877 MB (82%)** |
| Inference speed on t2.micro | Same (no GPU either way) | Same | — |

---

## Issue 7 — `[Errno 98] Address already in use` on port 8010

### Error
```
ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 8010): address already in use
```

### Root Cause
Diagnosis with `sudo ss -tlnp | grep 8010`:
```
LISTEN  docker-proxy  pid=2430702  0.0.0.0:8010
LISTEN  uvicorn       pid=1624616  0.0.0.0:8011
LISTEN  uvicorn       pid=1829415  0.0.0.0:8012
```
Port 8010 was bound by a Docker container (`docker-proxy`). Ports 8011 and 8012 were
occupied by pre-existing uvicorn processes from earlier deployment attempts.

### Fix
Used port **8013** (next available). Updated the systemd service file accordingly:
```bash
ExecStart=.../uvicorn shemes_rag:app --host 0.0.0.0 --port 8013 --workers 1
```

---

## Issue 8 — `ERR_CONNECTION_TIMED_OUT` from browser

### Error
```
This site can't be reached — 13.205.59.184 took too long to respond.
ERR_CONNECTION_TIMED_OUT
```

### Root Cause
EC2 Security Group had no inbound rule for TCP port 8013. All packets to that port were
silently dropped at the AWS network layer, appearing as a timeout to the client.

### Fix
AWS Console → EC2 → Security Groups → Edit inbound rules → Add rule:

| Type | Protocol | Port | Source |
|---|---|---|---|
| Custom TCP | TCP | 8013 | 0.0.0.0/0 |

### Impact
| State | External access |
|---|---|
| Before (no inbound rule) | Timeout — all connections dropped |
| After (rule added) | API and `/docs` accessible from internet |

---

## Issue 9 — Setup script prints `401 - Unauthorized` as the API URL

### Error
```
API docs: http://<?xml version="1.0"...><title>401 - Unauthorized</title>...:8010/docs
```

### Root Cause
`setup_ec2.sh` used IMDSv1 to fetch the instance's public IP:
```bash
curl -s http://169.254.169.254/latest/meta-data/public-ipv4
```
Newer EC2 instances enforce **IMDSv2** (token-required), so the unauthenticated IMDSv1 call
returns HTTP 401 HTML. The script interpolated that HTML into the final print statement.

### Impact
Cosmetic only — the service itself installed and started correctly. The printed URL was
incorrect but no functionality was affected.

### Fix (workaround)
Replace the metadata curl with an IMDSv2-aware call, or simply hard-code the known IP:
```bash
# IMDSv2 compatible
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
PUBLIC_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/public-ipv4)
echo "API docs: http://$PUBLIC_IP:8013/docs"
```

---

## Summary Table

| # | Issue | Severity | Root Cause | Fixed In |
|---|---|---|---|---|
| 1 | Streamlit ReadTimeout crash | High | Slow Pinecone call + missing exception handler | `streamlit_app.py` |
| 2 | 404 on generate-silver / delete | High | Zero vector undefined with cosine + filter | `shemes_rag.py` |
| 3 | JSONDecodeError on error paths | Medium | Non-JSON backend errors not handled | `streamlit_app.py` |
| 4 | python3.11 not found | High | Ubuntu 20.04 missing from default apt | `setup_ec2.sh` |
| 5 | ModuleNotFoundError: fastapi | High | System uvicorn shadowing venv uvicorn | Run command |
| 6 | No space left on device | Critical | CUDA torch (~1.07 GB) on 8 GB disk | `setup_ec2.sh` + install order |
| 7 | Address already in use | Medium | Docker + stale uvicorn on ports 8010–8012 | Port changed to 8013 |
| 8 | ERR_CONNECTION_TIMED_OUT | High | Security Group missing inbound rule for 8013 | AWS Console |
| 9 | 401 in setup script output | Low | IMDSv2 not supported by plain curl | Cosmetic — no fix required |
