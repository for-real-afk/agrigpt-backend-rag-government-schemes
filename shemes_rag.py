"""
Simple RAG API with Gemini and Pinecone
FastAPI application with file upload and query endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import json
import uuid
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
import google.genai as genai
from google.genai import types
from PyPDF2 import PdfReader
import docx
import io
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="Schemes RAG API with Gemini",
    description="Upload documents and query them using Gemini AI",
    version="1.0.0"
)

# CORS — tighten allow_origins to your frontend domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "schemes-rag"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Collect all available Gemini API keys (GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...)
_raw_keys = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
]
GEMINI_API_KEYS = [k for k in _raw_keys if k]

# Groq fallback (free tier: 14,400 req/day — set GROQ_API_KEY in .env)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GEMINI_API_KEYS and not GROQ_API_KEY:
    raise ValueError("Set at least one GEMINI_API_KEY or GROQ_API_KEY in .env")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")

GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else None

print(f"Pinecone API Key loaded: {PINECONE_API_KEY[:10]}...")
print(f"Gemini API Key loaded: {GEMINI_API_KEY[:10]}...")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Create or connect to index
def setup_index():
    """Setup Pinecone index"""
    try:
        existing_indexes = [idx['name'] for idx in pc.list_indexes()]
        print(f"Existing indexes: {existing_indexes}")
    except Exception as e:
        print(f"Error listing indexes: {e}")
        raise
    
    if INDEX_NAME not in existing_indexes:
        print(f"Creating new index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # all-MiniLM-L6-v2 dimension
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    else:
        print(f"Using existing index: {INDEX_NAME}")
    
    return pc.Index(INDEX_NAME)

index = setup_index()

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int

class DeleteFileResponse(BaseModel):
    message: str
    filename: str
    deleted_chunks: int

# ── Eval dataset models ────────────────────────────────────────────────────────

EVAL_STORE_PATH = os.path.join(os.path.dirname(__file__), "eval_store.json")

def load_eval_store() -> dict:
    if os.path.exists(EVAL_STORE_PATH):
        with open(EVAL_STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"silver": [], "golden": []}

def save_eval_store(store: dict):
    with open(EVAL_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)

def _parse_llm_json(text: str) -> dict:
    """Strip markdown fences then parse JSON."""
    raw = text.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())

class GenerateSilverRequest(BaseModel):
    filename: str
    num_questions_per_chunk: int = 2
    max_chunks: int = 10

class PromoteRequest(BaseModel):
    silver_id: str
    question: Optional[str] = ""
    expected_answer: Optional[str] = ""

class RunEvalRequest(BaseModel):
    dataset_type: str = "golden"   # "silver" | "golden" | "both"
    top_k: int = 3
    use_llm_judge: bool = True

# Helper functions
def extract_text_from_file(file: UploadFile) -> str:
    """Extract text from different file types"""
    content = file.file.read()
    
    if file.filename.endswith('.txt'):
        return content.decode('utf-8')
    
    elif file.filename.endswith('.pdf'):
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    elif file.filename.endswith('.docx'):
        doc = docx.Document(io.BytesIO(content))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .txt, .pdf, or .docx")

def chunk_text(text: str) -> List[str]:
    """Split text into chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    
    return chunks

def get_embedding(text: str) -> List[float]:
    """Generate embedding using BGE instruction tuning"""
    try:
        embedding = embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

def get_query_embedding(text: str) -> List[float]:
    """Generate embedding for query using BGE instruction tuning"""
    try:
        instruction = "Represent this sentence for searching relevant passages: "
        embedding = embedding_model.encode(instruction + text)
        # embedding = embedding_model.encode(text)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        raise

# API Endpoints
@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "RAG API with Gemini",
        "endpoints": {
            "docs": "/docs",
            "upload": "/upload",
            "query": "/query",
            "health": "/health"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "total_vectors": stats.total_vector_count
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a document
    
    - **file**: Upload a .txt, .pdf, or .docx file
    - Returns: Confirmation with number of chunks processed
    """
    try:
        # Extract text from file
        text = extract_text_from_file(file)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="File is empty or unreadable")
        
        # Chunk the text
        chunks = chunk_text(text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks created from file")
        
        print(f"Processing {len(chunks)} chunks from {file.filename}")
        
        # Create embeddings and store in Pinecone
        vectors = []
        for i, chunk in enumerate(chunks):
            print(f"Generating embedding for chunk {i+1}/{len(chunks)}")
            embedding = get_embedding(chunk)
            
            vector_id = f"{file.filename}_{i}"
            metadata = {
                'text': chunk,
                'filename': file.filename,
                'chunk_index': i
            }
            
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
            print(f"Uploaded batch {i//batch_size + 1}")
        
        return UploadResponse(
            message="File uploaded and processed successfully",
            filename=file.filename,
            chunks_added=len(chunks)
        )
    
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def decompose_query(question: str) -> List[str]:
    """
    Detect multi-entity queries (e.g. 'jso1 and jso2') and split into focused
    sub-queries so each entity gets its own retrieval pass.
    Returns a list of sub-queries, or [question] if no decomposition needed.
    """
    prompt = f"""Does this question ask about multiple distinct named items (crop varieties, schemes, models, locations, etc.) that should each be looked up separately?

Question: {question}

If YES, rewrite as individual focused sub-questions.
If NO, return the original question only.

Respond ONLY with valid JSON (no markdown):
{{"decompose": true/false, "sub_queries": ["sub-question 1", "sub-question 2"]}}"""

    def _call_llm(text: str) -> str:
        """Try Gemini keys then Groq; return raw text or raise."""
        for api_key in GEMINI_API_KEYS:
            try:
                _c = genai.Client(api_key=api_key)
                return _c.models.generate_content(
                    model=GEMINI_MODEL, contents=text,
                    config=types.GenerateContentConfig(temperature=0)
                ).text
            except Exception:
                continue
        if GROQ_API_KEY:
            from openai import OpenAI as _OAI
            groq = _OAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
            return groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": text}],
                temperature=0,
            ).choices[0].message.content
        raise RuntimeError("No LLM available")

    try:
        raw = _call_llm(prompt)
        parsed = _parse_llm_json(raw)
        if parsed.get("decompose") and len(parsed.get("sub_queries", [])) > 1:
            print(f"Query decomposed into {len(parsed['sub_queries'])} sub-queries: {parsed['sub_queries']}")
            return parsed["sub_queries"]
        return [question]
    except Exception as e:
        print(f"decompose_query failed ({e}), using original query")
        return [question]


def retrieve_chunks(sub_query: str, top_k: int) -> List[Dict]:
    """Run a single Pinecone retrieval for one sub-query."""
    emb = get_query_embedding(sub_query)
    results = index.query(vector=emb, top_k=top_k, include_metadata=True)
    return results.get("matches", [])


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document database

    - **question**: Your question
    - **top_k**: Number of relevant chunks to retrieve (default: 3)
    - Returns: AI-generated answer with sources
    """
    try:
        RELEVANCE_THRESHOLD = 0.48
        MIN_RELEVANT_CHUNKS = 2

        # ── Step 1: decompose multi-entity queries ────────────────────────────
        sub_queries = decompose_query(request.question)

        # ── Step 2: retrieve top_k chunks per sub-query, merge & deduplicate ──
        # For multi-entity queries retrieve extra chunks per sub-query so each
        # entity's specific data is more likely to appear in the context.
        chunks_per_sq = request.top_k + 5 if len(sub_queries) > 1 else request.top_k
        seen_ids: set = set()
        all_matches: List[Dict] = []

        for sq in sub_queries:
            for match in retrieve_chunks(sq, chunks_per_sq):
                if match["id"] not in seen_ids:
                    seen_ids.add(match["id"])
                    all_matches.append(match)

        if not all_matches:
            return QueryResponse(
                answer="No relevant information found in the database.",
                sources=[]
            )

        # Sort merged results by score descending
        all_matches.sort(key=lambda m: m["score"], reverse=True)

        # ── Step 3: apply relevance threshold ────────────────────────────────
        relevant_matches = [m for m in all_matches if m["score"] >= RELEVANCE_THRESHOLD]

        # For multi-entity queries lower the bar slightly: even 1 relevant chunk
        # per sub-query is useful, so require at least min(len(sub_queries), 2) chunks
        min_required = min(len(sub_queries), MIN_RELEVANT_CHUNKS)
        if len(relevant_matches) < min_required:
            return QueryResponse(
                answer="This specific information is not covered in the provided document.",
                sources=[]
            )

        # ── Step 4: build context ─────────────────────────────────────────────
        context_parts = []
        sources = []

        for i, match in enumerate(relevant_matches):
            context_parts.append(f"[{i+1}] {match['metadata']['text']}")
            sources.append({
                'chunk_id': match['id'],
                'filename': match['metadata']['filename'],
                'score': float(match['score']),
                'text': match['metadata']['text']
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using Gemini / Groq
        prompt = f"""You are an agricultural document assistant. Answer using ONLY the context provided below.

Guidelines:
- Read the full context before answering.
- If the answer is explicitly stated in the context, quote or paraphrase it directly — do not recalculate values that are already given.
- If the answer requires arithmetic (e.g. date ranges, totals), show your working briefly.
- For list questions, count every item present in the context.
- If the information is not in the context, say "This information is not covered in the provided documents."
- Be concise. Use bullet points where helpful.

Context:
{context}

Question: {request.question}

Answer:"""
        
        answer = None
        last_error = None

        # Try each Gemini key first
        for api_key in GEMINI_API_KEYS:
            try:
                _client = genai.Client(api_key=api_key)
                response = _client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
                answer = response.text
                print(f"Used Gemini key ...{api_key[-6:]}")
                break
            except Exception as e:
                print(f"Key ...{api_key[-6:]} failed: {e}")
                last_error = e

        # Fallback to Groq if all Gemini keys failed
        if answer is None and GROQ_API_KEY:
            try:
                groq_client = OpenAI(
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1"
                )
                groq_response = groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                answer = groq_response.choices[0].message.content
                print(f"Used Groq fallback ({GROQ_MODEL})")
            except Exception as e:
                print(f"Groq fallback failed: {e}")
                last_error = e

        if answer is None:
            raise last_error
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/files")
async def list_files():
    """
    List all uploaded files

    - Returns: List of unique filenames stored in the database
    """
    try:
        stats = index.describe_index_stats()
        total = stats.total_vector_count

        if total == 0:
            return {"files": [], "total_vectors": 0}

        dummy_vector = [0.0] * 768
        results = index.query(
            vector=dummy_vector,
            top_k=min(10000, total),
            include_metadata=True
        )

        filenames = sorted(set(
            match['metadata']['filename']
            for match in results['matches']
            if match.get('metadata', {}).get('filename')
        ))

        return {"files": filenames, "total_vectors": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.delete("/files/{filename:path}", response_model=DeleteFileResponse)
async def delete_file(filename: str):
    """
    Delete a specific file and all its embeddings

    - **filename**: Name of the file to delete
    - Returns: Confirmation with number of chunks deleted
    """
    try:
        # Zero vector breaks cosine similarity with metadata filters; use a real embedding.
        query_emb = get_embedding("document text content")
        results = index.query(
            vector=query_emb,
            filter={"filename": {"$eq": filename}},
            top_k=10000,
            include_metadata=False
        )

        ids_to_delete = [match['id'] for match in results['matches']]

        if not ids_to_delete:
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found in database")

        batch_size = 1000
        for i in range(0, len(ids_to_delete), batch_size):
            index.delete(ids=ids_to_delete[i:i + batch_size])

        return DeleteFileResponse(
            message=f"File '{filename}' deleted successfully",
            filename=filename,
            deleted_chunks=len(ids_to_delete)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.delete("/clear")
async def clear_database():
    """
    Clear all vectors from the database
    
    - Returns: Confirmation message
    """
    try:
        stats = index.describe_index_stats()
        if stats.total_vector_count == 0:
            return {"message": "Database already empty"}
        index.delete(delete_all=True)
        return {"message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.get("/stats")
async def get_stats():
    """
    Get database statistics
    
    - Returns: Current database stats
    """
    try:
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": stats.index_fullness
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

# ── Eval endpoints ─────────────────────────────────────────────────────────────

@app.post("/eval/generate-silver")
async def generate_silver(request: GenerateSilverRequest):
    """
    Auto-generate Q&A pairs (silver dataset) from a file's chunks using Gemini.
    """
    # A zero vector has undefined cosine similarity and returns empty results
    # when combined with a metadata filter — use a real embedding instead.
    query_emb = get_embedding("main topics key findings important information")
    results = index.query(
        vector=query_emb,
        filter={"filename": {"$eq": request.filename}},
        top_k=request.max_chunks,
        include_metadata=True
    )

    if not results["matches"]:
        raise HTTPException(status_code=404, detail=f"No chunks found for '{request.filename}'")

    store = load_eval_store()
    new_pairs = []

    for match in results["matches"]:
        chunk_text = match["metadata"]["text"]

        prompt = f"""Generate {request.num_questions_per_chunk} factual question-answer pairs from the text below.
Rules:
- Each question must be answerable directly from the text
- Prefer specific, verifiable answers (numbers, names, percentages, model names)
- Return ONLY a valid JSON array, no extra text

Format: [{{"question": "...", "answer": "..."}}]

Text:
{chunk_text}"""

        try:
            resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            qa_list = _parse_llm_json(resp.text)
            for qa in qa_list:
                if not qa.get("question") or not qa.get("answer"):
                    continue
                pair = {
                    "id": f"s_{uuid.uuid4().hex[:8]}",
                    "question": qa["question"].strip(),
                    "expected_answer": qa["answer"].strip(),
                    "source_file": request.filename,
                    "chunk_text": chunk_text[:600],
                    "pair_type": "silver",
                    "created_at": datetime.utcnow().isoformat(),
                }
                store["silver"].append(pair)
                new_pairs.append(pair)
        except Exception as e:
            print(f"Silver gen error for chunk {match['id']}: {e}")
            continue

    save_eval_store(store)
    return {"generated": len(new_pairs), "pairs": new_pairs}


@app.get("/eval/pairs")
async def get_eval_pairs(pair_type: str = Query("all", enum=["silver", "golden", "all"])):
    """List eval pairs filtered by type."""
    store = load_eval_store()
    if pair_type == "silver":
        return {"pairs": store["silver"]}
    if pair_type == "golden":
        return {"pairs": store["golden"]}
    return {"pairs": store["silver"] + store["golden"]}


@app.post("/eval/promote")
async def promote_to_golden(request: PromoteRequest):
    """Promote a silver pair to the golden dataset (optionally editing Q/A)."""
    store = load_eval_store()
    silver = next((p for p in store["silver"] if p["id"] == request.silver_id), None)
    if not silver:
        raise HTTPException(status_code=404, detail=f"Silver pair '{request.silver_id}' not found")

    golden = {
        "id": f"g_{uuid.uuid4().hex[:8]}",
        "question": (request.question or silver["question"]).strip(),
        "expected_answer": (request.expected_answer or silver["expected_answer"]).strip(),
        "source_file": silver["source_file"],
        "chunk_text": silver.get("chunk_text", ""),
        "pair_type": "golden",
        "promoted_from": request.silver_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    store["golden"].append(golden)
    store["silver"] = [p for p in store["silver"] if p["id"] != request.silver_id]
    save_eval_store(store)
    return {"message": "Promoted to golden", "pair": golden}


@app.delete("/eval/pairs/{pair_id}")
async def delete_eval_pair(pair_id: str):
    """Delete a silver or golden pair by ID."""
    store = load_eval_store()
    before = len(store["silver"]) + len(store["golden"])
    store["silver"] = [p for p in store["silver"] if p["id"] != pair_id]
    store["golden"] = [p for p in store["golden"] if p["id"] != pair_id]
    if len(store["silver"]) + len(store["golden"]) == before:
        raise HTTPException(status_code=404, detail=f"Pair '{pair_id}' not found")
    save_eval_store(store)
    return {"message": f"Deleted '{pair_id}'"}


@app.post("/eval/run")
async def run_evaluation(request: RunEvalRequest):
    """
    Run evaluation over silver / golden / both datasets.
    Uses Gemini as an LLM judge to score each answer 0.0–1.0.
    """
    store = load_eval_store()
    if request.dataset_type == "silver":
        pairs = store["silver"]
    elif request.dataset_type == "golden":
        pairs = store["golden"]
    else:
        pairs = store["silver"] + store["golden"]

    if not pairs:
        raise HTTPException(status_code=400, detail=f"No '{request.dataset_type}' pairs found. Generate or promote some first.")

    results = []

    for pair in pairs:
        # ── RAG retrieval ──────────────────────────────────────────────────
        q_emb = get_query_embedding(pair["question"])
        matches = index.query(vector=q_emb, top_k=request.top_k, include_metadata=True)

        context_parts = []
        sources = []
        for i, m in enumerate(matches["matches"]):
            context_parts.append(f"[{i+1}] {m['metadata']['text']}")
            sources.append({"filename": m["metadata"]["filename"], "score": float(m["score"])})

        context = "\n\n".join(context_parts)

        rag_prompt = f"""Answer the question using only the context below. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {pair['question']}
Answer:"""

        rag_resp = client.models.generate_content(model="gemini-2.5-flash", contents=rag_prompt)
        rag_answer = rag_resp.text.strip()

        # ── LLM judge ─────────────────────────────────────────────────────
        score, passed, reasoning = 0.0, False, ""

        if request.use_llm_judge:
            judge_prompt = f"""You are an impartial evaluation judge for a RAG system.

Question: {pair['question']}
Expected Answer: {pair['expected_answer']}
RAG Answer: {rag_answer}

Score the RAG answer from 0.0 to 1.0:
- 1.0  All key facts from expected answer are present and correct
- 0.7  Most key facts present, minor gaps
- 0.4  Some relevant info but missing important facts
- 0.0  Wrong, hallucinated, or irrelevant

Respond ONLY with valid JSON (no markdown):
{{"score": 0.0, "passed": false, "reasoning": "one sentence"}}"""

            try:
                j_resp = client.models.generate_content(model="gemini-2.5-flash", contents=judge_prompt)
                j = _parse_llm_json(j_resp.text)
                score = float(j.get("score", 0.0))
                passed = bool(j.get("passed", score >= 0.5))
                reasoning = j.get("reasoning", "")
            except Exception as e:
                reasoning = f"Judge parse error: {e}"
        else:
            # keyword fallback
            key_words = pair["expected_answer"].lower().split()[:6]
            matches_kw = sum(1 for w in key_words if w in rag_answer.lower())
            score = round(matches_kw / max(len(key_words), 1), 2)
            passed = score >= 0.5
            reasoning = f"keyword match {matches_kw}/{len(key_words)}"

        results.append({
            "pair_id": pair["id"],
            "pair_type": pair["pair_type"],
            "source_file": pair.get("source_file", ""),
            "question": pair["question"],
            "expected_answer": pair["expected_answer"],
            "rag_answer": rag_answer,
            "sources": sources,
            "score": round(score, 3),
            "passed": passed,
            "reasoning": reasoning,
        })

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    avg_score = round(sum(r["score"] for r in results) / total, 3) if total else 0.0

    return {
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "avg_score": avg_score,
            "pass_rate": round(passed_count / total * 100, 1) if total else 0.0,
        },
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8013)