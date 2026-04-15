#!/bin/bash
# Smoke-test all RAG endpoints
# Usage: bash test_endpoints.sh [host]
# Example: bash test_endpoints.sh http://13.205.59.184:8013

BASE=${1:-http://localhost:8013}
PASS=0; FAIL=0

check() {
    local label=$1; local expected=$2; local actual=$3
    if echo "$actual" | grep -q "$expected"; then
        echo "  PASS  $label"
        ((PASS++))
    else
        echo "  FAIL  $label"
        echo "        Expected: $expected"
        echo "        Got:      $(echo $actual | head -c 200)"
        ((FAIL++))
    fi
}

echo ""
echo "========================================"
echo " AgriGPT RAG — Endpoint Tests"
echo " Target: $BASE"
echo "========================================"

# ── Core ──────────────────────────────────────────────────────────────────────
echo ""
echo "[ Core ]"

R=$(curl -s $BASE/)
check "GET  /" "RAG API" "$R"

R=$(curl -s $BASE/health)
check "GET  /health" "pinecone_connected" "$R"

R=$(curl -s $BASE/stats)
check "GET  /stats" "total_vectors" "$R"

# ── Documents ─────────────────────────────────────────────────────────────────
echo ""
echo "[ Documents ]"

R=$(curl -s $BASE/files)
check "GET  /files" "files" "$R"

# Upload a tiny test file
echo "This is a test document about agricultural schemes in India." > /tmp/test_doc.txt
R=$(curl -s -X POST $BASE/upload -F "file=@/tmp/test_doc.txt")
check "POST /upload" "chunks_added" "$R"

# List files again — test_doc.txt should appear
R=$(curl -s $BASE/files)
check "GET  /files (after upload)" "test_doc.txt" "$R"

# ── Query ─────────────────────────────────────────────────────────────────────
echo ""
echo "[ Query ]"

R=$(curl -s -X POST $BASE/query \
    -H "Content-Type: application/json" \
    -d '{"question":"What is this document about?","top_k":2}')
check "POST /query" "answer" "$R"

# ── Eval ──────────────────────────────────────────────────────────────────────
echo ""
echo "[ Eval ]"

R=$(curl -s "$BASE/eval/pairs?pair_type=all")
check "GET  /eval/pairs?pair_type=all" "pairs" "$R"

R=$(curl -s "$BASE/eval/pairs?pair_type=silver")
check "GET  /eval/pairs?pair_type=silver" "pairs" "$R"

R=$(curl -s "$BASE/eval/pairs?pair_type=golden")
check "GET  /eval/pairs?pair_type=golden" "pairs" "$R"

R=$(curl -s -X POST $BASE/eval/generate-silver \
    -H "Content-Type: application/json" \
    -d '{"filename":"test_doc.txt","num_questions_per_chunk":1,"max_chunks":2}')
check "POST /eval/generate-silver" "generated" "$R"

# Grab the first silver pair id for promote/delete tests
SILVER_ID=$(echo $R | python3 -c "import sys,json; pairs=json.load(sys.stdin).get('pairs',[]); print(pairs[0]['id'] if pairs else '')" 2>/dev/null)

if [ -n "$SILVER_ID" ]; then
    R=$(curl -s -X POST $BASE/eval/promote \
        -H "Content-Type: application/json" \
        -d "{\"silver_id\":\"$SILVER_ID\"}")
    check "POST /eval/promote" "Promoted" "$R"

    GOLDEN_ID=$(echo $R | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pair',{}).get('id',''))" 2>/dev/null)

    if [ -n "$GOLDEN_ID" ]; then
        R=$(curl -s -X POST $BASE/eval/run \
            -H "Content-Type: application/json" \
            -d '{"dataset_type":"golden","top_k":2,"use_llm_judge":false}')
        check "POST /eval/run" "summary" "$R"

        R=$(curl -s -X DELETE $BASE/eval/pairs/$GOLDEN_ID)
        check "DELETE /eval/pairs/{id}" "Deleted" "$R"
    fi
else
    echo "  SKIP  POST /eval/promote  (no silver pairs generated)"
    echo "  SKIP  POST /eval/run"
    echo "  SKIP  DELETE /eval/pairs/{id}"
fi

# ── Cleanup: delete test file ──────────────────────────────────────────────────
echo ""
echo "[ Cleanup ]"
R=$(curl -s -X DELETE "$BASE/files/test_doc.txt")
check "DELETE /files/{filename}" "deleted" "$R"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Results: $PASS passed, $FAIL failed"
echo "========================================"
echo ""
