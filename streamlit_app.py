"""
Streamlit frontend for Schemes RAG API — Silver / Golden eval workflow
"""

import streamlit as st
import requests
from urllib.parse import quote

API_URL = "http://localhost:8010"

st.set_page_config(
    page_title="Agriculture RAG Eval",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Agriculture Schemes RAG")
st.caption("Upload research PDFs · Build silver/golden eval datasets · Validate with LLM judge")


# ── Helpers ───────────────────────────────────────────────────────────────────

def api_get(path, timeout=30):
    try:
        return requests.get(f"{API_URL}{path}", timeout=timeout)
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        return None


def api_post(path, json=None, files=None, timeout=120):
    try:
        return requests.post(f"{API_URL}{path}", json=json, files=files, timeout=timeout)
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        return None


def api_delete(path, timeout=30):
    try:
        return requests.delete(f"{API_URL}{path}", timeout=timeout)
    except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
        return None


def show_error(msg):
    st.error(msg)


def resp_detail(resp) -> str:
    """Return the error detail from a response, safely handling non-JSON bodies."""
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text


def score_color(score: float) -> str:
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "orange"
    return "red"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Database Stats")
    if st.button("Refresh Stats", use_container_width=True):
        with st.spinner("Fetching…"):
            resp = api_get("/stats", timeout=30)
        if resp is None:
            st.warning("Unavailable (timeout / not running).")
        elif resp.status_code == 200:
            d = resp.json()
            st.metric("Total Vectors", d.get("total_vectors", 0))
            st.metric("Index Fullness", f"{d.get('index_fullness', 0):.2%}")
        else:
            st.warning("Could not fetch stats.")
    else:
        st.caption("Click to load stats.")

    st.divider()

    # Eval dataset counts
    resp_s = api_get("/eval/pairs?pair_type=silver", timeout=15)
    resp_g = api_get("/eval/pairs?pair_type=golden", timeout=15)
    n_silver = len(resp_s.json().get("pairs", [])) if resp_s and resp_s.status_code == 200 else "?"
    n_golden = len(resp_g.json().get("pairs", [])) if resp_g and resp_g.status_code == 200 else "?"
    st.metric("Silver pairs", n_silver)
    st.metric("Golden pairs", n_golden)
    st.divider()
    st.caption("Backend: " + API_URL)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_manage, tab_query, tab_dataset, tab_eval = st.tabs([
    "📤 Upload",
    "📁 Manage PDFs",
    "💬 Query",
    "🗂 Dataset",
    "🧪 Eval",
])


# ── Upload ────────────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload a Document")
    st.write("Supported: `.pdf` · `.txt` · `.docx`")

    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx"],
                                     label_visibility="collapsed")
    if uploaded_file:
        st.info(f"Selected: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
        if st.button("Upload & Process", type="primary"):
            with st.spinner(f"Processing {uploaded_file.name}…"):
                resp = api_post(
                    "/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")},
                    timeout=120,
                )
            if resp is None:
                show_error("Connection failed.")
            elif resp.status_code == 200:
                d = resp.json()
                st.success(f"Uploaded **{d['filename']}** — {d['chunks_added']} chunks added.")
                st.rerun()
            else:
                show_error(f"Upload failed: {resp_detail(resp)}")


# ── Manage PDFs ───────────────────────────────────────────────────────────────
with tab_manage:
    st.subheader("Uploaded Documents")
    if st.button("Refresh", key="refresh_files"):
        st.rerun()

    resp = api_get("/files")
    if resp is None:
        show_error("Cannot reach backend.")
    elif resp.status_code == 200:
        files = resp.json().get("files", [])
        total_v = resp.json().get("total_vectors", 0)
        if not files:
            st.info("No documents yet. Upload some PDFs first.")
        else:
            st.write(f"**{len(files)} file(s)** — {total_v:,} vectors")
            st.divider()
            for fname in files:
                c1, c2 = st.columns([5, 1])
                c1.write(f"📄 {fname}")
                if c2.button("Delete", key=f"del_{fname}"):
                    dr = api_delete(f"/files/{quote(fname, safe='')}", timeout=30)
                    if dr and dr.status_code == 200:
                        st.success(f"Deleted {fname} — {dr.json()['deleted_chunks']} chunks removed.")
                        st.rerun()
                    else:
                        show_error(f"Delete failed: {dr.text if dr else 'no response'}")


# ── Query ─────────────────────────────────────────────────────────────────────
with tab_query:
    st.subheader("Ask a Question")
    question = st.text_area("Question", placeholder="e.g. Which model forecasts maize prices best?",
                            height=80)
    c1, c2 = st.columns([3, 1])
    with c1:
        top_k = st.slider("Sources to retrieve", 1, 10, 3)
    with c2:
        st.write(""); st.write("")
        ask = st.button("Ask", type="primary", use_container_width=True)

    if ask:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching…"):
                resp = api_post("/query", json={"question": question.strip(), "top_k": top_k})
            if resp is None:
                show_error("Connection failed.")
            elif resp.status_code == 200:
                d = resp.json()
                st.divider()
                st.markdown("### Answer")
                st.write(d["answer"])
                if d.get("sources"):
                    with st.expander(f"Sources ({len(d['sources'])})"):
                        for i, s in enumerate(d["sources"], 1):
                            st.markdown(f"**[{i}] {s['filename']}** — score `{s['score']:.3f}`")
                            st.caption(s["text"])
                            if i < len(d["sources"]):
                                st.divider()
            else:
                show_error(f"Query error: {resp_detail(resp)}")


# ── Dataset tab ───────────────────────────────────────────────────────────────
with tab_dataset:
    sec = st.radio("Section", ["⚗️ Generate Silver", "📋 Review Silver", "🏅 Golden Dataset"],
                   horizontal=True)
    st.divider()

    # ── Generate Silver ───────────────────────────────────────────────────────
    if sec == "⚗️ Generate Silver":
        st.subheader("Generate Silver Q&A from a PDF")
        st.write(
            "Gemini reads chunks from the selected file and generates factual Q&A pairs automatically. "
            "These become the **silver dataset** — noisy but fast."
        )

        resp = api_get("/files")
        files = resp.json().get("files", []) if resp and resp.status_code == 200 else []

        if not files:
            st.warning("No files uploaded yet. Go to Upload tab first.")
        else:
            sel_file = st.selectbox("File to generate from", files)
            c1, c2 = st.columns(2)
            with c1:
                n_q = st.slider("Questions per chunk", 1, 4, 2)
            with c2:
                max_chunks = st.slider("Max chunks to use", 3, 20, 8)

            if st.button("Generate Silver Pairs", type="primary"):
                with st.spinner(f"Generating from {sel_file}… (this may take ~30s)"):
                    resp = api_post("/eval/generate-silver", json={
                        "filename": sel_file,
                        "num_questions_per_chunk": n_q,
                        "max_chunks": max_chunks,
                    }, timeout=180)

                if resp is None:
                    show_error("Connection failed.")
                elif resp.status_code == 200:
                    d = resp.json()
                    st.success(f"Generated **{d['generated']}** silver pairs from {sel_file}.")
                    for p in d["pairs"]:
                        with st.expander(f"[{p['id']}] {p['question'][:80]}"):
                            st.write(f"**A:** {p['expected_answer']}")
                            st.caption(f"Chunk: {p['chunk_text'][:200]}…")
                else:
                    show_error(f"Generation failed: {resp_detail(resp)}")

    # ── Review Silver ─────────────────────────────────────────────────────────
    elif sec == "📋 Review Silver":
        st.subheader("Review Silver Pairs")
        st.write("Approve good pairs → **promote to Golden**. Discard bad ones → **delete**.")

        resp = api_get("/eval/pairs?pair_type=silver")
        if resp is None:
            show_error("Cannot reach backend.")
        elif resp.status_code == 200:
            pairs = resp.json().get("pairs", [])
            if not pairs:
                st.info("No silver pairs yet. Generate some above.")
            else:
                st.write(f"**{len(pairs)} silver pair(s)**")
                for p in pairs:
                    with st.expander(f"[{p['id']}] {p['question'][:90]}"):
                        st.caption(f"Source: **{p['source_file']}**")
                        new_q = st.text_input("Question", value=p["question"], key=f"q_{p['id']}")
                        new_a = st.text_area("Expected Answer", value=p["expected_answer"],
                                             key=f"a_{p['id']}", height=80)
                        st.caption(f"Chunk preview: {p.get('chunk_text', '')[:200]}…")

                        c1, c2 = st.columns(2)
                        if c1.button("Promote to Golden", key=f"prom_{p['id']}", type="primary"):
                            r = api_post("/eval/promote", json={
                                "silver_id": p["id"],
                                "question": new_q,
                                "expected_answer": new_a,
                            })
                            if r and r.status_code == 200:
                                st.success("Promoted to golden!")
                                st.rerun()
                            else:
                                show_error(f"Promote failed: {r.text if r else 'no response'}")

                        if c2.button("Delete", key=f"delp_{p['id']}"):
                            r = api_delete(f"/eval/pairs/{p['id']}")
                            if r and r.status_code == 200:
                                st.rerun()
                            else:
                                show_error("Delete failed.")

    # ── Golden Dataset ────────────────────────────────────────────────────────
    elif sec == "🏅 Golden Dataset":
        st.subheader("Golden Dataset")
        st.write("These are your ground-truth Q&A pairs used for evaluation.")

        resp = api_get("/eval/pairs?pair_type=golden")
        if resp is None:
            show_error("Cannot reach backend.")
        elif resp.status_code == 200:
            pairs = resp.json().get("pairs", [])
            if not pairs:
                st.info("No golden pairs yet. Promote some from the Silver review tab.")
            else:
                # Group by source file
                by_file: dict = {}
                for p in pairs:
                    by_file.setdefault(p["source_file"], []).append(p)

                st.write(f"**{len(pairs)} golden pair(s)** across {len(by_file)} file(s)")

                for fname, fps in by_file.items():
                    st.markdown(f"#### 📄 {fname} ({len(fps)})")
                    for p in fps:
                        with st.expander(f"[{p['id']}] {p['question'][:90]}"):
                            st.write(f"**Expected:** {p['expected_answer']}")
                            if st.button("Delete", key=f"delg_{p['id']}"):
                                r = api_delete(f"/eval/pairs/{p['id']}")
                                if r and r.status_code == 200:
                                    st.rerun()


# ── Eval tab ──────────────────────────────────────────────────────────────────
with tab_eval:
    st.subheader("Run Evaluation")

    # Dataset selector
    c1, c2, c3 = st.columns(3)
    with c1:
        dataset_type = st.selectbox(
            "Dataset",
            ["golden", "silver", "both"],
            format_func=lambda x: {"golden": "🏅 Golden", "silver": "⚗️ Silver",
                                    "both": "⚗️+🏅 Both"}[x],
        )
    with c2:
        eval_top_k = st.slider("Sources per query", 1, 10, 3, key="ev_topk")
    with c3:
        use_judge = st.checkbox("LLM judge (Gemini)", value=True)
        st.caption("Unchecked = keyword fallback")

    # Show how many pairs will run
    resp = api_get(f"/eval/pairs?pair_type={dataset_type}")
    if resp and resp.status_code == 200:
        n_pairs = len(resp.json().get("pairs", []))
        if dataset_type == "both":
            resp2 = api_get("/eval/pairs?pair_type=all")
            n_pairs = len(resp2.json().get("pairs", [])) if resp2 and resp2.status_code == 200 else n_pairs
        st.info(f"**{n_pairs}** pair(s) will be evaluated.")
    else:
        n_pairs = 0

    run_btn = st.button(
        f"Run Eval on {dataset_type.title()} Dataset",
        type="primary",
        disabled=(n_pairs == 0),
    )

    if run_btn:
        with st.spinner(f"Running {n_pairs} evaluations with LLM judge… (~{n_pairs * 10}s)"):
            resp = api_post("/eval/run", json={
                "dataset_type": dataset_type,
                "top_k": eval_top_k,
                "use_llm_judge": use_judge,
            }, timeout=max(120, n_pairs * 20))

        if resp is None:
            show_error("Connection failed or timed out.")
        elif resp.status_code == 200:
            data = resp.json()
            summary = data["summary"]
            results = data["results"]

            # ── Summary metrics ───────────────────────────────────────────
            st.divider()
            st.markdown("## Eval Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total", summary["total"])
            m2.metric("Passed", summary["passed"])
            m3.metric("Failed", summary["failed"])
            m4.metric("Avg Score", f"{summary['avg_score']:.2f}")

            pass_rate = summary["pass_rate"]
            if pass_rate >= 75:
                st.success(f"Pass rate: **{pass_rate}%** — RAG is working well.")
            elif pass_rate >= 50:
                st.warning(f"Pass rate: **{pass_rate}%** — some gaps in retrieval.")
            else:
                st.error(f"Pass rate: **{pass_rate}%** — check uploads and chunking.")

            # ── Per-file breakdown ────────────────────────────────────────
            by_file: dict = {}
            for r in results:
                by_file.setdefault(r["source_file"], []).append(r)

            if len(by_file) > 1:
                st.divider()
                st.markdown("### Per-file breakdown")
                for fname, frs in by_file.items():
                    fp = sum(1 for r in frs if r["passed"])
                    fa = round(sum(r["score"] for r in frs) / len(frs), 2)
                    st.markdown(
                        f"📄 **{fname}** — {fp}/{len(frs)} passed · avg score {fa}"
                    )

            # ── Detailed results ──────────────────────────────────────────
            st.divider()
            st.markdown("### Detailed Results")

            passed_results = [r for r in results if r["passed"]]
            failed_results = [r for r in results if not r["passed"]]

            view = st.radio("Show", ["All", "Failed only", "Passed only"], horizontal=True)
            show_list = (
                results if view == "All"
                else (failed_results if view == "Failed only" else passed_results)
            )

            for r in show_list:
                badge = "✅ PASS" if r["passed"] else "❌ FAIL"
                score_txt = f"score {r['score']:.2f}"
                label = f"{badge} [{r['pair_id']}] {r['question'][:80]}… — {score_txt}"

                with st.expander(label):
                    st.caption(f"📄 {r['source_file']} · type: {r['pair_type']}")

                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Expected Answer**")
                        st.info(r["expected_answer"])
                    with c2:
                        st.markdown("**RAG Answer**")
                        st.write(r["rag_answer"])

                    st.markdown(f"**Judge reasoning:** {r['reasoning']}")

                    if r.get("sources"):
                        with st.expander("Retrieved sources"):
                            for s in r["sources"]:
                                st.caption(f"📄 {s['filename']} — score {s['score']:.3f}")
        else:
            show_error(f"Eval failed: {resp_detail(resp)}")
    else:
        if n_pairs == 0:
            st.write("No pairs in the selected dataset. Generate silver pairs and promote to golden first.")
