# src/pipeline/rag_server.py
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pathway as pw
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn


# -------------------- API models --------------------
class RetrieveRequest(BaseModel):
    query: str
    k: int = 5
    min_score: float = 0.0

class RetrieveHit(BaseModel):
    article_id: str
    title: str
    source_name: str
    url: str
    published_at: str
    snippet: str
    score: float = 0.0  # cosine similarity


class RetrieveResponse(BaseModel):
    query: str
    hits: List[RetrieveHit]


class AnswerRequest(BaseModel):
    query: str
    k: int = 5


class AnswerResponse(BaseModel):
    query: str
    answer: str
    citations: List[RetrieveHit]
    grounded: bool
    confidence: float = 0.0


class LatestResponse(BaseModel):
    hits: List[RetrieveHit]


# -------------------- text utils --------------------
_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(s: str) -> str:
    return " ".join(_WORD_RE.findall((s or "").lower()))


def _looks_like_latest_query(q: str) -> bool:
    qn = (q or "").lower()
    return any(x in qn for x in ["latest", "recent", "today", "summarize", "summary", "top news", "what happened"])


# -------------------- vector math (no numpy dependency) --------------------
def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# -------------------- Azure OpenAI: embeddings + chat --------------------
def _azure_embeddings(texts: List[str], settings) -> List[List[float]]:
    """
    Azure embeddings endpoint:
    POST {base}/openai/deployments/{emb_deployment}/embeddings?api-version=...
    """
    if not (
        getattr(settings, "azure_api_key", None)
        and getattr(settings, "azure_api_base", None)
        and getattr(settings, "azure_api_version", None)
        and getattr(settings, "azure_embeddings_deployment", None)
    ):
        raise RuntimeError("Azure embeddings are not configured in settings/.env")

    endpoint = (
        f"{settings.azure_api_base.rstrip('/')}/openai/deployments/"
        f"{settings.azure_embeddings_deployment}/embeddings"
        f"?api-version={settings.azure_api_version}"
    )

    headers = {"api-key": settings.azure_api_key, "Content-Type": "application/json"}

    # Azure embeddings API expects `input` (string or list of strings)
    payload = {"input": texts}
    r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    # Azure returns: {"data":[{"embedding":[...], "index":0}, ...]}
    out: List[List[float]] = [None] * len(texts)  # type: ignore
    for item in data.get("data", []):
        idx = item.get("index", 0)
        out[idx] = item.get("embedding", [])
    return out  # type: ignore


def _azure_chat_answer(query: str, hits: List[RetrieveHit], settings) -> str:
    """
    Guardrailed answering: context-only.
    """
    # Fallback deterministic summary if chat not configured
    if not (
        getattr(settings, "azure_api_key", None)
        and getattr(settings, "azure_api_base", None)
        and getattr(settings, "azure_api_version", None)
        and getattr(settings, "azure_chat_deployment", None)
    ):
        if not hits:
            return "I don't have enough information in the ingested articles to answer that."
        bullets: List[str] = []
        for h in hits[:3]:
            snippet = (h.snippet or "").strip().replace("\n", " ")[:180]
            bullets.append(f"- {h.title}: {snippet}")
        return "Here’s what the ingested articles say:\n" + "\n".join(bullets)

    endpoint = (
        f"{settings.azure_api_base.rstrip('/')}/openai/deployments/"
        f"{settings.azure_chat_deployment}/chat/completions"
        f"?api-version={settings.azure_api_version}"
    )

    ctx_parts = []
    for i, h in enumerate(hits, start=1):
        ctx_parts.append(
            f"[Doc {i}]\n"
            f"Title: {h.title}\n"
            f"Source: {h.source_name}\n"
            f"Published: {h.published_at}\n"
            f"URL: {h.url}\n"
            f"Snippet: {h.snippet}\n"
        )
    context = "\n\n".join(ctx_parts)

    system = (
        "You are a Live News Analyst.\n"
        "You MUST answer ONLY using the provided Context documents.\n"
        "If the answer is not clearly supported by the Context, reply exactly:\n"
        "I don't have enough information in the ingested articles to answer that.\n"
        "Be concise. When using info, cite like [Doc 1], [Doc 2]."
    )

    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
        ],
        "temperature": 0.2,
        "max_tokens": 450,
    }
    headers = {"api-key": settings.azure_api_key, "Content-Type": "application/json"}

    r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


# -------------------- Grounding / confidence --------------------
def _is_grounded(similarities: List[float]) -> Tuple[bool, float]:
    """
    Stronger grounding confidence:
    - Need a decent top similarity
    - Either strong top OR decent margin vs 2nd place
    """
    if not similarities:
        return (False, 0.0)
    top = similarities[0]
    second = similarities[1] if len(similarities) > 1 else 0.0
    margin = top - second

    # Tune these thresholds as you like:
    # - 0.30+ is usually "pretty confident"
    # - 0.24 with margin is okay
    grounded = (top >= 0.30) or (top >= 0.24 and margin >= 0.03)
    confidence = max(0.0, min(1.0, top))
    return grounded, confidence


# -------------------- server --------------------
def run_rag_server(docs: pw.Table, settings) -> None:
    """
    - Pathway streams docs to JSONL
    - Background thread tails JSONL and builds a vector index incrementally
    - Retrieval uses embeddings + cosine similarity
    """

    docs_text = docs.select(
        article_id=pw.this.article_id,
        title=pw.this.title,
        source_name=pw.this.source_name,
        url=pw.this.url,
        published_at=pw.this.published_at,
        text=pw.apply(lambda b: b.decode("utf-8") if b is not None else "", pw.this.data),
    )

    out_dir = "/app/_cache"
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "docs.jsonl")
    vec_path = os.path.join(out_dir, "vectors.jsonl")  # persistent embeddings cache

    pw.io.jsonlines.write(docs_text, jsonl_path)

    def _run_pw():
        pw.run()

    threading.Thread(target=_run_pw, daemon=True).start()

    # -------------------- in-memory stores --------------------
    docs_by_id: Dict[str, Dict[str, Any]] = {}
    vec_by_id: Dict[str, List[float]] = {}
    lock = threading.Lock()

    # Load any existing vectors cache (survives container restarts if volume-mounted; otherwise harmless)
    if os.path.exists(vec_path):
        try:
            with open(vec_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    aid = str(obj.get("article_id", ""))
                    emb = obj.get("embedding", [])
                    if aid and isinstance(emb, list) and emb:
                        vec_by_id[aid] = emb
        except Exception:
            pass

    def _read_all_docs() -> List[Dict[str, Any]]:
        if not os.path.exists(jsonl_path):
            return []
        items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                except Exception:
                    continue
        return items

    def _to_hit(it: Dict[str, Any], score: float = 0.0) -> RetrieveHit:
        text = str(it.get("text", ""))
        return RetrieveHit(
            article_id=str(it.get("article_id", "")),
            title=str(it.get("title", "")),
            source_name=str(it.get("source_name", "")),
            url=str(it.get("url", "")),
            published_at=str(it.get("published_at", "")),
            snippet=text[:400],
            score=score,
        )

    def _latest(k: int) -> List[RetrieveHit]:
        items = _read_all_docs()
        items.sort(key=lambda x: str(x.get("published_at", "")), reverse=True)
        return [_to_hit(it, 0.0) for it in items[:k]]

    # -------------------- vector index builder (tails docs.jsonl) --------------------
    def _index_worker():
        last_size = 0
        while True:
            try:
                if os.path.exists(jsonl_path):
                    size = os.path.getsize(jsonl_path)
                    if size < last_size:
                        last_size = 0

                    new_lines: List[str] = []
                    if size > last_size:
                        with open(jsonl_path, "r", encoding="utf-8") as f:
                            f.seek(last_size)
                            chunk = f.read()
                            last_size = f.tell()
                        new_lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]

                    # Parse docs, update docs_by_id, compute embeddings for missing ids
                    to_embed: List[Tuple[str, str]] = []
                    for line in new_lines:
                        try:
                            obj = json.loads(line)
                            if not isinstance(obj, dict):
                                continue
                            aid = str(obj.get("article_id", ""))
                            if not aid:
                                continue

                            with lock:
                                docs_by_id[aid] = obj

                            # Build text for embedding (title + body)
                            if aid not in vec_by_id:
                                title = str(obj.get("title", ""))
                                body = str(obj.get("text", ""))
                                emb_text = (title + "\n\n" + body).strip()
                                if emb_text:
                                    to_embed.append((aid, emb_text))
                        except Exception:
                            continue

                    # Batch embeddings to Azure (faster + cheaper)
                    if to_embed:
                        ids = [x[0] for x in to_embed]
                        texts = [x[1] for x in to_embed]
                        try:
                            embs = _azure_embeddings(texts, settings)
                            with lock:
                                for aid, emb in zip(ids, embs):
                                    if emb:
                                        vec_by_id[aid] = emb
                            # persist cache
                            with open(vec_path, "a", encoding="utf-8") as f:
                                for aid, emb in zip(ids, embs):
                                    if emb:
                                        f.write(json.dumps({"article_id": aid, "embedding": emb}) + "\n")
                        except Exception:
                            # If Azure embeddings temporarily fail, we'll retry on future loops
                            pass

                time.sleep(0.8)
            except Exception:
                time.sleep(1.0)

    # Seed index once at startup
    for obj in _read_all_docs():
        aid = str(obj.get("article_id", ""))
        if aid:
            docs_by_id[aid] = obj

    threading.Thread(target=_index_worker, daemon=True).start()

    # -------------------- API --------------------
    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        with lock:
            return {
                "ok": True,
                "docs_seen": len(docs_by_id),
                "vectors_ready": len(vec_by_id),
                "jsonl_path": jsonl_path,
                "vec_path": vec_path,
            }

    @app.get("/v1/latest", response_model=LatestResponse)
    def latest(k: int = 30):
        return LatestResponse(hits=_latest(k))

    @app.post("/v1/retrieve", response_model=RetrieveResponse)
    def retrieve(req: RetrieveRequest):
        q = (req.query or "").strip()
        if not q:
            return RetrieveResponse(query=req.query, hits=[])

        # If embeddings not configured or vectors empty, fallback to "latest"
        with lock:
            have_vecs = len(vec_by_id) > 0

        if not have_vecs:
            # fallback: very simple keyword-ish behavior using latest (keeps UX alive)
            hits = _latest(req.k)
            return RetrieveResponse(query=req.query, hits=hits)

        # Embed query
        try:
            q_emb = _azure_embeddings([q], settings)[0]
        except Exception:
            # temporary Azure issue → fallback to latest
            hits = _latest(req.k)
            return RetrieveResponse(query=req.query, hits=hits)

        scored: List[Tuple[float, str]] = []
        with lock:
            for aid, emb in vec_by_id.items():
                scored.append((_cosine(q_emb, emb), aid))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: req.k]

        hits: List[RetrieveHit] = []
        for sim, aid in top:
            with lock:
                doc = docs_by_id.get(aid)
            if doc:
                hits.append(_to_hit(doc, score=float(sim)))

        return RetrieveResponse(query=req.query, hits=hits)

    @app.post("/v2/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest):
        # Latest-style questions are grounded in latest docs
        if _looks_like_latest_query(req.query):
            hits = _latest(req.k)
            if not hits:
                return AnswerResponse(
                    query=req.query,
                    answer="I don't have enough information in the ingested articles to answer that.",
                    citations=[],
                    grounded=False,
                    confidence=0.0,
                )
            ans = _azure_chat_answer(req.query, hits, settings)
            grounded = "I don't have enough information in the ingested articles" not in ans
            return AnswerResponse(query=req.query, answer=ans, citations=hits, grounded=grounded, confidence=0.35)

        # Normal Q&A: retrieve by embeddings
        r = retrieve(RetrieveRequest(query=req.query, k=req.k))
        hits = r.hits

        if not hits:
            return AnswerResponse(
                query=req.query,
                answer="I don't have enough information in the ingested articles to answer that.",
                citations=[],
                grounded=False,
                confidence=0.0,
            )

        sims = [h.score for h in hits]
        grounded, conf = _is_grounded(sims)

        if not grounded:
            return AnswerResponse(
                query=req.query,
                answer="I don't have enough information in the ingested articles to answer that confidently. Try asking with keywords from the ingested articles.",
                citations=hits,
                grounded=False,
                confidence=conf,
            )

        ans = _azure_chat_answer(req.query, hits, settings)
        grounded2 = "I don't have enough information in the ingested articles" not in ans
        return AnswerResponse(query=req.query, answer=ans, citations=hits, grounded=grounded2, confidence=conf)

    @app.get("/events")
    def events():
        """
        SSE stream: pushes new_article events when docs.jsonl grows.
        """
        def gen():
            last_size = 0
            yield "event: hello\ndata: {\"ok\": true}\n\n"
            while True:
                try:
                    if os.path.exists(jsonl_path):
                        size = os.path.getsize(jsonl_path)
                        if size < last_size:
                            last_size = 0

                        if size > last_size:
                            with open(jsonl_path, "r", encoding="utf-8") as f:
                                f.seek(last_size)
                                new_data = f.read()
                                last_size = f.tell()

                            for line in new_data.splitlines():
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    obj = json.loads(line)
                                    hit = _to_hit(obj, score=0.0)
                                    yield f"event: new_article\ndata: {json.dumps(hit.model_dump())}\n\n"
                                except Exception:
                                    continue

                    time.sleep(0.8)
                except Exception:
                    time.sleep(1.0)

        return StreamingResponse(gen(), media_type="text/event-stream")

    host = getattr(settings, "host", "0.0.0.0")
    port = int(getattr(settings, "port", 8000))
    print(f"[RAG SERVER] Running on http://{host}:{port}", flush=True)
    print("[RAG SERVER] Endpoints: GET /healthz, GET /v1/latest, GET /events, POST /v1/retrieve, POST /v2/answer", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
