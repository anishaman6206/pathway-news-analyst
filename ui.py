import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None


# ----------------------------
# Config
# ----------------------------
DEFAULT_API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))
DEFAULT_MIN_SCORE = float(os.getenv("DEFAULT_MIN_SCORE", "0.28"))
DEFAULT_FEED_SIZE = int(os.getenv("DEFAULT_FEED_SIZE", "10"))
DEFAULT_POLL_SECONDS = float(os.getenv("POLL_SECONDS", "2.0"))

st.set_page_config(page_title="Live RAG News Analyst", layout="wide")

st.markdown(
    """
<style>
.badge {
  display:inline-block; padding:0.15rem 0.55rem; border-radius:999px;
  font-size:0.80rem; border:1px solid rgba(255,255,255,0.15);
  margin-right:0.35rem;
}
.small { opacity: 0.85; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# HTTP helpers
# ----------------------------
@st.cache_resource
def http_session() -> requests.Session:
    return requests.Session()


def api_url(base: str, path: str) -> str:
    base = (base or "").rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def safe_short(s: str, n: int = 200) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def get_health(api_base: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    try:
        r = http_session().get(api_url(api_base, "/healthz"), timeout=4)
        if r.status_code != 200:
            return False, {}, f"HTTP {r.status_code}"
        return True, r.json(), None
    except Exception as e:
        return False, {}, str(e)


def get_latest(api_base: str, k: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        r = http_session().get(api_url(api_base, "/v1/latest"), params={"k": int(k)}, timeout=6)
        if r.status_code != 200:
            return [], f"HTTP {r.status_code}: {r.text[:200]}"
        data = r.json()
        hits = data.get("hits", []) if isinstance(data, dict) else []
        return hits if isinstance(hits, list) else [], None
    except Exception as e:
        return [], str(e)


def post_retrieve(api_base: str, query: str, k: int, min_score: float) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    # backend may ignore min_score if you didn't add it; pydantic typically ignores extra fields
    payload = {"query": query, "k": int(k), "min_score": float(min_score)}
    try:
        r = http_session().post(api_url(api_base, "/v1/retrieve"), json=payload, timeout=25)
        if r.status_code != 200:
            return [], f"HTTP {r.status_code}: {r.text[:250]}"
        data = r.json()
        hits = data.get("hits", []) if isinstance(data, dict) else []
        return hits if isinstance(hits, list) else [], None
    except Exception as e:
        return [], str(e)


def post_answer(api_base: str, query: str, k: int, min_score: float) -> Tuple[Dict[str, Any], Optional[str]]:
    # IMPORTANT: your backend expects "query"
    payload = {"query": query, "k": int(k), "min_score": float(min_score)}
    try:
        r = http_session().post(api_url(api_base, "/v2/answer"), json=payload, timeout=60)
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {r.text[:250]}"
        data = r.json()
        return data if isinstance(data, dict) else {}, None
    except Exception as e:
        return {}, str(e)


# ----------------------------
# Intent detection (UI-side “assistant-ness”)
# ----------------------------
_SMALLTALK_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|hii+|heyy+|hola|sup|how are you|good (morning|afternoon|evening))\s*[!.]*\s*$",
    re.IGNORECASE,
)

_THANKS_RE = re.compile(r"^\s*(thanks|thank you|thx)\s*[!.]*\s*$", re.IGNORECASE)

# Mirror your backend's latest-query triggers + a couple extra for UX
_LATEST_KEYWORDS = [
    "latest",
    "recent",
    "today",
    "summarize",
    "summary",
    "top news",
    "what happened",
    "headlines",
    "what's new",
    "whats new",
]


def looks_like_latest_query(q: str) -> bool:
    qn = (q or "").lower()
    return any(k in qn for k in _LATEST_KEYWORDS)


def is_smalltalk(q: str) -> bool:
    return bool(_SMALLTALK_RE.match(q or ""))


def is_thanks(q: str) -> bool:
    return bool(_THANKS_RE.match(q or ""))


def help_text(live_hits: List[Dict[str, Any]]) -> str:
    titles = [h.get("title", "") for h in live_hits[:5] if isinstance(h, dict) and h.get("title")]
    examples = "\n".join([f"- {t}" for t in titles]) if titles else "- (No ingested articles yet)"
    return (
        "Here’s what I can do:\n\n"
        "1) Answer questions **using only ingested articles** (grounded)\n"
        "2) Summarize **latest** ingested articles\n"
        "3) Show you sources + similarity scores in the Inspector\n\n"
        "Try:\n"
        "- “summarize latest”\n"
        "- “what happened in global markets”\n"
        "- “give key points of the electric bus fleet story”\n\n"
        "Articles currently in the feed:\n"
        f"{examples}"
    )


def friendly_no_hits_reply(q: str, min_score: float, live_hits: List[Dict[str, Any]]) -> str:
    titles = [h.get("title", "") for h in live_hits[:6] if isinstance(h, dict) and h.get("title")]
    titles = [t for t in titles if t]
    suggestions = "\n".join([f"- {t}" for t in titles[:5]]) if titles else "- (No articles yet)"

    return (
        "I couldn’t find any ingested articles that match your question strongly enough.\n\n"
        f"Things you can try:\n"
        f"- Lower **Min Similarity** (currently {min_score:.2f})\n"
        f"- Ask using keywords that appear in the article titles/snippets\n"
        f"- Or simply ask: **“summarize latest”**\n\n"
        "What I currently have in the Live Articles feed:\n"
        f"{suggestions}"
    )


def soften_backend_refusal(ans_text: str, min_score: float, live_hits: List[Dict[str, Any]]) -> str:
    # If backend returns the standard refusal, keep it (good grounding),
    # but add a more assistant-like follow-up.
    if "i don't have enough information in the ingested articles" not in (ans_text or "").lower():
        return ans_text

    return (
        ans_text.strip()
        + "\n\n"
        + "If you want, I can still help:\n"
        + "- Ask **“summarize latest”** to get a quick briefing\n"
        + f"- Or lower **Min Similarity** (currently {min_score:.2f}) and retry\n"
        + "- Or click **Ask about this** on an article from the left."
    )


# ----------------------------
# Session state
# ----------------------------
def ss_init():
    defaults = {
        "api_base": DEFAULT_API_BASE,
        "top_k": DEFAULT_K,
        "min_score": DEFAULT_MIN_SCORE,
        "feed_size": DEFAULT_FEED_SIZE,
        "poll_seconds": DEFAULT_POLL_SECONDS,
        "live_mode": True,

        # chat turns stored as a list of dicts
        # each: {"id": int, "query": str, "answer": str, "grounded": bool, "confidence": float}
        "turns": [],
        "selected_turn": None,

        # sources
        "retrieved_by_turn": {},  # turn_id -> hits list
        "citations_by_turn": {},  # turn_id -> citations list

        # live feed
        "live_hits": [],
        "prev_live_ids": set(),
        "new_live_ids": set(),
        "last_live_fetch": 0.0,

        # input draft
        "draft": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ss_init()


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## Settings")
    st.text_input("API Base", key="api_base")

    st.toggle("Live Mode (auto refresh)", key="live_mode")
    st.slider("Poll interval (seconds)", 0.5, 10.0, float(st.session_state.poll_seconds), 0.5, key="poll_seconds")
    st.slider("Live feed size", 5, 50, int(st.session_state.feed_size), 5, key="feed_size")

    st.markdown("---")
    st.markdown("### Retrieval Controls")
    st.slider("Top K", 1, 12, int(st.session_state.top_k), 1, key="top_k")
    st.slider("Min Similarity", 0.0, 0.60, float(st.session_state.min_score), 0.01, key="min_score")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.turns = []
            st.session_state.selected_turn = None
            st.session_state.retrieved_by_turn = {}
            st.session_state.citations_by_turn = {}
            st.rerun()
    with c2:
        if st.button("Refresh now", use_container_width=True):
            st.session_state.last_live_fetch = 0.0
            st.rerun()

    ok, health, err = get_health(st.session_state.api_base)
    if ok:
        st.success("Backend connected")
        st.caption(f"docs_seen: {health.get('docs_seen', 0)} | vectors_ready: {health.get('vectors_ready', 0)}")
    else:
        st.error("Backend not reachable")
        if err:
            st.caption(err)

    if st.session_state.live_mode and st_autorefresh is None:
        st.warning("Install auto-refresh:\n`pip install streamlit-autorefresh`")


# ----------------------------
# Auto refresh
# ----------------------------
ok, health, err = get_health(st.session_state.api_base)

if st.session_state.live_mode and st_autorefresh is not None:
    st_autorefresh(interval=int(float(st.session_state.poll_seconds) * 1000), key="live_tick")


# ----------------------------
# Live feed updater
# ----------------------------
def update_live():
    if not ok:
        return
    now = time.time()
    if now - float(st.session_state.last_live_fetch) < float(st.session_state.poll_seconds):
        return

    hits, e = get_latest(st.session_state.api_base, int(st.session_state.feed_size))
    st.session_state.last_live_fetch = now
    if e is not None:
        return

    cur_ids = set(str(h.get("article_id", "")) for h in hits if isinstance(h, dict))
    prev_ids = st.session_state.prev_live_ids if isinstance(st.session_state.prev_live_ids, set) else set()
    new_ids = cur_ids - prev_ids

    st.session_state.live_hits = hits
    st.session_state.prev_live_ids = cur_ids
    st.session_state.new_live_ids = new_ids

    if new_ids and hasattr(st, "toast"):
        st.toast(f"{len(new_ids)} new article(s) ingested", icon="📰")


update_live()


# ----------------------------
# Header
# ----------------------------
st.title("Live RAG News Analyst")

b1, b2, b3 = st.columns([1.2, 1.2, 2.2])
with b1:
    st.markdown(f"<span class='badge'>{'Connected ✅' if ok else 'Disconnected ❌'}</span>", unsafe_allow_html=True)
with b2:
    st.markdown(
        f"<span class='badge'>docs_seen: {health.get('docs_seen', '-') if ok else '-'}</span>"
        f"<span class='badge'>vectors_ready: {health.get('vectors_ready', '-') if ok else '-'}</span>",
        unsafe_allow_html=True,
    )
with b3:
    st.caption(st.session_state.api_base)

st.divider()


# ----------------------------
# Layout
# ----------------------------
left, center, right = st.columns([1.15, 1.85, 1.35], gap="large")


# ----------------------------
# Left: Live Articles
# ----------------------------
with left:
    st.subheader("Live Articles")

    hits = st.session_state.live_hits if isinstance(st.session_state.live_hits, list) else []
    new_ids = st.session_state.new_live_ids if isinstance(st.session_state.new_live_ids, set) else set()

    if not ok:
        st.info("Start your backend, then refresh.")
    elif not hits:
        st.info("No articles yet. Insert/ingest docs, then wait a second.")
    else:
        for i, h in enumerate(hits):
            if not isinstance(h, dict):
                continue

            aid = str(h.get("article_id", ""))
            title = str(h.get("title", "(untitled)"))
            src = str(h.get("source_name", ""))
            published = str(h.get("published_at", ""))
            url = str(h.get("url", ""))
            snippet = safe_short(str(h.get("snippet", "")), 220)

            meta_bits = []
            if aid and aid in new_ids:
                meta_bits.append("NEW")
            if src:
                meta_bits.append(src)
            if published:
                meta_bits.append(published)
            meta = " • ".join(meta_bits)

            with st.container(border=True):
                st.markdown(f"**{title}**")
                if meta:
                    st.caption(meta)
                st.write(snippet)
                if url:
                    st.markdown(f"[open source]({url})")

                # IMPORTANT: don't include "summarize" in the draft, otherwise your backend will treat it as "latest query"
                if st.button("Ask about this", key=f"ask_{i}", use_container_width=True):
                    st.session_state.draft = f"What are the key points of the article titled: {title}?"
                    st.rerun()


# ----------------------------
# Center: Chat
# ----------------------------
with center:
    st.subheader("Chat")

    if not st.session_state.turns:
        with st.chat_message("assistant"):
            st.markdown(
                "Hey 👋 I’m your live news assistant.\n\n"
                "I can:\n"
                "- answer questions using only the ingested articles\n"
                "- summarize the latest feed\n\n"
                "Try: **summarize latest** or click **Ask about this** on the left."
            )

    # Render turns
    for t in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(t["query"])

        with st.chat_message("assistant"):
            st.markdown(t["answer"])
            grounded = bool(t.get("grounded", False))
            conf = float(t.get("confidence", 0.0) or 0.0)
            st.caption(f"grounded: {'✅' if grounded else '❌'} • confidence: {conf:.3f}")

    # Quick actions
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        if st.button("Summarize latest", use_container_width=True, disabled=not ok):
            st.session_state.draft = "summarize latest"
            st.rerun()
    with qa2:
        if st.button("What’s new today?", use_container_width=True, disabled=not ok):
            st.session_state.draft = "what happened today"
            st.rerun()
    with qa3:
        if st.button("Help", use_container_width=True):
            st.session_state.draft = "help"
            st.rerun()

    # Input form (safe)
    with st.form("chat_form", clear_on_submit=True):
        st.text_input("Your message", key="draft", placeholder="Say hi, ask for a summary, or ask about a topic…", disabled=not ok)
        submitted = st.form_submit_button("Send", type="primary", use_container_width=True, disabled=not ok)

    if submitted and ok:
        q = (st.session_state.draft or "").strip()
        if not q:
            st.warning("Type something first.")
            st.stop()

        # Create turn id
        turn_id = (st.session_state.turns[-1]["id"] + 1) if st.session_state.turns else 1
        st.session_state.selected_turn = turn_id

        live_hits_now = st.session_state.live_hits if isinstance(st.session_state.live_hits, list) else []
        min_score = float(st.session_state.min_score)

        # --- UI-side assistant handling ---
        if is_smalltalk(q):
            answer_text = (
                "Hello! 🙂\n\n"
                "I’m here and ready. Want a quick briefing?\n"
                "- Try **summarize latest**\n"
                "- Or click **Ask about this** on any article in the Live Articles panel."
            )
            st.session_state.turns.append({"id": turn_id, "query": q, "answer": answer_text, "grounded": False, "confidence": 0.0})
            st.session_state.retrieved_by_turn[turn_id] = []
            st.session_state.citations_by_turn[turn_id] = []
            st.rerun()

        if is_thanks(q):
            answer_text = "You’re welcome 🙂\n\nIf you want, ask **summarize latest** for a quick news rundown."
            st.session_state.turns.append({"id": turn_id, "query": q, "answer": answer_text, "grounded": False, "confidence": 0.0})
            st.session_state.retrieved_by_turn[turn_id] = []
            st.session_state.citations_by_turn[turn_id] = []
            st.rerun()

        if q.lower().strip() in {"help", "what can you do", "what do you do"}:
            answer_text = help_text(live_hits_now)
            st.session_state.turns.append({"id": turn_id, "query": q, "answer": answer_text, "grounded": False, "confidence": 0.0})
            st.session_state.retrieved_by_turn[turn_id] = []
            st.session_state.citations_by_turn[turn_id] = []
            st.rerun()

        # If user asks for "latest/summarize", call /v2/answer directly and align inspector with citations
        if looks_like_latest_query(q):
            with st.spinner("Briefing the latest…"):
                ans, aerr = post_answer(st.session_state.api_base, q, int(st.session_state.top_k), min_score)

            if aerr or not ans:
                answer_text = f"I couldn’t generate a briefing right now.\n\nError: {aerr or 'unknown'}"
                grounded = False
                conf = 0.0
                citations = []
            else:
                answer_text = str(ans.get("answer", "")).strip()
                grounded = bool(ans.get("grounded", False))
                conf = float(ans.get("confidence", 0.0) or 0.0)
                citations = ans.get("citations", []) if isinstance(ans.get("citations", []), list) else []
                answer_text = soften_backend_refusal(answer_text, min_score, live_hits_now)

            st.session_state.turns.append({"id": turn_id, "query": q, "answer": answer_text, "grounded": grounded, "confidence": conf})
            st.session_state.retrieved_by_turn[turn_id] = citations
            st.session_state.citations_by_turn[turn_id] = citations
            st.rerun()

        # Normal question: retrieve then answer
        with st.spinner("Searching sources…"):
            rhits, rerr = post_retrieve(st.session_state.api_base, q, int(st.session_state.top_k), min_score)

        if rerr:
            answer_text = f"I hit an error while searching sources:\n\n{rerr}"
            st.session_state.turns.append({"id": turn_id, "query": q, "answer": answer_text, "grounded": False, "confidence": 0.0})
            st.session_state.retrieved_by_turn[turn_id] = []
            st.session_state.citations_by_turn[turn_id] = []
            st.rerun()

        st.session_state.retrieved_by_turn[turn_id] = rhits

        if not rhits:
            # This is the core UX fix: always respond like an assistant when nothing hits
            answer_text = friendly_no_hits_reply(q, min_score, live_hits_now)
            st.session_state.turns.append({"id": turn_id, "query": q, "answer": answer_text, "grounded": False, "confidence": 0.0})
            st.session_state.citations_by_turn[turn_id] = []
            st.rerun()

        with st.spinner("Writing answer…"):
            ans, aerr = post_answer(st.session_state.api_base, q, int(st.session_state.top_k), min_score)

        if aerr or not ans:
            answer_text = f"I found sources, but couldn’t generate an answer.\n\nError: {aerr or 'unknown'}"
            grounded = False
            conf = 0.0
            citations = []
        else:
            answer_text = str(ans.get("answer", "")).strip()
            grounded = bool(ans.get("grounded", False))
            conf = float(ans.get("confidence", 0.0) or 0.0)
            citations = ans.get("citations", []) if isinstance(ans.get("citations", []), list) else []
            answer_text = soften_backend_refusal(answer_text, min_score, live_hits_now)

        st.session_state.turns.append({"id": turn_id, "query": q, "answer": answer_text, "grounded": grounded, "confidence": conf})
        st.session_state.citations_by_turn[turn_id] = citations
        st.rerun()


# ----------------------------
# Right: Inspector
# ----------------------------
with right:
    st.subheader("Inspector")

    if not st.session_state.turns:
        st.info("Ask something to inspect sources + citations.")
    else:
        turn_ids = [t["id"] for t in st.session_state.turns]
        default_tid = st.session_state.selected_turn if st.session_state.selected_turn in turn_ids else turn_ids[-1]

        selected_tid = st.selectbox(
            "Select a question",
            options=turn_ids,
            index=turn_ids.index(default_tid),
            format_func=lambda tid: f"Q{tid}: {st.session_state.turns[turn_ids.index(tid)]['query'][:55]}",
        )
        st.session_state.selected_turn = selected_tid

        tab1, tab2, tab3 = st.tabs(["Retrieved Sources", "Answer Citations", "Diagnostics"])

        with tab1:
            hits = st.session_state.retrieved_by_turn.get(selected_tid, [])
            if not hits:
                st.warning("No retrieved sources for this turn.")
            else:
                for h in hits:
                    if not isinstance(h, dict):
                        continue
                    title = str(h.get("title", "(untitled)"))
                    score = h.get("score", None)
                    src = str(h.get("source_name", ""))
                    published = str(h.get("published_at", ""))
                    url = str(h.get("url", ""))
                    snippet = safe_short(str(h.get("snippet", "")), 320)

                    score_txt = f"{float(score):.3f}" if score is not None else "—"
                    meta = " • ".join([x for x in [f"score: {score_txt}", src, published] if x])

                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.caption(meta)
                        st.write(snippet)
                        if url:
                            st.markdown(f"[open source]({url})")

        with tab2:
            cits = st.session_state.citations_by_turn.get(selected_tid, [])
            if not cits:
                st.info("No citations for this turn.")
            else:
                for c in cits:
                    if not isinstance(c, dict):
                        continue
                    title = str(c.get("title", "(untitled)"))
                    score = c.get("score", None)
                    src = str(c.get("source_name", ""))
                    published = str(c.get("published_at", ""))
                    url = str(c.get("url", ""))
                    snippet = safe_short(str(c.get("snippet", "")), 320)

                    score_txt = f"{float(score):.3f}" if score is not None else "—"
                    meta = " • ".join([x for x in [f"score: {score_txt}", src, published] if x])

                    with st.container(border=True):
                        st.markdown(f"**{title}**")
                        st.caption(meta)
                        st.write(snippet)
                        if url:
                            st.markdown(f"[open source]({url})")

        with tab3:
            ok2, health2, err2 = get_health(st.session_state.api_base)
            if ok2:
                st.success("Backend OK")
                st.json(health2)
            else:
                st.error("Backend not reachable")
                if err2:
                    st.caption(err2)
