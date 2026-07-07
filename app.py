from dotenv import load_dotenv
load_dotenv()  # MUST be before any core/ or utils/ imports (they read env vars at import time)

import streamlit as st

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_questions

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Summarize",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Minimal purple-themed styling ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    padding-top: 3.5rem;
    max-width: 1100px;
}

/* Card-style containers */
.info-card {
    border: 1px solid rgba(168,85,247,0.25);
    background: rgba(168,85,247,0.05);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

.info-card h4 {
    margin-top: 0;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: #c084fc;
}

/* Chat bubbles */
.chat-msg {
    padding: 0.6rem 0.9rem;
    border-radius: 10px;
    margin-bottom: 0.5rem;
    max-width: 85%;
    line-height: 1.5;
    font-size: 0.92rem;
}

.chat-user {
    background: rgba(168,85,247,0.18);
    margin-left: auto;
    text-align: right;
}

.chat-bot {
    background: rgba(128,128,128,0.12);
    margin-right: auto;
}

/* Gradient title */
.app-title {
    font-weight: 800;
    font-size: 2.6rem;
    line-height: 1.3;
    padding-top: 0.1em;
    margin-bottom: 0;
    background: linear-gradient(90deg, #c084fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.app-subtitle {
    opacity: 0.55;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    margin-bottom: 1.5rem;
}

/* Primary button → purple gradient */
button[kind="primary"] {
    background: linear-gradient(90deg, #a855f7, #818cf8) !important;
    border: none !important;
}

/* Active tab underline → purple */
.stTabs [aria-selected="true"] {
    color: #c084fc !important;
    border-bottom-color: #c084fc !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Defaults ───────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar: input controls ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 AI Video Summarizer")
    st.caption("Transcribe, summarise, and chat with any recorded video.")
    st.divider()

    source = st.text_input(
        "YouTube URL or local file path",
        placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4",
    )
    language = st.selectbox("Spoken language", ["english", "hinglish"], index=0)
    run_btn = st.button("⚡ Analyse", use_container_width=True, type="primary")

    if st.session_state.result:
        st.divider()
        if st.button("🗑️ Clear session", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.rerun()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="app-title">AI Video Summarizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Transcribe · Summarise · Chat with your video</p>',
    unsafe_allow_html=True,
)

# ─── Run pipeline ─────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.result = None
        st.session_state.chat_history = []

        try:
            with st.status("Running pipeline…", expanded=True) as status:
                status.write("🔊 Processing audio…")
                chunks = process_input(source)

                status.write("📝 Transcribing…")
                transcript = transcribe_all(chunks, language)

                status.write("🏷️ Generating title…")
                title = generate_title(transcript)

                status.write("📋 Summarising…")
                summary = summarize(transcript)

                status.write("🔍 Extracting action items, decisions & questions…")
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

                status.write("🧠 Building chat index…")
                rag_chain = build_rag_chain(transcript)

                status.update(label="✅ Analysis complete", state="complete", expanded=False)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.rerun()

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ─── Results ──────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    st.subheader(r["title"])
    st.divider()

    tab_summary, tab_transcript, tab_items, tab_chat = st.tabs(
        ["📋 Summary", "📝 Transcript", "✅ Action Items", "💬 Chat"]
    )

    with tab_summary:
        st.markdown(f'<div class="info-card">{r["summary"]}</div>', unsafe_allow_html=True)

    with tab_transcript:
        st.text_area("Full transcript", r["transcript"], height=350, label_visibility="collapsed")

    with tab_items:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f'<div class="info-card"><h4>Action Items</h4>{r["action_items"]}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="info-card"><h4>Key Decisions</h4>{r["key_decisions"]}</div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="info-card"><h4>Open Questions</h4>{r["open_questions"]}</div>',
                unsafe_allow_html=True,
            )

    with tab_chat:
        for msg in st.session_state.chat_history:
            css_class = "chat-user" if msg["role"] == "user" else "chat-bot"
            label = "You" if msg["role"] == "user" else "🤖 Assistant"
            st.markdown(
                f'<div class="chat-msg {css_class}"><b>{label}</b><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

        if not st.session_state.chat_history:
            st.caption("Ask a question about the meeting to get started.")

        user_input = st.chat_input("Ask something about this meeting…")
        if user_input:
            with st.spinner("Thinking…"):
                answer = ask_questions(r["rag_chain"], user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

else:
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 0;">
            <div style="font-size:3rem;">🎬</div>
            <h2 style="margin-top:0.5rem;">Ready to Analyse</h2>
            <p style="opacity:0.6;">Paste a YouTube URL or local file path in the sidebar, choose your
            language, and hit <b>Analyse</b> to get started.</p>
            <div style="margin-top:1rem;">
                <span style="background:rgba(168,85,247,0.15); color:#c084fc; border:1px solid rgba(168,85,247,0.35);
                border-radius:6px; padding:0.25rem 0.7rem; font-size:0.75rem; margin-right:0.4rem;">TRANSCRIPTION</span>
                <span style="background:rgba(129,140,248,0.15); color:#a5b4fc; border:1px solid rgba(129,140,248,0.35);
                border-radius:6px; padding:0.25rem 0.7rem; font-size:0.75rem; margin-right:0.4rem;">SUMMARISATION</span>
                <span style="background:rgba(192,132,252,0.15); color:#e9d5ff; border:1px solid rgba(192,132,252,0.35);
                border-radius:6px; padding:0.25rem 0.7rem; font-size:0.75rem;">RAG CHAT</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )