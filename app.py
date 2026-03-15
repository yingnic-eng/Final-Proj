"""
ProfessorGPT - AI-Powered Lecture Assistant
BANA 275 Final Project - Group 5
"""

import streamlit as st
import anthropic
import json
import re
import time
from transcript_processor import process_transcript
from rag_engine import RAGEngine

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProfessorGPT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a6b5a; }
    .sub-header  { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 20px;
        padding: 6px 18px;
        font-weight: 500;
    }
    .flashcard {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e0e0e0;
        min-height: 120px;
        cursor: pointer;
    }
    .exam-correct { background: #eaf3de; border-left: 4px solid #639922; padding: 8px 12px; border-radius: 6px; }
    .exam-wrong   { background: #fcebeb; border-left: 4px solid #e24b4a; padding: 8px 12px; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────
for key in ["transcript", "notes", "flashcards", "qa_pairs",
            "exam_questions", "chat_history", "rag_engine", "processed"]:
    if key not in st.session_state:
        st.session_state[key] = None if key not in ["chat_history"] else []
st.session_state.setdefault("processed", False)

# ── Anthropic client ────────────────────────────────────────────────────────
@st.cache_resource
def get_client():
    # Works on Streamlit Cloud (st.secrets) and locally (.env / environment variable)
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return anthropic.Anthropic(api_key=api_key)

client = get_client()

# ── Helpers ────────────────────────────────────────────────────────────────
def call_claude(prompt: str, system: str = "", max_tokens: int = 2000) -> str:
    """Call Claude API and return text response."""
    kwargs = dict(
        model="claude-sonnet-4-5",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text


def parse_json_response(text: str):
    """Safely extract JSON from a model response."""
    clean = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", clean)
        if match:
            return json.loads(match.group(0))
        raise


# ── Generation functions ───────────────────────────────────────────────────
def generate_notes(transcript: str) -> str:
    return call_claude(
        prompt=f"""Given this lecture transcript, generate comprehensive structured study notes in markdown.

Use ## for major sections, ### for subsections, bullet points for key ideas.
Be thorough — cover every topic from the lecture with clear explanations.

TRANSCRIPT:
{transcript[:4000]}""",
        system="You are an expert academic note-taker. Produce clear, organized, exam-ready notes.",
    )


def generate_flashcards(transcript: str) -> list:
    raw = call_claude(
        prompt=f"""Extract 10 key terms and concepts from this transcript and create flashcards.

Return ONLY a JSON array, no markdown, no explanation:
[{{"term": "Term name", "definition": "1-2 sentence definition"}}, ...]

TRANSCRIPT:
{transcript[:4000]}""",
    )
    return parse_json_response(raw)


def generate_qa(transcript: str) -> list:
    raw = call_claude(
        prompt=f"""Create 8 meaningful study Q&A pairs from this lecture transcript.

Return ONLY a JSON array:
[{{"question": "Question?", "answer": "Detailed 2-3 sentence answer"}}, ...]

TRANSCRIPT:
{transcript[:4000]}""",
    )
    return parse_json_response(raw)


def generate_exam(transcript: str) -> list:
    raw = call_claude(
        prompt=f"""Create a 6-question multiple choice practice exam from this lecture.

Return ONLY a JSON array:
[{{
  "question": "Question text?",
  "options": ["A. option", "B. option", "C. option", "D. option"],
  "correct": 0,
  "explanation": "Why this answer is correct."
}}, ...]

"correct" is the 0-based index of the correct option.

TRANSCRIPT:
{transcript[:4000]}""",
    )
    return parse_json_response(raw)


def rag_chat(question: str, transcript: str, history: list) -> str:
    """RAG-powered Q&A grounded in lecture transcript."""
    system = f"""You are ProfessorGPT, an AI lecture assistant.
Answer questions using ONLY the provided lecture content.
If the answer isn't in the transcript, say so honestly.

LECTURE TRANSCRIPT:
{transcript[:3500]}"""

    messages = history + [{"role": "user", "content": question}]
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=800,
        system=system,
        messages=messages,
    )
    return response.content[0].text


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 ProfessorGPT")
    st.markdown("*AI-Powered Lecture Assistant*")
    st.divider()

    if st.session_state.processed:
        word_count = len(st.session_state.transcript.split())
        st.success(f"✓ Lecture loaded\n{word_count:,} words")
        st.divider()
        st.markdown("**Quick Navigation**")
        st.markdown("📝 Notes | 🃏 Flashcards")
        st.markdown("💬 Q&A | 🤖 Chat | 📋 Exam")
    else:
        st.info("Upload a lecture to get started")

    st.divider()
    st.markdown("**About**")
    st.caption("Built with Claude AI · RAG · Embeddings · LLM")
    st.caption("BANA 275 Final Project — Group 5")


# ── Main header ────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">Professor<span style="color:#ef9f27">GPT</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Transform any lecture into smart study tools — powered by AI</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_upload, tab_notes, tab_flash, tab_qa, tab_chat, tab_exam = st.tabs([
    "📤 Upload", "📝 Notes", "🃏 Flashcards", "💬 Q&A", "🤖 Ask AI", "📋 Exam Prep"
])

# ──────────────────────────────────────────────────────────────────────────
# TAB 1 — UPLOAD
# ──────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload Your Lecture")
    col1, col2 = st.columns([3, 1])

    with col1:
        upload_mode = st.radio(
            "Input type",
            ["Paste transcript", "Upload file"],
            horizontal=True,
        )

        if upload_mode == "Paste transcript":
            transcript_text = st.text_area(
                "Lecture transcript",
                height=250,
                placeholder="Paste your lecture transcript here...",
            )
        else:
            uploaded = st.file_uploader(
                "Upload transcript (.txt) or audio (.mp3/.mp4)",
                type=["txt", "mp3", "mp4", "pdf"],
            )
            transcript_text = ""
            if uploaded:
                if uploaded.type == "text/plain":
                    transcript_text = uploaded.read().decode("utf-8")
                    st.success(f"File loaded: {uploaded.name}")
                else:
                    st.info("Audio/video transcription requires Whisper API (see SETUP.md).")

    with col2:
        st.markdown("**Sample Lectures**")
        if st.button("🧠 ML Fundamentals"):
            st.session_state["demo_transcript"] = DEMO_ML
        if st.button("📈 Finance Basics"):
            st.session_state["demo_transcript"] = DEMO_FINANCE

    if "demo_transcript" in st.session_state:
        transcript_text = st.session_state["demo_transcript"]

    if st.button("✨ Process Lecture", type="primary", use_container_width=True):
        if not transcript_text.strip():
            st.error("Please provide a transcript first.")
        else:
            with st.spinner("Processing your lecture..."):
                st.session_state.transcript = transcript_text

                progress = st.progress(0, "Generating notes...")
                st.session_state.notes = generate_notes(transcript_text)
                progress.progress(25, "Creating flashcards...")
                st.session_state.flashcards = generate_flashcards(transcript_text)
                progress.progress(50, "Building Q&A pairs...")
                st.session_state.qa_pairs = generate_qa(transcript_text)
                progress.progress(75, "Preparing exam questions...")
                st.session_state.exam_questions = generate_exam(transcript_text)
                progress.progress(100, "Done!")
                st.session_state.processed = True
                st.session_state.chat_history = []

            st.success("✅ Lecture processed! Switch to any tab to explore your study materials.")
            st.balloons()


# ──────────────────────────────────────────────────────────────────────────
# TAB 2 — NOTES
# ──────────────────────────────────────────────────────────────────────────
with tab_notes:
    st.subheader("Smart Notes")
    if not st.session_state.processed:
        st.info("Process a lecture first to generate notes.")
    else:
        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("↻ Regenerate"):
                with st.spinner("Regenerating notes..."):
                    st.session_state.notes = generate_notes(st.session_state.transcript)
            if st.download_button(
                "⬇ Download",
                data=st.session_state.notes,
                file_name="lecture_notes.md",
                mime="text/markdown",
            ):
                pass
        with col1:
            st.markdown(st.session_state.notes)


# ──────────────────────────────────────────────────────────────────────────
# TAB 3 — FLASHCARDS
# ──────────────────────────────────────────────────────────────────────────
with tab_flash:
    st.subheader("Flashcards")
    if not st.session_state.processed:
        st.info("Process a lecture first to generate flashcards.")
    else:
        if st.button("↻ Regenerate Flashcards"):
            with st.spinner("Regenerating..."):
                st.session_state.flashcards = generate_flashcards(st.session_state.transcript)

        cards = st.session_state.flashcards or []
        st.caption(f"{len(cards)} cards generated")

        cols = st.columns(2)
        for i, card in enumerate(cards):
            with cols[i % 2]:
                with st.expander(f"**{card['term']}**", expanded=False):
                    st.write(card["definition"])


# ──────────────────────────────────────────────────────────────────────────
# TAB 4 — Q&A
# ──────────────────────────────────────────────────────────────────────────
with tab_qa:
    st.subheader("Practice Q&A")
    if not st.session_state.processed:
        st.info("Process a lecture first to generate Q&A pairs.")
    else:
        if st.button("↻ Regenerate Q&A"):
            with st.spinner("Regenerating..."):
                st.session_state.qa_pairs = generate_qa(st.session_state.transcript)

        pairs = st.session_state.qa_pairs or []
        st.caption(f"{len(pairs)} questions generated")

        for i, pair in enumerate(pairs, 1):
            with st.expander(f"**Q{i}: {pair['question']}**"):
                st.markdown(f"**Answer:** {pair['answer']}")


# ──────────────────────────────────────────────────────────────────────────
# TAB 5 — CHAT
# ──────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask AI — RAG-Powered Chat")
    if not st.session_state.processed:
        st.info("Process a lecture first to enable lecture-grounded chat.")
    else:
        # Display history
        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.write(msg["content"])

        # Input
        user_input = st.chat_input("Ask anything about your lecture...")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = rag_chat(
                        user_input,
                        st.session_state.transcript,
                        st.session_state.chat_history,
                    )
                st.write(answer)

            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────
# TAB 6 — EXAM
# ──────────────────────────────────────────────────────────────────────────
with tab_exam:
    st.subheader("Practice Exam")
    if not st.session_state.processed:
        st.info("Process a lecture first to generate a practice exam.")
    else:
        if st.button("↻ New Exam"):
            with st.spinner("Generating new exam..."):
                st.session_state.exam_questions = generate_exam(st.session_state.transcript)
            st.rerun()

        questions = st.session_state.exam_questions or []
        answers = {}
        letter = ["A", "B", "C", "D"]

        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")
            opts = [opt.replace(f"{letter[j]}. ", "") for j, opt in enumerate(q["options"])]
            answers[i] = st.radio(
                f"q{i}",
                options=range(len(opts)),
                format_func=lambda x, opts=opts: f"{letter[x]}. {opts[x]}",
                label_visibility="collapsed",
                key=f"exam_q_{i}",
            )
            st.divider()

        if st.button("📊 Grade My Exam", type="primary"):
            score = sum(1 for i, q in enumerate(questions) if answers.get(i) == q["correct"])
            pct = round(score / len(questions) * 100)
            emoji = "🎉" if pct >= 80 else "📚" if pct >= 60 else "💪"
            st.metric("Score", f"{score}/{len(questions)} ({pct}%)", f"{emoji}")

            st.subheader("Review")
            for i, q in enumerate(questions):
                chosen = answers.get(i)
                correct = q["correct"]
                if chosen == correct:
                    st.markdown(f'<div class="exam-correct">✓ Q{i+1}: Correct!</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="exam-wrong">✗ Q{i+1}: You chose {letter[chosen] if chosen is not None else "?"} — '
                        f'Correct: {letter[correct]}. {q.get("explanation", "")}</div>',
                        unsafe_allow_html=True,
                    )
                st.write("")


# ── Demo transcripts (truncated inline) ───────────────────────────────────
DEMO_ML = """Welcome to today's lecture on Machine Learning Fundamentals.
Machine learning is a subset of artificial intelligence that allows systems to learn and improve from experience without being explicitly programmed.
Supervised Learning: Models train on labeled datasets. Linear regression predicts continuous values; logistic regression performs classification using the sigmoid function.
The bias-variance tradeoff: high bias = underfitting, high variance = overfitting. Regularization (L1 Lasso, L2 Ridge) controls overfitting.
Unsupervised Learning: K-means clustering partitions data. PCA reduces dimensionality by projecting onto directions of maximum variance.
Neural Networks: Layers of connected nodes. Backpropagation computes gradients. ReLU activation introduces non-linearity. Deep learning = many hidden layers.
Evaluation: accuracy, precision, recall, F1-score, AUC-ROC. Always use train/validation/test splits and cross-validation."""

DEMO_FINANCE = """Today's lecture covers Personal Finance Fundamentals.
Budgeting: The 50/30/20 rule allocates 50% to needs, 30% to wants, 20% to savings. Zero-based budgeting assigns every dollar a job.
Compound Interest: A = P(1+r/n)^(nt). Starting early is the single biggest factor in wealth building. Einstein called it the eighth wonder of the world.
Investment Vehicles: 401k offers employer matching (free money). Roth IRA uses post-tax dollars for tax-free growth. Index funds track the market with low fees.
Risk and Return: Higher risk typically means higher potential return. Diversification reduces unsystematic risk. Modern Portfolio Theory optimizes the risk-return tradeoff.
Debt Management: Avalanche method pays high-interest debt first. Snowball method pays smallest balances first for psychological momentum.
Emergency Fund: 3-6 months of expenses in a liquid account. This is the foundation of financial security."""
