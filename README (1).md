# ProfessorGPT — AI-Powered Lecture Assistant

Transform any lecture transcript into smart study tools: structured notes, flashcards, Q&A, RAG-powered chat, and practice exams. This repository includes the **ProfessorGPT Streamlit app** and a standalone **note generator** CLI that outputs Word (.docx) notes using an external LLM.

For step-by-step usage of the app (tabs, workflow, tips), see **[USER_GUIDE.md](USER_GUIDE.md)**. For installation and deployment, see **[SETUP.md](SETUP.md)**.

Youtube Demo Link: https://www.youtube.com/watch?v=fjgACeMp9Lo
---

## Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key**  
   Create a `.env` file with your Anthropic API key (see SETUP.md).

3. **Run the app**
   ```bash
   streamlit run app.py
   ```
   The app opens at **http://localhost:8501**. Use the **Upload** tab to paste or upload a transcript, then explore the **Notes**, **Flashcards**, **Q&A**, **Ask AI**, and **Exam Prep** tabs. Full workflow is described in [USER_GUIDE.md](USER_GUIDE.md).

---

## Python Code in This Directory

| File | Purpose |
|------|--------|
| **`app.py`** | ProfessorGPT Streamlit app: six-tab UI (Upload, Notes, Flashcards, Q&A, Ask AI, Exam Prep), Claude-based generation (notes, flashcards, Q&A, exam), and RAG-style chat grounded in the lecture. |
| **`rag_engine.py`** | RAG engine: chunks the transcript, builds TF-IDF vectors, and retrieves top-k relevant chunks by cosine similarity for context injection (e.g. for chat). |
| **`transcript_processor.py`** | Text cleaning and normalization (Unicode, caption artifacts, timestamps, whitespace). Optional Whisper-based audio transcription. |
| **`note_generator.py`** | Standalone CLI that generates a Word (.docx) note from a transcription file using an external LLM (OpenAI-compatible API). |

### `app.py` — Main application

- **UI:** Tab-based layout, session state, regenerate and download actions.
- **Generation:** `generate_notes`, `generate_flashcards`, `generate_qa`, `generate_exam` call the Claude API with structured prompts; outputs are cached in session state.
- **Chat:** `rag_chat` answers questions using the current transcript as context (lecture-grounded Q&A).
- **Imports:** `transcript_processor.process_transcript`, `rag_engine.RAGEngine` (RAG engine is available for retrieval; chat currently uses full transcript in the prompt).

### `rag_engine.py` — Retrieval

- **RAGEngine:** Chunks transcript (e.g. 300-word windows, 50-word overlap), builds TF-IDF matrix, exposes `retrieve(query)` returning the most relevant chunks.
- **Use case:** Feed retrieved context into the LLM for grounded answers. Can be swapped for dense embeddings + vector DB later without changing the app interface.

### `transcript_processor.py` — Preprocessing

- **process_transcript(raw):** Cleans and normalizes raw text (Unicode, `[Music]`, `(applause)`, timestamps, extra spaces/newlines).
- **transcribe_audio(file_path, api_key):** Optional; transcribes audio/video via OpenAI Whisper and runs the result through `process_transcript`.

### `note_generator.py` — Standalone CLI

Generates a Word (.docx) note from a single input file. **Separate from the Streamlit app:** uses your own API key and OpenAI-compatible endpoint.

**Usage:**
```bash
python note_generator.py <input_file> --api-key <KEY> [-o output.docx] [--model MODEL]
```

- **Input file:** Path to `.txt`, `.pdf`, or `.json` transcription (required).
- **API key:** Required. Use `--api-key KEY` or `-k KEY` (e.g. OpenAI API key). Set `OPENAI_API_BASE` in the environment for an OpenAI-compatible endpoint.
- **Output:** Default: `<input_name>_note.docx`. Use `-o path/to/file.docx` for a custom path.
- **Model:** Optional `--model` (default: `gpt-4o-mini`).

**Examples:**
```bash
python note_generator.py transcript.txt --api-key sk-...
python note_generator.py slides.pdf -k sk-... -o my_notes.docx
python note_generator.py "C:\path\to\transcript.json" --api-key sk-... --model gpt-4o
```

**Output structure (docx):** Summary, Main Points (bullets), Detailed Notes.

---

## Output Overview

- **ProfessorGPT app:** Notes (Markdown, downloadable), 10 flashcards, 8 Q&A pairs, 6-question practice exam, and an AI chat that uses your lecture content. See [USER_GUIDE.md](USER_GUIDE.md) for details.
- **note_generator.py:** Single Word document with Summary, Main Points, and Detailed Notes.

---

## Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** — How to use ProfessorGPT (tabs, workflow, tips, FAQ).
- **[SETUP.md](SETUP.md)** — Installation, `.env`, and deployment (Streamlit Cloud, Railway, Docker).
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — System design, pipeline, and components.
