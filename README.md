# Note Generator

Generates a comprehensive Word (.docx) note from a lecture or speech transcription using an **external LLM** (e.g. OpenAI). Pass the **input file path** and **API key** together on the command line.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python note_generator.py <input_file> --api-key <KEY> [-o output.docx] [--model MODEL]
```

- **Input file:** Path to `.txt`, `.pdf`, or `.json` transcription (required).
- **API key:** Required. Use `--api-key KEY` or `-k KEY` (e.g. OpenAI API key). Set `OPENAI_API_BASE` in the environment for an OpenAI-compatible endpoint.
- **Output:** By default, saved as `<input_name>_note.docx`. Use `-o path/to/file.docx` for a custom path.
- **Model:** Optional `--model` (default: `gpt-4o-mini`).

## Examples

```bash
python note_generator.py transcript.txt --api-key sk-...
python note_generator.py slides.pdf -k sk-... -o my_notes.docx
python note_generator.py "C:\path\to\transcript.json" --api-key sk-... --model gpt-4o
```

## Output structure

The LLM-generated document includes:

1. **Summary** – Concise overview of the content.
2. **Main Points** – Bulleted key takeaways.
3. **Detailed Notes** – Structured notes suitable for study.
