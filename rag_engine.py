"""
RAG Engine — Retrieval-Augmented Generation for ProfessorGPT
Chunks transcript, creates TF-IDF embeddings, retrieves relevant context.
"""

import re
import math
from collections import Counter


class RAGEngine:
    """
    Lightweight RAG engine using TF-IDF for semantic retrieval.
    In production, swap _embed() for sentence-transformers or OpenAI embeddings
    and use a vector DB like Pinecone or ChromaDB.
    """

    CHUNK_SIZE = 300    # words per chunk
    CHUNK_OVERLAP = 50  # word overlap between chunks
    TOP_K = 3           # chunks to retrieve per query

    def __init__(self, transcript: str):
        self.transcript = transcript
        self.chunks = self._chunk(transcript)
        self.tfidf_matrix = self._build_tfidf()

    # ── Chunking ────────────────────────────────────────────────────────────
    def _chunk(self, text: str) -> list[str]:
        """Split transcript into overlapping windows."""
        words = text.split()
        chunks = []
        step = self.CHUNK_SIZE - self.CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.CHUNK_SIZE])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    # ── TF-IDF ──────────────────────────────────────────────────────────────
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z]{2,}\b", text.lower())

    @staticmethod
    def _tf(tokens: list[str]) -> dict:
        counts = Counter(tokens)
        total = len(tokens) or 1
        return {w: c / total for w, c in counts.items()}

    def _build_tfidf(self) -> list[dict]:
        tokenized = [self._tokenize(c) for c in self.chunks]
        n = len(tokenized)
        df = Counter(w for doc in tokenized for w in set(doc))
        idf = {w: math.log(n / (1 + freq)) for w, freq in df.items()}

        matrix = []
        for tokens in tokenized:
            tf = self._tf(tokens)
            tfidf = {w: tf[w] * idf.get(w, 0) for w in tf}
            matrix.append(tfidf)
        return matrix

    @staticmethod
    def _cosine(a: dict, b: dict) -> float:
        keys = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in keys)
        mag_a = math.sqrt(sum(v ** 2 for v in a.values()))
        mag_b = math.sqrt(sum(v ** 2 for v in b.values()))
        return dot / (mag_a * mag_b + 1e-9)

    # ── Public API ──────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> str:
        """Return the most relevant chunks for a query, joined as context."""
        q_tokens = self._tokenize(query)
        q_vec = self._tf(q_tokens)

        scores = [self._cosine(q_vec, chunk_vec) for chunk_vec in self.tfidf_matrix]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.TOP_K]
        return "\n\n---\n\n".join(self.chunks[i] for i in sorted(top_indices))

    def get_stats(self) -> dict:
        return {
            "chunks": len(self.chunks),
            "vocab_size": len({w for vec in self.tfidf_matrix for w in vec}),
            "word_count": len(self.transcript.split()),
        }
