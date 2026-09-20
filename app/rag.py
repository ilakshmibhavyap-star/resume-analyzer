import os
from typing import List, Dict, Any

import faiss
import numpy as np
from google import genai
from google.genai import types


# ---- Settings (override with environment variables) -------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.6-flash")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "6"))


# ---- Gemini client -----------------------------------------------------------
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=API_KEY)


# ---- Text chunking -----------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """

    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ---- Embeddings --------------------------------------------------------------
def embed_texts(
    texts: List[str],
    batch_size: int = 100,
) -> np.ndarray:
    """
    Generate Gemini embeddings for a list of texts.

    Returns:
        numpy array with shape:
        (number_of_texts, EMBED_DIM)
    """

    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
        )

        for embedding in result.embeddings:
            embeddings.append(embedding.values)

    vectors = np.asarray(embeddings, dtype=np.float32)

    return vectors


# ---- FAISS index -------------------------------------------------------------
def build_index(chunks: List[str]):
    """
    Build a FAISS index from text chunks.

    Returns:
        FAISS index and corresponding chunks.
    """

    if not chunks:
        raise ValueError("Cannot build an index from empty chunks.")

    vectors = embed_texts(chunks)

    if vectors.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch. "
            f"Expected {EMBED_DIM}, got {vectors.shape[1]}."
        )

    # Normalize vectors so inner product behaves like cosine similarity.
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)

    return index, chunks


# ---- Retrieval ---------------------------------------------------------------
def retrieve(
    query: str,
    index,
    chunks: List[str],
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant chunks for a query.
    """

    if not query:
        return []

    if index is None or index.ntotal == 0:
        return []

    query_vector = embed_texts([query])

    faiss.normalize_L2(query_vector)

    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vector, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        results.append(
            {
                "chunk": chunks[idx],
                "score": float(score),
                "index": int(idx),
            }
        )

    return results


# ---- Prompt ------------------------------------------------------------------
def build_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Build the grounded RAG prompt.
    """

    context_parts = []

    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"--- Resume Context {i} ---\n"
            f"{item['chunk']}"
        )

    context = "\n\n".join(context_parts)

    return f"""
You are a professional resume analysis assistant.

Answer the user's question using ONLY the resume context provided below.

Rules:
1. Do not invent information that is not present in the resume.
2. If the answer cannot be determined from the provided resume context, clearly say so.
3. Be concise and specific.
4. When appropriate, mention the relevant technologies, projects, responsibilities, and experience.
5. Do not make assumptions about the candidate's experience.
6. Keep the response professional and suitable for recruiters or interviewers.

Resume Context:
{context}

User Question:
{query}
""".strip()


# ---- Generation --------------------------------------------------------------
def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Generate an answer using the retrieved resume context.
    """

    if not retrieved_chunks:
        return (
            "I could not find relevant information in the resume "
            "to answer this question."
        )

    prompt = build_prompt(query, retrieved_chunks)

    response = client.models.generate_content(
        model=GEN_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
    )

    if not response or not response.text:
        return "No answer was generated."

    return response.text.strip()


# ---- RAG pipeline ------------------------------------------------------------
def ask(
    query: str,
    index,
    chunks: List[str],
    top_k: int = TOP_K,
) -> str:
    """
    Complete RAG pipeline:

    Query
      ↓
    Embedding
      ↓
    FAISS retrieval
      ↓
    Gemini generation
      ↓
    Answer
    """

    retrieved_chunks = retrieve(
        query=query,
        index=index,
        chunks=chunks,
        top_k=top_k,
    )

    return generate_answer(
        query=query,
        retrieved_chunks=retrieved_chunks,
    )
