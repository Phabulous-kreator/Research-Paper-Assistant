import os
import tempfile
from typing import List, Dict, Tuple

import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL_NAME = "deepset/roberta-base-squad2"
CHAT_MODEL_NAME = "gpt-4.1-mini"


def get_openai_client(api_key: str):
    """
    Build an OpenAI client if an API key was provided, otherwise None.
    """
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def load_embedding_model() -> SentenceTransformer:
    """
    Load the local embedding model.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def load_qa_pipeline():
    """
    Load the local question-answering pipeline.
    """
    return pipeline("question-answering", model=QA_MODEL_NAME, tokenizer=QA_MODEL_NAME)


def extract_text_from_pdfs(uploaded_files) -> List[Dict]:
    """
    Extract text from uploaded PDF files.

    Returns:
    [
        {
            "source_file": "paper.pdf",
            "page_number": 1,
            "text": "..."
        }
    ]
    """
    pages = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        try:
            reader = PdfReader(temp_path)

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()

                if text:
                    pages.append(
                        {
                            "source_file": uploaded_file.name,
                            "page_number": i + 1,
                            "text": text,
                        }
                    )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return pages


def chunk_text(
    pages: List[Dict],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> List[Dict]:
    """
    Split page text into overlapping character chunks.
    """
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(
                    {
                        "source_file": page["source_file"],
                        "page_number": page["page_number"],
                        "text": chunk,
                    }
                )

            if end == text_len:
                break

            start += chunk_size - overlap

    return chunks


def create_embeddings(
    embedding_model: SentenceTransformer,
    chunks: List[Dict],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Create embeddings for all chunks locally.
    """
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index using cosine similarity style search
    via inner product on normalized embeddings.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def embed_query(
    embedding_model: SentenceTransformer,
    query: str,
) -> np.ndarray:
    """
    Embed a user query locally.
    """
    vector = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.array(vector, dtype="float32")


def retrieve_chunks(
    embedding_model: SentenceTransformer,
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks.
    """
    query_vector = embed_query(embedding_model, query)
    scores, indices = index.search(query_vector, top_k)

    retrieved = []
    for idx in indices[0]:
        if idx != -1:
            retrieved.append(chunks[idx])

    return retrieved


def build_context(retrieved_chunks: List[Dict], max_context_chars: int = 3500) -> str:
    """
    Build a context string from retrieved chunks, capped to a safe size
    for the local QA model.
    """
    context_parts = []
    current_length = 0

    for i, chunk in enumerate(retrieved_chunks, start=1):
        piece = (
            f"[Source {i}] File: {chunk['source_file']} | Page: {chunk['page_number']}\n"
            f"{chunk['text']}\n\n"
        )

        if current_length + len(piece) > max_context_chars:
            break

        context_parts.append(piece)
        current_length += len(piece)

    return "".join(context_parts).strip()


def answer_question(
    qa_pipeline,
    question: str,
    retrieved_chunks: List[Dict],
    openai_client: OpenAI = None,
) -> Tuple[str, str]:
    """
    Answer a question from retrieved context.

    Uses OpenAI for a more fluent, generative answer when a client is
    provided (i.e. OPENAI_API_KEY is set); otherwise falls back to the
    local extractive QA model.

    Returns:
    - answer text
    - context used
    """
    context = build_context(retrieved_chunks)

    if not context.strip():
        return "I could not find relevant information in the uploaded papers.", context

    if openai_client is not None:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a research paper assistant. "
                        "Answer only from the provided context. "
                        "If the answer is not in the context, say you could not find it in the uploaded papers. "
                        "Be clear, accurate, and concise. "
                        "When useful, mention source file names and page numbers."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nContext:\n{context}",
                },
            ],
        )
        answer = response.choices[0].message.content.strip()
        return answer, context

    result = qa_pipeline(
        question=question,
        context=context,
    )

    answer = result.get("answer", "").strip()
    score = result.get("score", 0.0)

    if not answer or score < 0.1:
        return "I could not confidently find the answer in the uploaded papers.", context

    return answer, context