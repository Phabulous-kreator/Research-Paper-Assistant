import os

import streamlit as st
from dotenv import load_dotenv

from rag_utils import (
    load_embedding_model,
    load_qa_pipeline,
    get_openai_client,
    extract_text_from_pdfs,
    chunk_text,
    create_embeddings,
    build_faiss_index,
    retrieve_chunks,
    answer_question,
)

load_dotenv()

st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Paper Assistant")
st.write(
    "Upload one or more research papers in PDF format, then ask questions and get grounded answers."
)

openai_client = get_openai_client(os.getenv("OPENAI_API_KEY"))

if openai_client:
    st.caption("Using OpenAI to generate answers. Embeddings and retrieval still run locally.")
else:
    st.caption("Free local RAG version: no OpenAI API key required. Add one to your .env file for better answers.")

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()

@st.cache_resource
def get_qa_pipeline():
    return load_qa_pipeline()

embedding_model = get_embedding_model()
qa_pipeline = None if openai_client else get_qa_pipeline()

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "papers_loaded" not in st.session_state:
    st.session_state.papers_loaded = False

uploaded_files = st.file_uploader(
    "Upload PDF papers",
    type=["pdf"],
    accept_multiple_files=True,
)

col1, col2 = st.columns(2)

with col1:
    process_clicked = st.button("Process Papers")

with col2:
    clear_clicked = st.button("Clear Session")

if clear_clicked:
    st.session_state.chunks = None
    st.session_state.faiss_index = None
    st.session_state.papers_loaded = False
    st.rerun()

if process_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Extracting text from PDFs..."):
            pages = extract_text_from_pdfs(uploaded_files)

        if not pages:
            st.error("No readable text was extracted from the uploaded PDFs.")
            st.stop()

        with st.spinner("Chunking paper text..."):
            chunks = chunk_text(pages, chunk_size=1200, overlap=200)

        with st.spinner("Creating local embeddings and building vector index..."):
            embeddings = create_embeddings(embedding_model, chunks)
            faiss_index = build_faiss_index(embeddings)

        st.session_state.chunks = chunks
        st.session_state.faiss_index = faiss_index
        st.session_state.papers_loaded = True

        st.success(f"Processed {len(uploaded_files)} paper(s) into {len(chunks)} chunks.")

if st.session_state.papers_loaded:
    st.subheader("Ask a Question")

    question = st.text_input(
        "Enter your question",
        placeholder="What is the paper's main contribution?",
    )

    top_k = st.slider(
        "Number of source chunks to retrieve",
        min_value=3,
        max_value=8,
        value=5,
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant chunks..."):
                retrieved = retrieve_chunks(
                    embedding_model=embedding_model,
                    query=question,
                    index=st.session_state.faiss_index,
                    chunks=st.session_state.chunks,
                    top_k=top_k,
                )

            with st.spinner("Generating grounded answer..."):
                answer, context_used = answer_question(
                    qa_pipeline=qa_pipeline,
                    question=question,
                    retrieved_chunks=retrieved,
                    openai_client=openai_client,
                )

            st.subheader("Answer")
            st.write(answer)

            with st.expander("Retrieved source chunks"):
                for i, chunk in enumerate(retrieved, start=1):
                    st.markdown(
                        f"**Source {i}** — `{chunk['source_file']}` | Page **{chunk['page_number']}**"
                    )
                    st.write(chunk["text"])
                    st.divider()

            with st.expander("Context used by the QA model"):
                st.write(context_used)
else:
    st.info("Upload and process papers first, then ask questions.")