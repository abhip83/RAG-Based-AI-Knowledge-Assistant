import streamlit as st
import os

# from app.config import TOP_K
from ingestion.loader import load_document
from embeddings.embedder import get_embedding_model
from vectorstore.faiss_db import create_faiss_index
from retrieval.retriever import retrieve_docs
from llm.generator import generate_answer
from evaluation.metrics import context_precision, faithfulness

st.set_page_config(page_title="RAG AI Assistant")

st.title("📄 RAG-Based AI Knowledge Assistant")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    file_path = f"data/raw_docs/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    documents = load_document(file_path)
    embedding_model = get_embedding_model()
    vectorstore = create_faiss_index(documents, embedding_model)

    query = st.text_input("Ask a question")

    if query:
        retrieved_docs = retrieve_docs(vectorstore, query, TOP_K)
        context = " ".join([doc.page_content for doc in retrieved_docs])

        answer = generate_answer(context, query)

        st.subheader("🤖 Answer")
        st.write(answer)

        # Evaluation
        precision = context_precision(
            [doc.page_content for doc in retrieved_docs],
            query,
            embedding_model
        )

        faith = faithfulness(answer, context, embedding_model)

        st.subheader("📊 Evaluation Metrics")
        st.write(f"Context Precision: {precision:.3f}")
        st.write(f"Faithfulness Score: {faith:.3f}")
