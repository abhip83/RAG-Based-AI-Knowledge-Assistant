from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.cleaner import clean_text

def create_faiss_index(documents, embedding_model):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    texts = []
    for doc in documents:
        cleaned = clean_text(doc.page_content)
        texts.extend(splitter.split_text(cleaned))

    vectorstore = FAISS.from_texts(texts, embedding_model)
    return vectorstore
