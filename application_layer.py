# application_layer.py
import tempfile
import os
from pathlib import Path
import streamlit as st
from model_layer import get_llm
from data_layer import build_vector_store


@st.cache_resource(show_spinner=False)
def load_dependencies():
    """Import động để tối ưu tốc độ load app"""
    modules = {}
    try:
        from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_community.llms import Ollama
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from sentence_transformers import CrossEncoder

        modules.update({
            'PDFPlumberLoader': PDFPlumberLoader, 'Docx2txtLoader': Docx2txtLoader,
            'RecursiveCharacterTextSplitter': RecursiveCharacterTextSplitter,
            'HuggingFaceEmbeddings': HuggingFaceEmbeddings, 'FAISS': FAISS, 'Ollama': Ollama,
            'BM25Retriever': BM25Retriever, 'EnsembleRetriever': EnsembleRetriever,
            'CrossEncoder': CrossEncoder
        })
    except ImportError as e:
        print(f"Thiếu thư viện: {e}")
    return modules


mods = load_dependencies()


def process_file(uploaded_file, chunk_size, chunk_overlap) -> list:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        loader = mods['PDFPlumberLoader'](tmp_path) if suffix == '.pdf' else mods['Docx2txtLoader'](tmp_path)
        docs = loader.load()
        for doc in docs: doc.metadata['source_file'] = uploaded_file.name

        splitter = mods['RecursiveCharacterTextSplitter'](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        for i, chunk in enumerate(chunks): chunk.metadata['chunk_id'] = i
        return chunks
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)


def create_vector_store(all_chunks: list):
    return build_vector_store(mods, all_chunks)


def answer_question(query: str, vector_store, all_chunks, history, use_rerank, k, use_hybrid):
    llm = get_llm(mods)
    if not vector_store:
        prompt = f"Lịch sử: {history[-3:]}\n\nNgười dùng: {query}\nTrợ lý:"
        return {"answer": llm.invoke(prompt).strip(), "sources": []}

    # Retrieval logic
    search_kwargs = {"k": k}
    base = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    retriever = base
    if use_hybrid and all_chunks:
        bm25 = mods['BM25Retriever'].from_documents(all_chunks)
        bm25.k = k
        retriever = mods['EnsembleRetriever'](retrievers=[bm25, base], weights=[0.4, 0.6])

    relevant_docs = retriever.invoke(query)

    if use_rerank and len(relevant_docs) > 0:
        model = mods['CrossEncoder']('cross-encoder/ms-marco-MiniLM-L-6-v2')
        scores = model.predict([(query, d.page_content) for d in relevant_docs])
        ranked = sorted(zip(scores, relevant_docs), key=lambda x: x[0], reverse=True)
        relevant_docs = [d for s, d in ranked[:3]]

    context = "\n\n".join([d.page_content for d in relevant_docs])
    sources = [{"file": d.metadata.get("source_file"), "preview": d.page_content[:100]} for d in relevant_docs]

    prompt = f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}\nTrả lời:"
    return {"answer": llm.invoke(prompt).strip(), "sources": sources}