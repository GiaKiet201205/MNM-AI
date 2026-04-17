# application_layer.py
import re
import tempfile
import os
import math
from pathlib import Path
import streamlit as st
from model_layer import get_llm
from data_layer import build_vector_store


@st.cache_resource(show_spinner=False)
def load_dependencies():
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


def _clean_text(text: str) -> str:
    """Normalize PDF-extracted text: fix spacing, line breaks, hyphenation."""
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)   # join hyphenated line-breaks
    text = re.sub(r'[ \t]{2,}', ' ', text)           # collapse multiple spaces/tabs
    text = re.sub(r'\s*\n\s*', '\n', text)            # trim spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text)            # max 2 consecutive newlines
    return text.strip()


def process_file(uploaded_file, chunk_size, chunk_overlap) -> list:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        loader = mods['PDFPlumberLoader'](tmp_path) if suffix == '.pdf' else mods['Docx2txtLoader'](tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source_file'] = uploaded_file.name
        splitter = mods['RecursiveCharacterTextSplitter'](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
        return chunks
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def create_vector_store(all_chunks: list):
    return build_vector_store(mods, all_chunks)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def answer_question(query: str, vector_store, all_chunks, history, use_rerank, k, use_hybrid):
    llm = get_llm(mods)

    if not vector_store:
        prompt = f"Lịch sử: {history[-3:]}\n\nNgười dùng: {query}\nTrợ lý:"
        return {"answer": llm.invoke(prompt).strip(), "sources": []}

    # ── Retrieval ──────────────────────────────────────────────────────
    if use_hybrid and all_chunks:
        base = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        bm25 = mods['BM25Retriever'].from_documents(all_chunks)
        bm25.k = k
        retriever = mods['EnsembleRetriever'](retrievers=[bm25, base], weights=[0.4, 0.6])
        relevant_docs = retriever.invoke(query)
        doc_scores = [0.82] * len(relevant_docs)
    else:
        scored = vector_store.similarity_search_with_score(query, k=k)
        relevant_docs = [doc for doc, _ in scored]
        doc_scores = [float(1.0 / (1.0 + dist)) for _, dist in scored]

    if use_rerank and len(relevant_docs) > 0:
        model = mods['CrossEncoder']('cross-encoder/ms-marco-MiniLM-L-6-v2')
        ce_scores = model.predict([(query, d.page_content) for d in relevant_docs])
        pairs = sorted(zip(ce_scores, relevant_docs), key=lambda x: x[0], reverse=True)[:k]
        relevant_docs = [d for _, d in pairs]
        doc_scores = [_sigmoid(float(s)) for s, _ in pairs]

    # ── Build numbered context (enables [1][2] inline citations) ───────
    context = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(relevant_docs)])

    # ── Sources with cleaned text ──────────────────────────────────────
    sources = [{
        "file": d.metadata.get("source_file", "Unknown"),
        "page": (d.metadata.get("page", 0) or 0) + 1,
        "chunk_id": d.metadata.get("chunk_id", 0),
        "preview": _clean_text(d.page_content[:150]),
        "full_content": _clean_text(d.page_content),
        "query": query,
        "score": doc_scores[i] if i < len(doc_scores) else 0.8,
    } for i, d in enumerate(relevant_docs)]

    # ── Prompt ─────────────────────────────────────────────────────────
    hist_lines = [
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content'][:300]}"
        for m in (history[-4:-1] or [])
    ]
    hist_block = ("\nLịch sử hội thoại:\n" + "\n".join(hist_lines) + "\n") if hist_lines else ""

    prompt = f"""Bạn là trợ lý AI thông minh hỗ trợ đọc hiểu tài liệu.

Quy tắc trả lời:
- Trả lời súc tích, đúng trọng tâm, KHÔNG chép nguyên văn tài liệu
- Dùng bullet (–) hoặc đánh số khi liệt kê nhiều ý
- Sau mỗi thông tin, trích nguồn bằng [1], [2], [3] tương ứng với đoạn tài liệu
- Trả lời bằng ngôn ngữ của câu hỏi
- Nếu tài liệu không đủ thông tin, nói rõ
{hist_block}
Tài liệu tham khảo:
{context}

Câu hỏi: {query}
Trả lời:"""

    answer = llm.invoke(prompt).strip()
    for src in sources:
        src["answer"] = answer
    return {"answer": answer, "sources": sources}


def get_chunk_stats(chunks: list) -> dict:
    if not chunks:
        return {"num_chunks": 0, "avg_length": 0, "total_chars": 0, "min_length": 0, "max_length": 0}
    lengths = [len(c.page_content) for c in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_length": round(sum(lengths) / len(lengths)),
        "total_chars": sum(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }
