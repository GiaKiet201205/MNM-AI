# application_layer.py
import re
import tempfile
import os
import math
import time
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
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
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


# ═══════════════════════════════════════════════════════════════════
#  CÂU 9 — RE-RANKING VỚI CROSS-ENCODER
# ═══════════════════════════════════════════════════════════════════

def rerank_with_cross_encoder(query: str, documents: list, k: int) -> tuple:
    """
    Câu 9: Re-ranking với Cross-Encoder.

    So sánh với Bi-Encoder (FAISS):
    - Bi-Encoder: embed query và chunk RIÊNG LẺ → cosine similarity → nhanh nhưng kém chính xác
    - Cross-Encoder: đọc ĐỒNG THỜI query + chunk → hiểu ngữ cảnh sâu hơn → chính xác hơn

    Returns:
        (docs đã rerank, scores, thời gian ms)
    """
    model = mods['CrossEncoder']('cross-encoder/ms-marco-MiniLM-L-6-v2')

    t_start = time.time()
    pairs = [(query, doc.page_content) for doc in documents]
    ce_scores = model.predict(pairs)
    elapsed_ms = round((time.time() - t_start) * 1000)

    ranked = sorted(zip(ce_scores, documents), key=lambda x: x[0], reverse=True)[:k]
    reranked_docs = [d for _, d in ranked]
    doc_scores = [_sigmoid(float(s)) for s, _ in ranked]

    return reranked_docs, doc_scores, elapsed_ms


# ═══════════════════════════════════════════════════════════════════
#  CÂU 10 — SELF-RAG
# ═══════════════════════════════════════════════════════════════════

def rewrite_query(query: str, llm) -> str:
    """
    Câu 10 - Bước 1: Query Rewriting.

    Vấn đề: Câu hỏi mơ hồ như "nó hoạt động thế nào?" → FAISS tìm sai.
    Giải pháp: LLM tự cải thiện câu hỏi thành dạng rõ ràng hơn trước khi search.

    Ví dụ:
        "nó hoạt động thế nào?" → "Hệ thống RAG trong SmartDoc AI hoạt động như thế nào?"
        "so sánh 2 cái đó"      → "So sánh Bi-Encoder và Cross-Encoder về độ chính xác"
    """
    prompt = f"""Viết lại câu hỏi sau thành dạng rõ ràng và cụ thể hơn để tìm kiếm trong tài liệu đạt kết quả tốt nhất.
Chỉ trả về câu hỏi đã viết lại, không giải thích thêm bất kỳ điều gì.

Câu hỏi gốc: {query}
Câu hỏi viết lại:"""
    try:
        result = llm.invoke(prompt).strip()
        if result and 5 < len(result) < 400:
            return result
    except Exception:
        pass
    return query


def self_evaluate(query: str, context: str, answer: str, llm) -> dict:
    """
    Câu 10 - Bước 2: Self-Evaluation + Confidence Scoring.

    LLM tự đánh giá câu trả lời có thực sự dựa trên context lấy từ tài liệu không.

    3 mức:
        SUPPORTED (90%)           → Hoàn toàn từ tài liệu
        PARTIALLY_SUPPORTED (55%) → Một phần từ tài liệu
        NOT_SUPPORTED (20%)       → Không có trong tài liệu
    """
    prompt = f"""Đánh giá xem câu trả lời bên dưới có được hỗ trợ bởi ngữ cảnh không.
Chỉ trả về đúng 1 trong 3 giá trị sau, không giải thích:
SUPPORTED
PARTIALLY_SUPPORTED
NOT_SUPPORTED

Câu hỏi: {query}
Ngữ cảnh: {context[:600]}
Câu trả lời: {answer[:300]}

Đánh giá:"""
    try:
        raw = llm.invoke(prompt).strip().upper()
        if "NOT_SUPPORTED" in raw or ("NOT" in raw and "SUPPORT" in raw):
            label = "NOT_SUPPORTED"
        elif "PARTIALLY" in raw or "PARTIAL" in raw:
            label = "PARTIALLY_SUPPORTED"
        elif "SUPPORTED" in raw:
            label = "SUPPORTED"
        else:
            label = "PARTIALLY_SUPPORTED"
    except Exception:
        label = "PARTIALLY_SUPPORTED"

    score_map = {"SUPPORTED": 0.90, "PARTIALLY_SUPPORTED": 0.55, "NOT_SUPPORTED": 0.20}
    icon_map = {"SUPPORTED": "🟢", "PARTIALLY_SUPPORTED": "🟡", "NOT_SUPPORTED": "🔴"}
    label_vn = {
        "SUPPORTED": "Có cơ sở từ tài liệu",
        "PARTIALLY_SUPPORTED": "Một phần từ tài liệu",
        "NOT_SUPPORTED": "Không tìm thấy trong tài liệu",
    }
    return {
        "label": label,
        "label_vn": label_vn[label],
        "icon": icon_map[label],
        "score": score_map[label],
    }


# ═══════════════════════════════════════════════════════════════════
#  HÀM CHÍNH — Answer the question
# ═══════════════════════════════════════════════════════════════════

def answer_question(query: str, vector_store, all_chunks, history, use_rerank, k, use_hybrid):
    """
    Pipeline RAG tích hợp Câu 9 & 10.
    Chữ ký hàm GIỮ NGUYÊN so với app.py → không conflict khi merge.

    Câu 9: Cross-Encoder reranking + đo latency
    Câu 10: Query rewriting + Self-evaluation (đọc từ session_state)
    """
    llm = get_llm(mods)

    # Đọc trạng thái Self-RAG từ session_state (toggle trong sidebar)
    use_self_rag = st.session_state.get('use_self_rag', False)

    # ── Không có file → chat thông thường ────────────────────────
    if not vector_store:
        prompt = f"Lịch sử: {history[-3:]}\n\nNgười dùng: {query}\nTrợ lý:"
        return {
            "answer": llm.invoke(prompt).strip(),
            "sources": [],
            "latency": None,
            "self_rag": None,
        }

    # ── CÂU 10 Bước 1: Query Rewriting ───────────────────────────
    original_query = query
    search_query = query

    if use_self_rag:
        search_query = rewrite_query(query, llm)

    # ── Retrieve — đo thời gian Bi-Encoder ───────────────────────
    if use_hybrid and all_chunks:
        base = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        bm25 = mods['BM25Retriever'].from_documents(all_chunks)
        bm25.k = k
        retriever = mods['EnsembleRetriever'](retrievers=[bm25, base], weights=[0.4, 0.6])
        t0 = time.time()
        relevant_docs = retriever.invoke(search_query)
        retrieve_ms = round((time.time() - t0) * 1000)
        doc_scores = [0.82] * len(relevant_docs)
    else:
        t0 = time.time()
        scored = vector_store.similarity_search_with_score(search_query, k=k)
        retrieve_ms = round((time.time() - t0) * 1000)
        relevant_docs = [doc for doc, _ in scored]
        doc_scores = [float(1.0 / (1.0 + dist)) for _, dist in scored]

    # ── CÂU 9: Re-ranking với Cross-Encoder ──────────────────────
    rerank_ms = 0
    if use_rerank and relevant_docs:
        relevant_docs, doc_scores, rerank_ms = rerank_with_cross_encoder(search_query, relevant_docs, k)

    # ── Build numbered context (enables [1][2] inline citations) ───────
    context = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(relevant_docs)])

    sources = [{
        "file": d.metadata.get("source_file", "Unknown"),
        "page": (d.metadata.get("page", 0) or 0) + 1,
        "chunk_id": d.metadata.get("chunk_id", 0),
        "preview": _clean_text(d.page_content[:150]),
        "full_content": _clean_text(d.page_content),
        "query": search_query,
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

    # ── CÂU 10 Bước 2: Self-Evaluation ───────────────────────────
    self_rag_meta = None
    if use_self_rag:
        evaluation = self_evaluate(query, context, answer, llm)
        self_rag_meta = {
            "original_query": original_query,
            "rewritten_query": search_query,
            "was_rewritten": search_query != original_query,
            "evaluation": evaluation,
        }

    # ── Latency metadata (Câu 9) ──────────────────────────────────
    latency_meta = {
        "retrieve_ms": retrieve_ms,
        "rerank_ms": rerank_ms,
        "total_ms": retrieve_ms + rerank_ms,
        "reranking_used": use_rerank,
    }

    return {
        "answer": answer,
        "sources": sources,
        "latency": latency_meta,
        "self_rag": self_rag_meta,
    }
