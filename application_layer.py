import re
import tempfile
import os
import math
import time
from pathlib import Path
import streamlit as st
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
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
    """
    Xử lý tệp tin đầu vào với cơ chế Hybrid OCR để đọc ảnh và con dấu.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # Khởi tạo loader mặc định
        loader = mods['PDFPlumberLoader'](tmp_path) if suffix == '.pdf' else mods['Docx2txtLoader'](tmp_path)
        docs = loader.load()

        # Xử lý nâng cao cho định dạng PDF (Xử lý ảnh và con dấu)
        if suffix == '.pdf':
            # Chuyển đổi toàn bộ trang PDF thành hình ảnh để sẵn sàng OCR
            pages_as_images = convert_from_path(tmp_path)

            for i, doc in enumerate(docs):
                # Ngưỡng kiểm tra: Nếu trang có ít hơn 100 ký tự văn bản số, khả năng cao là ảnh hoặc con dấu
                is_likely_image_or_seal = len(doc.page_content.strip()) < 100

                if is_likely_image_or_seal:
                    # Sử dụng Tesseract để quét chữ từ con dấu/hình ảnh
                    ocr_text = pytesseract.image_to_string(pages_as_images[i], lang='vie+eng')
                    if ocr_text.strip():
                        doc.page_content += f"\n[Nội dung từ ảnh/con dấu]:\n{ocr_text}"

                doc.metadata['source_file'] = uploaded_file.name
                doc.metadata['page'] = i
        else:
            for doc in docs:
                doc.metadata['source_file'] = uploaded_file.name

        # Chia nhỏ văn bản thành các Chunks
        splitter = mods['RecursiveCharacterTextSplitter'](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i

        return chunks

    except Exception as e:
        print(f"Lỗi khi xử lý tệp: {e}")
        return []
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


def rerank_with_cross_encoder(query: str, documents: list, k: int) -> tuple:
    model = mods['CrossEncoder']('cross-encoder/ms-marco-MiniLM-L-6-v2')
    t_start = time.time()
    pairs = [(query, doc.page_content) for doc in documents]
    ce_scores = model.predict(pairs)
    elapsed_ms = round((time.time() - t_start) * 1000)

    ranked = sorted(zip(ce_scores, documents), key=lambda x: x[0], reverse=True)[:k]
    reranked_docs = [d for _, d in ranked]
    doc_scores = [_sigmoid(float(s)) for s, _ in ranked]

    return reranked_docs, doc_scores, elapsed_ms


def rewrite_query(query: str, llm) -> str:
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
        if "NOT_SUPPORTED" in raw:
            label = "NOT_SUPPORTED"
        elif "PARTIALLY" in raw:
            label = "PARTIALLY_SUPPORTED"
        else:
            label = "SUPPORTED"
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


def answer_question(query: str, vector_store, all_chunks, history, use_rerank, k, use_hybrid):
    llm = get_llm(mods)
    use_self_rag = st.session_state.get('use_self_rag', False)

    if not vector_store:
        prompt = f"Bạn là trợ lý AI chuyên nghiệp. Yêu cầu bắt buộc: Chỉ sử dụng tiếng Việt.\n\nLịch sử: {history[-3:]}\n\nNgười dùng: {query}\nTrợ lý:"
        return {
            "answer": llm.invoke(prompt).strip(),
            "sources": [],
            "latency": None,
            "self_rag": None,
        }

    original_query = query
    search_query = rewrite_query(query, llm) if use_self_rag else query

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

    rerank_ms = 0
    if use_rerank and relevant_docs:
        relevant_docs, doc_scores, rerank_ms = rerank_with_cross_encoder(search_query, relevant_docs, k)

    context = "\n\n".join([f"[{i + 1}] {d.page_content}" for i, d in enumerate(relevant_docs)])

    sources = [{
        "file": d.metadata.get("source_file", "Unknown"),
        "page": (d.metadata.get("page", 0) or 0) + 1,
        "chunk_id": d.metadata.get("chunk_id", 0),
        "preview": _clean_text(d.page_content[:150]),
        "full_content": _clean_text(d.page_content),
        "query": search_query,
        "score": doc_scores[i] if i < len(doc_scores) else 0.8,
    } for i, d in enumerate(relevant_docs)]

    hist_lines = [f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content'][:200]}" for m in history[-3:]]
    hist_block = ("\nLịch sử hội thoại:\n" + "\n".join(hist_lines) + "\n") if hist_lines else ""

    prompt = f"""Bạn là trợ lý AI thông minh hỗ trợ đọc hiểu tài liệu.
Quy tắc:
- Trả lời súc tích, trích nguồn dạng [1], [2]
- Ưu tiên thông tin từ hình ảnh/con dấu nếu có nhãn [Nội dung từ ảnh/con dấu]
- Nếu tài liệu không đủ thông tin, nói rõ

{hist_block}
Tài liệu tham khảo:
{context}

Câu hỏi: {query}
Trả lời:"""

    answer = llm.invoke(prompt).strip()
    for src in sources:
        src["answer"] = answer
    self_rag_meta = {
        "original_query": original_query,
        "rewritten_query": search_query,
        "was_rewritten": search_query != original_query,
        "evaluation": self_evaluate(query, context, answer, llm),
    } if use_self_rag else None

    latency_meta = {
        "retrieve_ms": retrieve_ms,
        "rerank_ms": rerank_ms,
        "total_ms": retrieve_ms + rerank_ms,
        "reranking_used": use_rerank,
    }

    return {"answer": answer, "sources": sources, "latency": latency_meta, "self_rag": self_rag_meta}