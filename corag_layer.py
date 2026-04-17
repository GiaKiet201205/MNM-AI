# corag_layer.py
# CO-RAG: Corrective Retrieval-Augmented Generation
# Nâng cấp so với RAG thường: đánh giá, sửa lỗi, và tinh chỉnh retrieval tự động

import re
from typing import Optional


# ─── BƯỚC 1: ĐÁNH GIÁ ĐỘ LIÊN QUAN (Relevance Grader) ─────────────────────
def grade_documents(llm, query: str, docs: list) -> dict:
    """
    Chấm điểm từng document xem có thực sự liên quan đến query không.
    Đây là điểm khác biệt cốt lõi so với RAG thường (RAG thường lấy top-k mà không kiểm tra).
    """
    graded = []
    relevant_count = 0

    for doc in docs:
        prompt = f"""Đánh giá xem đoạn văn bản dưới đây có liên quan đến câu hỏi không.
Chỉ trả lời "YES" hoặc "NO", không giải thích thêm.

Câu hỏi: {query}
Đoạn văn: {doc.page_content[:300]}

Trả lời:"""
        try:
            result = llm.invoke(prompt).strip().upper()
            is_relevant = result.startswith("YES")
        except Exception:
            is_relevant = True  # fallback: coi là liên quan nếu lỗi

        graded.append({"doc": doc, "relevant": is_relevant})
        if is_relevant:
            relevant_count += 1

    relevance_ratio = relevant_count / len(docs) if docs else 0
    return {
        "graded_docs": graded,
        "relevant_count": relevant_count,
        "total_count": len(docs),
        "relevance_ratio": relevance_ratio,
        "quality": "high" if relevance_ratio >= 0.6 else ("medium" if relevance_ratio >= 0.3 else "low")
    }


# ─── BƯỚC 2: TÁI VIẾT QUERY (Query Rewriter) ────────────────────────────────
def rewrite_query(llm, original_query: str) -> str:
    """
    Nếu retrieval ban đầu không tốt, tự động viết lại câu hỏi để tìm kiếm hiệu quả hơn.
    RAG thường không có bước này.
    """
    prompt = f"""Hãy viết lại câu hỏi sau để tìm kiếm tốt hơn trong tài liệu.
Giữ nguyên ý nghĩa nhưng dùng từ khóa cụ thể hơn, rõ ràng hơn.
Chỉ trả về câu hỏi đã viết lại, không giải thích.

Câu hỏi gốc: {original_query}
Câu hỏi đã viết lại:"""
    try:
        rewritten = llm.invoke(prompt).strip()
        # Làm sạch output
        rewritten = rewritten.replace('"', '').replace("'", '').strip()
        return rewritten if len(rewritten) > 5 else original_query
    except Exception:
        return original_query


# ─── BƯỚC 3: KIỂM TRA HALLUCINATION (Hallucination Grader) ──────────────────
def check_hallucination(llm, answer: str, context: str) -> dict:
    """
    Kiểm tra xem câu trả lời có bịa đặt thông tin không có trong tài liệu không.
    Đây là tính năng quan trọng nhất phân biệt CO-RAG với RAG.
    """
    prompt = f"""Dựa vào ngữ cảnh được cung cấp, hãy đánh giá câu trả lời.
Trả lời theo định dạng:
GROUNDED: YES hoặc NO
CONFIDENCE: số từ 0 đến 100

Ngữ cảnh: {context[:500]}
Câu trả lời: {answer[:300]}

Đánh giá:"""
    try:
        result = llm.invoke(prompt).strip()
        grounded = "YES" in result.upper().split("GROUNDED:")[-1][:10] if "GROUNDED:" in result.upper() else True

        conf_match = re.search(r'CONFIDENCE[:\s]+(\d+)', result, re.IGNORECASE)
        confidence = int(conf_match.group(1)) if conf_match else (75 if grounded else 30)
        confidence = max(0, min(100, confidence))
    except Exception:
        grounded, confidence = True, 70

    return {"grounded": grounded, "confidence": confidence}


# ─── PIPELINE CO-RAG CHÍNH ──────────────────────────────────────────────────
def answer_with_corag(query: str, vector_store, all_chunks, history, use_rerank, k, use_hybrid, mods) -> dict:
    """
    Pipeline CO-RAG đầy đủ:
    1. Retrieve → 2. Grade → 3. (Nếu xấu) Rewrite & Retrieve lại → 4. Generate → 5. Check hallucination
    """
    from model_layer import get_llm
    llm = get_llm(mods)

    steps_log = []  # Ghi lại các bước để hiển thị trên UI

    # ── Không có tài liệu → trả lời thẳng ──────────────────────────────────
    if not vector_store:
        prompt = f"Lịch sử: {history[-3:]}\n\nNgười dùng: {query}\nTrợ lý:"
        answer = llm.invoke(prompt).strip()
        return {
            "answer": answer, "sources": [],
            "corag_steps": [{"step": "⚡ Không có tài liệu", "detail": "Trả lời từ kiến thức mô hình"}],
            "confidence": 60, "grounded": True,
            "query_rewritten": False, "rewritten_query": query,
            "relevance_quality": "n/a", "relevant_ratio": 0
        }

    # ── BƯỚC 1: Retrieve ban đầu ─────────────────────────────────────────────
    search_kwargs = {"k": k + 2}  # Lấy thêm để có dữ liệu grading
    base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)

    if use_hybrid and all_chunks:
        bm25 = mods['BM25Retriever'].from_documents(all_chunks)
        bm25.k = k + 2
        retriever = mods['EnsembleRetriever'](retrievers=[bm25, base_retriever], weights=[0.4, 0.6])
    else:
        retriever = base_retriever

    retrieved_docs = retriever.invoke(query)
    steps_log.append({"step": f"🔍 Retrieve lần 1", "detail": f"Tìm được {len(retrieved_docs)} đoạn văn"})

    # ── BƯỚC 2: Grade documents ──────────────────────────────────────────────
    grade_result = grade_documents(llm, query, retrieved_docs)
    relevant_docs = [item["doc"] for item in grade_result["graded_docs"] if item["relevant"]]
    steps_log.append({
        "step": f"📊 Đánh giá độ liên quan",
        "detail": f"{grade_result['relevant_count']}/{grade_result['total_count']} đoạn liên quan — Chất lượng: {grade_result['quality'].upper()}"
    })

    # ── BƯỚC 3: Nếu chất lượng thấp → Rewrite query & Retrieve lại ──────────
    query_rewritten = False
    rewritten_query = query
    if grade_result["quality"] in ("low", "medium"):
        rewritten_query = rewrite_query(llm, query)
        steps_log.append({"step": "✏️ Viết lại câu hỏi", "detail": f'"{rewritten_query}"'})

        # Retrieve lại với query mới
        new_docs = retriever.invoke(rewritten_query)
        new_grade = grade_documents(llm, rewritten_query, new_docs)
        new_relevant = [item["doc"] for item in new_grade["graded_docs"] if item["relevant"]]

        # Gộp và dedup
        seen_ids = {id(d) for d in relevant_docs}
        for doc in new_relevant:
            if id(doc) not in seen_ids:
                relevant_docs.append(doc)
                seen_ids.add(id(doc))

        steps_log.append({
            "step": "🔍 Retrieve lần 2 (với query mới)",
            "detail": f"Tìm thêm {len(new_relevant)} đoạn liên quan"
        })
        query_rewritten = True
        # Cập nhật quality dựa trên combined result
        grade_result["quality"] = new_grade["quality"] if new_grade["relevance_ratio"] > grade_result["relevance_ratio"] else grade_result["quality"]

    # Fallback nếu vẫn không có gì
    if not relevant_docs:
        relevant_docs = retrieved_docs[:k]
        steps_log.append({"step": "⚠️ Fallback", "detail": "Dùng top docs ban đầu vì không tìm được docs liên quan"})

    # ── Reranking (nếu bật) ──────────────────────────────────────────────────
    if use_rerank and len(relevant_docs) > 1:
        model = mods['CrossEncoder']('cross-encoder/ms-marco-MiniLM-L-6-v2')
        scores = model.predict([(query, d.page_content) for d in relevant_docs])
        ranked = sorted(zip(scores, relevant_docs), key=lambda x: x[0], reverse=True)
        relevant_docs = [d for _, d in ranked[:k]]
        steps_log.append({"step": "🎯 Reranking", "detail": f"Sắp xếp lại, giữ top {k}"})

    # ── BƯỚC 4: Sinh câu trả lời ─────────────────────────────────────────────
    context = "\n\n".join([d.page_content for d in relevant_docs[:k]])
    history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-4:]])

    final_query = rewritten_query if query_rewritten else query
    prompt = f"""Bạn là trợ lý AI thông minh. Hãy trả lời câu hỏi dựa trên ngữ cảnh tài liệu.
Nếu thông tin không có trong tài liệu, hãy nói rõ điều đó.

Lịch sử hội thoại:
{history_text}

Ngữ cảnh từ tài liệu:
{context}

Câu hỏi: {final_query}
Trả lời chi tiết:"""

    answer = llm.invoke(prompt).strip()
    steps_log.append({"step": "💬 Sinh câu trả lời", "detail": f"Dựa trên {len(relevant_docs[:k])} đoạn tài liệu"})

    # ── BƯỚC 5: Kiểm tra Hallucination ──────────────────────────────────────
    hall_check = check_hallucination(llm, answer, context)
    steps_log.append({
        "step": f"🛡️ Kiểm tra Hallucination",
        "detail": f"Grounded: {'✅ Có căn cứ' if hall_check['grounded'] else '⚠️ Có thể suy diễn'} | Confidence: {hall_check['confidence']}%"
    })

    sources = [{"file": d.metadata.get("source_file", "?"), "preview": d.page_content[:120]} for d in relevant_docs[:k]]

    return {
        "answer": answer,
        "sources": sources,
        "corag_steps": steps_log,
        "confidence": hall_check["confidence"],
        "grounded": hall_check["grounded"],
        "query_rewritten": query_rewritten,
        "rewritten_query": rewritten_query,
        "relevance_quality": grade_result["quality"],
        "relevant_ratio": grade_result["relevance_ratio"]
    }