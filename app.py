# app.py — SmartDoc AI với chế độ so sánh RAG vs CO-RAG
import re
import streamlit as st
import uuid
from data_layer import load_chat_history, save_chat_history
from application_layer import process_file, create_vector_store, answer_question, get_chunk_stats
from corag_layer import answer_with_corag

st.set_page_config(page_title="SmartDoc AI", page_icon="✨", layout="wide")

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }

/* ── Mode selector tabs ── */
.mode-bar {
    display: flex; gap: 8px; padding: 12px 0; margin-bottom: 8px;
}
.mode-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.badge-rag { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb30; }
.badge-corag { background: #1a1a2e; color: #a78bfa; border: 1px solid #7c3aed30; }
.badge-compare { background: #1a2e1a; color: #4ade80; border: 1px solid #16a34a30; }

/* ── Panels ── */
.panel-rag {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #2563eb40;
    border-radius: 12px; padding: 16px;
    box-shadow: 0 4px 24px #2563eb15;
}
.panel-corag {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%);
    border: 1px solid #7c3aed40;
    border-radius: 12px; padding: 16px;
    box-shadow: 0 4px 24px #7c3aed15;
}
.panel-header-rag {
    font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08em;
    color: #60a5fa; text-transform: uppercase; margin-bottom: 12px;
    padding-bottom: 8px; border-bottom: 1px solid #2563eb30;
}
.panel-header-corag {
    font-weight: 700; font-size: 0.85rem; letter-spacing: 0.08em;
    color: #a78bfa; text-transform: uppercase; margin-bottom: 12px;
    padding-bottom: 8px; border-bottom: 1px solid #7c3aed30;
}

/* ── CO-RAG step log ── */
.step-log {
    background: #0a0a14; border: 1px solid #7c3aed25;
    border-radius: 8px; padding: 10px 14px; margin-top: 10px;
}
.step-item {
    display: flex; gap: 10px; padding: 5px 0;
    border-bottom: 1px solid #ffffff08; font-size: 0.78rem;
}
.step-item:last-child { border-bottom: none; }
.step-name { color: #a78bfa; font-weight: 600; min-width: 220px; }
.step-detail { color: #94a3b8; }

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: #ffffff10; border-radius: 99px; height: 6px;
    margin: 6px 0 2px; overflow: hidden;
}
.conf-bar-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #7c3aed, #a78bfa);
    transition: width 0.5s ease;
}
.conf-label { font-size: 0.72rem; color: #64748b; }

/* ── Quality badge ── */
.quality-high { color: #4ade80; font-weight: 700; }
.quality-medium { color: #fbbf24; font-weight: 700; }
.quality-low { color: #f87171; font-weight: 700; }

/* ── Welcome ── */
.welcome-title {
    text-align: center; font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #4ade80 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-top: 5vh; letter-spacing: -0.02em;
}
.welcome-sub {
    text-align: center; color: #475569; font-size: 0.95rem; margin-top: 8px;
}

/* ── Source chips ── */
.src-chip {
    display: inline-flex; align-items: center; gap: 4px;
    background: #1e293b; border: 1px solid #334155;
    border-radius: 6px; padding: 3px 10px; font-size: 0.72rem;
    color: #94a3b8; margin: 3px 2px;
}

/* ── Citation cards ── */
.cit-label {
    font-size: .7rem; color: #6b7280; font-weight: 600;
    text-transform: uppercase; letter-spacing: .09em;
    margin: 16px 0 8px; display: flex; align-items: center; gap: 6px;
}
.cit-card {
    background: #141420; border: 1px solid #2a2a3d;
    border-radius: 10px; padding: 11px 13px;
    transition: border-color .18s, box-shadow .18s; margin-bottom: 4px;
}
.cit-card:hover { border-color: #6d28d9; box-shadow: 0 0 0 3px rgba(109,40,217,.12); }
.cit-row { display: flex; align-items: flex-start; gap: 8px; }
.cit-badge {
    flex-shrink: 0; width: 20px; height: 20px;
    background: #4c1d95; color: #ddd6fe;
    border-radius: 5px; font-size: .65rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center; margin-top: 1px;
}
.cit-info { flex: 1; min-width: 0; }
.cit-fname {
    font-size: .78rem; font-weight: 600; color: #c4b5fd;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.cit-page { font-size: .67rem; color: #6b7280; margin-top: 2px; }
.cit-sep { border-top: 1px solid #2a2a3d; margin: 8px 0; }
.cit-snip {
    font-size: .71rem; color: #94a3b8; line-height: 1.5; font-style: italic;
    border-left: 2px solid #4c1d95; padding-left: 7px;
    overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}
.cit-bar { height: 2px; background: #2a2a3d; border-radius: 1px; margin-top: 8px; }
.cit-bar-fill {
    height: 100%; background: linear-gradient(90deg, #4c1d95, #7c3aed); border-radius: 1px;
}
</style>
""", unsafe_allow_html=True)


def highlight_keywords(text: str, query: str, answer: str = "") -> str:
    # Combine words from both question and answer for richer highlighting
    combined = f"{query} {answer}"
    seen = set()
    keywords = []
    for w in combined.split():
        w_clean = re.sub(r'[^\w]', '', w)
        if len(w_clean) > 3 and w_clean.lower() not in seen:
            seen.add(w_clean.lower())
            keywords.append(w_clean)
    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(lambda m: f"**{m.group(0)}**", text)
    return text


def display_sources(sources):
    if not sources:
        return

    n = len(sources)
    st.markdown(
        f'<div class="cit-label">📎 Nguồn tham khảo &nbsp;·&nbsp; {n} đoạn</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(min(n, 3))
    for i, src in enumerate(sources):
        with cols[i % len(cols)]:
            fname = src["file"]
            fname_disp = (fname[:22] + "…") if len(fname) > 22 else fname
            snip = src["preview"][:110] + "…" if len(src["preview"]) > 110 else src["preview"]
            score = src.get("score", 0.8)
            score_pct = int(min(1.0, max(0.0, score)) * 100)

            st.markdown(f"""
<div class="cit-card">
  <div class="cit-row">
    <div class="cit-badge">{i+1}</div>
    <div class="cit-info">
      <div class="cit-fname" title="{fname}">📄 {fname_disp}</div>
      <div class="cit-page">Trang {src['page']} &middot; Đoạn #{src['chunk_id']}</div>
    </div>
  </div>
  <div class="cit-sep"></div>
  <div class="cit-snip">"{snip}"</div>
  <div class="cit-bar"><div class="cit-bar-fill" style="width:{score_pct}%"></div></div>
</div>""", unsafe_allow_html=True)

            with st.popover("Xem đầy đủ →", use_container_width=True):
                st.markdown(f"**📄 {fname}**")
                st.caption(f"Trang {src['page']} · Đoạn #{src['chunk_id']} · Độ liên quan {score_pct}%")
                st.divider()
                full = src.get("full_content", src["preview"])
                full = highlight_keywords(
                    full,
                    src.get("query", ""),
                    src.get("answer", ""),
                )
                st.markdown(full)


# ─── Init ─────────────────────────────────────────────────────────────────────
def init_session():
    saved = load_chat_history()
    if saved:
        st.session_state.chat_sessions = saved
        st.session_state.current_id = list(saved.keys())[-1]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions = {
            new_id: {"name": "Trò chuyện mới", "history": [], "documents": {}, "all_chunks_data": []}
        }
        st.session_state.current_id = new_id

    defaults = {
        'chunk_size': 1000,
        'chunk_overlap': 100,
        'use_hybrid': False,
        'use_reranking': False,
        'retriever_k': 3,
        'confirm_del_id': None,
        'confirm_clear_docs': False,
        'app_mode': 'rag',
        'selected_docs': []
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


if 'chat_sessions' not in st.session_state:
    init_session()

active_id = st.session_state.current_id
active_session = st.session_state.chat_sessions[active_id]

# Vector store management
if active_session["all_chunks_data"] and 'vector_store' not in st.session_state:
    st.session_state.vector_store = create_vector_store(active_session["all_chunks_data"])
elif not active_session["all_chunks_data"] and 'vector_store' in st.session_state:
    if 'vector_store' in st.session_state:
        del st.session_state.vector_store


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Lịch sử trò chuyện")

    if st.button("➕ Trò chuyện mới", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {
            "name": "Trò chuyện mới", "history": [], "documents": {}, "all_chunks_data": []
        }
        st.session_state.current_id = new_id
        if 'vector_store' in st.session_state:
            del st.session_state.vector_store
        save_chat_history(st.session_state.chat_sessions)
        st.rerun()

    for sid, sdata in reversed(list(st.session_state.chat_sessions.items())):
        col_name, col_del = st.columns([0.8, 0.2])
        with col_name:
            if st.button(sdata['name'], key=f"s_{sid}", use_container_width=True,type="secondary"):
                st.session_state.current_id = sid
                if 'vector_store' in st.session_state:
                    del st.session_state.vector_store
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_{sid}"):
                st.session_state.confirm_del_id = sid

    if st.session_state.confirm_del_id:
        st.warning("Xóa chat này?")
        c1, c2 = st.columns(2)
        if c1.button("Xác nhận", key="y_c"):
            del st.session_state.chat_sessions[st.session_state.confirm_del_id]
            st.session_state.confirm_del_id = None
            if not st.session_state.chat_sessions:
                init_session()
            else:
                st.session_state.current_id = list(st.session_state.chat_sessions.keys())[0]
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()
        if c2.button("Hủy", key="n_c"):
            st.session_state.confirm_del_id = None
            st.rerun()

    st.markdown("---")

    # Metadata Filtering
    st.markdown("### 📂 Bộ lọc tài liệu")
    available_files = list(active_session["documents"].keys())
    if available_files:
        st.session_state.selected_docs = st.multiselect(
            "Chỉ truy vấn trong file :",
            options=available_files,
            default=available_files if not st.session_state.selected_docs else st.session_state.selected_docs
        )
    else:
        st.caption("Chưa có tài liệu nào.")

    st.markdown("---")

    # Mode Selector
    st.markdown("### 🔀 Chế độ hoạt động")
    mode_options = {
        "RAG (Chuẩn)": "rag",
        "CO-RAG (Nâng cao)": "corag",
    }
    st.session_state.app_mode = mode_options[st.radio(
        "Chọn chế độ:",
        list(mode_options.keys()),
        index=list(mode_options.values()).index(st.session_state.app_mode),
        label_visibility="collapsed"
    )]

    if st.session_state.app_mode == "corag":
        st.success("🛡️ CO-RAG: Tự đánh giá và kiểm tra hallucination", icon="✅")

    st.markdown("---")

    # ── [Cấu hình hệ thống] ──
    with st.expander("⚙️ Cấu hình hệ thống"):
        st.session_state.chunk_size = st.slider("Chunk Size", 500, 2000, st.session_state.chunk_size)
        st.session_state.chunk_overlap = st.slider("Chunk Overlap", 0, 200, st.session_state.chunk_overlap)

        # Hybrid Search Toggle
        st.session_state.use_hybrid = st.toggle("🔀 Hybrid Search", value=st.session_state.use_hybrid)
        st.session_state.use_reranking = st.toggle("🎯 Rerank (Cross-Encoder)", value=st.session_state.use_reranking)
        st.session_state.retriever_k = st.slider("Top-K Retrieval", 1, 10, st.session_state.retriever_k)

# Helper: Render một message trong history
def render_history_message(msg, mode):
    """Vẽ một message từ history, hỗ trợ cả RAG và CO-RAG metadata"""
    role = msg["role"]
    avatar = "🧑‍💻" if role == "user" else ("🟣" if mode == "corag" else "🔵")

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])
        if role == "assistant" and msg.get("sources"):
            display_sources(msg["sources"])
        if role == "assistant" and msg.get("corag_meta") and mode == "corag":
            _render_corag_meta(msg["corag_meta"])


def _render_corag_meta(meta: dict):
    """Hiển thị thông tin bổ sung của CO-RAG (steps, confidence, quality...)"""
    conf = meta.get("confidence", 0)
    grounded = meta.get("grounded", True)
    quality = meta.get("relevance_quality", "n/a")
    rewritten = meta.get("query_rewritten", False)
    rewritten_q = meta.get("rewritten_query", "")
    steps = meta.get("corag_steps", [])

    # Confidence bar
    bar_color = "#4ade80" if conf >= 70 else ("#fbbf24" if conf >= 40 else "#f87171")
    quality_class = f"quality-{quality}" if quality in ("high", "medium", "low") else ""
    ground_icon = "✅" if grounded else "⚠️"

    st.markdown(f"""
    <div style="margin-top:10px; padding:10px; background:#0a0a14; border-radius:8px; border:1px solid #7c3aed25;">
        <div style="display:flex; gap:16px; font-size:0.78rem; margin-bottom:8px; flex-wrap:wrap;">
            <span>{ground_icon} <b style="color:#94a3b8">Grounded</b></span>
            <span>📊 Chất lượng: <span class="{quality_class}">{quality.upper()}</span></span>
            {"<span>✏️ Query đã viết lại: <i style='color:#a78bfa'>" + rewritten_q + "</i></span>" if rewritten else ""}
        </div>
        <div class="conf-label">Confidence: {conf}%</div>
        <div class="conf-bar-wrap"><div class="conf-bar-fill" style="width:{conf}%; background:linear-gradient(90deg,{bar_color},{bar_color}aa)"></div></div>
    </div>
    """, unsafe_allow_html=True)

    if steps:
        with st.expander("🔍 Xem các bước CO-RAG"):
            step_html = '<div class="step-log">'
            for s in steps:
                step_html += f'<div class="step-item"><span class="step-name">{s["step"]}</span><span class="step-detail">{s["detail"]}</span></div>'
            step_html += '</div>'
            st.markdown(step_html, unsafe_allow_html=True)


# ─── Main Area ────────────────────────────────────────────────────────────────
mode = st.session_state.app_mode

# Welcome screen
if not active_session["history"]:
    st.markdown('<div class="welcome-title">✨ SmartDoc AI</div>', unsafe_allow_html=True)
    mode_desc = {
        "rag": "Chế độ RAG chuẩn — truy xuất nhanh, đơn giản",
        "corag": "Chế độ CO-RAG nâng cao — tự đánh giá & sửa lỗi retrieval"
    }
    st.markdown(f'<div class="welcome-sub">{mode_desc[mode]}</div>', unsafe_allow_html=True)

# Hiển thị lịch sử hội thoại và trích dẫn

for msg in active_session["history"]:
    render_history_message(msg, mode)

# ── Upload / Clear Files ──
with st.container():
    if active_session["documents"]:
        st.markdown("**📄 Tài liệu trong chat này:**")
        # Hiển thị danh sách file
        for doc_name in active_session["documents"].keys():
            st.caption(f"✅ {doc_name}")

        # Thống kê kỹ thuật (Yêu cầu nâng cao cho đồ án)
        stats = get_chunk_stats(active_session["all_chunks_data"])
        st.caption(
            f"📊 {stats['num_chunks']} chunks · TB {stats['avg_length']} ký tự · "
            f"Min {stats['min_length']} · Max {stats['max_length']} "
            f"· Size={st.session_state.chunk_size} / Overlap={st.session_state.chunk_overlap}"
        )

    col_up, col_clr = st.columns([0.7, 0.3])
    with col_up:
        up_files = st.file_uploader(
            "Tải thêm tài liệu (PDF/DOCX)", type=['pdf', 'docx'],
            accept_multiple_files=True, label_visibility="collapsed"
        )
    with col_clr:
        if active_session["documents"] and st.button("🗑️ Xóa toàn bộ Vector", use_container_width=True):
            st.session_state.confirm_clear_docs = True

    if up_files:
        new_f = [f for f in up_files if f.name not in active_session["documents"]]
        if new_f and st.button(f"⚡ Đang xử lý {len(new_f)} file mới..."):
            for f in new_f:
                chunks = process_file(f, st.session_state.chunk_size, st.session_state.chunk_overlap)
                if chunks:
                    active_session["all_chunks_data"].extend(chunks)
                    active_session["documents"][f.name] = True
            # Cập nhật Vector Store ngay sau khi tải
            st.session_state.vector_store = create_vector_store(active_session["all_chunks_data"])
            stats = get_chunk_stats(active_session["all_chunks_data"])
            st.info(
                f"📊 Tổng: **{stats['num_chunks']}** chunks | "
                f"TB: **{stats['avg_length']}** ký tự | "
                f"Min: {stats['min_length']} | Max: {stats['max_length']}"
            )
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()

    if st.session_state.confirm_clear_docs:
        st.error("Xác nhận xóa sạch tài liệu trong chat này?")
        b1, b2 = st.columns(2)
        if b1.button("✅ Xóa hết"):
            active_session["documents"], active_session["all_chunks_data"] = {}, []
            if 'vector_store' in st.session_state:
                del st.session_state.vector_store
            st.session_state.confirm_clear_docs = False
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()
        if b2.button("❌ Hủy"):
            st.session_state.confirm_clear_docs = False
            st.rerun()


# ─── Chat Input ───────────────────────────────────────────────────────────────
if query := st.chat_input("Hỏi SmartDoc..."):
    active_session["history"].append({"role": "user", "content": query})
    if len(active_session["history"]) == 1:
        active_session["name"] = query[:25]

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query)

    # Lấy tài nguyên hệ thống
    vs = st.session_state.get('vector_store')
    chunks = active_session["all_chunks_data"]
    from application_layer import load_dependencies
    mods = load_dependencies()
    # ── Chế độ RAG đơn  ─────────────────────────────────
    if mode == "rag":
        with st.chat_message("assistant", avatar="🔵"):
            with st.spinner("⚡ RAG đang xử lý..."):
                # Cấy các tham số từ HEAD vào lời gọi hàm của Remote
                res = answer_question(
                    query, vs, chunks,
                    history=active_session["history"][:-1],  # [Câu 6]: Hội thoại
                    use_rerank=st.session_state.use_reranking,
                    k=st.session_state.retriever_k,
                    use_hybrid=st.session_state.use_hybrid,  # [Câu 7]: Hybrid Search
                )
                st.markdown(res["answer"])
                if res.get("sources"):
                    display_sources(res["sources"])

                # Lưu vào history kèm nguồn trích dẫn
                active_session["history"].append({
                    "role": "assistant",
                    "content": res["answer"],
                    "sources": res.get("sources", [])
                })

    # ── Chế độ CO-RAG đơn ──────────────────────
    elif mode == "corag":
        with st.chat_message("assistant", avatar="🟣"):
            with st.spinner("🛡️ CO-RAG đang đánh giá và xử lý..."):
                # CO-RAG cũng cần filter_files và history để thông minh hơn
                if vs is None:
                    st.warning("Chưa có tài liệu.")
                else:
                    cres = answer_with_corag(
                    query, vs, chunks,
                    history=active_session["history"][:-1],
                    use_rerank=st.session_state.use_reranking,
                    k=st.session_state.retriever_k,
                    use_hybrid=st.session_state.use_hybrid,
                    mods=mods,
                )
                st.markdown(cres["answer"])
                if cres.get("sources"):
                    display_sources(cres["sources"])

                # Hiển thị các bước suy luận đặc trưng của CO-RAG
                meta = {k: cres[k] for k in ["corag_steps", "confidence", "grounded",
                                             "query_rewritten", "rewritten_query",
                                             "relevance_quality", "relevant_ratio"] if k in cres}
                _render_corag_meta(meta)

                active_session["history"].append({
                    "role": "assistant", "content": cres["answer"],
                    "sources": cres.get("sources", []), "corag_meta": meta
                })

    save_chat_history(st.session_state.chat_sessions)
    st.rerun()