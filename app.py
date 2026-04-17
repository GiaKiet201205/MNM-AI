# app.py
import re
import streamlit as st
import uuid
from data_layer import load_chat_history, save_chat_history
from application_layer import process_file, create_vector_store, answer_question, get_chunk_stats

st.set_page_config(page_title="SmartDoc AI", page_icon="✨", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    .welcome-title { text-align: center; font-size: 2.5rem; font-weight: 600; margin-top: 5vh; }

    /* ── Citation cards ── */
    .cit-label {
        font-size: .7rem; color: #6b7280; font-weight: 600;
        text-transform: uppercase; letter-spacing: .09em;
        margin: 16px 0 8px; display: flex; align-items: center; gap: 6px;
    }
    .cit-card {
        background: #141420; border: 1px solid #2a2a3d;
        border-radius: 10px; padding: 11px 13px;
        transition: border-color .18s, box-shadow .18s;
        margin-bottom: 4px;
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
        height: 100%; background: linear-gradient(90deg, #4c1d95, #7c3aed);
        border-radius: 1px;
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


def init_session():
    saved = load_chat_history()
    if saved:
        st.session_state.chat_sessions = saved
        st.session_state.current_id = list(saved.keys())[-1]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions = {
            new_id: {"name": "Trò chuyện mới", "history": [], "documents": {}, "all_chunks_data": []}}
        st.session_state.current_id = new_id

    # Cấu hình mặc định
    defaults = {'chunk_size': 1000, 'chunk_overlap': 100, 'use_hybrid': False, 'use_reranking': False, 'retriever_k': 3,
                'confirm_del_id': None, 'confirm_clear_docs': False}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v


if 'chat_sessions' not in st.session_state: init_session()

# Lấy session hiện tại
active_id = st.session_state.current_id
active_session = st.session_state.chat_sessions[active_id]

# Quản lý Vector Store theo Session (Sandbox)
if active_session["all_chunks_data"] and 'vector_store' not in st.session_state:
    st.session_state.vector_store = create_vector_store(active_session["all_chunks_data"])
elif not active_session["all_chunks_data"] and 'vector_store' in st.session_state:
    del st.session_state.vector_store

# ─── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💬 Lịch sử trò chuyện")
    if st.button("➕ Trò chuyện mới", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chat_sessions[new_id] = {"name": "Trò chuyện mới", "history": [], "documents": {},
                                                  "all_chunks_data": []}
        st.session_state.current_id = new_id
        if 'vector_store' in st.session_state: del st.session_state.vector_store
        save_chat_history(st.session_state.chat_sessions)
        st.rerun()

    for sid, sdata in reversed(list(st.session_state.chat_sessions.items())):
        col_name, col_del = st.columns([0.8, 0.2])
        with col_name:
            if st.button(sdata['name'], key=f"s_{sid}", use_container_width=True,
                         type="primary" if sid == active_id else "secondary"):
                st.session_state.current_id = sid
                if 'vector_store' in st.session_state: del st.session_state.vector_store
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_{sid}"): st.session_state.confirm_del_id = sid

    if st.session_state.confirm_del_id:
        st.warning("Xóa chat này?")
        c1, c2 = st.columns(2)
        if c1.button("✅", key="y_c"):
            del st.session_state.chat_sessions[st.session_state.confirm_del_id]
            st.session_state.confirm_del_id = None
            if not st.session_state.chat_sessions:
                init_session()
            else:
                st.session_state.current_id = list(st.session_state.chat_sessions.keys())[0]
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()
        if c2.button("❌", key="n_c"):
            st.session_state.confirm_del_id = None
            st.rerun()

    st.markdown("---")

    # TRẢ LẠI NGUYÊN BẢN CẤU HÌNH HỆ THỐNG
    with st.expander("⚙️ Cấu hình hệ thống"):
        st.session_state.chunk_size = st.slider("Chunk Size", 500, 2000, st.session_state.chunk_size)
        st.session_state.chunk_overlap = st.slider("Chunk Overlap", 0, 200, st.session_state.chunk_overlap)
        st.session_state.use_hybrid = st.toggle("🔀 Hybrid Search", value=st.session_state.use_hybrid)
        st.session_state.use_reranking = st.toggle("🎯 Rerank (Cross-Encoder)", value=st.session_state.use_reranking)

# ─── Main Area ───────────────────────────────────────────────────────
if not active_session["history"]:
    st.markdown('<div class="welcome-title">✨ SmartDoc AI</div>', unsafe_allow_html=True)

for msg in active_session["history"]:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "✨"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            display_sources(msg["sources"])

# Upload & Clear File (Gemini style)
with st.container():
    # --- PHẦN THÊM MỚI: HIỂN THỊ DANH SÁCH FILE ĐÃ LƯU ---
    if active_session["documents"]:
        st.markdown("**📄 Các tài liệu đang được dùng trong chat này:**")
        for doc_name in active_session["documents"].keys():
            st.caption(f"✅ {doc_name}")
        stats = get_chunk_stats(active_session["all_chunks_data"])
        st.caption(
            f"📊 {stats['num_chunks']} chunks · TB {stats['avg_length']} ký tự · "
            f"Min {stats['min_length']} · Max {stats['max_length']} "
            f"· Size={st.session_state.chunk_size} / Overlap={st.session_state.chunk_overlap}"
        )

    col_up, col_clr = st.columns([0.7, 0.3])
    with col_up:
        up_files = st.file_uploader("Tải thêm tài liệu cho chat này", type=['pdf', 'docx'], accept_multiple_files=True, label_visibility="collapsed")
    with col_clr:
        if active_session["documents"] and st.button("🗑️ Clear Vector Store", use_container_width=True):
            st.session_state.confirm_clear_docs = True

    if up_files:
        new_f = [f for f in up_files if f.name not in active_session["documents"]]
        if new_f and st.button(f"⚡ Đọc {len(new_f)} file mới"):
            for f in new_f:
                chunks = process_file(f, st.session_state.chunk_size, st.session_state.chunk_overlap)
                if chunks:
                    active_session["all_chunks_data"].extend(chunks)
                    active_session["documents"][f.name] = True
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
            if 'vector_store' in st.session_state: del st.session_state.vector_store
            st.session_state.confirm_clear_docs = False
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()
        if b2.button("❌ Hủy"):
            st.session_state.confirm_clear_docs = False
            st.rerun()

# Input
if query := st.chat_input("Hỏi SmartDoc..."):
    active_session["history"].append({"role": "user", "content": query})
    if len(active_session["history"]) == 1: active_session["name"] = query[:25]

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query)
    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Đang nghĩ..."):
            res = answer_question(
                query, st.session_state.get('vector_store'),
                active_session["all_chunks_data"], active_session["history"],
                st.session_state.use_reranking, st.session_state.retriever_k,
                st.session_state.use_hybrid,
            )
            st.markdown(res["answer"])
            if res.get("sources"):
                display_sources(res["sources"])
            active_session["history"].append(
                {"role": "assistant", "content": res["answer"], "sources": res.get("sources", [])})
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()
