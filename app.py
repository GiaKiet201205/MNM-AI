# app.py
import streamlit as st
import uuid
from data_layer import load_chat_history, save_chat_history
from application_layer import process_file, create_vector_store, answer_question

st.set_page_config(page_title="SmartDoc AI", page_icon="✨", layout="wide")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    .welcome-title { text-align: center; font-size: 2.5rem; font-weight: 600; margin-top: 5vh; }
    .citation-box { background: #2b2b2b; padding: 10px; border-radius: 5px; margin-top: 5px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


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
        st.session_state.chunk_overlap = st.slider("Chunk Overlap", 50, 200, st.session_state.chunk_overlap)
        st.session_state.use_hybrid = st.toggle("🔀 Hybrid Search", value=st.session_state.use_hybrid)
        st.session_state.use_reranking = st.toggle("🎯 Rerank (Cross-Encoder)", value=st.session_state.use_reranking)

# ─── Main Area ───────────────────────────────────────────────────────
if not active_session["history"]:
    st.markdown('<div class="welcome-title">✨ SmartDoc AI</div>', unsafe_allow_html=True)

for msg in active_session["history"]:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "✨"):
        st.markdown(msg["content"])

# Upload & Clear File (Gemini style)
with st.container():
    # --- PHẦN THÊM MỚI: HIỂN THỊ DANH SÁCH FILE ĐÃ LƯU ---
    if active_session["documents"]:
        st.markdown("**📄 Các tài liệu đang được dùng trong chat này:**")
        for doc_name in active_session["documents"].keys():
            st.caption(f"✅ {doc_name}")
    # ----------------------------------------------------

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
            res = answer_question(query, st.session_state.get('vector_store'), active_session["all_chunks_data"],
                                  active_session["history"], st.session_state.use_reranking,
                                  st.session_state.retriever_k, st.session_state.use_hybrid)
            st.markdown(res["answer"])
            active_session["history"].append(
                {"role": "assistant", "content": res["answer"], "sources": res.get("sources", [])})
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()