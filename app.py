# app.py
import streamlit as st
import uuid
from data_layer import load_chat_history, save_chat_history
from application_layer import process_file, create_vector_store, answer_question

st.set_page_config(page_title="SmartDoc AI", page_icon="✨", layout="wide")

# (Yêu cầu Giao diện): CSS tùy chỉnh để hiển thị trích dẫn (Citation) chuyên nghiệp
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    .welcome-title { text-align: center; font-size: 2.5rem; font-weight: 600; margin-top: 5vh; }
    .citation-box { 
        background: #2b2b2b; 
        padding: 12px; 
        border-radius: 8px; 
        margin-top: 8px; 
        font-size: 0.85rem; 
        border-left: 4px solid #00d4ff;
        color: #e0e0e0;
    }
    .source-tag {
        background: #444;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        color: #00d4ff;
    }
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

    # Cấu hình mặc định (Bổ sung tham số cho các câu yêu cầu đồ án)
    defaults = {
        'chunk_size': 1000,
        'chunk_overlap': 100,
        'use_hybrid': False,
        'use_reranking': False,
        'retriever_k': 3,
        'confirm_del_id': None,
        'confirm_clear_docs': False,
        'selected_docs': []  # Dành cho Yêu cầu Câu 8
    }
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
    if 'vector_store' in st.session_state:
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

    # (Yêu cầu Câu 8): Multi-doc Metadata Filtering
    st.markdown("### 📂 Bộ lọc tài liệu")
    available_files = list(active_session["documents"].keys())
    if available_files:
        st.session_state.selected_docs = st.multiselect(
            "Chỉ truy vấn trong file (Câu 8):",
            options=available_files,
            default=available_files
        )
    else:
        st.caption("Chưa có tài liệu nào.")

    # Cấu hình hệ thống nâng cao
    with st.expander("⚙️ Cấu hình hệ thống"):
        st.session_state.chunk_size = st.slider("Chunk Size", 500, 2000, st.session_state.chunk_size)
        st.session_state.chunk_overlap = st.slider("Chunk Overlap", 50, 200, st.session_state.chunk_overlap)

        # (Yêu cầu Câu 7): Hybrid Search Toggle
        st.session_state.use_hybrid = st.toggle("🔀 Hybrid Search", value=st.session_state.use_hybrid)
        st.session_state.use_reranking = st.toggle("🎯 Rerank (Cross-Encoder)", value=st.session_state.use_reranking)
        st.session_state.retriever_k = st.number_input("Số lượng Context (k)", 1, 10, st.session_state.retriever_k)

# ─── Main Area ───────────────────────────────────────────────────────
if not active_session["history"]:
    st.markdown('<div class="welcome-title">✨ SmartDoc AI</div>', unsafe_allow_html=True)

# Hiển thị lịch sử hội thoại và trích dẫn
for msg in active_session["history"]:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "✨"):
        st.markdown(msg["content"])

        # Hiển thị trích dẫn nguồn (Yêu cầu hiển thị minh bạch dữ liệu)
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Xem nguồn trích dẫn"):
                for idx, src in enumerate(msg["sources"]):
                    st.markdown(f"""
                    <div class="citation-box">
                        <span class="source-tag">Nguồn #{idx + 1}</span> 
                        <b>File:</b> {src.get('source', 'N/A')} | 
                        <b>Trang:</b> {src.get('page', 'N/A')}<br>
                        <hr style="margin: 5px 0; border: 0.5px solid #444;">
                        <i>"{src.get('content', '')[:250]}..."</i>
                    </div>
                    """, unsafe_allow_html=True)

# Upload & Quản lý File
with st.container():
    if active_session["documents"]:
        st.markdown("**📄 Tài liệu hiện có trong phiên này:**")
        cols = st.columns(3)
        for i, doc_name in enumerate(active_session["documents"].keys()):
            cols[i % 3].caption(f"✅ {doc_name}")

    col_up, col_clr = st.columns([0.7, 0.3])
    with col_up:
        up_files = st.file_uploader("Tải lên PDF hoặc DOCX", type=['pdf', 'docx'], accept_multiple_files=True,
                                    label_visibility="collapsed")
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

# (Yêu cầu Câu 6, 7, 8): Xử lý câu hỏi
if query := st.chat_input("Hỏi SmartDoc..."):
    active_session["history"].append({"role": "user", "content": query})
    if len(active_session["history"]) == 1: active_session["name"] = query[:25]

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Đang tìm kiếm dữ liệu..."):
            # Gọi hàm xử lý với đầy đủ các tham số yêu cầu của đồ án
            res = answer_question(
                query=query,
                vector_store=st.session_state.get('vector_store'),
                all_chunks=active_session["all_chunks_data"],
                chat_history=active_session["history"][:-1],  # (Câu 6): Truyền lịch sử hội thoại
                use_reranking=st.session_state.use_reranking,
                k=st.session_state.retriever_k,
                use_hybrid=st.session_state.use_hybrid,  # (Câu 7): Hybrid Search
                filter_files=st.session_state.selected_docs  # (Câu 8): Lọc file theo Metadata
            )

            st.markdown(res["answer"])

            # Lưu câu trả lời kèm thông tin trích dẫn vào session
            active_session["history"].append({
                "role": "assistant",
                "content": res["answer"],
                "sources": res.get("sources", [])
            })
            save_chat_history(st.session_state.chat_sessions)
            st.rerun()