# data_layer.py
import os
import json
from langchain_core.documents import Document

DB_FILE = "smartdoc_history.json"


def load_chat_history():
    """Đọc lịch sử từ JSON, có kiểm tra file rỗng và convert ngược lại Document object"""
    if os.path.exists(DB_FILE) and os.path.getsize(DB_FILE) > 0:
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Khôi phục dictionary thành object Document của LangChain
                for sid in data:
                    if "all_chunks_data" in data[sid]:
                        data[sid]["all_chunks_data"] = [
                            Document(page_content=d["page_content"], metadata=d["metadata"])
                            for d in data[sid]["all_chunks_data"]
                        ]
                return data
        except Exception as e:
            print(f"Lỗi load history: {e}")
            return None
    return None


def save_chat_history(chat_sessions):
    """Chuyển Document object thành dict để có thể lưu vào JSON"""
    serializable_data = {}
    for sid, sdata in chat_sessions.items():
        temp_session = sdata.copy()
        if "all_chunks_data" in sdata:
            temp_session["all_chunks_data"] = [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in sdata["all_chunks_data"]
            ]
        serializable_data[sid] = temp_session

    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=2)


def build_vector_store(mods, all_chunks: list):
    """Xây dựng và trả về FAISS Vector Store"""
    embedder = mods['HuggingFaceEmbeddings'](
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return mods['FAISS'].from_documents(all_chunks, embedder)