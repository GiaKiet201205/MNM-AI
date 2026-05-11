import sqlite3
import json
import os
from datetime import datetime
from langchain_core.documents import Document

DB_FILE = "smartdoc_database.db"


def get_connection():
    """Thiết lập kết nối đến SQLite và trả về đối tượng connection."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Khởi tạo cấu trúc các bảng nếu chưa tồn tại."""
    conn = get_connection()
    cursor = conn.cursor()

    # Bảng lưu trữ phiên trò chuyện
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Bảng lưu trữ tin nhắn (History)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            sources TEXT,
            latency TEXT,
            self_rag TEXT,
            corag_meta TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        )
    ''')

    # Bảng lưu trữ tài liệu và các đoạn văn bản (Chunks)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            file_name TEXT,
            chunks TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()


def save_session(session_id, name):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT OR IGNORE INTO sessions (id, name) VALUES (?, ?)', (session_id, name))
    conn.commit()
    conn.close()


def get_all_sessions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM sessions ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows


def save_message(session_id, role, content, sources=None, latency=None, self_rag=None, corag_meta=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO messages (session_id, role, content, sources, latency, self_rag, corag_meta)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id, role, content,
        json.dumps(sources, ensure_ascii=False) if sources else None,
        json.dumps(latency, ensure_ascii=False) if latency else None,
        json.dumps(self_rag, ensure_ascii=False) if self_rag else None,
        json.dumps(corag_meta, ensure_ascii=False) if corag_meta else None
    ))
    conn.commit()
    conn.close()


def load_history(session_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC', (session_id,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        history.append({
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources"]) if row["sources"] else [],
            "latency": json.loads(row["latency"]) if row["latency"] else None,
            "self_rag": json.loads(row["self_rag"]) if row["self_rag"] else None,
            "corag_meta": json.loads(row["corag_meta"]) if row["corag_meta"] else None
        })
    return history


def save_document_chunks(session_id, file_name, chunks):
    conn = get_connection()
    cursor = conn.cursor()
    # Chuyển đổi list Document thành list dict để lưu JSON
    serializable_chunks = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in chunks
    ]
    cursor.execute('''
        INSERT INTO documents (session_id, file_name, chunks)
        VALUES (?, ?, ?)
    ''', (session_id, file_name, json.dumps(serializable_chunks, ensure_ascii=False)))
    conn.commit()
    conn.close()


def load_all_chunks(session_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT chunks FROM documents WHERE session_id = ?', (session_id,))
    rows = cursor.fetchall()
    conn.close()

    all_chunks = []
    for row in rows:
        chunks_data = json.loads(row["chunks"])
        all_chunks.extend([
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in chunks_data
        ])
    return all_chunks


def delete_session_data(session_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()

def update_session_name(session_id, new_name):
        """Cập nhật tên hiển thị của phiên trò chuyện trong database."""
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sessions 
            SET name = ? 
            WHERE id = ?
        ''', (new_name, session_id))
        conn.commit()
        conn.close()


def build_vector_store(mods, all_chunks: list):
    """Giữ nguyên logic FAISS nhưng được gọi từ dữ liệu SQLite."""
    embedder = mods['HuggingFaceEmbeddings'](
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return mods['FAISS'].from_documents(all_chunks, embedder)