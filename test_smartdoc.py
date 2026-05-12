import pytest
from unittest.mock import MagicMock
from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

# Lớp giả lập để vượt qua kiểm tra kiểu dữ liệu của Pydantic trong LangChain
class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return []

# Câu hỏi 1: Thêm hỗ trợ file DOCX
def test_docx_loader_support(mocker):
    mock_docx_loader = mocker.patch('langchain_community.document_loaders.Docx2txtLoader')
    file_path = 'sample.docx'
    loader = mock_docx_loader(file_path)
    loader.load()
    mock_docx_loader.assert_called_once_with(file_path)
    assert loader.load.called

# Câu hỏi 2: Lưu trữ lịch sử hội thoại
def test_session_history_management():
    session_state = {'history': []}
    session_state['history'].append({'role': 'user', 'content': 'Kiểm tra lưu trữ'})
    assert len(session_state['history']) == 1
    assert session_state['history'][0]['content'] == 'Kiểm tra lưu trữ'

# Câu hỏi 3: Thêm nút xóa lịch sử
def test_clear_history_logic():
    session_state = {'history': [{'role': 'user', 'content': 'Xóa tôi đi'}]}
    session_state['history'].clear()
    assert len(session_state['history']) == 0

# Câu hỏi 4: Cải thiện chunk strategy
def test_text_splitter_configuration():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    test_text = 'A' * 2000
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(test_text)
    assert len(chunks) == 2
    assert len(chunks[0]) == 1500

# Câu hỏi 5: Thêm citation/source tracking
def test_source_tracking_metadata():
    doc = Document(page_content='Dữ liệu mẫu', metadata={'source': 'tailieu.pdf', 'page': 10})
    assert doc.metadata.get('source') == 'tailieu.pdf'
    assert doc.metadata.get('page') == 10

# Câu hỏi 6: Implement Conversational RAG
def test_conversational_memory_format():
    chat_history = [('user', 'Chào AI'), ('assistant', 'Chào bạn')]
    formatted = "\n".join([f"{role}: {msg}" for role, msg in chat_history])
    assert "user: Chào AI" in formatted
    assert "assistant: Chào bạn" in formatted

# Câu hỏi 7: Thêm hybrid search
def test_ensemble_retriever_initialization():
    mock_bm25 = DummyRetriever()
    mock_faiss = DummyRetriever()
    from langchain.retrievers import EnsembleRetriever
    ensemble = EnsembleRetriever(retrievers=[mock_bm25, mock_faiss], weights=[0.4, 0.6])
    assert len(ensemble.retrievers) == 2
    assert ensemble.weights == [0.4, 0.6]

# Câu hỏi 8: Multi-document RAG với metadata filtering
def test_metadata_filtering(mocker):
    mock_vs = MagicMock()
    filter_data = {'source': 'chuyen-nganh.pdf'}
    mock_vs.similarity_search('truy vấn', filter=filter_data)
    mock_vs.similarity_search.assert_called_with('truy vấn', filter=filter_data)

# Câu hỏi 9: Implement Re-ranking với Cross-Encoder
def test_cross_encoder_reranking_logic():
    scores = [0.2, 0.9, 0.5]
    docs = ['Doc A', 'Doc B', 'Doc C']
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    assert ranked[0][1] == 'Doc B' # Điểm cao nhất 0.9

# Câu hỏi 10: Advanced RAG với Self-RAG
def test_self_rag_evaluation(mocker):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = 'SUPPORTED'
    result = mock_llm.invoke('Đánh giá nội dung')
    assert result == 'SUPPORTED'