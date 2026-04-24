def get_llm(mods, model_name="qwen2.5:7b", temperature=0.7):
    """Khởi tạo và trả về LLM Inference Engine (Ollama)"""
    # Sử dụng module Ollama từ dictionary 'mods' được truyền từ tầng application
    return mods['Ollama'](model=model_name, temperature=temperature)