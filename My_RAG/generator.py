from ollama import Client
from pathlib import Path
import yaml


def load_ollama_config() -> dict:
    configs_folder = Path(__file__).parent.parent / "configs"
    config_paths = [
        configs_folder / "config_local.yaml",
        configs_folder / "config_submit.yaml",
    ]
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError("No configuration file found.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]


def format_context(context_chunks: list) -> str:
    """
    將 context chunks 格式化為結構化的參考資料
    """
    formatted_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        content = chunk['page_content'].strip()
        formatted_parts.append(f"[參考資料 {i}]\n{content}")
    
    return "\n\n".join(formatted_parts)


def generate_answer(query: str, context_chunks: list, language: str = "en") -> str:
    """
    根據語言生成對應的 Prompt 並調用 LLM
    
    Args:
        query: 用戶的問題
        context_chunks: 檢索到的參考資料列表
        language: 語言 ('zh' 或 'en')，由 main.py 傳入
    
    主要改進：
    1. 雙語 Prompt（中文/英文分離）
    2. 移除句數限制
    3. 強調只使用提供的參考資料
    4. 結構化的 Context 呈現
    5. 要求完整回答
    """
    # 格式化 context
    formatted_context = format_context(context_chunks)
    
    # 根據語言選擇 Prompt
    if language == "zh":
        prompt = f"""你是一個專業的問答助手。請根據以下提供的參考資料回答問題。

【參考資料】
{formatted_context}

【問題】
{query}

【回答要求】
1. 你的答案必須完全基於上述參考資料，不要使用參考資料以外的知識
2. 如果參考資料中沒有足夠的資訊來回答問題，請明確說明「根據提供的參考資料，無法完整回答此問題」
3. 請提供清晰、完整、準確的答案，包含所有相關的重要細節
4. 直接回答問題，不需要額外的開場白或結尾
5. 如果參考資料中的資訊有衝突，請指出並說明

【答案】"""
    else:  # English
        prompt = f"""You are a professional question-answering assistant. Answer the question based solely on the provided reference materials.

[Reference Materials]
{formatted_context}

[Question]
{query}

[Requirements]
1. Your answer must be based entirely on the reference materials above. Do not use knowledge outside the provided references.
2. If the reference materials do not contain sufficient information to answer the question, clearly state "Based on the provided reference materials, this question cannot be fully answered."
3. Provide a clear, complete, and accurate answer that includes all relevant important details.
4. Answer directly without unnecessary preamble or conclusion.
5. If there are conflicting information in the reference materials, point it out and explain.

[Answer]"""
    
    # 調用 LLM
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    # 調整 LLM 參數以獲得更好的輸出
    response = client.generate(
        model=ollama_config["model"],
        prompt=prompt,
        options={
            "temperature": 0.1,      # 降低隨機性，提高一致性
            "num_predict": 512,      # 允許更長的回答
            "top_p": 0.9,           # Nucleus sampling
            "top_k": 40,            # Top-K sampling
        }
    )
    
    return response["response"]


if __name__ == "__main__":
    # 測試中文
    print("=== 測試中文 ===")
    query_zh = "台積電2023年的營收是多少？"
    context_chunks_zh = [
        {"page_content": "台積電在2023年第一季度營收達到1000億美元。相較於去年同期成長了25%。"},
        {"page_content": "這個成績在半導體產業中非常優異，主要得益於先進製程的需求增加。"}
    ]
    answer_zh = generate_answer(query_zh, context_chunks_zh, language="zh")
    print(f"問題：{query_zh}")
    print(f"答案：{answer_zh}")
    
    print("\n" + "="*50 + "\n")
    
    # 測試英文
    print("=== 測試英文 ===")
    query_en = "What is the capital of France?"
    context_chunks_en = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer_en = generate_answer(query_en, context_chunks_en, language="en")
    print(f"Question: {query_en}")
    print(f"Answer: {answer_en}")