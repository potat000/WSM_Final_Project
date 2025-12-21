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
        meta = chunk['metadata']  # 先把 metadata 變數抓出來，程式碼比較乾淨
            
        # 使用 .get() 安全讀取，並用 or 串接不同領域可能的欄位名稱
        # 邏輯：有公司名就用公司名，沒有就找法院名，再沒有就找醫院/病患...
        source_name = (
            meta.get('company_name') or 
            meta.get('court_name') or 
            meta.get('hospital_patient_name')
        )
        print(f"測試 source ({i}):", source_name)
        formatted_parts.append(f"[source:{source_name}]\n{content}")
    
    return "\n\n".join(formatted_parts)


def generate_answer(query: str, context_chunks: list, language: str = "en") -> str:
    """
    根據語言生成對應的 Prompt 並調用 LLM
    
    ⭐ 修正版：
    1. temperature 保持低值 (0.1)
    2. Prompt 非常嚴格（不允許推論）
    3. 目標：重現 ID 10 的效果
    """
    # 格式化 context
    formatted_context = format_context(context_chunks)
    
    # 根據語言選擇 Prompt（嚴格版）
    if language == "zh":
        prompt = f"""你是一个专业的问答助手。请根据以下提供的参考资料回答问题。

    【参考资料，皆与问题提及之公司相关】
    {formatted_context}

    【问题】
    {query}

    【重要规则】

    你的答案必须完全基于上述参考资料

    不要使用参考资料以外的知识或信息

    不要推测、推论或猜测

    如果参考资料中没有足够的信息来回答问题，请明确说明「根据提供的参考资料，无法完整回答此问题」

    请提供清晰、完整、准确的答案，包含所有相关的重要细节

    直接回答问题，不需要额外的开场白或结尾

    【答案】"""
    else:  # English
        prompt = f"""You are a professional question-answering assistant. Answer the question based solely on the provided reference materials.

    [Reference Materials,all are relevant to the companies mentioned in the query]
    {formatted_context}

    [Question]
    {query}

    [Critical Rules]
    1. Your answer must be based entirely on the reference materials above
    2. Do not use knowledge or information outside the provided references
    3. Do not speculate, infer, or guess
    4. If the provided documents do not contain the answer, explicitly state 'Unable to answer
    5. Provide a clear, complete, and accurate answer that includes all relevant important details
    6. Answer directly without unnecessary preamble or conclusion

    [Answer]"""
    
    # 調用 LLM
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    # ⭐ 修正：temperature 保持低值
    response = client.generate(
        model=ollama_config["model"],
        prompt=prompt,
        options={
            "temperature": 0.0,      # 修正：保持低值（不是 0.3）
            "num_predict": 512,      # 可以測試 512
            "top_p": 0.9,
            "top_k": 40,
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