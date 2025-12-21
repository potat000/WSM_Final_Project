# 可以考慮替換成langchain的RecursiveCharacterTextSplitter 在助教的12/3投影片
from langchain_text_splitters import RecursiveCharacterTextSplitter
from generator import load_ollama_config
from ollama import Client
from tqdm import tqdm
def _generate_chunk_context(language, doc_text, chunk_text, metadata=None):
    """
    Generate contextual description for a specific chunk using Ollama.
    This follows Anthropic's Contextual Retrieval approach.
    """
    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    
    # Truncate doc_text to avoid token limits (keep first 6000 chars for smaller models)
    doc_text_truncated = doc_text[:6000]
    
    # Extract subject name from metadata
    subject_name = ""
    if metadata:
        if "company_name" in metadata:
            subject_name = metadata["company_name"]
        elif "hospital_patient_name" in metadata:
            subject_name = metadata["hospital_patient_name"]
        elif "court_name" in metadata:
            subject_name = metadata["court_name"]
    print("metadata name",subject_name)
    if language == "zh":
        subject_instruction = ""
        if subject_name:
            subject_instruction = f"\n重要：本文档的主体是「{subject_name}」。"
        
        prompt = f"""<document>
{doc_text_truncated}
</document>

以下是我们想要在整个文档中定位的块：
<chunk>
{chunk_text}
</chunk>

任务：为这个块生成一个上下文描述（50-80字），说明这段内容在整个文档中的位置和作用。{subject_instruction}

要求：
- 必须用简体中文回答
- 必须说明这段内容位于文档的哪个部分（如：文档开头、第二部分、结尾部分、财务指标部分等）
- 必须包含主体名称（公司名称/法院名称/医院名称）
- 如果是法律文档，必须包含法院名称和被告人/当事人姓名
- 如果是病历文档，必须包含医院名称和患者姓名
- 禁止直接复制原文内容
- 禁止使用代词如"该公司"、"该患者"、"本文档"等

格式示例：「这段内容位于[文档位置]，描述了[主体名称]的[主要内容]。」

请直接输出上下文描述："""
    else:
        subject_instruction = ""
        if subject_name:
            subject_instruction = f"\nIMPORTANT: The subject of this document is \"{subject_name}\"."
        
        prompt = f"""<document>
{doc_text_truncated}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Task: Generate a short context (30-50 words) describing WHERE this chunk is located in the document and its role.{subject_instruction}

Requirements:
- You MUST respond in English only
- MUST specify the location in the document (e.g., "at the beginning", "in the financial section", "at the end")
- MUST include the explicit subject name (company name/court name/hospital name)
- For legal documents, include both court name and defendant/party names
- For medical records, include both hospital name and patient name
- DO NOT copy the original text directly
- DO NOT use pronouns like "the company", "this document", "it", etc.

Format example: "This section, located in [document position], describes [subject name]'s [main content]."

Output ONLY the context description:"""
    
    try:
        response = client.generate(
            model=ollama_config["model"],  # Uses your configured model
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
            }
        )
        context = response["response"].strip()
        return context
    except Exception as e:
        print(f"Error generating chunk context: {e}")
        return ""
    
def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    """
    使用 RecursiveCharacterTextSplitter 進行文檔分塊，並加入 Contextual Retrieval
    """
    
    # --- 參數設定保持不變 ---
    if language == "zh":
        chunk_size = 128
        chunk_overlap = 25
    else:
        chunk_size = 512
        chunk_overlap = 100
    
    chunks = []
    
    # 根據語言選擇分隔符 (保持不變)
    if language == "zh":
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    else:
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False,
    )
    
    # 計算總處理量以便顯示進度 (Optional)
    print(f"開始處理文檔並生成上下文 (Language: {language})...這可能需要一段時間")
    
    # 處理每個文檔
    for doc_index, doc in enumerate(docs):
        # 檢查文檔格式
        if 'content' not in doc or not isinstance(doc['content'], str):
            continue
        if 'language' not in doc:
            continue
        
        # 只處理匹配語言的文檔
        if doc['language'] != language:
            continue
        
        # 使用 RecursiveCharacterTextSplitter 分塊
        full_doc_text = doc['content']
        doc_chunks = text_splitter.split_text(full_doc_text)
        
        print(f"正在處理文檔 {doc_index + 1}/{len(docs)}: 產生了 {len(doc_chunks)} 個 Chunks")

        # 為每個 chunk 生成上下文並添加 metadata
        # 使用 tqdm 顯示進度，因為 LLM 生成很慢
        for chunk_index, chunk_text in tqdm(enumerate(doc_chunks), total=len(doc_chunks), desc=f"Doc {doc_index+1} Context"):
            
            # --- 核心修改開始 ---
            
            # 1. 呼叫 LLM 生成上下文
            # 注意：這裡傳入 doc 是為了讓函數內能讀取 metadata (公司名/醫院名)
            context = _generate_chunk_context(language, full_doc_text, chunk_text, metadata=doc)
            print("context:\n",context)
            # 2. 組合內容：上下文 + 換行 + 原始 Chunk
            if context:
                combined_content = f"{context}\n\n{chunk_text}"
            else:
                combined_content = chunk_text
                
            # --- 核心修改結束 ---

            # 複製文檔的 metadata
            chunk_metadata = doc.copy()
            chunk_metadata.pop('content', None)  # 移除原始長文內容
            chunk_metadata['chunk_index'] = chunk_index
            chunk_metadata['total_chunks'] = len(doc_chunks)
            
            # 建議：保留原始文本在 metadata，方便後續引用顯示或 Debug
            chunk_metadata['original_text'] = chunk_text 
            # 建議：也可以把生成的 context 存一份在 metadata
            chunk_metadata['generated_context'] = context

            # 構建 chunk 對象
            chunk = {
                'page_content': combined_content, # 這裡是包含上下文的豐富內容
                'metadata': chunk_metadata
            }
            chunks.append(chunk)
    
    return chunks
