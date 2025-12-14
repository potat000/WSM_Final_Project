# 可以考慮替換成langchain的RecursiveCharacterTextSplitter 在助教的12/3投影片
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, language, chunk_size=1000, chunk_overlap=200):
    """
    使用 RecursiveCharacterTextSplitter 進行文檔分塊
    
    Args:
        docs: 文檔列表
        language: 語言 ('zh' 或 'en')
        chunk_size: 分塊大小（字元數）
        chunk_overlap: 重疊大小（字元數）
    
    Returns:
        chunks: 分塊列表，每個包含 'page_content' 和 'metadata'
    """
    
    if language == "zh":
        chunk_size = 450
        chunk_overlap = 90
    else:
        chunk_size = 500
        chunk_overlap = 100
        # Generation best
        # (800, 120) --> 如果沒有要優化 prompt（配置最平衡）
    
    chunks = []
    
    # 根據語言選擇不同的分隔符
    if language == "zh":
        # 中文分隔符：段落 → 句號 → 逗號 → 字元
        separators = [
            "\n\n",   # 段落
            "\n",     # 換行
            "。",     # 句號
            "！",     # 驚嘆號
            "？",     # 問號
            "；",     # 分號
            "，",     # 逗號
            " ",      # 空格
            "",       # 字元
        ]
    else:  # English
        # 英文分隔符：段落 → 句號 → 逗號 → 單字
        separators = [
            "\n\n",   # 段落
            "\n",     # 換行
            ". ",     # 句號
            "! ",     # 驚嘆號
            "? ",     # 問號
            "; ",     # 分號
            ", ",     # 逗號
            " ",      # 空格
            "",       # 字元
        ]
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=False,
    )
    
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
        text = doc['content']
        doc_chunks = text_splitter.split_text(text)
        
        # 為每個 chunk 添加 metadata
        for chunk_index, chunk_text in enumerate(doc_chunks):
            # 複製文檔的 metadata
            chunk_metadata = doc.copy()
            chunk_metadata.pop('content', None)  # 移除原始內容
            chunk_metadata['chunk_index'] = chunk_index
            chunk_metadata['total_chunks'] = len(doc_chunks)
            
            # 構建 chunk 對象（與原始格式保持一致）
            chunk = {
                'page_content': chunk_text,
                'metadata': chunk_metadata
            }
            chunks.append(chunk)
    
    return chunks
