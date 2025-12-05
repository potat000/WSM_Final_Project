from typing import List, Dict, Union 
from database import ChromaDBManager
# 模擬向量
DUMMY_VECTOR = [0.123, 0.456, 0.789] 

zh_texts = ["中文 Chunk A: 這是 Google 的最新技術。", "中文 Chunk B: TSMC 的報告指出效能提升。"]
zh_embeddings = [DUMMY_VECTOR, DUMMY_VECTOR]
zh_metadatas = [{"company_name": "Google"}, {"company_name": "TSMC"}]
zh_ids = ["zh_001", "zh_002"]

en_texts = ["English Chunk A: Microsoft released a patch.", "English Chunk B: Apple's M4 chip is fast."]
en_embeddings = [DUMMY_VECTOR, DUMMY_VECTOR]
en_metadatas = [{"company_name": "Microsoft"}, {"company_name": "Apple"}]
en_ids = ["en_003", "en_004"]


# ----------------------------------------------------
# 使用 ChromaDBManager 類別
# ----------------------------------------------------

# 1. 初始化 Manager，同時創建兩個 Collection
manager = ChromaDBManager(
    persist_directory="./my_db",
    collection_names=['docs_zh', 'docs_en'] # 在初始化時創建
)

# 2. 存儲中文數據
manager.save_chunks_to_chroma(
    collection_name="docs_zh",
    texts=zh_texts,
    #embeddings=zh_embeddings,
    metadatas=zh_metadatas,
    ids=zh_ids
)

# 3. 存儲英文數據
manager.save_chunks_to_chroma(
    collection_name="docs_en",
    texts=en_texts,
    #embeddings=en_embeddings,
    metadatas=en_metadatas,
    ids=en_ids
)

# 4. (可選) 驗證存儲結果
zh_col = manager.get_collection("documents_zh")
if zh_col:
    count = zh_col.count()
    print(f"\n驗證結果: docs_zh 中共有 {count} 條數據。")

###  進行檢索
# 1. 執行中文檢索，不帶過濾條件
query_zh_no_filter = "人工智慧的最新發展"
results1 = manager.query_chunks(
    collection_name="docs_zh",
    query_text=query_zh_no_filter,
    top_k=2
)
print(results1)
# 輸出結果 (results1 是一個字典，您通常會使用 results1['documents'][0] 來獲取文本內容)

# 2. 執行中文檢索，帶有公司過濾條件
query_zh_with_filter = "報告最新進度"
results2 = manager.query_chunks(
    collection_name="docs_zh",
    query_text=query_zh_with_filter,
    top_k=3,
    where_filter={"company_name": {"$eq": "TSMC"}} # 只在 TSMC 的文檔中進行搜索
)
# 3. 執行英文檢索
query_en = "security patch release"
results3 = manager.query_chunks(
    collection_name="docs_en",
    query_text=query_en,
    top_k=1,
    where_filter={"company_name": {"$eq": "Microsoft"}} # 只在 Microsoft 的文檔中進行搜索
)
