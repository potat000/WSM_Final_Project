from typing import List, Dict, Union 
from database import ChromaDBManager
import chromadb
# # 模擬向量
# DUMMY_VECTOR = [0.123, 0.456, 0.789] 

# zh_texts = ["中文 Chunk A: 這是 Google 的最新技術。", "中文 Chunk B: TSMC 的報告指出效能提升。"]
# zh_embeddings = [DUMMY_VECTOR, DUMMY_VECTOR]
# zh_metadatas = [{"company_name": "Google"}, {"company_name": "TSMC"}]
# zh_ids = ["zh_001", "zh_002"]

# en_texts = ["English Chunk A: Microsoft released a patch.", "English Chunk B: Apple's M4 chip is fast."]
# en_embeddings = [DUMMY_VECTOR, DUMMY_VECTOR]
# en_metadatas = [{"company_name": "Microsoft"}, {"company_name": "Apple"}]
# en_ids = ["en_003", "en_004"]


# # ----------------------------------------------------
# # 使用 ChromaDBManager 類別
# # ----------------------------------------------------

# # 1. 初始化 Manager，同時創建兩個 Collection
# manager = ChromaDBManager(
#     persist_directory="./my_db",
#     collection_names=['docs_zh', 'docs_en'] # 在初始化時創建
# )

# # 2. 存儲中文數據
# manager.save_chunks_to_chroma(
#     collection_name="docs_zh",
#     texts=zh_texts,
#     #embeddings=zh_embeddings,
#     metadatas=zh_metadatas,
#     ids=zh_ids
# )

# # 3. 存儲英文數據
# manager.save_chunks_to_chroma(
#     collection_name="docs_en",
#     texts=en_texts,
#     #embeddings=en_embeddings,
#     metadatas=en_metadatas,
#     ids=en_ids
# )

# # 4. (可選) 驗證存儲結果
# zh_col = manager.get_collection("documents_zh")
# if zh_col:
#     count = zh_col.count()
#     print(f"\n驗證結果: docs_zh 中共有 {count} 條數據。")

# ###  進行檢索
# # 1. 執行中文檢索，不帶過濾條件
# query_zh_no_filter = "人工智慧的最新發展"
# results1 = manager.query_chunks(
#     collection_name="docs_zh",
#     query_text=query_zh_no_filter,
#     top_k=2
# )
# print(results1)
# # 輸出結果 (results1 是一個字典，您通常會使用 results1['documents'][0] 來獲取文本內容)

# # 2. 執行中文檢索，帶有公司過濾條件
# query_zh_with_filter = "報告最新進度"
# results2 = manager.query_chunks(
#     collection_name="docs_zh",
#     query_text=query_zh_with_filter,
#     top_k=3,
#     where_filter={"company_name": {"$eq": "TSMC"}} # 只在 TSMC 的文檔中進行搜索
# )
# # 3. 執行英文檢索
# query_en = "security patch release"
# results3 = manager.query_chunks(
#     collection_name="docs_en",
#     query_text=query_en,
#     top_k=1,
#     where_filter={"company_name": {"$eq": "Microsoft"}} # 只在 Microsoft 的文檔中進行搜索
# )

if __name__ == "__main__":
    db_path = "./my_vector_db"
    client = chromadb.PersistentClient(path=db_path)

    def audit_collection(collection_name, expected_lang):
        print(f"\n🕵️‍♀️ 正在審計 Collection: {collection_name} (預期語言: {expected_lang})")
        
        try:
            coll = client.get_collection(collection_name)
        except:
            print("❌ Collection 不存在")
            return

        # 讀取所有 metadata (不讀取 embedding 以節省記憶體)
        data = coll.get(include=["metadatas", "documents"])
        
        wrong_count = 0
        total = len(data["ids"])
        
        for i in range(total):
            meta = data["metadatas"][i]
            doc = data["documents"][i]
            
            # 判斷依據 1: 檢查 Metadata (如果你的原始資料有 language 欄位)
            if meta and "language" in meta:
                if meta["language"] != expected_lang:
                    wrong_count += 1
                    if wrong_count <= 3: # 只印出前幾個錯誤範例
                        print(f"  ⚠️ 發現錯誤 Metadata! ID: {data['ids'][i]}, Meta: {meta}")
            
            # 判斷依據 2: 簡單的內容偵測 (備用方案)
            # 如果預期是中文，但前50字裡面英文單字太多，可能就是混入了
            # 這只是一個粗略的 heuristic
            if expected_lang == "zh":
                # 簡單檢查：如果一段話裡面英文字元超過 80% 可能是錯的
                english_char_count = sum(1 for c in doc if c.isascii())
                if len(doc) > 0 and (english_char_count / len(doc)) > 0.8:
                    print(f"  ⚠️ 內容疑似英文 (在中文庫中): {doc[:50]}...")
                    
        if wrong_count == 0:
            print(f"✅ 檢查完畢：所有 {total} 筆資料看起來都符合 Metadata 標記。")
        else:
            print(f"❌ 警告：發現 {wrong_count} 筆資料可能放錯位置！")

    # 執行檢查
    audit_collection("docs_zh", "zh")
    audit_collection("docs_en", "en")