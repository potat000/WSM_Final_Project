import argparse
import os
import re

from chunker import chunk_documents
from database import ChromaDBManager
from generator import generate_answer
from retriever import create_dense_retriever,create_bm25_retriever,create_pyserini_retriever,SimpleHybridRetriever
from reranker import Reranker
from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from generator import _domain_router_en,_domain_router_zh
# Reranker 配置
USE_REMOTE_RERANKER = True  # True: 提交環境(遠程API), False: 本地測試

# 語言特定配置
LANGUAGE_CONFIG = {
    "zh": {
        "use_rerank": True,
        "stage1_top_k": 20,
        "final_top_k": 3
    },
    "en": {
        "use_rerank": True,
        "stage1_top_k": 20,
        "final_top_k": 5
    }
}

def load_chunks_from_chroma(collection):
    """從現有的 ChromaDB collection 讀取所有資料並還原成 chunks 列表"""
    print("正在從 ChromaDB 讀取快取資料...")
    
    # 讀取所有資料 (包含 document 和 metadata)
    # limit=None 確保讀取全部，include 參數確保我們拿到需要的欄位
    results = collection.get(include=["documents", "metadatas"])
    
    loaded_chunks = []
    total = len(results["ids"])
    
    for i in range(total):
        chunk = {
            "page_content": results["documents"][i],
            "metadata": results["metadatas"][i]
        }
        loaded_chunks.append(chunk)
        
    print(f"✅ 成功從 DB 復原 {len(loaded_chunks)} 個 Chunks (跳過 LLM 生成)")
    return loaded_chunks

def prepare_chroma_data(chunks):
    """準備 ChromaDB 需要的數據格式"""
    texts = []
    metadatas = []
    ids = []

    for idx, chunk in enumerate(chunks):
        texts.append(chunk["page_content"])

        # 提取 metadata (只保留基本類型)
        metadata = {}
        if "metadata" in chunk:
            for key, value in chunk["metadata"].items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
                else:
                    metadata[key] = str(value)
        metadatas.append(metadata)
        ids.append(str(idx))

    return texts, metadatas, ids


def main(
    query_path,
    docs_path,
    language,
    output_path,
    chroma_path="./my_vector_db",
    top_k=3
):
    # 根據語言獲取配置
    lang_config = LANGUAGE_CONFIG.get(language, {})
    use_rerank = lang_config.get("use_rerank", False)
    stage1_top_k = lang_config.get("stage1_top_k", 20)
    final_top_k = lang_config.get("final_top_k", 3)
    
    print(f"\n{'=' * 60}")
    print(f"配置信息:")
    print(f"  語言: {language}")
    print(f"  使用 Reranker: {use_rerank}")
    if use_rerank:
        print(f"  Reranker 模式: {'遠程API' if USE_REMOTE_RERANKER else '本地模型'}")
        print(f"  Stage 1 候選數: {stage1_top_k}")
        print(f"  Stage 2 最終數: {final_top_k}")
    else:
        print(f"  檢索數量: {final_top_k}")
    print(f"{'=' * 60}\n")
    
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2.載入公司名單 (進行query正規化搜索)
    company_pattern = None
    try:
        if os.path.exists('./dragonball_dataset/company_names.txt'):
            with open('./dragonball_dataset/company_names.txt', 'r', encoding='utf-8') as f:
                # 讀取並去除空白
                company_list = [line.strip() for line in f if line.strip()]
            
            # 關鍵：按長度由大到小排序，避免「華夏娛樂」只匹配到「華夏」
            company_list.sort(key=len, reverse=True)
            
            if company_list:
                # 建立 Regex Pattern: (华夏娱乐有限公司|农业发展有限公司|...)
                pattern_str = '|'.join(map(re.escape, company_list))
                company_pattern = re.compile(f"({pattern_str})")
                print(f"✅ 已載入 {len(company_list)} 間公司名單用於過濾。")
        else:
            print("⚠️ 警告：找不到 company_names.txt，將不會進行公司過濾。")
    except Exception as e:
        print(f"⚠️ 載入公司名單時發生錯誤: {e}")

    # 3. Chunk Documents
    chunks = []
    chroma_manager = None
    collection_name = f"docs_{language}"
    
    print(f"\n{'=' * 60}")
    print("Initializing ChromaDB & Checking Cache...")
    print(f"{'=' * 60}")

    try:
        # 先連接 ChromaDB
        chroma_manager = ChromaDBManager(
            persist_directory=chroma_path, collection_names=[collection_name]
        )
        collection = chroma_manager.get_collection(collection_name)
        existing_count = collection.count() if collection else 0

        # 判斷是否需要重新 Chunking
        if existing_count > 0:
            # A計畫：DB 裡有資料 -> 直接拿出來用
            print(f"✅ 檢測到現有索引 ({existing_count} items)，跳過 LLM 生成步驟。")
            chunks = load_chunks_from_chroma(collection)
        else:
            # B計畫：DB 是空的 -> 執行昂貴的 Chunking + LLM Context
            print("⚠️ 未檢測到索引，開始執行文檔分塊與 LLM 上下文生成 (這會花點時間)...")
            chunks = chunk_documents(docs_for_chunking, language)
            print(f"Created {len(chunks)} chunks.")
            
            # 清洗 metadata (保留原本邏輯)
            print("Cleaning metadata entities...")
            for chunk in chunks:
                meta = chunk.get('metadata') if isinstance(chunk, dict) else chunk.metadata
                if meta:
                    if "hospital_patient_name" in meta and meta["hospital_patient_name"]:
                        full_name = meta["hospital_patient_name"]
                        clean_name = full_name.split('_')[0] 
                        meta["hospital_patient_name"] = clean_name
            
            # 存入 ChromaDB
            print(f"Building ChromaDB index for {len(chunks)} chunks...")
            texts, metadatas, ids = prepare_chroma_data(chunks)
            success = chroma_manager.save_chunks_to_chroma(
                collection_name=collection_name,
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                batch_size=500,
            )
            if not success:
                print("⚠️ ChromaDB indexing failed.")

    except Exception as e:
        print(f"⚠️ ChromaDB Error: {e}")
        # 如果 DB 掛了，迫不得已只好現場重算 (Fallback)
        if not chunks:
            print("Fallback: Re-calculating chunks in memory...")
            chunks = chunk_documents(docs_for_chunking, language)

    # 4. Create Retriever
    ## 強制修改成兩個retriever結果都要用到並作hybrid
    print(f"\n{'=' * 60}")
    print("Creating retriever...")
    print(f"{'=' * 60}")

    dense_retriever = create_dense_retriever(
        chunks=chunks,
        language=language,
        chroma_manager=chroma_manager,
    )
    
    pyserini_retriever = create_pyserini_retriever(
        chunks=chunks,
        language=language
    )
    # 設定權重
    if language == "zh":
        # 中文環境：通常 BM25 對專有名詞更準，權重給高一點
        weights = {"dense": 0.4, "sparse": 0.6}
    else:
        # 英文環境：一般預設 0.5/0.5 或視情況調整
        weights = {"dense": 0.5, "sparse": 0.5}

    print(f"Initializing Hybrid Retriever with weights: {weights}")
    hybrid_retriever = SimpleHybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=pyserini_retriever,
        weights=weights,
        language=language
    )    

    # 5. Initialize Reranker (if needed)
    reranker = None
    if use_rerank:
        print(f"\n{'=' * 60}")
        print("Initializing Reranker...")
        print(f"{'=' * 60}")
        
        try:
            reranker = Reranker(
                mode="remote" if USE_REMOTE_RERANKER else "local"
            )
            print(f"✅ Reranker initialized successfully")
        except Exception as e:
            print(f"⚠️ Reranker initialization failed: {e}")
            print("⚠️ 將使用單階段檢索")
            use_rerank = False

    # 6. Process Queries
    print(f"\n{'=' * 60}")
    print("Processing queries...")
    print(f"{'=' * 60}")

    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query["query"]["content"]
        # 改用 findall 抓取所有公司名稱
        target_companies = []
        if company_pattern:
            # findall 會回傳一個 list，包含所有匹配的字串
            found = company_pattern.findall(query_text)
            
            # 去除重複 (set) 並過濾雜訊
            target_companies = list(set(found))
            
            if target_companies:
                print(f"偵測到公司: {target_companies}")  # 除錯: 應該要看到 ['CleanCo', 'Retail Emporium']

        # 建立 ChromaDB 需要的 filter
        where_filter = None
        
        if target_companies:
            # 定義你要搜尋的所有 Metadata 欄位名稱
            # 請確保這裡的 key 與你 ingest 入庫時的 key 一模一樣
            search_keys = ["company_name", "court_name", "hospital_patient_name"]
            
            # 建立所有可能的組合條件
            # 邏輯：(公司名是A OR 法院名是A OR 醫院名是A) OR (公司名是B OR ...)
            or_conditions = []
            for entity in target_companies:
                for key in search_keys:
                    or_conditions.append({key: entity})
            
            # 生成 Filter
            if len(or_conditions) == 1:
                # 極少見情況：只搜一個名稱且只搜一個欄位
                where_filter = or_conditions[0]
            else:
                # 絕大多數情況都會走這裡，因為每個名稱都要搜 3 個欄位
                where_filter = {"$or": or_conditions}
        
        # Stage 1: 檢索候選文檔
        if use_rerank:
            # 使用 reranker: 先檢索更多候選
            retrieve_k = stage1_top_k
        else:
            # 不使用 reranker: 直接檢索最終數量
            retrieve_k = final_top_k
            
        retrieved_chunks = hybrid_retriever.retrieve(
                query_text, 
                top_k=retrieve_k, 
                where_filter=where_filter
            )
        #retrieved_chunks = dense_retriever.retrieve(query_text,retrieve_k,where_filter)
        # =================================================
        # 🟢 新增：去重邏輯 (Deduplication)
        # =================================================
        seen_ids = set()
        unique_chunks = []
        for chunk in retrieved_chunks:
            # 優先嘗試抓取 metadata 裡的 id，如果沒有則退而求其次用內容本身當 key
            # 假設 chunk 是 dict 或 object，這裡做個相容性處理
            if isinstance(chunk, dict):
                c_id = chunk.get("metadata", {}).get("id") or chunk.get("page_content")
            else: # 假設是 Document 物件
                c_id = chunk.metadata.get("id") or chunk.page_content
            
            if c_id not in seen_ids:
                seen_ids.add(c_id)
                unique_chunks.append(chunk)
        
        # 將去重後的結果指派回去
        retrieved_chunks = unique_chunks
        # =================================================
        # Stage 2: Reranking（如果啟用）
        if use_rerank and reranker is not None and retrieved_chunks:
            print("執行reranker!")
            retrieved_chunks = reranker.rerank(
                query=query_text,
                chunks=retrieved_chunks,
                top_k=final_top_k,
                return_scores=True,
            )

        # 生成答案
        if language == "zh":
            query_domain = _domain_router_zh(query_text,retrieved_chunks)
            answer = generate_answer(query_text, retrieved_chunks, language, query_domain)
        else:
            query_domain = _domain_router_en(query_text,retrieved_chunks)
            answer = generate_answer(query_text, retrieved_chunks,language,query_domain)

        query["prediction"]["content"] = answer
        
        # 儲存 References（根據語言分離策略）
        if language == "zh":
            # 中文：保存所有 chunks
            query["prediction"]["references"] = [
                chunk["page_content"] for chunk in retrieved_chunks
            ]
        else: 
            query["prediction"]["references"] = [
                chunk["page_content"] for chunk in retrieved_chunks
            ]


    # 7. Save Results
    save_jsonl(output_path, queries)
    print(f"\n{'=' * 60}")
    print(f"✅ Predictions saved at '{output_path}'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG System with Optional Hybrid Retrieval"
    )

    # 原有的基本參數
    parser.add_argument("--query_path", required=True, help="Path to the query file")
    parser.add_argument("--docs_path", required=True, help="Path to the documents file")
    parser.add_argument("--language", required=True, help="Language (zh or en)")
    parser.add_argument("--output", required=True, help="Path to the output file")

    parser.add_argument(
        "--chroma_path", default="./my_vector_db", help="ChromaDB storage path"
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of chunks to retrieve"
    )
    args = parser.parse_args()

    main(
        query_path=args.query_path,
        docs_path=args.docs_path,
        language=args.language,
        output_path=args.output,
        chroma_path=args.chroma_path,
        top_k=args.top_k
    )
