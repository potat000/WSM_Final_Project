"""
RAG System with Two-Stage Retrieval and Reranking

預設配置（提交環境）:
- use_rerank=True: 啟用 reranking
- use_remote_rerank=True: 使用遠端 API（避免 CPU 超時）
- stage1_top_k=20: Stage 1 檢索 20 個候選
- top_k=5: Stage 2 rerank 後返回 5 個最終結果

性能提升（基於測試）:
- 詞精確度: +139%
- 詞召回率: +62.5%
- 答案相似度: +144%
"""

import argparse
import os
import re

from chunker import chunk_documents
from database import ChromaDBManager
from generator import generate_answer
from reranker import Reranker
from retriever import create_bm25_retriever, create_dense_retriever
from tqdm import tqdm
from utils import load_jsonl, save_jsonl


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
    use_hybrid=False,
    use_rerank=True,
    use_remote_rerank=True,
    chroma_path="./my_vector_db",
    retrieval_method="rrf",
    top_k=5,
    stage1_top_k=20,
    rerank_model="BAAI/bge-reranker-v2-m3",
    remote_rerank_url="http://ollama-gateway:11434/rerank",
    alpha=0.5,
    rrf_k=60,
):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 載入公司名單 (進行query正規化搜索)
    company_pattern = None
    try:
        if os.path.exists("./dragonball_dataset/company_names.txt"):
            with open(
                "./dragonball_dataset/company_names.txt", "r", encoding="utf-8"
            ) as f:
                # 讀取並去除空白
                company_list = [line.strip() for line in f if line.strip()]

            # 關鍵：按長度由大到小排序，避免「華夏娛樂」只匹配到「華夏」
            company_list.sort(key=len, reverse=True)

            if company_list:
                # 建立 Regex Pattern: (华夏娱乐有限公司|农业发展有限公司|...)
                pattern_str = "|".join(map(re.escape, company_list))
                company_pattern = re.compile(f"({pattern_str})")
                print(f"✅ 已載入 {len(company_list)} 間公司名單用於過濾。")
        else:
            print("⚠️ 警告：找不到 company_names.txt，將不會進行公司過濾。")
    except Exception as e:
        print(f"⚠️ 載入公司名單時發生錯誤: {e}")

    # 2. Chunk Documents
    print("Chunking documents...")
    chunks = chunk_documents(docs_for_chunking, language)
    print(f"Created {len(chunks)} chunks.")

    # 3. Initialize ChromaDB
    chroma_manager = None
    print(f"\n{'=' * 60}")
    print("Initializing ChromaDB for Hybrid Retrieval...")
    print(f"{'=' * 60}")

    try:
        collection_name = f"docs_{language}"

        # 初始化 ChromaDBManager
        chroma_manager = ChromaDBManager(
            persist_directory=chroma_path, collection_names=[collection_name]
        )

        # 檢查是否需要建立索引
        collection = chroma_manager.get_collection(collection_name)
        existing_count = collection.count() if collection else 0

        if existing_count == 0:
            print(f"Building ChromaDB index for {len(chunks)} chunks...")
            print("Cleaning metadata entities...")
            for chunk in chunks:
                # 判斷 chunks 是字典還是物件 (根據您的實作調整)
                # 假設 chunk 是字典，且 metadata 在 chunk['metadata']
                # 如果 chunk 是 LangChain Document 物件，請改用 chunk.metadata
                meta = (
                    chunk.get("metadata") if isinstance(chunk, dict) else chunk.metadata
                )
                if meta:
                    # 1. 清洗醫院名稱 (去除 _病患名)
                    if (
                        "hospital_patient_name" in meta
                        and meta["hospital_patient_name"]
                    ):
                        full_name = meta["hospital_patient_name"]
                        # 只保留底線前的部分
                        clean_name = full_name.split("_")[0]
                        meta["hospital_patient_name"] = clean_name
            # 準備數據
            texts, metadatas, ids = prepare_chroma_data(chunks)
            # 存入 ChromaDB
            success = chroma_manager.save_chunks_to_chroma(
                collection_name=collection_name,
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                batch_size=500,
            )

            if not success:
                print("⚠️  ChromaDB indexing failed, using BM25 only")
                chroma_manager = None
        else:
            print(f"✅ Using existing ChromaDB index ({existing_count} items)")

    except Exception as e:
        print(f"⚠️  ChromaDB initialization failed: {e}")
        print("Falling back to BM25 only")
        chroma_manager = None

    # 4. Create Retriever
    print(f"\n{'=' * 60}")
    print("Creating retriever...")
    print(f"{'=' * 60}")

    dense_retriever = create_dense_retriever(
        chunks=chunks,
        language=language,
        chroma_manager=chroma_manager,
    )
    bm25_retriever = create_bm25_retriever(chunks=chunks, language=language)

    # 5. Initialize Reranker (if enabled)
    reranker = None
    if use_rerank:
        print(f"\n{'=' * 60}")
        print("Initializing Reranker...")
        print(f"{'=' * 60}")
        reranker = Reranker(
            model_name=rerank_model,
            use_remote=use_remote_rerank,
            remote_api_url=remote_rerank_url,
        )
        print(f"Reranker mode: {'Remote API' if use_remote_rerank else 'Local Model'}")
        print(f"Stage 1 retrieval: top-{stage1_top_k}")
        print(f"Stage 2 rerank: top-{top_k}")

    print(f"Top-k: {top_k}")
    print("Retriever created successfully.")

    # 6. Process Queries
    print(f"\n{'=' * 60}")
    print("Processing queries...")
    print(f"{'=' * 60}")

    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query["query"]["content"]

        # 1. 改用 findall 抓取所有公司名稱
        target_companies = []
        if company_pattern:
            # findall 會回傳一個 list，包含所有匹配的字串
            # 注意：如果你的 regex 有多個括號 group，這裡回傳的格式可能會變 tuple，需視 regex 寫法而定
            # 假設你的 pattern 是簡單的 (CleanCo|Retail Emporium|...)
            found = company_pattern.findall(query_text)

            # 去除重複 (set) 並過濾雜訊
            target_companies = list(set(found))

            if target_companies:
                print(
                    f"偵測到公司: {target_companies}"
                )  # 除錯: 應該要看到 ['CleanCo', 'Retail Emporium']

        # 建立 ChromaDB 需要的 filter
        where_filter = None

        if target_companies:
            # 1. 定義你要搜尋的所有 Metadata 欄位名稱
            # 請確保這裡的 key 與你 ingest 入庫時的 key 一模一樣
            search_keys = ["company_name", "court_name", "hospital_patient_name"]

            # 2. 建立所有可能的組合條件
            # 邏輯：(公司名是A OR 法院名是A OR 醫院名是A) OR (公司名是B OR ...)
            or_conditions = []
            for entity in target_companies:
                for key in search_keys:
                    or_conditions.append({key: entity})

            # 3. 生成 Filter
            if len(or_conditions) == 1:
                # 極少見情況：只搜一個名稱且只搜一個欄位
                where_filter = or_conditions[0]
            else:
                # 絕大多數情況都會走這裡，因為每個名稱都要搜 3 個欄位
                where_filter = {"$or": or_conditions}
        # 檢索相關文檔
        if language == "en":
            # dense retriever
            print("英文檢索")
            # 如果使用 reranker，先檢索更多候選
            retrieve_k = stage1_top_k if use_rerank else top_k
            retrieved_chunks = dense_retriever.retrieve(
                query_text, top_k=retrieve_k, where_filter=where_filter
            )
        else:
            print("中文檢索")
            # BM25Retriever
            retrieve_k = stage1_top_k if use_rerank else top_k
            retrieved_chunks = bm25_retriever.retrieve(query_text, top_k=retrieve_k)

        # 應用 Reranker (如果啟用)
        if use_rerank and reranker is not None and retrieved_chunks:
            print(f"Reranking {len(retrieved_chunks)} candidates to top-{top_k}...")
            retrieved_chunks = reranker.rerank(
                query=query_text,
                chunks=retrieved_chunks,
                top_k=top_k,
                return_scores=True,
            )

        # 生成答案
        if language == "zh":
            answer = generate_answer(query_text, retrieved_chunks, language)
        else:
            answer = generate_answer(query_text, [retrieved_chunks[0]], language)
        # if language == "zh":
        #     answer = generate_answer(query_text, retrieved_chunks, language)
        # elif language == "en" and not multi_ref:
        #     print("單一檢索元")
        #     answer = generate_answer(query_text,[retrieved_chunks[0]],language)
        # else:
        #     answer = generate_answer(query_text, retrieved_chunks, language)
        # answer = generate_answer(query_text, retrieved_chunks, language)
        query["prediction"]["content"] = answer
        print(retrieved_chunks)
        # 儲存 References（根據語言分離策略）
        if language == "zh":
            # 中文：保存所有 chunks
            query["prediction"]["references"] = [
                chunk["page_content"] for chunk in retrieved_chunks
            ]
        else:  # English
            # 英文：只保存第一個
            # if multi_ref:
            #     query["prediction"]["references"] = [chunk["page_content"] for chunk in retrieved_chunks]
            # else:
            #     query["prediction"]["references"] = [retrieved_chunks[0]["page_content"]]

            # query["prediction"]["references"] = [
            #     chunk["page_content"] for chunk in retrieved_chunks
            # ]
            query["prediction"]["references"] = [retrieved_chunks[0]["page_content"]]

    # 7. Save Results
    save_jsonl(output_path, queries)
    print(f"\n{'=' * 60}")
    print(f"✅ Predictions saved at '{output_path}'")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG System with Optional Hybrid Retrieval and Reranking"
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
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (final output)",
    )

    # Reranker 相關參數 - 默認啟用遠程 API 模式
    parser.add_argument(
        "--use_rerank",
        action="store_true",
        default=True,
        help="Enable reranking (two-stage retrieval) - DEFAULT: True",
    )
    parser.add_argument(
        "--use_remote_rerank",
        action="store_true",
        default=True,
        help="Use remote API for reranking (suitable for CPU-only environments) - DEFAULT: True",
    )
    parser.add_argument(
        "--stage1_top_k",
        type=int,
        default=20,
        help="Number of candidates to retrieve in stage 1 (before reranking)",
    )
    parser.add_argument(
        "--rerank_model",
        default="BAAI/bge-reranker-v2-m3",
        help="Reranker model name (for local mode)",
    )
    parser.add_argument(
        "--remote_rerank_url",
        default="http://ollama-gateway:11434/rerank",
        help="Remote reranker API URL (for remote mode)",
    )

    args = parser.parse_args()

    main(
        query_path=args.query_path,
        docs_path=args.docs_path,
        language=args.language,
        output_path=args.output,
        chroma_path=args.chroma_path,
        top_k=args.top_k,
        use_rerank=args.use_rerank,
        use_remote_rerank=args.use_remote_rerank,
        stage1_top_k=args.stage1_top_k,
        rerank_model=args.rerank_model,
        remote_rerank_url=args.remote_rerank_url,
    )
