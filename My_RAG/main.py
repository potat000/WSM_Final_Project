import argparse

from chunker import chunk_documents
from database import ChromaDBManager
from generator import generate_answer
from retriever import create_retriever
from reranker import Reranker, HybridRerankRetriever
from tqdm import tqdm
from utils import load_jsonl, save_jsonl
import os
import re

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
    use_rerank=False,
    chroma_path="./my_vector_db",
    retrieval_method="rrf",
    top_k=3,
    stage1_top_k=20,
    rerank_model="BAAI/bge-reranker-v2-m3",
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

    # 2. Chunk Documents
    print("Chunking documents...")
    chunks = chunk_documents(docs_for_chunking, language)
    print(f"Created {len(chunks)} chunks.")

    # 3. Initialize ChromaDB (如果使用混合檢索)
    chroma_manager = None
    if use_hybrid:
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

    base_retriever = create_retriever(
        chunks=chunks,
        language=language,
        chroma_manager=chroma_manager,
        use_hybrid=use_hybrid and chroma_manager is not None,
        only_dense=True 
    )
    
    # 5. Add Re-ranking Layer (如果啟用)
    if use_rerank:
        print(f"\n{'=' * 60}")
        print("Initializing Re-ranker (Two-Stage Retrieval)...")
        print(f"{'=' * 60}")
        
        reranker = Reranker(model_name=rerank_model)
        retriever = HybridRerankRetriever(
            base_retriever=base_retriever,
            reranker=reranker,
            stage1_top_k=stage1_top_k,
            stage2_top_k=top_k
        )
        
        print(f"Stage 1 (Retrieval): Top-{stage1_top_k}")
        print(f"Stage 2 (Re-ranking): Top-{top_k}")
        print(f"Re-ranker Model: {rerank_model}")
    else:
        retriever = base_retriever

    # 設定混合檢索參數
    if hasattr(retriever, "set_params"):
        retriever.set_params(alpha=alpha, rrf_k=rrf_k)
        if use_hybrid:
            print(f"Method: {retrieval_method.upper()}")
            if retrieval_method == "weighted":
                print(f"  - BM25 weight: {alpha}")
                print(f"  - Vector weight: {1 - alpha}")
            else:
                print(f"  - RRF k: {rrf_k}")

    if not use_rerank:
        print(f"Top-k: {top_k}")
    print("Retriever created successfully.")

    # 6. Process Queries
    print(f"\n{'=' * 60}")
    print("Processing queries...")
    print(f"{'=' * 60}")

    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query["query"]["content"]
        target_company = None
        if company_pattern:
            match = company_pattern.search(query_text)
            if match:
                target_company = match.group(1)
                print(f"偵測到公司: {target_company}") # 除錯用
                
        # 建立 ChromaDB 需要的 filter 格式
        # 如果有抓到公司，就設定 where={"company_name": "xxx"}，否則為 None
        where_filter = {"company_name": target_company} if target_company else None
        # 檢索相關文檔
        if (
            hasattr(retriever, "retrieve")
            and "method" in retriever.retrieve.__code__.co_varnames
        ):
            # HybridRetriever or HybridRerankRetriever
            retrieved_chunks = retriever.retrieve(
                query_text, top_k=top_k, method=retrieval_method, where_filter= where_filter
            )
        else:
            # BM25Retriever
            retrieved_chunks = retriever.retrieve(query_text, top_k=top_k)
            # Dense Retriever
            #retrieved_chunks = retriever.retrieve(query, top_k=top_k,where_filter=where_filter)

        # 生成答案
        answer = generate_answer(query_text, retrieved_chunks, language)
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
            query["prediction"]["references"] = [retrieved_chunks[0]["page_content"]]


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

    # 新增: 混合檢索參數 (可選)
    parser.add_argument(
        "--use_hybrid",
        action="store_true",
        help="Enable hybrid retrieval (BM25 + Vector)",
    )
    parser.add_argument(
        "--chroma_path", default="./my_vector_db", help="ChromaDB storage path"
    )
    parser.add_argument(
        "--retrieval_method",
        default="rrf",
        choices=["rrf", "weighted"],
        help="Merge method (rrf or weighted)",
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="BM25 weight for weighted method (0-1)"
    )
    parser.add_argument("--rrf_k", type=int, default=60, help="RRF smoothing parameter")
    
    # Re-ranking 參數
    parser.add_argument(
        "--use_rerank",
        action="store_true",
        help="Enable re-ranking (Two-Stage Retrieval)",
    )
    parser.add_argument(
        "--stage1_top_k",
        type=int,
        default=20,
        help="Stage 1 retrieval count (before re-ranking)",
    )
    parser.add_argument(
        "--rerank_model",
        default="BAAI/bge-reranker-v2-m3",
        help="Re-ranker model name",
    )

    args = parser.parse_args()

    main(
        query_path=args.query_path,
        docs_path=args.docs_path,
        language=args.language,
        output_path=args.output,
        use_hybrid=args.use_hybrid,
        use_rerank=args.use_rerank,
        chroma_path=args.chroma_path,
        retrieval_method=args.retrieval_method,
        top_k=args.top_k,
        stage1_top_k=args.stage1_top_k,
        rerank_model=args.rerank_model,
        alpha=args.alpha,
        rrf_k=args.rrf_k,
    )
