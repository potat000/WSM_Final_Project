from typing import Dict, List, Optional

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
import os
import json
import shutil
import subprocess
from pyserini.search.lucene import LuceneSearcher

class SimpleHybridRetriever:
    def __init__(self, dense_retriever, sparse_retriever, weights={"dense": 0.5, "sparse": 0.5}):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.weights = weights

    def _normalize_scores(self, results):
        """Min-Max Normalization"""
        if not results:
            return []
        scores = [r["score"] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return results
        for r in results:
            r["normalized_score"] = (r["score"] - min_score) / (max_score - min_score)
        return results

    def _matches_filter(self, metadata, where_filter):
        """
        手動檢查 metadata 是否符合 ChromaDB 風格的 filter
        支援簡單的 {"key": "value"} 和 {"$or": [...]}
        """
        if not where_filter:
            return True
        if not metadata:
            return False

        # 處理 $or 邏輯 (您的代碼中有用到)
        if "$or" in where_filter:
            conditions = where_filter["$or"]
            for cond in conditions:
                # 假設 cond 是單一鍵值對 {'company_name': 'XXX'}
                k, v = list(cond.items())[0]
                if metadata.get(k) == v:
                    return True
            return False
            
        # 處理單一條件 (一般情況)
        for key, value in where_filter.items():
            if metadata.get(key) != value:
                return False
        return True

    def retrieve(self, query, top_k=5, where_filter=None):
        # 1. 擴大候選範圍 (因為過濾後數量會變少)
        candidate_k = top_k * 3 
        
        # 2. 執行檢索
        # Dense 自帶過濾，所以結果已經是乾淨的
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_k, where_filter=where_filter)
        
        # Sparse (Pyserini) 不帶過濾，會回傳髒資料
        sparse_results = self.sparse_retriever.retrieve(query, top_k=candidate_k)

        # 3. 對 Sparse 結果進行「後處理過濾」 (關鍵修正！)
        if where_filter:
            filtered_sparse = []
            for doc in sparse_results:
                # 這裡假設 doc 裡有 'metadata' 欄位，且內容結構正確
                if self._matches_filter(doc.get("metadata"), where_filter):
                    filtered_sparse.append(doc)
            sparse_results = filtered_sparse

        # 4. 正規化
        dense_results = self._normalize_scores(dense_results)
        sparse_results = self._normalize_scores(sparse_results)

        # 5. 加權融合
        fused_scores = {}
        doc_map = {}

        for doc in dense_results:
            cid = doc.get("chunk_id") or doc.get("id")
            doc_map[cid] = doc
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (doc.get("normalized_score", 0.0) * self.weights["dense"])

        for doc in sparse_results:
            cid = doc.get("chunk_id") or doc.get("id")
            if cid not in doc_map:
                doc_map[cid] = doc
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (doc.get("normalized_score", 0.0) * self.weights["sparse"])

        # 6. 排序與取 Top K
        sorted_ids = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        
        final_results = []
        for cid, score in sorted_ids:
            chunk = doc_map[cid].copy()
            chunk["score"] = score
            if "normalized_score" in chunk:
                del chunk["normalized_score"]
            final_results.append(chunk)

        return final_results
    
class PyseriniRetriever:
    """
    基於 Pyserini (Lucene) 的 BM25 檢索器。
    特點：
    1. 不佔用大量 RAM (索引存於 Disk)
    2. 檢索速度極快
    3. 內建多語言 Analyzer (不需要手動 jieba)
    """

    def __init__(self, chunks, language, index_path="./pyserini_index"):
        self.chunks = chunks
        self.language = language
        self.index_path = f"{index_path}_{language}"  # 區分中英文索引路徑
        
        # 1. 準備索引資料
        # 如果索引已經存在，我們可以選擇跳過重建，或者強制重建 (這裡設為強制重建以確保資料最新)
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
            
        self._build_index()

        # 2. 載入 Searcher
        self.searcher = LuceneSearcher(self.index_path)
        
        # 3. 設定語言參數 (這很重要，決定了它如何分詞)
        if language == "zh":
            self.searcher.set_language("zh")  # 使用 Lucene 內建的中文分詞
        else:
            self.searcher.set_language("en")  # 使用標準英文分詞

    def _build_index(self):
        """將 chunks 轉換為 JSONL 並調用 Pyserini 建立索引"""
        print(f"正在為 {len(self.chunks)} 筆資料建立 Pyserini 索引...")
        
        # 建立暫存資料夾放 input.jsonl
        input_dir = "./temp_pyserini_input"
        if os.path.exists(input_dir):
            shutil.rmtree(input_dir)
        os.makedirs(input_dir)

        jsonl_path = os.path.join(input_dir, "docs.jsonl")
        
        # 將 chunks 寫入符合 Pyserini 格式的 JSONL (id, contents)
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for idx, chunk in enumerate(self.chunks):
                doc = {
                    "id": str(idx),  # 使用 list index 作為 id，方便找回
                    "contents": chunk["page_content"]
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        # 構建索引指令
        # Pyserini 通常建議使用 subprocess 調用命令行工具進行索引
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", input_dir,
            "--index", self.index_path,
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ]

        # 如果是中文，需要在索引階段指定語言
        if self.language == "zh":
            cmd.extend(["--language", "zh"])

        # 執行索引命令 (靜默模式，若要看 log 可移除 stdout=subprocess.DEVNULL)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 清理暫存檔
        shutil.rmtree(input_dir)
        print("索引建立完成。")

    def retrieve(self, query, top_k=3):
        """檢索並返回帶分數的結果"""
        
        # 執行搜尋
        # Pyserini 會自動根據 set_language 處理 query 的分詞
        hits = self.searcher.search(query, k=top_k)

        results = []
        for hit in hits:
            # hit.docid 是我們在 _build_index 時存入的 str(idx)
            chunk_idx = int(hit.docid)
            
            # 複製原始 chunk 並加入分數
            chunk = self.chunks[chunk_idx].copy()
            chunk["score"] = float(hit.score)
            chunk["chunk_id"] = str(chunk_idx)
            
            # (可選) Pyserini 其實也存了 raw content，可以用 hit.raw 取得
            # 但既然我們有 self.chunks，直接查表最快
            
            results.append(chunk)

        return results

class BM25Retriever:
    """BM25 關鍵字檢索器"""

    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk["page_content"] for chunk in chunks]

        if language == "zh":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=3):
        """檢索並返回帶分數的結果"""
        if self.language == "zh":
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.split(" ")

        # 獲取所有文檔的分數
        scores = self.bm25.get_scores(tokenized_query)

        # 獲取 top_k 的索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 構建結果列表
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            chunk["chunk_id"] = str(idx)
            results.append(chunk)

        return results


# class HybridRetriever:
#     """混合檢索器：BM25 + Dense Embedding (ChromaDB)"""

#     def __init__(self, chunks, language, chroma_manager=None):
#         self.chunks = chunks
#         self.language = language

#         # BM25 檢索器
#         self.bm25_retriever = BM25Retriever(chunks, language)

#         # ChromaDB 檢索器
#         self.chroma_manager = chroma_manager
#         self.collection_name = f"docs_{language}" if chroma_manager else None

#         # 可調參數
#         self.alpha = 0.7  # BM25 權重 (0-1)，vector 權重為 (1-alpha)
#         self.rrf_k = 60  # RRF 平滑參數

#     def retrieve(self, query, top_k=3, method="rrf",where_filter=None):
#         """
#         混合檢索主函數

#         Args:
#             query: 查詢文本
#             top_k: 最終返回數量
#             method: 合併方法 ("rrf" 或 "weighted")

#         Returns:
#             List of chunks with scores
#         """
#         # 如果沒有 ChromaDB，退回到純 BM25
#         if self.chroma_manager is None:
#             print("使用純BM25")
#             return self.bm25_retriever.retrieve(query, top_k)

#         # 1. BM25 檢索 (取 2*top_k 增加覆蓋率)
#         bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)

#         # 2. Vector 檢索 (取 2*top_k)
#         try:
#             chroma_results = self.chroma_manager.query_chunks(
#                 collection_name=self.collection_name, query_text=query, top_k=top_k * 2,where_filter=where_filter
#             )

#             # 將 ChromaDB 結果轉換為統一格式
#             vector_results = self._parse_chroma_results(chroma_results)

#             if not vector_results:
#                 print("Warning: Vector search returned no results, using BM25 only")
#                 return bm25_results[:top_k]

#         except Exception as e:
#             print(f"Vector search failed: {e}, falling back to BM25 only")
#             return bm25_results[:top_k]

#         # 3. 合併結果
#         if method == "rrf":
#             merged_results = self._reciprocal_rank_fusion(
#                 bm25_results, vector_results, top_k
#             )
#         else:  # weighted
#             merged_results = self._weighted_merge(bm25_results, vector_results, top_k)

#         return merged_results

#     def _parse_chroma_results(self, chroma_results: Optional[Dict]) -> List[Dict]:
#         """
#         將 ChromaDB 的查詢結果轉換為統一格式

#         ChromaDB 返回格式:
#         {
#             'ids': [['id1', 'id2', ...]],
#             'documents': [['text1', 'text2', ...]],
#             'metadatas': [[{...}, {...}, ...]],
#             'distances': [[0.5, 0.7, ...]]  # 距離越小越相似
#         }
#         """
#         if not chroma_results:
#             return []

#         # ChromaDB 返回的是嵌套列表
#         ids = chroma_results.get("ids", [[]])[0]
#         documents = chroma_results.get("documents", [[]])[0]
#         metadatas = chroma_results.get("metadatas", [[]])[0]
#         distances = chroma_results.get("distances", [[]])[0]

#         results = []
#         for i, doc_id in enumerate(ids):
#             chunk_id = str(doc_id)

#             # 只印第一筆確認 ID 轉換
#             if i == 0:
#                 print(
#                     f"\n🔍 [ID Mapping Check] 原始={doc_id!r} ({type(doc_id).__name__}) → 轉換後={chunk_id!r} ({type(chunk_id).__name__})"
#                 )

#             # 將距離轉換為相似度分數 (距離越小，相似度越高)
#             # 使用公式: similarity = 1 / (1 + distance)
#             distance = distances[i] if i < len(distances) else 1.0
#             similarity_score = 1.0 / (1.0 + distance)

#             result = {
#                 "chunk_id": str(doc_id),
#                 "page_content": documents[i] if i < len(documents) else "",
#                 "metadata": metadatas[i] if i < len(metadatas) else {},
#                 "score": similarity_score,
#                 "distance": distance,
#             }
#             results.append(result)

#         return results

#     def _weighted_merge(
#         self, bm25_results: List[Dict], vector_results: List[Dict], top_k: int
#     ) -> List[Dict]:
#         """
#         方法 A: 加權平均合併
#         適合: 當你想明確控制 BM25 和 Vector 的影響力
#         """
#         scores = {}
#         chunk_data = {}

#         # 正規化 BM25 分數到 0-1
#         max_bm25 = max([r["score"] for r in bm25_results], default=1e-6)
#         if max_bm25 == 0:
#             max_bm25 = 1e-6

#         # 處理 BM25 結果
#         for result in bm25_results:
#             chunk_id = result["chunk_id"]  # 已經是字串
#             normalized_score = result["score"] / max_bm25
#             scores[chunk_id] = self.alpha * normalized_score
#             chunk_data[chunk_id] = result

#         # 處理 Vector 結果
#         for result in vector_results:
#             chunk_id = result["chunk_id"]
#             vector_score = result["score"]  # 已經是相似度分數 (0-1)

#             if chunk_id in scores:
#                 scores[chunk_id] += (1 - self.alpha) * vector_score
#             else:
#                 scores[chunk_id] = (1 - self.alpha) * vector_score
#                 chunk_data[chunk_id] = result

#         # 排序並構建最終結果
#         sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

#         merged_results = []
#         for chunk_id in sorted_ids[:top_k]:
#             chunk = chunk_data[chunk_id].copy()
#             chunk["score"] = scores[chunk_id]
#             chunk["chunk_id"] = chunk_id
#             merged_results.append(chunk)

#         return merged_results

#     def _reciprocal_rank_fusion(
#         self, bm25_results: List[Dict], vector_results: List[Dict], top_k: int
#     ) -> List[Dict]:
#         """
#         方法 B: Reciprocal Rank Fusion (RRF)
#         適合: 不確定哪個檢索器更好時，讓數據說話

#         公式: score(d) = Σ 1 / (k + rank_i(d))
#         """
#         scores = {}
#         chunk_data = {}

#         # 處理 BM25 排名
#         for rank, result in enumerate(bm25_results):
#             chunk_id = result["chunk_id"]
#             scores[chunk_id] = 1.0 / (self.rrf_k + rank + 1)
#             chunk_data[chunk_id] = result

#         # 處理 Vector 排名 (累加)
#         for rank, result in enumerate(vector_results):
#             chunk_id = result["chunk_id"]
#             rrf_score = 1.0 / (self.rrf_k + rank + 1)

#             if chunk_id in scores:
#                 scores[chunk_id] += rrf_score
#                 print("叮咚 累加重複計算！")
#             else:
#                 scores[chunk_id] = rrf_score
#                 chunk_data[chunk_id] = result

#         # 排序並構建最終結果
#         sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

#         merged_results = []
#         for chunk_id in sorted_ids[:top_k]:
#             chunk = chunk_data[chunk_id].copy()
#             chunk["score"] = scores[chunk_id]
#             chunk["chunk_id"] = chunk_id
#             merged_results.append(chunk)

#         return merged_results

#     def set_params(self, alpha=None, rrf_k=None):
#         """動態調整參數"""
#         if alpha is not None:
#             self.alpha = alpha
#         if rrf_k is not None:
#             self.rrf_k = rrf_k

class DenseRetriever:
    def __init__(self, chunks, language, chroma_manager=None):
        self.chunks = chunks
        self.language = language

        # ChromaDB 檢索器
        self.chroma_manager = chroma_manager
        self.collection_name = f"docs_{language}" if chroma_manager else None


    def retrieve(self, query, top_k=3, where_filter=None):
        """
        混合檢索主函數

        Args:
            query: 查詢文本
            top_k: 最終返回數量
            method: 合併方法 ("rrf" 或 "weighted")

        Returns:
            List of chunks with scores
        """
        if self.chroma_manager is None:
            print("沒有初始化chroma")

        # Vector 檢索
        try:
            chroma_results = self.chroma_manager.query_chunks(
                collection_name=self.collection_name, query_text=query, top_k=top_k,where_filter=where_filter
            )

            # 將 ChromaDB 結果轉換為統一格式
            vector_results = self._parse_chroma_results(chroma_results)

        except Exception as e:
            print(f"Vector search failed: {e}, falling back to BM25 only")

        return vector_results

    def _parse_chroma_results(self, chroma_results: Optional[Dict]) -> List[Dict]:
        """
        將 ChromaDB 的查詢結果轉換為統一格式

        ChromaDB 返回格式:
        {
            'ids': [['id1', 'id2', ...]],
            'documents': [['text1', 'text2', ...]],
            'metadatas': [[{...}, {...}, ...]],
            'distances': [[0.5, 0.7, ...]]  # 距離越小越相似
        }
        """
        if not chroma_results:
            return []

        # ChromaDB 返回的是嵌套列表
        ids = chroma_results.get("ids", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]

        results = []
        for i, doc_id in enumerate(ids):
            # 將距離轉換為相似度分數 (距離越小，相似度越高)
            # 使用公式: similarity = 1 - distance
            distance = distances[i]
            similarity_score = 1.0 - distance

            result = {
                "chunk_id": int(doc_id) if doc_id.isdigit() else doc_id,
                "page_content": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "score": similarity_score,
                "distance": distance,
            }
            results.append(result)

        return results

def create_dense_retriever(chunks, language, chroma_manager=None):
    print("Creating Dense Retriever")
    return DenseRetriever(chunks, language, chroma_manager)

def create_bm25_retriever(chunks, language):
    print("Creating BM25 Retriever...")
    return BM25Retriever(chunks, language)

def create_pyserini_retriever(chunks, language):
    print("Creating pyserini Retriever...")
    return PyseriniRetriever(chunks, language)
# def create_retriever(chunks, language, chroma_manager=None, use_hybrid=True):
#     """
#     創建檢索器

#     Args:
#         chunks: 文檔塊列表
#         language: 語言 ("zh" 或 "en")
#         chroma_manager: ChromaDBManager 實例 (可選)
#         use_hybrid: 是否使用混合檢索

#     Returns:
#         BM25Retriever 或 HybridRetriever
#     """
#     if use_hybrid and chroma_manager is not None:
#         print("Creating Hybrid Retriever (BM25 + Vector)...")
#         return HybridRetriever(chunks, language, chroma_manager)
#     else:
#         print("Creating BM25 Retriever...")
#         return BM25Retriever(chunks, language)
