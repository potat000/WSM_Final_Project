from typing import Dict, List, Optional

import jieba
import numpy as np
from rank_bm25 import BM25Okapi


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

    def retrieve(self, query, top_k=5):
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
            chunk["chunk_id"] = idx
            results.append(chunk)

        return results


class HybridRetriever:
    """混合檢索器：BM25 + Dense Embedding (ChromaDB)"""

    def __init__(self, chunks, language, chroma_manager=None):
        self.chunks = chunks
        self.language = language

        # BM25 檢索器
        self.bm25_retriever = BM25Retriever(chunks, language)

        # ChromaDB 檢索器
        self.chroma_manager = chroma_manager
        self.collection_name = f"docs_{language}" if chroma_manager else None

        # 可調參數
        self.alpha = 0.7  # BM25 權重 (0-1)，vector 權重為 (1-alpha)
        self.rrf_k = 60  # RRF 平滑參數

    def retrieve(self, query, top_k=3, method="rrf",where_filter=None):
        """
        混合檢索主函數

        Args:
            query: 查詢文本
            top_k: 最終返回數量
            method: 合併方法 ("rrf" 或 "weighted")

        Returns:
            List of chunks with scores
        """
        # 如果沒有 ChromaDB，退回到純 BM25
        if self.chroma_manager is None:
            print("使用純BM25")
            return self.bm25_retriever.retrieve(query, top_k)

        # 1. BM25 檢索 (取 2*top_k 增加覆蓋率)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)

        # 2. Vector 檢索 (取 2*top_k)
        try:
            chroma_results = self.chroma_manager.query_chunks(
                collection_name=self.collection_name, query_text=query, top_k=top_k * 2,where_filter=where_filter
            )

            # 將 ChromaDB 結果轉換為統一格式
            vector_results = self._parse_chroma_results(chroma_results)

            if not vector_results:
                print("Warning: Vector search returned no results, using BM25 only")
                return bm25_results[:top_k]

        except Exception as e:
            print(f"Vector search failed: {e}, falling back to BM25 only")
            return bm25_results[:top_k]

        # 3. 合併結果
        if method == "rrf":
            merged_results = self._reciprocal_rank_fusion(
                bm25_results, vector_results, top_k
            )
        else:  # weighted
            merged_results = self._weighted_merge(bm25_results, vector_results, top_k)

        return merged_results

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
            # 使用公式: similarity = 1 / (1 + distance)
            distance = distances[i] if i < len(distances) else 1.0
            similarity_score = 1.0 / (1.0 + distance)

            result = {
                "chunk_id": int(doc_id) if doc_id.isdigit() else doc_id,
                "page_content": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "score": similarity_score,
                "distance": distance,
            }
            results.append(result)

        return results

    def _weighted_merge(
        self, bm25_results: List[Dict], vector_results: List[Dict], top_k: int
    ) -> List[Dict]:
        """
        方法 A: 加權平均合併
        適合: 當你想明確控制 BM25 和 Vector 的影響力
        """
        scores = {}
        chunk_data = {}

        # 正規化 BM25 分數到 0-1
        max_bm25 = max([r["score"] for r in bm25_results], default=1e-6)
        if max_bm25 == 0:
            max_bm25 = 1e-6

        # 處理 BM25 結果
        for result in bm25_results:
            chunk_id = result["chunk_id"]
            normalized_score = result["score"] / max_bm25
            scores[chunk_id] = self.alpha * normalized_score
            chunk_data[chunk_id] = result

        # 處理 Vector 結果
        for result in vector_results:
            chunk_id = result["chunk_id"]
            vector_score = result["score"]  # 已經是相似度分數 (0-1)

            if chunk_id in scores:
                scores[chunk_id] += (1 - self.alpha) * vector_score
            else:
                scores[chunk_id] = (1 - self.alpha) * vector_score
                chunk_data[chunk_id] = result

        # 排序並構建最終結果
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        merged_results = []
        for chunk_id in sorted_ids[:top_k]:
            chunk = chunk_data[chunk_id].copy()
            chunk["score"] = scores[chunk_id]
            chunk["chunk_id"] = chunk_id
            merged_results.append(chunk)

        return merged_results

    def _reciprocal_rank_fusion(
        self, bm25_results: List[Dict], vector_results: List[Dict], top_k: int
    ) -> List[Dict]:
        """
        方法 B: Reciprocal Rank Fusion (RRF)
        適合: 不確定哪個檢索器更好時，讓數據說話

        公式: score(d) = Σ 1 / (k + rank_i(d))
        """
        scores = {}
        chunk_data = {}

        # 處理 BM25 排名
        for rank, result in enumerate(bm25_results):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = 1.0 / (self.rrf_k + rank + 1)
            chunk_data[chunk_id] = result

        # 處理 Vector 排名 (累加)
        for rank, result in enumerate(vector_results):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (self.rrf_k + rank + 1)

            if chunk_id in scores:
                scores[chunk_id] += rrf_score
                print("叮咚 累加重複計算！")
            else:
                scores[chunk_id] = rrf_score
                chunk_data[chunk_id] = result

        # 排序並構建最終結果
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        merged_results = []
        for chunk_id in sorted_ids[:top_k]:
            chunk = chunk_data[chunk_id].copy()
            chunk["score"] = scores[chunk_id]
            chunk["chunk_id"] = chunk_id
            merged_results.append(chunk)

        return merged_results

    def set_params(self, alpha=None, rrf_k=None):
        """動態調整參數"""
        if alpha is not None:
            self.alpha = alpha
        if rrf_k is not None:
            self.rrf_k = rrf_k

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
        # 如果沒有 ChromaDB，退回到純 BM25
        if self.chroma_manager is None:
            print("沒有初始化chroma")

        # Vector 檢索 (取 2*top_k)
        try:
            chroma_results = self.chroma_manager.query_chunks(
                collection_name=self.collection_name, query_text=query, top_k=top_k * 2,where_filter=where_filter
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

def create_retriever(chunks, language, chroma_manager=None, use_hybrid=True):
    """
    創建檢索器

    Args:
        chunks: 文檔塊列表
        language: 語言 ("zh" 或 "en")
        chroma_manager: ChromaDBManager 實例 (可選)
        use_hybrid: 是否使用混合檢索

    Returns:
        BM25Retriever 或 HybridRetriever
    """
    if use_hybrid and chroma_manager is not None:
        print("Creating Hybrid Retriever (BM25 + Vector)...")
        return HybridRetriever(chunks, language, chroma_manager)
    else:
        print("Creating BM25 Retriever...")
        return BM25Retriever(chunks, language)
