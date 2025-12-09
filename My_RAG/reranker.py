"""
Re-ranking Module using Cross-Encoder
根據 Tutorial4 的建議實作 Two-Stage Retrieval
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-Encoder Re-ranker for Two-Stage Retrieval
    
    Stage 1: BM25/Hybrid retrieves many candidates (high recall)
    Stage 2: Cross-Encoder re-ranks for precision
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化 Re-ranker
        
        Args:
            model_name: Cross-Encoder 模型名稱
                       - BAAI/bge-reranker-v2-m3: 多語言（Tutorial 推薦）
                       - BAAI/bge-reranker-base: 英文
                       - BAAI/bge-reranker-large: 英文（更準但更慢）
        """
        print(f"Loading re-ranker model: {model_name}")
        try:
            self.model = CrossEncoder(model_name)
            print(f"✅ Re-ranker loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load re-ranker: {e}")
            print("💡 Falling back to score-based ranking")
            self.model = None
    
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[Dict]:
        """
        使用 Cross-Encoder 重新排序檢索結果
        
        Args:
            query: 查詢文本
            chunks: 候選文檔列表，每個包含 'page_content' 和 'score'
            top_k: 返回的最終結果數量
            return_scores: 是否在結果中包含 rerank score
            
        Returns:
            重新排序後的 top-k chunks
        """
        if not chunks:
            return []
        
        # 如果模型加載失敗，使用原始分數排序
        if self.model is None:
            return self._fallback_ranking(chunks, top_k)
        
        # 構建 (query, document) 對
        query_doc_pairs = [
            [query, chunk['page_content']] 
            for chunk in chunks
        ]
        
        # Cross-Encoder 計算相關性分數
        try:
            # predict() 返回相關性分數（越高越相關）
            rerank_scores = self.model.predict(query_doc_pairs)
            
            # 按分數排序（降序）
            ranked_indices = np.argsort(rerank_scores)[::-1]
            
            # 取 top-k
            top_indices = ranked_indices[:top_k]
            
            # 構建結果
            reranked_chunks = []
            for idx in top_indices:
                chunk = chunks[idx].copy()
                
                if return_scores:
                    chunk['rerank_score'] = float(rerank_scores[idx])
                    chunk['original_score'] = chunk.get('score', 0)
                
                # 更新 score 為 rerank score
                chunk['score'] = float(rerank_scores[idx])
                
                reranked_chunks.append(chunk)
            
            return reranked_chunks
            
        except Exception as e:
            print(f"⚠️  Re-ranking failed: {e}")
            print("💡 Falling back to original ranking")
            return self._fallback_ranking(chunks, top_k)
    
    def _fallback_ranking(self, chunks: List[Dict], top_k: int) -> List[Dict]:
        """
        Fallback：使用原始檢索分數排序
        """
        # 按原始 score 排序
        sorted_chunks = sorted(
            chunks, 
            key=lambda x: x.get('score', 0), 
            reverse=True
        )
        return sorted_chunks[:top_k]


class HybridRerankRetriever:
    """
    結合 Hybrid Retrieval + Re-ranking 的完整檢索器
    
    Pipeline:
    1. Stage 1: Hybrid (BM25 + Vector) 檢索大量候選（top_k=20-50）
    2. Stage 2: Cross-Encoder re-rank 精選結果（top_k=5）
    """
    
    def __init__(
        self, 
        base_retriever,
        reranker: Reranker = None,
        stage1_top_k: int = 20,
        stage2_top_k: int = 5
    ):
        """
        Args:
            base_retriever: 基礎檢索器（BM25Retriever 或 HybridRetriever）
            reranker: Re-ranking 模型
            stage1_top_k: Stage 1 檢索數量（建議 20-50）
            stage2_top_k: Stage 2 最終返回數量（通常 3-5）
        """
        self.base_retriever = base_retriever
        self.reranker = reranker or Reranker()
        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k
    
    def retrieve(self, query: str, top_k: int = 5, method: str = "rrf") -> List[Dict]:
        """
        Two-Stage Retrieval
        
        Args:
            query: 查詢文本
            top_k: 最終返回數量（會覆蓋 stage2_top_k）
            method: Hybrid 方法（如果使用 HybridRetriever）
            
        Returns:
            Re-ranked top-k chunks
        """
        # Stage 1: 檢索大量候選
        if hasattr(self.base_retriever, 'retrieve'):
            # 檢查是否是 HybridRetriever
            if 'method' in self.base_retriever.retrieve.__code__.co_varnames:
                candidates = self.base_retriever.retrieve(
                    query, 
                    top_k=self.stage1_top_k,
                    method=method
                )
            else:
                # BM25Retriever
                candidates = self.base_retriever.retrieve(
                    query, 
                    top_k=self.stage1_top_k
                )
        else:
            raise ValueError("base_retriever must have retrieve() method")
        
        # Stage 2: Re-rank
        reranked_results = self.reranker.rerank(
            query, 
            candidates, 
            top_k=top_k or self.stage2_top_k
        )
        
        return reranked_results
    
    def set_params(self, alpha=None, rrf_k=None, stage1_top_k=None, stage2_top_k=None):
        """動態調整參數"""
        if hasattr(self.base_retriever, 'set_params'):
            self.base_retriever.set_params(alpha=alpha, rrf_k=rrf_k)
        
        if stage1_top_k is not None:
            self.stage1_top_k = stage1_top_k
        
        if stage2_top_k is not None:
            self.stage2_top_k = stage2_top_k


# 便捷函數
def create_reranker(model_name: str = "BAAI/bge-reranker-v2-m3") -> Reranker:
    """
    創建 Re-ranker
    
    推薦模型（Tutorial4）：
    - BAAI/bge-reranker-v2-m3: 多語言，支援中英文（推薦）
    - BAAI/bge-reranker-base: 英文
    - BAAI/bge-reranker-large: 英文，更準確但更慢
    """
    return Reranker(model_name)


if __name__ == "__main__":
    # 測試 Re-ranker
    print("=" * 60)
    print("Testing Re-ranker")
    print("=" * 60)
    
    # 創建 Re-ranker
    reranker = create_reranker()
    
    # 測試數據
    query = "What is the capital of France?"
    
    chunks = [
        {
            "page_content": "Paris is the capital and most populous city of France.",
            "score": 0.6,
            "chunk_id": 0
        },
        {
            "page_content": "France is a country in Western Europe.",
            "score": 0.8,
            "chunk_id": 1
        },
        {
            "page_content": "The Eiffel Tower is located in Paris.",
            "score": 0.7,
            "chunk_id": 2
        },
        {
            "page_content": "French cuisine is world famous.",
            "score": 0.5,
            "chunk_id": 3
        }
    ]
    
    print(f"\nQuery: {query}")
    print(f"\nOriginal ranking (by score):")
    for i, chunk in enumerate(sorted(chunks, key=lambda x: x['score'], reverse=True), 1):
        print(f"  {i}. [score={chunk['score']:.2f}] {chunk['page_content'][:50]}...")
    
    # Re-rank
    reranked = reranker.rerank(query, chunks, top_k=3, return_scores=True)
    
    print(f"\nAfter re-ranking:")
    for i, chunk in enumerate(reranked, 1):
        print(f"  {i}. [rerank={chunk['rerank_score']:.4f}, orig={chunk['original_score']:.2f}] {chunk['page_content'][:50]}...")
    
    print("\n" + "=" * 60)
    print("Test completed!")
