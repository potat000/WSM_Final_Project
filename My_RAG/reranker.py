from typing import List, Dict
import numpy as np
import requests
from sentence_transformers import CrossEncoder


class RemoteFlagReranker:
    """
    遠程 API Reranker（按照助教提供的接口）
    與官方 FlagReranker 接口相同，但內部調用遠程 API
    """

    def __init__(self, api_url: str = "http://ollama-gateway:11434/rerank"):
        """
        Args:
            api_url: rerank endpoint URL
        """
        self.api_url = api_url
        print(f"✅ Initialized RemoteFlagReranker with API: {api_url}")

    def compute_score(self, pairs, max_length=1024):
        """
        計算 query-document pairs 的相關性分數
        
        Args:
            pairs: list of [text1, text2]
            max_length: 最大長度（API 固定為 1024）
        
        Returns:
            scores: np.ndarray of relevance scores
        """
        MAX_BATCH_SIZE = 32  # API 限制

        if len(pairs) > MAX_BATCH_SIZE:
            # 分批處理
            all_scores = []
            for i in range(0, len(pairs), MAX_BATCH_SIZE):
                batch = pairs[i : i + MAX_BATCH_SIZE]
                batch_scores = self._compute_batch(batch)
                all_scores.extend(batch_scores)
            return np.array(all_scores)
        else:
            scores = self._compute_batch(pairs)
            return np.array(scores)

    def _compute_batch(self, pairs):
        """處理單個批次（≤32 pairs）"""
        payload = {"pairs": [{"text1": a, "text2": b} for a, b in pairs]}

        try:
            resp = requests.post(self.api_url, json=payload, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"API request failed ({resp.status_code}): {resp.text}"
                )
            scores = resp.json()["scores"]
            return scores
        except Exception as e:
            print(f"⚠️  Remote reranker API call failed: {e}")
            return [0.0] * len(pairs)


class Reranker:
    """
    統一的 Reranker 介面
    支持本地 CrossEncoder 和遠程 API
    """

    def __init__(
        self,
        mode: str = "remote",
        api_url: str = "http://ollama-gateway:11434/rerank",
        model_name: str = "BAAI/bge-reranker-v2-m3",
    ):
        """
        初始化 Reranker
        
        Args:
            mode: "remote" (遠程 API) 或 "local" (本地模型)
            api_url: 遠程 API URL
            model_name: 本地模型名稱
        """
        self.model = None
        self.mode = mode

        if mode == "remote":
            print(f"🌐 Using remote reranker API")
            try:
                self.model = RemoteFlagReranker(api_url)
            except Exception as e:
                print(f"❌ Failed to initialize remote reranker: {e}")
                self.model = None
        elif mode == "local":
            print(f"💻 Loading local reranker model: {model_name}")
            try:
                self.model = CrossEncoder(model_name)
                print("✅ Local reranker loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load local reranker: {e}")
                print("💡 Try: pip install sentence-transformers")
                self.model = None
        else:
            raise ValueError(f"不支持的模式: {mode}，請使用 'remote' 或 'local'")

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5,
        return_scores: bool = False,
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
        query_doc_pairs = [[query, chunk["page_content"]] for chunk in chunks]

        try:
            # 根據模式調用不同的方法
            if self.mode == "remote":
                # RemoteFlagReranker 使用 compute_score
                rerank_scores = self.model.compute_score(query_doc_pairs, max_length=1024)
            else:
                # CrossEncoder 使用 predict
                rerank_scores = self.model.predict(query_doc_pairs)

            # 確保是 numpy array
            if not isinstance(rerank_scores, np.ndarray):
                rerank_scores = np.array(rerank_scores)

            # 按分數排序（降序）
            ranked_indices = np.argsort(rerank_scores)[::-1]

            # 取 top-k
            top_indices = ranked_indices[:top_k]

            # 構建結果
            reranked_chunks = []
            for idx in top_indices:
                chunk = chunks[idx].copy()

                if return_scores:
                    chunk["rerank_score"] = float(rerank_scores[idx])
                    chunk["original_score"] = chunk.get("score", 0)

                # 更新 score 為 rerank score
                chunk["score"] = float(rerank_scores[idx])

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
        sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        return sorted_chunks[:top_k]


if __name__ == "__main__":
    # 測試 Reranker
    print("=" * 60)
    print("Testing Reranker")
    print("=" * 60)

    # 測試數據
    query = "What is the capital of France?"

    chunks = [
        {
            "page_content": "Paris is the capital and most populous city of France.",
            "score": 0.6,
            "chunk_id": 0,
        },
        {
            "page_content": "France is a country in Western Europe.",
            "score": 0.8,
            "chunk_id": 1,
        },
        {
            "page_content": "The Eiffel Tower is located in Paris.",
            "score": 0.7,
            "chunk_id": 2,
        },
    ]

    print(f"\nQuery: {query}")
    print("\nOriginal ranking:")
    for i, chunk in enumerate(sorted(chunks, key=lambda x: x["score"], reverse=True), 1):
        print(f"  {i}. [score={chunk['score']:.2f}] {chunk['page_content'][:50]}...")

    # 測試本地模式（如果有安裝 sentence-transformers）
    try:
        print("\n--- Testing Local Mode ---")
        reranker = Reranker(mode="local")
        reranked = reranker.rerank(query, chunks, top_k=3, return_scores=True)
        
        print("\nAfter re-ranking (local):")
        for i, chunk in enumerate(reranked, 1):
            print(f"  {i}. [rerank={chunk.get('rerank_score', 0):.4f}] {chunk['page_content'][:50]}...")
    except:
        print("\n⚠️  Local reranker not available")

    print("\n" + "=" * 60)
    print("Test completed!")