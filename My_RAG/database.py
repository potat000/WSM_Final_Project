import chromadb
from typing import List, Dict, Union, Optional, Any
import os # 用於檢查持久化路徑

class ChromaDBManager:
    """
    用於管理 ChromaDB 客戶端和數據 Collection 的類別。
    負責初始化持久化客戶端，並提供批量數據存儲功能。
    """

    def __init__(self, persist_directory: str = "./my_vector_db", collection_names: List[str] = None):
        """
        初始化 ChromaDB 客戶端，並創建指定的 Collection。

        Args:
            persist_directory: 數據持久化存儲的路徑。
            collection_names: 要創建或獲取的 Collection 名稱列表（例如: ['documents_zh', 'documents_en']）。
        """
        self.persist_directory = persist_directory
        self.client: Optional[chromadb.Client] = None
        self.collections: Dict[str, chromadb.Collection] = {}

        # 1. 初始化 ChromaDB 客戶端
        try:
            # 確保路徑存在
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            print(f"✅ ChromaDB 客戶端初始化成功，數據存儲在: {self.persist_directory}")
        except Exception as e:
            print(f"❌ 錯誤: 初始化 Chroma 客戶端失敗。{e}")
            return

        # 2. 創建或獲取 Collection
        if collection_names:
            self._initialize_collections(collection_names)

    def _initialize_collections(self, collection_names: List[str]):
        """創建或獲取指定的 Collection，並將其實例儲存在類別屬性中。"""
        if self.client is None:
            return

        for name in collection_names:
            try:
                collection = self.client.get_or_create_collection(name=name)
                self.collections[name] = collection
                print(f"   -> Collection '{name}' 已準備就緒。")
            except Exception as e:
                print(f"❌ 錯誤: 無法創建或獲取 Collection '{name}': {e}")
    
    def get_collection(self, name: str) -> Optional[chromadb.Collection]:
        """通過名稱獲取已初始化的 Collection 實例。"""
        return self.collections.get(name)

    def save_chunks_to_chroma(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: List[Dict[str, Union[str, int, float]]],
        ids: List[str],
        batch_size: int = 500
    ) -> bool:
        """
        將文本區塊、向量和元數據存入指定的 ChromaDB Collection。
        
        Args:
            collection_name: 要存入的 Collection 名稱。
            ... (其他參數定義與之前相同)
            
        Returns:
            bool: 存儲操作是否成功。
        """
        if self.client is None:
            print("❌ 錯誤: Chroma 客戶端未初始化。無法存儲數據。")
            return False
            
        # 嘗試從已初始化的列表獲取 Collection，如果不存在，則動態創建
        collection = self.collections.get(collection_name)
        if collection is None:
            try:
                collection = self.client.get_or_create_collection(name=collection_name)
                self.collections[collection_name] = collection # 存入字典以供後續使用
            except Exception as e:
                print(f"❌ 錯誤: 無法獲取或創建 Collection '{collection_name}': {e}")
                return False

        try:
            total_chunks = len(ids)
            if total_chunks == 0:
                print(f"ℹ️ 警告: 待存儲數據為空。Collection: '{collection_name}'。")
                return True
                
            print(f"🚀 開始存儲 {total_chunks} 條數據到 Collection: '{collection_name}'...")

            # 批量寫入 (Batch Processing) 以優化性能
            for i in range(0, total_chunks, batch_size):
                end_index = min(i + batch_size, total_chunks)
                
                # 獲取當前批次的數據
                batch_ids = ids[i:end_index]
                batch_texts = texts[i:end_index]
                #batch_embeddings = embeddings[i:end_index]
                batch_metadatas = metadatas[i:end_index]
                
                # 執行寫入操作
                collection.add(
                    #embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                )
                print(f"  -> 已完成存儲: {end_index}/{total_chunks} 條")

            print(f"✅ 成功將所有數據存入 Collection: '{collection_name}'。")
            return True

        except Exception as e:
            print(f"❌ 存儲數據到 ChromaDB 發生錯誤: {e}")
            return False
        
    def query_chunks(
            self,
            collection_name: str,
            query_text: str,
            top_k: int = 5,
            where_filter: Optional[Dict[str, Any]] = None
        ) -> Optional[Dict[str, List[Any]]]:
            """
            在指定的 Collection 中執行向量相似度檢索。
            
            Args:
                collection_name: 要查詢的 Collection 名稱（例如: 'zh'）。
                query_text: 用戶輸入的查詢字串。
                top_k: 希望返回的最相關的 Chunk 數量。
                where_filter: 用於 Metadata 過濾的字典（例如: {"company": {"$eq": "TSMC"}}）。

            Returns:
                Optional[Dict]: ChromaDB 返回的檢索結果字典，包含 IDs, documents, metadatas 等。
                            如果 Collection 不存在或發生錯誤，則返回 None。
            """
            collection = self.get_collection(collection_name)
            if collection is None:
                print(f"❌ 錯誤: Collection '{collection_name}' 不存在或未初始化。")
                return None

            print(f"\n🔍 正在 Collection '{collection_name}' 中檢索...")
            print(f"   - 查詢: {query_text}")
            print(f"   - 數量: Top {top_k}")
            
            if where_filter:
                print(f"   - 過濾條件 (Where): {where_filter}")
            
            try:
                # 執行 ChromaDB 的 query 函式
                results = collection.query(
                    query_texts=[query_text],  # 查詢文本列表
                    n_results=top_k,           # 返回的結果數量
                    where=where_filter         # Metadata 過濾條件
                )
                
                # 檢查結果是否為空
                if not results or not results['documents'] or not results['documents'][0]:
                    print("ℹ️ 檢索結果為空。")
                    return None
                
                print(f"✅ 檢索成功，找到 {len(results['documents'][0])} 條結果。")
                return results
            
            except Exception as e:
                print(f"❌ 錯誤: 執行檢索查詢時發生錯誤: {e}")
                return None