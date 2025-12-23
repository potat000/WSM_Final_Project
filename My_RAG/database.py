import chromadb
from typing import List, Dict, Union, Optional, Any
import os
import chromadb.utils.embedding_functions as embedding_functions
from generator import load_ollama_config
class ChromaDBManager:
    def __init__(self, persist_directory: str = "./my_vector_db", collection_names: List[str] = None):
        # ==================================================
        # 1. 優先初始化屬性 (這段非常重要，必須放在最上面)
        # ==================================================
        self.persist_directory = persist_directory
        self.client: Optional[chromadb.Client] = None
        self.collections: Dict[str, chromadb.Collection] = {}
        self.embedding_fns: Dict[str, Any] = {}  # 👈 這裡必須先定義它是空字典，下面才能用 ['en']
        self.host = ""
        self.base_url = ""

        # ==================================================
        # 2. 讀取設定
        # ==================================================
        try:
            config = load_ollama_config()
            self.host = config['host']
            # 確保 host 結尾沒有 '/'
            self.base_url = f"{self.host.rstrip('/')}/api/embeddings"
            
            # 從設定檔取得模型名稱
            model_en = config['embedding_model_en']
            model_zh = config['embedding_model_zh']
            
            # 設定 Embedding Functions
            self.embedding_fns['en'] = embedding_functions.OllamaEmbeddingFunction(
                url=self.base_url,
                model_name=model_en
            )
            
            self.embedding_fns['zh'] = embedding_functions.OllamaEmbeddingFunction(
                url=self.base_url,
                model_name=model_zh
            )
            
            print(f"✅ Embedding 設定完成 (Host: {self.base_url})")
            print(f"   - EN: {model_en}")
            print(f"   - ZH: {model_zh}")

        except Exception as e:
            print(f"❌ 設定讀取失敗或 Embedding 初始化錯誤: {e}")
            # 這裡建議讓程式繼續跑，或者根據你的需求決定是否 return
            return

        # ==================================================
        # 3. 初始化 ChromaDB 客戶端
        # ==================================================
        try:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            print(f"✅ ChromaDB 客戶端初始化成功，數據存儲在: {self.persist_directory}")
        except Exception as e:
            print(f"❌ 錯誤: 初始化 Chroma 客戶端失敗。{e}")
            return

        # 4. 創建或獲取 Collection
        if collection_names:
            self._initialize_collections(collection_names)

    def _select_embedding_fn(self, collection_name: str):
        """
        根據 collection 名稱決定使用哪個 Embedding Function。
        """
        name_lower = collection_name.lower()
        if any(keyword in name_lower for keyword in ['zh', 'cn', 'chinese']):
            return self.embedding_fns.get('zh')
        else:
            # 預設使用英文，如果英文也沒有則回傳 None (會報錯)
            return self.embedding_fns.get('en')

    def _initialize_collections(self, collection_names: List[str]):
        if self.client is None:
            return

        for name in collection_names:
            try:
                ef = self._select_embedding_fn(name)
                if ef is None:
                    print(f"⚠️ 無法為 Collection '{name}' 找到合適的 Embedding Function，跳過初始化。")
                    continue

                # 嘗試讀取模型名稱供 Log 使用 (OllamaFunction 物件通常有 _model_name 屬性)
                model_log = getattr(ef, "_model_name", "Unknown")

                collection = self.client.get_or_create_collection(
                    name=name,
                    embedding_function=ef 
                )
                self.collections[name] = collection
                print(f"   -> Collection '{name}' 就緒 (Model: {model_log})")
            except Exception as e:
                print(f"❌ 錯誤: 無法創建 Collection '{name}': {e}")
    
    def get_collection(self, name: str) -> Optional[chromadb.Collection]:
        return self.collections.get(name)

    def save_chunks_to_chroma(self, collection_name: str, texts: List[str], metadatas: List[Dict], ids: List[str], batch_size: int = 500) -> bool:
        if self.client is None: return False
        
        collection = self.collections.get(collection_name)
        if collection is None:
            # 嘗試 Lazy Load
            try:
                ef = self._select_embedding_fn(collection_name)
                if ef is None: raise ValueError("No embedding function found")
                collection = self.client.get_or_create_collection(name=collection_name, embedding_function=ef)
                self.collections[collection_name] = collection
            except Exception as e:
                print(f"❌ 錯誤: 無法獲取 Collection '{collection_name}': {e}")
                return False

        try:
            total_chunks = len(ids)
            if total_chunks == 0: return True
            print(f"🚀 [Chroma] 開始存儲 {total_chunks} 筆資料到 '{collection_name}'...")

            for i in range(0, total_chunks, batch_size):
                end_index = min(i + batch_size, total_chunks)
                collection.add(
                    documents=texts[i:end_index],
                    metadatas=metadatas[i:end_index],
                    ids=ids[i:end_index],
                )
                print(f"   -> 進度: {end_index}/{total_chunks}")

            print(f"✅ 存儲完成: '{collection_name}'")
            return True
        except Exception as e:
            print(f"❌ 存儲失敗: {e}")
            return False

    def query_chunks(self, collection_name: str, query_text: str, top_k: int = 5, where_filter: Optional[Dict] = None) -> Optional[Dict]:
        collection = self.get_collection(collection_name)
        if not collection:
            print(f"⚠️ Collection '{collection_name}' 尚未載入。")
            return None
        try:
            return collection.query(query_texts=[query_text], n_results=top_k, where=where_filter)
        except Exception as e:
            print(f"❌ 查詢失敗: {e}")
            return None
        



# class ChromaDBManager:
#     """
#     用於管理 ChromaDB 客戶端和數據 Collection 的類別。
#     支援多語言（中/英）使用不同的 Embedding Model。
#     """

#     def __init__(self, persist_directory: str = "./my_vector_db", collection_names: List[str] = None):
#         self.persist_directory = persist_directory
#         self.client: Optional[chromadb.Client] = None
#         self.collections: Dict[str, chromadb.Collection] = {}
        
#         # 存放不同語言的 Embedding Function
#         self.embedding_fns: Dict[str, Any] = {}

#         # =================================================
#         # 1. 設定模型路徑 & 初始化 Embedding Functions
#         # =================================================
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         models_root = os.path.join(current_dir, "..", "models")

#         # --- 設定英文模型 (Example: all-MiniLM-L6-v2) ---
#         en_model_path = os.path.join(models_root, "all-MiniLM-L6-v2")
#         try:
#             # 如果本地沒模型，這裡可以改回用預設下載，或者保持報錯
#             self.embedding_fns['en'] = embedding_functions.SentenceTransformerEmbeddingFunction(
#                 model_name=en_model_path 
#             )
#             print(f"✅ 英文 Embedding 模型載入成功: {en_model_path}")
#         except Exception as e:
#             print(f"⚠️ 英文模型載入失敗 (將無法處理英文 Collection): {e}")

#         # --- 設定中文模型 (Example: text2vec-base-chinese or bge-large-zh) ---
#         # 假設你有下載中文模型放在 '../models/text2vec-base-chinese'
#         zh_model_path = os.path.join(models_root, "bge-small-zh-v1.5") 
#         try:
#             self.embedding_fns['zh'] = embedding_functions.SentenceTransformerEmbeddingFunction(
#                 model_name=zh_model_path
#             )
#             print(f"✅ 中文 Embedding 模型載入成功: {zh_model_path}")
#         except Exception as e:
#             print(f"⚠️ 中文模型載入失敗 (或是路徑錯誤)，請確認 path: {zh_model_path}")
#             # 如果沒有專用中文模型，可以 fallback 到英文模型 (視需求而定)
#             if 'en' in self.embedding_fns:
#                  print("   -> 將使用英文模型作為中文的備用方案。")
#                  self.embedding_fns['zh'] = self.embedding_fns['en']
#             else:
#                  raise e

#         # 2. 初始化 ChromaDB 客戶端
#         try:
#             os.makedirs(persist_directory, exist_ok=True)
#             self.client = chromadb.PersistentClient(path=self.persist_directory)
#             print(f"✅ ChromaDB 客戶端初始化成功，數據存儲在: {self.persist_directory}")
#         except Exception as e:
#             print(f"❌ 錯誤: 初始化 Chroma 客戶端失敗。{e}")
#             return

#         # 3. 創建或獲取 Collection
#         if collection_names:
#             self._initialize_collections(collection_names)

#     def _select_embedding_fn(self, collection_name: str):
#         """
#         根據 collection 名稱決定使用哪個 Embedding Function。
#         規則: 
#           - 名稱包含 'zh', 'cn', 'chinese' -> 使用中文模型
#           - 其他 -> 使用英文模型
#         """
#         name_lower = collection_name.lower()
#         if any(keyword in name_lower for keyword in ['zh', 'cn', 'chinese']):
#             return self.embedding_fns.get('zh')
#         else:
#             return self.embedding_fns.get('en')

#     def _initialize_collections(self, collection_names: List[str]):
#         if self.client is None:
#             return

#         for name in collection_names:
#             try:
#                 # 動態選擇對應的 Embedding Function
#                 ef = self._select_embedding_fn(name)
#                 print(name)
#                 print(ef)
#                 collection = self.client.get_or_create_collection(
#                     name=name,
#                     embedding_function=ef # 這裡傳入對應語言的 function
#                 )
#                 self.collections[name] = collection
#                 print(f"   -> Collection '{name}' 已準備就緒 (使用模型: {ef.models if hasattr(ef, 'models') else 'Unknown'})。")
#             except Exception as e:
#                 print(f"❌ 錯誤: 無法創建或獲取 Collection '{name}': {e}")
    
#     def get_collection(self, name: str) -> Optional[chromadb.Collection]:
#         return self.collections.get(name)

#     def save_chunks_to_chroma(
#         self,
#         collection_name: str,
#         texts: List[str],
#         metadatas: List[Dict[str, Union[str, int, float]]],
#         ids: List[str],
#         batch_size: int = 500
#     ) -> bool:
#         if self.client is None:
#             return False
            
#         collection = self.collections.get(collection_name)
#         if collection is None:
#             try:
#                 # 動態選擇 Embedding Function
#                 ef = self._select_embedding_fn(collection_name)
#                 collection = self.client.get_or_create_collection(name=collection_name, embedding_function=ef)
#                 self.collections[collection_name] = collection
#             except Exception as e:
#                 print(f"❌ 錯誤: 無法獲取或創建 Collection '{collection_name}': {e}")
#                 return False

#         try:
#             total_chunks = len(ids)
#             if total_chunks == 0:
#                 return True
                
#             print(f"🚀 開始存儲 {total_chunks} 條數據到 Collection: '{collection_name}'...")

#             for i in range(0, total_chunks, batch_size):
#                 end_index = min(i + batch_size, total_chunks)
#                 batch_ids = ids[i:end_index]
#                 batch_texts = texts[i:end_index]
#                 batch_metadatas = metadatas[i:end_index]
                
#                 # ChromaDB 會自動使用 create_collection 時綁定的 embedding_function 來計算向量
#                 collection.add(
#                     documents=batch_texts,
#                     metadatas=batch_metadatas,
#                     ids=batch_ids,
#                 )
#                 print(f"  -> 已完成存儲: {end_index}/{total_chunks} 條")

#             print(f"✅ 成功將所有數據存入 Collection: '{collection_name}'。")
#             return True

#         except Exception as e:
#             print(f"❌ 存儲數據到 ChromaDB 發生錯誤: {e}")
#             return False

#     # query_chunks 方法不需要修改，因為 Collection 已經記住了它該用哪個 Embedding Function
#     def query_chunks(self, collection_name: str, query_text: str, top_k: int = 5, where_filter: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, List[Any]]]:
#         # ... (保持原本的代碼) ...
#         collection = self.get_collection(collection_name)
#         # ...
#         results = collection.query(
#             query_texts=[query_text], # 這裡 Chroma 會自動呼叫該 Collection 對應的中文或英文模型來轉向量
#             n_results=top_k,
#             where=where_filter
#         )
#         # ...
#         return results