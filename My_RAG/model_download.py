from sentence_transformers import SentenceTransformer

# 1. 指定下載路徑 (您的專案資料夾/models/default_model)
save_path = "./models/all-MiniLM-L6-v2"

print(f"⏳ 正在下載模型到: {save_path} ...")

# 2. 下載並儲存
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save(save_path)

print("✅ 下載完成！請確認 'models/all-MiniLM-L6-v2' 資料夾內有檔案。")