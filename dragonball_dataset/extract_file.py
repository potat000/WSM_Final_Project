import json
import random

def random_sample_data(input_file, output_zh_file, output_en_file, sample_size=100):
    # 建立兩個暫存清單來放所有的候選資料
    zh_pool = []
    en_pool = []
    
    print(f"正在讀取並分類檔案: {input_file} (這可能需要一點時間)...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                
                try:
                    record = json.loads(line)
                    lang = record.get('language')
                    
                    # 將資料分類放入池中
                    if lang == 'zh':
                        zh_pool.append(record)
                    elif lang == 'en':
                        en_pool.append(record)
                        
                except json.JSONDecodeError:
                    continue

        print(f"讀取完畢。統計結果：")
        print(f"- 中文總數: {len(zh_pool)} 筆")
        print(f"- 英文總數: {len(en_pool)} 筆")

        # --- 隨機抽樣邏輯 ---
        
        # 1. 處理中文
        # 如果池子裡的資料少於目標數量 (200)，就全拿；否則隨機抽 200 筆
        current_zh_limit = min(len(zh_pool), sample_size)
        zh_samples = random.sample(zh_pool, current_zh_limit)
        
        # 2. 處理英文
        current_en_limit = min(len(en_pool), sample_size)
        en_samples = random.sample(en_pool, current_en_limit)

        # --- 寫入檔案 ---
        
        print(f"正在寫入隨機抽樣後的中文資料 ({len(zh_samples)} 筆)...")
        with open(output_zh_file, 'w', encoding='utf-8') as f:
            for item in zh_samples:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"正在寫入隨機抽樣後的英文資料 ({len(en_samples)} 筆)...")
        with open(output_en_file, 'w', encoding='utf-8') as f:
            for item in en_samples:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        print("隨機抽樣完成！")

    except FileNotFoundError:
        print(f"錯誤: 找不到檔案 {input_file}")

# --- 設定區 ---
input_filename = 'dragonball_queries.jsonl'  # 您的來源檔案
output_zh = 'queries_new/queries_zh_100.jsonl'
output_en = 'queries_new/queries_en_100.jsonl'

if __name__ == "__main__":
    # 設定隨機種子 (Optional): 如果希望每次執行抽出來的結果都一樣，可以解開下面這行
    random.seed(42) 
    random_sample_data(input_filename, output_zh, output_en)