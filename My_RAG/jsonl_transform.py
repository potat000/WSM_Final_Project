import json

# 設定輸入與輸出檔名
input_file = './dragonball_dataset/dragonball_docs.jsonl'  # 您的原始檔案
output_file = './dragonball_dataset/company_names.txt'

companies = []

# 讀取 JSONL 並提取 company_name
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if 'company_name' in data:
                    companies.append(data['company_name'])

    # 寫入 TXT 檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        for company in companies:
            f.write(company + '\n')

    print(f"成功提取 {len(companies)} 間公司名稱至 {output_file}")

except FileNotFoundError:
    print(f"找不到檔案: {input_file}")
except json.JSONDecodeError:
    print("JSON 解析錯誤，請檢查檔案格式")