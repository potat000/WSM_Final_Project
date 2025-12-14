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
                if 'court_name' in data:
                    companies.append(data['court_name'])
                if 'hospital_patient_name' in data and data['hospital_patient_name']:
                    # 修改重點：使用 split 切割字串，並只取第一個部分 ([0])
                    # 例如: "雷峰市人民医院_马某某" -> ["雷峰市人民医院", "马某某"] -> 取 "雷峰市人民医院"
                    full_name = data['hospital_patient_name']
                    hospital_name = full_name.split('_')[0]
                    companies.append(hospital_name)

    # 寫入 TXT 檔案
    with open(output_file, 'w', encoding='utf-8') as f:
        for company in companies:
            f.write(company + '\n')

    print(f"成功提取 {len(companies)} 間公司名稱至 {output_file}")

except FileNotFoundError:
    print(f"找不到檔案: {input_file}")
except json.JSONDecodeError:
    print("JSON 解析錯誤，請檢查檔案格式")