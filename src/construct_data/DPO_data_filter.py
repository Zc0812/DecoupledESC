import json
from loguru import logger
import os

path = "path"
save_path = "save_path"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_data(path, data):
    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode, encoding='utf-8') as f:
        if mode == 'a':
            f.write('\n') 
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Data saved to {path} in {'append' if mode == 'a' else 'write'} mode.")
    
datas = load_json(path)

Yes_num = 0

result = []

for i, d in enumerate(datas):

    try:
        if d["remained"]["Should the sample be retained"] == "Yes":
            Yes_num += 1
            result.append(d)
    except Exception as e:
        print(d)

print(f"Yes_num: {Yes_num} \nlen(datas): {len(datas)}")
save_data(save_path, result)

