import json
import os
from loguru import logger

def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def load_json_lines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_l = json.loads(line.strip())
            data.append(data_l)
    return data
    
def save_data(path, data):
    mode = 'a' if os.path.exists(path) else 'w'
    with open(path, mode, encoding='utf-8') as f:
        if mode == 'a':
            f.write('\n')
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Data saved to {path} in {'append' if mode == 'a' else 'write'} mode.")

def _norm(s):
    return ' '.join(s.strip().split())

