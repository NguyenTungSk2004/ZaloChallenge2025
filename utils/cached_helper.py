# Cache VLM vào disk
import csv
import hashlib
import json
import os

def save_temp_results(results, temp_file_path):
    """Lưu kết quả tạm thời"""
    sorted_results = sorted(results, key=lambda x: x['index'])
    csv_data = [{'id': r['id'], 'answer': r['answer']} for r in sorted_results]
    
    with open(temp_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'answer'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"💾 Backup: {len(results)} kết quả -> {temp_file_path}")

def get_vlm_cache(video_path):
    """Load VLM description từ cache"""
    cache_dir = "cached_vlm"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.md5(video_path.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)['vlm_description']
    return None

def save_vlm_cache(video_path, vlm_description):
    """Lưu VLM description vào cache"""
    cache_dir = "cached_vlm"
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = hashlib.md5(video_path.encode()).hexdigest()[:8]
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({'vlm_description': vlm_description}, f, ensure_ascii=False)