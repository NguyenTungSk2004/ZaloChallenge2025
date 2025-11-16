from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from ultralytics import YOLO
import json 
import csv
import os
import logging
import warnings
import gc  # Garbage collector

# TẮT LOGGING
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig
)

from sentence_transformers import CrossEncoder

from modules.tracker import BestFrameTracker
from modules.extract_frames import extract_frames_to_queue
from modules.vlm import describe_frame_with_prompt
from config import USE_GPU, USE_BLIP2_GPU
from modules.qa import lm_generate

# ------------------------------------------------------------
# 1. NF4 QUANTIZATION CONFIG
# ------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# ------------------------------------------------------------
# 2. LOAD BLIP2 VLM (4-BIT NF4)
# ------------------------------------------------------------
model_path = "models/blip2-opt-2.7b"
device_blip = "cuda" if torch.cuda.is_available() and USE_BLIP2_GPU else "cpu"
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

# ------------------------------------------------------------
# YOLO, EMBEDDING, RERANKER
# ------------------------------------------------------------
model_path_yolo = "models/yolo/best.pt"
yolo_detector = YOLO(model_path_yolo)
EMB_PATH = "models/bkai_vn_bi_encoder"
embeddings = HuggingFaceEmbeddings(
    model_name=EMB_PATH,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': False}
)

DB_PATH = "Vecto_Database/db_bienbao_2"
vectordb = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

RERANK_PATH = "models/ViRanker"
device = "cuda" if torch.cuda.is_available() and USE_BLIP2_GPU else "cpu"
reranker = CrossEncoder(RERANK_PATH, device=device)
# ------------------------------------------------------------
# 3. LOAD PHI-3 MINI (4BIT NF4)
# ------------------------------------------------------------
LLM_PATH = "microsoft/phi-3-mini-4k-instruct"

llm = None
tokenizer = None

try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16  # Thêm để tiết kiệm memory
    )

except Exception as e:
    print(f"LỖI KHI TẢI LLM: {e}")
    exit(1)

# ------------------------------------------------------------
# HELPER: MEMORY CLEANUP
# ------------------------------------------------------------
def cleanup_memory():
    """Dọn dẹp memory sau mỗi case"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_memory_usage():
    """In memory usage hiện tại"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  [MEM] GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# ------------------------------------------------------------
# LOGIC XỬ LÝ CHÍNH
# ------------------------------------------------------------
INPUT_JSON_FILE = 'public_test/public_test.json'
OUTPUT_CSV_FILE = 'submission.csv'
VLM_PROMPT_TEMPLATE = (
    "Question: Describe the environment and context of the car. "
    "Also, what is the **exact text or symbol** visible on the traffic sign associated with the bounding box {bbox}? Answer:"
)

# 1. Đọc dữ liệu JSON
try:
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        full_data_object = json.load(f)
    data_list = full_data_object.get('data', [])
except Exception as e:
    exit(1)

# GIỚI HẠN CHỈ TEST 5 CÂU ĐẦU TIÊN
data_list = data_list[:5]

submission_results = []
total_cases = len(data_list)

# 2. Lặp qua từng test case
for index, test_case in enumerate(data_list):
    case_id = test_case['id']
    video_path = test_case['video_path']
    question = test_case['question']
    choices = test_case['choices']
    
    print(f"\n{'='*60}")
    print(f"Case {index + 1}/{total_cases}: ID={case_id}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"{'='*60}")

    # 2.1. EXTRACT FRAMES & TRACKING
    tracker = BestFrameTracker()
    
    try:
        frames_queue = extract_frames_to_queue(video_path)
    except Exception as e:
        submission_results.append({'id': case_id, 'answer': "ERR"})
        cleanup_memory()
        continue

    frame_count = 0
    while True:
        frame = frames_queue.get()
        if frame is None:
            break
        frame_count += 1

        results = yolo_detector.track(frame, tracker="bytetrack.yaml", verbose=False)
        if not results or len(results) == 0:
            continue

        for box in results[0].boxes:
            if box.id is None:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            track_id = int(box.id)
            conf = float(box.conf)
            cls_id = int(box.cls)
            cls_name = results[0].names[cls_id]

            tracker.update_track(frame, track_id, (x1, y1, x2, y2), conf, cls_name)

    # 2.2. VLM CAPTIONING
    all_caption = ""

    for track_id, frameData in tracker.best_frames.items():
        box = frameData.box_info
        prompt = VLM_PROMPT_TEMPLATE.format(bbox=box.bbox)
        
        try:
            caption = describe_frame_with_prompt(frameData.frame, prompt, processor, model)
            all_caption += (
                f"\nCaption Frame {track_id}: {caption} "
                f"Information: [label: '{box.class_name}', score: {frameData.score:.3f}]"
            )
        except Exception as e:
            continue

    vlm_description = all_caption.strip()
    
    # Truncate nếu quá dài (giới hạn ~2000 chars để tránh OOM)
    if len(vlm_description) > 2000:
        vlm_description = vlm_description[:2000] + "..."
    
    # 2.3. LLM FINAL ANSWER
    llm_question_with_choices = question + "\n" + "\n".join(choices)
    
    try:
        # Debug: print question
        
        final_answer_raw = lm_generate(
            llm=llm,
            tokenizer=tokenizer,
            retriever=retriever,
            reranker=reranker,
            vlm_description=vlm_description,
            question=llm_question_with_choices,
        )
        
        # Xử lý đặc biệt cho OOM
        if final_answer_raw == "OOM":
            answer_letter = "ERR"
        else:
            # Trích xuất chữ cái đáp án
            answer_letter = final_answer_raw.strip().upper()
            
            # Loại bỏ dấu chấm
            if '.' in answer_letter:
                answer_letter = answer_letter.split('.')[0].strip()
            
            # Tìm chữ cái A/B/C/D đầu tiên
            found = False
            for char in answer_letter:
                if char in ['A', 'B', 'C', 'D']:
                    answer_letter = char
                    found = True
                    break
            
            if not found:
                # Fallback: lấy ký tự đầu
                if len(answer_letter) > 0:
                    answer_letter = answer_letter[0]
                else:
                    answer_letter = "UNK"
            
            # Validate cuối cùng
            if answer_letter not in ['A', 'B', 'C', 'D']:
                answer_letter = "UNK"

    except Exception as e:
        import traceback
        traceback.print_exc()
        answer_letter = "ERR"

    print(f"Final answer: {answer_letter}")    
    print_memory_usage()
    
    # 3. Lưu kết quả
    submission_results.append({
        'id': case_id,
        'answer': answer_letter
    })
    
    # 4. Cleanup memory sau mỗi case
    cleanup_memory()
    

# 5. Viết kết quả ra CSV

with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(submission_results)

# 6. Thống kê kết quả
answer_stats = {}
for result in submission_results:
    ans = result['answer']
    answer_stats[ans] = answer_stats.get(ans, 0) + 1

print(f"\nFile output: {OUTPUT_CSV_FILE}")
print("Hoàn tất!")