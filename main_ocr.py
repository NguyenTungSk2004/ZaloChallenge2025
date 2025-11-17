from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from ultralytics import YOLO
import json 
import csv
import os
import logging
import warnings
import gc

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
from modules.ocr_helper import TrafficSignOCR 
from modules.qa import lm_generate

# ============================================================
# QUANTIZATION CONFIG
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# ============================================================
# LOAD MODELS
# ============================================================
print("="*60)
print("LOADING MODELS")
print("="*60)

# 1. BLIP2 VLM
print("\n[1/6] Loading BLIP2 VLM...")
model_path = "models/blip2-opt-2.7b"
device_blip = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("✓ BLIP2 loaded!")

# 2. OCR READER - NEW!
print("\n[2/6] Loading OCR")
ocr_reader = None
try:
    ocr_reader = TrafficSignOCR(backend='easy', use_gpu=True)
except Exception as e:
    print(f"  Warning: OCR failed({e})")

# 3. YOLO
print("\n[3/6] Loading YOLO...")
model_path_yolo = "models/yolo/best.pt"
yolo_detector = YOLO(model_path_yolo)
print("✓ YOLO loaded!")

# 4. EMBEDDINGS & VECTORDB
print("\n[4/6] Loading Embeddings & VectorDB...")
EMB_PATH = "models/bkai-foundation-models/bkai_vn_bi_encoder"
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
print("✓ VectorDB loaded!")

# 5. RERANKER
print("\n[5/6] Loading Reranker...")
RERANK_PATH = "models/namdp-ptit/ViRanker"
device = "cuda" if torch.cuda.is_available() else "cpu"
reranker = CrossEncoder(RERANK_PATH, device=device)
print("✓ Reranker loaded!")

# 6. PHI-3 MINI LLM
print("\n[6/6] Loading Phi-3 Mini...")
LLM_PATH = "models/microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("✓ Phi-3 loaded!")

print("\n" + "="*60)
print("ALL MODELS LOADED!")
print("="*60 + "\n")

# ============================================================
# CLEANUP HELPER
# ============================================================
def cleanup_memory():
    """Dọn dẹp memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ============================================================
# MAIN PROCESSING
# ============================================================
INPUT_JSON_FILE = 'public_test/public_test.json'
OUTPUT_CSV_FILE = 'submission.csv'

# VLM Prompt template
VLM_PROMPT_TEMPLATE = (
    "Question: Describe the traffic sign. What type is it? "
    "What information does it show? Answer:"
)

# Read test data
try:
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        full_data_object = json.load(f)
    data_list = full_data_object.get('data', [])
except Exception as e:
    print(f"Lỗi khi đọc file JSON: {e}")
    exit(1)

submission_results = []
total_cases = len(data_list)

print(f"\n{'='*60}")
print(f"BẮT ĐẦU XỬ LÝ {total_cases} TEST CASES")
print(f"OCR: {'ENABLED ✓' if ocr_reader else 'DISABLED ✗'}")
print(f"{'='*60}\n")

# Process each case
for index, test_case in enumerate(data_list):
    case_id = test_case['id']
    video_path = test_case['video_path']
    question = test_case['question']
    choices = test_case['choices']
    
    print(f"\n{'='*60}")
    print(f"Case {index + 1}/{total_cases}: ID={case_id}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    
    # STEP 1: EXTRACT FRAMES & TRACKING
    tracker = BestFrameTracker()
    
    try:
        frames_queue = extract_frames_to_queue(video_path)
    except Exception as e:
        print(f"[ERROR] Lỗi đọc video: {e}")
        submission_results.append({'id': case_id, 'answer': "ERR"})
        cleanup_memory()
        continue

    print("[STEP 1/3] Tracking objects...")
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
    
    print(f"  Tracked {len(tracker.best_frames)} signs from {frame_count} frames")

    # STEP 2: VLM CAPTIONING + OCR
    print("[STEP 2/3] VLM + OCR processing...")
    all_caption = ""

    for track_id, frameData in tracker.best_frames.items():
        box = frameData.box_info
        
        # 2.1. VLM Caption
        prompt = VLM_PROMPT_TEMPLATE.format(bbox=box.bbox)
        vlm_caption = describe_frame_with_prompt(frameData.frame, prompt, processor, model)
        
        # 2.2. OCR Text (NEW!)
        ocr_text = ""
        if ocr_reader is not None:
            try:
                ocr_text = ocr_reader.read_text_from_roi(
                    frame=frameData.frame,
                    bbox=box.bbox,
                    conf_threshold=0.5,  # Confidence threshold
                    enhance=True,        # Tăng cường ảnh
                    padding=5            # Padding xung quanh bbox
                )
                if ocr_text:
                    ocr_text = ocr_reader.clean_traffic_text(ocr_text)
            except Exception as e:
                print(f"    [OCR Warning] Track {track_id}: {e}")
                ocr_text = ""
        
        # 2.3. Combine VLM + OCR
        description = (
            f"\nSign {track_id}: {vlm_caption} "
            f"[Class: {box.class_name}, Conf: {frameData.score:.2f}]"
        )
        
        if ocr_text:
            description += f"\n  OCR Text: '{ocr_text}'"
            print(f"  Track {track_id}: OCR detected '{ocr_text}'")
        
        all_caption += description

    vlm_description = all_caption.strip()
    
    # Truncate nếu quá dài
    if len(vlm_description) > 2000:
        vlm_description = vlm_description[:2000] + "..."
        print(f"  [INFO] Description truncated to 2000 chars")
    
    print(f"  Total description: {len(vlm_description)} chars")
    
    # STEP 3: LLM FINAL ANSWER
    print("[STEP 3/3] LLM answering...")
    llm_question_with_choices = question + "\n" + "\n".join(choices)
    
    try:
        final_answer_raw = lm_generate(
            llm=llm,
            tokenizer=tokenizer,
            retriever=retriever,
            reranker=reranker,
            vlm_description=vlm_description,
            question=llm_question_with_choices,
        )
        
        print(f"  [DEBUG] LLM output: '{final_answer_raw}'")
        
        # Extract answer
        if final_answer_raw == "OOM":
            print("  [ERROR] Out of Memory!")
            answer_letter = "ERR"
        else:
            answer_letter = final_answer_raw.strip().upper()
            
            if '.' in answer_letter:
                answer_letter = answer_letter.split('.')[0].strip()
            
            # Find A/B/C/D
            found = False
            for char in answer_letter:
                if char in ['A', 'B', 'C', 'D']:
                    answer_letter = char
                    found = True
                    break
            
            if not found and len(answer_letter) > 0:
                answer_letter = answer_letter[0]
            
            if answer_letter not in ['A', 'B', 'C', 'D']:
                print(f"  [WARNING] Invalid answer '{answer_letter}', using UNK")
                answer_letter = "UNK"

    except Exception as e:
        print(f"  [ERROR] LLM exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        answer_letter = "ERR"
    
    print(f"  ✓ Final answer: {answer_letter}")
    
    # Save result
    submission_results.append({
        'id': case_id,
        'answer': answer_letter
    })
    
    # Cleanup
    cleanup_memory()
    
    print(f"\nProgress: {index + 1}/{total_cases} ({(index+1)/total_cases*100:.1f}%)")

# Write results to CSV
print(f"\n{'='*60}")
print("GHI KẾT QUẢ RA FILE")
print(f"{'='*60}")

with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['id', 'answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(submission_results)

# Statistics
answer_stats = {}
for result in submission_results:
    ans = result['answer']
    answer_stats[ans] = answer_stats.get(ans, 0) + 1

print(f"\n{'='*60}")
print("THỐNG KÊ KẾT QUẢ:")
print(f"{'='*60}")
print(f"Tổng số cases: {len(submission_results)}")
for ans, count in sorted(answer_stats.items()):
    percentage = count / len(submission_results) * 100
    print(f"  {ans}: {count} cases ({percentage:.1f}%)")
print(f"\nFile output: {OUTPUT_CSV_FILE}")
print("Hoàn tất!")