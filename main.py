from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from ultralytics import YOLO
# NEW IMPORTS FOR 4BIT NF4
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig
)

from sentence_transformers import CrossEncoder

# ❌ COMMENT Llamacpp – no longer used
# from langchain_community.llms import LlamaCpp

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
    bnb_4bit_quant_type="nf4",        # NF4 quantization
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# ------------------------------------------------------------
# 2. LOAD BLIP2 VLM (4-BIT NF4)
# ------------------------------------------------------------
model_path = "models/blip2-opt-2.7b"
device_blip = "cuda" if torch.cuda.is_available() and USE_BLIP2_GPU else "cpu"

processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

# OLD (FP16 FLOAT)
# model = AutoModelForImageTextToText.from_pretrained(
#     model_path,
#     dtype=torch.float16 if device_blip == "cuda" else torch.float32,
#     low_cpu_mem_usage=True
# ).to(device_blip)

#  NEW – QUANTIZED NF4
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

# ------------------------------------------------------------
# YOLO
# ------------------------------------------------------------
model_path_yolo = "models/yolo/best.pt"
yolo_detector = YOLO(model_path_yolo)
tracker = BestFrameTracker()

# ------------------------------------------------------------
# EMBEDDING + CHROMA
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# RERANKER
# ------------------------------------------------------------
RERANK_PATH = "models/ViRanker"
device = "cuda" if torch.cuda.is_available() and USE_BLIP2_GPU else "cpu"
reranker = CrossEncoder(RERANK_PATH, device=device)

# ------------------------------------------------------------
# 3. LOAD PHI-3 MINI (TRANSFORMERS, 4BIT NF4)
# ------------------------------------------------------------

# OLD LlamaCpp (COMMENT)
# LLM_PATH = "models/Phi-3-gguf/Phi-3-mini-4k-instruct-q4.gguf"
# llm = AutoModelForCausalLM.from_pretrained(... llamacpp ...)

#  NEW:
LLM_PATH = "microsoft/phi-3-mini-4k-instruct"

print("Đang load Phi-3-mini NF4 bằng transformers...")

tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    quantization_config=bnb_config,   # QUANTIZED 4BIT NF4
    device_map="auto",
    low_cpu_mem_usage=True
)

print("Phi-3 đã load xong!")

# ------------------------------------------------------------
# TEST CASE
# ------------------------------------------------------------
test_case = {
            "id": "train_0002",
            "question": "Phần đường trong video cho phép các phương tiện đi theo hướng nào khi đến nút giao?",
            "choices": [
                "A. Đi thẳng",
                "B. Đi thẳng và rẽ phải",
                "C. Đi thẳng, rẽ trái và rẽ phải",
                "D. Rẽ trái và rẽ phải"
            ],
            "video_path": "train/videos/2b840c67_386_clip_002_0008_0018_Y.mp4",
}

# ------------------------------------------------------------
# EXTRACT FRAMES
# ------------------------------------------------------------
frames_queue = extract_frames_to_queue(test_case['video_path'])
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

# ------------------------------------------------------------
# VLM CAPTIONING
# ------------------------------------------------------------
all_caption = ""

# SỬA PROMPT TRONG VÒNG LẶP CAPTIONING
for track_id, frameData in tracker.best_frames.items():
    box = frameData.box_info

    # Yêu cầu VLM mô tả CẢ HÌNH ẢNH và ĐỌC NỘI DUNG BIỂN BÁO
    prompt = (
        f"Question: Describe the environment and context of the car. "
        f"Also, what is the **exact text or symbol** visible on the traffic sign associated with the bounding box {box.bbox}? Answer:"
    )

    caption = describe_frame_with_prompt(frameData.frame, prompt, processor, model)

    all_caption += (
        f"\n Caption Frame {track_id}: {caption} "
        f"Information the traffic sign:[label: '{box.class_name}', score: '{frameData.score}']"
    )

vlm_description = all_caption
question = test_case["question"]
choices = test_case["choices"]

# ------------------------------------------------------------
# LLM FINAL ANSWER
# ------------------------------------------------------------
final_answer = lm_generate(
    llm=llm,
    tokenizer=tokenizer,          # REQUIRED khi chuyển từ LlamaCPP sang transformers
    retriever=retriever,
    reranker=reranker,
    vlm_description=vlm_description,
    question=question + "\n" + "\n".join(choices),
)

print("Final Answer:", final_answer)
