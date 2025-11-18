from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from ultralytics import YOLO
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig
)
from sentence_transformers import CrossEncoder

def load_models():
    """Load models với tối ưu hóa tốc độ"""
    
    # Clear GPU cache trước khi load
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 1. NF4 QUANTIZATION CONFIG
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,  # Tắt để tăng tốc
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. LOAD BLIP2 VLM
    model_path = "models/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.eval()  # Chỉ eval, không .half() cho quantized model

    # 3. YOLO
    model_path_yolo = "models/yolo/best.pt"
    yolo_detector = YOLO(model_path_yolo)
    yolo_detector.model.eval()

    # 4. EMBEDDING + CHROMA
    # EMB_PATH = "models/bkai-foundation-models/vietnamese-bi-encoder"
    EMB_PATH = "models/bkai-foundation-models/bkai_vn_bi_encoder" # máy thầy
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_PATH,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    DB_PATH = "Vecto_Database/db_bienbao_2"
    vectordb = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 5. RERANKER
    RERANK_PATH = "models/namdp-ptit/ViRanker"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(RERANK_PATH, device=device)

    # 6. LOAD PHI-3 MINI
    LLM_PATH = "models/microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    llm.eval()  # Chỉ eval, không .half() cho quantized model
    
    return {
        'processor': processor,
        'model': model,
        'yolo_detector': yolo_detector,
        'retriever': retriever,
        'reranker': reranker,
        'llm': llm,
        'tokenizer': tokenizer
    }
