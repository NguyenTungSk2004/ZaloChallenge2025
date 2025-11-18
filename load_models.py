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

def get_quantization_config():
    """Tạo config quantization NF4"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16
    )

def load_model_yolo(model_path="models/yolo/best.pt"):
    """Load YOLO model"""
    print("🔄 Đang load YOLO model...")
    yolo_detector = YOLO(model_path)
    yolo_detector.model.eval()
    print("✅ YOLO model đã load thành công")
    return yolo_detector

def load_model_vlm(model_path="models/blip2-opt-2.7b"):
    """Load VLM (Vision Language Model) - BLIP2"""
    print("🔄 Đang load VLM model...")
    bnb_config = get_quantization_config()
    
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.eval()
    print("✅ VLM model đã load thành công")
    return processor, model

def load_model_embeddings_and_retriever(
    emb_path="models/bkai-foundation-models/bkai_vn_bi_encoder",
    db_path="Vecto_Database/db_bienbao_2"
):
    """Load Embeddings và Retriever"""
    print("🔄 Đang load Embeddings và Retriever...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_path,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    print("✅ Embeddings và Retriever đã load thành công")
    return retriever

def load_model_reranker(model_path="models/namdp-ptit/ViRanker"):
    """Load Reranker model"""
    print("🔄 Đang load Reranker...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(model_path, device=device)
    print("✅ Reranker đã load thành công")
    return reranker

def load_model_llm(model_path="models/microsoft/Phi-3-mini-4k-instruct"):
    """Load LLM (Language Model) - Phi-3 Mini"""
    print("🔄 Đang load LLM model...")
    bnb_config = get_quantization_config()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    llm.eval()
    print("✅ LLM model đã load thành công")
    return llm, tokenizer

def load_models(models_to_load = ['yolo', 'vlm', 'retriever', 'reranker', 'llm']):
    """
    Load chỉ những models cần thiết
    
    Args:
        models_to_load (list): Danh sách models cần load
                              ['yolo', 'vlm', 'retriever', 'reranker', 'llm']
    """
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    models = {}
    
    if 'yolo' in models_to_load:
        models['yolo'] = load_model_yolo()
    
    if 'vlm' in models_to_load:
        processor, model = load_model_vlm()
        models['processor'] = processor
        models['model'] = model
    
    if 'retriever' in models_to_load:
        models['retriever'] = load_model_embeddings_and_retriever()
    
    if 'reranker' in models_to_load:
        models['reranker'] = load_model_reranker()
    
    if 'llm' in models_to_load:
        llm, tokenizer = load_model_llm()
        models['llm'] = llm
        models['tokenizer'] = tokenizer
    
    return models
