from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from peft import PeftModel
import torch
from ultralytics import YOLO
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig
)
from transformers import Qwen3VLForConditionalGeneration
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

# Load environment variables
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'models/yolo_best.pt')
LLM_MODEL_PATH = os.getenv('LLM_MODEL_PATH', 'models/Qwen/Qwen3-4B')
VLM_MODEL_PATH = os.getenv('VLM_MODEL_PATH', 'models/Qwen/Qwen3-VL-2B-Instruct')
RERANKER_MODEL_PATH = os.getenv('RERANKER_MODEL_PATH', 'models/namdp-ptit/ViRanker')
VECTOR_DB_PATH = os.getenv('VECTOR_DB_PATH', 'Vecto_Database/db_bienbao_2')
EMBEDDING_PATH = os.getenv('EMBEDDING_PATH', 'models/bkai-foundation-models/vietnamese-bi-encoder')
ADAPTER_LLM_PATH = os.getenv('ADAPTER_LLM_PATH', 'models/qwen-3-4b-finetuned')

def load_model_embeddings_and_retriever(emb_path, db_path):
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

def load_model_reranker(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(model_path, device=device)
    print("✅ Reranker đã load thành công")
    return reranker

def get_quantization_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16
    )

def load_model_yolo(model_path):
    """Load YOLO model"""
    yolo_detector = YOLO(model_path)
    yolo_detector.model.eval()
    print("✅ YOLO model đã load thành công")
    return yolo_detector

def load_model_vlm(model_path):
    bnb_config = get_quantization_config()
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    model.eval()
    print("✅ VLM model đã load thành công")
    return processor, model

def load_model_llm(base_model_id, adapter_path):
    bnb_config = get_quantization_config() 

    llm = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    llm = PeftModel.from_pretrained(llm, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    
    # Fix lỗi padding token thường gặp ở Qwen/Llama nếu chưa có
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm.eval()
    print("✅ LLM model (Fine-tuned) đã load thành công")
    return llm, tokenizer

def load_models():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    models = {}
    
    models['yolo'] = load_model_yolo(YOLO_MODEL_PATH)
    models['reranker'] = load_model_reranker(RERANKER_MODEL_PATH)
    models['retriever'] = load_model_embeddings_and_retriever(
        emb_path=EMBEDDING_PATH,
        db_path=VECTOR_DB_PATH
    )
    
    models['vlm_processor'], models['vlm'] = load_model_vlm(VLM_MODEL_PATH)
    models['llm'], models['llm_tokenizer'] = load_model_llm(LLM_MODEL_PATH, ADAPTER_LLM_PATH)
    
    return models