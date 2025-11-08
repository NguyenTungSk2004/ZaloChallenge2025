import torch
from llama_cpp import Llama
import llama_cpp
model_path = 'E:/Zalo Challenge 2025/Build_RAG/model/Phi-3-gguf/Phi-3-mini-4k-instruct-q4.gguf'
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print("CUDA verion:", torch.version.cuda)

# llm = Llama(model_path="E:/Zalo Challenge 2025/Build_RAG/model/Phi-3-gguf/Phi-3-mini-4k-instruct-q4.gguf", n_gpu_layers=10)
# print("✅ GPU LlamaCpp chạy OK")

print(llama_cpp.__version__)
print(llama_cpp.llama_print_system_info())