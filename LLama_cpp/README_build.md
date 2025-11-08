# 🚀 Hướng dẫn build llama-cpp-python có CUDA cho Windows

> Áp dụng cho môi trường `rag` (Python 3.11, CUDA 12.6, GPU RTX 2060 trở lên, AE lưu ý xem máy cài CUDA, python gì nhé)

---

## 1️⃣ Chuẩn bị môi trường

### Gỡ sạch Torch cũ
```bash
pip uninstall torch torchvision torchaudio -y
Cài CMake (>=3.31.7)
Tải tại: https://github.com/Granddyser/windows-llama-cpp-python-cuda-guide
Trong quá trình cài: chọn “Add CMake to PATH for all users”

Kiểm tra CUDA
bash
Copy code
echo %CUDA_PATH%
nvcc --version
Đảm bảo xuất ra như:

arduino
Copy code
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.6, V12.6.20
2️⃣ Clone và build llama-cpp-python
bash
Copy code
git clone https://github.com/abetlen/llama-cpp-python.git
cd llama-cpp-python
git submodule update --init --recursive
3️⃣ Cấu hình build GPU
GPU	Compute Capability	CMake option
GTX 10xx (Pascal)	6.1	-DCMAKE_CUDA_ARCHITECTURES=61
RTX 20xx (Turing)	7.5	-DCMAKE_CUDA_ARCHITECTURES=75 ✅
RTX 30xx (Ampere)	8.6	-DCMAKE_CUDA_ARCHITECTURES=86
RTX 40xx (Ada)	8.9	-DCMAKE_CUDA_ARCHITECTURES=89

4️⃣ Build và cài đặt
bash
Copy code
conda activate rag
set FORCE_CMAKE=1
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75
pip install -e . --force-reinstall --no-cache-dir
Nếu build thành công, bạn sẽ thấy thư mục build/ sinh ra trong llama-cpp-python/.

5️⃣ Kiểm tra GPU hoạt động
python
Copy code
from llama_cpp import Llama
import llama_cpp

print(llama_cpp.__version__)
print(llama_cpp.llama_print_system_info())

llm = Llama(model_path="E:/Zalo Challenge 2025/Build_RAG/model/Phi-3-gguf/Phi-3-mini-4k-instruct-q4.gguf", n_gpu_layers=999)
print("✅ GPU LlamaCpp chạy OK")
Nếu bạn thấy trong log có dòng “CUDA = 1 | cuBLAS = 1” là GPU đã được kích hoạt 🎉