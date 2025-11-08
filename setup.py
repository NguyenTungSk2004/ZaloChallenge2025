from setuptools import setup, find_packages

# Đọc nội dung requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="rag_legal_assistant",
    version="0.1.0",
    author="Do Nam",
    description="RAG hệ thống hỏi đáp văn bản luật Việt Nam sử dụng Phi-3 GGUF + LangChain + Chroma.",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=requirements,
    python_requires=">=3.10",
)
