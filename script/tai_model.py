from huggingface_hub import snapshot_download

repo_id = "vinai/phobert-base-v2"
local_dir = "E:/Zalo Challenge 2025/Build_RAG/model/vinai"

snapshot_download(repo_id=repo_id, local_dir=local_dir)