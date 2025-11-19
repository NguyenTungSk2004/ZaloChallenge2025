from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-4B",
    local_dir="./models/Qwen/Qwen3-4B",
    local_dir_use_symlinks=False
)

snapshot_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    local_dir="./models/Qwen/Qwen2-VL-2B-Instruct",
    local_dir_use_symlinks=False
)