from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    local_dir="./models/microsoft/Phi-3-mini-4k-instruct",
    local_dir_use_symlinks=False
)