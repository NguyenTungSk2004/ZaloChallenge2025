from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="microsoft/Phi-3-mini-4k-instruct",
#     local_dir="./models/microsoft/Phi-3-mini-4k-instruct",
#     local_dir_use_symlinks=False
# )

# snapshot_download(
#     repo_id="namdp-ptit/ViRanker",
#     local_dir="./models/namdp-ptit/ViRanker",
#     local_dir_use_symlinks=False
# )

# snapshot_download(
#     repo_id="bkai-foundation-models/vietnamese-bi-encoder",
#     local_dir="./models/bkai-foundation-models/vietnamese-bi-encoder",
#     local_dir_use_symlinks=False
# )

# snapshot_download(
#     repo_id="Salesforce/blip2-opt-2.7b",
#     local_dir="./models/blip2-opt-2.7b",
#     local_dir_use_symlinks=False
# )

# snapshot_download(
#     repo_id="google/gemma-3-4b-it",
#     local_dir="./models/google/gemma-3-4b-it",
#     local_dir_use_symlinks=False
# )

snapshot_download(
    repo_id="Qwen/Qwen3-VL-8B-Instruct",
    local_dir="./models/Qwen3-VL-8B",
    local_dir_use_symlinks=False
)
