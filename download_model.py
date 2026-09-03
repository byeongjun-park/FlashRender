"""Download the Wan2.1-Fun base model and the FlashRender checkpoints into `models/`."""

from modelscope import snapshot_download as ms_snapshot_download
from huggingface_hub import snapshot_download as hf_snapshot_download

BASE_MODEL = "PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera"
BASE_DIR = "models/PAI/Wan2.1-Fun-V1.1-1.3B-Control-Camera"

FLASHRENDER_REPO = "byeongjun-park/FlashRender"
CKPT_DIR = "models/checkpoints"
# config.json is also the file the Hub uses to count downloads, so keep it in the pattern.
FLASHRENDER_PATTERNS = ["*.safetensors", "config.json"]


def main():
    # Download base model
    ms_snapshot_download(BASE_MODEL, local_dir=BASE_DIR)

    # Download FlashRender: all three checkpoints (stage 1 / 2 / 3) in one shot
    path = hf_snapshot_download(
        repo_id=FLASHRENDER_REPO,
        local_dir=CKPT_DIR,
        allow_patterns=FLASHRENDER_PATTERNS,
    )
    print(f"[FlashRender] {path}")


if __name__ == "__main__":
    main()
