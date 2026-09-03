import os
import pandas as pd
from glob import glob
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import torch.distributed as dist
import imageio
from PIL import Image
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd
from pathlib import Path

def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size, device


@hydra.main(config_path="configs", config_name="base.yaml", version_base="1.1")
def main(cfg: DictConfig):
    rank, local_rank, world_size, device = setup_ddp()

    orig_cwd = get_original_cwd()

    train_files = sorted(glob(f'{cfg.dataset_path}/*/train/*/*/*/*.mp4'))

    my_files = train_files[rank::world_size]

    caption_processor = AutoProcessor.from_pretrained(cfg.blip_path)
    captioner = (
        Blip2ForConditionalGeneration
        .from_pretrained(cfg.blip_path, torch_dtype=torch.float16)
        .to(device)
        .eval()
    )

    data = []
    pbar = tqdm(my_files, disable=(rank != 0), desc=f"rank{rank}")

    with torch.inference_mode():
        for file in pbar:
            reader = imageio.get_reader(file)
            mid_frame = reader.get_data((reader.count_frames() - 1) // 2)
            reader.close()

            pil_image = Image.fromarray(mid_frame)
            inputs = caption_processor(images=pil_image, return_tensors="pt").to(device, torch.float16)
            generated_ids = captioner.generate(**inputs)
            generated_text = caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            data.append({
                "file_name": str(Path(file).resolve().relative_to(Path(cfg.dataset_path).resolve())),
                "text": generated_text,
            })

    out_dir = os.path.join(orig_cwd, "dataset")
    os.makedirs(out_dir, exist_ok=True)
    part_path = os.path.join(out_dir, f"metadata_rank{rank}.csv")
    pd.DataFrame(data).to_csv(part_path, index=False)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        part_files = sorted(glob(os.path.join(out_dir, "metadata_rank*.csv")))
        dfs = [pd.read_csv(p) for p in part_files]
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(os.path.join(out_dir, "metadata.csv"), index=False)
        print("metadata.csv generated successfully!")

        for p in part_files:
            os.remove(p)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
