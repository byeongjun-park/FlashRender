<div align="center">

# ⚡ FlashRender ⚡
### Few-Step Generative Rendering via Camera-Controlled Video MeanFlow

<a href="https://arxiv.org/abs/26xx.xxxxx"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-26xx.xxxxx-b31b1b.svg"></a>
<a href="https://byeongjun-park.github.io/FlashRender/"><img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>
<a href="https://huggingface.co/byeongjun-park/FlashRender"><img alt="HuggingFace" src="https://img.shields.io/badge/🤗%20HuggingFace-model-blue"></a>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-lightgrey.svg"></a>

</div>

---

<div align="center">

<table>
<tr>
<td><img src="assets/readme/la-la-land_pair.gif" width="380"></td>
<td><img src="assets/readme/joker_pair.gif" width="380"></td>
</tr>
<tr>
<td><img src="assets/readme/mammoths_pair.gif" width="380"></td>
<td><img src="assets/readme/chess-monkey_pair.gif" width="380"></td>
</tr>
<tr>
<td colspan="2" align="center"><small><b>Input Video</b> (left) &nbsp;·&nbsp; <b>Target Retake</b> (right)</small></td>
</tr>
</table>
<p align="center">
<i>🔥 High-quality camera-controlled video-to-video generation in just 4 sampling steps!! 🔥</i>
</p>

</div>

## 📑 Table of Contents

- [Setup](#-setup)
- [Training](#-training)
- [Inference and Evaluation](#-inference-and-evaluation)
- [Citation](#-citation)
- [License](#-license)

## 🔧 Setup

1. **Create the environment and download checkpoints**

   ```shell
   conda env create -f environment.yml
   conda activate flashrender
   pip install -r requirements.txt
   pip install flash_attn --no-build-isolation
   pip install --no-build-isolation git+https://github.com/mohammadasim98/met3r
   pip install --no-build-isolation -e vipe
   python download_model.py
   ```

2. **Prepare the training datasets**

   Download [MultiCamVideo](https://huggingface.co/datasets/KwaiVGI/MultiCamVideo-Dataset) and [SyncCamVideo](https://huggingface.co/datasets/KlingTeam/SynCamVideo-Dataset), and place them as:

   ```
   <path/to/dataset>/
   ├── MultiCamVideo/
   │   └── train/
   │       ├── f18_aperture10/
   │       ├── f24_aperture5/
   │       ├── f35_aperture2.4/
   │       └── f50_aperture2.4/
   └── SyncCamVideo/
       └── train/
           └── f24_aperture5/
   ```

3. **Extract features**

   ```shell
   # Captioning is optional — ours is already provided in dataset/metadata.csv
   torchrun --nproc-per-node=8 text_captioning.py dataset_path=<path/to/dataset>

   torchrun --nproc-per-node=8 extract_features.py \
       dataset_path=<path/to/dataset> meta_path=dataset/metadata.csv
   ```

## 🚀 Training

FlashRender retakes an input video from a target camera trajectory in **4-NFE** using a three-stage training recipe, run on 8 GPUs and configured through `configs/base.yaml`.

1. **Stage 1** (`train.py`) — Multi-step flow-matching model (`t = r`)

   ```shell
   torchrun --nproc-per-node=8 train.py dataset_path=<path/to/dataset>
   ```

2. **Stage 2** (`train_meanflow.py`) — Few-step MeanFlow model, resumed from Stage 1

   ```shell
   torchrun --nproc-per-node=8 train_meanflow.py \
       meanflow.resume_checkpoint=models/checkpoints/epoch_20.safetensors
   ```

3. **Stage 3** (`train_onpolicy.py`) — On-policy flow-map distillation of the Stage 2 student

   ```shell
   torchrun --nproc-per-node=8 train_onpolicy.py \
       onpolicy.teacher_checkpoint=models/checkpoints/epoch_20.safetensors \
       onpolicy.student_checkpoint=models/checkpoints/meanflow_epoch_20.safetensors
   ```

## 🎬 Inference and Evaluation

1. **Download checkpoints** (skip if you already ran `download_model.py` during setup)

   | Stage | Checkpoint file | Download |
   |:---:|---|---|
   | 1 | `epoch_20.safetensors` | [Checkpoint (Stage 1)](https://huggingface.co/byeongjun-park/FlashRender/blob/main/epoch_20.safetensors) |
   | 2 | `meanflow_epoch_20.safetensors` | [Checkpoint (Stage 2)](https://huggingface.co/byeongjun-park/FlashRender/blob/main/meanflow_epoch_20.safetensors) |
   | 3 | `onpolicy_epoch_5.safetensors` | [Checkpoint (Stage 3)](https://huggingface.co/byeongjun-park/FlashRender/blob/main/onpolicy_epoch_5.safetensors) |

2. **Run inference**

   ```shell
   # Stage 1: multi-step checkpoint
   python inference.py eval.ckpt_path=models/checkpoints/epoch_20.safetensors \
       eval.num_inference_steps=50 eval.inference_mode=mul \
       eval.cam_type=1 eval.video_path=example_data/bear

   # Stage 2: MeanFlow checkpoint
   python inference.py eval.ckpt_path=models/checkpoints/meanflow_epoch_20.safetensors \
       eval.num_inference_steps=4 eval.inference_mode=any \
       eval.cam_type=1 eval.video_path=example_data/bear

   # Stage 3: on-policy-distilled checkpoint
   python inference.py eval.ckpt_path=models/checkpoints/onpolicy_epoch_5.safetensors \
       eval.num_inference_steps=4 eval.inference_mode=any \
       eval.cam_type=1 eval.video_path=example_data/bear
   ```

3. **Evaluate a generated video** (the example below evaluates the output of the final checkpoint)

   ```shell
   python evaluate.py --data_path results/bear/onpolicy_epoch5_4NFE_cam1.mp4
   ```

## 📚 Citation

If you find this repository helpful for your project, please consider citing our work. :)

```bibtex

```

## 📄 License

This project is released under the [Apache License 2.0](LICENSE).
