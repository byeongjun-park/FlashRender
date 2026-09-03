import torch
import random
import numpy as np
import os
import re
import json
import pandas as pd
from einops import rearrange
import imageio
import torchvision
from torchvision.transforms import v2
from PIL import Image

from flashrender_utils.model_utils import parse_matrix


def crop_and_resize(image, width, height):
    img_width, img_height = image.size
    scale = max(width / img_width, height / img_height)
    return torchvision.transforms.functional.resize(
        image,
        (round(img_height * scale), round(img_width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )


def detach_worker_stdin(worker_id):
    """Point a dataloader worker's stdin at /dev/null.

    imageio spawns ffmpeg with the worker's stdin inherited. If that stdin is a
    tty whose master has since closed (e.g. the terminal that launched torchrun),
    ffmpeg's keyboard-interaction poll busy-spins instead of exiting, so
    count_frames() never returns and the worker hangs forever.
    """
    fd = os.open(os.devnull, os.O_RDONLY)
    if fd != 0:
        os.dup2(fd, 0)
        os.close(fd)


class HybridTensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, steps_per_epoch, height, width):
        synthetic_metadata = pd.read_csv('dataset/metadata.csv')
        self.synth_path = [os.path.join(base_path, file_name) for file_name in synthetic_metadata["file_name"]]
        self.forward_synth_path = [i + ".landscape_tensors.pth" for i in self.synth_path if (os.path.exists(i + ".landscape_tensors.pth"))]
        self.steps_per_epoch = steps_per_epoch

        # FIXME: Hardcoded parameters
        self.height, self.width = height, width
        target_size = 518
        new_width = target_size
        new_height = round(self.height * (new_width / self.width) / 14) * 14
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(self.height, self.width)),
            v2.Resize(size=(new_height, new_width), antialias=True),
            v2.ToTensor(),
        ])

    def get_synthetic_data(self, index):
        path = self.forward_synth_path

        while True:
            try:
                data = {}
                data_id = torch.randint(0, len(path), (1,))[0]
                data_id = (data_id + index) % len(path) # For fixed seed.
                path_tgt = path[data_id]

                # load the condition latent
                match = re.search(r'cam(\d+)', path_tgt)
                tgt_idx = int(match.group(1))
                cond_idx = random.randint(1, 10)
                path_cond = re.sub(r'cam(\d+)', f'cam{cond_idx:02}', path_tgt)

                # load the target trajectory
                base_path = path_tgt.rsplit('/', 2)[0]
                tgt_camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")
                with open(tgt_camera_path, 'r') as file:
                    cam_data = json.load(file)
                
                cam_idx = list(range(81))

                # Intrinsic matrix
                focal = float(base_path.split('train/f')[-1].split('_aperture')[0])
                focal = focal / 23.76  # 23.76: sensor height/width
                
                sample_wh_ratio = self.width / self.height
                pose_wh_ratio = 1  # 1280 / 1280 = 1

                if pose_wh_ratio > sample_wh_ratio:
                    resized_ori_w = self.height * pose_wh_ratio
                    focal_x = resized_ori_w * focal / self.width
                    focal_y = focal
                else:
                    resized_ori_h = self.width / pose_wh_ratio
                    focal_x = focal
                    focal_y = resized_ori_h * focal / self.height
                
                intrinsics = torch.tensor([0, focal_x, focal_y, 0.5, 0.5, 0, 0]).unsqueeze(0).repeat(len(cam_idx), 1)

                # rel poses
                cond_traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{cond_idx:02d}"]) for idx in cam_idx]
                cond_traj = np.stack(cond_traj).transpose(0, 2, 1)
                cond_traj = cond_traj[:, :, [1, 2, 0, 3]]
                cond_traj[:, :3, 1] *= -1.
                cond_traj[:, :3, 3] /= 100
                cond_ref_w2c = np.linalg.inv(cond_traj[0])
                _cond_c2ws = torch.as_tensor(cond_ref_w2c[None] @ cond_traj)
                cond_c2ws = rearrange(_cond_c2ws[:, :3], 'b c d -> b (c d)')
                cond_cam_params = torch.cat([intrinsics, cond_c2ws], dim=1)

                tgt_traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{tgt_idx:02d}"]) for idx in cam_idx]
                tgt_traj = np.stack(tgt_traj).transpose(0, 2, 1)
                tgt_traj = tgt_traj[:, :, [1, 2, 0, 3]]
                tgt_traj[:, :3, 1] *= -1.
                tgt_traj[:, :3, 3] /= 100
                _tgt_c2ws = torch.as_tensor(cond_ref_w2c[None] @ tgt_traj)
                tgt_c2ws = rearrange(_tgt_c2ws[:, :3], 'b c d -> b (c d)')
                tgt_cam_params = torch.cat([intrinsics, tgt_c2ws], dim=1)

                rel_poses = torch.linalg.inv(_cond_c2ws) @ _tgt_c2ws
                rel_poses = rearrange(rel_poses[:, :3], 'b c d -> b (c d)')

                data_tgt = torch.load(path_tgt, weights_only=True, map_location="cpu")["y"]
                data_cond = torch.load(path_cond, weights_only=True, map_location="cpu")

                if data_tgt.shape[1] != data_cond["y"].shape[1]:
                    print(f"ERROR WHEN LOADING: src-tgt latent shape mismatch")
                    index = random.randrange(len(path))
                    continue

                data = {
                    "input_latents": data_tgt,
                    'traj': tgt_cam_params,
                    'traj_cond': cond_cam_params,
                    'rel_poses': rel_poses,
                }
                data.update(data_cond)

                tgt_video_path = path_tgt[:path_tgt.find(".mp4") + len(".mp4")]
                tgt_reader = imageio.get_reader(tgt_video_path)
                tgt_nframes = tgt_reader.count_frames()

                if tgt_nframes != (data_tgt.shape[1] - 1) * 4 + 1:
                    print(f"ERROR WHEN LOADING: tgt video-latent shape mismatch")
                    tgt_reader.close()
                    index = random.randrange(len(path))
                    continue

                tgt_frames = []
                for frame_id in range(tgt_nframes):
                    frame = tgt_reader.get_data(frame_id)
                    frame = Image.fromarray(frame)
                    frame = crop_and_resize(frame, self.width, self.height)
                    frame = self.frame_process(frame)
                    tgt_frames.append(frame)
                tgt_reader.close()

                data['tgt_video_frames'] = torch.stack(tgt_frames, dim=0)
                break

            except Exception:
                index = random.randrange(len(path))

        return data

    def __getitem__(self, index):
        return self.get_synthetic_data(index)

    def __len__(self):
        return self.steps_per_epoch
    
class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, meta_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832):
        metadata = pd.read_csv(meta_path)
        self.path = [os.path.join(base_path, file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()

        self.path = self.path
        self.text = self.text

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = crop_and_resize(frame, self.width, self.height)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames

    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames


    def __getitem__(self, index):
        while True:
            try:
                path = self.path[index]
                prompt = self.text[index]
                video = self.load_video(path)

                data = {
                    'path': path,
                    'prompt': prompt,
                    'v_cond': video,
                }
                break
            except:
                index += 1

        return data

    def __len__(self):
        return len(self.path)