import json
import torch
import torch.nn as nn
import numpy as np
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R
from flashrender_utils.vggt_alignment_loss import VGGTAlignmentLoss
from copy import deepcopy


def adjust_to_FlashRender(model, cfg, training=True):
    dim = model.dim
    model.use_align = training and cfg.train.use_align

    model.time_embedding_r = deepcopy(model.time_embedding)
    model.rel_pose_embedding = nn.Sequential(
        nn.Linear(48, dim),
        nn.SiLU(),
        nn.Linear(dim, dim),
    )

    nn.init.xavier_uniform_(model.rel_pose_embedding[0].weight)
    nn.init.zeros_(model.rel_pose_embedding[0].bias)
    nn.init.zeros_(model.rel_pose_embedding[2].weight)
    nn.init.zeros_(model.rel_pose_embedding[2].bias)

    for block in model.blocks:
        phase_dim = block.num_heads * (2 * model.head_dim // 3 // 2)
        block.self_attn.rope_phase_qk = nn.Sequential(nn.SiLU(), nn.Linear(dim, phase_dim, bias=False))
        block.self_attn.rope_phase_vo = nn.Sequential(nn.SiLU(), nn.Linear(dim, phase_dim, bias=False))
        block.self_attn.rope_phase_qk[-1].weight.data.zero_()
        block.self_attn.rope_phase_vo[-1].weight.data.zero_()
        block.rel_pose_src_token = nn.Parameter(torch.zeros(1, 1, dim))
        block.rel_pose_tgt_token = nn.Parameter(torch.zeros(1, 1, dim))

    if model.use_align:
        model.alignment_loss = VGGTAlignmentLoss(dim)

    return model

# camera-related functions
def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


def save_video(frames, save_path, fps):
    if isinstance(frames, torch.Tensor):
        frames = (frames.detach().cpu() * 255).to(torch.uint8)

    writer = imageio.get_writer(save_path, fps=fps, codec='libx264', ffmpeg_params=["-crf", "10"])

    for frame in frames:
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()

def _build_traj_opencv_c2w(data: dict, cam_type: int) -> np.ndarray:
    cam_idx = list(range(81))
    traj = [parse_matrix(data[f"frame{idx}"][f"cam{int(cam_type):02d}"]) for idx in cam_idx]
    traj = np.stack(traj).transpose(0, 2, 1)       # -> (N,4,4)
    traj = traj[:, :, [1, 2, 0, 3]]     
    traj[:, :3, 1] *= -1.                
    traj[:, :3, 3] /= 100.0   
    traj = np.linalg.inv(traj[0])[None] @ traj
    return traj                          # (N,4,4) OpenCV-style c2w

def _rel_rotvec_and_local_trans(T_abs: np.ndarray):
    R_abs = T_abs[:, :3, :3]
    t_abs = T_abs[:, :3, 3]
    N = T_abs.shape[0]
    rotvecs, dlocal = [], []
    for i in range(N-1):
        R_rel = R_abs[i].T @ R_abs[i+1]
        rv = R.from_matrix(R_rel).as_rotvec()     # axis*angle
        rotvecs.append(rv)
        dx = R_abs[i].T @ (t_abs[i+1] - t_abs[i]) # local translation
        dlocal.append(dx)
    return np.asarray(rotvecs), np.asarray(dlocal)

def _resample(arr: np.ndarray, M: int) -> np.ndarray:
    if M <= 1: return np.zeros((0, arr.shape[1]), dtype=arr.dtype)
    n = arr.shape[0]
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, M-1)
    return np.vstack([np.interp(dst, src, arr[:,j]) for j in range(arr.shape[1])]).T

def make_camera_trajectory(cfg, eval_frames, eps=1e-6):
    with open(cfg.eval.camera_path, "r") as f:
        data = json.load(f)

    T_abs = _build_traj_opencv_c2w(data, cfg.eval.cam_type)     # (N,4,4)
    R0 = T_abs[0, :3, :3].copy()
    t0 = T_abs[0, :3, 3].copy()

    rotvecs, dlocal = _rel_rotvec_and_local_trans(T_abs)  # (N-1,3), (N-1,3)

    rv_sum = rotvecs.sum(axis=0)
    axis_norm = np.linalg.norm(rv_sum)
    if axis_norm < eps:
        base_axis = rotvecs[0] if np.linalg.norm(rotvecs[0]) > eps else np.array([0,0,1.0])
        u = base_axis / (np.linalg.norm(base_axis) + 1e-12)
    else:
        u = rv_sum / axis_norm

    theta_steps = (rotvecs @ u)  # (N-1,)

    src = np.linspace(0.0, 1.0, len(theta_steps))
    dst = np.linspace(0.0, 1.0, eval_frames-1)
    theta_steps_resampled = np.interp(dst, src, theta_steps) * cfg.eval.cam_speed

    dlocal_resampled = _resample(dlocal, eval_frames) * cfg.eval.cam_speed  # (num_frame-1, 3)

    traj = []
    Rk = R0.copy()
    tk = t0.copy()
    traj.append(np.block([[Rk, tk.reshape(3,1)],[np.zeros((1,3)), np.ones((1,1))]]))

    theta_cum = 0.0
    for i in range(eval_frames-1):
        theta_cum += theta_steps_resampled[i]
        R_rel = R.from_rotvec(u * theta_cum).as_matrix()
        Rk = R_rel @ R0
        tk = tk + (Rk @ dlocal_resampled[i])

        Ti = np.eye(4)
        Ti[:3, :3] = Rk
        Ti[:3, 3]  = tk
        traj.append(Ti.copy())

    return np.stack(traj)