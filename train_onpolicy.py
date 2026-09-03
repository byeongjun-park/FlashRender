import os
import random
import contextlib
from copy import deepcopy
from datetime import datetime, timedelta

import hydra
import torch
import torch.nn.functional as F
import lightning as pl
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file, load_file
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from einops import rearrange

from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from flashrender_utils.model_utils import adjust_to_FlashRender
from flashrender_utils.dmd2_utils import (
    variational_score_distillation_loss,
    gan_loss_generator,
    gan_loss_discriminator,
    load_video_discriminator,
)
from dataset import HybridTensorDataset, detach_worker_stdin


_STAGE1_TRAINABLE_KEYWORDS = [
    "self_attn", "cross_attn", "rel_pose_embedding"
]
_STAGE2_TRAINABLE_KEYWORDS = [
    "self_attn", "cross_attn", "time_embedding", "time_embedding_r", "time_projection", "rel_pose_embedding"
]


class OnPolicyModule(pl.LightningModule):
    def __init__(self, cfg, use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False):
        super().__init__()
        # Alternating generator / fake-score updates -> disable Lightning auto step/backward.
        self.automatic_optimization = False
        self.cfg = cfg
        self.op = cfg.onpolicy

        # --- Model loading ---
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            redirect_common_files=False,
            model_configs=[ModelConfig(
                model_id=cfg.model_path,
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                skip_download=True,
            )],
        )
        # 1000-step training scheduler (used by sigma->timestep conversion).
        self.pipe.scheduler.set_timesteps(1000, training=True)

        student = adjust_to_FlashRender(getattr(self.pipe, "dit"), cfg, training=False).to(torch.bfloat16)
        setattr(self.pipe, "dit", student)

        self.pipe.freeze_except([])
        self._set_student_trainable(self.pipe.dit)

        # --- Student init: flow-map pretrained checkpoint (falls back to the teacher ckpt). ---
        teacher_ckpt = self.op.teacher_checkpoint
        if not teacher_ckpt or not os.path.isfile(teacher_ckpt):
            raise FileNotFoundError(f"onpolicy.teacher_checkpoint not found: {teacher_ckpt}")
        student_ckpt = self.op.get("student_checkpoint", None) or teacher_ckpt
        if not os.path.isfile(student_ckpt):
            raise FileNotFoundError(f"onpolicy.student_checkpoint not found: {student_ckpt}")
        student_state = load_file(student_ckpt)
        missing, unexpected = self.pipe.dit.load_state_dict(student_state, strict=False)
        print(f"[OnPolicy] student init from {student_ckpt}: "
              f"loaded {len(student_state)}, missing={len(missing)}, unexpected={len(unexpected)}")

        # --- Teacher (real score): frozen MULTI-STEP model, queried as a denoiser (r=t). ---
        self.teacher_dit = deepcopy(self.pipe.dit).to(torch.bfloat16)
        teacher_state = load_file(teacher_ckpt)
        self.teacher_dit.load_state_dict(teacher_state, strict=False)
        self.teacher_dit.eval().requires_grad_(False)
        print(f"[OnPolicy] teacher (real score) from {teacher_ckpt}: loaded {len(teacher_state)}")

        self.fake_dit = deepcopy(self.pipe.dit).to(torch.bfloat16)
        self.fake_dit.load_state_dict(teacher_state, strict=False)   # teacher init (NOT the student)
        self.fake_dit.requires_grad_(False)
        self._set_student_trainable(self.fake_dit, keywords=_STAGE1_TRAINABLE_KEYWORDS)
        print(f"[OnPolicy] fake score (critic) initialised from TEACHER ckpt {teacher_ckpt}")

        # --- Discriminator on frozen-teacher features ---
        self.use_gan = float(self.op.get("gan_loss_weight_gen", 0.0)) > 0
        self.feature_indices = sorted(self.op.get("feature_indices", [15, 22, 29]))
        self._feat_cache = {}
        self._capture_on = False
        if self.use_gan:
            self.discriminator = load_video_discriminator(
                feature_indices=list(self.feature_indices),
                num_blocks=len(self.pipe.dit.blocks),
                disc_type=self.op.get("disc_type", "dit_simple_conv3d"),
                inner_dim=self.pipe.dit.dim,
            ).to(torch.bfloat16)
            # Forward hooks capture teacher block activations for the discriminator.
            for idx in self.feature_indices:
                self.teacher_dit.blocks[idx].register_forward_hook(self._make_capture_hook(idx))
            print(f"[OnPolicy] GAN ENABLED (weight={self.op.gan_loss_weight_gen}, feats={self.feature_indices})")
        else:
            self.discriminator = None

        # --- Static conditioning / hyper-params ---
        self.nega_prompt_emb = torch.load("dataset/nega_prompt_emb.pth", weights_only=True, map_location="cpu")
        self.extra_inputs = ["input_latents", "context", "clip_feature", "y", "traj", "traj_cond", "rel_poses"]
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.max_timestep_boundary = cfg.train.max_timestep_boundary
        self.min_timestep_boundary = cfg.train.min_timestep_boundary
        self.alignment_layer_index = cfg.train.alignment_layer_index

        # --- AnyFlow any-step: step count s sampled from this list each iter (use [4] to lock 4-step). ---
        self.num_inference_steps_list = [int(s) for s in self.op.num_inference_steps_list]
        assert all(s >= 1 for s in self.num_inference_steps_list), "num_inference_steps_list entries must be >= 1"

        # Iteration counter for the student_update_freq alternation schedule.
        self._op_iter = 0

    # ------------------------------------------------------------------ setup
    def _set_student_trainable(self, dit, keywords=_STAGE2_TRAINABLE_KEYWORDS):
        for name, module in dit.named_modules():
            if any(k in name for k in keywords):
                module.train()
                module.requires_grad_(True)
        for block in dit.blocks:
            block.rel_pose_src_token.requires_grad_(True)
            block.rel_pose_tgt_token.requires_grad_(True)

    def _make_capture_hook(self, idx):
        def hook(_module, _inp, out):
            if self._capture_on:
                self._feat_cache[idx] = out
        return hook

    @contextlib.contextmanager
    def _capturing(self):
        """Arm the teacher-block forward hooks for the duration of one teacher forward call."""
        self._capture_on = True
        try:
            yield
        finally:
            self._capture_on = False

    def _collect_feats(self, h, w):
        """Reshape captured teacher block tokens [B, 2*f*h*w, C] into disc feats [B, C, f, h, w]."""
        feats = []
        for idx in self.feature_indices:
            tok = self._feat_cache[idx]  # [B, (k f h w), C], k=2 (denoise | conditioning halves)
            feat = rearrange(tok, "b (k f h w) c -> k b c f h w", k=2, h=h, w=w)[0]
            feats.append(feat.contiguous())
        return feats

    # ------------------------------------------------------------- conditioning
    def forward_preprocess(self, data):
        """Identical to train.py: run the pipeline units to build all conditioning."""
        inputs_posi, inputs_nega = {}, {}
        inputs_shared = {
            "height": data["input_latents"].shape[3] * 8,
            "width": data["input_latents"].shape[4] * 8,
            "num_frames": (data["input_latents"].shape[2] - 1) * 4 + 1,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "alignment_layer_index": self.alignment_layer_index,
            "context_images": data["tgt_video_frames"],
            "nega_prompt_emb": self.nega_prompt_emb,
            "max_iterations": self.trainer.num_training_batches * self.trainer.max_epochs,
        }
        for extra_input in self.extra_inputs:
            inputs_shared[extra_input] = data[extra_input]
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )
        return {**inputs_shared, **inputs_posi}

    def _base_cond(self, inputs):
        """Conditioning kwargs shared by every model_fn call (context/timestep overridden per call)."""
        drop = {"context", "timestep", "timestep_r", "latents", "distill",
                "max_iterations", "nega_prompt_emb"}
        return {k: v for k, v in inputs.items() if k not in drop}

    # ----------------------------------------------------------------- helpers
    def _sigma_to_timestep(self, sigma):
        return sigma * self.pipe.scheduler.num_train_timesteps

    def _model_v(self, dit, x_t, sigma_t, sigma_r, context, base_cond):
        """Call model_fn with a given DiT and return the flow-map velocity prediction.

        ``sigma_t``/``sigma_r`` arrive as [B,1,1,1,1]; sinusoidal_embedding_1d needs a
        1-D [B] timestep, so flatten before converting to timestep units.
        """
        ts_t = self._sigma_to_timestep(sigma_t.reshape(-1))
        ts_r = self._sigma_to_timestep(sigma_r.reshape(-1))
        v = self.pipe.model_fn(
            dit=dit, latents=x_t, timestep=ts_t, timestep_r=ts_r,
            context=context, alignment=False, **base_cond,
        )[0]
        return v

    @staticmethod
    def _v_to_x0(x_t, v, sigma):
        # Forward process: x_t = (1 - sigma) * x0 + sigma * noise, velocity v = noise - x0.
        return x_t - sigma * v

    @staticmethod
    def _forward_process(x0, eps, sigma):
        return (1 - sigma) * x0 + sigma * eps

    def _sample_sigma(self, batch_size, device, dtype):
        """Shifted uniform sigma in [min_sigma, max_sigma] (matches the flow-matching schedule)."""
        shift = self.op.sigma_shift
        u = torch.rand(batch_size, device=device)
        sigma = shift * u / (1 + (shift - 1) * u)
        sigma = sigma.clamp(self.op.min_sigma, self.op.max_sigma)
        return sigma.reshape(-1, 1, 1, 1, 1).to(dtype)

    @staticmethod
    def _as_sigma(sigma, x):
        b = x.shape[0]
        if torch.is_tensor(sigma):
            return sigma.reshape(b, 1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        return torch.full((b, 1, 1, 1, 1), float(sigma), device=x.device, dtype=x.dtype)

    def _broadcast_int(self, value):
        t = torch.tensor([value], device=self.device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(t, src=0)
        return int(t.item())

    def _rollout_sigmas(self, s):
        shift = self.op.sigma_shift
        sig = torch.linspace(1.0, 0.0, s + 1)[:-1]     # s values (sigma_min=0, extra_one_step)
        sig = shift * sig / (1 + (shift - 1) * sig)     # apply shift
        return sig.tolist() + [0.0]

    # ------------------------------------------- Full-trajectory flow-map rollout
    def _flow_map_step(self, x, sigma_a, sigma_b, context, base_cond):
        sa = self._as_sigma(sigma_a, x)
        sb = self._as_sigma(sigma_b, x)
        v = self._model_v(self.pipe.dit, x, sa, sb, context, base_cond)
        return x + v * (sb - sa)

    def training_rollout(self, noise, context, base_cond, requires_grad):
        s = self._broadcast_int(random.choice(self.num_inference_steps_list))
        sigmas = self._rollout_sigmas(s)                 # len s+1, trailing 0.0
        whole = bool(self.op.whole_chain_grad) and requires_grad

        def seg_ctx(grad_on):
            return contextlib.nullcontext() if (requires_grad and grad_on) else torch.no_grad()

        x = noise
        for i in range(s):
            grad_on = whole or (i == s - 1)  # last transition always keeps grad when requires_grad
            with seg_ctx(grad_on):
                x = self._flow_map_step(x, sigmas[i], sigmas[i + 1], context, base_cond)
        return x

    # ------------------------------------------------------------- DMD steps
    def _student_step(self, inputs, gen, real, ctx_pos, ctx_neg, base_cond, h, w):
        # ``gen`` is the on-policy student x0 (grad through the chain), rolled out in
        # training_step so its detached copy can be reused by the fake-score update.

        # DMD / VSD gradient at a random re-noise level (scores off-graph, gen detached).
        eps = torch.randn_like(real)
        sigma = self._sample_sigma(real.shape[0], real.device, real.dtype)
        x_t = self._forward_process(gen.detach(), eps, sigma)

        with torch.no_grad():
            v_cond = self._model_v(self.teacher_dit, x_t, sigma, sigma, ctx_pos, base_cond)
            if self.op.guidance_scale is not None:
                v_uncond = self._model_v(self.teacher_dit, x_t, sigma, sigma, ctx_neg, base_cond)
                v_teacher = v_uncond + self.op.guidance_scale * (v_cond - v_uncond)
            else:
                v_teacher = v_cond
            teacher_x0 = self._v_to_x0(x_t, v_teacher, sigma)

            v_fake = self._model_v(self.fake_dit, x_t, sigma, sigma, ctx_pos, base_cond)
            fake_x0 = self._v_to_x0(x_t, v_fake, sigma)

        dmd_loss = self.op.dmd_weight * variational_score_distillation_loss(gen, teacher_x0, fake_x0)
        logs = {"dmd_loss": dmd_loss.detach()}
        loss = dmd_loss

        # GAN generator loss. The discriminator scores frozen-teacher
        # features of the STUDENT's own sample; grad flows teacher->x_t_gen->gen->(whole rollout).
        # x_t_gen keeps gen's grad (unlike the VSD x_t which detaches gen), and reuses the SAME
        # (eps, sigma). The teacher forward here runs WITH grad (teacher params are frozen, so only
        # activations carry the graph into the student).
        gan_gen = torch.zeros((), device=real.device, dtype=gen.dtype)
        if self.use_gan:
            x_t_gen = self._forward_process(gen, eps, sigma)
            with self._capturing():
                self._model_v(self.teacher_dit, x_t_gen, sigma, sigma, ctx_pos, base_cond)
            feats_fake = self._collect_feats(h, w)
            gan_gen = gan_loss_generator(self.discriminator(feats_fake))
            loss = loss + float(self.op.gan_loss_weight_gen) * gan_gen
            logs["gan_loss_gen"] = gan_gen.detach()

        return loss, logs

    def _fake_step(self, gen, real, ctx_pos, base_cond, h, w):
        # Fake score tracks the student distribution via denoising-score-matching in x0-space.
        # ``gen`` is a detached on-policy student sample (fresh re-noise each call).
        eps = torch.randn_like(gen)
        sigma = self._sample_sigma(gen.shape[0], gen.device, gen.dtype)
        x_t = self._forward_process(gen, eps, sigma)

        v_fake = self._model_v(self.fake_dit, x_t, sigma, sigma, ctx_pos, base_cond)
        fake_x0 = self._v_to_x0(x_t, v_fake, sigma)
        fake_loss = F.mse_loss(fake_x0, gen)  # gen detached
        logs = {"fake_score_loss": fake_loss.detach()}
        loss = fake_loss

        # GAN discriminator loss. Frozen teacher extracts features for
        # the student sample (fake) and real data; grad lives only in the discriminator heads.
        if self.use_gan:
            gan_r1 = torch.zeros((), device=real.device, dtype=fake_loss.dtype)
            with torch.no_grad():
                with self._capturing():
                    self._model_v(self.teacher_dit, x_t.detach(), sigma, sigma, ctx_pos, base_cond)
                feats_fake = self._collect_feats(h, w)

                if self.op.get("gan_use_same_t_noise", True):
                    sigma_real, eps_real = sigma, eps
                else:
                    sigma_real = self._sample_sigma(real.shape[0], real.device, real.dtype)
                    eps_real = torch.randn_like(real)
                x_real_t = self._forward_process(real, eps_real, sigma_real)

                with self._capturing():
                    self._model_v(self.teacher_dit, x_real_t, sigma_real, sigma_real, ctx_pos, base_cond)
                feats_real = self._collect_feats(h, w)

            real_logit = self.discriminator(feats_real)
            fake_logit = self.discriminator(feats_fake)
            gan_disc = gan_loss_discriminator(real_logit, fake_logit)

            r1_w = float(self.op.get("gan_r1_reg_weight", 0.0))
            if r1_w > 0:
                with torch.no_grad():
                    real_alpha = real + float(self.op.gan_r1_reg_alpha) * torch.randn_like(real)
                    x_real_alpha = self._forward_process(real_alpha, eps_real, sigma_real)
                    with self._capturing():
                        self._model_v(self.teacher_dit, x_real_alpha, sigma_real, sigma_real, ctx_pos, base_cond)
                    feats_real_alpha = self._collect_feats(h, w)
                real_alpha_logit = self.discriminator(feats_real_alpha)
                gan_r1 = F.mse_loss(real_logit, real_alpha_logit)

            loss = fake_loss + gan_disc + r1_w * gan_r1
            logs["gan_loss_disc"] = gan_disc.detach()
            logs["gan_loss_r1"] = gan_r1.detach()

        return loss, logs

    # ---------------------------------------------------------------- training
    def training_step(self, batch, batch_idx):
        self.pipe.device = self.device
        opts = self.optimizers()
        opt_student, opt_fake = opts[0], opts[1]
        opt_disc = opts[2] if isinstance(opts, (list, tuple)) and len(opts) > 2 else None

        with torch.no_grad():
            inputs = self.forward_preprocess(batch)

        real = inputs["input_latents"]
        ctx_pos = inputs["context"]
        ctx_neg = self.nega_prompt_emb.to(device=ctx_pos.device, dtype=ctx_pos.dtype)
        base_cond = self._base_cond(inputs)
        clip = self.op.grad_clip
        patch = self.pipe.dit.patch_size
        h = real.shape[3] // patch[1]
        w = real.shape[4] // patch[2]

        # STRICT alternation: every iter updates the student XOR the fake score, never both.
        # student_update_freq = f -> student on iter % f == 0, fake otherwise, so the critic
        # trains on fresh on-policy samples on the fake iters. f=1 is degenerate (student only,
        # critic never trains) -- use f>=2.
        freq = int(self.op.get("student_update_freq", 5))
        do_student = (self._op_iter % freq == 0)
        self._op_iter += 1

        if do_student:
            noise = torch.randn_like(real)
            gen = self.training_rollout(noise, ctx_pos, base_cond, requires_grad=True)
            loss, logs = self._student_step(inputs, gen, real, ctx_pos, ctx_neg, base_cond, h, w)
            opt_student.zero_grad()
            self.manual_backward(loss)
            if clip > 0:
                self.clip_gradients(opt_student, gradient_clip_val=clip)
            opt_student.step()
            logs["total_loss"] = loss.detach()
        else:
            # fake-only iter: fresh on-policy sample under no_grad (no student graph).
            # Also trains the GAN discriminator when use_gan (fake:disc share this branch).
            with torch.no_grad():
                fake_samples = self.training_rollout(
                    torch.randn_like(real), ctx_pos, base_cond, requires_grad=False)
            loss, logs = self._fake_step(fake_samples, real, ctx_pos, base_cond, h, w)
            opt_fake.zero_grad()
            if opt_disc is not None:
                opt_disc.zero_grad()
            self.manual_backward(loss)
            if clip > 0:
                self.clip_gradients(opt_fake, gradient_clip_val=clip)
                if opt_disc is not None:
                    self.clip_gradients(opt_disc, gradient_clip_val=clip)
            opt_fake.step()
            if opt_disc is not None:
                opt_disc.step()

        self.log_dict(logs, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        student_params = [p for p in self.pipe.dit.parameters() if p.requires_grad]
        opt_student = torch.optim.AdamW(student_params, lr=self.op.student_lr)
        fake_params = [p for p in self.fake_dit.parameters() if p.requires_grad]
        opt_fake = torch.optim.AdamW(fake_params, lr=self.op.fake_score_lr)
        if not self.use_gan:
            return [opt_student, opt_fake]
        opt_disc = torch.optim.AdamW(self.discriminator.parameters(),
                                     lr=float(self.op.get("discriminator_lr", self.op.fake_score_lr)))
        return [opt_student, opt_fake, opt_disc]

    def on_train_epoch_end(self):
        # Save the student's trainable parameters (same format as train.py checkpoints).
        if self.trainer.is_global_zero:
            checkpoint_dir = os.path.join(self.trainer.default_root_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            trainable = [n.replace("pipe.dit.", "") for n, p in self.named_parameters()
                         if p.requires_grad and n.startswith("pipe.dit.")]
            state_dict = self.pipe.dit.state_dict()
            state_dict = {k: v.detach().cpu() for k, v in state_dict.items() if k in trainable}
            save_file(state_dict, os.path.join(checkpoint_dir, f"onpolicy_epoch_{self.trainer.current_epoch + 1}.safetensors"))
        self.trainer.strategy.barrier()


@hydra.main(config_path="configs", config_name="base.yaml", version_base="1.1")
def main(cfg: DictConfig):
    rank = int(os.environ.get("RANK", "0"))
    seed_everything(42 + rank, workers=True)

    if not cfg.wandb.disabled:
        os.environ["WANDB_API_KEY"] = cfg.wandb.wandb_api_key
        wandb_logger = WandbLogger(
            project=cfg.wandb.project_name,
            name=f"onpolicy-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=OmegaConf.to_container(cfg),
        )
    else:
        wandb_logger = False

    model = OnPolicyModule(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.onpolicy.get("max_epochs", cfg.train.max_epochs),
        accelerator="gpu",
        devices="auto",
        # Alternating updates leave some params grad-free each step -> allow unused params.
        strategy=DDPStrategy(timeout=timedelta(hours=12), find_unused_parameters=True),
        precision="bf16-mixed",
        default_root_dir=cfg.output_path,
        log_every_n_steps=1,
        logger=wandb_logger,
        enable_checkpointing=False,
    )

    dataset = HybridTensorDataset(
        cfg.dataset_path,
        cfg.train.steps_per_epoch * trainer.world_size * cfg.train.batch_size,
        cfg.extract.height,
        cfg.extract.width,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataloader_num_workers,
        worker_init_fn=detach_worker_stdin,
    )

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
