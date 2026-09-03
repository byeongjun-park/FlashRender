import os
import hydra
import torch
import lightning as pl
from datetime import datetime
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from safetensors.torch import save_file
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from flashrender_utils.model_utils import adjust_to_FlashRender
from dataset import HybridTensorDataset, detach_worker_stdin
from lightning.pytorch import seed_everything
from datetime import timedelta
from lightning.pytorch.strategies import DDPStrategy


class TrainingModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
    ):
        super().__init__()

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            redirect_common_files=False,
            model_configs=[ModelConfig(model_id=cfg.model_path, origin_file_pattern="diffusion_pytorch_model*.safetensors", skip_download=True)],
        )

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Add modules
        model = adjust_to_FlashRender(getattr(self.pipe, 'dit'), cfg, training=True)
        model = model.to(torch.bfloat16)
        setattr(self.pipe, 'dit', model)

        self.pipe.freeze_except([])

        module_name_list = ['self_attn', 'cross_attn', 'rel_pose_embedding']
        for name, module in self.pipe.dit.named_modules():
            if any(keyword in name for keyword in module_name_list):
                module.train()
                module.requires_grad_(True)

        # Newly added modules by adjust_to_FlashRender: all trainable
        for block in self.pipe.dit.blocks:
            block.rel_pose_src_token.requires_grad_(True)
            block.rel_pose_tgt_token.requires_grad_(True)

        if hasattr(self.pipe.dit, "alignment_loss"):
            self.pipe.dit.alignment_loss.train()
            self.pipe.dit.alignment_loss.requires_grad_(True)
            if hasattr(self.pipe.dit.alignment_loss, "vggt_model"):
                self.pipe.dit.alignment_loss.vggt_model.eval()
                self.pipe.dit.alignment_loss.vggt_model.requires_grad_(False)

        # Bake cfg
        self.nega_prompt_emb = torch.load('dataset/nega_prompt_emb.pth', weights_only=True, map_location='cpu')

        # Store other configs
        self.extra_inputs = ['input_latents', 'context', 'clip_feature', 'y', 'traj', 'traj_cond', 'rel_poses']


        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.max_timestep_boundary = cfg.train.max_timestep_boundary
        self.min_timestep_boundary = cfg.train.min_timestep_boundary

        self.learning_rate = cfg.train.lr
        self.alignment_layer_index = cfg.train.alignment_layer_index
        self.align_loss_weight = cfg.train.get('align_loss_weight', 0.5)

    def forward_preprocess(self, data):        
        # CFG-sensitive parameters
        inputs_posi = {}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "height": data["input_latents"].shape[3] * 8,
            "width": data["input_latents"].shape[4] * 8,
            "num_frames": (data["input_latents"].shape[2] - 1) * 4 + 1,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "alignment_layer_index": self.alignment_layer_index,    # for RETA
            "distill": False,                                       # Stage 1 never distills; see train_meanflow.py
            "context_images": data['tgt_video_frames'],             # for RETA
            "nega_prompt_emb": self.nega_prompt_emb,                # for Baking CFG (One-Step)
            'max_iterations': self.trainer.num_training_batches * self.trainer.max_epochs,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)

        return {**inputs_shared, **inputs_posi}

    def training_step(self, batch, batch_idx):
        self.pipe.device = self.device

        with torch.no_grad():
            inputs = self.forward_preprocess(batch)
            
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss_dict = self.pipe.training_loss(**models, **inputs)
        total_loss = 0
        for k, v in loss_dict.items():
            if v is None:
                continue
            
            self.log(k, v, sync_dist=True)

            if k == 'alignment_loss':
                v = v * self.align_loss_weight

            total_loss += v

        return total_loss

    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            checkpoint_dir = os.path.join(self.trainer.default_root_dir, 'checkpoints') 
            os.makedirs(checkpoint_dir, exist_ok=True)

            trainable_param_names = [n.replace('pipe.dit.', '') for n, p in self.named_parameters() if p.requires_grad]
            state_dict = self.pipe.dit.state_dict()
            state_dict = {k: v.detach().cpu() for k, v in state_dict.items() if k in trainable_param_names}
            save_file(state_dict, os.path.join(checkpoint_dir, f"epoch_{self.trainer.current_epoch + 1}.safetensors"))

        self.trainer.strategy.barrier()
        
@hydra.main(config_path="configs", config_name="base.yaml", version_base="1.1")
def main(cfg: DictConfig):
    rank = int(os.environ.get("RANK", "0"))
    base = 42
    seed = base + rank
    seed_everything(seed, workers=True)

    if not cfg.wandb.disabled:
        os.environ["WANDB_API_KEY"] = cfg.wandb.wandb_api_key
        wandb_logger = WandbLogger(
            project=cfg.wandb.project_name,
            name = f"stage1-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=OmegaConf.to_container(cfg),
        )
    else:
        wandb_logger = False

    model = TrainingModule(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        accelerator="gpu",
        devices='auto',
        strategy=DDPStrategy(timeout=timedelta(hours=12)),
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
        cfg.extract.width
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataloader_num_workers,
        worker_init_fn=detach_worker_stdin,
    )
    
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    main()
