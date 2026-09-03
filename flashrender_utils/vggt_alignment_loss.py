import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from vggt import VGGT

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class VGGTAlignmentLoss(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
        self.vggt_projector = nn.Conv3d(dim, self.vggt_model.embed_dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def vggt_processor(self, images: Tensor): 
        # images: b t c h w 
        b = images.shape[0] 
        images = rearrange(images, 'b f c h w -> (b f) c h w')
        images_resized = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
        images_resized = rearrange(images_resized, '(b f) c h w -> b f c h w', b=b)
        images_resized = torch.clamp(images_resized, 0.0, 1.0)
        return images_resized
    
    def forward(self, latents, images, h, w):
        _, latents_src = latents 

        latents_src = self.vggt_projector(latents_src)
        latents_src = rearrange(latents_src, 'b c f h w -> b c (f h w)')
        latent_norm = F.normalize(latents_src, p=2, dim=1)

        with torch.no_grad():
            images = self.vggt_processor(images) # [B, 3, T, H, W]
            patch_start_idx, vggt_feat = self.vggt_model.shortcut_forward(images, num_layers=1) #  24x []
            img_h, img_w = images.shape[-2:]
            h_tok, w_tok = img_h // 14, img_w // 14
            vggt_feat = rearrange(vggt_feat[:, :, patch_start_idx:], 'kb t (h w) c -> kb t h w c', h=h_tok, w=w_tok)
            vggt_feat = torch.cat(
                [torch.repeat_interleave(vggt_feat[:, 0:1], repeats=4, dim=1), vggt_feat[:, 1:]], dim=1
            )
            vggt_feat = rearrange(vggt_feat, 'b f h w c -> b c f h w')
            vggt_feat = F.interpolate(vggt_feat, size=(vggt_feat.shape[2]//4, h, w), mode='trilinear', align_corners=False)
            vggt_feat = rearrange(vggt_feat, 'b c f h w -> b c (f h w)')
            vggt_norm = F.normalize(vggt_feat, p=2, dim=1)

        alignment_loss = mean_flat(-(vggt_norm * latent_norm).sum(dim=1)).mean()
        return alignment_loss

