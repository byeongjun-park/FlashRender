"""DMD2 (Distribution Matching Distillation v2) utilities.

The loss functions here are ported from NVIDIA FastGen
(``fastgen/methods/common_loss.py``). They operate in x0-space, which is why the
training code converts the flow-matching velocity prediction into an x0 estimate
via ``x0 = x_t - sigma * v`` before calling :func:`variational_score_distillation_loss`.

The discriminator (``Discriminator_VideoDiT``) is vendored in-repo at
``flashrender_utils/third_party/fastgen_discriminators.py`` -- a verbatim copy of FastGen's
``fastgen/networks/discriminators.py``, which only depends on ``torch`` -- so this
repo is self-contained and does not need a separate FastGen checkout on disk.
"""

import torch
import torch.nn.functional as F

from flashrender_utils.third_party.fastgen_discriminators import Discriminator_VideoDiT


# ---------------------------------------------------------------------------
# Losses (ported from fastgen/methods/common_loss.py)
# ---------------------------------------------------------------------------
def variational_score_distillation_loss(gen_data, teacher_x0, fake_score_x0, additional_scale=None):
    """VSD loss driving the student distribution towards the teacher distribution.

    The gradient is ``(fake_score_x0 - teacher_x0) * w`` where ``w`` normalises by
    the per-sample teacher/student discrepancy. Everything except ``gen_data`` is
    treated as a constant target.
    """
    dims = tuple(range(1, teacher_x0.ndim))

    with torch.no_grad():
        original_dtype = gen_data.dtype
        gen_data_fp32 = gen_data.float()
        teacher_x0_fp32 = teacher_x0.float()

        diff_abs_mean = (gen_data_fp32 - teacher_x0_fp32).abs().mean(dim=dims, keepdim=True)
        w_fp32 = 1 / (diff_abs_mean + 1e-6)

        if additional_scale is not None:
            w_fp32 = w_fp32 * additional_scale.float().reshape(-1, *([1] * (w_fp32.ndim - 1)))

        w = w_fp32.to(dtype=original_dtype)

        vsd_grad = (fake_score_x0 - teacher_x0) * w
        pseudo_target = gen_data - vsd_grad

    loss = 0.5 * F.mse_loss(gen_data, pseudo_target, reduction="mean")
    return loss


def gan_loss_generator(fake_logits):
    """Non-saturating GAN generator loss."""
    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    return F.softplus(-fake_logits).mean()


def gan_loss_discriminator(real_logits, fake_logits):
    """Non-saturating GAN discriminator loss."""
    assert fake_logits.ndim == 2, f"fake_logits has shape {fake_logits.shape}"
    assert real_logits.ndim == 2, f"real_logits has shape {real_logits.shape}"
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()


# ---------------------------------------------------------------------------
# Discriminator loader (vendored copy, see module docstring)
# ---------------------------------------------------------------------------
def load_video_discriminator(feature_indices, num_blocks, disc_type, inner_dim):
    """Instantiate ``Discriminator_VideoDiT`` (vendored in-repo).

    Args:
        feature_indices: DiT block indices whose activations feed the discriminator.
        num_blocks: total number of DiT blocks (used to validate ``feature_indices``).
        disc_type: architecture key, see ``Discriminator_VideoDiT.ARCHITECTURES``.
        inner_dim: channel dim of the captured DiT features (== ``dit.dim``).

    Returns:
        An ``nn.Module`` mapping ``List[[B, inner_dim, T, H, W]] -> [B, num_features]``.
    """
    return Discriminator_VideoDiT(
        feature_indices=set(feature_indices),
        num_blocks=num_blocks,
        disc_type=disc_type,
        inner_dim=inner_dim,
    )
