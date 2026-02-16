"""Diffusion noise scheduler for ARC-IT.

Implements a simple linear noise schedule for training and inference.
During training, noise is added to output grid embeddings and the model
learns to denoise. During inference, we start from pure noise and
iteratively denoise to produce the output grid.

This is a minimal, clean implementation suitable for our discrete-output
use case. The continuous diffusion operates on patch embeddings, and we
apply CrossEntropy loss on the decoded discrete logits.
"""

import torch
import torch.nn as nn


class DiffusionScheduler(nn.Module):
    """Linear noise schedule for training and inference."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to clean samples.

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_0: (B, N, C) clean patch embeddings.
            noise: (B, N, C) Gaussian noise (same shape as x_0).
            timesteps: (B,) integer timesteps.

        Returns:
            (B, N, C) noisy patch embeddings at time t.
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

    @torch.no_grad()
    def denoise_step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Single DDPM denoising step: x_t -> x_{t-1}.

        Args:
            model_output: (B, N, C) predicted noise from the Sana backbone.
            x_t: (B, N, C) current noisy state.
            t: Current timestep (integer).

        Returns:
            (B, N, C) slightly less noisy state x_{t-1}.
        """
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        # Predict x_0 from noise prediction
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]
        pred_mean = sqrt_recip_alpha * (
            x_t - beta_t / self.sqrt_one_minus_alphas_cumprod[t] * model_output
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            return pred_mean + sigma * noise
        else:
            return pred_mean

    @torch.no_grad()
    def inference_loop(
        self,
        denoise_fn,
        shape: tuple,
        conditioning: torch.Tensor,
        num_inference_steps: int = 50,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Full denoising loop from pure noise to clean output.

        Args:
            denoise_fn: Callable(x_t, conditioning, timestep) -> noise prediction.
            shape: (B, N, C) shape of the output.
            conditioning: (B, M, C) encoder conditioning.
            num_inference_steps: Number of denoising steps.
            device: Target device.

        Returns:
            (B, N, C) denoised patch embeddings.
        """
        if device is None:
            device = conditioning.device

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Evenly spaced timesteps for inference
        step_size = self.num_train_timesteps // num_inference_steps
        timesteps = list(range(self.num_train_timesteps - 1, -1, -step_size))

        for t in timesteps:
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            noise_pred = denoise_fn(x, conditioning, t_tensor)
            x = self.denoise_step(noise_pred, x, t)

        return x
