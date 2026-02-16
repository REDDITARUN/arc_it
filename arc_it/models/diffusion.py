"""Diffusion noise scheduler for ARC-IT.

Implements a simple linear noise schedule for training and inference.
During training, noise is added to output grid embeddings and the model
learns to denoise (x_0 prediction). During inference, we support both
single-step prediction and multi-step DDPM with x_0 parameterization.

The model predicts x_0 directly (not epsilon), so inference uses the
DDPM posterior formula for x_0 prediction.
"""

import torch
import torch.nn as nn


class DiffusionScheduler(nn.Module):
    """Linear noise schedule with x_0 prediction for training and inference."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps

        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Prepend alpha_bar_0 = 1.0 for the posterior at t=1
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Posterior variance: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        self.register_buffer("posterior_variance", posterior_variance)

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise"""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps][:, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

    @torch.no_grad()
    def denoise_step_x0(
        self,
        x_0_pred: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """Single DDPM denoising step with x_0 prediction: x_t -> x_{t-1}.

        Uses the DDPM posterior mean formula for x_0 parameterization:
        mu_posterior = (sqrt(alpha_bar_{t-1}) * beta_t / (1-alpha_bar_t)) * x_0_pred
                     + (sqrt(alpha_t) * (1-alpha_bar_{t-1}) / (1-alpha_bar_t)) * x_t
        """
        if t == 0:
            return x_0_pred

        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod_prev[t]
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]

        # Posterior mean coefficients
        coeff_x0 = torch.sqrt(alpha_bar_t_prev) * beta_t / (1.0 - alpha_bar_t)
        coeff_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)

        posterior_mean = coeff_x0 * x_0_pred + coeff_xt * x_t

        # Add noise scaled by posterior variance
        noise = torch.randn_like(x_t)
        posterior_std = torch.sqrt(self.posterior_variance[t])
        return posterior_mean + posterior_std * noise

    @torch.no_grad()
    def inference_loop(
        self,
        denoise_fn,
        shape: tuple,
        conditioning: torch.Tensor,
        num_inference_steps: int = 50,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Full denoising loop from pure noise to clean output (x_0 prediction).

        The denoise_fn is expected to predict x_0 directly (not epsilon).
        """
        if device is None:
            device = conditioning.device

        x = torch.randn(shape, device=device)

        step_size = max(self.num_train_timesteps // num_inference_steps, 1)
        timesteps = list(range(self.num_train_timesteps - 1, -1, -step_size))

        for t in timesteps:
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x_0_pred = denoise_fn(x, conditioning, t_tensor)
            x = self.denoise_step_x0(x_0_pred, x, t)

        return x

    @torch.no_grad()
    def single_step_predict(
        self,
        denoise_fn,
        shape: tuple,
        conditioning: torch.Tensor,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Single-step prediction: pass noise at max timestep, get x_0 directly.

        This is faster than the full loop and works well because the model
        is trained to predict x_0 from any noise level, and at t=T-1 the
        conditioning signal dominates the prediction.
        """
        if device is None:
            device = conditioning.device

        noise = torch.randn(shape, device=device)
        t_max = self.num_train_timesteps - 1
        t_tensor = torch.full((shape[0],), t_max, device=device, dtype=torch.long)
        return denoise_fn(noise, conditioning, t_tensor)
