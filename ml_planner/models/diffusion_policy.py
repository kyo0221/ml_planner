import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditional_unet1d import ConditionalUNet1D
from .vision_encoder import VisionEncoder


class DiffusionPolicy(nn.Module):
    NUM_COMMANDS = 4

    def __init__(
        self,
        action_dim: int,
        pred_horizon: int,
        n_obs_steps: int,
        diffusion_step_embed_dim: int = 256,
        global_cond_dim: int = 512,
        down_dims: tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.n_obs_steps = n_obs_steps

        self.vision_encoder = VisionEncoder()
        vision_dim = self.vision_encoder.feature_dim * n_obs_steps

        self.command_encoder = nn.Sequential(
            nn.Linear(self.NUM_COMMANDS, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        self.global_condition = nn.Sequential(
            nn.Linear(vision_dim + 128, global_cond_dim),
            nn.SiLU(),
            nn.Linear(global_cond_dim, global_cond_dim),
        )
        self.denoiser = ConditionalUNet1D(
            input_dim=action_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

    def encode_condition(self, obs_images: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        image_feature = self.vision_encoder(obs_images).flatten(start_dim=1)
        command_feature = self.command_encoder(command)
        return self.global_condition(torch.cat((image_feature, command_feature), dim=1))

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, obs_images: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        global_cond = self.encode_condition(obs_images, command)
        return self.denoiser(noisy_actions, timesteps, global_cond)

    def compute_loss(
        self,
        obs_images: torch.Tensor,
        command: torch.Tensor,
        actions: torch.Tensor,
        noise_scheduler,
    ) -> torch.Tensor:
        batch_size = actions.shape[0]
        timesteps = torch.randint(
            low=0,
            high=noise_scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=actions.device,
        )
        noise = torch.randn_like(actions)
        noisy_actions = noise_scheduler.add_noise(actions, noise, timesteps)
        pred_noise = self.forward(noisy_actions, timesteps, obs_images, command)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample_actions(
        self,
        obs_images: torch.Tensor,
        command: torch.Tensor,
        noise_scheduler,
        num_inference_steps: int,
    ) -> torch.Tensor:
        global_cond = self.encode_condition(obs_images, command)
        device = obs_images.device
        sample = torch.randn(
            obs_images.shape[0],
            self.pred_horizon,
            self.action_dim,
            device=device,
            dtype=obs_images.dtype,
        )
        noise_scheduler.set_timesteps(num_inference_steps, device=device)
        for timestep in noise_scheduler.timesteps:
            timestep_batch = torch.full((sample.shape[0],), int(timestep), device=device, dtype=torch.long)
            model_output = self.denoiser(sample, timestep_batch, global_cond)
            sample = noise_scheduler.step(model_output, timestep, sample).prev_sample
        return sample

    def get_config(self) -> dict:
        return {
            'action_dim': self.action_dim,
            'pred_horizon': self.pred_horizon,
            'n_obs_steps': self.n_obs_steps,
        }
