import torch.nn as nn
import torch
from models.transformer import ActionTransformer
from diffusion.sampler import reverse_diffusion_sampling
from models.observation_encoder import ObservationEncoder

class policy(nn.Module):
    def __init__(self, horizon=10, action_dims=7):
        super().__init__()
        self.obs_encoder = ObservationEncoder()
        self.diffusion_transformer = ActionTransformer()
        self.H = horizon
        self.D = action_dims
        self.B = 1

    def forward(self, rgb_static, rgb_gripper, language_instruction):
        """
        rgb_static: [1, 3, 200, 200]
        rgb_gripper: [1, 3, 84, 84]
        robot_obs: [1, 15]
        instruction: str
        """
        s, g, l = self.obs_encoder(rgb_static, rgb_gripper, language_instruction)
        clean_action_chunk = reverse_diffusion_sampling(self.diffusion_transformer, self.B, s, g, l, H=self.H, D=self.D, sampling_steps=20)

        assert clean_action_chunk.shape == (self.B, self.H, self.D)

        return clean_action_chunk

