import gym
import torch as th
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

class GridExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.fc_out = nn.Linear(288, features_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = nn.Flatten()(x)
        x = F.relu(self.fc_out(x))
        return x
