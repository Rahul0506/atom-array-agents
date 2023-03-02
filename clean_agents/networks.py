import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from torch import einsum


class MaskedCategorical(Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):
        self.mask = mask
        if mask is None:
            super(MaskedCategorical, self).__init__(logits=logits)
        else:
            self.mask_value = torch.tensor(
                torch.finfo(logits.dtype).min, dtype=logits.dtype, device=logits.device
            )
            logits = torch.where(self.mask, logits, self.mask_value)
            super(MaskedCategorical, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -p_log_p.sum(-1)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MaskedAgent(nn.Module):
    def __init__(self, envs, device):
        super().__init__()

        # Specify params
        feature_dims = 64
        kernel1_size, kernel2_size = 3, 3
        conv2_channels = 32

        # Calculate values
        input_width = envs.single_observation_space.shape[1]
        conv1_out = input_width + 2 - kernel1_size + 1
        conv2_out = conv1_out - kernel2_size + 1
        flattened_dims = (conv2_out**2) * conv2_channels

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel1_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, conv2_channels, kernel2_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flattened_dims, feature_dims),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(  # Observation -> Valuation
            self.extractor,
            layer_init(nn.Linear(feature_dims, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = (
            nn.Sequential(  # Observation + Valid Actions -> Action -> Next Observation
                self.extractor,
                layer_init(nn.Linear(feature_dims, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            )
        )

        self.masking_kernel = torch.tensor(
            [
                [
                    [[0, 1000, 0], [10, 10000, 1], [0, 100, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 100000, 0], [0, 0, 0]],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)

        # Get action mask and set logits
        has_atoms = torch.amax(x, (1, 2, 3))

        padded = F.pad(x, (1,) * 4, value=2)
        detect = F.conv2d(padded, self.masking_kernel)
        inter_masks = torch.amax(detect, (1, 2, 3))

        inter_masks = inter_masks + inter_masks * (has_atoms == 2)
        masks_ = torch.column_stack(
            (
                inter_masks % 10000 < 2000,
                inter_masks % 1000 < 200,
                inter_masks % 100 < 20,
                inter_masks % 10 < 2,
                torch.logical_and(has_atoms == 1, inter_masks % 100000 >= 10000),
                has_atoms == 2,
            )
        ).to(x.device)

        probs = MaskedCategorical(logits=logits, mask=masks_)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class SeparateNets(MaskedAgent):
    def __init__(self, envs, device):
        super().__init__(envs, device)

        # Specify params
        feature_dims = 64
        kernel1_size, kernel2_size = 3, 3
        conv2_channels = 32

        # Calculate values
        input_width = envs.single_observation_space.shape[1]
        conv1_out = input_width + 2 - kernel1_size + 1
        conv2_out = conv1_out - kernel2_size + 1
        flattened_dims = (conv2_out**2) * conv2_channels

        self.critic = nn.Sequential(  # Observation -> Valuation
            nn.Conv2d(3, 32, kernel1_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, conv2_channels, kernel2_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(flattened_dims, feature_dims),
            nn.ReLU(),
            layer_init(nn.Linear(feature_dims, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = (
            nn.Sequential(  # Observation + Valid Actions -> Action -> Next Observation
                nn.Conv2d(3, 32, kernel1_size, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, conv2_channels, kernel2_size),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(flattened_dims, feature_dims),
                nn.ReLU(),
                layer_init(nn.Linear(feature_dims, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
            )
        )

        self.masking_kernel = torch.tensor(
            [
                [
                    [[0, 1000, 0], [10, 10000, 1], [0, 100, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 100000, 0], [0, 0, 0]],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )


class AgentOld5x5(MaskedAgent):
    def __init__(self, envs, device):
        super().__init__(envs, device)

        feature_dims = 64

        self.extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(288, feature_dims),
        )

        self.critic = nn.Sequential(
            self.extractor,
            layer_init(nn.Linear(feature_dims, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            self.extractor,
            layer_init(nn.Linear(feature_dims, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
