import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = layer_init(nn.Linear(num_inputs, 64))
        self.affine2 = layer_init(nn.Linear(64, 64))

        self.action_mean = layer_init(nn.Linear(64, num_outputs), std=0.01)
        # self.action_mean.weight.data.mul_(0.1)
        # self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = layer_init(nn.Linear(num_inputs, 64))
        self.affine2 = layer_init(nn.Linear(64, 64))
        self.value_head = layer_init(nn.Linear(64, 1), std=1.0)
        # self.value_head.weight.data.mul_(0.1)
        # self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values
