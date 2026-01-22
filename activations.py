import torch
from torch import nn
import math


def _xavier_init(model):
    """
    Performs the Xavier weight initialization.
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity='linear')
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(module.bias, -bound, bound)
            # nn.init.he_(module.weight)


class GrowingCosine(nn.Module):

    def forward(self, x):
        return x * torch.cos(x)


class ELiSH(nn.Module):

    def forward(self, x):
        return torch.where(x > 0, x * torch.sigmoid(x), (torch.exp(x) - 1) * torch.sigmoid(x))


class SinLU(nn.Module):

    def forward(self, x):
        return (x + torch.sin(x)) * torch.sigmoid(x)


class ParameterSinLU(nn.Module):
    def __init__(self):
        super(ParameterSinLU, self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return torch.sigmoid(x) * (x + self.a * torch.sin(self.b * x))


nonlinearities = {'silu': nn.SiLU(), 'gelu': nn.GELU(), 'selu': nn.SELU(), 'leaky': nn.LeakyReLU(),
                  'grow': GrowingCosine(), 'elish': ELiSH(), 'sinlu': SinLU(), 'psinlu': ParameterSinLU(),
                  'mish': nn.Mish()}
