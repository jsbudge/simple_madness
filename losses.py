from torch import nn, Tensor
import torch

class KLDivProb(nn.Module):

    def forward(self, mu1, logvar1, mu2=None, logvar2=None):

        if mu2 is None:
            mu2 = torch.zeros(1, device=mu1.device)

        if logvar2 is None:
            logvar2 = torch.zeros(1, device=mu1.device)

        return torch.sum(0.5 * (
                logvar2 - logvar1 + (torch.exp(logvar1) + (mu1 - mu2).pow(2))
                / torch.exp(logvar2) - torch.ones(1, device=mu1.device)
        ), 1)