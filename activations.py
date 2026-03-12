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


class TimeScaledLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=None, bias=True):
        super(TimeScaledLinear, self).__init__()
        self.l0 = nn.Linear(in_features, in_features, bias=bias)
        self.activation = activation if activation is not None else nn.SiLU()
        self.l1 = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, dt):
        return self.activation(self.l1(self.activation(self.l0(x) * dt)))


nonlinearities = {'silu': nn.SiLU(), 'gelu': nn.GELU(), 'selu': nn.SELU(), 'leaky': nn.LeakyReLU(),
                  'grow': GrowingCosine(), 'elish': ELiSH(), 'sinlu': SinLU(), 'psinlu': ParameterSinLU(),
                  'mish': nn.Mish()}


class GatedTransition(nn.Module):
    def __init__(self, z_dim, u_dim, hid_dim):
        super(GatedTransition, self).__init__()

        self.gate = nn.Sequential(nn.Linear(z_dim + u_dim, hid_dim),
                                  nn.ReLU(),
                                  nn.Linear(hid_dim, z_dim),
                                  nn.Sigmoid())

        self.proposed_mean = nn.Sequential(nn.Linear(z_dim + u_dim, hid_dim),
                                           nn.ReLU(),
                                           nn.Linear(hid_dim, z_dim))

        self.z_to_mu = nn.Linear(z_dim + u_dim, z_dim)
        # modify the default initialization of z_to_mu
        # so that it starts out as the identity function
        self.z_to_mu.weight.data = torch.eye(z_dim)
        self.z_to_mu.bias.data = torch.zeros(z_dim)

        self.z_to_logvar = nn.Linear(z_dim, z_dim)
        self.relu = nn.ReLU()

    def forward(self, z_t_1, u=None):
        u = u if u is not None else torch.zeros(z_t_1.size())
        aug_z = torch.cat([z_t_1, u], dim=-1)
        gate = self.gate(aug_z)
        proposed_mean = self.proposed_mean(aug_z)
        mu = (1 - gate) * self.z_to_mu(aug_z) + gate * proposed_mean
        logvar = self.z_to_logvar(self.relu(proposed_mean))
        # sampling
        eps = torch.randn(z_t_1.size())
        z_t = mu + eps * torch.exp(.5 * logvar)
        return z_t, mu, logvar


class Combiner(nn.Module):
    # PostNet
    def __init__(self, z_dim, hid_dim):
        super(Combiner, self).__init__()
        self.z_dim = z_dim
        self.z_to_hidden = nn.Linear(z_dim, hid_dim)
        self.hidden_to_mu = nn.Linear(hid_dim, z_dim)
        self.hidden_to_logvar = nn.Linear(hid_dim, z_dim)
        self.tanh = nn.Tanh()

    def forward(self, z_t_1, h_rnn):
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.5 * (self.tanh(self.z_to_hidden(z_t_1)) + h_rnn)
        # use the combined hidden state
        # to compute the mean used to sample z_t
        mu = self.hidden_to_mu(h_combined)
        # use the combined hidden state
        # to compute the scale used to sample z_t
        logvar = self.hidden_to_logvar(h_combined)
        eps = torch.randn(z_t_1.size())
        z_t = mu + eps * torch.exp(.5 * logvar)
        return z_t, mu, logvar


class Emitter(nn.Module):
    def __init__(self, z_dim, hid_dim, input_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.z_to_hidden = nn.Linear(z_dim, hid_dim)
        self.hidden_to_hidden = nn.Linear(hid_dim, hid_dim)
        self.hidden_to_input_mu = nn.Linear(hid_dim, input_dim)
        self.logvar = nn.Parameter(torch.ones(input_dim))
        self.relu = nn.ReLU()

    def forward(self, z_t):
        h1 = self.relu(self.z_to_hidden(z_t))
        h2 = self.relu(self.hidden_to_hidden(h1))
        mu = self.hidden_to_input_mu(h2)
        # return mu  # x_t
        eps = torch.randn(z_t.size(0), self.input_dim)
        x_t = mu + eps * torch.exp(.5 * self.logvar)
        return x_t, mu, self.logvar