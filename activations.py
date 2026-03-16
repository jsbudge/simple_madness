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

        self.gate = nn.Sequential(nn.Linear(z_dim, hid_dim),
                                  nn.SiLU(),
                                  nn.Linear(hid_dim, z_dim),
                                  nn.Sigmoid())

        self.proposed_mean = nn.Sequential(nn.Linear(z_dim, hid_dim),
                                           nn.LeakyReLU(),
                                           nn.Linear(hid_dim, z_dim))

        self.z_to_mu = nn.Linear(z_dim, z_dim)
        # modify the default initialization of z_to_mu
        # so that it starts out as the identity function
        self.z_to_mu.weight.data = torch.eye(z_dim)
        self.z_to_mu.bias.data = torch.zeros(z_dim)

        self.z_to_logvar = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_dim, z_dim),
            nn.Softplus()
        )

        self.u_to_mu = nn.Sequential(
            nn.Linear(u_dim, z_dim),
            nn.SiLU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, z_t_1, u=None):
        u = u if u is not None else torch.zeros(1)
        gate = self.gate(z_t_1)
        proposed_mean = self.proposed_mean(z_t_1) + self.u_to_mu(u)
        mu = (1 - gate) * self.z_to_mu(z_t_1) + gate * proposed_mean
        logvar = self.z_to_logvar(proposed_mean)
        return mu, logvar


class Combiner(nn.Module):
    # PostNet
    def __init__(self, z_dim, u_dim, hid_dim):
        super(Combiner, self).__init__()
        self.z_dim = z_dim
        self.hid_dim = hid_dim
        self.z_to_hidden = nn.Linear(z_dim, hid_dim)
        self.hidden_to_mu = nn.Linear(hid_dim, z_dim)
        self.u_to_mu = nn.Linear(u_dim, z_dim)
        self.hidden_to_logvar = nn.Sequential(
            nn.Linear(hid_dim, z_dim),
            nn.Softplus()
        )
        self.tanh = nn.Softsign()

    def forward(self, z_t_1, h_rnn, u):
        # combine the rnn hidden state with a transformed version of z_t_1
        h_combined = 0.3333333333333333333 * (self.tanh(self.z_to_hidden(z_t_1)) +
                                              h_rnn[..., :self.hid_dim] + h_rnn[..., -self.hid_dim:])
        # use the combined hidden state
        # to compute the mean used to sample z_t
        mu = self.hidden_to_mu(h_combined) + self.u_to_mu(u)
        # use the combined hidden state
        # to compute the scale used to sample z_t
        logvar = self.hidden_to_logvar(h_combined)
        return mu, logvar


class Emitter(nn.Module):
    """
    Parametrizes F_k, the distribution used to sample output values of x given
    state z.
    """
    def __init__(self, z_dim, hid_dim, input_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.z_to_x = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.SiLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(.1),
            nn.SiLU(),
            nn.Linear(hid_dim, input_dim),
        )

    def forward(self, z_t):
        mu = self.z_to_x(z_t)
        return mu