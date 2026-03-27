from copy import deepcopy
from typing import Any
import numpy as np
from activations import _xavier_init, GatedTransition, Emitter, Combiner, nonlinearities, MultiHeadAttention
from losses import KLDivProb
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn import functional as tf
from torch.distributions import MultivariateNormal
from scipy.linalg import sqrtm

from schedulers import CosineWarmupScheduler


class DKF(LightningModule):
    # Structured Inference Networks
    # Current version ignores backward RNN outputs
    def __init__(self, x_dim, z_dim=50, u_dim=2, trans_dim=30, emission_dim=30,
                 rnn_dim=100, num_rnn_layers=1, alpha: float = 1.1, k_param: int = 0, annealing_factor=.01, *args, **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.trans = GatedTransition(z_dim, u_dim, trans_dim)
        self.emitter = Emitter(z_dim, emission_dim, x_dim)
        self.combiner = Combiner(z_dim, u_dim, rnn_dim)

        self.z_0 = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim))
        self.h_0 = nn.Parameter(torch.zeros(2, 1, rnn_dim))
        self.c_0 = nn.Parameter(torch.zeros(2, 1, rnn_dim))

        self._lambda = alpha ** 2 * (z_dim + k_param) - z_dim
        self.emitter_log_sigma = nn.Parameter(torch.ones(x_dim))
        self.mean_weights = torch.tensor(np.array([(self._lambda / (z_dim + self._lambda))
                                                   if i == 0 else 1 / (2 * (z_dim + self._lambda)) for i in
                                                   range(2 * z_dim + 1)]), dtype=torch.float32, requires_grad=False)

        # corresponding learning 'l' in the original code
        # latent state transition matrix substitute
        self.rnn = nn.LSTM(input_size=x_dim,
                          hidden_size=rnn_dim,
                          batch_first=True,
                          bidirectional=True,
                          num_layers=num_rnn_layers)

        self.kldiv = KLDivProb()

    def infer(self, x, y, u):

        assert x.size() == y.size()

        batch_size, T_max, x_dim = x.size()
        h_0 = self.h_0.expand(2, batch_size, self.hparams.rnn_dim).contiguous()
        c_0 = self.c_0.expand(2, batch_size, self.hparams.rnn_dim).contiguous()
        rnn_out, _ = self.rnn(x, (h_0, c_0))

        # encode x which can contain missing values
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))
        kl_states = torch.zeros((batch_size, T_max))
        rec_losses = torch.zeros((batch_size, T_max))

        for t in range(T_max):
            # p(z_t|z_{t-1}, u{t})
            z_prior_mu, z_prior_logvar = self.trans(z_prev, u[:, t])
            # q(z_t|z_{t-1},x_{t:T})
            z_mu, z_logvar = self.combiner(z_prior_mu, rnn_out[:, t], u[:, t])

            z_t = z_mu + torch.rand_like(z_mu) * torch.sqrt(torch.exp(z_logvar))
            # p(x_t|z_t)
            x_t = self.emitter(z_t) + torch.rand(batch_size, x_dim).to(self.device) * torch.sqrt(torch.exp(self.emitter_log_sigma.to(self.device)))

            # compute loss
            kl_states[:, t] = self.kldiv(z_mu, z_logvar, z_prior_mu, z_prior_logvar)

            # error between x and y
            rec_losses[:, t] = nn.MSELoss(reduction='none')(
                x_t.contiguous().view(-1),
                # x_mu.contiguous().view(-1),
                y[:, t].contiguous().view(-1)
            ).view(batch_size, -1).mean(dim=1)

            z_prev = z_t

        return rec_losses.mean(), kl_states.mean()

    def predict(self, x, u, pred_steps=1, num_sample=100, step_by_step=True):
        """ x should contain the prediction period
        """
        # Outputs
        x_hat = torch.zeros(x.size())  # predictions
        x_025 = torch.zeros(x.size())
        x_975 = torch.zeros(x.size())

        batch_size, T_max, _ = x.size()
        assert batch_size == 1
        z_prev = self.z_0.expand(num_sample, self.z_0.size(0))

        if not step_by_step:
            # hide test inputs
            x = deepcopy(x)
            x[:, -pred_steps:] = 0.

        h_0 = self.h_0.expand(2, batch_size, self.hparams.rnn_dim).contiguous()
        c_0 = self.c_0.expand(2, batch_size, self.hparams.rnn_dim).contiguous()
        rnn_out, _ = self.rnn(x, (h_0, c_0))
        rnn_out = rnn_out.expand(num_sample,
                                 rnn_out.size(1), rnn_out.size(2))

        for t in range(T_max - pred_steps):
            # z_t: (num_sample, z_dim)
            z_t, z_mu = self.combiner(z_prev, rnn_out[:, t], u[:, t])
            x_mu = self.emitter(z_t)

            x_covar = torch.diag(torch.sqrt(torch.exp(.5 * self.emitter_log_sigma)))
            x_samples = MultivariateNormal(
                x_mu, covariance_matrix=x_covar).sample()

            x_hat[:, t] = x_samples.mean(0)
            x_025[:, t] = x_samples.quantile(0.025, 0)
            x_975[:, t] = x_samples.quantile(0.975, 0)

            z_prev = z_mu

        for t in range(T_max - pred_steps, T_max):

            rnn_out, _ = self.rnn(x[:, :t], (h_0, c_0))
            rnn_out = rnn_out.expand(num_sample, rnn_out.size(1), rnn_out.size(2))

            z_t_1, z_mu = self.combiner(z_prev, rnn_out[:, -1], u[:, t])
            z_t, z_mu = self.trans(z_t_1, u[:, t])
            x_mu = self.emitter(z_t)

            if not step_by_step:
                # overwrite the next point x with the mean emission value
                x[:, t] = torch.unsqueeze(x_mu.mean(axis=0), 0)

            x_covar = torch.diag(torch.sqrt(torch.exp(.5 * self.emitter_log_sigma)))
            x_samples = MultivariateNormal(
                x_mu, covariance_matrix=x_covar).sample()

            x_hat[:, t] = x_samples.mean(0)
            x_025[:, t] = x_samples.quantile(0.025, 0)
            x_975[:, t] = x_samples.quantile(0.975, 0)

        return x_hat, x_025, x_975, z_mu, z_t

    def get_sigmas(self, z, p):
        """generates sigma points"""

        tmp_mat = (self.hparams.z_dim + self._lambda) * p

        # print spr_mat
        spr_mat, _ = torch.linalg.cholesky_ex(tmp_mat)

        ret = torch.cat([z.unsqueeze(-2), z - tmp_mat, z + tmp_mat], dim=-2)
        # ret = x.repeat(self.n_sigmas, 1)

        return ret

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and self.logger:
            self.logger.log_graph(self, self.example_input_array)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        self.manual_backward(train_loss)
        opt.zero_grad()
        opt.step()
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_train_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        x, u, y, _ = batch
        rec_loss, kl_loss = self.infer(x, y, u)
        total_loss = rec_loss + self.hparams.annealing_factor * kl_loss
        # return rec_loss.item(), kl_loss.item(), total_loss.item()

        self.log_dict({f'{kind}_total_loss': total_loss, f'{kind}_kl_loss': kl_loss, f'{kind}_rec_loss': rec_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return total_loss

    def save_model(self, filename):
        """ dkf.pth """
        torch.save(self.to('cpu').state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))


class MMClassifier(LightningModule):

    def __init__(self, data_sz: int, state_sz: int = 4, game_sz: int = 36, activation: str = 'silu', num_heads: int = 10, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.encode = nn.Sequential(
            nn.Linear(data_sz, state_sz),
            nonlinearities[activation],
            nn.Linear(state_sz, state_sz),
        )

        '''self.home_encode = nn.Sequential(
            nn.Linear(state_sz + 1, state_sz),
            nn.Sigmoid(),
        )'''

        assert state_sz % num_heads == 0, 'State size must be divisible by num_heads'

        self.attention = MultiHeadAttention(state_sz, num_heads)

        self.connection = nn.Sequential(
            nn.Conv1d(game_sz, game_sz, 1),
            nn.SiLU(),
            nn.Conv1d(game_sz, 1, 1),
            nn.LayerNorm(state_sz),
        )

        self.classify = nn.Sequential(
            nn.Linear(state_sz * 2 + 1, state_sz),
            nonlinearities[activation],
            nn.Dropout(p=0.3),
            nn.Linear(state_sz, 1),
            nn.Sigmoid(),
        )

        _xavier_init(self)

    def forward(self, x, y, home):
        xmask = (x[..., 0] > -999).float()
        ymask = (y[..., 0] > -999).float()
        x = self.encode(x)
        y = self.encode(y)
        x0, xa = self.attention(x, y, y, xmask)
        y0, ya = self.attention(y, x, x, ymask)
        x0 = self.connection(x0).squeeze(1)
        y0 = self.connection(y0).squeeze(1)
        return self.classify(torch.cat([x0, y0, home], dim=-1))

    def loss_function(self, y, y_pred):
        return torch.mean((y_pred - y)**2)
        # return tf.binary_cross_entropy(y, y_pred)

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and self.logger:
            self.logger.log_graph(self, self.example_input_array)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss)
        opt.step()
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay)
        scheduler = CosineWarmupScheduler(optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        team, opp, home, targets = batch

        res = self.forward(team, opp, home)
        train_loss = self.loss_function(res, targets)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss
