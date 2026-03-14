from copy import deepcopy
from typing import Any
import numpy as np
from activations import GrowingCosine, ParameterSinLU, _xavier_init, GatedTransition, Emitter, Combiner
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
        self.z_q_0 = nn.Parameter(torch.zeros(z_dim), requires_grad=False)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim), requires_grad=False)

        self.emitter_log_sigma = nn.Parameter(torch.ones(x_dim))
        self._lambda = alpha ** 2 * (z_dim + k_param) - z_dim
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
        rnn_out, _ = self.rnn(x, (h_0, h_0))

        # encode x which can contain missing values
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))
        kl_states = torch.zeros((batch_size, T_max))
        rec_losses = torch.zeros((batch_size, T_max))

        for t in range(T_max):
            # p(z_t|z_{t-1}, u{t})
            z_prior_mu, z_prior_logvar = self.trans(z_prev, u[:, t])
            # q(z_t|z_{t-1},x_{t:T})
            z_mu, z_logvar = self.combiner(z_prev, rnn_out[:, t], u[:, t])

            # sample z distribution using cholesky
            sig_pts = self.get_sigmas(z_mu,
                                      torch.diag_embed(torch.sqrt(torch.exp(.5 * z_logvar)),
                                                       offset=0, dim1=-2, dim2=-1))
            z_t = torch.mean(sig_pts * self.mean_weights[None, :, None], dim=-2)
            # p(x_t|z_t)
            x_t = torch.mean(self.emitter(sig_pts) * self.mean_weights[None, :, None], dim=-2)

            # compute loss
            kl_states[:, t] = self.kldiv(
                z_mu, z_logvar, z_prior_mu, z_prior_logvar)

            # error between x and y
            rec_losses[:, t] = nn.MSELoss(reduction='none')(
                x_t.contiguous().view(-1),
                # x_mu.contiguous().view(-1),
                y[:, t].contiguous().view(-1)
            ).view(batch_size, -1).mean(dim=1)

            z_prev = z_t

        return rec_losses.mean(), kl_states.mean()

    def filter(self, x, num_sample=100):

        # Outputs
        x_hat = torch.zeros(x.size())  # predictions
        x_025 = torch.zeros(x.size())
        x_975 = torch.zeros(x.size())

        batch_size, T_max, x_dim = x.size()
        assert batch_size == 1
        z_prev = self.z_0.expand(num_sample, self.z_0.size(0))

        h_0 = self.h_0.expand(1, 1, self.hparams.rnn_dim).contiguous()
        rnn_out, _ = self.rnn(x, h_0)
        rnn_out = rnn_out.expand(num_sample,
                                 rnn_out.size(1), rnn_out.size(2))

        for t in range(T_max):
            # z_t: (num_sample, z_dim)
            z_t, z_mu, z_logvar = self.combiner(z_prev, rnn_out[:, t])
            x_t, x_mu, x_logvar = self.emitter(z_t)
            # x_hat[:, t] = x_mu

            x_covar = torch.diag(torch.sqrt(torch.exp(.5 * x_logvar)))
            x_samples = MultivariateNormal(
                x_mu, covariance_matrix=x_covar).sample()
            # # sampling z_t and computing quantiles
            # x_samples = MultivariateNormal(
            #     loc=x_mu, covariance_matrix=x_covar).sample_n(num_sample)

            x_hat[:, t] = x_samples.mean(0)
            x_025[:, t] = x_samples.quantile(0.025, 0)
            x_975[:, t] = x_samples.quantile(0.975, 0)

            # x_hat[:, t] = x_t.mean(0)
            # x_025[:, t] = x_t.quantile(0.025, 0)
            # x_975[:, t] = x_t.quantile(0.975, 0)

            z_prev = z_t
            # z_prev = z_mu

        return x_hat, x_025, x_975

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

        h_0 = self.h_0.expand(1, 1, self.hparams.rnn_dim).contiguous()
        rnn_out, _ = self.rnn(x[:, :T_max - pred_steps], h_0)
        rnn_out = rnn_out.expand(num_sample,
                                 rnn_out.size(1), rnn_out.size(2))

        for t in range(T_max - pred_steps):
            # z_t: (num_sample, z_dim)
            z_t, z_mu, _ = self.combiner(z_prev, rnn_out[:, t], u[:, t])
            _, x_mu, x_logvar = self.emitter(z_t)

            x_covar = torch.diag(torch.sqrt(torch.exp(.5 * x_logvar)))
            x_samples = MultivariateNormal(
                x_mu, covariance_matrix=x_covar).sample()

            x_hat[:, t] = x_samples.mean(0)
            x_025[:, t] = x_samples.quantile(0.025, 0)
            x_975[:, t] = x_samples.quantile(0.975, 0)

            z_prev = z_mu

        for t in range(T_max - pred_steps, T_max):

            rnn_out, _ = self.rnn(x[:, :t], h_0)
            rnn_out = rnn_out.expand(num_sample, rnn_out.size(1), rnn_out.size(2))

            z_t_1, z_mu, _ = self.combiner(z_prev, rnn_out[:, -1], u[:, t])
            z_t, z_mu, _ = self.trans(z_t_1, u[:, t])
            _, x_mu, x_logvar = self.emitter(z_t)

            if not step_by_step:
                # overwrite the next point x with the mean emission value
                x[:, t] = torch.unsqueeze(x_mu.mean(axis=0), 0)

            x_covar = torch.diag(torch.sqrt(torch.exp(.5 * x_logvar)))
            x_samples = MultivariateNormal(
                x_mu, covariance_matrix=x_covar).sample()

            x_hat[:, t] = x_samples.mean(0)
            x_025[:, t] = x_samples.quantile(0.025, 0)
            x_975[:, t] = x_samples.quantile(0.975, 0)

        return x_hat, x_025, x_975

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
        self.lr_schedulers().step()
        opt.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_train_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay,
                                      betas=self.hparams.betas,
                                      eps=1e-7)
        if self.hparams.scheduler_gamma is None:
            return optimizer
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


class NeuralKF(LightningModule):

    def __init__(self, state_sz: int, meas_size: int = 4, alpha: float = 1.1, k_param: float = 0., beta_filter: float = 2.,
                 meas_path: str = None, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.q = nn.Parameter(torch.randn(state_sz), requires_grad=True)
        self.R = nn.Parameter(torch.randn(meas_size), requires_grad=True)
        self.p = torch.eye(state_sz, requires_grad=False, device=self.device)
        self._lambda = alpha ** 2 * (state_sz + k_param) - state_sz

        # Sigmas for weights to sample UKF
        self.n_sigmas = 1 + state_sz * 2
        self.covar_weights = torch.tensor(np.array([(self._lambda / (state_sz + self._lambda)) + (1 - alpha ** 2 + beta_filter)
                                       if i == 0 else 1 / (2 * (state_sz + self._lambda)) for i in
                                       range(self.n_sigmas)]), dtype=torch.float32, requires_grad=False)
        self.mean_weights = torch.tensor(np.array([(self._lambda / (state_sz + self._lambda))
                                      if i == 0 else 1 / (2 * (state_sz + self._lambda)) for i in range(self.n_sigmas)]), dtype=torch.float32, requires_grad=False)
        self.x_hat_selector = torch.zeros(self.n_sigmas, requires_grad=False)
        self.x_hat_selector[0] = 1.

        # State function receives the state of the team and the encoded stats of its opponents
        # as a forcing function.
        # self.state_function = TimeScaledLinear(state_sz + force_sz, state_sz)
        self.state_function = nn.Sequential(
            nn.Linear(state_sz, state_sz),
            ParameterSinLU(),
            nn.Linear(state_sz, state_sz),
            nn.Tanh(),
        )

        # Measurement function to go from state to measurements
        self.meas_function = Measurement.load_from_checkpoint(meas_path, strict=False, weights_only=False)
        for param in self.meas_function.parameters():
            param.requires_grad = False
        # self.meas_function.requires_grad = False

        _xavier_init(self.state_function)

    def get_sigmas(self, x):
        """generates sigma points"""

        tmp_mat = (self.hparams.state_sz + self._lambda) * self.p.to(self.device)

        # print spr_mat
        spr_mat, _ = torch.linalg.cholesky_ex(tmp_mat)

        ret = torch.cat([x, x - spr_mat, x + spr_mat], dim=0)
        # ret = x.repeat(self.n_sigmas, 1)

        return ret

    def predict(self, sigmas, dt):
        sigmas_out = self.state_function(sigmas) * dt
        x_out = torch.sum(self.mean_weights.to(self.device)[:, None] * sigmas_out, dim=-2)

        diff = sigmas_out - x_out
        p_hat = self.covar_weights.to(self.device) * diff.T @ diff + dt * torch.diag(self.q)

        return x_out, sigmas_out, p_hat

    def update(self, curr_x, sigmas, u, data, p_hat):
        y = self.meas_function(sigmas, u.repeat(self.n_sigmas, 1))
        y_mu = self.x_hat_selector.to(self.device) @ y

        y_diff = y - y_mu

        # Measurement covariance
        p_yy = (self.covar_weights.to(self.device) * y_diff.T @ y_diff) + torch.diag(self.R)

        p_xy = (sigmas - curr_x).T @ (self.covar_weights.to(self.device)[:, None] * y_diff)

        k = p_xy @ torch.linalg.inv(p_yy)
        innovation = data - y_mu

        x = k @ innovation.T
        p_hat = p_hat - (k @ (p_yy @ k.T))
        return x.flatten(), p_hat, y_mu


    def forward(self, x_hat, u, z, dt):
        x_hat = self.meas_function.embedding_function(x_hat)
        u = self.meas_function.embedding_function(u)
        sigmas = self.get_sigmas(x_hat)
        x, sigmas, p_hat = self.predict(sigmas, dt)
        x, p_hat, y_mu = self.update(x, sigmas, u, z, p_hat)
        self.p = p_hat.detach()
        return x, y_mu

    def loss_function(self, y, y_pred):
        return tf.binary_cross_entropy(y, y_pred)

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero and self.logger:
            self.logger.log_graph(self, self.example_input_array)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        train_loss = self.train_val_get(batch, batch_idx)
        opt.zero_grad()
        self.manual_backward(train_loss, retain_graph=True)
        opt.step()

    def validation_step(self, batch, batch_idx):
        self.train_val_get(batch, batch_idx, 'val')

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()

        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val_loss"])
        else:
            sch.step()

    def on_validation_epoch_end(self) -> None:
        self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.lr,
                                      weight_decay=self.hparams.weight_decay,
                                      betas=self.hparams.betas,
                                      eps=1e-7)
        if self.hparams.scheduler_gamma is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120,
                                                               eta_min=self.hparams.scheduler_gamma)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        team, opp, targets, _ = batch
        # team_emb = self.meas_function.embedding_function(team)
        # opp_emb = self.meas_function.embedding_function(opp)
        # sigmas = self.get_sigmas(team_emb)


        _, y_mu = self.forward(team, opp, targets, 1.)
        train_loss = self.loss_function(y_mu, targets.flatten())

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss
