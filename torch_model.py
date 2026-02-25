from copy import deepcopy
from typing import Any
import numpy as np
from activations import GrowingCosine, ParameterSinLU, _xavier_init, TimeScaledLinear
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn import functional as tf
from scipy.linalg import sqrtm


class Predictor(LightningModule):

    def __init__(self, init_size: int = 70, state_size: int = 100, lr: float = 1e-5, weight_decay: float = 0.0,
                 encoded_sz: int = 10, sigma: float = 10., scheduler_gamma: float = .7,
                 betas: tuple[float, float] = (.9, .99), *args: Any, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.state_encoder = nn.Sequential(
            nn.Linear(init_size, init_size),
            ParameterSinLU(),
            nn.Linear(init_size, init_size),
            GrowingCosine(),
            nn.Linear(init_size, state_size)
        )

        self.nkf = NeuralKF(state_size, state_size)

        self.predict_head = nn.Sequential(
            nn.Linear(state_size, state_size),
            ParameterSinLU(),
            nn.Linear(state_size, state_size),
            GrowingCosine(),
            nn.Linear(state_size, 1),
            nn.Sigmoid()
        )

        _xavier_init(self)

    def forward(self, x, opp, game_res, y):
        """

        :param x: Team stats to encode.
        :param opp: NxM array of stats for opponents, where N is the number of games played by the team against them.
        :param game_res: NxS Kalman filter measurement prediction of game results, where S is the number of stats to include.
        :param y: This is the stats of the team in the tournament to run through predict_head
        :return: probability of team x winning against team y.
        """
        x = self.state_encoder(x)
        opp = self.state_encoder(opp)

        # for o in opp[0]:
        #     x = self.nkf(x, o, game_res, .1)
        x = self.nkf(x, opp[0][0], game_res[0], .1)

        x = self.predict_head(x.reshape(1, -1))
        return x

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=self.hparams.scheduler_gamma)
        '''scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=self.params['step_size'],
                                                         factor=self.params['scheduler_gamma'], threshold=1e-5)'''

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def train_val_get(self, batch, batch_idx, kind='train'):
        team, opp, z, y, targets = batch

        results = self.forward(team, opp, z, y)
        train_loss = self.loss_function(results, targets)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss


class NeuralKF(LightningModule):

    def __init__(self, state_sz: int, force_sz: int, meas_size: int = 4, alpha: float = 1.1, k: float = 0., beta_filter: float = 2.):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.q = nn.Parameter(torch.randn(state_sz), requires_grad=True)
        self.R = nn.Parameter(torch.randn(meas_size), requires_grad=True)
        self.p = torch.eye(state_sz, requires_grad=False)
        self._lambda = alpha ** 2 * (state_sz + k) - state_sz

        # Sigmas for weights to sample UKF
        self.n_sigmas = 1 + state_sz * 2
        self.covar_weights = torch.tensor(np.array([(self._lambda / (state_sz + self._lambda)) + (1 - alpha ** 2 + beta_filter)
                                       if i == 0 else 1 / (2 * (state_sz + self._lambda)) for i in
                                       range(self.n_sigmas)]), dtype=torch.float32, requires_grad=False)
        self.mean_weights = torch.tensor(np.array([(self._lambda / (state_sz + self._lambda))
                                      if i == 0 else 1 / (2 * (state_sz + self._lambda)) for i in range(self.n_sigmas)]), dtype=torch.float32, requires_grad=False)

        # State function receives the state of the team and the encoded stats of its opponents
        # as a forcing function.
        # self.state_function = TimeScaledLinear(state_sz + force_sz, state_sz)
        self.state_function = nn.Sequential(
            nn.Linear(state_sz + force_sz, state_sz),
            nn.SiLU(),
            nn.Linear(state_sz, state_sz),
        )

        # Measurement function to go from state to measurements
        self.meas_function = nn.Sequential(
            nn.Linear(state_sz, state_sz),
            nn.SiLU(),
            nn.Linear(state_sz, meas_size),
            nn.Sigmoid(),
        )

        _xavier_init(self)

    def get_sigmas(self, x):
        """generates sigma points"""

        tmp_mat = (self.hparams.state_sz + self._lambda) * self.p

        # print spr_mat
        # spr_mat, _ = torch.linalg.cholesky_ex(tmp_mat)

        ret = torch.cat([x, x - tmp_mat, x + tmp_mat], dim=0)
        # ret = x.repeat(self.n_sigmas, 1)

        return ret

    def predict(self, sigmas, dt):
        sigmas_out = self.state_function(sigmas) * dt
        x_out = torch.sum(self.mean_weights[:, None] * sigmas_out, dim=-2)

        diff = sigmas_out - x_out
        self.p = self.covar_weights * diff.T @ diff + dt * torch.diag(self.q)

        return x_out, sigmas_out

    def update(self, curr_x, sigmas, data):
        y = self.meas_function(sigmas)
        y_mu = self.meas_function(curr_x)

        y_diff = y - y_mu
        x_diff = sigmas - curr_x

        # Measurement covariance
        p_yy = (self.covar_weights * y_diff.T @ y_diff) + torch.diag(self.R)

        p_xy = x_diff.T @ (self.covar_weights[:, None] * y_diff)

        k = p_xy @ torch.linalg.inv(p_yy)

        x = k @ (data - y_mu.reshape(1, -1)).T
        self.p = self.p - (k @ (p_yy @ k.T))
        return x.flatten()

    def forward(self, x, u, z, dt):
        sigmas = self.get_sigmas(x)
        x, sigmas = self.predict(torch.cat([sigmas, u.repeat(self.n_sigmas, 1)], dim=-1), dt)
        x = self.update(x, sigmas, z)
        return x
