from copy import deepcopy
from typing import Any
import numpy as np
from activations import GrowingCosine, ParameterSinLU, _xavier_init, TimeScaledLinear
import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.nn import functional as tf
from scipy.linalg import sqrtm


class Measurement(LightningModule):

    def __init__(self, init_size: int = 70, state_size: int = 100, meas_size: int = 4, lr: float = 1e-5, weight_decay: float = 0.0,
                  scheduler_gamma: float = .7, betas: tuple[float, float] = (.9, .99), *args: Any, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.embedding_function = nn.Sequential(
            nn.Linear(init_size, state_size),
            ParameterSinLU(),
            nn.Linear(state_size, state_size),
            nn.Tanh(),
        )

        self.measure_function = nn.Sequential(
            nn.Linear(state_size * 2, state_size),
            ParameterSinLU(),
            nn.Linear(state_size, state_size),
            GrowingCosine(),
            nn.Linear(state_size, meas_size),
            nn.Sigmoid()
        )

        _xavier_init(self)

    def forward(self, x, y):
        """

        :param x: Team stats to encode.
        :param y: This is the stats of the team in the tournament to run through predict_head
        :return: probability of team x winning against team y.
        """
        return self.measure_function(torch.cat([x, y]))

    def training_forward(self, x, y):
        x = self.embedding_function(x)
        y = self.embedding_function(y)
        return self.measure_function(torch.cat([x, y], dim=-1))

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
        team, opp, targets, _ = batch

        results = self.training_forward(team, opp)
        train_loss = self.loss_function(results, targets)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss

    def minimization(self, x, x_hat, y, targets):
        results = self.training_forward(torch.tensor(np.concatenate((x_hat, x))), torch.tensor(y))
        return self.loss_function(results, torch.tensor(targets)).data.numpy().flatten()


class StateTransition(LightningModule):
    def __init__(self, init_size: int = 70, state_size: int = 100, meas_size: int = 10, lr: float = 1e-5, weight_decay: float = 0.0,
                  scheduler_gamma: float = .7, betas: tuple[float, float] = (.9, .99), *args: Any, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.transition = nn.Sequential(
            nn.Linear(state_size, state_size),
            ParameterSinLU(),
            nn.Linear(state_size, state_size),
            nn.Tanh(),
        )

        # This is only for training purposes
        self.measure = Measurement(init_size, state_size, meas_size)
        self.measure.requires_grad = False

        _xavier_init(self)

    def forward(self, x):
        """

        :param x: Team stats to encode.
        :param y: This is the stats of the team in the tournament to run through predict_head
        :return: probability of team x winning against team y.
        """
        return self.transition(x)

    def training_forward(self, x, y):
        x = self.transition(x)
        y = self.transition(y)
        return self.measure(torch.cat([x, y]))

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
        team, opp, targets = batch

        results = self.training_forward(team, opp)
        train_loss = self.loss_function(results, targets)

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss


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
        x_out = torch.sum(self.mean_weights[:, None] * sigmas_out, dim=-2)

        diff = sigmas_out - x_out
        p_hat = self.covar_weights * diff.T @ diff + dt * torch.diag(self.q)

        return x_out, sigmas_out, p_hat

    def update(self, curr_x, sigmas, data, p_hat):
        y = self.meas_function(sigmas)
        y_mu = torch.sum(y @ self.x_hat_selector, dim=0)

        y_diff = y - y_mu

        # Measurement covariance
        p_yy = (self.covar_weights * y_diff.T @ y_diff) + torch.diag(self.R)

        p_xy = (sigmas - curr_x).T @ (self.covar_weights[:, None] * y_diff)

        k = p_xy @ torch.linalg.inv(p_yy)
        innovation = data - y_mu

        x = k @ innovation.T
        p_hat = p_hat - (k @ (p_yy @ k.T))
        return x.flatten(), p_hat, y_mu

    def training_forward(self, sigmas, u):
        sigmas = self.state_function(sigmas)
        y = self.meas_function.measure_function(torch.cat([sigmas, u.repeat(self.n_sigmas, 1)], dim=-1))
        return (self.x_hat_selector.to(self.device) @ y).flatten()


    def forward(self, sigmas, u, z, dt):
        x, sigmas, p_hat = self.predict(torch.cat([sigmas, u.repeat(self.n_sigmas, 1)], dim=-1), dt)
        x, p_hat, y_mu = self.update(x, sigmas, z, p_hat)
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
        team_emb = self.meas_function.embedding_function(team)
        opp_emb = self.meas_function.embedding_function(opp)
        sigmas = self.get_sigmas(team_emb)

        y_mu = self.training_forward(sigmas, opp_emb)
        train_loss = self.loss_function(y_mu, targets.flatten())

        self.log_dict({f'{kind}_loss': train_loss}, on_epoch=True,
                      prog_bar=True, rank_zero_only=True)
        return train_loss
