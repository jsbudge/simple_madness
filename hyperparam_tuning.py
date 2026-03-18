import optuna
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import KalmanDataModuleCV
from torch_model import MMClassifier
import numpy as np
import yaml
from functools import partial

def objective(trial: optuna.Trial, cf=None, d0=None, d1=None):
    cf['class_model']['lr'] = trial.suggest_float('lr', 1e-7, 1e-1, log=True)
    cf['class_model']['warmup'] = trial.suggest_categorical('warmup', [100, 3500, 5000])
    cf['class_model']['state_sz'] = trial.suggest_int('state_sz', 10, 200)
    cf['class_model']['activation'] = trial.suggest_categorical('activation', ['silu', 'psinlu', 'leaky', 'mish'])

    model = MMClassifier(**cf['class_model'])
    train_params = dict(logger=loggers.TensorBoardLogger(cf['class_model']['training']['log_dir'], name='class_opt'),
                        max_epochs=cf['class_model']['training']['max_epochs'], num_sanity_val_steps=0,
                  log_every_n_steps=cf['class_model']['training']['log_epoch'], callbacks=
                  [EarlyStopping(monitor='train_loss', patience=cf['class_model']['training']['patience'],
                                 check_finite=True),
                   StochasticWeightAveraging(swa_lrs=1e-6,
                                             swa_epoch_start=cf['class_model']['training']['swa_start']),
                   ModelCheckpoint(monitor='train_loss')])
    trainer = Trainer(**train_params)
    trainer.fit(model, datamodule=d1)
    return trainer.callback_metrics['val_loss'].data.cpu().numpy()

torch.set_float32_matmul_precision('medium')
# torch.autograd.set_detect_anomaly(True)
gpu_num = 0
device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
seed_everything(np.random.randint(1, 2048), workers=True)

with open('./run_params.yaml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

print('Initial dataset...')
data = KalmanDataModuleCV(**config['class_model']['dataloader'])
data.setup()

print('tournament datasets...')
config['class_model']['dataloader']['is_tourney'] = True
tdata = KalmanDataModuleCV(**config['class_model']['dataloader'])
tdata.setup()

study = optuna.create_study(direction='minimize',
                                storage='sqlite:///db.sqlite3',
                                study_name='madness',
                            load_if_exists=True)
objective = partial(objective, cf=config, d0=data, d1=tdata)
study.optimize(objective, n_trials=100)

study.best_params  # E.g. {'x': 2.002108042}