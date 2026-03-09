import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import GameDataModuleCV
from torch_model import StateTransition, NeuralKF
import numpy as np
from tqdm import tqdm
import itertools
import yaml
from pathlib import Path
from scipy.optimize import minimize


if __name__ == '__main__':
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

    data = GameDataModuleCV(**config['dataloader'], game_num=1)
    data.setup()

    # Get the model, experiment, logger set up for measurement function and embedding training
    config['model']['init_size'] = data.train_dataset.data_len
    mdl_name = f"{config['model']['name']}"
    model = NeuralKF(**config['model'], meas_path=f"{config['model']['training']['weights_path']}/Measurement.ckpt")
    logger = loggers.TensorBoardLogger(config['model']['training']['log_dir'], version=0, name=mdl_name)
    expected_lr = max((config['model']['lr'] * config['model']['scheduler_gamma'] ** (config['model']['training']['max_epochs'] *
                                                                config['model']['training']['swa_start'])), 1e-9)
    print("======= Training =======")
    trainer = Trainer(logger=logger, max_epochs=config['model']['training']['max_epochs'],
                      default_root_dir=config['model']['training']['weights_path'], num_sanity_val_steps=0,
                      log_every_n_steps=config['model']['training']['log_epoch'], callbacks=
                      [EarlyStopping(monitor='train_loss', patience=config['model']['training']['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=config['model']['training']['swa_start']),
                       ModelCheckpoint(monitor='train_loss')])
    model.train()
    trainer.fit(model, datamodule=data)
    model.eval()
    if config['model']['training']['save_model']:
        trainer.save_checkpoint(f"{config['model']['training']['weights_path']}/{mdl_name}.ckpt")


