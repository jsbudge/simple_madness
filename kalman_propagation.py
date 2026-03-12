import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import GameDataModuleCV
from torch_model import NeuralKF
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
    device = 'cpu'  # f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
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
    model = NeuralKF.load_from_checkpoint(f"{config['model']['training']['weights_path']}/NeuralKF.ckpt",
                                          strict=False, weights_only=False)
    model.eval()

    game_data = pd.read_csv(Path(f'{config["load_data"]["save_path"]}/GameDataBasic.csv'))

    gd = game_data.loc[:, 2023, :, :]
    kfs = {}

    for idx, row in gd.iterrows():
        if idx[2] not in kfs.keys():
            kfs[idx[2]] = NeuralKF.load_from_checkpoint(f"{config['model']['training']['weights_path']}/NeuralKF.ckpt",
                                          strict=False, weights_only=False)
        if idx[3] not in kfs.keys():
            kfs[idx[3]] = NeuralKF.load_from_checkpoint(f"{config['model']['training']['weights_path']}/NeuralKF.ckpt",
                                          strict=False, weights_only=False)
        t_state = data.train_dataset.get_state(idx[1], idx[2])
        o_state = data.train_dataset.get_state(idx[1], idx[3])


        kfs[idx[2]](t_state, o_state)







