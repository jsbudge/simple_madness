import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import GameDataModuleCV
from torch_model import StateTransition, Measurement
import numpy as np
from tqdm import tqdm
import itertools
import threading
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

    data = GameDataModuleCV(**config['dataloader'])
    data.setup()

    # Get the model, experiment, logger set up for measurement function and embedding training
    config['measure_model']['init_size'] = data.train_dataset.data_len
    mdl_name = f"{config['measure_model']['name']}"
    model = Measurement(**config['measure_model'])
    logger = loggers.TensorBoardLogger(config['measure_model']['training']['log_dir'], version=0, name=mdl_name)

    print("======= Training =======")
    for rnd in range(5):
        trainer = Trainer(logger=logger, max_epochs=config['measure_model']['training']['max_epochs'],
                          default_root_dir=config['measure_model']['training']['weights_path'], num_sanity_val_steps=0,
                          log_every_n_steps=config['measure_model']['training']['log_epoch'], callbacks=
                          [ModelCheckpoint(monitor='train_loss')])
        print(f'Training round {rnd}')
        model.train()
        trainer.fit(model, datamodule=data)
        model.eval()
        def min_save(x, y, label, dt):
            min_eval = minimize(model.minimization, x[0, 6:].data.numpy(),
                                (x[0, :6].data.numpy(), y.data.numpy().flatten(),
                                 label.data.numpy().flatten()))
            torch.save(torch.tensor(np.concatenate((x[0, :6].data.numpy(), min_eval['x'])), dtype=torch.float32),
                       Path(f'{config["load_data"]["save_path"]}/state_vectors/{dt[0, 0]}_{dt[0, 1]}.pt'))
        '''threads = [threading.Thread(target=min_save, args=d) for d in data.train_dataloader()]
        threads += [threading.Thread(target=min_save, args=d) for d in data.val_dataloader()]
        for thread in threads:
            thread.start()
        for thread in tqdm(threads):
            thread.join()'''
        for x, y, label, dt in tqdm(data.train_dataloader()):
            min_eval = minimize(model.minimization, x[0, 6:].data.numpy(),
                                (x[0, :6].data.numpy(), y.data.numpy().flatten(),
                                 label.data.numpy().flatten()))
            torch.save(torch.tensor(np.concatenate((x[0, :6].data.numpy(), min_eval['x'])), dtype=torch.float32),
                       Path(f'{config["load_data"]["save_path"]}/state_vectors/{dt[0, 0]}_{dt[0, 1]}.pt'))
        for x, y, label, dt in tqdm(data.val_dataloader()):
            min_eval = minimize(model.minimization, x[0, 6:].data.numpy(),
                                (x[0, :6].data.numpy(), y.data.numpy().flatten(),
                                 label.data.numpy().flatten()))
            torch.save(torch.tensor(np.concatenate((x[0, :6].data.numpy(), min_eval['x'])), dtype=torch.float32),
                       Path(f'{config["load_data"]["save_path"]}/state_vectors/{dt[0, 0]}_{dt[0, 1]}.pt'))

    if config['measure_model']['training']['save_model']:
        trainer.save_checkpoint(f"{config['measure_model']['training']['weights_path']}/{mdl_name}.ckpt")

