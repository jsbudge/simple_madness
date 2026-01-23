import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import EncoderDataModule, PredictorDataModule
from model import Encoder, Predictor
from sklearn.decomposition import KernelPCA
import numpy as np
from tqdm import tqdm
import itertools
import yaml


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    gpu_num = 0
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)

    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # First, run the encoding to try and reduce the dimension of the data
    fnme = f'MNormalized{method}EncodedData' if prenorm else f'M{method}EncodedData'
    enc_df = pd.read_csv(f'{config["dataloader"]["datapath"]}/{fnme}.csv').set_index(['season', 'tid'])

    data = PredictorDataModule(**config['dataloader'], file=enc_df)
    data.setup()

    # Get the model, experiment, logger set up
    config['model']['init_size'] = data.train_dataset.data_len
    mdl_name = f"{config['model']['name']}_{fnme}"
    model = Predictor(**config['model'])
    logger = loggers.TensorBoardLogger(config['model']['training']['log_dir'], version=0, name=mdl_name)
    expected_lr = max((config['model']['lr'] * config['model']['scheduler_gamma'] ** (config['model']['training']['max_epochs'] *
                                                                config['model']['training']['swa_start'])), 1e-9)
    trainer = Trainer(logger=logger, max_epochs=config['model']['training']['max_epochs'],
                      default_root_dir=config['model']['training']['weights_path'],
                      log_every_n_steps=config['model']['training']['log_epoch'], callbacks=
                      [EarlyStopping(monitor='train_loss', patience=config['model']['training']['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=config['model']['training']['swa_start']),
                       ModelCheckpoint(monitor='train_loss')])

    print("======= Training =======")
    try:
        if config['model']['training']['warm_start']:
            trainer.fit(model, ckpt_path=f"{config['model']['training']['weights_path']}/{mdl_name}.ckpt",
                        datamodule=data)
        else:
            trainer.fit(model, datamodule=data)
    except KeyboardInterrupt:
        if trainer.is_global_zero:
            print('Training interrupted.')
        else:
            print('adios!')
            exit(0)
    if config['model']['training']['save_model']:
        trainer.save_checkpoint(f"{config['model']['training']['weights_path']}/{mdl_name}.ckpt")

    t0, t1, label = next(iter(data.train_dataloader()))

    check = model(t0.to(model.device), t1.to(model.device))
    check = np.concatenate((check.cpu().data.numpy(), label.cpu().data.numpy()), axis=-1)


