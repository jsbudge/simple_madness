import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import KalmanDataModuleCV
from torch_model import MMClassifier
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm


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

    data = KalmanDataModuleCV(**config['class_model']['dataloader'])
    data.setup()

    # Get the model, experiment, logger set up for measurement function and embedding training
    # config['class_model']['init_size'] = data.train_dataset.data_len
    mdl_name = f"{config['class_model']['name']}"
    model = MMClassifier(**config['class_model'])
    logger = loggers.TensorBoardLogger(config['class_model']['training']['log_dir'], version=0, name=mdl_name)
    expected_lr = 1e-6
    print("======= Training =======")
    trainer = Trainer(logger=logger, max_epochs=config['class_model']['training']['max_epochs'],
                      default_root_dir=config['class_model']['training']['weights_path'], num_sanity_val_steps=0,
                      log_every_n_steps=config['class_model']['training']['log_epoch'], callbacks=
                      [EarlyStopping(monitor='train_loss', patience=config['class_model']['training']['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=config['class_model']['training']['swa_start']),
                       ModelCheckpoint(monitor='train_loss')])
    try:
        trainer.fit(model, datamodule=data)
    except KeyboardInterrupt:
        print('Training interrupted by user.')

    config['class_model']['dataloader']['is_tourney'] = True
    tdata = KalmanDataModuleCV(**config['class_model']['dataloader'])
    tdata.setup()
    model.lr = 1e-7
    model.max_iters = 40000

    trainer = Trainer(logger=logger, max_epochs=100,
                      default_root_dir=config['class_model']['training']['weights_path'], num_sanity_val_steps=0,
                      log_every_n_steps=config['class_model']['training']['log_epoch'], callbacks=
                      [EarlyStopping(monitor='train_loss', patience=config['class_model']['training']['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=1e-6,
                                                 swa_epoch_start=config['class_model']['training']['swa_start']),
                       ModelCheckpoint(monitor='train_loss')])

    print("======= Finetuning =======")
    try:
        trainer.fit(model, datamodule=tdata)
    except KeyboardInterrupt:
        print('Training interrupted by user.')

    datapath = './data'

    avs = pd.read_csv(Path(f'{datapath}/Averages.csv')).set_index(['season', 'tid'])

    from utils.dataframe_utils import getPossMatches

    ps = getPossMatches(avs, 2026, False, False, datapath)
    psw = getPossMatches(avs, 2026, False, False, datapath, 'W')
    model.eval()
    dt_idx = pd.concat([ps[0].reset_index()[['gid', 'season', 'tid', 'oid']],
                        psw[0].reset_index()[['gid', 'season', 'tid', 'oid']]])
    dt_idx['gid'] = np.arange(dt_idx.shape[0])
    results = pd.DataFrame(index=dt_idx[['gid', 'season', 'tid', 'oid']].set_index(['gid', 'season', 'tid', 'oid']).index, columns=['Res'])
    dt_idx = dt_idx.values
    for gm in tqdm(dt_idx):
        results.loc[gm, 'Res'] = 1 - model(tdata.loadTeam(gm[1], gm[2]).unsqueeze(0).to(model.device),
                                       tdata.loadTeam(gm[1], gm[3]).unsqueeze(0).to(model.device),
              torch.zeros((1, 1), dtype=torch.float32, device=model.device)).item()
    results.to_csv(Path(f'{datapath}/mlp_results.csv'))










