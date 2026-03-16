import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import GameDataModuleCV
from torch_model import DKF
import numpy as np
from utils.dataframe_utils import prepFrame, getMatches, date_weight, normalize, getPossMatches
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
    # config['model']['init_size'] = data.train_dataset.data_len
    mdl_name = f"{config['model']['name']}"
    model = DKF(**config['model'])
    logger = loggers.TensorBoardLogger(config['model']['training']['log_dir'], version=0, name=mdl_name)
    expected_lr = max((config['model']['lr'] * config['model']['scheduler_gamma'] ** (config['model']['training']['max_epochs'] *
                                                                config['model']['training']['swa_start'])), 1e-9)
    print("======= Training =======")
    trainer = Trainer(logger=logger, max_epochs=config['model']['training']['max_epochs'],
                      default_root_dir=config['model']['training']['weights_path'], num_sanity_val_steps=0,
                      log_every_n_steps=config['model']['training']['log_epoch'], callbacks=
                      [EarlyStopping(monitor='train_total_loss', patience=config['model']['training']['patience'],
                                     check_finite=True),
                       StochasticWeightAveraging(swa_lrs=expected_lr,
                                                 swa_epoch_start=config['model']['training']['swa_start']),
                       ModelCheckpoint(monitor='train_total_loss')])
    try:
        trainer.fit(model, datamodule=data)
    except KeyboardInterrupt:
        print('Training interrupted by user.')
    model.eval()
    if config['model']['training']['save_model']:
        trainer.save_checkpoint(f"{config['model']['training']['weights_path']}/{mdl_name}.ckpt")

    game_sets = {}
    elo_data = pd.read_csv(Path(f'{data.train_dataset.datapath}/GameDataAdv.csv')).groupby(['season', 'tid']).last()[['t_elo']]
    raw_data = pd.read_csv(Path(f'{data.train_dataset.datapath}/kalman_data.csv')).drop(columns=['tid']).rename(columns={'oid': 'tid'}).groupby(['season', 'tid']).first()
    raw_data = raw_data[[col for col in raw_data.columns if 'o_' in col]]
    raw_data['o_elo'] = (elo_data.loc[raw_data.index, 't_elo'] - elo_data['t_elo'].mean()) / elo_data['t_elo'].std()
    raw_data['o_gloc'] = 0

    season = 2026
    final = pd.DataFrame()
    for gender in ['M', 'W']:
        for s in range(2011, 2027):
            _, u_data = getPossMatches(raw_data, s, use_seed=False, datapath=data.train_dataset.datapath, gender=gender)
            tids = list(set(u_data.index.get_level_values(2)))
            results = pd.DataFrame(columns=['season', 'tid'] + [f't_{i}' for i in range(50)])
            results['season'] = np.ones(len(tids)).astype(int) * s
            results['tid'] = tids
            results = results.set_index(['season', 'tid'])
            for tid in tqdm(tids):
                # tid results
                try:
                    x, u, y = data.train_dataset.get_team((s, tid))
                except Exception as e:
                    try:
                        x, u, y = data.val_dataset.get_team((s, tid))
                    except Exception as e:
                        continue
                txh, txmin, txmax, z_mu, z_t = model.predict(x.unsqueeze(0), u.unsqueeze(0))
                results.loc[(s, tid), results.columns] = z_mu[-1].data.cpu().numpy()
            final = pd.concat([final, results])

    final.to_csv(f'{data.train_dataset.datapath}/dkf_data.csv')






