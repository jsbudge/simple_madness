import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from dataloader import KalmanDataModuleCV
from torch_model import MMClassifier
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
    device = 'cpu'  # f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
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
    expected_lr = max((config['class_model']['lr'] * config['class_model']['scheduler_gamma'] ** (config['class_model']['training']['max_epochs'] *
                                                                config['class_model']['training']['swa_start'])), 1e-9)
    print("======= Training =======")
    trainer = Trainer(logger=logger, max_epochs=config['class_model']['training']['max_epochs'], accelerator=device,
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
    model.eval()
    if config['class_model']['training']['save_model']:
        trainer.save_checkpoint(f"{config['class_model']['training']['weights_path']}/{mdl_name}.ckpt")

    game_sets = {}
    elo_data = pd.read_csv(Path(f'{data.train_dataset.datapath}/GameDataAdv.csv')).groupby(['season', 'tid']).last()[['t_elo']]
    raw_data = pd.read_csv(Path(f'{data.train_dataset.datapath}/kalman_data.csv')).drop(columns=['tid']).rename(columns={'oid': 'tid'}).groupby(['season', 'tid']).first()
    raw_data = raw_data[[col for col in raw_data.columns if 'o_' in col]]
    raw_data['o_elo'] = (elo_data.loc[raw_data.index, 't_elo'] - elo_data['t_elo'].mean()) / elo_data['t_elo'].std()
    raw_data['o_gloc'] = 0

    _, u_data = getPossMatches(raw_data, 2023, use_seed=True, datapath=data.train_dataset.datapath)
    results = pd.DataFrame(index=u_data.index, columns=['txh', 'txh_o', 'txmin', 'txmin_o', 'txmax', 'txmax_o', 'oxh',
                                                        'oxh_t', 'oxmin', 'oxmin_t', 'oxmax', 'oxmax_t'])
    for idx, row in tqdm(u_data.iterrows()):
        # tid results
        x, u, y = data.val_dataset.get_team((idx[1], idx[2]))
        x = torch.cat([x, torch.zeros(1, 2)], dim=0).unsqueeze(0).float()
        u = torch.cat([u, torch.tensor(u_data.loc[idx].values).unsqueeze(0)], dim=0).unsqueeze(0).float()
        txh, txmin, txmax = model.predict(x, u)
        results.loc[idx, ['txh', 'txh_o', 'txmin', 'txmin_o', 'txmax', 'txmax_o']] = \
            [txh[0, -1, 0].data.numpy(), txmin[0, -1, 0].data.numpy(), txmax[0, -1, 0].data.numpy(),
             txh[0, -1, 1].data.numpy(), txmin[0, -1, 1].data.numpy(), txmax[0, -1, 1].data.numpy()]

        # oid results
        x, u, y = data.val_dataset.get_team((idx[1], idx[3]))
        x = torch.cat([x, torch.zeros(1, 2)], dim=0).unsqueeze(0).float()
        u = torch.cat([u, torch.tensor(u_data.loc[idx].values).unsqueeze(0)], dim=0).unsqueeze(0).float()
        oxh, oxmin, oxmax = model.predict(x, u)
        results.loc[idx, ['oxh', 'oxh_t', 'oxmin', 'oxmin_t', 'oxmax', 'oxmax_t']] = \
            [oxh[0, -1, 0].data.numpy(), oxmin[0, -1, 0].data.numpy(), oxmax[0, -1, 0].data.numpy(),
             oxh[0, -1, 1].data.numpy(), oxmin[0, -1, 1].data.numpy(), oxmax[0, -1, 1].data.numpy()]

    from bracket import generateBracket, applyResultsToBracket, scoreBracket
    from scipy.special import erf

    truth_br = generateBracket(2023, True, datapath=data.train_dataset.datapath)
    test_br = generateBracket(2023, True, datapath=data.train_dataset.datapath)
    rfc_results = pd.DataFrame(index=u_data.index, columns=['Res'], data=1 - .5 * (1 + erf((-(results['txh'].values - results['oxh'].values).astype(float)) / (.2 * np.sqrt(2)))))
    res = []
    for _ in range(100):
        test_br = applyResultsToBracket(test_br, rfc_results, select_random=True, random_limit=1.)
        print(f'Final score of {scoreBracket(test_br, truth_br)}')






