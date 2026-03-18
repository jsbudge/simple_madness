import numpy as np
from pathlib import Path
import pandas as pd
import yaml
from bracket import generateBracket, applyResultsToBracket, scoreBracket
from utils.dataframe_utils import prepFrame, getMatches, getPossMatches, normalize
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
# from sklearn.feature_selection import RFECV
from utils.sklearn_utils import get_legendre_pipeline, SeasonalSplit
from sklearn.decomposition import TruncatedSVD

with open('./run_params.yaml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

datapath = config['dataloader']['datapath']
ncomp = 10
features = pd.read_csv(Path(f'{datapath}/Averages.csv')).set_index(['season', 'tid'])
svd = TruncatedSVD(n_components=ncomp)
features = normalize(pd.DataFrame(data=svd.fit_transform(features), index=features.index, columns=[f't_{col}' for col in range(ncomp)]))

tids = pd.read_csv(Path(f'{datapath}/MNCAATourneyCompactResults.csv'))
tids = prepFrame(tids)
gids = prepFrame(pd.read_csv(Path(f'{datapath}/MRegularSeasonCompactResults.csv')))
stids = prepFrame(pd.read_csv(Path(f'{datapath}/MSecondaryTourneyCompactResults.csv')))
method_results = pd.DataFrame(index=tids.index, columns=['Truth'],
                              data=tids['t_score'] - tids['o_score'] > 0).astype(np.float32)

# print('Getting features for feature elimination.')
d0, d1 = getMatches(tids, features)
X_feat = d0 - d1
y_feat = method_results.loc[d0.index]

kernels = [('matern', kernels.Matern()), ('rbf', kernels.RBF()), ('rat_quad', kernels.RationalQuadratic(length_scale=10.))]

results = pd.DataFrame(columns=[k[0] for k in kernels])

for degree in range(2, 7):
    print(f'Legendre feature degree: {degree}.')
    leg_feats = get_legendre_pipeline(degree=degree, include_bias=False)
    nfeats = pd.DataFrame(index=features.index, data=leg_feats.fit_transform(features))
    X_nfeat = getMatches(tids, nfeats, diff=True)

    s_cv = SeasonalSplit()
    for col_name, ker in kernels:
        gpc = GaussianProcessClassifier(kernel=ker)
        print(f'Kernel: {col_name}.')
        total_res = []
        for test, train in s_cv.split(X_nfeat, y_feat.index):
            gpc.fit(X_nfeat.loc[train], np.ravel(y_feat.loc[train]))
            season = test.get_level_values(1)[0]
            print(f'Running seasonal split for {season}.')
            truth_br = generateBracket(season, True, datapath=datapath)
            test_br = generateBracket(season, True, datapath=datapath)
            ps = getPossMatches(nfeats.loc[nfeats.index.get_level_values(0) == season], season, diff=True, datapath=datapath)
            rfc_results = pd.DataFrame(index=ps.index, columns=['Res', 'res1'], data=gpc.predict_proba(ps))
            res = []
            for _ in range(100):
                test_br = applyResultsToBracket(test_br, rfc_results, select_random=True, random_limit=.6)
                res.append(scoreBracket(test_br, truth_br))
            print(f'Average score of {np.mean(res)} with STD of {np.std(res)}.')
            total_res.append(np.mean(res))
        results.loc[degree, col_name] = np.mean(total_res)

best_performer = np.where(results == np.max(results))
best_degree = results.index[best_performer[0][0]]
best_kernel = kernels[best_performer[1][0]]

leg_feats = get_legendre_pipeline(degree=best_degree, include_bias=False)
nfeats = pd.DataFrame(index=features.index, data=leg_feats.fit_transform(features))
X_nfeat = getMatches(tids, nfeats, diff=True)

gpc = GaussianProcessClassifier(kernel=best_kernel[1])

gpc.fit(X_nfeat, np.ravel(y_feat))

from utils.dataframe_utils import getPossMatches

ps = getPossMatches(nfeats, 2026, True, False, datapath)
final_results = pd.DataFrame(index=ps.index, columns=['Res', 'res1'], data=gpc.predict_proba(ps))

ps = getPossMatches(nfeats, 2026, True, False, datapath, 'W')

final_results = pd.concat([final_results, pd.DataFrame(index=ps.index, columns=['Res', 'res1'], data=gpc.predict_proba(ps))])
final_results.to_csv(f'{datapath}/gpc_results.csv')







