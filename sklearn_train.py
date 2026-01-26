import numpy as np
from pathlib import Path
import pandas as pd
import yaml
from bracket import generateBracket, applyResultsToBracket, scoreBracket
from utils.dataframe_utils import prepFrame, getMatches, getPossMatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from utils.sklearn_utils import LegendreScalarPolynomialFeatures, get_legendre_pipeline, SeasonalSplit

with open('./run_params.yaml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

datapath = config['dataloader']['datapath']
features = pd.read_csv(Path(f'{datapath}/NormalizedEloAverages.csv')).set_index(['tid', 'season'])
tids = pd.read_csv(Path(f'{datapath}/MNCAATourneyCompactResults.csv'))
tids = prepFrame(tids)
gids = prepFrame(pd.read_csv(Path(f'{datapath}/MRegularSeasonCompactResults.csv')))
stids = prepFrame(pd.read_csv(Path(f'{datapath}/MSecondaryTourneyCompactResults.csv')))
method_results = pd.DataFrame(index=tids.index, columns=['Truth'],
                              data=tids['t_score'] - tids['o_score'] > 0).astype(np.float32)

print('Getting features for feature elimination.')
d0, d1 = getMatches(tids, features)
X_feat = d0 - d1
y_feat = method_results.loc[d0.index]

# First, find features that work best to get tournament results
print('Fitting CV for feature elimination.')
rfc = RandomForestClassifier(n_estimators=128)
rfe = RFECV(estimator=rfc, verbose=1, cv=4)
rfe.fit(X_feat, np.ravel(y_feat))

results = pd.DataFrame(columns=[100, 128, 200, 255, 301, 512, 1024])

for degree in range(2, 9):
    print(f'Legendre feature degree: {degree}.')
    leg_feats = get_legendre_pipeline(degree=degree, include_bias=False)
    nfeats = pd.DataFrame(index=features.index, data=leg_feats.fit_transform(features[rfe.feature_names_in_[rfe.ranking_ <= 3]]))
    dn0, dn1 = getMatches(tids, nfeats)
    X_nfeat = dn0 - dn1

    s_cv = SeasonalSplit()
    for n_est in results.columns:
        rfc = RandomForestClassifier(n_estimators=n_est)
        print(f'Estimators: {n_est}.')
        total_res = []
        for test, train in s_cv.split(X_nfeat, y_feat.index):
            rfc.fit(X_nfeat.loc[train], np.ravel(y_feat.loc[train]))
            season = test.get_level_values(1)[0]
            print(f'Running seasonal split for {season}.')
            truth_br = generateBracket(season, True, datapath=datapath)
            test_br = generateBracket(season, True, datapath=datapath)
            ps = getPossMatches(nfeats.loc[nfeats.index.get_level_values(1) == season], season, diff=True, datapath=datapath)
            rfc_results = pd.DataFrame(index=ps.index, columns=['Res', 'res1'], data=rfc.predict_proba(ps))
            res = []
            for _ in range(500):
                test_br = applyResultsToBracket(test_br, rfc_results, select_random=True, random_limit=.1)
                res.append(scoreBracket(test_br, truth_br))
            print(f'Average score of {np.mean(res)} with STD of {np.std(res)}.')
            total_res.append(np.mean(res))
        results.loc[degree, n_est] = np.mean(total_res)







