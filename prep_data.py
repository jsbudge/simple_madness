import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.decomposition import TruncatedSVD
from utils.dataframe_utils import prepFrame, addAdvStatstoFrame, addSeasonalStatsToFrame, normalize, glicko_vol, getMatches
from scipy.optimize import basinhopping
from tqdm import tqdm

with open('./run_params.yaml', 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

m_season_data_fnme = Path(f"{config['load_data']['data_path']}/MRegularSeasonDetailedResults.csv")
msdf = pd.read_csv(m_season_data_fnme)
msdf = pd.concat((msdf, pd.read_csv(Path(f"{config['load_data']['data_path']}/WRegularSeasonDetailedResults.csv"))),
                 ignore_index=True)

sdf = prepFrame(msdf)
adf = addAdvStatstoFrame(sdf).fillna(0.)
avdf = adf.groupby(['season', 'tid']).mean()
# countdf = adf.groupby(['season', 'tid']).count()
stddf = adf.groupby(['season', 'tid']).std()

o_cols = np.sort([c for c in adf.columns if c[:2] == 'o_'])
t_cols = np.sort([c for c in adf.columns if c[:2] == 't_'])

infdf = pd.DataFrame(index=sdf.index)

print('Running influence calculations...')
for tidx, tgrp in tqdm(adf.groupby(['tid'])):
    # Use median to reduce outlier influence in calculation
    o_av = adf.loc[adf.index.get_level_values(2) != tidx].groupby(['season', 'tid']).mean()
    # Join each game's o_fg% with that team's average fg%, excluding tidx (to remove bias)
    nchck = tgrp[o_cols].merge(o_av[t_cols], left_on=['season', 'oid'], right_on=['season', 'tid'])
    # Join with the std so we can calculate the # of stds each game was affected by the team
    nstd = tgrp[o_cols].merge(stddf[t_cols], left_on=['season', 'oid'], right_on=['season', 'tid'])
    nstd = nstd.fillna(1.)
    # Get the number of standard deviations away from the mean this team made its opponent go
    inf_data = (nchck[o_cols].values - nchck[t_cols].values) / nstd[t_cols].values

    # These stats are how the team affects its opponents, not how the team is affected
    infdf.loc[tgrp.index, [f'{c}_inf' for c in nstd.columns if c[:2] == 't_']] = inf_data

avdf[infdf.columns] = infdf.groupby(['season', 'tid']).mean().values

# Get resiliency stats - how variable the team is compared to the rest of the world
# Formulated so that a higher resiliency score means you have less variance than the average team
avdf[[f'{c}_res' for c in stddf.columns]] = stddf - adf.groupby(['season', 'tid']).std().groupby(['season']).mean()

# Run elo ratings
print('Running elo ratings...')
m_cond_data_fnme = Path(f"{config['load_data']['data_path']}/MRegularSeasonCompactResults.csv")
mcdf = pd.read_csv(m_cond_data_fnme)
mcdf = pd.concat((mcdf, pd.read_csv(Path(f"{config['load_data']['data_path']}/WRegularSeasonCompactResults.csv"))),
                 ignore_index=True)
mcdf = mcdf.loc[mcdf['Season'] > 2001]

# Don't duplicate for the losers because we want to play each game once
scdf = prepFrame(mcdf, False)
tids = list(set(scdf.index.get_level_values(2)))

# curr_elo = np.ones(max(tids) + 1) * 1500
scdf = scdf.sort_values(by=['season', 'daynum'])
scdf['mov'] = scdf['t_score'] - scdf['o_score']
scdf['t_elo'] = 1500.
scdf['o_elo'] = 1500.


def runElo(x):
    scarray = scdf.reset_index().values
    curr_elo = np.ones(max(tids) + 1) * 1500
    curr_seas = 2001
    mu_reg = x[0]
    margin = x[1]
    k = x[2]
    for n in range(scarray.shape[0]):
        if curr_seas != scarray[n, 1]:
            # Regress everything to the mean
            for val in curr_elo:
                val += ((1 - mu_reg) * val + mu_reg * 1500 - val)
        curr_seas = scarray[n, 1]
        t_elo = curr_elo[int(scarray[n, 2])]
        o_elo = curr_elo[int(scarray[n, 3])]
        scarray[n, -2:] = [t_elo, o_elo]
        hc_adv = x[3] * scarray[n, 7]
        elo_diff = max(t_elo + hc_adv - o_elo, -400)
        elo_shift = 1. / (10. ** (-elo_diff / 400.) + 1.)
        exp_margin = margin + 0.006 * elo_diff
        final_elo_update = k * ((abs(scarray[n, 9]) + 3.) ** 0.8) / exp_margin * (1 - elo_shift) * np.sign(
            scarray[n, 9])
        curr_elo[int(scarray[n, 2])] += final_elo_update
        curr_elo[int(scarray[n, 3])] -= final_elo_update
    return scarray


def optElo(x):
    sc_x = runElo(x)
    return 1 - np.logical_and((sc_x[:, 10] - sc_x[:, 11] > 0), sc_x[:, 9] > 0).sum() / sc_x.shape[0]


elo_params = np.array([0.3896076731384477, 6.51988202753904, 34.11927604457895, 0.17251109126016217])
if config['load_data']['run_elo_opt']:
    print('Optimizing elo...')
    opt_res = basinhopping(optElo, elo_params, minimizer_kwargs=dict(bounds=[(0.1, 10.), (0.1, 100.), (.1, 100), (-10., 10.)]))
    # Add them to the adv frame (make sure the game ids are the same, though)
    sc_out = runElo(opt_res['x'])
else:
    print('Not optimizing elo.')
    sc_out = runElo(elo_params)
scdf = pd.DataFrame(index=scdf.index, columns=scdf.columns, data=sc_out[:, 4:])
joiner_df = sdf.reset_index()[['season', 'tid', 'oid', 'daynum', 'gid']].merge(
    scdf.reset_index()[['season', 'tid', 'oid', 'daynum', 't_elo', 'o_elo']],
    on=['season', 'tid', 'oid', 'daynum'])
joiner_df = joiner_df.set_index(['gid', 'season', 'tid', 'oid'])

adf.loc[joiner_df.index, ['t_elo', 'o_elo']] = joiner_df[['t_elo', 'o_elo']]
adf.loc[np.isnan(adf['t_elo']), ['t_elo', 'o_elo']] = joiner_df[['o_elo', 't_elo']].values

avdf[['t_elo', 'o_elo']] = adf[['t_elo', 'o_elo']].groupby(['season', 'tid']).mean()


# Run Glicko ratings
# Don't duplicate for the losers because we want to play each game once
scdf = prepFrame(mcdf, False)
tids = list(set(scdf.index.get_level_values(2)))

# curr_elo = np.ones(max(tids) + 1) * 1500
scdf = scdf.sort_values(by=['season', 'daynum'])
scdf['mov'] = scdf['t_score'] - scdf['o_score']
scdf['t_glicko'] = 1500.
scdf['o_glicko'] = 1500.
def runGlicko2(x):
    scale = 173.7178
    scarray = scdf.reset_index().values
    curr_gl = np.zeros(max(tids) + 1)
    curr_gl[tids] = 1500.
    curr_vol = np.ones(max(tids) + 1) * .06
    curr_rd = np.ones(max(tids) + 1) * x[1] / scale
    curr_seas = 2001
    tau = x[0]
    for n in range(scarray.shape[0]):
        if curr_seas != scarray[n, 1]:
            # Update volatility to reflect change in season
            for val in curr_rd:
                val = x[1] / scale
            for val in curr_vol:
                val = .06
            for val in curr_gl:
                val += (1500. - val) * x[2]
            # Shift mean back to 1500
            curr_gl -= (curr_gl[tids].mean() - 1500.)
        curr_seas = scarray[n, 1]
        w = scarray[n, 9] > 0.
        # Get scaled r0 values into the results array
        scarray[n, -2:] = [curr_gl[int(scarray[n, 2])], curr_gl[int(scarray[n, 3])]]

        # compute new vol rating
        vol0, v0, E0, g0 = glicko_vol(float(w), curr_vol[int(scarray[n, 2])], curr_rd[int(scarray[n, 3])],
                                                  curr_gl[int(scarray[n, 2])], (curr_gl[int(scarray[n, 3])] - 1500) / scale, tau)
        vol1, v1, E1, g1 = glicko_vol(float(not w), curr_vol[int(scarray[n, 3])], curr_rd[int(scarray[n, 2])],
                              curr_gl[int(scarray[n, 3])], (curr_gl[int(scarray[n, 2])] - 1500) / scale, tau)
        curr_rd[int(scarray[n, 2])] = 1 / (np.sqrt(1 / (curr_rd[int(scarray[n, 2])]**2 + vol0**2)) + 1 / v0)
        curr_rd[int(scarray[n, 3])] = 1 / (np.sqrt(1 / (curr_rd[int(scarray[n, 3])] ** 2 + vol1 ** 2)) + 1 / v1)

        curr_gl[int(scarray[n, 2])] += scale * curr_rd[int(scarray[n, 2])]**2 * (g0 * (w - E0))
        curr_gl[int(scarray[n, 3])] += scale * curr_rd[int(scarray[n, 3])] ** 2 * (g1 * (float(not w) - E1))
    return scarray

def optGlicko(x):
    sc_x = runGlicko2(x)
    return 1 - np.logical_and((sc_x[:, 10] - sc_x[:, 11] > 0), sc_x[:, 9] > 0).sum() / sc_x.shape[0]


gl_params = np.array([0.5, 350., .15])
if config['load_data']['run_elo_opt']:
    print('Optimizing elo...')
    opt_res = basinhopping(optGlicko, gl_params, minimizer_kwargs=dict(bounds=[(0.1, 1.), (100., 1000.)]))
    # Add them to the adv frame (make sure the game ids are the same, though)
    sc_out = runGlicko(opt_res['x'])
else:
    print('Not optimizing elo.')
    sc_out = runGlicko2(gl_params)
scdf = pd.DataFrame(index=scdf.index, columns=scdf.columns, data=sc_out[:, 4:])
joiner_df = sdf.reset_index()[['season', 'tid', 'oid', 'daynum', 'gid']].merge(
    scdf.reset_index()[['season', 'tid', 'oid', 'daynum', 't_glicko', 'o_glicko']],
    on=['season', 'tid', 'oid', 'daynum'])
joiner_df = joiner_df.set_index(['gid', 'season', 'tid', 'oid'])

adf.loc[joiner_df.index, ['t_glicko', 'o_glicko']] = joiner_df[['t_glicko', 'o_glicko']]
adf.loc[np.isnan(adf['t_glicko']), ['t_glicko', 'o_glicko']] = joiner_df[['o_glicko', 't_glicko']].values

avdf[['t_glicko', 'o_glicko']] = adf[['t_glicko', 'o_glicko']].groupby(['season', 'tid']).mean()

# Consolidate massey ordinals in a logical way
ord_fnme = Path(f"{config['load_data']['data_path']}/MMasseyOrdinals.csv")
ord_df = pd.read_csv(ord_fnme)
ord_df = ord_df.pivot_table(index=['Season', 'TeamID', 'RankingDayNum'], columns=['SystemName'])
ord_df.columns = ord_df.columns.droplevel(0)

ord_id = sdf.reset_index()[['season', 'tid', 'oid', 'daynum', 'gid']].rename(
    columns={'season': 'Season', 'tid': 'TeamID', 'daynum': 'RankingDayNum'})
ord_id = ord_id[ord_id['Season'] > 2002]
# adf[['t_rank', 'o_rank']] = 0.

print('Running ranking consolidation...')
if config['load_data']['run_rank_opt']:
    av_acc = dict(zip(ord_df.columns, np.zeros(ord_df.shape[1])))
for t in tqdm(tids):
    try:
        t_ords = ord_df.loc[:, t, :]
    except KeyError:
        continue
    ord_id_local = ord_id.loc[np.logical_or(ord_id['TeamID'] == t, ord_id['oid'] == t)].set_index(['Season', 'RankingDayNum'])
    check = ord_id_local.join(t_ords, how='left', lsuffix='_left', rsuffix='_right')
    check = check.loc[check['TeamID'] == t]
    check['t_rank'] = check.reset_index().ffill().drop(
        columns=['Season', 'RankingDayNum', 'TeamID', 'gid', 'oid']).mean(axis=1, skipna=True).values
    check = check.reset_index().rename(columns={'Season': 'season', 'TeamID': 'tid'}).set_index(['gid', 'season', 'tid', 'oid'])[['t_rank']]
    # check = check.fillna(999.)
    adf.loc[check.index, 't_rank'] = check['t_rank']
adf['t_rank'] = adf['t_rank'].fillna(999.)

ind_0 = adf.reset_index().drop_duplicates(subset=['gid'], keep='first').set_index(['gid', 'season', 'tid', 'oid']).index
ind_1 = adf.reset_index().drop_duplicates(subset=['gid'], keep='last').set_index(['gid', 'season', 'tid', 'oid']).index
adf.loc[ind_0, 'o_rank'] = adf.loc[ind_1, 't_rank'].values
adf.loc[ind_1, 'o_rank'] = adf.loc[ind_0, 't_rank'].values
avdf[['t_rank', 'o_rank']] = adf[['t_rank', 'o_rank']].groupby(['season', 'tid']).mean()

adf[['t_score', 'o_score', 'numot']] = sdf[['t_score', 'o_score', 'numot']]

# Add in seasonal stats to the avdf frame
print('Adding seasonal stats to frame...')
avdf = addSeasonalStatsToFrame(adf, avdf, True)

print('Adding conference stats to frame...')
conf = pd.read_csv(Path(f"{config['load_data']['data_path']}/MTeamConferences.csv"))
conf = pd.concat((conf, pd.read_csv(Path(f"{config['load_data']['data_path']}/WTeamConferences.csv"))),
                 ignore_index=True)
conf = conf.rename(columns={'TeamID': 'tid', 'Season': 'season'})
# cj_df = conf.set_index(['season', 'tid']).join(avdf[['t_elo', 't_rank']])
cj_df = conf.set_index(['season', 'tid']).join(avdf[['t_elo']])
# Set mean
conf = conf.merge(cj_df.groupby(['season', 'ConfAbbrev']).mean(), on=['season', 'ConfAbbrev'])
# conf = conf.rename(columns={'t_elo': 'conf_meanelo', 't_rank': 'conf_meanrank'})
conf = conf.rename(columns={'t_elo': 'conf_meanelo'})
# Set max
conf = conf.merge(cj_df.groupby(['season', 'ConfAbbrev']).max(), on=['season', 'ConfAbbrev'])
# conf = conf.rename(columns={'t_elo': 'conf_maxelo', 't_rank': 'conf_minrank'})
conf = conf.rename(columns={'t_elo': 'conf_maxelo'})
# Set min
conf = conf.merge(cj_df.groupby(['season', 'ConfAbbrev']).min(), on=['season', 'ConfAbbrev'])
# conf = conf.rename(columns={'t_elo': 'conf_minelo', 't_rank': 'conf_maxrank'})
conf = conf.rename(columns={'t_elo': 'conf_minelo'})
avdf = avdf.merge(conf, on=['season', 'tid']).drop(columns=['ConfAbbrev']).set_index(['season', 'tid'])

# Save out the files so we can use them later
if config['load_data']['save_files']:
    adf.to_csv(Path(f'{config["load_data"]["save_path"]}/GameDataAdv.csv'))
    sdf.to_csv(Path(f'{config["load_data"]["save_path"]}/GameDataBasic.csv'))

# Create a dataframe of the tournament results with average data
ncaa_tdf = pd.DataFrame()
for gender in ['M', 'W']:
    ncaa_fnme = f'{config["load_data"]["data_path"]}/{gender}NCAATourneyCompactResults.csv'
    ncaa_tdf = pd.concat([ncaa_tdf, prepFrame(pd.read_csv(ncaa_fnme))])

    # Add in secondary tourney results
    sec_fnme = f'{config["load_data"]["data_path"]}/{gender}SecondaryTourneyCompactResults.csv'
    sc_tdf = pd.read_csv(sec_fnme)
    ncaa_tdf = pd.concat([ncaa_tdf, prepFrame(sc_tdf)])
ncaa_tdf['t_win'] = ncaa_tdf['t_score'] - ncaa_tdf['o_score'] > 0
ncaa_tdf = ncaa_tdf.sort_index().loc[:, 2011:, :, :]

n0, n1 = getMatches(ncaa_tdf, avdf[['t_elo', 't_glicko', 't_rank']])
ncaa_tdf['t_expwin'] = (n0['t_elo'] - n1['t_elo']) > 0

if config['load_data']['save_files']:
    ncaa_tdf.to_csv(Path(f'{config["load_data"]["save_path"]}/TourneyResults.csv'))

# Add coaches - impute for the girls
coaches = pd.read_csv(Path(f"{config['load_data']['data_path']}/MTeamCoaches.csv")).rename(columns={'TeamID': 'tid', 'Season': 'season'})
coaches['span'] = coaches['LastDayNum'] - coaches['FirstDayNum']
coaches['coach_exp'] = coaches[['span', 'CoachName']].groupby(['CoachName']).cumsum()
coaches['coach_exp'] -= coaches['span']  # Make sure this is representative of the start of the season
tourney_wins = coaches.loc[coaches['LastDayNum'] == 154]
win_df = tourney_wins.join(ncaa_tdf.groupby(['season', 'tid']).sum(), how='left', on=['season', 'tid']).fillna(0)
win_df['coach_twins'] = win_df[['CoachName', 't_win']].groupby(['CoachName']).cumsum()
win_df['coach_twins'] -= win_df['t_win']
win_df['coach_winoverexp'] = win_df[['CoachName', 't_expwin']].groupby(['CoachName']).cumsum()
win_df['coach_winoverexp'] -= win_df['t_expwin']
win_df['coach_winoverexp'] = win_df['coach_twins'] - win_df['coach_winoverexp']
coaches = coaches.loc[coaches['LastDayNum'] == 154]
coaches = coaches.reset_index().merge(win_df[['season', 'CoachName', 'coach_twins', 'coach_winoverexp']], on=['season', 'CoachName'])
coaches = coaches.set_index(['season', 'tid']).drop(columns=['index', 'FirstDayNum', 'LastDayNum', 'CoachName', 'span'])
coaches = (coaches - coaches.mean()) / coaches.std()

avdf = normalize(avdf, to_season=True).join(coaches, how='left', on=['season', 'tid']).fillna(0.)
if config['load_data']['save_files']:
    avdf = avdf.groupby(['season', 'tid']).first()
    avdf.to_csv(Path(f'{config["load_data"]["save_path"]}/Averages.csv'))
