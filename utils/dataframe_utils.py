import numpy as np
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from itertools import permutations


def normalize(df: DataFrame, transform=None, to_season: bool = False):
    """
    Normalize a frame to have a mean of zero and standard deviation of one.
    :param transform: SKlearn transformer.
    :param df: Frame of seasonal data.
    :param to_season: if True, calculates norms based on season instead of overall.
    :return: Frame of normalized seasonal data.
    """
    rdf = df.copy()
    if transform is not None:
        if to_season:
            for idx, grp in rdf.groupby(['season']):
                rdf.loc[grp.index] = transform.fit_transform(grp)
        else:
            rdf = pd.DataFrame(index=rdf.index, columns=rdf.columns, data=transform.fit_transform(rdf))
    else:
        if to_season:
            mu_df = df.groupby(['season']).mean()
            std_df = df.groupby(['season']).std()
            for idx, grp in rdf.groupby(['season']):
                rdf.loc[grp.index] = (grp - mu_df.loc[idx]) / std_df.loc[idx]
        else:
            rdf = (rdf - rdf.mean()) / rdf.std()
    return rdf


def getMatches(gids: DataFrame, team_feats: DataFrame, season: int = None, diff: bool = False, sort: bool = True):
    """
    Given a set of games, creates a frame with the chosen stats for predicting purposes.
    :param sort:
    :param gids: frame of games to get matches for. Only uses index. Should have index of [GameID, Season, TID, OID].
    :param team_feats: frame of features to use for matches. Should have index of [Season, TID].
    :param season: Season(s) for which data is wanted. If None, gets it all.
    :param diff: if True, returns differences of features. If False, returns two frames with features.
    :return: Returns either one frame or two, based on diff parameter, of game features.
    """
    if season is not None:
        g = gids.loc(axis=0)[:, season, :, :]
    else:
        g = gids.copy()
    ids = ['gid', 'season', 'tid', 'oid']
    gsc = g.reset_index()[ids]
    g1 = gsc.merge(team_feats, on=['season', 'tid']).set_index(ids)
    g2 = gsc.merge(team_feats, left_on=['season', 'oid'],
                   right_on=['season', 'tid']).set_index(ids)
    if diff:
        return (g1 - g2).sort_index() if sort else (g1 - g2)
    else:
        if sort:
            return g1.sort_index(), g2.sort_index()
        else:
            return g1, g2


def getPossMatches(team_feats, season, diff=False, use_seed=True, datapath=None, gender='M'):
    """
    Gets the possible matches in any season of all the teams in the tournament.
    :param use_seed: If True, only gets the tournament participants. Otherwise uses every team that's played that season
    :param team_feats: Frame of features wanted for hypothetical matchups. Should have index of [Season, TID]
    :param season: Season for which data is wanted.
    :param diff: if True, returns differences of features. If False, returns two frames with features.
    :return: Returns either one frame or two, based on diff parameter, of game features.
    """
    if use_seed:
        sd = pd.read_csv(f'{datapath}/{gender}NCAATourneySeeds.csv')
    else:
        sd = pd.read_csv(f'{datapath}/{gender}RegularSeasonCompactResults.csv')
    sd = sd.loc[sd['Season'] == season]['TeamID'].values
    teams = list(set(sd))
    matches = [[x, y] for (x, y) in permutations(teams, 2)]
    poss_games = pd.DataFrame(data=matches, columns=['tid', 'oid'])
    poss_games['season'] = season
    poss_games['gid'] = np.arange(poss_games.shape[0])
    gsc = poss_games.set_index(['gid', 'season'])
    g1 = gsc.merge(team_feats, left_on=['tid', 'season'],
                   right_on=['tid', 'season'], right_index=True).sort_index()
    g1 = g1.reset_index().set_index(['gid', 'season', 'tid', 'oid'])
    g2 = gsc.merge(team_feats, left_on=['oid', 'season'],
                   right_on=['tid', 'season'],
                   right_index=True).sort_index()
    g2 = g2.reset_index().set_index(['gid', 'season', 'tid', 'oid'])
    if diff:
        return g1 - g2
    else:
        return g1, g2


def prepFrame(df: DataFrame, full_frame: bool = True) -> DataFrame:
    df = df.rename(columns={'WLoc': 'gloc'})
    df['gloc'] = df['gloc'].map({'A': -1, 'N': 0, 'H': 1})

    # take the frame and convert it into team/opp format
    iddf = df[[c for c in df.columns if c[0] != 'W' and c[0] != 'L']]
    iddf = iddf.rename(columns=dict([(c, c.lower()) for c in iddf.columns]))
    iddf['gid'] = iddf.index.values
    wdf = df[[c for c in df.columns if c[0] == 'W']]
    ldf = df[[c for c in df.columns if c[0] == 'L']]

    # First the winners get to be the team
    tdf = wdf.rename(columns=dict([(c, f't_{c[1:].lower()}') for c in wdf.columns]))
    odf = ldf.rename(columns=dict([(c, f'o_{c[1:].lower()}') for c in ldf.columns]))
    fdf = tdf.merge(odf, left_index=True, right_index=True).merge(iddf, left_index=True, right_index=True)

    # then the losers get to do it
    if full_frame:
        tdf = ldf.rename(columns=dict([(c, f't_{c[1:].lower()}') for c in ldf.columns]))
        odf = wdf.rename(columns=dict([(c, f'o_{c[1:].lower()}') for c in wdf.columns]))
        full_ldf = tdf.merge(odf, left_index=True, right_index=True).merge(iddf, left_index=True, right_index=True)
        full_ldf['gloc'] = -full_ldf['gloc']
        fdf = pd.concat([fdf, full_ldf])

    # Final clean up of column names and ID setting
    fdf = fdf.rename(columns={'t_teamid': 'tid', 'o_teamid': 'oid'})
    fdf = fdf.set_index(['gid', 'season', 'tid', 'oid']).sort_index()

    return fdf


def addAdvStatstoFrame(df: DataFrame, add_to_frame: bool = False) -> DataFrame:
    """
    Given a properly indexed dataframe, adds some advanced stats and returns it
    :param df: DataFrame with index keys of [gid, season, tid, oid] and the necessary columns
    :param add_to_frame: if True, adds it to df. Otherwise returns a new DataFrame with the columns.
    :return: DataFrame with new stats. Same index.
    """
    out_df = pd.DataFrame(index=df.index)
    # First order derived stats
    out_df['t_fg%'] = df['t_fgm'] / df['t_fga']
    out_df['t_fg2%'] = (df['t_fgm'] - df['t_fgm3']) / (df['t_fga'] - df['t_fga3'])
    out_df['t_fg3%'] = df['t_fgm3'] / df['t_fga3']
    out_df['t_efg%'] = (df['t_fgm'] + .5 * df['t_fgm3']) / df['t_fga']
    out_df['t_ts%'] = df['t_score'] / (2 * (df['t_fga'] + .44 * df['t_fta']))
    out_df['t_econ'] = df['t_ast'] + df['t_stl'] - df['t_to']
    out_df['t_poss'] = .96 * (df['t_fga'] - df['t_or'] + df['t_to'] + .44 * df['t_fta'])
    out_df['t_offrat'] = df['t_score'] * 100 / out_df['t_poss']
    out_df['t_r%'] = (df['t_or'] + df['t_dr']) / (df['t_or'] + df['t_dr'] + df['o_or'] + df['o_dr'])
    out_df['t_ast%'] = df['t_ast'] / df['t_fgm']
    out_df['t_3two%'] = df['t_fga3'] / df['t_fga']
    out_df['t_ft/a'] = df['t_fta'] / (df['t_fga'] * 2 + df['t_fga3'])
    out_df['t_ft%'] = df['t_ftm'] / df['t_fta']
    out_df['t_to%'] = df['t_to'] / out_df['t_poss']
    out_df['t_extraposs'] = df['t_or'] + df['t_stl'] + df['o_pf']
    out_df['t_mov'] = df['t_score'] - df['o_score']
    out_df['t_rmar'] = (df['t_or'] + df['t_dr']) - (df['o_or'] + df['o_dr'])
    out_df['t_tomar'] = df['t_to'] - df['o_to']
    out_df['t_a/to'] = df['t_ast'] - df['t_to']
    out_df['t_blkperp'] = df['t_blk'] / out_df['t_poss']
    out_df['t_domf'] = (df['t_or'] - df['o_or']) * 1.2 + (df['t_dr'] - df['o_dr']) * 1.07 + \
                       (df['o_to'] - df['t_to']) * 1.5
    out_df['t_score%'] = (df['t_fgm'] + df['t_fgm3'] * .5 + df['t_ftm'] * .3 + df['t_pf'] * .5) / (
            df['t_fga'] + df['t_fta'] * .3 + df['t_to'])
    out_df['t_pie'] = df['t_score'] + df['t_fgm'] + df['t_ftm'] - df['t_fga'] - df['t_fta'] + \
                      df['t_dr'] + (.5 * df['t_or']) + df['t_ast'] + df['t_stl'] + .5 * df['t_blk'] - \
                      df['t_pf'] - df['t_to']
    out_df['o_pie'] = df['o_score'] + df['o_fgm'] + df['o_ftm'] - df['o_fga'] - df['o_fta'] + \
                      df['o_dr'] + (.5 * df['o_or']) + df['o_ast'] + df['o_stl'] + .5 * df['o_blk'] - \
                      df['o_pf'] - df['o_to']
    out_df['t_or%'] = df['t_or'] / (df['t_fga'] - df['t_fgm'])

    out_df['o_fg%'] = df['o_fgm'] / df['o_fga']
    out_df['o_fg2%'] = (df['o_fgm'] - df['o_fgm3']) / (df['o_fga'] - df['o_fga3'])
    out_df['o_fg3%'] = df['o_fgm3'] / df['o_fga3']
    out_df['o_efg%'] = (df['o_fgm'] + .5 * df['o_fgm3']) / df['o_fga']
    out_df['o_ts%'] = df['o_score'] / (2 * (df['o_fga'] + .44 * df['o_fta']))
    out_df['o_econ'] = df['o_ast'] + df['o_stl'] - df['o_to']
    out_df['o_poss'] = .96 * (df['o_fga'] - df['o_or'] + df['o_to'] + .44 * df['o_fta'])
    out_df['o_offrat'] = df['o_score'] * 100 / out_df['o_poss']
    out_df['o_r%'] = 1 - out_df['t_r%']
    out_df['o_ast%'] = df['o_ast'] / df['o_fgm']
    out_df['o_3two%'] = df['o_fga3'] / df['o_fga']
    out_df['o_ft/a'] = df['o_fta'] / (df['o_fga'] * 2 + df['o_fga3'])
    out_df['o_ft%'] = df['o_ftm'] / df['o_fta']
    out_df['o_to%'] = df['o_to'] / out_df['o_poss']
    out_df['o_extraposs'] = df['o_or'] + df['o_stl'] + df['t_pf']
    out_df['o_mov'] = df['o_score'] - df['t_score']
    out_df['o_rmar'] = (df['o_or'] + df['o_dr']) - (df['t_or'] + df['t_dr'])
    out_df['o_tomar'] = df['o_to'] - df['t_to']
    out_df['o_a/to'] = df['o_ast'] - df['o_to']
    out_df['o_blkperp'] = df['o_blk'] / out_df['o_poss']
    out_df['o_domf'] = (df['o_or'] - df['t_or']) * 1.2 + (df['o_dr'] - df['t_dr']) * 1.07 + \
                       (df['t_to'] - df['o_to']) * 1.5
    out_df['o_score%'] = (df['o_fgm'] + df['o_fgm3'] * .5 + df['o_ftm'] * .3 + df['o_pf'] * .5) / (
            df['o_fga'] + df['o_fta'] * .3 + df['o_to'])
    out_df['o_or%'] = df['o_or'] / (df['o_fga'] - df['o_fgm'])

    # Second order derived stats
    out_df['t_defrat'] = out_df['o_offrat']
    out_df['o_defrat'] = out_df['t_offrat']
    out_df['t_gamescore'] = 40 * out_df['t_efg%'] + 20 * out_df['t_r%'] + \
                            15 * out_df['t_ft/a'] + 25 - 25 * out_df['t_to%']
    out_df['o_gamescore'] = 40 * out_df['o_efg%'] + 20 * out_df['o_r%'] + \
                            15 * out_df['o_ft/a'] + 25 - 25 * out_df['o_to%']
    out_df['t_prodposs'] = out_df['t_poss'] - df['t_to'] - (df['t_fga'] - df['t_fgm'] + .44 * df['t_ftm'])
    out_df['o_prodposs'] = out_df['o_poss'] - df['o_to'] - (df['o_fga'] - df['o_fgm'] + .44 * df['o_ftm'])
    out_df['t_prodposs%'] = out_df['t_prodposs'] / out_df['t_poss']
    out_df['o_prodposs%'] = out_df['o_prodposs'] / out_df['o_poss']
    out_df['t_gamecontrol'] = out_df['t_poss'] / (out_df['o_poss'] + out_df['t_poss'])
    out_df['o_gamecontrol'] = 1 - out_df['t_gamecontrol']
    out_df['t_sos'] = df['t_score'] / out_df['t_poss'] - df['o_score'] / out_df['o_poss']
    out_df['o_sos'] = df['o_score'] / out_df['o_poss'] - df['t_score'] / out_df['t_poss']
    # out_df['t_tie'] = out_df['t_pie'] / (out_df['t_pie'] + out_df['o_pie'])
    # out_df['o_tie'] = out_df['o_pie'] / (out_df['t_pie'] + out_df['o_pie'])

    # third order derived stats
    eff_model = np.polyfit(out_df['t_efg%'], out_df['t_offrat'], 1)
    out_df['t_offeff'] = out_df['t_offrat'] - np.poly1d(eff_model)(out_df['t_efg%'])
    out_df['o_offeff'] = out_df['o_offrat'] - np.poly1d(eff_model)(out_df['o_efg%'])

    return df.merge(out_df, right_index=True, left_index=True) if add_to_frame else out_df


def addSeasonalStatsToFrame(sdf: DataFrame, df: DataFrame, add_to_frame: bool = True, pyth_exp: float = 13.91):
    """
    Adds some end-of-season stats to a dataframe.
    :param sdf: Frame with seasonal stats. Needs the stats listed in the function or it will error.
    :param df: Frame with team stats, id of [season, tid]
    :param add_to_frame: if True, adds the stats to df. Otherwise returns a new frame.
    :param pyth_exp: Float with the pythagorean win exponential. Generally accepted to be 13.91, but can be changed if desired.
    :return: Either df with the new columns or the new dataframe.
    """
    out_df = pd.DataFrame(index=df.index)
    dfapp = sdf.groupby(['season', 'tid'])
    out_df['t_closegame%'] = dfapp.apply(lambda x: sum(np.logical_or(abs(x['t_mov']) < 4, x['numot'] > 0)) / x.shape[0])
    out_df['t_win%'] = dfapp.apply(lambda x: sum(x['t_mov'] > 0) / x.shape[0])
    out_df['t_pythwin%'] = dfapp.apply(
        lambda grp: sum(grp['t_score'] ** pyth_exp) / sum(grp['t_score'] ** pyth_exp + grp['o_score'] ** pyth_exp))
    out_df['t_owin%'] = sdf.reset_index().merge(out_df['t_win%'].reset_index(),
                                                left_on=['season', 'oid'],
                                                right_on=['season', 'tid']).groupby(['season',
                                                                                     'tid_x']).mean()['t_win%'].values
    # Opponents' opponent win percentage calculations, for RPI
    oo_win = sdf.reset_index().merge(out_df['t_owin%'], left_on=['season', 'oid'], right_on=['season', 'tid']).groupby(
        ['season', 'tid']).mean()['t_owin%']
    out_df['t_rpi'] = .25 * out_df['t_win%'] + .5 * out_df['t_owin%'] + .25 * oo_win
    out_df['t_opythwin%'] = sdf.reset_index().merge(out_df['t_pythwin%'].reset_index(),
                                                    left_on=['season', 'oid'],
                                                    right_on=['season', 'tid']).groupby(['season',
                                                                                         'tid_x']).mean()[
        't_pythwin%'].values
    # Opponents' opponent win percentage calculations, for RPI
    oo_win = \
        sdf.reset_index().merge(out_df['t_opythwin%'], left_on=['season', 'oid'], right_on=['season', 'tid']).groupby(
            ['season', 'tid']).mean()['t_opythwin%']
    out_df['t_pythrpi'] = .25 * out_df['t_pythwin%'] + .5 * out_df['t_opythwin%'] + .25 * oo_win
    out_df['t_expwin%'] = dfapp.apply(lambda x: sum(x['t_elo'] > x['o_elo']) / x.shape[0])
    out_df['t_luck'] = out_df['t_win%'] - out_df['t_pythwin%']

    return df.merge(out_df, right_index=True, left_index=True) if add_to_frame else out_df


def loadTeamNames(datapath: str = './data', gender='M'):
    """
    Create a dict of team names and ids so we know who's behind the numbers.
    :return: Dict of teamIDs and names.
    """
    df = pd.concat(
        [pd.read_csv(f'{datapath}/{gender}Teams.csv'), pd.read_csv(f'{datapath}/{gender}Teams.csv')]).sort_index()
    ret = {}
    for idx, row in df.iterrows():
        ret[row['TeamID']] = row['TeamName']
        ret[row['TeamName']] = row['TeamID']
    return ret


def gauss_weight(df, col, sigma=None):
    sigma = df[f't_{col}'].std() if sigma is None else sigma
    df_weight = np.exp(-(df[f't_{col}'] - df[f'o_{col}']) ** 2 / (2 * sigma ** 2))
    return df.mul(df_weight, axis=0).groupby(['season', 'tid']).sum().mul(
        1 / df_weight.groupby(['season', 'tid']).sum(), axis=0)


def col_weight(df, col):
    df_weight = df[col]
    return df.mul(df_weight, axis=0).groupby(['season', 'tid']).sum().mul(
        1 / df_weight.groupby(['season', 'tid']).sum(), axis=0)


def date_weight(df, dates):
    df_weight = dates['daynum']
    return df.mul(df_weight, axis=0).groupby(['season', 'tid']).sum().mul(
        1 / df_weight.groupby(['season', 'tid']).sum(), axis=0)
