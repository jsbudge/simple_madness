import numpy as np
import pandas as pd
from anytree import Node, NodeMixin, RenderTree, AsciiStyle
from anytree.iterators.zigzaggroupiter import ZigZagGroupIter
from dataframe_utils import loadTeamNames, getPossMatches


def scoreBracket(br, truth_br, score_type=None):
    rnds = br.getRounds()
    score = 0
    ngames = 0
    for rnd in range(6):
        for game in rnds[rnd]:
            gm = br[game]
            if gm.children:
                ngames += 1
                if gm.tid == truth_br[game].tid:
                    if score_type is None:
                        # This is the ESPN scoring
                        score += 32 / len(rnds[rnd]) * 10.
                    elif score_type == 'log_loss':
                        score += np.log(gm.win_perc)
                else:
                    if score_type == 'log_loss':
                        score += np.log(1 - gm.win_perc)
    return score


def generateBracket(season: int, use_results: bool = True, datapath: str = './data', gender='M'):
    seeds = pd.read_csv(f'{datapath}/{gender}NCAATourneySeeds.csv')
    seeds = seeds.loc[seeds['Season'] == season]
    slots = pd.read_csv(f'{datapath}/{gender}NCAATourneySlots.csv')
    slots = slots.loc[slots['Season'] == season]
    seedslots = pd.read_csv(f'{datapath}/{gender}NCAATourneySeedRoundSlots.csv').rename(columns={'GameSlot': 'Slot'})
    structure = slots.merge(seedslots[['Slot', 'GameRound']], on='Slot')
    structure = structure.loc[np.logical_not(structure.duplicated(['Season', 'Slot'], keep='first'))].sort_values(
        'GameRound')
    ret_br = Bracket('R6CH')
    for rnd in range(6, -1, -1):
        for g_idx, row in structure.loc[structure['GameRound'] == rnd].iterrows():
            ret_br.addGame(row['Slot'], row['StrongSeed'])
            ret_br.addGame(row['Slot'], row['WeakSeed'])
    # Add in the seeds that don't have games
    rnds = ret_br.getRounds()
    if not use_results:
        for r in rnds:
            for gmid in r:
                gm = ret_br[gmid]
                if not gm.has_children:
                    gm.tid = seeds.loc[seeds['Seed'] == gmid, 'TeamID'].values[0]
                    gm.slot_win = gmid
    else:
        results = pd.read_csv(f'{datapath}/{gender}NCAATourneyCompactResults.csv')
        results = results.loc[results['Season'] == season]
        for n in range(len(rnds) - 1, -1, -1):
            r = rnds[n]
            for gid in r:
                gm = ret_br[gid]
                if gm.has_children:
                    if np.any(results):
                        try:
                            rg = results.loc[np.logical_or(np.logical_and(results['WTeamID'] == gm.children[0].tid,
                                                                          results['LTeamID'] == gm.children[1].tid),
                                np.logical_and(results['LTeamID'] == gm.children[0].tid,
                                               results['WTeamID'] == gm.children[1].tid))]
                            sel_tid = rg['WTeamID'].values[0]
                            slot_win = gm.children[0].slot_win if gm.children[0].tid == sel_tid else gm.children[1].slot_win
                        except IndexError:
                            rg = seedslots.loc[seedslots['Slot'] == gid]
                            if rg['Seed'].values[0] in results['WTeamID'] or rg['Seed'].values[0] in results['LTeamID']:
                                sel_tid = seeds.loc[seeds['Seed'] == rg['Seed'].values[0], 'TeamID'].values[0]
                                slot_win = rg['Seed'].values[0]
                            else:
                                sel_tid = seeds.loc[seeds['Seed'] == rg['Seed'].values[1], 'TeamID'].values[0]
                                slot_win = rg['Seed'].values[1]
                        gm.tid = sel_tid
                        gm.slot_win = slot_win
                    else:
                        # This is a tournament that has yet to be played
                        continue
                else:
                    gm.tid = seeds.loc[seeds['Seed'] == gid, 'TeamID'].values[0]
                    gm.slot_win = gid
    return ret_br


class Game(NodeMixin):
    tid: int
    win_perc: float = 0.
    slot_win: str

    def __init__(self, id, round=None, parent=None, data=None, children=None):
        super().__init__()
        self.parent = parent
        self.data = data
        self.id = id
        self.round = round
        if children:
            self.children = children

    @property
    def has_children(self):
        return True if self.children else False

    def __str__(self):
        return f'{self.id}, Round {self.round}'


class Bracket(object):
    _root: Game
    node_dict: dict

    def __init__(self, root_node_id=None):
        nd = Game(root_node_id, round=6)
        self._root = nd
        if root_node_id is not None:
            self.node_dict = {root_node_id: nd}
        else:
            self.node_dict = {}

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, nd):
        self._root = nd
        self.node_dict['root'] = nd

    def addGame(self, parent_id, nd_id):
        nd = Game(nd_id, parent=self.node_dict[parent_id], round=self.node_dict[parent_id].round - 1)
        self.node_dict[nd.id] = nd

    def addData(self, data: pd.DataFrame):
        rnds = self.getRounds()
        for r in rnds:
            for gm in r:
                if self.node_dict[gm].has_children:
                    self.node_dict[gm].data = (data.loc(axis=0)[:, self.node_dict[gm].children[0].tid],
                                               data.loc(axis=0)[:, self.node_dict[gm].children[1].tid])
                else:
                    self.node_dict[gm].data = data.loc(axis=0)[:, self.node_dict[gm].tid]

    def __getitem__(self, nd_id):
        return self.node_dict[nd_id]

    def __str__(self, datapath = "D:\\\\madness_data\\data"):
        tnames = loadTeamNames(datapath)
        rstr = ''
        for pre, fill, node in RenderTree(self.root, style=AsciiStyle()):
            treestr = pre + node.id + ' ' + tnames[node.tid]
            if hasattr(node, 'win_perc') and node.has_children:
                wp = node.win_perc if node.win_perc > .5 else 1 - node.win_perc
                treestr += f': {100 * wp:.2f}%'
            rstr += '\n' + treestr.ljust(4)
        return rstr

    def __len__(self):
        return self.root.size

    def getRounds(self):
        return [[node.id for node in children] for children in ZigZagGroupIter(self.root)]

    def getSubmission(self, bracket_number: int = 1, save_file_path: str = None):
        sample = pd.read_csv('./data/sample_submission.csv')
        sample['Bracket'] = bracket_number
        rounds = self.getRounds()

        # Apply results frame in backwards order, since rounds starts at the championship
        for rnd in range(len(rounds) - 1, -1, -1):
            for gmid in rounds[rnd]:
                gm = self.node_dict[gmid]
                if gm.has_children:
                    sample.loc[sample['Slot'] == gmid, 'Team'] = gm.slot_win
        if save_file_path is not None:
            sample.to_csv(f'{save_file_path}/submission_{bracket_number}.csv')
        return sample


def buildSubmission(menbrackets: list, womenbrackets: list, save_file_path: str = None):
    master_sub = pd.read_csv('./data/sample_submission.csv')
    rounds = menbrackets[0].getRounds()
    res = pd.DataFrame()

    for idx, (mbr, wbr) in enumerate(zip(menbrackets, womenbrackets)):
        sample = master_sub.copy()
        sample['Bracket'] = idx + 1
        # Apply results frame in backwards order, since rounds starts at the championship
        for rnd in range(len(rounds) - 1, -1, -1):
            for gmid in rounds[rnd]:
                gm = mbr[gmid]
                if gm.has_children:
                    sample.loc[np.logical_and(sample['Slot'] == gmid,
                                              sample['Tournament'] == 'M'), 'Team'] = gm.slot_win
                try:
                    gm = wbr[gmid]
                    if gm.has_children:
                        sample.loc[np.logical_and(sample['Slot'] == gmid,
                                                  sample['Tournament'] == 'W'), 'Team'] = gm.slot_win
                except KeyError:
                    pass
        res = pd.concat([res, sample])
    if save_file_path is not None:
        res.to_csv(f'{save_file_path}/submission.csv')
    return res


def applyResultsToBracket(br: Bracket, res: pd.DataFrame,
                          select_random: bool = False, random_limit: float = 1.) -> Bracket:
    """
    Given a Bracket and a DataFrame of results, applies the results to the Bracket.
    This assumes the bracket has the seed tids already loaded.
    :param random_limit: The percentage difference from .5 at which to not select at random. Only use if select_random == True
    :param select_random: If True, randomly selects a team to move forward using the probabilities given.
    :param br: Bracket instance. Builds a tournament result from the ground up.
    :param res: DataFrame with index (GameID, Season, TID, OID) and column Res
    :return: Reference of br with filled tids.
    """
    rounds = br.getRounds()

    # Apply results frame in backwards order, since rounds starts at the championship
    for rnd in range(len(rounds) - 1, -1, -1):
        for gmid in rounds[rnd]:
            gm = br[gmid]
            if gm.has_children:
                gm_res = res.loc(axis=0)[:, :, gm.children[0].tid, gm.children[1].tid]['Res'].values[0]
                gm.win_perc = gm_res
                if select_random and abs(gm_res - .5) * 2 < random_limit:
                    gm.tid = gm.children[1].tid if np.random.rand() < gm_res else gm.children[0].tid
                    gm.slot_win = gm.children[1].slot_win if np.random.rand() < gm_res else gm.children[0].slot_win
                else:
                    gm.tid = gm.children[1].tid if gm_res > .5 else gm.children[0].tid
                    gm.slot_win = gm.children[1].slot_win if gm_res > .5 else gm.children[0].slot_win
    return br
