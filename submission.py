import pandas as pd
from utils.dataframe_utils import getPossMatches
import numpy as np

results_frame = pd.read_csv('results.csv')
submission_frame = pd.read_csv('./data/SampleSubmissionStage2.csv')

for idx, row in submission_frame.iterrows():
    ids = row['ID'].split('_')
    submission_frame.loc[idx, 'Pred'] = results_frame.loc[:, ids[0], ids[1], ids[2]]['Res'].values[0]







