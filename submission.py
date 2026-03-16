import pandas as pd
from tqdm import tqdm
from bracket import generateBracket, applyResultsToBracket, scoreBracket

model = 'rfc'
results_frame = pd.read_csv(f'./data/{model}_results.csv').set_index(['gid', 'season', 'tid', 'oid'])
submission_frame = pd.read_csv('./data/SampleSubmissionStage2.csv')

'''for idx, row in tqdm(submission_frame.iterrows()):
    ids = [int(i) for i in row['ID'].split('_')]
    submission_frame.loc[idx, 'Pred'] = results_frame.loc[:, ids[0], ids[1], ids[2]]['Res'].values[0]'''

test_br = generateBracket(2026, False, datapath='./data')
for i in range(10):
    test_br = applyResultsToBracket(test_br, results_frame, select_random=True, random_limit=1.)
    with open(f'./brackets/{model}_{i}.txt', 'w') as f:
        f.write(str(test_br))









