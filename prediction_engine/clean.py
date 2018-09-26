import pandas as pd
from functools import reduce

path = '~/ML/Lazarus/Diabetes-Data/csvs/'


def clean():
    ALQ_I = pd.read_csv(path + 'ALQ_I.csv')
    CDQ_I = pd.read_csv(path + 'CDQ_I.csv')
    HSQ_I = pd.read_csv(path + 'HSQ_I.csv')
    DIQ_I = pd.read_csv(path + 'DIQ_I.csv')
    # INQ_I = pd.read_csv(path + 'INQ_I.csv')
    # MCQ_I = pd.read_csv(path + 'MCQ_I.csv')
    PAQ_I = pd.read_csv(path + 'PAQ_I.csv')
    WHQ_I = pd.read_csv(path + 'WHQ_I.csv')

    # Drop uneeded columns
    ALQ_I.drop(['ALQ160', 'ALQ110', 'ALQ141U'], 1, inplace=True)
    HSQ_I.drop(['HSQ500', 'HSQ510', 'HSQ520', 'HSQ571',
                'HSQ580', 'HSQ590', 'HSAQUEX'], 1, inplace=True)

    DIQ_keep = ['SEQN', 'DIQ175A', 'DIQ175B', 'DIQ175C', 'DIQ175D',
                'DIQ175G', 'DIQ175H', 'DIQ175I', 'DIQ175J',
                'DIQ175K', 'DIQ175L', 'DIQ175M', 'DIQ175N', 'DIQ175O']

    # Impute zeros into missing values since they are just negative responses
    DIQ_I = DIQ_I[DIQ_keep]
    DIQ_I.fillna(0, inplace=True)

    # Add diabete scolumn since all respondants have it
    DIQ_I['Diabetes'] = 1

    for c in CDQ_I.columns:
        if c != 'CDQ001' and c != 'SEQN':
            CDQ_I.drop(c, 1, inplace=True)

    PAQ_I_keep = ['SEQN', 'PAQ605']
    PAQ_I = PAQ_I[PAQ_I_keep]
    WHQ_I_keep = ['SEQN', 'WHD020', 'WHQ030']
    WHQ_I = WHQ_I[WHQ_I_keep]

    list_x15_16 = [ALQ_I, CDQ_I, HSQ_I, DIQ_I, PAQ_I, WHQ_I]

    # Merge all data into one list
    merged = reduce(lambda left, right: pd.merge(left, right, on='SEQN',
                    how='outer'), list_x15_16)

    print(merged)
    merged['Diabetes'].fillna(0)

    merged.to_csv(path + 'merged.csv')


clean()
