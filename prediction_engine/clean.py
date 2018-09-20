import pandas as pd
from functools import reduce

path = '~/ML/Lazarus/Diabetes-Data/csvs/'


def clean():
    ALQ_I = pd.read_csv(path + 'ALQ_I.csv')
    CDQ_I = pd.read_csv(path + 'CDQ_I.csv')
    HSQ_I = pd.read_csv(path + 'HSQ_I.csv')
    DIQ_I = pd.read_csv(path + 'DIQ_I.csv')
    INQ_I = pd.read_csv(path + 'INQ_I.csv')
    MCQ_I = pd.read_csv(path + 'MCQ_I.csv')
    PAQ_I = pd.read_csv(path + 'PAQ_I.csv')
    WHD_I = pd.read_csv(path + 'WHD_I.csv')

    # Drop uneeded columns
    ALQ_I.drop(['ALQ160', 'ALQ110', 'ALQ141U'])
    HSQ_I.drop(['HSQ500', 'HSQ510', 'HSQ520', 'HSQ571',
                'HSQ580', 'HSQ590', 'QUEX'])

    DIQ_keep = ['DIQ175A', 'DIQ175B', 'DIQ175C', 'DIQ175D',
                'DIQ175G', 'DIQ175H', 'DIQ175I', 'DIQ175J',
                'DIQ175K', 'DIQ175L', 'DIQ175M', 'DIQ175N', 'DIQ175O']

    # Impute zeros into missing values since they are just negative responses
    DIQ_I = DIQ_I[DIQ_keep]
    DIQ_I.fillna(0)

    # Add diabete scolumn since all respondants have it
    DIQ_I['Diabetes'] = [x for x in range(DIQ_I.shape[1])]

    for c in CDQ_I.columns:
        if c != 'CDQ001':
            CDQ_I.drop(c)

    list_x15_16 = list(ALQ_I, CDQ_I, HSQ_I, DIQ_I, INQ_I,
                       MCQ_I, PAQ_I, WHD_I)

    # Merge all data into one list
    merged = reduce(lambda left, right: pd.merge(left, right, on='SEQ',
                    how='outer'), list_x15_16)

    merged.to_csv(path + 'merged.csv')
