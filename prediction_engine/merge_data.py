import pandas as pd
from functools import reduce

path = '~/ML/Lazarus/Diabetes-Data'

DIQ_I = pd.read_csv(path + 'csvs/DIQ_I.csv')
ALQ_I = pd.read_csv(path + 'csvs/ALQ_I.csv')
BPQ_I = pd.read_csv(path + 'csvs/BPQ_I.csv')
CDQ_I = pd.read_csv(path + 'csvs/CDQ_I.csv')
DIQ_I = pd.read_csv(path + 'csvs/DIQ_I.csv')
PAQ_I = pd.read_csv(path + 'csvs/PAQ_I.csv')
SMQ_I = pd.read_csv(path + 'csvs/SMQ_I.csv')
WHQ_I = pd.read_csv(path + 'csvs/WHQ_I.csv')

# Impute NaNs to be zeros for pain settings
CDQ_I['CDQ009A', 'CDQ009B', 'CDQ009C',
      'CDQ009D', 'CDQ009E', 'CDQ009F', 'CDQ009G', 'CDQ009H'].fillna(0)

list_x15_16 = list(DIQ_I, ALQ_I, BPQ_I, CDQ_I, DIQ_I, PAQ_I, SMQ_I, WHQ_I)

# Merge datasets
merged = reduce(lambda left, right: pd.merge(left, right, on='SEQ',
                how='outer'), list_x15_16)
