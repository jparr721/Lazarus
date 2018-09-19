import pandas as pd
from functools import reduce

path = '~/ML/Lazarus/Diabetes-Data'

DIQ_I = pd.read_csv(path + 'csvs/DIQ_I.csv')
ALQ_I = pd.read_csv(path + 'csvs/ALQ_I.csv')
BPQ_I = pd.read_csv(path + 'csvs/BPQ_I.csv')
CDQ_I = pd.read_csv(path + 'csvs/CDQ_I.csv')
HSQ_I = pd.read_csv(path + 'csvs/HSQ_I.csv')
PAQ_I = pd.read_csv(path + 'csvs/PAQ_I.csv')
PFQ_I = pd.read_csv(path + 'csvs/PFQ_I.csv')
WHQ_I = pd.read_csv(path + 'csvs/WHQ_I.csv')
MCQ_I = pd.read_csv(path + 'csvs/MCQ_I.csv')
INQ_I = pd.read_csv(path + 'csvs/INQ_I.csv')

# Impute NaNs to be zeros for pain settings
CDQ_I['CDQ009A', 'CDQ009B', 'CDQ009C',
      'CDQ009D', 'CDQ009E', 'CDQ009F', 'CDQ009G', 'CDQ009H'].fillna(0)

list_x15_16 = list(DIQ_I, ALQ_I, BPQ_I, CDQ_I,
                   DIQ_I, PAQ_I, WHQ_I, MCQ_I, INQ_I)

# Merge datasets
merged = reduce(lambda left, right: pd.merge(left, right, on='SEQ',
                how='outer'), list_x15_16)

merged.to_csv(path + 'csvs/merged.csv')
