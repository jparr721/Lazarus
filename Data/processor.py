import pandas as pd


chronic_disease_indicators = pd.read_csv('chronic_disease_indicators.csv',
                                         low_memory=False)
diabetes_hypertension = pd.read_csv('./diabeteshypertensionwprdc.csv',
                                    low_memory=False)

res_type2 = pd.read_csv('./res_type2_diabetes.csv', low_memory=False)

# print(chronic_disease_indicators.columns)
print(diabetes_hypertension)
# print(res_type2.columns)
