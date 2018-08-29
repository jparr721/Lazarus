import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


diabetes_data = pd.read_csv('../Diabetes-Data/diabetes.csv')
y = diabetes_data.Outcome
X = diabetes_data.drop(['Outcome'], axis=1)


