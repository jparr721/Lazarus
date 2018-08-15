import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt


imputer = Imputer()

# Clean the data
data = pd.read_csv('./Diabetes-Data/diabetes2015.csv')

analyzers = [col for col in list(data.columns)]

X = data[analyzers]

# Impute to fix NaN's
imputed_X = X.copy()
imputed_X = imputer.fit_transform(imputed_X)

# Make the k means cluster
km = KMeans(n_clusters=10,
            init='random',
            random_state=0)

# Fit the k means cluster
y_km = km.fit_predict(imputed_X)

plt.scatter(imputed_X[:, 0], imputed_X[:, 1], c=y_km, s=50, cmap='viridis')
plt.grid()
plt.show()
