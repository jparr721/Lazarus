import pandas as pd
from sklean.model_selection import train_test_split
from sklearn.preprocessing import StandardScalar
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.svm import SVC


class SVMDiabetesClassifier(object):
    def __init(self, kernel='rbf', C=1.0, random_state=1):

