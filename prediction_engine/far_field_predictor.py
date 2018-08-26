import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes_data = pd.read_csv('../Diabetes-Data/diabetes.csv')
target = diabetes_data.Outcome
predictors = diabetes_data.drop(['Outcome'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(predictors,
                                                    target, train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=0)


# Score our dataset to see how it stacks up with a given algo
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return mean_absolute_error(y_test, predictions)


print(score_dataset(X_train, X_test, y_train, y_test))
