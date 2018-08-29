import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    # Marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contour(xx1, xx2, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max)
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # Highlight test samples
    if test_idx:
        # Plot all samples
        X_test = X[test_idx, :]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test Set')


def load_data(path):
    diabetes_data = pd.read_csv(path)
    y = diabetes_data.Outcome
    X = diabetes_data.drop(['Outcome'], axis=1)
    return X, y


def main():
    X, y = load_data('../Diabetes-Data/diabetes.csv')

    # Split the data 30% test and 70% train, stratify for proporion
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, train_size=0.7,
                                                        test_size=0.3,
                                                        random_state=0,
                                                        stratify=y)

    sc = StandardScaler()
    svm = SVC(kernel='linear', C=1.0, random_state=1)

    # Estimate the mean and standard deviation of each feature
    sc.fit(X_train)

    # Standardize the inputs
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # X_combined_std = np.vstack((X_train_std, X_test_std))
    # y_combined = np.hstack((y_train, y_test))

    svm.fit(X_train_std, y_train)

    # Make predictions about the standardized input
    y_pred = svm.predict(X_test_std)

    # Predict the accuracy of the predictions and misclassifications
    print('Misclassified samples: {}'.format((y_test != y_pred).sum()))

    # Find accuracy of the results
    print('Accuracy of results: {}'.format(accuracy_score(y_test, y_pred)))

    # Show the test accuracy
    print('Test accuracy: {}'.format(svm.score(X_test_std, y_test)))

    # plot_decision_regions(X_combined_std,
    #                       y_combined,
    #                       svm,
    #                       range(105, 150))

    # plt.xlabel('Biomedical Indicators')
    # plt.ylabel('Outcome')
    # plt.legend(loc='upper left')
    # plt.show()


main()
