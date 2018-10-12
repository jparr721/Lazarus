import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import datetime


def calculate_field_average(stats):
    avg_result_accuracy = (sum([stat[0] for stat in stats]) /
                           len([stat[0] for stat in stats]))
    avg_misclassified = (sum([stat[1] for stat in stats]) /
                         len([stat[1] for stat in stats]))
    avg_training_accuracy = (sum([stat[2] for stat in stats]) /
                             len([stat[2] for stat in stats]))
    avg_test_accuracy = (sum([stat[3] for stat in stats]) /
                         len([stat[3] for stat in stats]))

    return (avg_result_accuracy,
            avg_misclassified,
            avg_training_accuracy,
            avg_test_accuracy)


def log_results(stats, console=False):
    """
    Takes an output tuple of
    values and logs each to a
    file with some formatting
    """
    now = datetime.datetime.now()

    logfile = open('samples.txt', 'a')
    logfile.write('\nRun time: {}'.format(str(now)))

    for output in stats:
        formatted_text = '''
        Results Accuracy: {}
        Misclassified Samples: {}
        Training Accuracy: {}
        Test Accuracy: {}
        '''.format(output[0], output[1], output[2], output[3])

        if console:
            print(formatted_text)
        else:
            logfile.write(formatted_text)

    avg = calculate_field_average(stats)

    averages = '''
    Average Results Accuracy: {}
    Average Misclassified Samples: {}
    Average Training Accuracy: {}
    Average Test Accuracy: {}
    '''.format(avg[0], avg[1], avg[2], avg[3])

    if console:
        print('\n')
        print(averages)
        print('## END')
    else:
        logfile.write(averages)
        logfile.write('## END SAMPLE')


def plot_decision_regions(X, y, classifier, test_idx=None,
                          resolution=0.02):
    """
    Makes a color coded plot of whatever
    learning model you inject
    """

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
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # Plot all samples
        X_test = X[test_idx, :]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test Set')


def load_data(path):
    diabetes_data = pd.read_csv(path)
    y = diabetes_data.Diabetes
    X = diabetes_data.drop(['Diabetes'], axis=1)
    return X, y


def classify():
    # X, y = load_data('../Diabetes-Data/diabetes.csv')
    X, y, = load_data('../Diabetes-Data/csvs/merged.csv')

    # Split the data 30% test and 70% train, stratify for proporion
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, train_size=0.7,
                                                        test_size=0.3,
                                                        stratify=y)

    sc = StandardScaler()
    svm = SVC(kernel='rbf', C=3.0, random_state=1)

    # Estimate the mean and standard deviation of each feature
    sc.fit(X_train)

    # Standardize the inputs
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    svm.fit(X_train_std, y_train)

    # Make predictions about the standardized input
    y_pred = svm.predict(X_test_std)

    # # Predict the accuracy of the predictions and misclassifications
    # print('Misclassified samples: {}'.format((y_test != y_pred).sum()))

    # # Find accuracy of the results
    # print('Accuracy of results: {}'.format(accuracy_score(y_test, y_pred)))

    # # Show the training accuracy
    # print('Train accuracy: {}'.format(svm.score(X_train_std, y_train)))

    # # Show the test accuracy
    # print('Test accuracy: {}'.format(svm.score(X_test_std, y_test)))

    # Assign accuracies to variables
    results_accuracy = accuracy_score(y_test, y_pred)
    misclassified_samples = (y_test != y_pred).sum()
    train_accuracy = svm.score(X_train_std, y_train)
    test_accuracy = svm.score(X_test_std, y_test)

    return (results_accuracy,
            misclassified_samples,
            train_accuracy,
            test_accuracy)


def main():
    resultslist = []
    for i in range(20):
        result = classify()

        resultslist.append(result)

    log_results(resultslist, True)


main()
