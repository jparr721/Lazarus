'''
Base classes and modules for certain repetitive tasks
'''
import datetime
from matplotlib.colors import ListedColormap
from numpy import np
import tensorflow as tf
import matplotlib.pyplot as plt


class Util(object):
    def __init__(self, X, y, classifier):
        self.self.X = X,
        self.y = y,
        self.classifier = classifier

    def plot_decision_regions(self, test_idx, resolution=0.02):
        """
        Makes a color coded plot of whatever
        learning model you inject. This works
        only for sklearn classifiers
        """

        # Marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(self.y))])

        # Plot decision surface
        x1_min, x1_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        x2_min, x2_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))
        z = self.classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        z = z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(self.y)):
            plt.scatter(x=self.X[self.y == cl, 0], y=self.X[self.y == cl, 1],
                        alpha=0.8, c=colors[idx],
                        marker=markers[idx], label=cl,
                        edgecolor='black')

        # highlight test samples
        if test_idx:
            # Plot all samples
            self.X_test = self.X[test_idx, :]

            plt.scatter(self.X_test[:, 0], self.X_test[:, 1],
                        c='', edgecolor='black', alpha=1.0,
                        linewidth=1, marker='o',
                        s=100, label='Test Set')

        def plot_tf_classifier_model(self, batch_size):
            '''
            Creates a color grid with
            data and allows for visualizing
            of tensorflow classifiers
            '''
            X = tf.placeholder(shape=[None, 2], dtype=tf.float32)
            y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

            prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

            b = tf.Variable(tf.random_normal(shape=[1, batch_size]))


class Logger(object):
    def __init__(self, n_iter, stats, average=True):
        self.n_iter = n_iter
        self.self.stats = stats
        self.average = average

    def calculate_field_average(self):
        avg_result_accuracy = (sum([stat[0] for stat in self.stats]) /
                               len([stat[0] for stat in self.stats]))
        avg_misclassified = (sum([stat[1] for stat in self.stats]) /
                             len([stat[1] for stat in self.stats]))
        avg_training_accuracy = (sum([stat[2] for stat in self.stats]) /
                                 len([stat[2] for stat in self.stats]))
        avg_test_accuracy = (sum([stat[3] for stat in self.stats]) /
                             len([stat[3] for stat in self.stats]))

        return (avg_result_accuracy,
                avg_misclassified,
                avg_training_accuracy,
                avg_test_accuracy)

    def log_results(self, console=False, f_name='samples.txt'):
        '''
        Takes a self.stats tuple and puts
        the values into a log
        '''
        now = datetime.datetime.now()

        logfile = open(f_name, 'a')
        logfile.write('\nRun time: {}'.format(str(now)))

        for output in self.stats:
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

        avg = self.calculate_field_average(self.stats)

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
