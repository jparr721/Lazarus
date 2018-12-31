#pragma once

#include <Eigen/Dense>


class MultiLayerPerceptron {
  public:
    MultiLayerPerceptron(
        int n_hidden, int l2, int epochs, float eta, bool shuffle, int minibatch_size, int seed);
    ~MultiLayerPerceptron() = default;

    float forward_propagate(Eigen::Vector1f input_layer);
    float predict(Eigen::Vector1f input_layer);
    MultiLayerPerceptron& fit(
        Eigen::MatrixXf X_train, Eigen::Vector1f y_train, Eigen::MatrixXf X_valid, Eigen::Vector1f y_valid);
  private:
    float l2 = 0;
    float eta = 0.001;

    int epochs = 100;
    int n_hidden = 30;
    int minibatch_size = 1;
    int seed;

    bool shuffle = true;

    float sigmoid(float z);
    float compute_cost(Eigen::Vector1d class_labels, Eigen::MatrixXf output);
    Eigen::Vector1d onehot(Eigen::Vector1f class_labels, int n_classes);
};
