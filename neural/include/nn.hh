#pragma once

#include <Eigen/Dense>

#include <tuple>


class MultiLayerPerceptron {
  public:
    MultiLayerPerceptron(
        int n_hidden, int l2, int epochs, float eta, bool shuffle, int minibatch_size, int seed);
    ~MultiLayerPerceptron() = default;

    float predict(Eigen::MatrixXf input_layer);
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

    std::tuple<float, float, float, float>
      forward_propagate(Eigen::MatrixXf input_layer);
    Eigen::MatrixXf onehot(Eigen::Vector1f class_labels, int n_classes);
};
