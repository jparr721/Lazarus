#ifndef NEURAL_NN_H
#define NEURAL_NN_H

#include <Eigen/Dense>

#include <tuple>
#include <memory>


class MultiLayerPerceptron {
  public:
    MultiLayerPerceptron(
        int n_hidden, int l2, int epochs, float eta, bool shuffle, int minibatch_size, int seed);

    double predict(Eigen::MatrixXd input_layer);
    MultiLayerPerceptron& fit(
        Eigen::MatrixXd X_train, Eigen::MatrixXd y_train, Eigen::MatrixXd X_valid, Eigen::MatrixXd y_valid);
  private:
    float l2 = 0;
    float eta = 0.001;

    int epochs = 100;
    int n_hidden = 30;
    int minibatch_size = 1;
    int seed;

    bool shuffle = true;

    template<typename MatrixDimensionalityType>
    MatrixDimensionalityType sigmoid(MatrixDimensionalityType z);
    float compute_cost(Eigen::Vector1d class_labels, Eigen::MatrixXd output);

    Eigen::MatrixXd weights;
    Eigen::VectorXd bias;

    std::tuple<Eigen::VectorXd,
               Eigen::VectorXd,
               Eigen::VectorXd,
               Eigen::VectorXd>
    forward_propagate(Eigen::MatrixXd input_layer);
    Eigen::MatrixXd onehot(Eigen::Vector1f class_labels, int n_classes);
};

#endif
