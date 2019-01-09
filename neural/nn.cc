#include <neural/nn.h>

#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

MultiLayerPerceptron::MultiLayerPerceptron(
  int n_hidden, int l2, int epochs, float eta, bool shuffle, int minibatch_size, int seed):
  n_hidden(n_hidden), l2(l2), epochs(epochs), eta(eta), shuffle(shuffle), minibatch_size(minibatch_size), seed(seed) {}

template<typename MatrixDimensionalityType>
MatrixDimensionalityType MultiLayerPerceptron::sigmoid(MatrixDimensionalityType z) {
  // Allows us to sigmiod a scalar or element-wise in a matrix/vector
  return 1f / (1f + std::exp(-z));
}

Eigen::MatrixXd MultiLayerPerceptron::onehot(Eigen::MatrixXd class_labels, int n_classes) {
  // Make the identity  matrix based on the number of available class labels
  Eigen::MatrixXd encoded_values = MatrixXd::Identity(class_labels.rows(), class_labels.cols());
  return encoded_values.transpose();
}

double MultiLayerPerceptron::predict(Eigen::MatrixXd input_layer) {
  std::tuple<Eigen::VectorXd,
             Eigen::VectorXd,
             Eigen::VectorXd,
             Eigen::VectorXd> propagated_ouput =
  forward_propagate(input_layer);

  Eigen::VectorXd z_output = std::get<2>(propagated_ouput);

  MatrixXd::Index max_row, max_col;
  return z_output.maxCoeff(&max_row, &max_col);
}

std::tuple<Eigen::VectorXd,
           Eigen::VectorXd,
           Eigen::VectorXd,
           Eigen::VectorXd>
MultiLayerPerceptron::forward_propagate(Eigen::MatrixXd input_layer) {
  // input_layer multiplied by all of our hidden layers
  Eigen::VectorXd z_h = input_layer.dot(weights) + bias;

  // Calculate our activation of the net input layer
  Eigen::VectorXd a_h = sigmoid(z_h);

  // Calculate the net input of our output layer
  Eigen::MatrixXd z_out = a_h.dot(weights) + bias;

  // Calculate sigmoid for continuous ouput
  Eigen::MatrixXd a_out = sigmoid(z_out);

  // Return all of our values as specified in the vectorized nn
  return std::make_typle(z_h, a_h, z_out, a_out);
}

MultiLayerPerceptron& MultiLayerPerceptron::fit(
        Eigen::MatrixXd X_train, Eigen::MatrixXd y_train, Eigen::MatrixXd X_valid, Eigen::MatrixXd y_valid) {
  std::random_device rd{};
  //Use merseinne(or however it's spelled) for random distr.
  std::mt19937 generator{rd()};
  std::normal_distribution<double> dist(0.0, 1.0);

  int n_output = y_train.cols();
  int n_features = X_train.cols() - 1;


  // Initialize weights
  bias = Eigen::VectorXd::Zeros(n_output);
  // Initialze weights as standard normal distribution
  // This prevents issues during optimization
  weights = Eigen::MatrixXd(X_train.rows(), X_train.cols());
  for (size_t i = 0u; i < weights.rows(); ++i) {
    for (size_t j = 0u; j < weigts.cols(); ++j) {
      weights(i, j) = dist(generator());
    }
  }

  // Our output dict to track returns
  std::unordered_map<std::string, std::vector<double>> eval = {
    "cost", {},
    "train_acc", {},
    "valid_acc", {}
  };

  Eigen::MatrixXd y_train_encoded = onehot<Eigen::MatrixXd>(y_train, n_output);

  // Begin backpropagation setup
  for (int i = 0; i < epochs; ++i) {
    Eigen::VectorXd indices;
    if (!shuffle) {
      // Initialize our index vector
      for (int j = 0; j < X_train.rows(); ++j)
        indices(j) = j;
    } else {
      indices::Random(X_train.rows());
    }

    for (int j = 0; j < indices.row(); ++j) {
      int batch_idx = indices(j);

      std::tuple<Eigen::VectorXd,
                 Eigen::VectorXd,
                 Eigen::VectorXd,
                 Eigen::VectorXd> propagated_output = forward_propagate(X_train(batch_idx));

      // Backpropagation
      Eigen::VectorXd z_h = std::get<0>(propagated_output);
      Eigen::VectorXd a_h = std::get<1>(propagated_output);
      Eigen::VectorXd z_out = std::get<2>(propagated_output);
      Eigen::VectorXd a_out = std::get<3>(propagated_output);

      // Compute the error at each node at each layer
      Eigen::VectorXd sigma_out = a_out - y_train_encoded[batch_idx];

      Eigen::MatrixXd sigmoid_derivative_h = a_h * (1.0 - a_h);

    }
  }
}
