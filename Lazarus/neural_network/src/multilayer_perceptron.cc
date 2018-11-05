#include <neural_network/multilayer_perceptron.hpp>

#include <cmath>

namespace neural_network {
  /**
   * The sigmoid function computes the sigmoid
   * of a given input z
   *
   * @param z {double} - The input value
   */
  double MultiLayerPerceptron::sigmoid(double z) {
    return 1 / (1 + exp(-z));
  }

  /**
   * Predict the label of an input
   * given a trained neural network
   *
   */
} // namespace neural_network
