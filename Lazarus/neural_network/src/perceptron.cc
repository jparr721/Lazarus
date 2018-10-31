/**
 * Copyright 2018-present, Grand Valley State University DEN Research Lab
 *
 *
 * Implemented by: Jarred Parr
 */

#include <neural_network/perceptron.hpp>

#include <algorithm>
#include <random>

namespace neural_network {
  Perceptron::Perceptron(float eta, int iter, int random_state, int data_size) : _W(data_size) {
    this->eta = eta;
    this->iter = iter;
    this->random_state = random_state;
  }

  std::tuple<std::vector<std::vector<float>>, std::vector<float>> fit(
    std::vector<std::vector<float>> X, std::vector<float> y) {
    for (int i = 0; i < this->iter; ++i) {
      int net_input = this->net_input(X);

    }
  }

  float Perceptron::net_input(std::vector<std::vector<float>> X) {
    return std::inner_product(X, this->_W);
  }

} // namespace neural_network
