/**
 * Copyright 2018-present, Grand Valley State University DEN Research Lab
 *
 * The Adaptive Linear Neuron, also just called a Perceptron, is a basic
 * classifier and the foundation to most neural networks. It can be used
 * in places where a full multilayer network is not needed. This uses the
 * gradient descent and is therefore not the most optimum implementation.
 *
 * Implemented by: Jarred Parr
 */

#include <neural_network/perceptron.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace neural_network {
  Perceptron::Perceptron(float eta, int iter, int random_state, int data_size) : W(data_size) {
    this->eta = eta;
    this->iter = iter;
    this->random_state = random_state;
  }

  void Perceptron::fit(const std::vector<float>& X, const std::vector<float>& y) {
    float errorsum  = 0.0;
    float cost = 0.0;
    std::vector<float> squared_errors;

    for (int i = 0; i < this->iter; ++i) {
      int net_input = this->net_input(X);
      auto output = this->activation(X);
      std::transform(
            y.begin(),
            y.end(),
            output.begin(),
            std::back_inserter(this->errors),
            std::minus<float>());
      auto W_rest(this->W.begin() + 1, this->W.end());
      for (auto val : W_rest) {
        val += this->eta * std::inner_product(X.begin(), X.end(), this->errors.begin(), 0.0);
      }

      for (const auto& e : this->errors) {
        errorsum += e;
        squared_errors.push_back(std::pow(e, 2));
      }
      this->W[0] += this->eta * errorsum;

      for (const auto& se : squared_errors) cost += se;

      this->costs.push_back(cost / 2.0);
    }

    return;
  }

  float Perceptron::predict(std::vector<float> X) {
    return this->activation(this->net_input(X)) >= 0.0
      ? 1
      : -1;
  }

  std::vector<float> Perceptron::activation(const std::vector<float>& X) const {
    return X;
  }

  float Perceptron::net_input(const std::vector<float>& X) const {
    float first = this->W[0];
    std::vector<float> rest(this->W.begin() + 1, this->W.end());
    return std::inner_product(X.begin(), X.end(), rest.begin(), 0.0) + first;
  }

} // namespace neural_network
