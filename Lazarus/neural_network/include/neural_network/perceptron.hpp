#ifndef NEURALNETWORK_PERCEPTRON_HPP
#define NEURALNETWORK_PERCEPTRON_HPP

#include <tuple>
#include <vector>

namespace neural_network {
class Perceptron {
  public:
    Perceptron(float eta, int iter, int random_state);
    Perceptron(const Perceptron &p);
    ~Perceptron() = default;

    std::vector<std::vector<float>>
    float net_input(std::vector<std::vector<float>> X);
    float predict(std::vector<std::vector<float>> X);
    std::tuple<std::vector<std::vector<float>>, std::vector<float>> fit(
        std::vector<std::vector<float>> X, std::vector<float> y);
    // Linear activation function
    auto activation(std::vector<std::vector<float>> X);
  private:
    float eta;
    int iter, random_state;
    std::vector<float> errors;
    std::vector<float> W;
    std::vector<float> cost;
};
} // namespace neural_network

#endif
