#ifndef NEURALNETWORK_PERCEPTRON_HPP
#define NEURALNETWORK_PERCEPTRON_HPP

#include <tuple>
#include <vector>

namespace neural_network {
class Perceptron {
  public:
    Perceptron(float eta, int iter, int random_state, int data_size);
    Perceptron(const Perceptron &p);
    ~Perceptron() = default;

    float net_input(const std::vector<float>& X) const;
    float predict(std::vector<float> X);
    void fit(const std::vector<float>& X, const std::vector<float>& y);
    // Linear activation function
    std::vector<float> activation(const std::vector<float>& X) const;
  private:
    float eta;
    int iter, random_state;
    std::vector<float> errors;
    std::vector<float> W;
    std::vector<float> costs;
};
} // namespace neural_network

#endif
