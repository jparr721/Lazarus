#ifndef NEURALNETWORK_MLP_HPP
#define NEURALNETWORK_MLP_HPP

#include <boost/numeric/ublas/matrix.hpp>

namespace neural_network {
class MultiLayerPerceptron {
  public:
    MultiLayerPerceptron() = default;
    MultiLayerPerceptron(const MultiLayerPerceptron &mlp);
    ~MultiLayerPerceptron() = default;

    double sigmoid(double z);
    void predict(
        double theta1,
        double theta2,
        const boost::numeric::ublas::matrix<double>& X);
    void sigmoidGradient();
    void init_weights();
    void cost_function();
    void compute_numerical_gradient();
};
}

#endif
