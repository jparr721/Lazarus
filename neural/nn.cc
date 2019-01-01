#include <neural/nn.hh>
#include <cmath>
#include <random>

MultiLayerPerceptron::MultiLayerPerceptron(
  int n_hidden, int l2, int epochs, float eta, bool shuffle, int minibatch_size, int seed):
  n_hidden(n_hidden), l2(l2), epochs(epochs), eta(eta), shuffle(shuffle), minibatch_size(minibatch_size), seed(seed) {}

float MultiLayerPerceptron::sigmoid(float z) {
  return 1f / (1f + std::exp(-z));
}

Eigen::MatrixXf MultiLayerPerceptron::onehot(Eigen::MatrixXf class_labels, int n_classes) {
  Eigen::MatrixXf (class_labels.cols());
}

float MultiLayerPerceptron::predict(Eigen::MatrixXf input_layer) {
  std::tuple<
    Eigen::Vectorxf,
    Eigen::VectorXf,
    Eigen::VectorXf,
    Eigen::VectorXf> propagated_ouput =
    this->forward_propagate(input_layer);

  Eigen::VectorXf z_output = std::get<2>(propagated_ouput);

  MatrixXf::Index max_row, max_col;
  return z_output.maxCoeff(&max_row, &max_col);
}

std::tuple<float, float, float, float>
MultiLayerPerceptron::forward_propagate(Eigen::MatrixXf input_layer) {

}
