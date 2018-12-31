#include <neural/nn.hh>

MultiLayerPerceptron::MultiLayerPerceptron(
  int n_hidden, int l2, int epochs, float eta, bool shuffle, int minibatch_size, int seed):
  n_hidden(n_hidden), l2(l2), epochs(epochs), eta(eta), shuffle(shuffle), minibatch_size(minibatch_size), seed(seed) {}
