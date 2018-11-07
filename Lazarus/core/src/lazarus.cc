#include <lazarus/lazarus.hpp>
#include <neural_network/perceptron.hpp>
#include <util/util.hpp>

namespace lazarus {
  float test_perceptron() {
    util::Util u;
    neural_network::Perceptron p = new Perceptron();

    auto data = u.read_input_file("../../data/iris.data");

    // Init our X and y to 0
    std::vector<float> y(data.size(), 0);
    std::vector<float> X(data.size(), 0);

    // Create our X and y vectors
    for (int i = 0; i < data.size(); ++i) {
      y.push_back(data[i][data[i].size() - 1]);
      for (int j = 0; j < data[i].size() - 1; ++j) {
        X.push_back(data[i][j]);
      }
    }

    delete p;
  }

} // namespace lazarus
