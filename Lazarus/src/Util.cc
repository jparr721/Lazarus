#include "../include/lazarus/Util.hpp"
/* #include <include/Util.hpp> */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <iostream>
#include <random>
#include <stdexcept>

namespace util {
  template <typename T>
  std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> Util::train_test_split(
      std::vector<T> data,
      double training_size,
      double test_size,
      double validation_size
      ) {
    if (training_size > 1 || test_size > 1 || validation_size > 1) {
      throw std::invalid_argument("Numbers must be < 1");
    }
    std::vector<T> random_data(data);
    std::vector<T> random_training;
    std::vector<T> random_test;
    std::vector<T> random_validation;

    // Make our randomizer algorithm
    std::random_device rd;
    std::mt19937 g(rd());

    int dataset_size = data.size();
    double training_number = round(dataset_size * training_size);
    double test_number = round(dataset_size * test_size);
    double validation_number = round(dataset_size * validation_size);

    // Shuffle our random list
    std::shuffle(random_data.begin(), random_data.end(), g);
    random_training = this->slice(random_data, 0, training_number);
    random_test = this->slice(random_data, training_number + 1, (training_number + 1) + test_number);
    random_validation = this->slice(
        random_data,
        (training_number + 1 + test_number),
        (training_number + 1 + test_number) + validation_number);

    return std::make_tuple(random_training, random_test, random_validation);
  }

  template <typename T>
  std::vector<T> slice(const std::vector<T>& v, int start, int end) {
    auto first = v.cbegin() + start;
    auto last = v.cbegin() + end + 1;

    return std::vector<T> (first, last);
  }
} // namespace util
