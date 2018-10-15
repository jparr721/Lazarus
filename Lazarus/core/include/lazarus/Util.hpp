#ifndef LAZARUS_UTIL_HPP
#define LAZARUS_UTIL_HPP

#include <tuple>
#include <vector>

namespace util {
class Util {
  public:
    // Return tuple of train, test, validation sets from a given vector of generic type
    template <typename T>
    std::tuple<std::vector<T>, std::vector<T>, std::vector<T>> train_test_split(
        std::vector<T> data,
        double training_size = 0.6,
        double test_size = 0.2,
        double validation_size = 0.2);

    template <typename T>
    std::vector<T> slice(const std::vector<T>& v, int start, int end);
};
} // namespae util

#endif
