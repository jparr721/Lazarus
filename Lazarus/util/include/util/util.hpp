#ifndef LAZARUS_UTIL_HPP
#define LAZARUS_UTIL_HPP

#include <vector>

namespace util {

// Ensures the op types are standard
enum op {
  ADD = 0,
  SUBTRACT = 1,
  MULTIPLY = 2,
  DIVIDE = 3,
};

template <typename T>
class Util {
  public:
    Util() = default;
    ~Util() = default;
    auto matrix_vector_calculation(
        std::vector<std::vector<T>> matrix,
        std::vector<T> vector,
        op o);

  private:

};
} // namespace util

#endif
