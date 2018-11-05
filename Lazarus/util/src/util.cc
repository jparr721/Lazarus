#include <util/Util.hpp>

#include <stdexcept>

namespace util {
auto matrix_vector_calculation(
    std::vector<std::vector<T>> matrix,
    std::vector<T> vector,
    op o) {
  int cols = matrix[0].size();
  if (cols != vector.size()) {
    throw std::runtime_error(std::string("Dimension mismatch"));
  }

  for (int i = 0; i < matrix.size(); ++i) {
    for (int j = 0; j < matrix[i].size(); ++j) {

    }
  }
}
} // namespace util
