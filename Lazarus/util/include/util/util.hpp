#ifndef LAZARUS_UTIL_HPP
#define LAZARUS_UTIL_HPP

#include <string>
#include <vector>

namespace util {

class Util {
  public:
    Util() = default;
    ~Util() = default;
    std::vector<std::vector<float>> read_input_file(std::string filename);

  private:

};
} // namespace util

#endif
