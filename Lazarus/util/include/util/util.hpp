#ifndef LAZARUS_UTIL_HPP
#define LAZARUS_UTIL_HPP

#include <string>

namespace util {

template <typename T>
class Util {
  public:
    Util() = default;
    ~Util() = default;
    auto read_input_file(std::string filename);

  private:

};
} // namespace util

#endif
