#include <util/Util.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace util {
  auto read_input_file(std::string filename) {
    std::vector<std::vector<std::string>> input;
    std::vector<std::string> record;
    std::ifstream in(filename);

    while (in) {
      std::string row;
      if (!std::getline(in, row)) {
        throw std::runtime_error(std::string("Unable to read file properly"));
        break;
      }

      std::istringstream ss(row);

      while (ss) {
        std::string s;
        if (!std::getline(row, s, ',')) {
          throw std::runtime_error(std::string("Unable to read row of data"));
          break;
        }
        record.push_back(s);
      }

      input.push_back(s);
    }
    if (!in.eof()) {
      throw std::runtime_error(std::string("Error! Unable to read file, or it may not exist. Please check your file path");
    }
  }
} // namespace util
