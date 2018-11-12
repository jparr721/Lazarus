#include <util/util.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace util {
  std::vector<std::vector<float>> read_input_file(std::string filename) {
    std::vector<std::vector<float>> input;
    std::vector<float> record;
    std::ifstream in(filename);

    std::string row;
    while (std::getline(in, row)) {
      std::istringstream iss(row);

      while (iss.good()) {
        std::string value;
        std::getline(iss, value, ',');
        record.push_back(std::stof(value.c_str()));
      }
      input.push_back(record);
    }

    return input;
  }
} // namespace util
