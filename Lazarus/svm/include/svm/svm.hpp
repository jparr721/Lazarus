#ifndef SVM_SVM_HPP
#define SVM_SVM_HPP

#include <string>
#include <vector>

namespace svm {
class SVM {
  public:
    SVM(std::vector<std::vector<int>> W, std::vector<int> y);
};
} // namespace svm

#endif
