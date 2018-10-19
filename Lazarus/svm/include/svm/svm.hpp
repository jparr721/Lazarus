#ifndef SVM_SVM_HPP
#define SVM_SVM_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <string>
#include <vector>

namespace svm {
class SVM {
  public:
    std::vector<std::vector<int>> gradient;
    SVM(
        const boost::numeric::ublas::matrix<float>& W,
        const boost::numeric::ublas::vector<float>& X,
        const boost::numeric::ublas::vector<float>& y,
        int reg) : _W(W), _X(X), _y(y), _reg(reg) {};

  private:
    float loss = 0.0;
    boost::numeric::ublas::vector<float> scores;
    boost::numeric::ublas::matrix<float> _W;
    boost::numeric::ublas::vector<float> _X;
    boost::numeric::ublas::vector<float> _y;
    int _reg;
    void binarize(boost::numeric::ublas::vector<float> *margins);
    float fit(boost::numeric::ublas::vector<float> scores);
    float compute_gaussian_kernel(int x1, int x2, int sigma);
};
} // namespace svm

#endif
