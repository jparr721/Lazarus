#ifndef SVM_SVM_HPP
#define SVM_SVM_HPP

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <string>
#include <vector>

namespace svm {

enum KernelFunction {
  linear = 0,
  gaussian = 1,
};

class SVM {
  public:
    std::vector<std::vector<int>> gradient;
    SVM(
        const boost::numeric::ublas::matrix<float>& X,
        const boost::numeric::ublas::vector<float>& y,
        int C) : _X(X), _y(y), _C(C) {};

  private:
    float loss = 0.0;
    boost::numeric::ublas::vector<float> scores;
    boost::numeric::ublas::matrix<float> _X;
    boost::numeric::ublas::vector<float> _y;
    int _C;
    void binarize(boost::numeric::ublas::vector<float> *margins);
    float fit(int max_passes, int tol, KernelFunction kf);
    float compute_gaussian_kernel(int x1, int x2, int sigma);
    float compute_linear_kernel(boost::numeric::ublas::vector<float> x1,
                                boost::numeric::ublas::vector<float> x2);
};
} // namespace svm

#endif
