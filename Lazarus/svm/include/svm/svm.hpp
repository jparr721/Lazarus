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
  non_vectorized = 9,
};

class SVM {
  public:
    std::vector<std::vector<int>> gradient;
    SVM(
        const boost::numeric::ublas::matrix<float>& X,
        const boost::numeric::ublas::vector<float>& y,
        int C);

  private:
    float loss = 0.0;
    boost::numeric::ublas::vector<float> scores;
    boost::numeric::ublas::matrix<float> _X;
    boost::numeric::ublas::vector<float> _y;
    int _C;
    void binarize(boost::numeric::ublas::vector<float>& margins);
    float fit(int max_passes, int tol, KernelFunction kf, int sigma);
    float compute_gaussian_kernel(int x1, int x2, int sigma);
    auto compute_linear_kernel(boost::numeric::ublas::matrix<float> x1,
                                boost::numeric::ublas::matrix<float> x2);
};
} // namespace svm

#endif
