/**
 * Copyright 2018-present, Grand Valley State University DEN Research Lab
 *
 * The SVM classifier is a C++ implementation of the support vector machine
 * wide margin classifier. This class uses gradient descent to minimize the
 * loss function and functions on a gaussian or linear kernel.
 *
 * Implemented by: Jarred Parr
 */


#include <svm/svm.hpp>

#include <cmath>
#include <complex>

namespace svm {
  /// Support Vector Machine constructor, this will
  /// exist as the entry point for the application
  ///
  /// |param matrix<float> W - The weight matrix
  /// |param matrix<float> X - The feature matrix
  /// |param vector<float> y - The class label vector
  SVM::SVM(
        const boost::numeric::ublas::matrix<float>& X,
        const boost::numeric::ublas::vector<float>& y,
        int C) : _X(X), _y(y), _C(C) {};

  /// Binarize the matrix to standardize the data into one
  /// group or another for more straightforward binary
  /// classification
  ///
  /// |param vector<float> *margins - The values in each margin
  void SVM::binarize(boost::numeric::ublas::vector<float> *margins) {
    for (unsigned int i = 0; i < *margins.size(); ++i) {
       if (*margins(i) > 0) {
          *margins(i) = 1;
       }
    }

    return;
  }

  /// Computes th gaussian kernel which is the radial basis
  /// function kernel between two observations, x1 and x2
  ///
  /// |param int x1 - The first observation
  /// |param int x2 - The second observation
  ///
  /// Note: Observations do not need tomaintain a particular order
  /// when passed in as parameters
  float SVM::compute_gaussian_kernel(int x1, int x2, int sigma) {
    float sim = 0.0;

    std::complex<int> _x1(x1);
    std::complex<int> _x2(x2);

    return exp(- std::norm(x1 - x2)^2 / (2 * sigma ^ 2));
  }

  /// Computes the linear kernel between vectors x1 and x2
  /// and returns the value in the simulation
  ///
  /// |param vector<float> x1 - The first vector
  /// |param vector<float> x2 - The second vector
  float SVM::compute_linear_kernel(
      boost::numeric::ublas::vector<float> x1,
      boost::numeric::ublas::vector<float> x2
      ) {
    float sim = 0.0;

    sim = x1 * x2;

    return sim;
  }

  /// Trains an SVM classifier using a simplified verison
  /// of the SMO algorithm. This function uses the class members
  /// X, y, and C with X being the matrix of traininig examples,
  /// y being the column matrix of 0 or 1 values as the class
  /// labels. C is the standard SVM regularization parameter.
  ///
  /// |param int max_passes - The number of iterations over the dataset
  ///                     before the algorithm stops training
  /// |param int tol (optional) - The tolerance paramter for
  ///                         floating point values
  /// |param KernelFunction kf - The kernel function to be used in the
  ///                           classifier (linear or gaussian)
  float SVM::fit(
      int max_passes=5,
      int tol=1e-3,
      KernelFunction kf=KernelFunction::linear
      ) {
    int m = this->X.size1();
    int n = this->X.size2();

    // Change all occurances of 0 to -1 in Y
    for (auto i = 0; i < this->Y.size(); ++i) {
      if (this->Y(i) == 0) {
        this->Y(i) = -1;
      }
    }

    boost::numeric::ublas::vector<int> zeros();
  }

} // namespace svm
