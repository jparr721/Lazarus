/**
 * Copyright 2018-present, Grand Valley State University DEN Research Lab
 *
 * The SVM classifier is a C++ implementation of the support vector machine
 * wide margin classifier. This class uses gradient descent to minimize the
 * loss function and functions on a gaussian kernel.
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
        const boost::numeric::ublas::matrix<float>& W,
        const boost::numeric::ublas::matrix<float>& X,
        const boost::numeric::ublas::vector<float>& y,
        int reg) {};

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

  /// Fits the data to the support vector machine clasifier
  ///
  /// |param vector<float> scores - The scores for our initial
  /// that need to be optimized
  float SVM::fit(boost::numeric::ublas::vector<float> scores) {
    // Get our initial scores
    scores = boost::numeric::ublas::prod(this->X, this->W);
    boost::numeric::ublas::vector<float> yi_scores =
  }

  /// Computes th gaussian kernel which is the radial basis
  /// function kernel between two observations, x1 and x2
  ///
  /// |param int x1 - The first observation
  /// |param int x2 - The second observation
  ///
  /// Note: Observations do not maintain a particular order
  /// when passed in as parameters
  float SVM::compute_gaussian_kernel(int x1, int x2, int sigma) {
    float sim = 0.0;

    std::complex<int> _x1(x1);
    td::complex<int> _x2(x2);

    return exp(- std::norm(x1 - x2)^2 / (2 * sigma ^ 2));
  }
} // namespace svm
