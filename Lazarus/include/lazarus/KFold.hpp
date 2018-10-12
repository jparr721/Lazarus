#ifndef LAZARUS_KFOLD_H
#define LAZARUS_KFOLD_H

#include <algorithm>
#include <stdexcept>
#include <vector>

// Code adapted from https://sureshamrita.wordpress.com/2011/08/24/c-implementation-of-k-fold-cross-validation/

namespace kfold {
  template <class in>
  class KFold {
    public:
      KFold(int k, in beg, in _end);
      template <class out>
      void get_fold(int fold, out training, out testing);

    private:
      in beginning;
      in end;
      int K; // how many folds
      std::vector<int> fold;
  };

} // namespace kfold

#endif
