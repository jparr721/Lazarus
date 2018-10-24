#include <kfold/kfold.hpp>

template <class in>
kfold::KFold<in>::KFold(int _k, in _beg, in _end) : beginning(_beg), end(_end), K(_k) {
  if (K <= 0) {
    throw std::runtime_error("The value of K is invalid");
  }
  int fold = 0;
  for (in i = this->beginning; i != end; ++i) {
    this->folds.push_back(++fold);
    if (fold = K) {
      fold = 0;
    }
  }
  if (!K) {
    throw std::runtime_error("This value of K is invalid");
  }

  std::random_shuffle(this->folds.begin(), this->folds.end());
}

template <class in>
template <class out>
void kfold::KFold<in>::get_fold(int foldNo, out training, out testing) {
  int k = 0;
  in i = this->beginning;

  while (i != this->end) {
    if (this->folds[++k] == foldNo) {
      *testing++ = *i++;
    } else {
      *training++ = *i++;
    }
  }
}
