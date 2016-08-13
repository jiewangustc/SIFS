#ifndef SUPERVISED_DATA_REGRESSION_H_
#define SUPERVISED_DATA_REGRESSION_H_

#include <sdm/lib/io.hpp>

namespace sdm {

template <typename _Scalar, int _Options> class supervised_data_regression {
public:
  supervised_data_regression() {}
  explicit supervised_data_regression(const std::string &libsvm_format);
  supervised_data_regression(
      const Eigen::SparseMatrix<_Scalar, _Options, std::ptrdiff_t> &x,
      const Eigen::ArrayXd &y);
  virtual ~supervised_data_regression() {}

  int get_num_ins(void) const;
  int get_num_fea(void) const;

  Eigen::SparseMatrix<_Scalar, _Options, std::ptrdiff_t> x_;
  Eigen::ArrayXd y_;
  int num_ins_;
  int num_fea_;
};

template <typename _Scalar, int _Options>
supervised_data_regression<_Scalar, _Options>::supervised_data_regression(
    const std::string &libsvm_format) {
  sdm::load_libsvm(x_, y_, libsvm_format);
  num_ins_ = x_.rows();
  num_fea_ = x_.cols();
}

template <typename _Scalar, int _Options>
supervised_data_regression<_Scalar, _Options>::supervised_data_regression(
    const Eigen::SparseMatrix<_Scalar, _Options, std::ptrdiff_t> &x,
    const Eigen::ArrayXd &y)
    : x_(x), y_(y), num_ins_(), num_fea_() {
  num_ins_ = x_.rows();
  num_fea_ = x_.cols();
}

template <typename _Scalar, int _Options>
int supervised_data_regression<_Scalar, _Options>::get_num_ins(void) const {
  return num_ins_;
}

template <typename _Scalar, int _Options>
int supervised_data_regression<_Scalar, _Options>::get_num_fea(void) const {
  return num_fea_;
}
}

#endif
