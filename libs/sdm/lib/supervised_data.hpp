#ifndef SUPERVISED_DATA_H_
#define SUPERVISED_DATA_H_

#include <sdm/lib/io.hpp>

namespace sdm {

template <typename _Scalar, int _Options> class supervised_data {
public:
  supervised_data() {}
  explicit supervised_data(const std::string &libsvm_format);
  supervised_data(
      const Eigen::SparseMatrix<_Scalar, _Options, std::ptrdiff_t> &x,
      const Eigen::ArrayXd &y);
  virtual ~supervised_data() {}

  int get_num_ins(void) const;
  int get_num_fea(void) const;

  int get_num_err(const Eigen::VectorXd &w) const;
  _Scalar get_err(const Eigen::VectorXd &w) const;

  Eigen::SparseMatrix<_Scalar, _Options, std::ptrdiff_t> x_;
  Eigen::ArrayXd y_;
  int num_ins_;
  int num_fea_;
};

template <typename _Scalar, int _Options>
supervised_data<_Scalar, _Options>::supervised_data(const std::string &libsvm_format) {
  sdm::load_libsvm_binary(x_, y_, libsvm_format);
  num_ins_ = x_.rows();
  num_fea_ = x_.cols();
}

template <typename _Scalar, int _Options>
supervised_data<_Scalar, _Options>::supervised_data(
	const Eigen::SparseMatrix<_Scalar, _Options, std::ptrdiff_t> &x,
    const Eigen::ArrayXd &y)
    : x_(x), y_(y), num_ins_(), num_fea_() {
  num_ins_ = x_.rows();
  num_fea_ = x_.cols();
}

template <typename _Scalar, int _Options>
int supervised_data<_Scalar, _Options>::get_num_ins(void) const { return num_ins_; }

template <typename _Scalar, int _Options>
int supervised_data<_Scalar, _Options>::get_num_fea(void) const { return num_fea_; }

template <typename _Scalar, int _Options>
int supervised_data<_Scalar, _Options>::get_num_err(const Eigen::VectorXd &w) const {
  Eigen::VectorXd vw = w;
  if (num_fea_ != w.size())
    vw.conservativeResize(num_fea_);

  Eigen::ArrayXd prob = y_ * (x_ * vw).array();
  int miss = 0.0;
  for (int i = 0; i < num_ins_; ++i)
    miss += static_cast<int>(prob[i] < 0.0);
  return miss;
}

template <typename _Scalar, int _Options>
_Scalar supervised_data<_Scalar, _Options>::get_err(const Eigen::VectorXd &w) const {
  int miss = get_num_err(w);
  return static_cast<_Scalar>(miss) / num_ins_;
}
}

#endif
