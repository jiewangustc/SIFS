#ifndef SDM_UTILS_HPP_
#define SDM_UTILS_HPP_

#include "fmath.hpp"
#include "alias.hpp"

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdarg>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace sdm {

static inline void
get_cross_validation_index(const int num_fold, const int num_ins,
                           std::vector<std::vector<int>> &train_indexs,
                           std::vector<std::vector<int>> &valid_indexs,
                           const bool random_flag = false,
                           const bool seed_flag = false) {
  train_indexs.clear();
  train_indexs.resize(num_fold);
  valid_indexs.clear();
  valid_indexs.resize(num_fold);
  std::vector<int> whole_index(num_ins);
  std::iota(std::begin(whole_index), std::end(whole_index), 0);

  if (random_flag) {
    std::mt19937 engine;
    if (seed_flag) {
      std::random_device rnd;
      std::vector<std::uint_least32_t> v(10);
      std::generate(std::begin(v), std::end(v), std::ref(rnd));
      std::seed_seq seed(std::begin(v), std::end(v));
      engine.seed(seed);
    }
    std::shuffle(std::begin(whole_index), std::end(whole_index), engine);
  }

  int each_fold_num_ins = num_ins / num_fold;
  std::vector<int> each_fold_num_ins_vec(num_fold, each_fold_num_ins);
  int rest = num_ins % num_fold;
  for (int i = 0; i < rest; ++i)
    ++each_fold_num_ins_vec[i];

  auto start_it = std::begin(whole_index);
  auto end_it = std::end(whole_index);
  for (int i = 0; i < num_fold; ++i) {
    valid_indexs[i].reserve(each_fold_num_ins_vec[i]);
    end_it = start_it + each_fold_num_ins_vec[i];
    valid_indexs[i].insert(std::end(valid_indexs[i]), start_it, end_it);
    start_it = end_it;
  }

  for (int i = 0; i < num_fold; ++i) {
    std::sort(std::begin(valid_indexs[i]), std::end(valid_indexs[i]));
    train_indexs[i].reserve(num_ins - each_fold_num_ins_vec[i]);
  }

  for (int i = 0; i < num_fold; ++i)
    for (int j = 0; j < num_fold; ++j)
      if (i != j)
        train_indexs[j].insert(std::end(train_indexs[j]),
                               std::begin(valid_indexs[i]),
                               std::end(valid_indexs[i]));

  for (int i = 0; i < num_fold; ++i)
    std::sort(std::begin(train_indexs[i]), std::end(train_indexs[i]));
}

// misc
static inline std::vector<std::string> split_string(const std::string &str,
                                                    const std::string &delim) {
  std::vector<std::string> res;
  std::string::size_type current = 0, found, delimlen = delim.size();
  while ((found = str.find(delim, current)) != std::string::npos) {
    res.push_back(std::string(str, current, found - current));
    current = found + delimlen;
  }
  res.push_back(std::string(str, current, str.size() - current));
  return res;
}

static inline int count_lines(const std::string file_name) {
  int num_lines = 0;
  std::ifstream data_file;
  std::string line;
  data_file.open(file_name);
  if (data_file.fail()) {
    std::cerr << "Unable to open " << file_name << '\n';
    exit(8);
  }
  while (std::getline(data_file, line))
    ++num_lines;
  data_file.close();

  return num_lines;
}

// for cygwin(GCC) can't use std::stoi
static inline int str2int(const std::string &s) {
  int r = 0;
  bool neg = false;
  int index = 0;
  while (s[index] == ' ')
    ++index;
  if (s[index] == '-') {
    neg = true;
    ++index;
  } else if (s[index] == '+') {
    ++index;
  }
  while (s[index] >= '0' && s[index] <= '9') {
    r = (r * 10.0) + (s[index] - '0');
    ++index;
  }
  if (neg) {
    r = -r;
  }
  return r;
}

template <typename ValueType> ValueType naive_atot(const char *p) {
  ValueType r = 0.0;
  bool neg = false;
  while (*p == ' ')
    ++p;
  if (*p == '-') {
    neg = true;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  while (*p >= '0' && *p <= '9') {
    r = (r * 10.0) + (*p - '0');
    ++p;
  }
  if (*p == '.') {
    ValueType f = 0.0;
    int n = 0;
    ++p;
    while (*p >= '0' && *p <= '9') {
      f = (f * 10.0) + (*p - '0');
      ++p;
      ++n;
    }
    r += f / std::pow(10.0, n);
  }
  if (neg) {
    r = -r;
  }
  return r;
}

template <typename ValueType> ValueType naive_atot(const std::string &s) {
  ValueType r = 0.0;
  bool neg = false;
  auto it = s.begin();
  while (*it == ' ')
    ++it;
  if (*it == '-') {
    neg = true;
    ++it;
  } else if (*it == '-') {
    ++it;
  }
  while (*it >= '0' && *it <= '9') {
    r = (r * 10.0) + (*it - '0');
    ++it;
  }
  if (*it == '.') {
    ++it;
    ValueType f = 0.0;
    int n = 0;
    while (*it >= '0' && *it <= '9') {
      f = (f * 10.0) + (*it - '0');
      ++it;
      ++n;
    }
    r += f / std::pow(10.0, n);
  }
  if (neg)
    r = -r;
  return r;
}

template <typename RealType1, typename RealType2>
inline bool lessPair(const std::pair<RealType1, RealType2> &l,
                     const std::pair<RealType1, RealType2> &r) {
  return l.second < r.second;
}

template <typename RealType1, typename RealType2>
inline bool greaterPair(const std::pair<RealType1, RealType2> &l,
                        const std::pair<RealType1, RealType2> &r) {
  return l.second > r.second;
}

template <typename RealType> inline double norm_cdf(const RealType x) {
  constexpr double a1 = 0.254829592;
  constexpr double a2 = -0.284496736;
  constexpr double a3 = 1.421413741;
  constexpr double a4 = -1.453152027;
  constexpr double a5 = 1.061405429;
  constexpr double p = 0.3275911;

  int sign = 1;
  if (x < static_cast<RealType>(0.0))
    sign = -1;
  double tmp = fabs(x) / sqrt(2.0);

  // A&S formula 7.1.26
  double t = 1.0 / (1.0 + p * tmp);
  double y = 1.0 -
             (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t *
                 fmath::exp(-tmp * tmp);

  return 0.5 * (1.0 + sign * y);
}

template <typename RealType> inline double normsinv(const RealType pos) {

  constexpr double A1 = (-3.969683028665376e+01);
  constexpr double A2 = 2.209460984245205e+02;
  constexpr double A3 = (-2.759285104469687e+02);
  constexpr double A4 = 1.383577518672690e+02;
  constexpr double A5 = (-3.066479806614716e+01);
  constexpr double A6 = 2.506628277459239e+00;

  constexpr double B1 = (-5.447609879822406e+01);
  constexpr double B2 = 1.615858368580409e+02;
  constexpr double B3 = (-1.556989798598866e+02);
  constexpr double B4 = 6.680131188771972e+01;
  constexpr double B5 = (-1.328068155288572e+01);

  constexpr double C1 = (-7.784894002430293e-03);
  constexpr double C2 = (-3.223964580411365e-01);
  constexpr double C3 = (-2.400758277161838e+00);
  constexpr double C4 = (-2.549732539343734e+00);
  constexpr double C5 = 4.374664141464968e+00;
  constexpr double C6 = 2.938163982698783e+00;

  constexpr double D1 = 7.784695709041462e-03;
  constexpr double D2 = 3.224671290700398e-01;
  constexpr double D3 = 2.445134137142996e+00;
  constexpr double D4 = 3.754408661907416e+00;

  constexpr double P_LOW = 0.02425;
  /* P_high = 1 - p_low*/
  constexpr double P_HIGH = 0.97575;

  double x = 0.0;
  double q, r;
  if ((0 < pos) && (pos < P_LOW)) {
    q = sqrt(-2 * fmath::log(pos));
    x = (((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6) /
        ((((D1 * q + D2) * q + D3) * q + D4) * q + 1);
  } else {
    if ((P_LOW <= pos) && (pos <= P_HIGH)) {
      q = pos - 0.5;
      r = q * q;
      x = (((((A1 * r + A2) * r + A3) * r + A4) * r + A5) * r + A6) * q /
          (((((B1 * r + B2) * r + B3) * r + B4) * r + B5) * r + 1);
    } else {
      if ((P_HIGH < pos) && (pos < 1)) {
        q = sqrt(-2 * fmath::log(1 - pos));
        x = -(((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6) /
            ((((D1 * q + D2) * q + D3) * q + D4) * q + 1);
      } else {
        std::cout << "CDF position > 1 : " << pos << std::endl;
        x = 1.0;
      }
    }
  }

  /*
  // If you are composiling this under UNIX OR LINUX, you may uncomment this
  // block for better accuracy.
  if(( 0 < pos)&&(pos < 1)){
    e = 0.5 * erfc(-x/sqrt(2)) - pos;
    u = e * sqrt(2*M_PI) * exp(x*x/2);
    x = x - u/(1 + x*u/2);
  }
  */

  return x;
}

template <typename ValueType>
bool compare(const ValueType &lhs, const ValueType &rhs,
             const ValueType eps = 1e-12) {
  return (std::abs(lhs - rhs) <= std::max(lhs, rhs) * eps ||
          std::abs(lhs - rhs) <= eps);
}

template <typename ValueType, int _Rows, int _Cols, int _Option>
ValueType compare(const Eigen::Matrix<ValueType, _Rows, _Cols, _Option> &mat,
                  const ValueType value, const ValueType eps = 1e-12) {
  ValueType v;
  int num_eq = 0;
  if (_Option == Eigen::RowMajor) {
    for (int i = 0; i < mat.rows(); ++i) {
      for (int j = 0; j < mat.cols(); ++j) {
        v = mat.coeffRef(i, j);
        if (compare(v, value, eps))
          ++num_eq;
      }
    }
  } else {
    for (int j = 0; j < mat.cols(); ++j) {
      for (int i = 0; i < mat.rows(); ++i) {
        v = mat.coeffRef(i, j);
        if (compare(v, value, eps))
          ++num_eq;
      }
    }
  }
  return num_eq;
}

} // namespace sdm

#endif
