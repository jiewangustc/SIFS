#ifndef SDM_STATS_HPP_
#define SDM_STATS_HPP_

#include "utils.hpp"

#include <boost/math/distributions.hpp>

namespace sdm {

// Eigen's Matrix class has mean function

template <typename RealType> double mean(const std::vector<RealType> &vec) {
  double sum = 0.0;
  if (!vec.empty()) {
    for (auto i = vec.begin(); i != vec.end(); ++i)
      sum += *i;
    sum /= static_cast<RealType>(vec.size());
  }
  return sum;
}

// trimProb% trimmed mean
template <typename RealType, int Options>
inline double
trimedMean(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
           const double trimProb = 0.5) {
  int l = vec.size();
  Eigen::Matrix<RealType, Eigen::Dynamic, 1> v_copy = vec;
  std::sort(v_copy.data(), v_copy.data() + v_copy.size());
  return (v_copy.segment((int)(nearbyint(l * trimProb)),
                         (int)(nearbyint(l * (1.0 - trimProb))))).mean();
}

// ----- scale mesure -----
template <typename RealType, int Options>
inline double
vari(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
     const double the_mean) {
  double v = 0.0;
  if (vec.size() > 2)
    v = ((vec.array() - the_mean).square().sum()) / (vec.size() - 1);
  return v;
}

template <typename RealType, int Options>
inline double vari(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> &mat,
    const double the_mean) {
  double v = 0.0;
  if (mat.rows() * mat.cols() > 2)
    v = ((mat.array() - the_mean).square().sum()) /
        (mat.rows() * mat.cols() - 1);
  return v;
}

template <typename RealType, int Options>
inline double
vari(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  return vari(vec, vec.mean());
}

template <typename RealType, int Options>
inline double vari(const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic,
                                       Options> &mat) {
  return vari(mat, mat.mean());
}

template <typename RealType, int Options>
inline double
stdv(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
     const double the_mean) {
  return sqrt(vari(vec, the_mean));
}

template <typename RealType, int Options>
inline double stdv(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> &mat,
    const double the_mean) {
  return sqrt(vari(mat, the_mean));
}

template <typename RealType, int Options>
inline double
stdv(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  return sqrt(vari(vec));
}

template <typename RealType, int Options>
inline double stdv(const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic,
                                       Options> &mat) {
  return sqrt(vari(mat));
}

template <typename RealType, int Options>
inline double
MLvari(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
       const double the_mean) {
  double v = 0.0;
  if (!vec.empty())
    v = ((vec.array() - the_mean).squared().sum()) / (vec.size());
  return v;
}

template <typename RealType, int Options>
inline double MLvari(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> &mat,
    const double the_mean) {
  double v = 0.0;
  if (mat.rows() * mat.cols() > 1)
    v = ((mat.array() - the_mean).square().sum()) / (mat.rows() * mat.cols());
  return v;
}

template <typename RealType, int Options>
inline double
MLvari(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  return MLvari(vec, vec.mean());
}

template <typename RealType, int Options>
inline double MLvari(const Eigen::Matrix<RealType, Eigen::Dynamic,
                                         Eigen::Dynamic, Options> &mat) {
  return MLvari(mat, mat.mean());
}

template <typename RealType, int Options>
inline double
MLstdv(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
       const double the_mean) {
  return sqrt(MLvari(vec, the_mean));
}

template <typename RealType, int Options>
inline double MLstdv(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> &mat,
    const double the_mean) {
  return sqrt(MLvari(mat, the_mean));
}

template <typename RealType, int Options>
inline double
MLstdv(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  return sqrt(MLvari(vec));
}

template <typename RealType, int Options>
inline double MLstdv(const Eigen::Matrix<RealType, Eigen::Dynamic,
                                         Eigen::Dynamic, Options> &mat) {
  return sqrt(MLvari(mat));
}

template <typename RealType, int Options>
inline double
quantileSorted(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
               const double order) {
  // SDMAssert(0.0 <= order && order <= 1.0);
  int l = vec.size();
  double tmp = (l - 1) * order;
  int lowerIndex = int(floor(tmp));
  int upperIndex = int(ceil(tmp));
  tmp -= lowerIndex;
  return (1.0 - tmp) * vec[lowerIndex] + tmp * vec[upperIndex];
}

template <typename RealType, int Options>
inline double
quantileUnsorted(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
                 const double order) {
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> vv = vec;
  std::sort(vv.data(), vv.data() + vv.size());
  return quantileSorted(vv, order);
}

template <typename RealType, int Options>
inline double
quantile(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
         const double order) {
  return quantileUnsorted(vec, order);
}

template <typename RealType, int Options>
inline double
quartileRange(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> vv = vec;
  std::sort(vv.data(), vv.data() + vv.size());
  return quantileSorted(vv, 0.75) - quantileSorted(vv, 0.25);
}

template <typename RealType, int Options>
inline double meanAbsoluteDeviation(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  return meanAbsoluteDeviation(vec, vec.maen());
}

template <typename RealType, int Options>
inline double meanAbsoluteDeviation(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
    const double the_mean) {
  int l = vec.size();
  double ans = 0.0;
  if (!vec.empty()) {
    for (int i = 0; i < l; ++i)
      ans += fabs(vec[i] - the_mean);
    ans /= static_cast<double>(l);
  }
  return ans;
}

template <typename RealType, int Options>
RealType range(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  return static_cast<RealType>(vec.maxCoeff() - vec.minCoeff());
}

template <typename RealType, int Options>
inline double stdvByQuartileRange(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> vv = vec;
  std::sort(vv.data(), vv.data() + vv.size());
  boost::math::normal_distribution<double> n_dstr(0.0, 1.0);
  double x = boost::math::quantile(n_dstr, 0.75);
  return (quantileSorted(vv, 0.75) - quantileSorted(vv, 0.25)) / x;
}

template <typename RealType, int Options>
inline double
lpNorm(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
       const int order) {
  double ans;
  if (order == 1) {
    ans = vec.cwiseAbs().sum();
  } else if (order == 2) {
    ans = vec.norm();
  } else {
    using std::pow;
    ans = pow(vec.cwiseAbs().array().pow(order).sum(), RealType(1) / order);
  }
  return ans;
}

template <typename RealType, int Options>
inline double
moment(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
       const int order) {
  double lp = lpNorm(vec, order);
  return lp / static_cast<double>(vec.size());
}

template <typename RealType, int Options>
inline double
normalizedMoment(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
                 const int order, const double the_mean,
                 const double the_stdv) {
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> vv =
      ((vec.array() - the_mean) / the_stdv).matrix();
  return moment(vv, order);
}

template <typename RealType, int Options>
inline double
normalizedMoment(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
                 const int order) {
  double m = vec.mean();
  double s = stdv(vec, m);
  return normalizedMoment(vec, order, m, s);
}

template <typename RealType, int Options>
inline double
skew(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  return normalizedMoment(vec, 3);
}

template <typename RealType, int Options>
inline double
skew(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
     const double the_mean, const double the_stdv) {
  return normalizedMoment(vec, 3, the_mean, the_stdv);
}

template <typename RealType, int Options>
inline double
kurt(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  return normalizedMoment(vec, 4) - 3.0;
}

template <typename RealType, int Options>
inline double
kurt(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
     const double the_mean, const double the_stdv) {
  return normalizedMoment(vec, 4, the_mean, the_stdv) - 3.0;
}

template <typename RealType, int Options>
double median(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  return quantile(vec, 0.5);
}

template <typename RealType, int Options>
double medianAbsoluteDeviation(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
    double the_median) {
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> mad =
      ((vec.array() - the_median).abs()).matrix();
  return median(mad);
}

template <typename RealType, int Options>
double medianAbsoluteDeviation(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  double the_median = median(vec);
  return medianAbsoluteDeviation(vec, the_median);
}

template <typename RealType, int Options>
void meanAndVari(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
                 double &the_mean, double &the_vari) {
  the_mean = vec.mean();
  the_vari = vari(vec, the_mean);
}

template <typename RealType, int Options>
void meanAndStdv(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec,
                 double &the_mean, double &the_stdv) {
  the_mean = vec.mean();
  the_stdv = stdv(vec, the_mean);
}

template <typename RealType, int Options>
double coefficientOfVariation(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  double m, s;
  meanAndVari(vec, m, s);
  // if (fabs(m) < 1e-12)
  //   SDMError("in coefficientOfVariation(...), the mean is almost 0");
  return s / m;
}

template <typename RealType, int Options>
Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> covMat(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> &mat,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &mean_vec) {
  int n = mat.cols();
  // SDMAssert(2 <= n);
  int p = mat.rows();
  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> ans(p, p);
  double ans_jk = 0.0;
  for (int j = 0; j < p; ++j) {
    for (int k = j; k < p; ++k) {
      for (int i = 0; i < n; ++i)
        ans_jk += (mat.coeffRef(i, j) - mean_vec[j]) *
                  (mat.coeffRef(i, k) - mean_vec[k]);
      ans.coeffRef(j, k) = ans.coeffRef(k, j) = ans_jk / (n - 1);
    }
  }
  return ans;
}

template <typename RealType, int Options>
Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options>
correlationMat(const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic,
                                   Options> &covMat) {
  int p = covMat.cols();
  // if (p != covMat.rows())
  //   SDMError("covMat must be square matrix");
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> diag(p);
  for (int i = 0; i < p; ++i)
    diag[i] = sqrt(covMat.coeffRef(i, i));
  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> ans = covMat;
  for (int i = 0; i < p; ++i) {
    for (int j = i + 1; j < p; ++j)
      ans.coeffRef(i, j) = ans.coeffRef(j, i) /= (diag[i] * diag[j]);
    ans.coeffRef(i, i) = 1.0;
  }
  return ans;
}

template <typename RealType, int Options>
Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> pooledCovMat(
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> &
        data1,
    const Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> &
        data2,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &meanVec1,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &meanVec2) {
  int n1 = data1.cols();
  int n2 = data2.cols();
  int n = n1 + n2;
  int p = data1.rows();
  // SDMAssert(2 <= n1 && 2 <= n2 && data2.rows() == p);

  Eigen::Matrix<RealType, Eigen::Dynamic, Eigen::Dynamic, Options> ans(p, p);
  double ans_jk = 0.0;
  for (int j = 0; j < p; ++j) {
    for (int k = j; k < p; ++k) {
      for (int i = 0; i < n1; ++i)
        ans_jk += (data1.coeffRef(i, j) - meanVec1[j]) *
                  (data1.coeffRef(i, k) - meanVec1[k]);
      for (int i = 0; i < n2; ++i)
        ans_jk += (data2.coeffRef(i, j) - meanVec2[j]) *
                  (data2.coeffRef(i, k) - meanVec2[k]);
      ans.coeffRef(j, k) = ans.coeffRef(k, j) = ans_jk / (n - 2);
    }
  }
  return ans;

  // return static_cast<double>(n1 - 1)/(n - 2)*covMat(data1, meanVec1) +
  // static_cast<double>(n2 - 1)/(n - 2)*covMat(data2, meanVec2);
}

template <typename RealType, int Options>
RealType correlationCoefficient(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &v1,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &v2) {
  int l = v1.size();
  // SDMAssert(v2.size() == l);
  RealType m1 = v1.mean();
  RealType m2 = v2.mean();
  RealType enumerator = 0.0, denominator1 = 0.0, denominator2 = 0.0;
  for (int i = 0; i < l; i++) {
    RealType tmp1 = v1[i] - m1;
    RealType tmp2 = v2[i] - m2;
    enumerator += tmp1 * tmp2;
    denominator1 += tmp1 * tmp1;
    denominator2 += tmp2 * tmp2;
  }
  return enumerator / sqrt(denominator1) / sqrt(denominator2);
}

template <typename RealType, int Options>
Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> correlationCoefficientTest(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &v1,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &v2) {
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> ans(3);
  RealType r = ans[0] = correlationCoefficient(v1, v2);
  RealType df = static_cast<RealType>(v1.size() - 2.0);
  RealType t = ans[1] = r * sqrt(df) / sqrt(1.0 - r * r);
  boost::math::students_t_distribution<RealType> st_dstr(df);
  ans[2] = 1.0 - boost::math::cdf(st_dstr, t);
  return ans;
}

template <typename RealType, int Options>
Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> extractEqualIndices(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &v,
    const RealType &val) {
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> ans(v.size());
  int count = 0;
  for (int i = 0; i < v.size(); ++i)
    if (val == v[i])
      ans[count++] = i;
  ans.conservativeResize(count);
  return ans;
}

template <typename RealType, int Options>
Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> doubleMeanRank(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &v,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &rank) {
  int l = v.size();
  // SDMAssert(l == rank.size());
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> dRank(l);
  for (int i = 0; i < l; ++i) {
    Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> tmpVec1 =
        extractEqualIndices(v, v[i]);
    int len = tmpVec1.size();
    if (len > 1) {
      Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> tmpVec2(len);
      for (int j = 0; j < len; ++j)
        tmpVec2[j] = rank[tmpVec1[j]];

      dRank[i] = tmpVec2.mean();
    } else {
      dRank[i] = rank[i];
    }
  }
  return dRank;
}

template <typename RealType, int Options>
Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> argsortDecOrder(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {

  using pair_sort = std::pair<int, RealType>;
  std::vector<pair_sort> sort_vec;
  for (int i = 0; i < vec.size(); ++i)
    sort_vec.push_back(i, vec[i]);
  std::sort(sort_vec.begin(), sort_vec.end(), lessPair);
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> ans(vec.size());
  for (int i = 0; i < vec.size(); ++i)
    ans[i] = (sort_vec[i])->first;
  return ans;
}

template <typename RealType, int Options>
Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options>
rankDecOrder(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &vec) {
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> idx =
      argsortDecOrder(vec);
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> ans(vec.size());
  for (int i = 0; i < vec.size(); ++i)
    ans[idx[i]] = i;
  return ans;
}

template <typename RealType, int Options>
double spearmanRankCorrelation(
    Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &v1,
    Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &v2) {
  int l = v1.size();
  // SDMAssert(v2.size() == l);
  // Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> rank1 =
  // rankDecOrder(v1);
  // Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> rank2 =
  // rankDecOrder(v2);
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> dRank1 =
      doubleMeanRank(v1, rankDecOrder(v1));
  Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> dRank2 =
      doubleMeanRank(v2, rankDecOrder(v2));
  double sum = 0.0;
  for (int i = 0; i < l; ++i) {
    double tmp_i = (dRank1[i] - dRank2[i]);
    sum += tmp_i * tmp_i;
  }
  return 1.0 - 6.0 * sum / (double)(l * (l * l - 1));
}

template <typename RealType, int Options>
double mixQuantile(const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &v,
                   const double order) {
  // SDMAssert(0.0 <= order && order <= 1.0);
  int numKernel = v.size() / 3;
  // SDMAssert(numKernel > 3);
  double ans = 0.0;
  boost::math::normal_distribution<double> n_dstr(0.0, 1.0);
  for (int i = 0; i < numKernel; ++i)
    ans +=
        (boost::math::quantile(n_dstr, order) * v[i * 3 + 1] + v[i * 3 + 2]) *
        v[i * 3];

  return ans;
}

template <typename RealType, int Options>
double quantileFromNormalMixture(
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &coefficient,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &mu,
    const Eigen::Matrix<RealType, Eigen::Dynamic, 1, Options> &sigma,
    const double order) {
  int numComponent = coefficient.size();
  // SDMAssert(mu.size() == numComponent && sigma.size() == numComponent);
  // SDMAssert(0.0 <= order && order <= 1.0);
  double ans = 0.0;
  boost::math::normal_distribution<double> n_dstr(0.0, 1.0);
  for (int k = 0; k < numComponent; ++k)
    ans += coefficient[k] *
           (boost::math::quantile(n_dstr, order) * sigma[k] + mu[k]);
  return ans;
}

} // namespace sdm

#endif
