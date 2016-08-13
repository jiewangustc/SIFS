#ifndef SDM_ALIAS_HPP_
#define SDM_ALIAS_HPP_

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>

namespace sdm {

template <class T, int _Option>
using sm_iit = typename Eigen::SparseMatrix<T, _Option, std::ptrdiff_t>::InnerIterator;

template <class T>
using vector_pair = std::pair<std::vector<T>, std::vector<T>>;

}

#endif
