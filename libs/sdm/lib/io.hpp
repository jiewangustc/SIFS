#ifndef SDM_IO_HPP_
#define SDM_IO_HPP_

#include "utils.hpp"

namespace sdm {

// for readline (used in load libsvm)
static char *sdm_line = nullptr;
static int sdm_max_line_len;
static inline char *readline(FILE *input) {
  if (fgets(sdm_line, sdm_max_line_len, input) == nullptr)
    return nullptr;

  while (strrchr(sdm_line, '\n') == nullptr) {
    sdm_max_line_len *= 2;
    sdm_line = (char *)realloc(sdm_line, sdm_max_line_len);
    int len = (int)strlen(sdm_line);
    if (fgets(sdm_line + len, sdm_max_line_len - len, input) == nullptr)
      break;
  }

  return sdm_line;
}

//////////////////////////////////////////
//             File IO
//////////////////////////////////////////
template <typename ValueType, int _Cols>
bool save(const Eigen::Matrix<ValueType, Eigen::Dynamic, _Cols> &mat,
          const std::string &file_name, const std::string &separater = " ",
          const std::string &header = "", const bool &size_info = false) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "cannot open the file for writing in save\n";
    return false;
  }
  if (header != "")
    output_file << header << std::endl;
  if (size_info) {
    output_file << "# " << mat.rows();
    if (mat.cols() != 1)
      output_file << " " << mat.cols();

    output_file << std::endl;
  }
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < _Cols; ++j)
      output_file << mat.coeffRef(i, j) << separater;
    output_file << std::endl;
  }
  return true;
}

template <typename ValueType, int Major, typename Index>
bool save(const Eigen::SparseMatrix<ValueType, Major, Index> &spa_mat,
          const std::string &file_name, const bool &size_info = false) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "cannot open the file for writing in save\n";
    return false;
  }
  if (size_info)
    output_file << "# " << spa_mat.rows() << " " << spa_mat.cols() << std::endl;

  using InnerIterator = typename Eigen::SparseMatrix<ValueType, Eigen::RowMajor,
                                                     Index>::InnerIterator;
  bool line_break_flag;
  for (int i = 0; i < spa_mat.rows(); ++i) {
    line_break_flag = false;
    for (InnerIterator it(spa_mat, i); it;) {
      output_file << it.index() << ":" << it.value();
      ++it;
      if (it)
        output_file << " ";
      line_break_flag = true;
    }
    if (line_break_flag)
      output_file << std::endl;
  }
  return true;
}

template <typename ValueType, int _Cols, int _Major, typename Index,
          typename ValueType1>
bool save_libsvm(
    const Eigen::Matrix<ValueType, Eigen::Dynamic, _Cols, _Major> &mat,
    const Eigen::Array<ValueType1, Eigen::Dynamic, 1> &y,
    const std::string &file_name, const bool &size_info = false) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "error: in save_libsvm (cannot open the output_file)\n";
    return false;
  }
  if (mat.rows() != y.rows()) {
    std::cerr << "error: in save_libsvm (mat size isn't equal y size)\n";
  }
  if (size_info)
    output_file << "# " << mat.rows() << " " << mat.cols() << std::endl;

  const std::string space = " ";
  const std::string colon = ":";
  for (int i = 0; i < mat.rows(); ++i) {
    output_file << y.coeffRef(i);
    for (int j = 0; j < mat.cols(); ++j)
      output_file << space << j + 1 << colon << mat.coeffRef(i, j);
    output_file << std::endl;
  }
  return true;
}

template <typename ValueType, int Major, typename Index, typename ValueType1>
bool save_libsvm(const Eigen::SparseMatrix<ValueType, Major, Index> &spa_mat,
                 const Eigen::Array<ValueType1, Eigen::Dynamic, 1> &y,
                 const std::string &file_name, const bool &size_info = false) {
  std::ofstream output_file(file_name);
  if (!output_file.is_open()) {
    std::cerr << "error: in save_libsvm (cannot open the output_file)\n";
    return false;
  }
  if (spa_mat.rows() != y.rows()) {
    std::cerr << "error: in save_libsvm (spa_mat size isn't equal y size)\n";
  }
  if (size_info)
    output_file << "# " << spa_mat.rows() << " " << spa_mat.cols() << std::endl;

  using InnerIterator = typename Eigen::SparseMatrix<ValueType, Eigen::RowMajor,
                                                     Index>::InnerIterator;
  bool line_break_flag;
  const std::string space = " ";
  const std::string colon = ":";
  for (int i = 0; i < spa_mat.rows(); ++i) {
    line_break_flag = false;
    output_file << y.coeffRef(i) << space;
    for (InnerIterator it(spa_mat, i); it;) {
      output_file << it.index()+1 << colon << it.value();
      ++it;
      if (it)
        output_file << space;
      line_break_flag = true;
    }
    if (line_break_flag)
      output_file << std::endl;
  }
  return true;
}

template <typename ValueType, int Major>
bool load(Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic, Major> &mat,
          const std::string &file_name) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  readline(fp);
  std::string buf = sdm_line;
  std::vector<std::string> vec1;
  vec1 = split_string(buf, " ");
  if (vec1.at(0) != "#") {
    std::vector<ValueType> tmp_vec;
    int n = 1, k = 0;
    while (1) {
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;
      tmp_vec.push_back(naive_atot<ValueType>(val));
      ++k;
    }
    while (readline(fp) != nullptr) {
      while (1) {
        char *val = strtok(nullptr, " \t");
        if (val == nullptr)
          break;
        tmp_vec.push_back(naive_atot<ValueType>(val));
      }
      ++n;
    }
    mat = Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>>(&tmp_vec[0], n, k);
  } else if (vec1.size() != 3) {
    std::cerr << "file's style error in load matrix" << std::endl;
    return false;
  } else {
    int x_rows = str2int(vec1.at(1));
    int x_cols = str2int(vec1.at(2));
    mat.resize(x_rows, x_cols);
    int n = 0, k = 0;
    while (readline(fp) != nullptr) {
      strtok(sdm_line, " \t");
      k = 0;
      while (1) {
        char *val = strtok(nullptr, " \t");
        if (val == nullptr)
          break;
        mat.coeffRef(n, k) = naive_atot<ValueType>(val);
        ++k;
      }
      ++n;
    }
  }
  fclose(fp);
  return true;
}

template <typename ValueType>
bool load(Eigen::Matrix<ValueType, Eigen::Dynamic, 1> &vec,
          const std::string &file_name) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  std::string buf, label_buf, tmp_st1, tmp_st2;
  std::getline(fs, buf);
  std::vector<std::string> vec1;
  vec1 = split_string(buf, " ");

  if (vec1.at(0) != "#") {
    std::vector<ValueType> tmp_vec;
    tmp_vec.push_back(naive_atot<ValueType>(buf));
    int n = 1;
    for (; std::getline(fs, buf); ++n)
      tmp_vec.push_back(naive_atot<ValueType>(buf));

    vec =
        Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>>(&tmp_vec[0], n);
  } else if (vec1.size() != 2) {
    std::cerr << "file's style error in load vector" << std::endl;
    return false;
  } else {
    int x_rows = str2int(vec1.at(1));
    vec.resize(x_rows, 1);
    int n = 0;
    for (; std::getline(fs, buf); ++n)
      vec.coeffRef(n, 0) = naive_atot<ValueType>(buf);
  }
  fs.close();
  return true;
}

template <typename ValueType>
bool load(Eigen::Array<ValueType, Eigen::Dynamic, 1> &vec,
          const std::string &file_name) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  std::string buf, label_buf, tmp_st1, tmp_st2;
  std::getline(fs, buf);
  std::vector<std::string> vec1;
  vec1 = split_string(buf, " ");

  if (vec1.at(0) != "#") {
    std::vector<ValueType> tmp_vec;
    tmp_vec.push_back(naive_atot<ValueType>(buf));
    int n = 1;
    for (; std::getline(fs, buf); ++n)
      tmp_vec.push_back(naive_atot<ValueType>(buf));

    vec =
        Eigen::Map<Eigen::Matrix<ValueType, Eigen::Dynamic, 1>>(&tmp_vec[0], n);
  } else if (vec1.size() != 2) {
    std::cerr << "file's style error in load vector" << std::endl;
    return false;
  } else {
    int x_rows = str2int(vec1.at(1));
    vec.resize(x_rows, 1);
    int n = 0;
    for (; std::getline(fs, buf); ++n)
      vec.coeffRef(n, 0) = naive_atot<ValueType>(buf);
  }
  fs.close();
  return true;
}

template <typename ValueType, int Major, typename Index>
bool load(Eigen::SparseMatrix<ValueType, Major, Index> &spa_mat,
          const std::string &file_name) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  using Tri = Eigen::Triplet<ValueType>;
  std::vector<Tri> tripletList;
  tripletList.reserve(1024);

  std::string buf;
  int n, d, k;
  n = d = 0;
  std::string::size_type idx0 = static_cast<std::string::size_type>(0);
  std::string::size_type idx1 = idx0, idx2 = idx0;
  double tmp = 0;
  while (std::getline(fs, buf)) {
    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cerr << "file format error in load SpaMat" << std::endl;
      return false;
    }
    idx1 = idx0, idx2 = idx0;
    do {
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      k = std::atoi((buf.substr(idx1, idx2 - idx1)).c_str());
      if (d < k)
        d = k;
      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);
      tmp = naive_atot<ValueType>((buf.substr(idx2, idx1 - idx2)).c_str());
      tripletList.push_back(Tri(n, k, tmp));
    } while (idx1 != std::string::npos);
    ++n;
  }
  fs.close();
  ++d;
  spa_mat.resize(n, d);
  spa_mat.setFromTriplets(tripletList.begin(), tripletList.end());
  spa_mat.makeCompressed();
  return true;
}

template <typename ValueType1, int Major, typename Index>
bool load_libsvm(Eigen::SparseMatrix<ValueType1, Major, Index> &spa_x,
                 Eigen::Array<ValueType1, Eigen::Dynamic, 1> &y,
                 const std::string &file_name) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  using Tri = Eigen::Triplet<ValueType1>;
  std::vector<Tri> tripletList;
  tripletList.reserve(1024);
  y.resize(1024);

  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  unsigned int n = 0, d = 0, k = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }
    y.coeffRef(n) = naive_atot<ValueType1>(p);

    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;

      tripletList.push_back(Tri(n, k, naive_atot<ValueType1>(val)));
    }

    if (static_cast<unsigned int>(y.size()) <= (++n))
      y.conservativeResize(y.size() * 2);
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  y.conservativeResize(n);

  spa_x.resize(n, d);
  spa_x.setFromTriplets(tripletList.begin(), tripletList.end());
  spa_x.makeCompressed();
  return true;
}

template <typename ValueType1, int Major>
bool load_libsvm(
    Eigen::Matrix<ValueType1, Eigen::Dynamic, Eigen::Dynamic, Major> &dense_x,
    Eigen::Array<ValueType1, Eigen::Dynamic, 1> &y,
    const std::string &file_name) {

  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  std::vector<ValueType1> x_vec;
  x_vec.reserve(1024); // estimation of non_zero_entries
  y.resize(1024);
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  ValueType1 vt1_0;

  unsigned int n = 0, d = 0, k = 0, pre_k = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }
    y.coeffRef(n) = naive_atot<ValueType1>(p);

    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;
      for (; pre_k < k; ++pre_k)
        x_vec.push_back(vt1_0);

      x_vec.push_back(naive_atot<ValueType1>(val));
      pre_k = k + 1;
    }

    if (static_cast<unsigned int>(y.size()) <= (++n))
      y.conservativeResize(y.size() * 2);
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  y.conservativeResize(n);
  dense_x = Eigen::Map<Eigen::Matrix<ValueType1, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(&x_vec[0], n, k + 1);
  return true;
}

template <typename ValueType1, int Major, typename Index>
bool load_libsvm_subsampling(
    Eigen::SparseMatrix<ValueType1, Major, Index> &spa_x,
    Eigen::Array<ValueType1, Eigen::Dynamic, 1> &y,
    const std::string &file_name, const int &num_ins, const int &num_fea,
    const bool &random_seed_flag = false) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "error: in load_libsvm_subsampling (file open error)\n";
    return false;
  }
  Eigen::Array<ValueType1, Eigen::Dynamic, 1> whole_y;
  using Tri = Eigen::Triplet<ValueType1>;
  std::vector<Tri> tripletList;
  std::vector<std::vector<Tri>> tri_ins;
  tripletList.reserve(1024);
  whole_y.resize(1024);

  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  int n = 0, d = 0, k = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }
    whole_y.coeffRef(n) = naive_atot<ValueType1>(p);
    std::vector<Tri> ins_tri_vec;
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;

      ins_tri_vec.push_back(Tri(n, k, naive_atot<ValueType1>(val)));
    }
    tri_ins.push_back(ins_tri_vec);
    if (static_cast<unsigned int>(whole_y.size()) <= (++n))
      whole_y.conservativeResize(whole_y.size() * 2);
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  whole_y.conservativeResize(n);

  int sub_num_ins = std::min(num_ins, n);
  int sub_num_fea = std::min(num_fea, d);
  y.resize(sub_num_ins);

  std::vector<int> ins_index(n);
  std::iota(std::begin(ins_index), std::end(ins_index), 0);

  std::vector<int> fea_index(d);
  std::iota(std::begin(fea_index), std::end(fea_index), 0);

  std::mt19937 engine;
  std::random_device rnd;
  std::vector<std::uint_least32_t> v(10);
  std::generate(std::begin(v), std::end(v), std::ref(rnd));
  std::seed_seq seed(std::begin(v), std::end(v));
  if (random_seed_flag)
    engine.seed(seed);
  std::shuffle(std::begin(ins_index), std::end(ins_index), engine);

  std::generate(std::begin(v), std::end(v), std::ref(rnd));
  if (random_seed_flag)
    engine.seed(seed);
  std::shuffle(std::begin(fea_index), std::end(fea_index), engine);

  Eigen::VectorXi flag_subsample_fea = Eigen::VectorXi::Zero(d);
  Eigen::VectorXi new_fea_index = Eigen::VectorXi::Zero(d);
  for (int j = 0; j < sub_num_fea; ++j)
    new_fea_index[fea_index[j]] = j;

  for (int j = 0; j < sub_num_fea; ++j)
    flag_subsample_fea[fea_index[j]] = 1;

  for (int i = 0; i < sub_num_ins; ++i) {
    y.coeffRef(i) = whole_y.coeffRef(ins_index[i]);
    for (auto &&j : tri_ins[ins_index[i]]) {
      if (flag_subsample_fea[j.col()])
        tripletList.push_back(Tri(i, new_fea_index[j.col()], j.value()));
    }
  }

  spa_x.resize(sub_num_ins, sub_num_fea);
  spa_x.setFromTriplets(tripletList.begin(), tripletList.end());
  spa_x.makeCompressed();
  return true;
}

template <typename ValueType1, int Major, typename Index>
bool load_libsvm_binary(Eigen::SparseMatrix<ValueType1, Major, Index> &spa_x,
                        Eigen::Array<ValueType1, Eigen::Dynamic, 1> &label,
                        const std::string &file_name) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  using Tri = Eigen::Triplet<ValueType1>;
  std::vector<Tri> tripletList;
  tripletList.reserve(1024);

  ValueType1 label_memo = 0.0, tmp_label = 0.0;
  bool label_flag = false;
  ValueType1 label_true, label_false;
  label_true = static_cast<ValueType1>(1.0);
  label_false = static_cast<ValueType1>(-1.0);

  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));
  label.resize(1024);

  unsigned int n = 0, d = 0, k = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }

    tmp_label = naive_atot<ValueType1>(p);
    if (n == 0)
      label_memo = tmp_label;
    if (label_flag) {
      label.coeffRef(n) = ((tmp_label == label_memo) ? 1.0 : -1.0);
    } else {
      if (label_memo == tmp_label) {
        label.coeffRef(n) = label_true;
      } else {
        if (label_memo > tmp_label) {
          label.coeffRef(n) = label_false;
        } else {
          for (unsigned int i = 0; i < n; ++i)
            label.coeffRef(i) = label_false;
          label.coeffRef(n) = label_true;
          label_memo = tmp_label;
          label_flag = true;
        }
      }
    }
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;

      tripletList.push_back(Tri(n, k, naive_atot<ValueType1>(val)));
    }

    if (static_cast<unsigned int>(label.size()) <= (++n))
      label.conservativeResize(label.size() * 2);
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  label.conservativeResize(n);

  spa_x.resize(n, d);
  spa_x.setFromTriplets(tripletList.begin(), tripletList.end());
  spa_x.makeCompressed();
  return true;
}

template <typename ValueType1, int Major, typename Index>
bool load_libsvm_binary_randomly(
    Eigen::SparseMatrix<ValueType1, Major, Index> &spa_x,
    Eigen::Array<ValueType1, Eigen::Dynamic, 1> &label,
    const std::string &file_name, const bool seed_flag = false) {
  int num_ins = count_lines(file_name);
  std::vector<int> whole_index(num_ins);
  std::iota(std::begin(whole_index), std::end(whole_index), 0);

  std::mt19937 engine;
  if (seed_flag) {
    std::random_device rnd;
    std::vector<std::uint_least32_t> v(10);
    std::generate(std::begin(v), std::end(v), std::ref(rnd));
    std::seed_seq seed(std::begin(v), std::end(v));
    engine.seed(seed);
  }
  std::shuffle(std::begin(whole_index), std::end(whole_index), engine);

  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  using Tri = Eigen::Triplet<ValueType1>;
  std::vector<Tri> tripletList;
  tripletList.reserve(1024);

  ValueType1 label_memo = 0.0, tmp_label = 0.0;
  bool label_flag = false;
  ValueType1 label_true, label_false;
  label_true = static_cast<ValueType1>(1.0);
  label_false = static_cast<ValueType1>(-1.0);

  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));
  label.resize(num_ins);

  unsigned int n = 0, d = 0, k = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }

    tmp_label = naive_atot<ValueType1>(p);
    if (n == 0)
      label_memo = tmp_label;
    if (label_flag) {
      label.coeffRef(whole_index[n]) = ((tmp_label == label_memo) ? 1.0 : -1.0);
    } else {
      if (label_memo == tmp_label) {
        label.coeffRef(whole_index[n]) = label_true;
      } else {
        if (label_memo > tmp_label) {
          label.coeffRef(whole_index[n]) = label_false;
        } else {
          for (unsigned int i = 0; i < n; ++i)
            label.coeffRef(whole_index[i]) = label_false;
          label.coeffRef(whole_index[n]) = label_true;
          label_memo = tmp_label;
          label_flag = true;
        }
      }
    }
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;

      tripletList.push_back(
          Tri(whole_index[n], k, naive_atot<ValueType1>(val)));
    }
    ++n;
  }
  fclose(fp);
  free(sdm_line);
  ++d;

  spa_x.resize(n, d);
  spa_x.setFromTriplets(tripletList.begin(), tripletList.end());
  spa_x.makeCompressed();
  return true;
}

template <typename ValueType1, int Major>
bool load_libsvm_binary(
    Eigen::Matrix<ValueType1, Eigen::Dynamic, Eigen::Dynamic, Major> &dense_x,
    Eigen::Array<ValueType1, Eigen::Dynamic, 1> &label,
    const std::string &file_name) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  std::vector<ValueType1> x_vec;
  x_vec.reserve(1024); // estimation of non_zero_entries
  label.resize(1024);
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  ValueType1 tmp, vt1_0;
  tmp = vt1_0 = static_cast<ValueType1>(0.0);
  ValueType1 label_memo = 0.0, tmp_label = 0.0;
  bool label_flag = false;
  ValueType1 label_true, label_false;
  label_true = static_cast<ValueType1>(1.0);
  label_false = static_cast<ValueType1>(-1.0);

  unsigned int n = 0, d = 0, k = 0, pre_k = 0;
  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }
    tmp_label = naive_atot<ValueType1>(p);
    if (n == 0)
      label_memo = tmp_label;
    if (label_flag) {
      label.coeffRef(n) = ((tmp_label == label_memo) ? 1.0 : -1.0);
    } else {
      if (label_memo == tmp_label) {
        label.coeffRef(n) = (label_true);
      } else {
        if (label_memo > tmp_label) {
          label.coeffRef(n) = (label_false);
        } else {
          for (int i = 0; i < n; ++i)
            label.coeffRef(i) = -1.0;
          label.coeffRef(n) = (label_true);
          label_memo = tmp_label;
          label_flag = true;
        }
      }
    }
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;
      for (; pre_k < k; ++pre_k)
        x_vec.push_back(vt1_0);

      x_vec.push_back(naive_atot<ValueType1>(val));
      pre_k = k + 1;
    }

    if (label.size() <= (++n))
      label.conservativeResize(label.size() * 2);
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  label.conservativeResize(n);
  dense_x = Eigen::Map<Eigen::Matrix<ValueType1, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>>(&x_vec[0], n, k + 1);
  return true;
}

template <typename ValueType1, int Major, typename Index>
bool equal_split_libsvm_binary(
    Eigen::SparseMatrix<ValueType1, Major, Index> &train_X,
    Eigen::ArrayXd &train_y,
    Eigen::SparseMatrix<ValueType1, Major, Index> &valid_X,
    Eigen::ArrayXd &valid_y, const std::string &file_name) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }

  using Tri = Eigen::Triplet<ValueType1>;
  std::vector<Tri> tripletList1, tripletList2;
  tripletList1.reserve(1024);
  tripletList2.reserve(1024);
  train_y.resize(1024);
  valid_y.resize(1024);
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  unsigned int n = 0, d = 0, k = 0;
  ValueType1 label_memo = static_cast<ValueType1>(0.0), label_tmp;
  bool flag_p = true, flag_n = true, flag_x = true;
  // flag_p : true if the previous positive instance was pushed to valid set
  // flag_n : true if the previous negative instance was pushed to valid set
  // flag_x : true if the present instance is pushed to train set

  while (readline(fp) != nullptr) {
    char *p = strtok(sdm_line, " \t\n");
    if (p == nullptr) {
      std::cerr << "error: empty line" << std::endl;
      return false;
    }
    label_tmp = naive_atot<ValueType1>(p);
    if (n == 0)
      label_memo = label_tmp;
    if (label_tmp == label_memo) {
      if (flag_p) {
        train_y.coeffRef(n) = 1.0;
        flag_p = false;
        flag_x = true;
      } else {
        valid_y.coeffRef(n) = 1.0;
        flag_p = true;
        flag_x = false;
      }
    } else {
      if (flag_n) {
        train_y.coeffRef(n) = -1.0;
        flag_n = false;
        flag_x = true;
      } else {
        valid_y.coeffRef(n) = -1.0;
        flag_n = true;
        flag_x = false;
      }
    }
    while (1) {
      char *idx = strtok(nullptr, ":");
      char *val = strtok(nullptr, " \t");
      if (val == nullptr)
        break;

      k = strtol(idx, nullptr, 10) - 1;
      if (d < k)
        d = k;
      if (flag_x) {
        tripletList1.push_back(Tri(n, k, naive_atot<ValueType1>(val)));
      } else {
        tripletList2.push_back(Tri(n, k, naive_atot<ValueType1>(val)));
      }
    }

    if (!flag_x)
      ++n;
    if (train_y.size() <= n) {
      train_y.conservativeResize(train_y.size() * 2);
      valid_y.conservativeResize(valid_y.size() * 2);
    }
  }
  fclose(fp);
  free(sdm_line);
  ++d;
  if (flag_x) {
    train_X.resize(n + 1, d);
    train_y.conservativeResize(n + 1);
  } else {
    train_X.resize(n, d);
    train_y.conservativeResize(n);
  }
  train_X.setFromTriplets(tripletList1.begin(), tripletList1.end());
  train_X.makeCompressed();

  valid_X.resize(n, d);
  valid_y.conservativeResize(n);
  valid_X.setFromTriplets(tripletList2.begin(), tripletList2.end());
  valid_X.makeCompressed();
  return true;
}

// for equal_split_libsvm_binary_for_cross_validation
// - vec_flag control where he present instance should push
static inline int get_to_distribute_index(const std::vector<bool> &vec_flag,
                                          const int &vec_size) {
  for (int i = 0; i < vec_size; ++i) {
    if (vec_flag[i] == true)
      return i;
  }
  std::cerr << "error in get_to_distribute_index" << std::endl;
  return 0;
}

static inline void change_vec_flag(std::vector<bool> &vec_flag,
                                   const int &vec_size, const int &index) {
  vec_flag[index] = false;
  int next_index = index + 1;
  if (next_index == vec_size) {
    vec_flag[0] = true;
  } else {
    vec_flag[next_index] = true;
  }
}

template <typename ValueType1, int Major, typename Index>
bool equal_split_libsvm_binary_for_cross_validation(
    std::vector<Eigen::SparseMatrix<ValueType1, Major, Index>> &vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<ValueType1, Major, Index>> &vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y, const std::string &file_name,
    const int &split_num) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error" << std::endl;
    return false;
  }
  vec_train_x.resize(split_num);
  vec_train_y.resize(split_num);
  vec_valid_x.resize(split_num);
  vec_valid_y.resize(split_num);
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  using Tri = Eigen::Triplet<ValueType1>;
  std::vector<ValueType1> *tmpy_array = new std::vector<ValueType1>[split_num];
  std::vector<Tri> *triplets_array = new std::vector<Tri>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::vector<int> vec_n(split_num, 0);
  std::vector<bool> vec_flag_p(split_num, false);
  std::vector<bool> vec_flag_n(split_num, false);
  int d, k, l, last_index, the_line_index;
  d = k = l = last_index = the_line_index = 0;
  ValueType1 label_memo, label_tmp;
  label_memo = static_cast<ValueType1>(0.0);
  label_tmp = label_memo;
  vec_flag_p.at(0) = true;
  vec_flag_n.at(0) = true;
  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (readline(fp) == nullptr) {
        last_index = i;
        goto LABEL;
      }
      char *p = strtok(sdm_line, " \t\n");
      if (p == nullptr) {
        std::cerr << "error: empty line" << std::endl;
        return false;
      }
      label_tmp = naive_atot<ValueType1>(p);
      if (l == 0)
        label_memo = label_tmp;

      if (label_tmp == label_memo) {
        the_line_index = get_to_distribute_index(vec_flag_p, split_num);
        (tmpy_array[the_line_index]).push_back(1.0);
        change_vec_flag(vec_flag_p, split_num, the_line_index);
      } else {
        the_line_index = get_to_distribute_index(vec_flag_n, split_num);
        (tmpy_array[the_line_index]).push_back(-1.0);
        change_vec_flag(vec_flag_n, split_num, the_line_index);
      }
      while (1) {
        char *idx = strtok(nullptr, ":");
        char *val = strtok(nullptr, " \t");
        if (val == nullptr)
          break;
        k = strtol(idx, nullptr, 10) - 1;
        if (d < k)
          d = k;
        (triplets_array[the_line_index])
            .push_back(
                Tri(vec_n[the_line_index], k, naive_atot<ValueType1>(val)));
      }
      ++vec_n[the_line_index];
      ++l;
    }
  }
LABEL:
  fclose(fp);
  free(sdm_line);
  ++d;
  std::vector<Tri> vec_x;
  std::vector<ValueType1> vec_y;

  for (int i = 0; i < split_num; ++i) {
    vec_train_x[i].resize(l - vec_n[i], d);
    vec_valid_x[i].resize(vec_n[i], d);

    vec_x.clear();
    vec_y.clear();
    int count_cv = 0;
    for (int j = 0; j < split_num; ++j) {
      if (j != i) {
        if (count_cv == 0) {
          vec_x.insert(vec_x.end(), (triplets_array[j]).begin(),
                       (triplets_array[j]).end());
        } else {
          for (auto it : (triplets_array[j]))
            vec_x.push_back(Tri((it.row() + count_cv), it.col(), it.value()));
        }
        vec_y.insert(vec_y.end(), (tmpy_array[j]).begin(),
                     (tmpy_array[j]).end());
        count_cv += (tmpy_array[j]).size();
      }
    }
    (vec_train_x[i]).setFromTriplets(vec_x.begin(), vec_x.end());
    (vec_valid_x[i])
        .setFromTriplets((triplets_array[i]).begin(),
                         (triplets_array[i]).end());

    (vec_train_x[i]).makeCompressed();
    (vec_valid_x[i]).makeCompressed();

    (vec_train_y[i]) = Eigen::Map<Eigen::ArrayXd>(&(vec_y)[0], (vec_y).size());
    (vec_valid_y[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  delete[] triplets_array;
  delete[] tmpy_array;
  return true;
}

template <typename ValueType1, int Major, typename Index>
bool merge_equal_split_libsvm_binary_for_cross_validation(
    std::vector<Eigen::SparseMatrix<ValueType1, Major, Index>> &vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<ValueType1, Major, Index>> &vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y, const std::string &file_name1,
    const std::string &file_name2, const int &split_num) {
  FILE *fp = fopen(file_name1.c_str(), "r");
  if (fp == nullptr) {
    std::cerr << "file open error file1" << std::endl;
    return false;
  }
  FILE *fp2 = fopen(file_name2.c_str(), "r");
  if (fp2 == nullptr) {
    std::cerr << "file open error file2" << std::endl;
    return false;
  }
  vec_train_x.resize(split_num);
  vec_train_y.resize(split_num);
  vec_valid_x.resize(split_num);
  vec_valid_y.resize(split_num);
  sdm_max_line_len = 1024;
  sdm_line = (char *)malloc(sdm_max_line_len * sizeof(char));

  using Tri = Eigen::Triplet<ValueType1>;
  std::vector<ValueType1> *tmpy_array = new std::vector<ValueType1>[split_num];
  std::vector<Tri> *triplets_array = new std::vector<Tri>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::vector<int> vec_n(split_num, 0);
  std::vector<bool> vec_flag_p(split_num, false);
  std::vector<bool> vec_flag_n(split_num, false);
  int d, k, l, last_index, the_line_index;
  d = k = l = last_index = the_line_index = 0;
  ValueType1 label_memo, label_tmp;
  label_memo = static_cast<ValueType1>(0.0);
  label_tmp = label_memo;
  vec_flag_p.at(0) = true;
  vec_flag_n.at(0) = true;
  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (readline(fp) == nullptr) {
        last_index = i;
        goto LABEL;
      }
      char *p = strtok(sdm_line, " \t\n");
      if (p == nullptr) {
        std::cerr << "error: empty line" << std::endl;
        return false;
      }
      label_tmp = naive_atot<ValueType1>(p);
      if (l == 0)
        label_memo = label_tmp;

      if (label_tmp == label_memo) {
        the_line_index = get_to_distribute_index(vec_flag_p, split_num);
        (tmpy_array[the_line_index]).push_back(1.0);
        change_vec_flag(vec_flag_p, split_num, the_line_index);
      } else {
        the_line_index = get_to_distribute_index(vec_flag_n, split_num);
        (tmpy_array[the_line_index]).push_back(-1.0);
        change_vec_flag(vec_flag_n, split_num, the_line_index);
      }
      while (1) {
        char *idx = strtok(nullptr, ":");
        char *val = strtok(nullptr, " \t");
        if (val == nullptr)
          break;
        k = strtol(idx, nullptr, 10) - 1;
        if (d < k)
          d = k;
        (triplets_array[the_line_index])
            .push_back(
                Tri(vec_n[the_line_index], k, naive_atot<ValueType1>(val)));
      }
      ++vec_n[the_line_index];
      ++l;
    }
  }
LABEL:
  fclose(fp);
  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (readline(fp2) == nullptr) {
        last_index = i;
        goto LABEL2;
      }
      char *p = strtok(sdm_line, " \t\n");
      if (p == nullptr) {
        std::cerr << "error: empty line" << std::endl;
        return false;
      }
      label_tmp = naive_atot<ValueType1>(p);

      if (label_tmp == label_memo) {
        the_line_index = get_to_distribute_index(vec_flag_p, split_num);
        (tmpy_array[the_line_index]).push_back(1.0);
        change_vec_flag(vec_flag_p, split_num, the_line_index);
      } else {
        the_line_index = get_to_distribute_index(vec_flag_n, split_num);
        (tmpy_array[the_line_index]).push_back(-1.0);
        change_vec_flag(vec_flag_n, split_num, the_line_index);
      }
      while (1) {
        char *idx = strtok(nullptr, ":");
        char *val = strtok(nullptr, " \t");
        if (val == nullptr)
          break;
        k = strtol(idx, nullptr, 10) - 1;
        if (d < k)
          d = k;
        (triplets_array[the_line_index])
            .push_back(
                Tri(vec_n[the_line_index], k, naive_atot<ValueType1>(val)));
      }
      ++vec_n[the_line_index];
      ++l;
    }
  }
LABEL2:
  fclose(fp2);

  ++d;
  std::vector<Tri> vec_x;
  std::vector<ValueType1> vec_y;

  for (int i = 0; i < split_num; ++i) {
    vec_train_x[i].resize(l - vec_n[i], d);
    vec_valid_x[i].resize(vec_n[i], d);

    vec_x.clear();
    vec_y.clear();
    int count_cv = 0;
    for (int j = 0; j < split_num; ++j) {
      if (j != i) {
        if (count_cv == 0) {
          vec_x.insert(vec_x.end(), (triplets_array[j]).begin(),
                       (triplets_array[j]).end());
        } else {
          for (auto it : (triplets_array[j]))
            vec_x.push_back(Tri((it.row() + count_cv), it.col(), it.value()));
        }
        vec_y.insert(vec_y.end(), (tmpy_array[j]).begin(),
                     (tmpy_array[j]).end());
        count_cv += (tmpy_array[j]).size();
      }
    }
    (vec_train_x[i]).setFromTriplets(vec_x.begin(), vec_x.end());
    (vec_valid_x[i])
        .setFromTriplets((triplets_array[i]).begin(),
                         (triplets_array[i]).end());

    (vec_train_x[i]).makeCompressed();
    (vec_valid_x[i]).makeCompressed();

    (vec_train_y[i]) = Eigen::Map<Eigen::ArrayXd>(&(vec_y)[0], (vec_y).size());
    (vec_valid_y[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  free(sdm_line);
  delete[] triplets_array;
  delete[] tmpy_array;
  return true;
}

} // namespace sdm

#endif
