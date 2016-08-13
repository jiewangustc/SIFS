#include "io.hpp"

using namespace sdm;
using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[]) {
  if (argc > 4) {
    std::string file_name = argv[1];
    int sub_num_ins = atof(argv[2]);
    int sub_num_fea = atof(argv[3]);
    bool random_seed_flag = static_cast<bool>(atoi(argv[4]));
    std::string output_file =
        file_name + "_" + to_string(sub_num_ins) + "_" + to_string(sub_num_fea);
    if (argc > 5)
      output_file = (argv[5]);
    Eigen::SparseMatrix<double, 1> sx;
    Eigen::ArrayXd y;
    load_libsvm_subsampling(sx, y, file_name, sub_num_ins, sub_num_fea,
                            random_seed_flag);
    save_libsvm(sx, y, output_file);
  }
  return 0;
}
