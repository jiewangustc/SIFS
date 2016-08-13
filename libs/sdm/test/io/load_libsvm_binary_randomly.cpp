#include "io.hpp"

using namespace sdm;
using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[]) {
  std::string file_name = argv[1];
  Eigen::SparseMatrix<double, 1> sx;
  Eigen::ArrayXd y;
  load_libsvm_binary_randomly(sx, y,file_name);
  for (int i = 0; i < sx.rows(); ++i)
  {
    cout << y[i] <<" " << sx.row(i) <<endl;
  }
  return 0;
}
