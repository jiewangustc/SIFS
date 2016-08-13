#include "utils.hpp"

using namespace sdm;
using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[]) {
  MatrixXd a = MatrixXd::Random(3,3);
  a.coeffRef(1,1) = 3.0;
  int numeq = compare(a, 3.0);
  cout << "num of equal value " << numeq <<endl;
  VectorXd v = VectorXd::Ones(5);
  numeq = compare(v, 1.0);
  cout << "num of equal value " << numeq <<endl;
}
