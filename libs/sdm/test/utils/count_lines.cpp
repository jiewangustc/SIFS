#include "utils.hpp"

using namespace sdm;
using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[]) {
  std::string file_name = argv[1];
  cout << count_lines(file_name) <<endl;
  return 0;
}
