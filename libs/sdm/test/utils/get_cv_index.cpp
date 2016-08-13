#include "utils.hpp"

using namespace sdm;
using namespace Eigen;
using namespace std;

int main(int argc, char const *argv[]) {
  int num_ins = 25;
  int num_fold = 3;
  bool random_flag = true;
  bool seed_flag = true;
  std::vector<std::vector<int>> trains, valids;

  get_cross_validation_index(num_fold, num_ins, trains, valids);
  cout << "valids" << endl;
  for (auto &v : valids) {
    cout << "# valid instance = " << v.size() << endl;
    for (auto i : v) {
      cout << i << ", ";
    }
    cout << "\n" << endl;
  }

  cout << "trains" << endl;
  for (auto &v : trains) {
    cout << "# train instance = " << v.size() << endl;
    for (auto i : v) {
      cout << i << ", ";
    }
    cout << "\n" << endl;
  }

  get_cross_validation_index(num_fold, num_ins, trains, valids, random_flag);
  cout << "random valids" << endl;
  for (auto &v : valids) {
    cout << "# valid instance = " << v.size() << endl;
    for (auto i : v) {
      cout << i << ", ";
    }
    cout << "\n" << endl;
  }

  cout << "random trains" << endl;
  for (auto &v : trains) {
    cout << "# train instance = " << v.size() << endl;
    for (auto i : v) {
      cout << i << ", ";
    }
    cout << "\n" << endl;
  }

  get_cross_validation_index(num_fold, num_ins, trains, valids, random_flag, seed_flag);
  cout << "random2 valids" << endl;
  for (auto &v : valids) {
    cout << "# valid instance = " << v.size() << endl;
    for (auto i : v) {
      cout << i << ", ";
    }
    cout << "\n" << endl;
  }

  cout << "random2 trains" << endl;
  for (auto &v : trains) {
    cout << "# train instance = " << v.size() << endl;
    for (auto i : v) {
      cout << i << ", ";
    }
    cout << "\n" << endl;
  }
  return 0;
}
