# Scaling Up Sparse Support Vector Machine by Simultaneous Feature and Sample Reduction
## About
This is the implementation of [Scaling Up Sparse Support Vector Machine by Simultaneous Feature and Sample Reduction](https://arxiv.org/abs/1607.06996v1) . We wrote the code in C++ along with the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) library for some numerical computations.


The goal of this project is to accelerate sparse SVM training by indentifying the inactive features and samples simultaneously. 


It is extremely efficient in dealing with big data problems, such as kddb with about 20 million of samples and 30 million of features. We can speed up the training process by 200-300 times on many real datasets.

## Usage
### Support platforms and Enviromental Requirement
```
Linux
gcc version > 4.8.0
cmake version > 2.8.12
```
### Compile
```
cd test
make
```

### Example
```
cd test
./start/ train_file_name -task=task_type -br.ub=1.0 -br.lb=0.05 -b.ns=10 -ar.ub=1.0 -ar.lb=0.01 -a.ns=100 -max.iter=10000 -tol=1e-9

option:
train_file_name: training data (libsvm format), e.g., [rcv1_train](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary)
  task_type = 0 : don't perform screening
            = 1 : both inactive feature and smaple screening
            = 2 : only inactive sample screening
            = 3 : only inactive feature screening
```
To see all the options, use
```
./start
```

### Acknowledgement 

We would like to acknowledge  the [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) library and the recent work [s3fs] (https://github.com/husk214/s3fs).

### Related Papers
```
@article{zhang2016scaling,
  title={Scaling Up Sparse Support Vector Machine by Simultaneous Feature and Sample Reduction},
  author={Zhang, Weizhong and Hong, Bin and Ye, Jieping and Cai, Deng and He, Xiaofei and Wang, Jie},
  journal={arXiv preprint arXiv:1607.06996},
  year={2016}
}
```
