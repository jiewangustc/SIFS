# SIFS: Scaling Up Sparse Support Vector Machine by Simultaneous Feature and Sample Reduction
## About
This is the implementation of "Scaling Up Sparse Support Vector Machine by Simultaneous Feature and Sample Reduction". 

## Usage
### Compile
```
cd test
make .
```

### Example
```
cd test
./start/ train_file_name -task=task_type -br.ub=1.0 -br.lb=0.05 -b.ns=10 -ar.ub=1.0 -ar.lb=0.01 -a.ns=100 -max.iter=10000 -tol=1e-9

option:
  task_type = 0 : don't perform screening
            = 1 : both inactive feature and smaple screening
            = 2 : only inactive sample screening
            = 3 : only inactive feature screening
```
