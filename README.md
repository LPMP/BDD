[![Build Status](https://travis-ci.com/LPMP/BDD.svg?branch=main)](https://travis-ci.com/LPMP/BDD)

# BDD

An integer linear program solver using a Lagrange decomposition into binary decision diagrams. Lagrange multipliers are updated through min-marginal averaging (a form of dual block coordinate ascent). Sequential and parallel CPU solvers are provided as well as a massively parallel GPU implementation.

## Installation

`git clone https://github.com/LPMP/BDD`

Then continue with creating a build folder and use cmake:

`mkdir build && cd build && cmake ..`

If CUDA-solvers are to be built, set `WITH_CUDA=ON` in cmake and ensure CUDA is available (tested on CUDA 11.2, later versions should also work).

## Command Line Usage

Given an input file ${input} in LP format, one can solve the problem via
`bdd_solver_cl -i ${input} -s ${solver}` 
where ${solver} is one of

* `mma` for sequential min-marginal averaging [1].
* `parallel_mma` for parallel CPU deferred min-marginal averaging [2].
* `mma_cuda` for parallel deferred min-marginal averaging on GPU (available if built with `WITH_CUDA=ON`) [2].
* `hybrid_parallel_mma` for parallel deferred min-marginal averaging [2] on CPU and GPU simultaneously (available if built with `WITH_CUDA=ON`). This solver might be faster when a few long constraints are present that would constitute a sequential bottleneck for the pure GPU solver.
* `subgradient` for subgradient ascent with adaptive step sizes.
* `lbfgs_parallel_mma` for L-BFGS using the parallel_mma [2] CPU solver as backbone. 
* `lbfgs_cuda_mma` for L-BFGS using the mma_cuda [2] GPU solver as backbone (available if built with `WITH_CUDA=ON`).

### Primal Rounding

In order to compute a primal solution from the dual one obtained through a min-marginal averaging scheme, we provide a perturbation based rounding heuristic

* `--incremental_primal`: Perturb costs iteratively to drive variables towards integrality. Parameters for this scheme are
    * `--incremental_initial_perturbation ${p}`: The initial perturbation magnitude.
    * `--incremental_perturbation_growth_rate ${x}`: The growth rate for increasing the perturbation after each round.

### Termination Criteria

For terminating dual optimization we provide three stopping criteria:

* `--max_iter ${max_iter}`: For terminating after a pre-specified number of iterations.
* `--improvement_slope ${p}$`: For terminating if improvement between iterations is less than ${p} of the improvement after the first iteration.
* `--tolerance ${p}$`: For terminating if improvement between iterations is less than ${p} of the initial lower bound.

### Variable Ordering

For computing BDDs for representing constraints and for sequentially visiting variables in the `mma` solver the variable order can be specified.

* `-o input`: Use the variable ordering as given in the input file.
* `-o bfs`: Use a breadth-first search through the variable-constraint adjacency matrix to determine a variable ordering starting from the most eccentric node.
* `-o cuthill`: Use the Cuthill McKee algorithm on the variable-constraint adjacency matrix to determina a variable ordering.

### Python interface

All solvers are exposed to Python. To install Python solver do:

```bash
git clone git@github.com:LPMP/BDD.git
cd BDD
python setup.py install
```
For information about Python interface see [test_bdd_solver_py.py](test/test_bdd_solver_py.py).

## Learned solver (DOGE-Train)
Please navigate to `DOGE` sub-folder.

## References
If you use this work please cite
* [`[1] - J. H. Lange and P. Swoboda. Efficient Message Passing for 0–1 ILPs with Binary Decision Diagrams. In ICML 2021.`](http://proceedings.mlr.press/v139/lange21a.html)
* [`[2] - A. Abbas and P. Swoboda. FastDOG: Fast Discrete Optimization on GPU. In CVPR 2022.`](https://arxiv.org/abs/2111.10270)
for the parallel solvers, and 
* [`[3] - A. Abbas and P. Swoboda. DOGE-Train: Discrete Optimization on GPU with End-to-end Training. In AAAI 2024.`](https://arxiv.org/abs/2205.11638)
for learned solvers.
