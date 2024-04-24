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
`bdd_solver_cl ${config.json}` 
where ${config.json} is a json configuration file.

It is structured as follows:
```
{
    "input": "${input file}",
    "variable order": "{input|bfs|minimum degree|cuthill}",
    "normalize constraints": "{true|false}",
    "precision": "{float|double}",
    "relaxation solver": "{sequential mma|parallel mma|cuda parallel mma|lbfgs parallel mma|cuda lbfgs parallel mma|subgradient}",
    "termination criteria": {
        "maximum iterations": ${integer},
        "improvement slope": ${real number},
        "minimum improvement": ${real number},
        "time limit": ${integer}
    },
    "perturbation rounding": {
        "initial perturbation": ${real number},
        "perturbation growth rate": ${real number},
        "inner iterations": ${integer},
        "outer iterations": ${integer}
    }
}
```

* `input`: File containing the optimization problem in .lp or .opb format, argument required.
* `variable order`: The order of optimization variables. Possible values are 
    * `input`: As encountered in the input file, this is the default.
    * `bfs`: Use a breadth-first search through the variable-constraint adjacency matrix to determine a variable ordering starting from the most eccentric node.
    * `minimum degree`: Use the minimum degree ordering.
    * `cuthill`: Use the Cuthill McKee algorithm on the variable-constraint adjacency matrix to determina a variable ordering.
* `normalize constraints`: Should variables in constraints be sorted according to the variable order? This argument is not required and defaults to true.
* `precision`: Can be either float or double precision for all floating point computations. The argument is not required and defaults to double.
* `relaxation solver`: Can be one of the following:
    * `sequential mma` for sequential min-marginal averaging [1].
    * `parallel mma` for parallel CPU deferred min-marginal averaging [2].
    * `cuda parallel mma` for parallel deferred min-marginal averaging on GPU (available if built with `WITH_CUDA=ON`) [2].
    * `lbfgs parallel mma` for L-BFGS using the parallel_mma [2] CPU solver as backbone. 
    * `lbfgs cuda parallel mma` for L-BFGS using the mma_cuda [2] GPU solver as backbone (available if built with `WITH_CUDA=ON`).
    * `subgradient` for subgradient ascent with adaptive step sizes.
* `termination criteria`: Terminate the relaxation optimization if either of the below stopping criteria is satisfied. The argument is not required.
    * `maximum iterations`: For terminating after a pre-specified number of iterations, default value 1000.
    * `improvement slope`: For terminating if improvement between iterations is less than fraction of the improvement after the first iteration, default value 1e-6.
    * `minimum improvement`: For terminating if improvement between iterations is less than the specified value, default value 1e-6.
    * `time limit`: For terminating if optimization takes longer than value in seconds, default value 3600.
* `perturbation rounding`: Compute primal solution by perturbing costs such that the relaxation solver solution becomes integral.
    * `initial perturbation`: By how much should costs be perturbed, default value 0.1.
    * `perturbation growth rate`: The factor specifying by how much perturbation should be increased in each perturbation round, default value 1.1.
    * `inner iterations`: For how many iterations should the relaxation solver run between perturbing costs, default value 100.
    * `outer iterations`: How many perturbation rounds should be performed, default value 100.
* `lbfgs`: If a LBFG-S solver is chosen, the following non-required parameters can be passed:
    * `history size`: how many past iterates should be used, default value 5.
    * `initial step size`: the initial step size for the LBFG-S step, default value 1e-6.
    * `required relative lb increase`: the required relative increase in the lower bound for a step to be considered successful, default value 1e-6.
    * `step size decrease factor`: the factor by which to decrease the step size if a step is unsuccessful, default value 0.8.
    * `step size increase factor`: the factor by which to increase the step size if a step is successful, default value 1.1.

### Python interface

All solvers are exposed to Python. To install Python solver do:

```bash
git clone git@github.com:LPMP/BDD.git
cd BDD
python setup.py install
```
To use Python solver only on CPU (e.g. GPU not available) replace last command by
```bash
WITH_CUDA=OFF python setup.py install
```

For running the solver via Python interface do:

```
from BDD.bdd_solver_py import bdd_solver as bdd_solver

solver = bdd_solver(input file)
solver.solve()
```

For more information about setting-up the solver especially from Python see this [guide](https://paulroetzer.github.io/posts/how-to-use-fastdog/).
The python interface is exposed via [bdd_solver_py.py](src/bdd_solver_py.py) and one example of use is in [test_bdd_solver_py.py](test/test_bdd_solver_py.py).

## Learned solver (DOGE-Train)
Please navigate to `DOGE` sub-folder.

## References
If you use this work please cite
* [`[1] - J. H. Lange and P. Swoboda. Efficient Message Passing for 0–1 ILPs with Binary Decision Diagrams. In ICML 2021.`](http://proceedings.mlr.press/v139/lange21a.html)
* [`[2] - A. Abbas and P. Swoboda. FastDOG: Fast Discrete Optimization on GPU. In CVPR 2022.`](https://arxiv.org/abs/2111.10270)
for the parallel solvers,
* [`[3] - A. Abbas and P. Swoboda. DOGE-Train: Discrete Optimization on GPU with End-to-end Training. In AAAI 2024.`](https://arxiv.org/abs/2205.11638)
for learned solvers, 
* [`[4] - Roetzer, Paul, et al. Fast Discrete Optimisation for Geometrically Consistent 3D Shape Matching. arXiv preprint arXiv:2310.08230 (2023).`](https://arxiv.org/abs/2310.08230) for LBFGS based parallel solvers.