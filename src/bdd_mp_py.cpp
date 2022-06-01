#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "bdd_parallel_mma_base.h"
#include "bdd_branch_instruction.h"
#include "ILP_input.h"
#include "bdd_preprocessor.h"

namespace py=pybind11;

using bdd_base_type = LPMP::bdd_parallel_mma_base<LPMP::bdd_branch_instruction<float,uint16_t>>;

PYBIND11_MODULE(bdd_mp_py, m) {
    m.doc() = "Python binding for solution of bdd-based message passing";

    py::class_<bdd_base_type>(m, "bdd_mp")
        .def(py::init([](const LPMP::ILP_input& ilp) {
                    LPMP::bdd_preprocessor bdd_pre(ilp);
                    auto* base = new bdd_base_type(bdd_pre.get_bdd_collection()); 
                    base->update_costs(ilp.objective().begin(), ilp.objective().begin(), ilp.objective().begin(), ilp.objective().end());
                    return base;
                    }))
    .def("min_marginals", [](bdd_base_type& base) { return base.min_marginals_stacked(); })
    .def("update_costs", [](bdd_base_type& base, const Eigen::Matrix<float, Eigen::Dynamic, 2>& delta) { return base.update_costs(delta); })
    .def("update_costs", [](bdd_base_type& base, const Eigen::Matrix<float, Eigen::Dynamic, 1>& delta) { return base.update_costs(delta); })
    .def("get_costs", &bdd_base_type::get_costs)
    .def("Lagrange_constraint_matrix", &bdd_base_type::Lagrange_constraint_matrix)
    .def("lower_bound", &bdd_base_type::lower_bound)
    .def("lower_bound_per_bdd", [](bdd_base_type& base) { return base.lower_bound_per_bdd(); })
    .def("nr_bdds", [](const bdd_base_type& base) { return base.nr_bdds(); })
    .def("nr_variables", [](const bdd_base_type& base) { return base.nr_variables(); })
    ;
}

