#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11_json.hpp"
#include "bdd_solver.h"
#include "ILP_input.h"

namespace py=pybind11;

PYBIND11_MODULE(bdd_solver_py, m) {
    m.doc() = "Bindings for BDD solver.";
    py::class_<LPMP::bdd_solver>(m, "bdd_solver")
        .def(py::init<>())
        .def(py::init<const std::string&>())
        .def(py::init<const nlohmann::json&>())
        .def("solve", py::overload_cast<>(&LPMP::bdd_solver::solve))
        .def("solve", py::overload_cast<nlohmann::json&>(&LPMP::bdd_solver::solve))
        .def("min_marginals", &LPMP::bdd_solver::min_marginals)
        .def("min_marginals_with_variable_names", &LPMP::bdd_solver::min_marginals_with_variable_names)
        .def("lower_bound", &LPMP::bdd_solver::lower_bound)
        ;
}