#include "ILP_input.h"
#include "ILP_parser.h"
#include "OPB_parser.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(ILP_instance_py, m) {
    m.doc() = "Python binding for ILP instance";

    py::class_<LPMP::ILP_input>(m, "ILP_instance")
        .def(py::init<>())
        .def("evaluate", [](const LPMP::ILP_input& ilp, const std::vector<char>& sol) { 
                return ilp.evaluate(sol.begin(), sol.end());
                })
        .def("feasible", [](const LPMP::ILP_input& ilp, const std::vector<char>& sol) { 
                return ilp.feasible(sol.begin(), sol.end());
                })
        .def("export_constraints", &LPMP::ILP_input::export_constraints)
        ;

    m.def("read_ILP", [](const std::string& filename) -> LPMP::ILP_input { return LPMP::ILP_parser::parse_file(filename); }); 
    m.def("read_OPB", [](const std::string& filename) -> LPMP::ILP_input { return LPMP::OPB_parser::parse_file(filename); }); 
}
