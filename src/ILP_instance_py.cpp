#include "ILP_input.h"
#include "ILP_parser.h"
#include "OPB_parser.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(ILP_instance_py, m) {
    m.doc() = "Python binding for ILP instance";

    py::enum_<LPMP::ILP_input::inequality_type>(m, "inequality_type")
        .value("smaller_equal", LPMP::ILP_input::inequality_type::smaller_equal)
        .value("greater_equal", LPMP::ILP_input::inequality_type::greater_equal)
        .value("equal", LPMP::ILP_input::inequality_type::equal)
        .export_values();

    py::class_<LPMP::ILP_input>(m, "ILP_instance")
        .def(py::init<>())
        .def("evaluate", [](const LPMP::ILP_input& ilp, const std::vector<char>& sol) { 
                return ilp.evaluate(sol.begin(), sol.end());
                })
        .def("feasible", [](const LPMP::ILP_input& ilp, const std::vector<char>& sol) { 
                return ilp.feasible(sol.begin(), sol.end());
                })
        .def("export_constraints", &LPMP::ILP_input::export_constraints)
        .def("add_new_variable_with_obj", [](LPMP::ILP_input& ilp, const std::string& var_name, const double coefficient) {
                auto index = ilp.add_new_variable(var_name);
                ilp.add_to_objective(coefficient, var_name);
                return index;
                })
        .def("add_new_constraint", [](LPMP::ILP_input& ilp, const std::string& constraint_name, const std::vector<std::string>& var_names, std::vector<int>& coeffs, const int rhs, const LPMP::ILP_input::inequality_type ineq_type) {
                assert(var_names.size() == coeffs.size());
                ilp.begin_new_inequality();
                ilp.set_inequality_identifier(constraint_name);
                for(size_t i = 0; i < var_names.size(); ++i)
                    ilp.add_to_constraint(coeffs[i], var_names[i]);

                ilp.set_inequality_type(ineq_type);
                ilp.set_right_hand_side(rhs);
                });

    m.def("read_ILP", [](const std::string& filename) -> LPMP::ILP_input { return LPMP::ILP_parser::parse_file(filename); }); 
    m.def("read_OPB", [](const std::string& filename) -> LPMP::ILP_input { return LPMP::OPB_parser::parse_file(filename); }); 
}
