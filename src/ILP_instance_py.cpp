#include "ILP_input.h"
#include "ILP_parser.h"
#include "OPB_parser.h"
#include "specialized_solvers/mrf_input.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;


// return a matrix with dim (n, m), n corresponding to variables, m corresponding to constraints.
Eigen::SparseMatrix<int> node_constraint_incidence_matrix(const LPMP::ILP_input& ilp)
{
    using T = Eigen::Triplet<int>;
    std::vector<T> coefficients;

    for(size_t c=0; c<ilp.nr_constraints(); ++c)
    {
        const auto& constr = ilp.constraints()[c];
        if(!constr.is_linear())
            throw std::runtime_error("Only linear constraints supported");
        assert(constr.monomials.size() == constr.coefficients.size());
        for(size_t monomial_idx=0; monomial_idx<constr.monomials.size(); ++monomial_idx)
        {
            const size_t var = constr.monomials(monomial_idx, 0);
            const int coeff = constr.coefficients[monomial_idx];
            coefficients.push_back(T(var, c, coeff));
        }
    }

    Eigen::SparseMatrix<int> A(ilp.nr_variables(), ilp.nr_constraints());
    A.setFromTriplets(coefficients.begin(), coefficients.end());

    return A;
}

Eigen::MatrixXi variable_constraint_bounds(const LPMP::ILP_input& ilp)
{
    Eigen::MatrixXi D(ilp.nr_variables() + ilp.nr_constraints(), 2);

    for(size_t i=0; i<ilp.nr_variables(); ++i)
    {
        D(i,0) = 0;
        D(i,1) = 1;
    }

    for(size_t c=0; c<ilp.nr_constraints(); ++c)
    {
        if(ilp.constraints()[c].ineq == LPMP::ILP_input::inequality_type::equal)
        {
            D(ilp.nr_variables() + c, 0) = ilp.constraints()[c].right_hand_side;
            D(ilp.nr_variables() + c, 1) = ilp.constraints()[c].right_hand_side;
        }
        else if(ilp.constraints()[c].ineq == LPMP::ILP_input::inequality_type::smaller_equal)
        {
            D(ilp.nr_variables() + c, 0) = std::numeric_limits<int>::min();
            D(ilp.nr_variables() + c, 1) = ilp.constraints()[c].right_hand_side; 
        }
        else if(ilp.constraints()[c].ineq == LPMP::ILP_input::inequality_type::greater_equal)
        {
            D(ilp.nr_variables() + c, 0) = ilp.constraints()[c].right_hand_side; 
            D(ilp.nr_variables() + c, 1) = std::numeric_limits<int>::max();
        } 
    }

    return D; 
}

PYBIND11_MODULE(ILP_instance_py, m) {
    m.doc() = "Python binding for ILP instance";

    py::enum_<LPMP::ILP_input::inequality_type>(m, "inequality_type")
        .value("smaller_equal", LPMP::ILP_input::inequality_type::smaller_equal)
        .value("greater_equal", LPMP::ILP_input::inequality_type::greater_equal)
        .value("equal", LPMP::ILP_input::inequality_type::equal)
        .export_values();

    py::class_<LPMP::ILP_input>(m, "ILP_instance")
        .def(py::init<>())
        .def("nr_constraints",&LPMP::ILP_input::nr_constraints)
        .def("nr_variables",&LPMP::ILP_input::nr_variables)
        .def("evaluate", [](const LPMP::ILP_input& ilp, const std::vector<char>& sol) { 
                return ilp.evaluate(sol.begin(), sol.end());
                })
        .def("feasible", [](const LPMP::ILP_input& ilp, const std::vector<char>& sol) { 
                return ilp.feasible(sol.begin(), sol.end());
                })
        .def("export_constraints", &LPMP::ILP_input::export_constraints)
        .def("objective", [](const LPMP::ILP_input& ilp) { return ilp.objective(); })
        .def("get_var_name", &LPMP::ILP_input::get_var_name)
        .def("get_var_index", &LPMP::ILP_input::get_var_index)
        .def("node_constraint_incidence_matrix", [](const LPMP::ILP_input& ilp) { 
                return std::make_tuple(
                        node_constraint_incidence_matrix(ilp),
                        variable_constraint_bounds(ilp)
                        );
                })
        .def("variable_constraint_bounds", [](const LPMP::ILP_input& ilp) { 
                return variable_constraint_bounds(ilp);
                })
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
        m.def("parse_ILP", [](const char* instance) -> LPMP::ILP_input { return LPMP::ILP_parser::parse_string(instance); }); 
        m.def("read_OPB", [](const std::string& filename) -> LPMP::ILP_input { return LPMP::OPB_parser::parse_file(filename); }); 
        m.def("read_MRF_UAI", [](const std::string& filename) -> LPMP::ILP_input { return LPMP::parse_mrf_uai_file(filename).convert_to_ilp(); });
}
