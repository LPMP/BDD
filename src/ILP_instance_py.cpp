#include "ILP_input.h"
#include "ILP_parser.h"
#include "OPB_parser.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;


// return a matrix with dim (n+m,n+m), first n entries corresponding to variables, last m entries corresponding to inequalities.
Eigen::SparseMatrix<int> node_constraint_incidence_matrix(const LPMP::ILP_input& ilp)
{
    using T = Eigen::Triplet<int>;
    std::vector<T> coefficients;

    for(size_t c=0; c<ilp.nr_constraints(); ++c)
    {
        for(const auto& l : ilp.constraints()[c].variables)
        {
            coefficients.push_back(T(l.var, ilp.nr_variables() + c, l.coefficient));
            coefficients.push_back(T(ilp.nr_variables() + c, l.var, l.coefficient));
        }
    }

    Eigen::SparseMatrix<int> A(ilp.nr_variables() + ilp.nr_constraints(), ilp.nr_variables() + ilp.nr_constraints());
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

    py::class_<LPMP::ILP_input>(m, "ILP_instance")
        .def(py::init<>())
        .def("evaluate", [](const LPMP::ILP_input& ilp, const std::vector<char>& sol) { 
                return ilp.evaluate(sol.begin(), sol.end());
                })
        .def("feasible", [](const LPMP::ILP_input& ilp, const std::vector<char>& sol) { 
                return ilp.feasible(sol.begin(), sol.end());
                })
        .def("export_constraints", &LPMP::ILP_input::export_constraints)
        .def("node_constraint_incidence_matrix", [](const LPMP::ILP_input& ilp) { 
                return std::make_tuple(
                        node_constraint_incidence_matrix(ilp),
                        variable_constraint_bounds(ilp)
                        );
                })
        ;

        m.def("read_ILP", [](const std::string& filename) -> LPMP::ILP_input { return LPMP::ILP_parser::parse_file(filename); }); 
        m.def("read_OPB", [](const std::string& filename) -> LPMP::ILP_input { return LPMP::OPB_parser::parse_file(filename); }); 
}
