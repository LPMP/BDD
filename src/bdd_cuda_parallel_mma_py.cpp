#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "bdd_cuda_parallel_mma.h"
#include "bdd_branch_instruction.h"
#include "ILP_input.h"
#include "bdd_preprocessor.h"
#include <sstream>

namespace py=pybind11;

using bdd_type = LPMP::bdd_cuda_parallel_mma<float>;

bdd_type create_solver(const py::bytes& s)
{
    std::istringstream ss(s);
    cereal::BinaryInputArchive archive(ss);
    bdd_type solver;
    archive(solver); 
    solver.init();
    return solver;
}

PYBIND11_MODULE(bdd_cuda_parallel_mma_py, m) {
    m.doc() = "Python binding for bdd-based solver using CUDA";

    py::class_<bdd_type>(m, "bdd_cuda_parallel_mma")
        .def(py::pickle(
                    [](const bdd_type& solver) {
                        std::stringstream ss;
                        cereal::BinaryOutputArchive archive(ss);
                        archive(solver);
                        return py::bytes(ss.str());
                },
                    [](const py::bytes& s) {
                        return create_solver(s);
                }))
        .def(py::init([](const LPMP::ILP_input& ilp) {
                    LPMP::bdd_preprocessor bdd_pre(ilp);
                    auto* base = new bdd_type(bdd_pre.get_bdd_collection());  //TODO: New?
                    base->update_costs(ilp.objective().begin(), ilp.objective().begin(), ilp.objective().begin(), ilp.objective().end());
                    return base;
                }))
        .def("__repr__", [](const bdd_type &solver) {
            return std::string("<bdd_cuda_parallel_mma>: ") + 
                "nr_variables: "+ std::to_string(solver.nr_variables()) +
                ", nr_bdds: "+ std::to_string(solver.nr_bdds()) +
                ", nr_variable_blocks: "+ std::to_string(solver.nr_variable_blocks());
                })
        .def("nr_variables", [](bdd_type& solver) { return solver.nr_variables(); })
        .def("nr_variables", [](bdd_type& solver, const int var_block_idx) { return solver.nr_variables(var_block_idx); })
        .def("nr_variable_blocks", &bdd_type::nr_variable_blocks)
        .def("nr_bdds", &bdd_type::nr_bdds)
        .def("lower_bound", &bdd_type::lower_bound)
        .def("min_marginal_diff_of_block", [](bdd_type& solver, const int var_block_idx, const long mm_diff_out_ptr) 
        {
            // Computes min-marginal of variables in block: 'block_idx' and sets the result in mm_diff_ptr. 
            // Assumes enough space is allocated. To query required space call: solver.nr_variables(block_idx).

            float* mm_diff_ptr = reinterpret_cast<float*>(mm_diff_out_ptr); // Points to memory allocated by Python.
            thrust::device_ptr<float> mm_diff_ptr_thrust = thrust::device_pointer_cast(mm_diff_ptr);
            solver.min_marginals_from_directional_costs(var_block_idx, 1.0, mm_diff_ptr_thrust);
        })
    ;
}

