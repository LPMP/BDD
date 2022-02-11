#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "bdd_cuda_learned_mma.h"
#include "bdd_branch_instruction.h"
#include "ILP_input.h"
#include "bdd_preprocessor.h"
#include <sstream>
#include "cuda_utils.h"

namespace py=pybind11;

using bdd_type = LPMP::bdd_cuda_learned_mma<float>;

bdd_type create_solver(const py::bytes& s)
{
    std::istringstream ss(s);
    cereal::BinaryInputArchive archive(ss);
    bdd_type solver;
    archive(solver); 
    solver.init();
    return solver;
}

PYBIND11_MODULE(bdd_cuda_learned_mma_py, m) {
    m.doc() = "Python binding for bdd-based solver using CUDA";

    py::class_<bdd_type>(m, "bdd_cuda_learned_mma")
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
            return std::string("<bdd_cuda_learned_mma>: ") + 
                "nr_variables: "+ std::to_string(solver.nr_variables()) +
                ", nr_bdds: "+ std::to_string(solver.nr_bdds()) +
                ", nr_layers: "+ std::to_string(solver.nr_layers());
                })
        .def("nr_primal_variables", [](bdd_type& solver) { return solver.nr_variables(); })
        .def("nr_layers", [](bdd_type& solver) { return solver.nr_layers(); })
        .def("nr_layers", [](bdd_type& solver, const int hop_index) { return solver.nr_layers(hop_index); })
        .def("nr_bdds", &bdd_type::nr_bdds)
        .def("lower_bound", &bdd_type::lower_bound)
        
        .def("primal_variable_index", [](bdd_type& solver, const long primal_variable_index_out_ptr)
        {
            int* ptr = reinterpret_cast<int*>(primal_variable_index_out_ptr); 
            const thrust::device_vector<int> primal_index_managed = solver.get_primal_variable_index();
            thrust::copy(primal_index_managed.begin(), primal_index_managed.end(), ptr);
        }, "Sets primal variables indices for all dual variables in the pre-allocated memory of size = nr_layers() pointed to by the input pointer in INT32 format.")
        
        .def("bdd_index", [](bdd_type& solver, const long bdd_index_out_ptr)
        {
            int* ptr = reinterpret_cast<int*>(bdd_index_out_ptr); 
            const thrust::device_vector<int> bdd_index_managed = solver.get_bdd_index();
            thrust::copy(bdd_index_managed.begin(), bdd_index_managed.end(), ptr);
        }, "Sets BDD indices for all dual variables in the pre-allocated memory of size = nr_layers() pointed to by the input pointer in INT32 format.")

        .def("get_solver_costs", [](bdd_type& solver, 
                                const long lo_cost_out_ptr,
                                const long hi_cost_out_ptr,
                                const long delta_lo_out_ptr,
                                const long delta_hi_out_ptr)
        {
            thrust::device_ptr<float> lo_cost_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(lo_cost_out_ptr));
            thrust::device_ptr<float> hi_cost_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(hi_cost_out_ptr));
            thrust::device_ptr<float> delta_lo_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(delta_lo_out_ptr));
            thrust::device_ptr<float> delta_hi_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(delta_hi_out_ptr));
            solver.get_solver_costs(lo_cost_out_ptr_thrust, hi_cost_out_ptr_thrust, delta_lo_out_ptr_thrust, delta_hi_out_ptr_thrust);
        },"Get the costs i.e., (lo_costs (size = nr_layers()), hi_costs (size = nr_layers()), delta_lo (size = nr_variables()), delta_hi (size = nr_variables())),\n"
        "and set in the memory pointed to by input pointers to preallocated memory. This method can be used to restore solver state by calling set_solver_costs().")

        .def("set_solver_costs", [](bdd_type& solver, 
            const long lo_cost_ptr,
            const long hi_cost_ptr,
            const long delta_lo_ptr,
            const long delta_hi_ptr)
        {
            thrust::device_ptr<float> lo_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(lo_cost_ptr));
            thrust::device_ptr<float> hi_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(hi_cost_ptr));
            thrust::device_ptr<float> delta_lo_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(delta_lo_ptr));
            thrust::device_ptr<float> delta_hi_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(delta_hi_ptr));
            solver.set_solver_costs(lo_cost_ptr_thrust, hi_cost_ptr_thrust, delta_lo_ptr_thrust, delta_hi_ptr_thrust);
        },"Set the costs i.e., (lo_costs (size = nr_layers()), hi_costs (size = nr_layers()), delta_lo (size = nr_variables()), delta_hi (size = nr_variables())),\n"
        "to set solver state.")

        .def("iterations", [](bdd_type& solver, 
                            const long dist_weights_ptr, 
                            const long mm_diff_out_ptr, 
                            const int num_itr, 
                            const float omega) 
        {
            thrust::device_ptr<float> distw_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(dist_weights_ptr));
            thrust::device_ptr<float> mm_diff_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(mm_diff_out_ptr));
            solver.iterations(distw_ptr_thrust, mm_diff_ptr_thrust, num_itr, omega);
        }, "Runs solver for num_itr many iterations using distribution weights *dist_weights_ptr and sets the min-marginals to distribute in *mm_diff_out_ptr.\n"
        "Both dist_weights_ptr and mm_diff_out_ptr should point to a memory containing nr_layers() many elements in FP32 format.")

        .def("grad_iterations", [](bdd_type& solver, 
                                const long dist_weights_ptr,
                                const long grad_lo_cost_ptr,
                                const long grad_hi_cost_ptr,
                                const long grad_mm_ptr,
                                const long grad_dist_weights_out_ptr,
                                const long grad_omega_out_ptr,
                                const float omega, 
                                const int track_grad_after_itr, 
                                const int track_grad_for_num_itr) 
        {
            thrust::device_ptr<const float> dist_weights_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(dist_weights_ptr));
            thrust::device_ptr<float> grad_lo_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_lo_cost_ptr));
            thrust::device_ptr<float> grad_hi_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_hi_cost_ptr));
            thrust::device_ptr<float> grad_mm_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_mm_ptr));
            thrust::device_ptr<float> grad_dist_weights_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_dist_weights_out_ptr));
            thrust::device_ptr<float> grad_omega_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_omega_out_ptr));
            solver.grad_iterations(dist_weights_ptr_thrust, 
                                grad_lo_cost_ptr_thrust, 
                                grad_hi_cost_ptr_thrust,
                                grad_mm_ptr_thrust,
                                grad_dist_weights_out_ptr_thrust,
                                grad_omega_out_ptr_thrust,
                                omega,
                                track_grad_after_itr,
                                track_grad_for_num_itr);

        }, "Implements backprop through iterations().\n"
            "dist_weights: distribution weights used in the forward pass.\n"
            "grad_lo_cost: Input: incoming grad w.r.t lo_cost which were output from iterations and Outputs in-place to compute grad. lo_cost before iterations.\n"
            "grad_hi_cost: Input: incoming grad w.r.t hi_cost which were output from iterations and Outputs in-place to compute grad. hi_cost before iterations.\n"
            "grad_mm: Input: incoming grad w.r.t min-marg. diff. which were output from iterations and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iterations.\n"
            "grad_dist_weights_out: Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()).\n"
            "grad_dist_weights_out_ptr_thrust:  Output: contains grad w.r.t omega (size = 1)."
            "omega: floating point scalar in [0, 1] to scale current min-marginal difference before subtracting. (Same value as used in forward pass).\n"
            "track_grad_after_itr: First runs the solver for track_grad_after_itr many iterations without tracking gradients and then backpropagates through only last track_grad_for_num_itr many itrs.\n"
            "track_grad_for_num_itr: See prev. argument")

        .def("distribute_delta", [](bdd_type& solver, const long dist_weights_ptr) 
        {
            thrust::device_ptr<float> distw_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(dist_weights_ptr));
            solver.distribute_delta(distw_ptr_thrust);
        }, "Distributes the deferred min-marginals back to lo and hi costs such that dual constraint are satisfied with equality.\n"
            "For distributing the weights pointed to by dist_weights_ptr are used.")

        .def("grad_distribute_delta", [](bdd_type& solver, 
            const long grad_lo_cost_ptr,
            const long grad_hi_cost_ptr,
            const long grad_dist_weights_out_ptr)
        {
            thrust::device_ptr<float> grad_lo_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_lo_cost_ptr));
            thrust::device_ptr<float> grad_hi_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_hi_cost_ptr));
            thrust::device_ptr<float> grad_dist_weights_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_dist_weights_out_ptr));
            solver.grad_distribute_delta(grad_lo_cost_ptr_thrust, grad_hi_cost_ptr_thrust, grad_dist_weights_out_ptr_thrust);
        }, "Backprop. through distribute_delta.")

        .def("all_min_marginal_differences", [](bdd_type& solver, const long mm_diff_out_ptr)
        {
            thrust::device_ptr<float> mm_diff_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(mm_diff_out_ptr));
            const auto mms = solver.min_marginals_cuda(false);
            const auto& mms_0 = std::get<1>(mms);
            const auto& mms_1 = std::get<2>(mms);
            thrust::transform(mms_1.begin(), mms_1.end(), mms_0.begin(), mm_diff_ptr_thrust, thrust::minus<float>());
        }, "Computes min-marginal differences = (m^1 - m^0) for ALL dual variables and sets in memory pointed to by *mm_diff_out_ptr.")

        .def("grad_all_min_marginal_differences", [](bdd_type& solver, 
            const long grad_mm_diff, 
            const long grad_lo_out, 
            const long grad_hi_out)
        {
            thrust::device_ptr<float> grad_mm_diff_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_mm_diff));
            thrust::device_ptr<float> grad_lo_out_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_lo_out));
            thrust::device_ptr<float> grad_hi_out_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_hi_out));
            solver.grad_mm_diff_all_hops(grad_mm_diff_thrust, grad_lo_out_thrust, grad_hi_out_thrust);
        }, "Computes gradient of all_min_marginal_differences().\n"
            "Receives grad. w.r.t output of all_min_marginal_differences() and computes grad_lo_cost, grad_hi_cost"
            "and sets the gradient in the memory pointed by the input pointers.")

        .def("perturb_costs", [](bdd_type& solver, 
            const long lo_pert_ptr,
            const long hi_pert_ptr)
        {
            thrust::device_ptr<float> lo_pert_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(lo_pert_ptr));
            thrust::device_ptr<float> hi_pert_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(hi_pert_ptr));
            solver.update_costs<float>(lo_pert_ptr_thrust, solver.nr_variables(), hi_pert_ptr_thrust, solver.nr_variables());
        }, "Perturb primal costs by memory pointed by lo_pert_ptr, hi_pert_ptr. Where both inputs should point to a memory of size nr_variables().")

        .def("grad_cost_perturbation", [](bdd_type& solver, 
            const long grad_lo_cost_ptr,
            const long grad_hi_cost_ptr,
            const long grad_lo_pert_out,
            const long grad_hi_pert_out)
        {
            thrust::device_ptr<float> grad_lo_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_lo_cost_ptr));
            thrust::device_ptr<float> grad_hi_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_hi_cost_ptr));
            thrust::device_ptr<float> grad_lo_pert_out_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_lo_pert_out));
            thrust::device_ptr<float> grad_hi_pert_out_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_hi_pert_out));
            solver.grad_cost_perturbation(grad_lo_thrust, grad_hi_thrust, grad_lo_pert_out_thrust, grad_hi_pert_out_thrust);
        }, "During primal rounding calling update_costs(lo_pert, hi_pert) changes the dual costs, the underlying primal objective vector also changes.\n"
            "Here we compute gradients of such pertubation operation assuming that distribution of (lo_pert, hi_pert) was done with isoptropic weights.")

        
    ;
}

