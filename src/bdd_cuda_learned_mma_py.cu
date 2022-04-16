#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "bdd_cuda_learned_mma.h"
#include "bdd_branch_instruction.h"
#include "ILP_input.h"
#include "two_dimensional_variable_array.hxx"
#include "bdd_preprocessor.h"
#include <sstream>
#include <fstream>
#include "cuda_utils.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

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

struct set_primal_indices {
    const unsigned long num_vars;
    __host__ __device__ int operator()(const int i)
    {
        return min(i, (int) num_vars);
    }
};

template<typename T>
std::vector<size_t> argsort(const std::vector<T> &array) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}

std::vector<float> get_constraint_matrix_coeffs(const LPMP::ILP_input& ilp, const bdd_type& solver)
{
    if (ilp.nr_constraints() != solver.nr_bdds())
    {
        std::cout<<"Number of constraints: "<<ilp.nr_constraints()<<", not equal to number of BDDs: "<<solver.nr_bdds()<<"\n";
        throw std::runtime_error("error");
    }
    const std::vector<size_t> bdd_to_constraint_map = solver.bdd_to_constraint_map();
    if (bdd_to_constraint_map.size() != solver.nr_bdds())
    {
        throw std::runtime_error("bdd_to_constraint_map not calculated.");
    }

    const size_t num_elements = solver.nr_layers();
    std::vector<int> var_indices_sorted(num_elements);
    std::vector<int> con_indices_sorted(num_elements);
    std::vector<int> cumm_num_vars_per_constraint(solver.nr_bdds() + 1);
    std::vector<size_t> indices(num_elements);
    { // Create COO representation for faster indexing later.
        thrust::device_vector<int> dev_primal_index = solver.get_primal_variable_index();
        const thrust::device_vector<int> dev_bdd_index = solver.get_bdd_index();
        thrust::device_vector<unsigned long> dev_indices(num_elements);
        thrust::sequence(dev_indices.begin(), dev_indices.end());

        thrust::device_vector<size_t> dev_bdd_to_constraint_map(bdd_to_constraint_map.begin(), bdd_to_constraint_map.end());
        thrust::device_vector<int> dev_con_index(dev_bdd_index.size());

        // Map bdd_index to constraint index:
        thrust::gather(dev_bdd_index.begin(), dev_bdd_index.end(), dev_bdd_to_constraint_map.begin(), dev_con_index.begin());

        auto first_key = thrust::make_zip_iterator(thrust::make_tuple(dev_con_index.begin(), dev_primal_index.begin()));
        auto last_key = thrust::make_zip_iterator(thrust::make_tuple(dev_con_index.end(), dev_primal_index.end()));
        thrust::sort_by_key(thrust::device, first_key, last_key, dev_indices.begin());

        thrust::device_vector<int> dev_cumm_num_vars_per_constraint(num_elements);
        auto new_last = thrust::reduce_by_key(dev_con_index.begin(), dev_con_index.end(), thrust::make_constant_iterator<int>(1), 
                                thrust::make_discard_iterator(), dev_cumm_num_vars_per_constraint.begin());
        const auto nr_con = std::distance(dev_cumm_num_vars_per_constraint.begin(), new_last.second);
        if (nr_con != solver.nr_bdds())
            throw std::runtime_error("con_indices reduced size mismatch.");
        dev_cumm_num_vars_per_constraint.resize(nr_con);
        thrust::inclusive_scan(dev_cumm_num_vars_per_constraint.begin(), dev_cumm_num_vars_per_constraint.end(), 
                                dev_cumm_num_vars_per_constraint.begin());
        thrust::copy(dev_con_index.begin(), dev_con_index.end(), con_indices_sorted.begin());
        thrust::copy(dev_primal_index.begin(), dev_primal_index.end(), var_indices_sorted.begin());
        thrust::copy(dev_indices.begin(), dev_indices.end(), indices.begin());
        thrust::copy(dev_cumm_num_vars_per_constraint.begin(), dev_cumm_num_vars_per_constraint.end(), 
                    cumm_num_vars_per_constraint.begin() + 1);
        cumm_num_vars_per_constraint[0] = 0;
    }

    std::vector<float> coefficients(num_elements, 0.0);
    int find_start_index = cumm_num_vars_per_constraint[0];
    for(size_t c = 0; c < ilp.nr_constraints(); ++c)
    {
        const auto& constr = ilp.constraints()[c];
        if(!constr.is_linear())
            throw std::runtime_error("Only linear constraints supported");
        assert(constr.monomials.size() == constr.coefficients.size());
        int find_end_index = cumm_num_vars_per_constraint[c + 1];
        for(size_t monomial_idx = 0; monomial_idx < constr.monomials.size(); ++monomial_idx)
        {
            const size_t var = constr.monomials(monomial_idx, 0);
            const int coeff = constr.coefficients[monomial_idx];
            // Find where does (c, var) occurs in solver variable and constraint indices:
            const auto it = std::find(var_indices_sorted.begin() + find_start_index, 
                                    var_indices_sorted.begin() + find_end_index, var);
            if (it == var_indices_sorted.begin() + find_end_index)
            {
                std::cout<<"ILP variable not found in BDD. Var: " + std::to_string(var)<<", Con: "<<c<<"\n";
                throw std::runtime_error("error");
            }
            else
            {
                const int index_to_place = indices[std::distance(var_indices_sorted.begin(), it)];
                coefficients[index_to_place] = coeff;
            }
        }
        find_start_index = find_end_index;
    }

    return coefficients;
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
        .def(py::init([](const LPMP::ILP_input& ilp, const bool compute_bdd_to_constraint_map = true) {
                    LPMP::bdd_preprocessor bdd_pre;
                    const auto constraint_to_bdd_map = bdd_pre.add_ilp(ilp);
                    auto* base = new bdd_type(bdd_pre.get_bdd_collection());
                    base->update_costs(ilp.objective().begin(), ilp.objective().begin(), ilp.objective().begin(), ilp.objective().end());
                    if (compute_bdd_to_constraint_map)
                        base->compute_bdd_to_constraint_map(constraint_to_bdd_map);
                    return base;
                }))
        .def("__repr__", [](const bdd_type &solver) {
            return std::string("<bdd_cuda_learned_mma>: ") + 
                "nr_variables: "+ std::to_string(solver.nr_variables()) +
                ", nr_bdds: "+ std::to_string(solver.nr_bdds()) +
                ", nr_layers: "+ std::to_string(solver.nr_layers());
                })
        .def("export_ss", [](const bdd_type& solver, const std::string output_path){
            std::ofstream os(output_path, std::ios::binary);
            cereal::BinaryOutputArchive archive(os);
            archive(solver);
            std::cout<<"Exported solver data to path: "<<output_path<<"\n";
        })
        .def("nr_primal_variables", [](const bdd_type& solver) { return solver.nr_variables(); })
        .def("nr_layers", [](const bdd_type& solver) { return solver.nr_layers(); })
        .def("nr_layers", [](const bdd_type& solver, const int hop_index) { return solver.nr_layers(hop_index); })
        .def("nr_bdds", &bdd_type::nr_bdds)
        .def("constraint_matrix_coeffs", [](const bdd_type& solver, const LPMP::ILP_input& ilp)
        {
            return get_constraint_matrix_coeffs(ilp, solver);
        }, "Computes the coefficients for each variable appearing in constraint."
        "\nAssumes that each BDD correspond to a linear constraint present in original ILP.")
        .def("bdd_to_constraint_map", &bdd_type::bdd_to_constraint_map)
        .def("lower_bound", &bdd_type::lower_bound)
        .def("lower_bound_per_bdd", [](bdd_type& solver, const long lb_out_ptr)
        {
            thrust::device_ptr<float> lb_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(lb_out_ptr));
            solver.lower_bound_per_bdd(lb_out_ptr_thrust);
        }, "Computes LB for each constraint and copies in the provided pointer to FP32 memory (size = nr_bdds()).")
        .def("solution_per_bdd", [](bdd_type& solver, const long sol_out_ptr)
        {
            thrust::device_ptr<float> sol_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(sol_out_ptr));
            solver.bdds_solution_cuda(sol_out_ptr_thrust);
        }, "Computes argmin for each constraint and copies in the provided pointer to FP32 memory (size = nr_layers()).")
        .def("terminal_layer_indices", [](bdd_type& solver, const long indices_out_ptr)
        {
            thrust::device_ptr<int> indices_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<int*>(indices_out_ptr));
            solver.terminal_layer_indices(indices_out_ptr_thrust);
        }, "Computes indices of dual variables which are actually just terminal nodes. Input argument to point to a INT32 memory of size = nr_bdds().")
        .def("primal_variable_index", [](bdd_type& solver, const long primal_variable_index_out_ptr)
        {
            int* ptr = reinterpret_cast<int*>(primal_variable_index_out_ptr); 
            const thrust::device_vector<int> primal_index_managed = solver.get_primal_variable_index();
            thrust::transform(primal_index_managed.begin(), primal_index_managed.end(), ptr, set_primal_indices({solver.nr_variables()}));
        }, "Sets primal variables indices for all dual variables in the pre-allocated memory of size = nr_layers() pointed to by the input pointer in INT32 format.\n"
        "Also contains entries for root/terminal nodes for which the values are equal to nr_variables().")
        
        .def("bdd_index", [](bdd_type& solver, const long bdd_index_out_ptr)
        {
            int* ptr = reinterpret_cast<int*>(bdd_index_out_ptr); 
            const thrust::device_vector<int> bdd_index_managed = solver.get_bdd_index();
            thrust::copy(bdd_index_managed.begin(), bdd_index_managed.end(), ptr);
        }, "Sets BDD indices for all dual variables in the pre-allocated memory of size = nr_layers() pointed to by the input pointer in INT32 format.")

        .def("get_primal_objective_vector", [](bdd_type& solver, const long primal_obj_out_ptr)
        {
            thrust::device_ptr<float> primal_obj_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(primal_obj_out_ptr));
            solver.compute_primal_objective_vec(primal_obj_out_ptr_thrust);
        }, "Computes primal objective vector from dual variables in the pre-allocated memory of size = nr_primal_variables() pointed to by the input pointer in FP32 format.")

        .def("get_solver_costs", [](const bdd_type& solver, 
                                const long lo_cost_out_ptr,
                                const long hi_cost_out_ptr,
                                const long deferred_mm_out_ptr)
        {
            thrust::device_ptr<float> lo_cost_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(lo_cost_out_ptr));
            thrust::device_ptr<float> hi_cost_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(hi_cost_out_ptr));
            thrust::device_ptr<float> deferred_mm_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(deferred_mm_out_ptr));
            solver.get_solver_costs(lo_cost_out_ptr_thrust, hi_cost_out_ptr_thrust, deferred_mm_out_ptr_thrust);
        },"Get the costs i.e., (lo_costs (size = nr_layers()), hi_costs (size = nr_layers()), deferred_mm_out_ptr_thrust (size = nr_variables()), \n"
        "and set in the memory pointed to by input pointers to preallocated memory. This method can be used to restore solver state by calling set_solver_costs().")

        .def("set_solver_costs", [](bdd_type& solver, 
            const long lo_cost_ptr,
            const long hi_cost_ptr,
            const long def_mm_ptr)
        {
            thrust::device_ptr<float> lo_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(lo_cost_ptr));
            thrust::device_ptr<float> hi_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(hi_cost_ptr));
            thrust::device_ptr<float> def_mm_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(def_mm_ptr));
            solver.set_solver_costs(lo_cost_ptr_thrust, hi_cost_ptr_thrust, def_mm_ptr_thrust);
        },"Set the costs i.e., (lo_costs (size = nr_layers()), hi_costs (size = nr_layers()), def_mm_ptr (size = nr_layers()) to set solver state.")

        .def("non_learned_iterations", [](bdd_type& solver, const float omega, const int max_num_itr, const float improvement_slope) 
        {
            const double lb_initial = solver.lower_bound();
            double lb_first_iter = std::numeric_limits<double>::max();
            double lb_prev = lb_initial;
            double lb_post = lb_prev;
            for (int itr = 0; itr < max_num_itr; itr++)
            {
                solver.iteration(omega);
                lb_prev = lb_post;
                lb_post = solver.lower_bound();
                if(itr == 0)
                    lb_first_iter = lb_post;
                if (std::abs(lb_prev - lb_post) < improvement_slope * std::abs(lb_initial - lb_first_iter))
                    break;
            }
        }, "Runs parallel_mma solver for a maximum of max_num_itr iterations and stops earlier if rel. improvement is less than improvement_slop .")

        .def("iterations", [](bdd_type& solver, 
                            const long dist_weights_ptr, 
                            const int num_itr, 
                            const float omega_scalar,
                            const double improvement_slope,
                            const long omega_vec_ptr,
                            const bool omega_vec_valid,
                            const int compute_history_for_itr,
                            const float beta,
                            const long sol_avg_ptr,
                            const long lb_first_order_avg_ptr,
                            const long lb_second_order_avg_ptr) 
        {
            thrust::device_ptr<float> distw_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(dist_weights_ptr));
            thrust::device_ptr<float> omega_vec_thrust, sol_avg_ptr_thrust, lb_first_ptr_thrust, lb_second_ptr_thrust;
            if (compute_history_for_itr)
            {
                sol_avg_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(sol_avg_ptr)); 
                lb_first_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(lb_first_order_avg_ptr)); 
                lb_second_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(lb_second_order_avg_ptr)); 
            }
            
            if (omega_vec_valid)
            {
                omega_vec_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(omega_vec_ptr));
                return solver.iterations(distw_ptr_thrust, num_itr, 1.0, improvement_slope, 
                                        sol_avg_ptr_thrust, lb_first_ptr_thrust, lb_second_ptr_thrust,
                                        compute_history_for_itr, beta, omega_vec_thrust);
            }
            else
                return solver.iterations(distw_ptr_thrust, num_itr, omega_scalar, improvement_slope, 
                                        sol_avg_ptr_thrust, lb_first_ptr_thrust, lb_second_ptr_thrust, 
                                        compute_history_for_itr, beta);
        }, "Runs solver for num_itr many iterations using distribution weights *dist_weights_ptr and sets the min-marginals to distribute in *mm_diff_ptr.\n"
        "dist_weights_ptr, mm_diff_ptr and sol_avg_ptr should point to a memory containing nr_layers() many elements in FP32 format.\n"
        "lb_first_order_avg_ptr and lb_second_order_avg_ptr should point to a memory containing nr_bdds() many elements in FP32 format.\n"
        "If omega_vec_valid == True, then omega_vec_ptr is used (size = nr_layers()) instead of omega_scalar."
        "First iteration used the deferred min-marginals in mm_diff_ptr to distribute.")

        .def("grad_iterations", [](bdd_type& solver, 
                                const long dist_weights_ptr,
                                const long grad_lo_cost_ptr,
                                const long grad_hi_cost_ptr,
                                const long grad_mm_ptr,
                                const long grad_dist_weights_out_ptr,
                                const long grad_omega_out_ptr,
                                const float omega_scalar, 
                                const int track_grad_after_itr, 
                                const int track_grad_for_num_itr,
                                const long omega_vec_ptr,
                                const bool omega_vec_valid,
                                const int num_caches) 
        {
            thrust::device_ptr<const float> dist_weights_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(dist_weights_ptr));
            thrust::device_ptr<float> grad_lo_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_lo_cost_ptr));
            thrust::device_ptr<float> grad_hi_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_hi_cost_ptr));
            thrust::device_ptr<float> grad_mm_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_mm_ptr));
            thrust::device_ptr<float> grad_dist_weights_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_dist_weights_out_ptr));
            thrust::device_ptr<float> grad_omega_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_omega_out_ptr));
            if (omega_vec_valid)
            {
                thrust::device_ptr<float> omega_vec_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(omega_vec_ptr));
                solver.grad_iterations(dist_weights_ptr_thrust, grad_lo_cost_ptr_thrust, grad_hi_cost_ptr_thrust,
                                grad_mm_ptr_thrust,grad_dist_weights_out_ptr_thrust, grad_omega_out_ptr_thrust,
                                1.0, track_grad_after_itr, track_grad_for_num_itr, num_caches, omega_vec_thrust);
            }
            else
                solver.grad_iterations(dist_weights_ptr_thrust, grad_lo_cost_ptr_thrust, grad_hi_cost_ptr_thrust,
                                grad_mm_ptr_thrust, grad_dist_weights_out_ptr_thrust, grad_omega_out_ptr_thrust,
                                omega_scalar, track_grad_after_itr, track_grad_for_num_itr, num_caches);


        }, "Implements backprop through iterations().\n"
            "dist_weights: distribution weights used in the forward pass.\n"
            "grad_lo_cost: Input: incoming grad w.r.t lo_cost which were output from iterations and Outputs in-place to compute grad. lo_cost before iterations.\n"
            "grad_hi_cost: Input: incoming grad w.r.t hi_cost which were output from iterations and Outputs in-place to compute grad. hi_cost before iterations.\n"
            "grad_mm: Input: incoming grad w.r.t min-marg. diff. which were output from iterations and Outputs in-place to compute grad. w.r.t deferred min-marginals used in iterations.\n"
            "grad_dist_weights_out: Output: contains grad w.r.t distribution weights, assumes the memory is already allocated (= nr_layers()).\n"
            "grad_omega_out_ptr:  Output: contains grad w.r.t omega (size = 1)."
            "omega: floating point scalar in [0, 1] to scale current min-marginal difference before subtracting. (Same value as used in forward pass).\n"
            "track_grad_after_itr: First runs the solver for track_grad_after_itr many iterations without tracking gradients and then backpropagates through only last track_grad_for_num_itr many itrs.\n"
            "track_grad_for_num_itr: See prev. argument.\n"
            "omega_vec_ptr: vector-valued (size = nr_layers()) damping weights used in forward pass (if omega_vec_valid == True, otherwise not used).\n")

        .def("distribute_delta", [](bdd_type& solver) 
        {
            solver.distribute_delta();
        }, "Distributes the deferred min-marginals back to lo and hi costs such that dual constraint are satisfied with equality.\n"
            "deferred min-marginals are zero-ed out after distributing.")

        .def("grad_distribute_delta", [](bdd_type& solver, 
            const long grad_lo_cost_ptr,
            const long grad_hi_cost_ptr,
            const long grad_def_mm_out_ptr)
        {
            thrust::device_ptr<float> grad_lo_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_lo_cost_ptr));
            thrust::device_ptr<float> grad_hi_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_hi_cost_ptr));
            thrust::device_ptr<float> grad_def_mm_out_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_def_mm_out_ptr));
            solver.grad_distribute_delta(grad_lo_cost_ptr_thrust, grad_hi_cost_ptr_thrust, grad_def_mm_out_ptr_thrust);
        }, "Backprop. through distribute_delta.")
        .def("grad_lower_bound_per_bdd", [](bdd_type& solver, const long grad_lb_per_bdd, const long grad_lo_cost_ptr, const long grad_hi_cost_ptr)
        {
            thrust::device_ptr<float> grad_lb_per_bdd_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_lb_per_bdd));
            thrust::device_ptr<float> grad_lo_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_lo_cost_ptr));
            thrust::device_ptr<float> grad_hi_cost_ptr_thrust = thrust::device_pointer_cast(reinterpret_cast<float*>(grad_hi_cost_ptr));
            solver.grad_lower_bound_per_bdd(grad_lb_per_bdd_thrust, grad_lo_cost_ptr_thrust, grad_hi_cost_ptr_thrust);
        }, "Backprop. through lower bound per BDD.")
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

