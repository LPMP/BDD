#include "bdd_cuda_learned_mma.h"
#include "ILP_parser.h"
#include "bdd_collection/bdd_collection.h"
#include "bdd_preprocessor.h"
#include "test.h"
#include "cuda_utils.h"

using namespace LPMP;
using namespace BDD;

const char * two_simplex = 
R"(Minimize
1 x_1 + 2 x_2 + 1 x_3
+2 x_4 + 1 x_5 + 2 x_6
Subject To
x_1 + x_2 + x_3 + x_4 = 1
x_4 + x_5 + x_6 = 2
End)";


struct sum_dist_weights {
    const int* primal_index;
    const double* dist_weights;
    double* dist_weights_sum;
    const unsigned long num_vars;
    __device__ void operator()(const int i)
    {
        const int primal_var = primal_index[i];
        if (primal_var < num_vars) // ignores terminal nodes.
            atomicAdd(&dist_weights_sum[primal_var], dist_weights[i]);
    }
};

struct normalize_dist_weights {
    const int* primal_index;
    const double* dist_weights_sum;
    double* dist_weights;
    const unsigned long num_vars;
    __device__ void operator()(const int i)
    {
        const int primal_var = primal_index[i];
        if (primal_var < num_vars) // ignores terminal nodes.
            dist_weights[i] /= dist_weights_sum[primal_var];
    }
};

template<typename SOLVER>
void project_dist_weights(SOLVER& solver, thrust::device_vector<double>& dist_weights, const thrust::device_vector<int>& primal_var_index)
{
    thrust::device_vector<double> dist_weights_sum(solver.nr_variables(), 0.0);

    sum_dist_weights sum_func({thrust::raw_pointer_cast(primal_var_index.data()), 
                                thrust::raw_pointer_cast(dist_weights.data()), 
                                thrust::raw_pointer_cast(dist_weights_sum.data()),
                                solver.nr_variables()});

    thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + dist_weights.size(), sum_func);

    normalize_dist_weights norm_func({thrust::raw_pointer_cast(primal_var_index.data()), 
                                    thrust::raw_pointer_cast(dist_weights_sum.data()),
                                    thrust::raw_pointer_cast(dist_weights.data()), 
                                    solver.nr_variables()});
    thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + dist_weights.size(), norm_func);
}

struct loss_func {
    const int* primal_index;
    const double* pred;
    const double* target;
    double* loss;
    double* loss_sign_match;
    const unsigned long num_vars;
    __device__ void operator()(const int i)
    {
        const int primal_var = primal_index[i];
        if (primal_var < num_vars)
        {
            if (primal_var < num_vars)
            {
                loss[i] = abs(pred[i] - target[primal_var]);
                if (pred[i] * target[primal_var] > 0.1)
                    loss_sign_match[i] = 0.0;
                else
                    loss_sign_match[i] = 1.0;
            }
        }
        else
            loss[i] = 0.0;
    }
};

struct loss_gradient_func {
    const int* primal_index;
    const double* pred;
    const double* target;
    double* grad;
    const unsigned long num_vars;
    __device__ void operator()(const int i)
    {
        const int primal_var = primal_index[i];
        if (primal_var < num_vars)
        {
            if (pred[i] > target[primal_var])
                grad[i] = 1.0;
            else
                grad[i] = -1.0;
        }
        else
            grad[i] = 0.0;
    }
};

struct grad_step_pert {
    const double* grad_lo_pert;
    const double* grad_hi_pert;
    double* lo_pert;
    double* hi_pert;
    double lr;
    __device__ void operator()(const int i)
    {
        lo_pert[i] = lo_pert[i] - lr * grad_lo_pert[i];
        hi_pert[i] = hi_pert[i] - lr * grad_hi_pert[i];
    }
};

void test_problem(const char* instance, const thrust::device_vector<double>& expected_mm_diff, const double omega = 0.5, const double tol = 1e-12)
{
    ILP_input ilp = ILP_parser::parse_string(instance);
    bdd_preprocessor bdd_pre(ilp);
    bdd_collection bdd_col = bdd_pre.get_bdd_collection();
    bdd_cuda_learned_mma<double> solver(bdd_col);

    for(size_t i=0; i<solver.nr_variables(); ++i)
        solver.set_cost(ilp.objective()[i], i);

    thrust::device_vector<double> dist_weights(solver.nr_layers(), 1.0);
    const thrust::device_vector<int> primal_var_index = solver.get_primal_variable_index();
    const thrust::device_vector<int> bdd_index = solver.get_bdd_index();
    print_vector(primal_var_index, "primal_var_index");
    print_vector(bdd_index, "bdd_index");
    project_dist_weights(solver, dist_weights, primal_var_index);

    thrust::device_vector<double> pert_lo(solver.nr_variables(), 0.0);
    thrust::device_vector<double> pert_hi(solver.nr_variables(), 0.0);
    thrust::device_vector<double> grad_pert_lo(solver.nr_variables());
    thrust::device_vector<double> grad_pert_hi(solver.nr_variables());
    thrust::device_vector<double> final_mm_diff(solver.nr_layers());
    thrust::device_vector<double> loss_grad_mm(solver.nr_layers());
    thrust::device_vector<double> grad_lo_costs(solver.nr_layers());
    thrust::device_vector<double> grad_hi_costs(solver.nr_layers());

    const int num_solver_itr = 5;
    double prev_loss = 0;
    double avg_loss_improvement_per_itr = 0;
    const int num_learning_itr = 100;
    double num_incorrect = solver.nr_layers();
    for(int learning_itr = 0; learning_itr < num_learning_itr; learning_itr++)
    {
        const auto orig_costs = solver.get_solver_costs();
        solver.update_costs(pert_lo, pert_hi); // Perturb costs.
        solver.iterations(dist_weights.data(), num_solver_itr, omega); // Dual iterations.

        const auto costs_before_dist = solver.get_solver_costs();
        solver.distribute_delta();
        const auto mms = solver.min_marginals_cuda(false);
        const auto& mms_0 = std::get<1>(mms);
        const auto& mms_1 = std::get<2>(mms);
        thrust::transform(mms_1.begin(), mms_1.end(), mms_0.begin(), final_mm_diff.begin(), thrust::minus<double>());
        thrust::device_vector<double> loss(final_mm_diff.size());
        thrust::device_vector<double> loss_sign_match(final_mm_diff.size());
        loss_func compute_loss({thrust::raw_pointer_cast(primal_var_index.data()),
                                thrust::raw_pointer_cast(final_mm_diff.data()),
                                thrust::raw_pointer_cast(expected_mm_diff.data()),
                                thrust::raw_pointer_cast(loss.data()),
                                thrust::raw_pointer_cast(loss_sign_match.data()),
                                solver.nr_variables()});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + solver.nr_layers(), compute_loss);
        const double loss_val = thrust::reduce(loss.begin(), loss.end());
        num_incorrect = thrust::reduce(loss_sign_match.begin(), loss_sign_match.end());
        prev_loss = loss_val;
        std::cout<<"Grad itr: "<<learning_itr<<", Loss: "<<loss_val<<", Num incorrect: "<<num_incorrect<<"/"<<solver.nr_layers()<<"\n";
        if (num_incorrect == 0.0)
            break;
        if (learning_itr > 0)
            avg_loss_improvement_per_itr += (prev_loss - loss_val);

        //Backward pass:
        loss_gradient_func compute_loss_grad({thrust::raw_pointer_cast(primal_var_index.data()),
                                            thrust::raw_pointer_cast(final_mm_diff.data()),
                                            thrust::raw_pointer_cast(expected_mm_diff.data()),
                                            thrust::raw_pointer_cast(loss_grad_mm.data()),
                                            solver.nr_variables()});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + solver.nr_layers(), compute_loss_grad);
        solver.grad_mm_diff_all_hops(loss_grad_mm.data(), grad_lo_costs.data(), grad_hi_costs.data());

        thrust::device_vector<double> grad_dist_weights(solver.nr_layers(), 0.0);
        thrust::device_vector<double> grad_def_mm(solver.nr_layers(), 0.0);
        thrust::device_vector<double> grad_omega(1, 0.0);
        
        solver.set_solver_costs(costs_before_dist);

        solver.grad_distribute_delta(grad_lo_costs.data(), grad_hi_costs.data(), grad_def_mm.data());

        solver.set_solver_costs(orig_costs); // reset to orig state.
        solver.update_costs(pert_lo, pert_hi); // Perturb costs to backprop through iterations().
        solver.grad_iterations(dist_weights.data(), grad_lo_costs.data(), grad_hi_costs.data(),
                                grad_def_mm.data(), grad_dist_weights.data(), grad_omega.data(),
                                omega, 0, num_solver_itr, 0);
        solver.set_solver_costs(orig_costs); // reset to orig state.
        solver.grad_cost_perturbation(grad_lo_costs.data(), grad_hi_costs.data(), grad_pert_lo.data(), grad_pert_hi.data());
        grad_step_pert grad_step_func({
            thrust::raw_pointer_cast(grad_pert_lo.data()),
            thrust::raw_pointer_cast(grad_pert_hi.data()),
            thrust::raw_pointer_cast(pert_lo.data()),
            thrust::raw_pointer_cast(pert_hi.data()),
            5e-2});
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + solver.nr_variables(), grad_step_func);
    }
    test(num_incorrect == 0.0);
}

int main(int argc, char** argv)
{
    std::vector<double> h_expected_mm_diff = {1.0, -1.0, 1.0, 1.0, -1.0, -1.0};
    thrust::device_vector<double> expected_mm_diff(h_expected_mm_diff.begin(), h_expected_mm_diff.end());
    test_problem(two_simplex, expected_mm_diff);
}
