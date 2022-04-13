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

const char * matching_3x3 = 
R"(Minimize
-2 x_11 - 1 x_12 - 1 x_13
-1 x_21 - 2 x_22 - 1 x_23
-1 x_31 - 1 x_32 - 2 x_33
Subject To
x_11 + x_12 + x_13 = 1
x_21 + x_22 + x_23 = 1
x_31 + x_32 + x_33 = 1
x_11 + x_21 + x_31 = 1
x_12 + x_22 + x_32 = 1
x_13 + x_23 + x_33 = 1
End)";

const char * short_chain_shuffled = 
R"(Minimize
+ 1 mu_2_1 + 1 mu_10 + 0 mu_1_1 + 0 mu_11
-1 mu_1_0 + 1 mu_00 + 2 mu_01 + 2 mu_2_0
Subject To
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_00 + mu_10 + mu_01 + mu_11 = 1
mu_1_0 - mu_00 - mu_01 = 0
mu_1_1 - mu_10 - mu_11 = 0
mu_2_0 - mu_00 - mu_10 = 0
mu_2_1 - mu_01 - mu_11 = 0
End)";

const char * long_chain = 
R"(Minimize
2 mu_0_0 - 1 mu_0_1 + 3 mu_1_0 - 1 mu_1_1
+ 3 mu_2_0 + 2 mu_2_1 - 1 mu_3_0 - 2 mu_3_1
- 2 mu_4_0 - 1 mu_4_1 + 1 mu_5_0 - 1 mu_5_1
+ 1 mu_6_0 + 1 mu_6_1 - 3 mu_7_0 + 2 mu_7_1
+ 0 mu_8_0 + 2 mu_8_1
+ 1 mu_01_00 - 2 mu_01_01 + 2 mu_01_10 - 1 mu_01_11
+ 0 mu_12_00 - 1 mu_12_01 + 1 mu_12_10 + 0 mu_12_11
- 1 mu_23_00 + 2 mu_23_01 + 1 mu_23_10 - 2 mu_23_11
+ 2 mu_34_00 + 0 mu_34_01 + 2 mu_34_10 + 2 mu_34_11
+ 1 mu_45_00 - 2 mu_45_01 - 3 mu_45_10 - 1 mu_45_11
- 2 mu_56_00 + 0 mu_56_01 + 1 mu_56_10 + 3 mu_56_11
- 1 mu_67_00 - 2 mu_67_01 - 1 mu_67_10 - 1 mu_67_11
+ 2 mu_78_00 + 0 mu_78_01 + 2 mu_78_10 + 3 mu_78_11
Subject To
mu_0_0 + mu_0_1 = 1
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_3_0 + mu_3_1 = 1
mu_4_0 + mu_4_1 = 1
mu_5_0 + mu_5_1 = 1
mu_6_0 + mu_6_1 = 1
mu_7_0 + mu_7_1 = 1
mu_8_0 + mu_8_1 = 1
mu_01_00 + mu_01_10 + mu_01_01 + mu_01_11 = 1
mu_12_00 + mu_12_10 + mu_12_01 + mu_12_11 = 1
mu_23_00 + mu_23_10 + mu_23_01 + mu_23_11 = 1
mu_34_00 + mu_34_10 + mu_34_01 + mu_34_11 = 1
mu_45_00 + mu_45_10 + mu_45_01 + mu_45_11 = 1
mu_56_00 + mu_56_10 + mu_56_01 + mu_56_11 = 1
mu_67_00 + mu_67_10 + mu_67_01 + mu_67_11 = 1
mu_78_00 + mu_78_10 + mu_78_01 + mu_78_11 = 1
mu_0_0 - mu_01_00 - mu_01_01 = 0
mu_0_1 - mu_01_10 - mu_01_11 = 0
mu_1_0 - mu_01_00 - mu_01_10 = 0
mu_1_1 - mu_01_01 - mu_01_11 = 0
mu_1_0 - mu_12_00 - mu_12_01 = 0
mu_1_1 - mu_12_10 - mu_12_11 = 0
mu_2_0 - mu_12_00 - mu_12_10 = 0
mu_2_1 - mu_12_01 - mu_12_11 = 0
mu_2_0 - mu_23_00 - mu_23_01 = 0
mu_2_1 - mu_23_10 - mu_23_11 = 0
mu_3_0 - mu_23_00 - mu_23_10 = 0
mu_3_1 - mu_23_01 - mu_23_11 = 0
mu_3_0 - mu_34_00 - mu_34_01 = 0
mu_3_1 - mu_34_10 - mu_34_11 = 0
mu_4_0 - mu_34_00 - mu_34_10 = 0
mu_4_1 - mu_34_01 - mu_34_11 = 0
mu_4_0 - mu_45_00 - mu_45_01 = 0
mu_4_1 - mu_45_10 - mu_45_11 = 0
mu_5_0 - mu_45_00 - mu_45_10 = 0
mu_5_1 - mu_45_01 - mu_45_11 = 0
mu_5_0 - mu_56_00 - mu_56_01 = 0
mu_5_1 - mu_56_10 - mu_56_11 = 0
mu_6_0 - mu_56_00 - mu_56_10 = 0
mu_6_1 - mu_56_01 - mu_56_11 = 0
mu_6_0 - mu_67_00 - mu_67_01 = 0
mu_6_1 - mu_67_10 - mu_67_11 = 0
mu_7_0 - mu_67_00 - mu_67_10 = 0
mu_7_1 - mu_67_01 - mu_67_11 = 0
mu_7_0 - mu_78_00 - mu_78_01 = 0
mu_7_1 - mu_78_10 - mu_78_11 = 0
mu_8_0 - mu_78_00 - mu_78_10 = 0
mu_8_1 - mu_78_01 - mu_78_11 = 0
End)";

const char * grid_graph_3x3 = 
R"(Minimize
2 mu_0_0 - 1 mu_0_1 + 3 mu_1_0 - 1 mu_1_1
+ 3 mu_2_0 + 2 mu_2_1 - 1 mu_3_0 - 2 mu_3_1
- 2 mu_4_0 - 1 mu_4_1 + 3 mu_5_0 - 1 mu_5_1
+ 1 mu_6_0 + 1 mu_6_1 - 3 mu_7_0 + 2 mu_7_1
+ 0 mu_8_0 + 2 mu_8_1
+ 1 mu_01_00 - 2 mu_01_01 + 2 mu_01_10 - 1 mu_01_11
+ 0 mu_12_00 + 1 mu_12_01 + 1 mu_12_10 + 0 mu_12_11
- 1 mu_03_00 + 2 mu_03_01 + 0 mu_03_10 - 2 mu_03_11
+ 2 mu_14_00 + 0 mu_14_01 + 2 mu_14_10 + 2 mu_14_11
+ 1 mu_25_00 - 2 mu_25_01 - 3 mu_25_10 - 1 mu_25_11
+ 0 mu_34_00 + 1 mu_34_01 + 1 mu_34_10 + 1 mu_34_11
- 1 mu_45_00 - 2 mu_45_01 + 4 mu_45_10 - 2 mu_45_11
- 2 mu_36_00 + 0 mu_36_01 + 1 mu_36_10 + 3 mu_36_11
+ 3 mu_47_00 - 2 mu_47_01 - 2 mu_47_10 - 1 mu_47_11
+ 0 mu_58_00 + 1 mu_58_01 + 1 mu_58_10 + 1 mu_58_11
- 1 mu_67_00 + 2 mu_67_01 - 1 mu_67_10 - 1 mu_67_11
+ 2 mu_78_00 + 0 mu_78_01 + 2 mu_78_10 + 2 mu_78_11
Subject To
mu_0_0 + mu_0_1 = 1
mu_1_0 + mu_1_1 = 1
mu_2_0 + mu_2_1 = 1
mu_3_0 + mu_3_1 = 1
mu_4_0 + mu_4_1 = 1
mu_5_0 + mu_5_1 = 1
mu_6_0 + mu_6_1 = 1
mu_7_0 + mu_7_1 = 1
mu_8_0 + mu_8_1 = 1
mu_01_00 + mu_01_10 + mu_01_01 + mu_01_11 = 1
mu_12_00 + mu_12_10 + mu_12_01 + mu_12_11 = 1
mu_03_00 + mu_03_10 + mu_03_01 + mu_03_11 = 1
mu_14_00 + mu_14_10 + mu_14_01 + mu_14_11 = 1
mu_25_00 + mu_25_10 + mu_25_01 + mu_25_11 = 1
mu_34_00 + mu_34_10 + mu_34_01 + mu_34_11 = 1
mu_45_00 + mu_45_10 + mu_45_01 + mu_45_11 = 1
mu_36_00 + mu_36_10 + mu_36_01 + mu_36_11 = 1
mu_47_00 + mu_47_10 + mu_47_01 + mu_47_11 = 1
mu_58_00 + mu_58_10 + mu_58_01 + mu_58_11 = 1
mu_67_00 + mu_67_10 + mu_67_01 + mu_67_11 = 1
mu_78_00 + mu_78_10 + mu_78_01 + mu_78_11 = 1
mu_0_0 - mu_01_00 - mu_01_01 = 0
mu_0_1 - mu_01_10 - mu_01_11 = 0
mu_0_0 - mu_03_00 - mu_03_01 = 0
mu_0_1 - mu_03_10 - mu_03_11 = 0
mu_1_0 - mu_01_00 - mu_01_10 = 0
mu_1_1 - mu_01_01 - mu_01_11 = 0
mu_1_0 - mu_12_00 - mu_12_01 = 0
mu_1_1 - mu_12_10 - mu_12_11 = 0
mu_1_0 - mu_14_00 - mu_14_01 = 0
mu_1_1 - mu_14_10 - mu_14_11 = 0
mu_2_0 - mu_12_00 - mu_12_10 = 0
mu_2_1 - mu_12_01 - mu_12_11 = 0
mu_2_0 - mu_25_00 - mu_25_01 = 0
mu_2_1 - mu_25_10 - mu_25_11 = 0
mu_3_0 - mu_03_00 - mu_03_10 = 0
mu_3_1 - mu_03_01 - mu_03_11 = 0
mu_3_0 - mu_34_00 - mu_34_01 = 0
mu_3_1 - mu_34_10 - mu_34_11 = 0
mu_3_0 - mu_36_00 - mu_36_01 = 0
mu_3_1 - mu_36_10 - mu_36_11 = 0
mu_4_0 - mu_14_00 - mu_14_10 = 0
mu_4_1 - mu_14_01 - mu_14_11 = 0
mu_4_0 - mu_34_00 - mu_34_10 = 0
mu_4_1 - mu_34_01 - mu_34_11 = 0
mu_4_0 - mu_45_00 - mu_45_01 = 0
mu_4_1 - mu_45_10 - mu_45_11 = 0
mu_4_0 - mu_47_00 - mu_47_01 = 0
mu_4_1 - mu_47_10 - mu_47_11 = 0
mu_5_0 - mu_25_00 - mu_25_10 = 0
mu_5_1 - mu_25_01 - mu_25_11 = 0
mu_5_0 - mu_45_00 - mu_45_10 = 0
mu_5_1 - mu_45_01 - mu_45_11 = 0
mu_5_0 - mu_58_00 - mu_58_01 = 0
mu_5_1 - mu_58_10 - mu_58_11 = 0
mu_6_0 - mu_36_00 - mu_36_10 = 0
mu_6_1 - mu_36_01 - mu_36_11 = 0
mu_6_0 - mu_67_00 - mu_67_01 = 0
mu_6_1 - mu_67_10 - mu_67_11 = 0
mu_7_0 - mu_47_00 - mu_47_10 = 0
mu_7_1 - mu_47_01 - mu_47_11 = 0
mu_7_0 - mu_67_00 - mu_67_10 = 0
mu_7_1 - mu_67_01 - mu_67_11 = 0
mu_7_0 - mu_78_00 - mu_78_01 = 0
mu_7_1 - mu_78_10 - mu_78_11 = 0
mu_8_0 - mu_58_00 - mu_58_10 = 0
mu_8_1 - mu_58_01 - mu_58_11 = 0
mu_8_0 - mu_78_00 - mu_78_10 = 0
mu_8_1 - mu_78_01 - mu_78_11 = 0
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
    const unsigned long num_vars;
    __device__ void operator()(const int i)
    {
        const int primal_var = primal_index[i];
        if (primal_var < num_vars)
            loss[i] = abs(pred[i] - target[i]);
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
            if (pred[i] > target[i])
                grad[i] = 1.0;
            else
                grad[i] = -1.0;
        }
        else
            grad[i] = 0.0;
    }
};

struct grad_step_dist_w {
    const double* grad_dist_w;
    double* dist_w;
    double lr;
    __device__ void operator()(const int i)
    {
        dist_w[i] = dist_w[i] - lr * grad_dist_w[i];
    }
};

struct grad_step_min_marginals {
    const double* grad_lo;
    const double* grad_hi;
    double* lo_costs;
    double* hi_costs;
    double lr;
    __device__ void operator()(const int i)
    {
        lo_costs[i] = lo_costs[i] - lr * grad_lo[i];
        hi_costs[i] = hi_costs[i] - lr * grad_hi[i];
    }
};

thrust::device_vector<double> compute_expected_mm_diff(const char* instance)
{
    ILP_input ilp = ILP_parser::parse_string(instance);
    bdd_preprocessor bdd_pre(ilp);
    bdd_collection bdd_col = bdd_pre.get_bdd_collection();
    bdd_cuda_parallel_mma<double> solver(bdd_col);

    for(size_t i=0; i<solver.nr_variables(); ++i)
        solver.set_cost(ilp.objective()[i], i);

    for(size_t iter=0; iter<200; ++iter)
        solver.iteration();

    thrust::device_vector<double> expected_mm_diff(solver.nr_layers());
    const auto mms = solver.min_marginals_cuda(false);
    const auto& mms_0 = std::get<1>(mms);
    const auto& mms_1 = std::get<2>(mms);
    thrust::transform(mms_1.begin(), mms_1.end(), mms_0.begin(), expected_mm_diff.begin(), thrust::minus<double>());
    return expected_mm_diff;
}

void test_problem(const char* instance, const double expected_lb, const double omega = 0.5, const double tol = 1e-12)
{
    const thrust::device_vector<double> expected_mm_diff = compute_expected_mm_diff(instance);
    ILP_input ilp = ILP_parser::parse_string(instance);
    bdd_preprocessor bdd_pre(ilp);
    bdd_collection bdd_col = bdd_pre.get_bdd_collection();
    bdd_cuda_learned_mma<double> solver(bdd_col);

    for(size_t i=0; i<solver.nr_variables(); ++i)
        solver.set_cost(ilp.objective()[i], i);

    thrust::device_vector<double> dist_weights(solver.nr_layers(), 1.0);
    const thrust::device_vector<int> primal_var_index = solver.get_primal_variable_index();


    thrust::device_vector<double> final_mm_diff(solver.nr_layers());
    thrust::device_vector<double> loss_grad_mm(solver.nr_layers());
    thrust::device_vector<double> grad_lo_costs(solver.nr_layers());
    thrust::device_vector<double> grad_hi_costs(solver.nr_layers());

    auto initial_costs = solver.get_solver_costs();
    const double initial_lb = solver.lower_bound();
    const int num_solver_itr = 50;
    double prev_loss = 0;
    double avg_loss_improvement_per_itr = 0;
    const int num_learning_itr = 5;
    for(int learning_itr = 0; learning_itr < num_learning_itr; learning_itr++)
    {
        solver.set_solver_costs(initial_costs); // reset to initial state.

        // Forward pass:
        project_dist_weights(solver, dist_weights, primal_var_index);
        // test(initial_lb == solver.lower_bound());
        solver.iterations(dist_weights.data(), num_solver_itr, omega);
        const auto costs_before_dist = solver.get_solver_costs();
        solver.distribute_delta();
        const auto costs_after_dist = solver.get_solver_costs();
        const auto mms = solver.min_marginals_cuda(false);
        const auto& mms_0 = std::get<1>(mms);
        const auto& mms_1 = std::get<2>(mms);
        thrust::transform(mms_1.begin(), mms_1.end(), mms_0.begin(), final_mm_diff.begin(), thrust::minus<double>());
        thrust::device_vector<double> loss(final_mm_diff.size());
        const thrust::device_vector<int> pv = solver.get_primal_variable_index();
        loss_func compute_loss({thrust::raw_pointer_cast(pv.data()),
                                thrust::raw_pointer_cast(final_mm_diff.data()),
                                thrust::raw_pointer_cast(expected_mm_diff.data()),
                                thrust::raw_pointer_cast(loss.data()),
                                solver.nr_variables()});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + solver.nr_layers(), compute_loss);
        const double loss_val = thrust::reduce(loss.begin(), loss.end());
        if (learning_itr > 0)
            avg_loss_improvement_per_itr += (prev_loss - loss_val);
        prev_loss = loss_val;
        std::cout<<"Grad itr: "<<learning_itr<<", Loss: "<<loss_val<<", LB: "<<solver.lower_bound()<<", Max. possible LB:  "<<expected_lb<<"\n";

        //Backward pass:
        loss_gradient_func compute_loss_grad({thrust::raw_pointer_cast(solver.get_primal_variable_index().data()),
                                            thrust::raw_pointer_cast(final_mm_diff.data()),
                                            thrust::raw_pointer_cast(expected_mm_diff.data()),
                                            thrust::raw_pointer_cast(loss_grad_mm.data()),
                                            solver.nr_variables()});

        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + solver.nr_layers(), compute_loss_grad);
        solver.grad_mm_diff_all_hops(loss_grad_mm.data(), grad_lo_costs.data(), grad_hi_costs.data());

        // Check min_marginal derivative sign:
        // const auto final_costs = solver.get_solver_costs();
        // thrust::device_vector<double> pert_lo_costs(std::get<0>(final_costs));
        // thrust::device_vector<double> pert_hi_costs(std::get<1>(final_costs));
        // grad_step_min_marginals grad_step_mm_func({
        //                                         thrust::raw_pointer_cast(grad_lo_costs.data()),
        //                                         thrust::raw_pointer_cast(grad_hi_costs.data()),
        //                                         thrust::raw_pointer_cast(pert_lo_costs.data()),
        //                                         thrust::raw_pointer_cast(pert_hi_costs.data()),
        //                                         5e-3});
        // thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + solver.nr_layers(), grad_step_mm_func);
        // initial_costs = std::make_tuple(pert_lo_costs, pert_hi_costs, std::get<2>(initial_costs), std::get<3>(initial_costs));
        // solver.set_solver_costs(std::make_tuple(pert_lo_costs, pert_hi_costs, std::get<2>(final_costs), std::get<3>(final_costs)));
        // const auto pmms = solver.min_marginals_cuda(false);
        // const auto& pmms_0 = std::get<1>(pmms);
        // const auto& pmms_1 = std::get<2>(pmms);
        // thrust::transform(pmms_1.begin(), pmms_1.end(), pmms_0.begin(), final_mm_diff.begin(), thrust::minus<double>());
        // thrust::device_vector<double> lossp(final_mm_diff.size());
        // loss_func compute_lossp({thrust::raw_pointer_cast(solver.get_primal_variable_index().data()),
        //                         thrust::raw_pointer_cast(final_mm_diff.data()),
        //                         thrust::raw_pointer_cast(expected_mm_diff.data()),
        //                         thrust::raw_pointer_cast(lossp.data()),
        //                         solver.nr_variables()});

        // thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + solver.nr_layers(), compute_lossp);
        // const double loss_valp = thrust::reduce(lossp.begin(), lossp.end());
        // std::cout<<"Grad itr: "<<learning_itr<<", Loss: "<<loss_valp<<", LB: "<<solver.lower_bound()<<", Max. possible:  "<<expected_lb<<"\n";

        thrust::device_vector<double> grad_dist_weights(solver.nr_layers(), 0.0);
        thrust::device_vector<double> grad_def_mm(solver.nr_layers(), 0.0);
        thrust::device_vector<double> grad_omega(1);

        solver.set_solver_costs(costs_before_dist);
        solver.grad_distribute_delta(grad_lo_costs.data(), grad_hi_costs.data(), grad_def_mm.data());

        solver.set_solver_costs(initial_costs); // reset to initial state.
        solver.grad_iterations(dist_weights.data(), grad_lo_costs.data(), grad_hi_costs.data(), 
                                grad_def_mm.data(), grad_dist_weights.data(), grad_omega.data(),
                                omega, 0, num_solver_itr, 10);
        
        grad_step_dist_w grad_step_func({
            thrust::raw_pointer_cast(grad_dist_weights.data()),
            thrust::raw_pointer_cast(dist_weights.data()),
            2.5e-4});
        thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + solver.nr_layers(), grad_step_func);
    }
    avg_loss_improvement_per_itr = avg_loss_improvement_per_itr / num_learning_itr;
    std::cout<<"\n Avg. loss improvment per gradient iteration: "<<avg_loss_improvement_per_itr<<" (should be positive)\n";
    test(avg_loss_improvement_per_itr > tol);

    solver.set_solver_costs(initial_costs); // reset to initial state.
    project_dist_weights(solver, dist_weights, primal_var_index);
    solver.iterations(dist_weights.data(), num_solver_itr, omega);
    solver.distribute_delta();
    std::cout<<"Final lower bound: "<<solver.lower_bound()<<", Max. possible:  "<<expected_lb<<"\n\n\n";

    // Check feasibility:
    std::vector<double> cost_vector_after = solver.get_primal_objective_vector_host();
    for(size_t i=0; i<solver.nr_variables(); ++i)
    {
        const auto diff = std::abs(ilp.objective()[i] - cost_vector_after[i]);
        std::stringstream buffer;
        buffer<<i<<" "<<ilp.objective()[i]<<" "<<cost_vector_after[i]<<" "<<diff<<"\n";
        test(diff <= tol, buffer.str());
    }
}

int main(int argc, char** argv)
{
    std::cout<<"two_simplex"<<"\n";
    test_problem(two_simplex, 3.0);
    std::cout<<"matching_3x3"<<"\n";
    test_problem(matching_3x3, -6.0);
    std::cout<<"short_chain_shuffled"<<"\n";
    test_problem(short_chain_shuffled, 1.0);
    std::cout<<"long_chain"<<"\n";
    test_problem(long_chain, -9.0);
    std::cout<<"grid_graph_3x3"<<"\n";
    test_problem(grid_graph_3x3, -8.0);
}
