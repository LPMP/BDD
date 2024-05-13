#pragma once

#include <vector>
#include "bdd_collection/bdd_collection.h"
#include "time_measure_util.h"
#include "bdd_logging.h"
#include <deque>
#ifdef WITH_CUDA
#include <thrust/for_each.h>
#include <thrust/inner_product.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#endif

namespace LPMP {

#ifdef WITH_CUDA
using namespace thrust::placeholders;
#endif

// LBFGS requires the following functions to be implemented in the SOLVER base class:
// VECTOR bdd_solutions_vec()
// void make_dual_feasible(VECTOR)
// VECTOR net_solver_costs()
// size_t nr_layers()
// void gradient_step(VECTOR)

constexpr static int lbfgs_default_history_size = 5;
constexpr static double lbfgs_default_init_step_size = 1e-6;
constexpr static double lbfgs_default_req_rel_lb_increase = 1e-6;
constexpr static double lbfgs_default_step_size_decrease_factor = 0.8;
constexpr static double lbfgs_default_step_size_increase_factor = 1.1;

template<class SOLVER, typename VECTOR, typename REAL, typename INT_VECTOR, bool CUDA_SOLVER>
class lbfgs : public SOLVER
{
    public:

        lbfgs(const BDD::bdd_collection& bdd_col, 
        const int _history_size,
        const double _init_step_size,
        const double _req_rel_lb_increase,
        const double _step_size_decrease_factor,
        const double _step_size_increase_factor);

        lbfgs(const BDD::bdd_collection& bdd_col, 
        const std::vector<double>& costs_hi,
        const int _history_size,
        const double _init_step_size,
        const double _req_rel_lb_increase,
        const double _step_size_decrease_factor,
        const double _step_size_increase_factor);

        lbfgs(const BDD::bdd_collection& bdd_col)
        : lbfgs(bdd_col, lbfgs_default_history_size, lbfgs_default_init_step_size, lbfgs_default_req_rel_lb_increase, lbfgs_default_step_size_decrease_factor, lbfgs_default_step_size_increase_factor)
        {}

        lbfgs(const BDD::bdd_collection& bdd_col, const std::vector<double>& costs_hi)
        : lbfgs(bdd_col, costs_hi, lbfgs_default_history_size, lbfgs_default_init_step_size, lbfgs_default_req_rel_lb_increase, lbfgs_default_step_size_decrease_factor, lbfgs_default_step_size_increase_factor)
        {}

        void iteration();

        void update_costs(const std::vector<REAL> &cost_lo, const std::vector<REAL> &cost_hi);
#ifdef WITH_CUDA
        void update_costs(const thrust::device_vector<REAL> &cost_0, const thrust::device_vector<REAL> &cost_1);
#endif

    private:
        void store_iterate(const INT_VECTOR &grad_f);
        VECTOR compute_update_direction(const INT_VECTOR& grad_f);
        void search_step_size_and_apply(const VECTOR &update);
        void flush_lbfgs_states();
        bool lbfgs_update_possible() const;
        void next_itr_without_storage();

        struct history
        {
            VECTOR s; // difference of x
            INT_VECTOR y; // different of grad_x f(x) in {-1, 0, 1}.
            REAL rho_inv;
        };
        std::deque<history> history;

        VECTOR prev_x;
        INT_VECTOR prev_grad_f;
        int m;
        double step_size;
        //double init_lb_increase;
        //bool init_lb_valid = false;
        double required_relative_lb_increase, step_size_decrease_factor, step_size_increase_factor;
        int num_unsuccessful_lbfgs_updates_ = 0;
        double initial_rho_inv = 0.0;

        std::deque<REAL> lb_history;

        bool prev_states_stored = false;
        bool initial_rho_inv_valid = false;

        double mma_lb_increase_per_time = 0.0;
        double lbfgs_lb_increase_per_time = 0.0;
        size_t mma_iterations = 0;
        size_t lbfgs_iterations = 0;

        enum class solver_type {mma, lbfgs};
        solver_type choose_solver() const;

        void mma_iteration();
        void lbfgs_iteration(const INT_VECTOR& grad_f);
};

}
