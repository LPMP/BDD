#pragma once

#include "bdd_collection/bdd_collection.h"
#include "bdd_logging.h"
#include <limits>
#ifdef WITH_CUDA
#include <thrust/device_vector.h>
#endif

namespace LPMP
{

    template <class SOLVER, typename REAL>
    class subgradient : public SOLVER
    {
    public:
        subgradient(const BDD::bdd_collection &bdd_col) : SOLVER(bdd_col) {}

        void iteration();

    private:
        void subgradient_step();
        void adaptive_subgradient_step(); // as in Section 7.5 of "MRF Energy Minimization and Beyond via Dual Decomposition" by Komodakis et al.
        
        REAL best_lb = -std::numeric_limits<REAL>::infinity();
        REAL delta = 1.0;
        bool dual_improvement = false;
        size_t iteration_ = 0;
    };

    template <class SOLVER, typename REAL>
    void subgradient<SOLVER, REAL>::iteration()
    {
        iteration_++;
        //subgradient_step();
        adaptive_subgradient_step();
    }

    template <class SOLVER, typename REAL>
    void subgradient<SOLVER, REAL>::adaptive_subgradient_step()
    {
        // parameters
        constexpr REAL rho_0 = 1.5;
        constexpr REAL rho_1 = 0.7;
        constexpr REAL delta_min = 0.001;

        if (best_lb == -std::numeric_limits<REAL>::infinity())
            best_lb = this->lower_bound();

        auto g = this->template bdds_solution_vec<REAL>();
        this->make_dual_feasible(g.begin(), g.end());
        const REAL g_norm_squared = std::inner_product(g.begin(), g.end(), g.begin(), 0.0);
        delta = dual_improvement ? rho_0 * delta : std::max(rho_1 * delta, delta_min);
        const REAL step_size = (best_lb - this->lower_bound() + delta) / (g_norm_squared + 0.0001);
        //bdd_log << "[subgradient] step size: " << step_size << ", gradient norm squared: " << g_norm_squared << ", best lb: " << best_lb << ", delta: " << delta << "\n";
        this->gradient_step(g.begin(), g.end(), step_size);

        if(this->lower_bound() > best_lb)
        {
            best_lb = this->lower_bound();
            dual_improvement = true;
        }
        else
            dual_improvement = false;
    }

    template <class SOLVER, typename REAL>
    void subgradient<SOLVER, REAL>::subgradient_step()
    {
        auto g = this->template bdds_solution_vec<REAL>();
        this->make_dual_feasible(g.begin(), g.end());
        const REAL step_size = 0.1 / REAL(iteration_);
        bdd_log << "[subgradient] step size: " << step_size << "\n";
        this->gradient_step(g.begin(), g.end(), step_size);
    }

}