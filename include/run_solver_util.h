#pragma once

#include <chrono>

namespace LPMP {

    template<typename SOLVER>
        void run_solver(SOLVER& s, const size_t max_iter, const double tolerance, const double improvement_slope, const double time_limit, const bool verbose = true)
        {
            assert(improvement_slope > 0.0 && improvement_slope < 1.0);
            assert(time_limit >= 0.0);
            assert(tolerance >= 0.0);

            if(verbose)
            {
                std::cout << "[bdd solver] termination criteria:\n";
                std::cout << "[bdd solver]     max iter = " << max_iter << "\n";
                std::cout << "[bdd solver]     time limit = " << time_limit << "s\n";
                std::cout << "[bdd solver]     tolerance = " << tolerance << ", lb_current-lb_prev < |tolerance*lb_prev|" << "\n";
                std::cout << "[bdd solver]     improvement_slope = " << improvement_slope << ", lb_current-lb_prev < tolerance*(lb_1-lb_0)" << "\n";
            }

            const auto start_time = std::chrono::steady_clock::now();
            const double lb_initial = s.lower_bound();
            double lb_first_iter = std::numeric_limits<double>::max();
            double lb_prev = lb_initial;
            double lb_post = lb_prev;
            if(verbose)
            {
                std::cout << "[bdd solver] initial lower bound = " << lb_prev;
                auto time = std::chrono::steady_clock::now();
                std::cout << ", time = " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000 << " s\n";
            }
            for(size_t iter=0; iter<max_iter; ++iter)
            {
                s.iteration();
                lb_prev = lb_post;
                lb_post = s.lower_bound();
                if(iter == 0)
                    lb_first_iter = lb_post;
                if(verbose)
                    std::cout << "[bdd solver] iteration " << iter << ", lower bound = " << lb_post;
                const auto time = std::chrono::steady_clock::now();
                double time_spent = (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000;
                if(verbose)
                    std::cout << ", time = " << time_spent << " s\n";
                if (time_spent > time_limit)
                {
                    if(verbose)
                        std::cout << "[bdd solver] Time limit reached." << std::endl;
                    break;
                }
                if (std::abs(lb_prev-lb_post) < std::abs(tolerance*lb_prev))
                {
                    if(verbose)
                        std::cout << "[bdd solver] Relative progress less than tolerance (" << tolerance << ")\n";
                    break;
                }
                if(std::abs(lb_prev - lb_post) < improvement_slope * std::abs(lb_initial - lb_first_iter))
                {
                    if(verbose)
                        std::cout << "[bdd solver] improvement smaller than " << 100*improvement_slope << "\% of initial improvement\n";
                    break;
                }
                if(lb_post == std::numeric_limits<double>::infinity())
                {
                    if(verbose)
                        std::cout << "[bdd solver] problem infeasible\n";
                    break;
                }
            }
            if(verbose)
                std::cout << "[bdd solver] final lower bound = " << s.lower_bound() << "\n"; 
        } 
}
