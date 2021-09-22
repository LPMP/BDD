#include "bdd_cuda_parallel_mma_sorting.h"
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>

namespace LPMP {
    void bdd_cuda_parallel_mma_sorting::iteration()
    {
        initialize_costs();
        forward_run();
        backward_run();
    }

    void bdd_cuda_parallel_mma_sorting::solve(const size_t max_iter, const double tolerance, const double time_limit)
    {
        const auto start_time = std::chrono::steady_clock::now();
        double lb_prev = this->lower_bound();
        double lb_post = lb_prev;
        std::cout << "initial lower bound = " << lb_prev;
        auto time = std::chrono::steady_clock::now();
        std::cout << ", time = " << (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000 << " s";
        std::cout << "\n";
        for(size_t iter=0; iter<max_iter; ++iter)
        {
            iteration();
            lb_prev = lb_post;
            lb_post = this->lower_bound();
            std::cout << "iteration " << iter << ", lower bound = " << lb_post;
            time = std::chrono::steady_clock::now();
            double time_spent = (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000;
            std::cout << ", time = " << time_spent << " s";
            std::cout << "\n";
            if (time_spent > time_limit)
            {
                std::cout << "Time limit reached." << std::endl;
                break;
            }
            if (std::abs(lb_prev-lb_post) < std::abs(tolerance*lb_prev))
            {
                std::cout << "Relative progress less than tolerance (" << tolerance << ")\n";
                break;
            }
        }
        std::cout << "final lower bound = " << this->lower_bound() << "\n"; 
    }
}