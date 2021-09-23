#include "bdd_cuda_parallel_mma_sorting.h"
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/discard_iterator.h>

namespace LPMP {

    struct is_terminal_func {
        __host__ __device__ bool operator()(const thrust::tuple<int, float>& t)
        {
            return thrust::get<0>(t) < 0;
        }
    };

    struct normalize_by_num_vars_in_bdd_func {
        int* num_vars_per_bdd;
        __host__ __device__ void operator()(thrust::tuple<int, float&> t) const
        {
            const int bdd_idx = thrust::get<0>(t);
            float& mm_diff = thrust::get<1>(t);

            mm_diff /= num_vars_per_bdd[bdd_idx];
        }
    };

    struct compute_update_func {
        int* num_bdds_per_var;
        float* sum_diff_primal_var;
        __host__ __device__ void operator()(thrust::tuple<int, float&> t) const
        {
            const int primal_index = thrust::get<0>(t);
            if(primal_index < 0)
                return; // Nothing needs to be done for terminal nodes.

            float& update = thrust::get<1>(t); // Here the value corresponds to mm_diff / num_vars
            update = sum_diff_primal_var[primal_index] / num_bdds_per_var[primal_index] - update;
        }
    };

    template<typename T>
    thrust::device_vector<int> repeat_values(const thrust::device_vector<T>& values, const thrust::device_vector<int>& counts)
    {
        thrust::device_vector<int> counts_sum(counts.size() + 1);
        counts_sum[0] = 0;
        thrust::inclusive_scan(counts.begin(), counts.end(), counts_sum.begin() + 1);
        
        int out_size = counts_sum.back();
        thrust::device_vector<int> output_indices(out_size, 0);

        thrust::scatter(thrust::constant_iterator<int>(1), thrust::constant_iterator<int>(1) + values.size(), counts_sum.begin(), output_indices.begin());

        thrust::inclusive_scan(output_indices.begin(), output_indices.end(), output_indices.begin());
        thrust::transform(output_indices.begin(), output_indices.end(), thrust::make_constant_iterator(1), output_indices.begin(), thrust::minus<int>());

        thrust::device_vector<T> out_values(out_size);
        thrust::gather(output_indices.begin(), output_indices.end(), values.begin(), out_values.begin());

        return out_values;
    }

    void bdd_cuda_parallel_mma_sorting::iteration()
    {
        initialize_costs();
        thrust::device_vector<int> mm_primal_index, mm_bdd_index;
        thrust::device_vector<float> diff_1_0;
        {
            thrust::device_vector<float> mm_0;
            std::tie(mm_primal_index, mm_bdd_index, mm_0, diff_1_0) = min_marginals_cuda();

            // Compute min-marginal difference (mm_1 - mm_0) and store in diff_1_0
            thrust::transform(diff_1_0.begin(), diff_1_0.end(), mm_0.begin(), diff_1_0.begin(), thrust::minus<float>());
        }

        // For now, sort such that same primal variables are contiguous. TODO Returned min-marginals should already satisfy this condition. 
        auto first_key = thrust::make_zip_iterator(thrust::make_tuple(mm_primal_index.begin(), mm_bdd_index.begin()));
        auto last_key = thrust::make_zip_iterator(thrust::make_tuple(mm_primal_index.end(), mm_bdd_index.end()));

        thrust::sort_by_key(first_key, last_key, diff_1_0.begin()); 

        // Normalize by number of variables in the BDD.
        {
            normalize_by_num_vars_in_bdd_func func_bdd_norm({thrust::raw_pointer_cast(num_vars_per_bdd_.data())});
            auto first = thrust::make_zip_iterator(thrust::make_tuple(mm_bdd_index.begin(), diff_1_0.begin()));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(mm_bdd_index.end(), diff_1_0.end()));

            thrust::for_each(first, last, func_bdd_norm);
        }

        // Compute sum of min-marginal differences per primal variable
        thrust::device_vector<float> sum_diff_primal_var(diff_1_0.size());
        {
            thrust::device_vector<int> sum_diff_primal_index(diff_1_0.size());
            auto last_primal = thrust::reduce_by_key(mm_primal_index.begin(), mm_primal_index.end(), diff_1_0.begin(), 
                                                    sum_diff_primal_index.begin(), sum_diff_primal_var.begin());
            const int num_primal_vars_inc_terminals = std::distance(sum_diff_primal_var.begin(), last_primal.second);
            
            // Now remove terminal nodes from these:
            auto first = thrust::make_zip_iterator(thrust::make_tuple(sum_diff_primal_index.begin(), sum_diff_primal_var.begin()));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(sum_diff_primal_index.begin() + num_primal_vars_inc_terminals, 
                                                                            sum_diff_primal_var.begin() + num_primal_vars_inc_terminals));
            auto last_valid_primal = thrust::remove_if(first, last, is_terminal_func());
            sum_diff_primal_var.resize(std::distance(first, last_valid_primal));
            // Here sum_diff_primal_index should resemble a thrust::sequence starting from 0.
        }

        // Compute the update:
        {
            compute_update_func update_func({thrust::raw_pointer_cast(num_bdds_per_var_.data()), thrust::raw_pointer_cast(sum_diff_primal_var.data())});
            auto first = thrust::make_zip_iterator(thrust::make_tuple(mm_primal_index.begin(), diff_1_0.begin()));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(mm_primal_index.end(), diff_1_0.end()));

            thrust::for_each(first, last, update_func);
        }
        // diff_1_0 now contains the update.
        // Now replicate the update values since each primal variable can occur multiple times within a BDD:
        diff_1_0 = repeat_values(diff_1_0, bdd_layer_width_);

        thrust::device_vector<float> diff_1_0_sorted(diff_1_0.size());
        thrust::gather(sorting_order_.begin(), sorting_order_.end(), diff_1_0.begin(), diff_1_0_sorted.begin());

        assert(diff_1_0_sorted.size() == hi_cost_.size());
        thrust::transform(hi_cost_.begin(), hi_cost_.end(), diff_1_0_sorted.begin(), hi_cost_.begin(), thrust::plus<float>());
        forward_state_valid_ = false;
        backward_state_valid_ = false;
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