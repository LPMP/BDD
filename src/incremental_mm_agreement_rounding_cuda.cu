#include "incremental_mm_agreement_rounding_cuda.h"
#include <iostream>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include "cuda_utils.h"
#include "time_measure_util.h"

#include "bdd_cuda_parallel_mma.h"
#include "bdd_cuda_learned_mma.h"
#include "run_solver_util.h"

namespace LPMP {

    namespace {
        enum class mm_type {
            zero,
            one,
            equal,
            inconsistent,
        };  
    }   

    template<typename REAL>
    struct mm_diff_direction_func {
        __host__ __device__ char operator()(const thrust::tuple<REAL, REAL> t) const
        {
            REAL mm_0 = thrust::get<0>(t);
            REAL mm_1 = thrust::get<1>(t);
            if(mm_0 + 1e-6 <= mm_1)
                return -1;
            else if(mm_1 + 1e-6 <= mm_0)
                return 1;
            else 
                return 0;
        }
    };

    struct fill_mm_type_func
    {
        mm_type* mm_types;
        const unsigned long nr_vars;

        __host__ __device__
        void operator()(const thrust::tuple<char, char, int> t) const
        {
            const char mm_min = thrust::get<0>(t);
            const char mm_max = thrust::get<1>(t);
            const int var = thrust::get<2>(t);
            if(var >= nr_vars)
                return;

            if (mm_min > 0)
                mm_types[var] = mm_type::one;
            else if (mm_max < 0)
                mm_types[var] = mm_type::zero;
            else if(mm_max == 0 && mm_min == 0)
                mm_types[var] = mm_type::equal;
            else
                mm_types[var] = mm_type::inconsistent;
        }
    };


    template<typename T>
    struct tuple_min_max
    {
        __host__ __device__
        thrust::tuple<T, T> operator()(const thrust::tuple<T, T>& t0, const thrust::tuple<T, T>& t1)
        {
            return thrust::make_tuple(min(thrust::get<0>(t0), thrust::get<0>(t1)), max(thrust::get<1>(t0), thrust::get<1>(t1)));
        }
    };

    template<typename REAL>
    thrust::device_vector<mm_type> compute_mm_types(const size_t nr_vars, const thrust::device_vector<REAL>& mm_0, const thrust::device_vector<REAL>& mm_1, const thrust::device_vector<int>& mm_vars)
    {
        assert(thrust::is_sorted(mm_vars.begin(), mm_vars.end()));
        thrust::device_vector<char> mm_diff_direction(mm_0.size());
        {
            auto first = thrust::make_zip_iterator(thrust::make_tuple(mm_0.begin(), mm_1.begin()));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(mm_0.end(), mm_1.end()));
            thrust::transform(first, last, mm_diff_direction.begin(), mm_diff_direction_func<REAL>());
        }

        thrust::device_vector<char> mm_diff_min(nr_vars + 1);
        thrust::device_vector<char> mm_diff_max(nr_vars + 1);
        {
            auto first_val = thrust::make_zip_iterator(thrust::make_tuple(mm_diff_direction.begin(), mm_diff_direction.begin()));
            auto first_out_val = thrust::make_zip_iterator(thrust::make_tuple(mm_diff_min.begin(), mm_diff_max.begin()));

            thrust::equal_to<int> binary_pred;
            auto new_end = thrust::reduce_by_key(mm_vars.begin(), mm_vars.end(), first_val, thrust::make_discard_iterator(), first_out_val, binary_pred, tuple_min_max<char>());
            const int out_size = thrust::distance(first_out_val, new_end.second);
            assert(out_size == mm_diff_min.size());
        }
        thrust::device_vector<mm_type> mm_types(nr_vars);
        {
            auto first_val = thrust::make_zip_iterator(thrust::make_tuple(mm_diff_min.begin(), mm_diff_max.begin(), thrust::make_counting_iterator<int>(0)));
            auto last_val = thrust::make_zip_iterator(thrust::make_tuple(mm_diff_min.end(), mm_diff_max.end(),  thrust::make_counting_iterator<int>(0) + mm_diff_min.size()));

            thrust::for_each(first_val, last_val, fill_mm_type_func({thrust::raw_pointer_cast(mm_types.data()), nr_vars}));
        }

        return mm_types;
    }   

    template<typename REAL>
    struct tuple_sum
    {
        __host__ __device__
        thrust::tuple<REAL, REAL> operator()(const thrust::tuple<REAL, REAL>& t0, const thrust::tuple<REAL, REAL>& t1)
        {
            return thrust::make_tuple(thrust::get<0>(t0) + thrust::get<0>(t1), thrust::get<1>(t0) + thrust::get<1>(t1));
        }
    };

    template<typename REAL>
    std::tuple<thrust::device_vector<REAL>, thrust::device_vector<REAL>> compute_mm_sums(const size_t nr_vars, const thrust::device_vector<REAL>& mm_0, const thrust::device_vector<REAL>& mm_1, const thrust::device_vector<int>& mm_vars)
    {   
        assert(mm_0.size() == mm_vars.size());
        assert(mm_1.size() == mm_vars.size());
        assert(thrust::is_sorted(mm_vars.begin(), mm_vars.end()));
        thrust::device_vector<REAL> mm_sums_0(nr_vars + 1);
        thrust::device_vector<REAL> mm_sums_1(nr_vars + 1);

        auto first_val = thrust::make_zip_iterator(thrust::make_tuple(mm_0.begin(), mm_1.begin()));
        auto first_out_val = thrust::make_zip_iterator(thrust::make_tuple(mm_sums_0.begin(), mm_sums_1.begin()));

        thrust::equal_to<int> binary_pred;
        auto new_end = thrust::reduce_by_key(mm_vars.begin(), mm_vars.end(), first_val, thrust::make_discard_iterator(), first_out_val, binary_pred, tuple_sum<REAL>());
        const int out_size = thrust::distance(first_out_val, new_end.second);
        assert(out_size == mm_sums_0.size());

        // remove terminal nodes:
        mm_sums_0.resize(nr_vars);
        mm_sums_1.resize(nr_vars);
        return {mm_sums_0, mm_sums_1};
    }

    template<typename REAL>
    struct mm_types_transform {
        const REAL delta;
        const REAL max_incon_mm_diff;
        const REAL decay_factor_inconsistent;
        const bool only_perturb_inconsistent;

        __host__ __device__
        thrust::tuple<REAL, REAL> operator()(const thrust::tuple<mm_type,REAL,REAL,int> t) const
        {
            const mm_type mmt = thrust::get<0>(t);
            const REAL mm_0 = thrust::get<1>(t);
            const REAL mm_1 = thrust::get<2>(t);

            if(mmt == mm_type::one)
            {
                if (!only_perturb_inconsistent)
                    return {delta, 0.0};
                else
                    return {0.0, 0.0};
            }
            else if(mmt == mm_type::zero)
            {
                if (!only_perturb_inconsistent)
                    return {0.0, delta};
                else
                    return {0.0, 0.0};
            }
            else
            {
                thrust::default_random_engine rng;
                thrust::uniform_real_distribution<float> dist(-delta, delta);
                const int id = blockIdx.x * blockDim.x + threadIdx.x;
                rng.discard(id); // TODO: have other source for randomness here!
                const float r = dist(rng);
                if(mmt == mm_type::equal)
                {
                    if(r < 0.0)
                        return {abs(r)*delta, 0.0};
                    else
                        return {0.0, abs(r)*delta};
                }

                else
                {
                    if(mm_0 < mm_1)
                        return {0.0, abs(r)*delta};
                    else
                        return {abs(r)*delta, 0.0};
                }
            }
            // else
            // {
            //     const REAL cur_abs_mm_diff = abs(mm_0 - mm_1);
            //     if(mm_0 < mm_1)
            //         return {0.0, delta * pow(cur_abs_mm_diff / max_incon_mm_diff, decay_factor_inconsistent)};
            //     else
            //         return {delta * pow(cur_abs_mm_diff / max_incon_mm_diff, decay_factor_inconsistent), 0.0};
            // }
        }
    };

    template<typename REAL>
    struct mm_abs_diff_func
    {
        __host__ __device__
        REAL operator()(const REAL& m1, const REAL& m0)
        {
            return abs(m1 - m0);
        }
    };

    template<typename REAL>
    struct is_consistent_func
    {
        const mm_type* mm_types;
        __host__ __device__
        bool operator()(const thrust::tuple<REAL, int>& t) const
        {
            const int var = thrust::get<1>(t);
            return mm_types[var] != mm_type::inconsistent; 
        }
    };

    template<typename REAL>
    REAL compute_max_inconsistent_mm_diff(const thrust::device_vector<mm_type>& mm_types, const thrust::device_vector<REAL>& mm_sum_0, const thrust::device_vector<REAL>& mm_sum_1)
    {
        thrust::device_vector<REAL> mm_abs_diff(mm_sum_0.size());
        thrust::transform(mm_sum_1.begin(), mm_sum_1.end(), mm_sum_0.begin(), mm_abs_diff.begin(), mm_abs_diff_func<REAL>());
        
        thrust::device_vector<int> inconsistent_primal_vars(mm_types.size());
        thrust::sequence(inconsistent_primal_vars.begin(), inconsistent_primal_vars.end());

        auto first = thrust::make_zip_iterator(thrust::make_tuple(mm_abs_diff.begin(), inconsistent_primal_vars.begin()));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(mm_abs_diff.end(), inconsistent_primal_vars.end()));

        auto new_last = thrust::remove_if(first, last, is_consistent_func<REAL>({thrust::raw_pointer_cast(mm_types.data())}));
        const int num_inconsistent = thrust::distance(first, new_last);
        mm_abs_diff.resize(num_inconsistent);
        return *thrust::max_element(mm_abs_diff.begin(), mm_abs_diff.end());
    }

struct mm_type_to_sol {
    __host__ __device__
        char operator()(const mm_type t)
        {
            if(t == mm_type::one)
                return 1;
            else if(t == mm_type::zero)
                return 0;
            assert(false);
            return -1; 
        }
};

    template<typename SOLVER>
        std::vector<char> incremental_mm_agreement_rounding_cuda(SOLVER& s, double init_delta, const double delta_growth_rate, const int num_itr_lb, const bool verbose)
        {
            assert(delta_growth_rate > 0);
            assert(init_delta > 0);
            MEASURE_FUNCTION_EXECUTION_TIME;

            const auto start_time = std::chrono::steady_clock::now();
            s.distribute_delta();
            const double lb = s.lower_bound();
            if (verbose)
            {
                std::cout<<"Lower bound after distributing delta: "<<lb<<"\n";
                std::cout << "[incremental primal rounding cuda] initial perturbation delta = " << init_delta << ", growth rate for perturbation " << delta_growth_rate << "\n";
            }

            double cur_delta = 1.0/delta_growth_rate * init_delta;

            for(size_t round=0; round<500; ++round)
            {
                cur_delta = min(cur_delta*delta_growth_rate, 1e6);
                const auto time = std::chrono::steady_clock::now();
                const double time_elapsed = (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000;
                if (verbose) std::cout << "[incremental primal rounding cuda] round " << round << ", cost delta " << cur_delta << ", time elapsed = " << time_elapsed << "\n";
                
                s.distribute_delta();
                const auto mms = s.min_marginals_cuda();
                const thrust::device_vector<int>& primal_vars = std::get<0>(mms);
                const auto& mms_0 = std::get<1>(mms);
                const auto& mms_1 = std::get<2>(mms);
                const auto mm_types = compute_mm_types(s.nr_variables(), mms_0, mms_1, primal_vars);
                const size_t nr_one_mms = thrust::count(mm_types.begin(), mm_types.end(), mm_type::one);
                const size_t nr_zero_mms = thrust::count(mm_types.begin(), mm_types.end(), mm_type::zero);
                const size_t nr_equal_mms = thrust::count(mm_types.begin(), mm_types.end(), mm_type::equal);
                const size_t nr_inconsistent_mms = thrust::count(mm_types.begin(), mm_types.end(), mm_type::inconsistent);
                if (nr_inconsistent_mms == 1)
                {
                    const size_t incon_index = thrust::distance(mm_types.begin(), thrust::find(mm_types.begin(), mm_types.end(), mm_type::inconsistent));
                    if (verbose) std::cout<<"Inconsistent index: "<<incon_index<<"\n";
                }

                if (verbose)
                {
                    std::cout << "[incremental primal rounding cuda] " <<
                    "#one min-marg diffs = " << nr_one_mms << " " << u8"\u2258" << " " << double(100*nr_one_mms)/double(s.nr_variables()) << "%, " <<
                    "#zero min-marg diffs = " << nr_zero_mms << " " << u8"\u2258" << " " << double(100*nr_zero_mms)/double(s.nr_variables()) << "%, " <<
                    "#equal min-marg diffs = " << nr_equal_mms << " " << u8"\u2258" << " " << double(100*nr_equal_mms)/double(s.nr_variables()) << "%, " <<
                    "#inconsistent min-marg diffs = " << nr_inconsistent_mms << " " << u8"\u2258" << " " << double(100*nr_inconsistent_mms)/double(s.nr_variables()) << "%\n";
                }
                // reconstruct solution from min-marginals
                if(nr_one_mms + nr_zero_mms == s.nr_variables())
                {
                    if (verbose) std::cout << "[incremental primal rounding cuda] reconstruct solution\n";
                    assert(mm_types.size() == s.nr_variables());
                    thrust::device_vector<char> device_sol(s.nr_variables());
                    thrust::transform(mm_types.begin(), mm_types.end(), device_sol.begin(), mm_type_to_sol{});
                    std::vector<char> sol(s.nr_variables());
                    thrust::copy(device_sol.begin(), device_sol.end(), sol.begin());
                    if (verbose) std::cout << "[incremental primal rounding cuda] reconstructed solution\n"<<"[incremental primal rounding cuda] Lower bound with 0 delta: "<<lb<<"\n";
                    return sol;
                }

                thrust::device_vector<double> cost_delta_0(s.nr_variables());
                thrust::device_vector<double> cost_delta_1(s.nr_variables());

                const auto mm_sums = compute_mm_sums(s.nr_variables(), mms_0, mms_1, primal_vars);
                const auto& mm_sums_0 = std::get<0>(mm_sums);
                const auto& mm_sums_1 = std::get<1>(mm_sums);
                const auto max_incon_mm_diff = compute_max_inconsistent_mm_diff(mm_types, mm_sums_0, mm_sums_1);

                auto delta_it_begin = thrust::zip_iterator(thrust::make_tuple(cost_delta_0.begin(), cost_delta_1.begin()));
                auto first = thrust::zip_iterator(thrust::make_tuple(mm_types.begin(), mm_sums_0.begin(), mm_sums_1.begin(), thrust::make_counting_iterator<int>(0)));
                auto last = thrust::zip_iterator(thrust::make_tuple(mm_types.end(), mm_sums_0.end(), mm_sums_1.end(), thrust::make_counting_iterator<int>(0) + mm_types.size()));

                thrust::transform(first, last, delta_it_begin, mm_types_transform<typename SOLVER::value_type>{cur_delta, max_incon_mm_diff, 2.0, false}); //nr_inconsistent_mms == 1});

                s.update_costs(cost_delta_0, cost_delta_1);
                run_solver(s, num_itr_lb, 1e-7, 0.0001, std::numeric_limits<double>::max(), false);
                if (verbose) std::cout << "[incremental primal rounding cuda] lower bound = " << s.lower_bound() << "\n";
            }

            if (verbose) std::cout << "[incremental primal rounding cuda] No solution found\n";
            return {};
        }

    template std::vector<char> incremental_mm_agreement_rounding_cuda(bdd_cuda_parallel_mma<float>& , double , const double, const int, const bool );
    template std::vector<char> incremental_mm_agreement_rounding_cuda(bdd_cuda_parallel_mma<double>& , double , const double, const int, const bool );
    template std::vector<char> incremental_mm_agreement_rounding_cuda(bdd_cuda_learned_mma<float>& , double , const double, const int, const bool );
    template std::vector<char> incremental_mm_agreement_rounding_cuda(bdd_cuda_learned_mma<double>& , double , const double, const int, const bool );
}
