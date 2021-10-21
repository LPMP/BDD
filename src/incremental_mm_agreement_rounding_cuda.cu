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
    struct fill_mm_type_func
    {
        mm_type* mm_types;
        const int nr_vars;

        __host__ __device__
        void operator()(const thrust::tuple<REAL, REAL, int> t) const
        {
            const REAL mm_0 = thrust::get<0>(t);
            const REAL mm_1 = thrust::get<1>(t);
            const int var = thrust::get<2>(t);
            if(var >= nr_vars)
                return;

            mm_type& cur_mm_type = mm_types[var];

            if(mm_0 + 1e-6 < mm_1)
            {
                if(cur_mm_type == mm_type::one || cur_mm_type == mm_type::inconsistent)
                    cur_mm_type = mm_type::inconsistent;
                else
                    cur_mm_type = mm_type::zero;
            }
            else if(mm_1 + 1e-6 < mm_0)
            {
                if(cur_mm_type == mm_type::zero || cur_mm_type == mm_type::inconsistent)
                    cur_mm_type = mm_type::inconsistent;
                else
                    cur_mm_type = mm_type::one; 
            }
            else 
            {
                assert(std::abs(mm_1 - mm_0) <= 1e-6);
            }
        }
    };

    template<typename REAL>
    thrust::device_vector<mm_type> compute_mm_types(const size_t nr_vars, const thrust::device_vector<REAL>& mm_0, const thrust::device_vector<REAL>& mm_1, const thrust::device_vector<int>& mm_vars)
    {   
        thrust::device_vector<mm_type> mm_types(nr_vars, mm_type::equal);
        auto it_begin = thrust::zip_iterator(thrust::make_tuple(mm_0.begin(), mm_1.begin(), mm_vars.begin()));
        auto it_end = thrust::zip_iterator(thrust::make_tuple(mm_0.end(), mm_1.end(), mm_vars.end()));
        fill_mm_type_func<REAL> func({thrust::raw_pointer_cast(mm_types.data()), nr_vars});
        thrust::for_each(it_begin, it_end, func);
        return mm_types;
    }   

    template<typename REAL>
    struct sum_mms_func
    {
        REAL* mm_sums_0;
        REAL* mm_sums_1;
        const int nr_vars;

        __host__ __device__
        void operator()(const thrust::tuple<REAL, REAL, int> t) const
        {
            const REAL mm_0 = thrust::get<0>(t);
            const REAL mm_1 = thrust::get<1>(t);
            const int var = thrust::get<2>(t);
            if(var >= nr_vars)
                return;

            atomicAdd(&mm_sums_0[var], mm_0);
            atomicAdd(&mm_sums_1[var], mm_1);
        }
    };

    template<typename REAL>
    std::tuple<thrust::device_vector<REAL>, thrust::device_vector<REAL>> compute_mm_sums(const size_t nr_vars, const thrust::device_vector<REAL>& mm_0, const thrust::device_vector<REAL>& mm_1, const thrust::device_vector<int>& mm_vars)
    {   
        assert(mm_0.size() == mm_vars.size());
        assert(mm_1.size() == mm_vars.size());
        thrust::device_vector<REAL> mm_sums_0(nr_vars, 0.0);
        thrust::device_vector<REAL> mm_sums_1(nr_vars, 0.0);
        auto it_begin = thrust::zip_iterator(thrust::make_tuple(mm_0.begin(), mm_1.begin(), mm_vars.begin()));
        auto it_end = thrust::zip_iterator(thrust::make_tuple(mm_0.end(), mm_1.end(), mm_vars.end()));
        sum_mms_func<REAL> func({thrust::raw_pointer_cast(mm_sums_0.data()), thrust::raw_pointer_cast(mm_sums_1.data()), nr_vars});
        thrust::for_each(it_begin, it_end, func);
        return {mm_sums_0, mm_sums_1};
    }   

    template<typename REAL>
    struct mm_types_transform {
        const REAL delta;

        __host__ __device__
        thrust::tuple<REAL, REAL> operator()(const thrust::tuple<mm_type,REAL,REAL> t)
        {
            const mm_type mmt = thrust::get<0>(t);
            const REAL mm_0 = thrust::get<1>(t);
            const REAL mm_1 = thrust::get<2>(t);

            if(mmt == mm_type::one)
            {
                return {delta, 0.0};
            }
            else if(mmt == mm_type::zero)
            {
                return {0.0, delta};
            }
            else if(mmt == mm_type::equal)
            {
                printf("equal min marginals not implemented\n");
                assert(false);
                // typically does not happen
            }
            else
            {   
                assert(mmt == mm_type::inconsistent);

                thrust::default_random_engine rng;
                thrust::uniform_real_distribution<float> dist(delta/5.0, delta);
                const int id = blockIdx.x * blockDim.x + threadIdx.x;
                rng.discard(id); // TODO: have other source for randomness here!
                const float r = dist(rng);

                if(mm_0 < mm_1)
                    return {0.0, 3.0*r};
                else
                    return {3.0*r, 0.0};
            }
        }
    };

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
        std::vector<char> incremental_mm_agreement_rounding_cuda(SOLVER& s, double init_delta, const double delta_growth_rate)
        {
            assert(delta_growth_rate > 0);
            assert(init_delta > 0);
            MEASURE_FUNCTION_EXECUTION_TIME;

            const auto start_time = std::chrono::steady_clock::now();

            std::cout << "[incremental primal rounding cuda] initial perturbation delta = " << init_delta << ", growth rate for perturbation " << delta_growth_rate << "\n";

            double cur_delta = 1.0/delta_growth_rate * init_delta;

            for(size_t round=0; round<10000; ++round)
            {
                cur_delta = cur_delta*delta_growth_rate;
                const auto time = std::chrono::steady_clock::now();
                const double time_elapsed = (double) std::chrono::duration_cast<std::chrono::milliseconds>(time - start_time).count() / 1000;
                std::cout << "[incremental primal rounding cuda] round " << round << ", cost delta " << cur_delta << ", time elapsed = " << time_elapsed << "\n";

                const auto mms = s.min_marginals_cuda();
                const thrust::device_vector<float>& mms_0 = std::get<0>(mms);
                const thrust::device_vector<float>& mms_1 = std::get<1>(mms);
                const thrust::device_vector<int>& primal_vars = s.primal_variable_index();
                const auto mm_types = compute_mm_types(s.nr_variables(), mms_0, mms_1, primal_vars);
                const size_t nr_one_mms = thrust::count(mm_types.begin(), mm_types.end(), mm_type::one);
                const size_t nr_zero_mms = thrust::count(mm_types.begin(), mm_types.end(), mm_type::zero);
                const size_t nr_equal_mms = thrust::count(mm_types.begin(), mm_types.end(), mm_type::equal);
                const size_t nr_inconsistent_mms = thrust::count(mm_types.begin(), mm_types.end(), mm_type::inconsistent);

                std::cout << "[incremental primal rounding cuda] " <<
                    "#one min-marg diffs = " << nr_one_mms << " " << u8"\u2258" << " " << double(100*nr_one_mms)/double(s.nr_variables()) << "%, " <<
                    "#zero min-marg diffs = " << nr_zero_mms << " " << u8"\u2258" << " " << double(100*nr_zero_mms)/double(s.nr_variables()) << "%, " <<
                    "#equal min-marg diffs = " << nr_equal_mms << " " << u8"\u2258" << " " << double(100*nr_equal_mms)/double(s.nr_variables()) << "%, " <<
                    "#inconsistent min-marg diffs = " << nr_inconsistent_mms << " " << u8"\u2258" << " " << double(100*nr_inconsistent_mms)/double(s.nr_variables()) << "%\n";

                // reconstruct solution from min-marginals
                if(nr_one_mms + nr_zero_mms == s.nr_variables())
                {
                    std::cout << "[incremental primal rounding cuda] reconstruct solution\n";
                    assert(mm_types.size() == s.nr_variables());
                    thrust::device_vector<char> device_sol(s.nr_variables());
                    thrust::transform(mm_types.begin(), mm_types.end(), device_sol.begin(), mm_type_to_sol{});
                    std::vector<char> sol(s.nr_variables());
                    thrust::copy(device_sol.begin(), device_sol.end(), sol.begin());
                    std::cout << "[incremental primal rounding cuda] reconstructed solution\n";
                    return sol;
                }

                thrust::device_vector<float> cost_delta_0(s.nr_variables());
                thrust::device_vector<float> cost_delta_1(s.nr_variables());

                const auto mm_sums = compute_mm_sums(s.nr_variables(), mms_0, mms_1, primal_vars);
                const auto& mm_sums_0 = std::get<0>(mm_sums);
                const auto& mm_sums_1 = std::get<1>(mm_sums);

                auto delta_it_begin = thrust::zip_iterator(thrust::make_tuple(cost_delta_0.begin(), cost_delta_1.begin()));
                auto it_begin = thrust::zip_iterator(thrust::make_tuple(mm_types.begin(), mm_sums_0.begin(), mm_sums_1.begin()));
                auto it_end = thrust::zip_iterator(thrust::make_tuple(mm_types.end(), mm_sums_0.end(), mm_sums_1.end()));
                thrust::transform(it_begin, it_end, delta_it_begin, mm_types_transform<typename SOLVER::value_type>{cur_delta});

                s.update_costs(cost_delta_0, cost_delta_1);
                for(size_t solver_iter=0; solver_iter<5; ++solver_iter)
                    s.iteration();
                std::cout << "[incremental primal rounding cuda] lower bound = " << s.lower_bound() << "\n";
            }

            std::cout << "[incremental primal rounding cuda] No solution found\n";
            return {};
        }

    template std::vector<char> incremental_mm_agreement_rounding_cuda(bdd_cuda_parallel_mma<float>& , double , const double );
    template std::vector<char> incremental_mm_agreement_rounding_cuda(bdd_cuda_parallel_mma<double>& , double , const double );
}
