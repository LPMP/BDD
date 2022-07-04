#include "bdd_multi_parallel_mma_base.h"
#include "two_dimensional_variable_array.hxx"
#include <limits>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <unordered_set>
#include <future>
#include "time_measure_util.h"
// TODO: remove
#include <stdio.h>

namespace LPMP {

    std::array<BDD::bdd_collection,2> split_bdd_collection(BDD::bdd_collection& bdd_col, const size_t gpu_th, const size_t cpu_th)
    {
        assert(gpu_th < cpu_th);
        // TODO: make parallel
        std::vector<size_t> nr_bdd_nodes_per_layer;
        std::vector<size_t> nr_bdds_with_hops;
        size_t nr_variables = 0;
        for (size_t bdd_nr = 0; bdd_nr < bdd_col.nr_bdds(); ++bdd_nr)
        {
            assert(bdd_col.is_qbdd(bdd_nr));
            const auto bdd_vars = bdd_col.variables(bdd_nr);
            nr_variables = std::max(nr_variables, *std::max_element(bdd_vars.begin(), bdd_vars.end()) + 1);
            if (bdd_vars.size() > nr_bdd_nodes_per_layer.size())
            {
                nr_bdd_nodes_per_layer.resize(bdd_vars.size(), 0);
                nr_bdds_with_hops.resize(bdd_vars.size()+1, 0);
            }
            ++nr_bdds_with_hops[bdd_vars.size()];
            size_t prev_var = bdd_col.root_variable(bdd_nr);
            size_t layer = 0;
            for (auto bdd_it = bdd_col.begin(bdd_nr); bdd_it != bdd_col.end(bdd_nr); ++bdd_it)
            {
                const auto& bdd_instr = *bdd_it;
                const size_t bdd_var = bdd_instr.index;
                if(bdd_var != prev_var)
                {
                    prev_var = bdd_var;
                    ++layer;
                }
                assert(layer < nr_bdd_nodes_per_layer.size());
                ++nr_bdd_nodes_per_layer[layer];
            }
            assert(layer+1 == bdd_vars.size());
        }

        for(size_t i=0; i<nr_bdds_with_hops.size(); ++i)
            if(nr_bdds_with_hops[i] > 0)
                std::cout << "Hop " << i << ": #BDDs = " << nr_bdds_with_hops[i] << "\n";

        if(nr_bdd_nodes_per_layer.size() < gpu_th)
        {
            std::cout << "[multi parallel mma base] all BDDs shorter than gpu threshold, put all BDDs onto GPU\n";
            return {BDD::bdd_collection(), bdd_col};
        }

        constexpr static size_t min_nr_bdd_nodes_per_layer = 1024;
        // first layer such that all subsequent ones have fewer than min_nr_bdd_nodes_per_layer BDDs per hop
        std::ptrdiff_t layer_th = std::min(cpu_th, nr_bdd_nodes_per_layer.size()-1);
        for(; layer_th>gpu_th; --layer_th)
        {
            if(nr_bdd_nodes_per_layer[layer_th] >= min_nr_bdd_nodes_per_layer)
            {
                layer_th++;
                break;
            }
        }
        std::cout << "[multi parallel mma base] Computed hop threshold for CPU/GPU split = " << layer_th << ", preset GPU threshold = " << gpu_th << ", preset CPU threshold << " << cpu_th << "\n";

        if(layer_th + 10 > nr_bdd_nodes_per_layer.size() && layer_th + 10 < cpu_th)
        {
            std::cout << "[multi parallel mma base] Too few slim layers, optimize everything on GPU\n";
            return {BDD::bdd_collection(), bdd_col};
        }

        std::cout << "[multi parallel mma base] Optimize BDDs with more than " << layer_th << " hops on CPU\n";

        std::vector<size_t> cpu_bdds;
        std::vector<size_t> gpu_bdds;
        size_t nr_cpu_bdd_nodes = 0;
        size_t nr_gpu_bdd_nodes = 0;
        std::unordered_set<size_t> cpu_vars;
        std::vector<size_t> nr_cpu_bdds_with_hops(nr_bdds_with_hops.size(), 0);
        std::unordered_set<size_t> gpu_vars;
        std::vector<size_t> nr_gpu_bdds_with_hops(nr_bdds_with_hops.size(), 0);

        for(size_t bdd_nr=0; bdd_nr<bdd_col.nr_bdds(); ++bdd_nr)
        {
            const auto bdd_vars = bdd_col.variables(bdd_nr);
            if(bdd_vars.size() > layer_th)
            {
                cpu_bdds.push_back(bdd_nr);
                cpu_vars.insert(bdd_vars.begin(), bdd_vars.end());
                nr_cpu_bdd_nodes += bdd_col.nr_bdd_nodes(bdd_nr);
                ++nr_cpu_bdds_with_hops[bdd_vars.size()];
            }
            else
            {
                gpu_bdds.push_back(bdd_nr);
                gpu_vars.insert(bdd_vars.begin(), bdd_vars.end());
                nr_gpu_bdd_nodes += bdd_col.nr_bdd_nodes(bdd_nr);
                ++nr_gpu_bdds_with_hops[bdd_vars.size()];
            }
        }

        std::cout << "[multi parallel mma base] #CPU BDDs = " << cpu_bdds.size() << ", #GPU BDDs = " << gpu_bdds.size() << "\n";
        std::cout << "[multi parallel mma base] #CPU vars = " << cpu_vars.size() << "/" << nr_variables << " variables = " << 100.0 * double(cpu_vars.size())/double(nr_variables) << "%, #GPU vars = " << gpu_vars.size() << "/" << nr_variables << " variables = " << 100.0 * double(gpu_vars.size())/double(nr_variables) << "%\n";
        std::cout << "[multi parallel mma base] #CPU BDD nodes = " << nr_cpu_bdd_nodes << " = " << 100.0 * double(nr_cpu_bdd_nodes)/double(nr_cpu_bdd_nodes + nr_gpu_bdd_nodes) << "%, #GPU BDD nodes = " << nr_gpu_bdd_nodes << " = " << 100.0 * double(nr_gpu_bdd_nodes)/double(nr_cpu_bdd_nodes + nr_gpu_bdd_nodes) << "%\n";

        for(size_t i=0; i<nr_cpu_bdds_with_hops.size(); ++i)
            if(nr_cpu_bdds_with_hops[i] > 0)
                std::cout << "CPU Hop " << i << ": #BDDs = " << nr_cpu_bdds_with_hops[i] << "\n";

        for(size_t i=0; i<nr_gpu_bdds_with_hops.size(); ++i)
            if(nr_gpu_bdds_with_hops[i] > 0)
                std::cout << "GPU Hop " << i << ": #BDDs = " << nr_gpu_bdds_with_hops[i] << "\n";

        BDD::bdd_collection cpu_bdd_col = bdd_col;
        cpu_bdd_col.remove(gpu_bdds.begin(), gpu_bdds.end());
        BDD::bdd_collection cuda_bdd_col = bdd_col;
        cuda_bdd_col.remove(cpu_bdds.begin(), cpu_bdds.end());

        return {cpu_bdd_col, cuda_bdd_col};
    }

    template<typename REAL>
    bdd_multi_parallel_mma_base<REAL>::bdd_multi_parallel_mma_base(BDD::bdd_collection &cpu_bdd_col, BDD::bdd_collection &cuda_bdd_col)
        : cpu_base(cpu_bdd_col),
        cuda_base(cuda_bdd_col)
    {
        std::vector<size_t> cpu_nr_bdds_per_var;
        cpu_nr_bdds_per_var.reserve(cpu_base.nr_variables());
        for (size_t i = 0; i < cpu_base.nr_variables(); ++i)
            cpu_nr_bdds_per_var.push_back(cpu_base.nr_bdds(i));
        total_nr_bdds_per_var_ = thrust::device_vector<size_t>(nr_variables(), 0);
        thrust::copy(cpu_nr_bdds_per_var.begin(), cpu_nr_bdds_per_var.end(), total_nr_bdds_per_var_.begin());
        thrust::transform(
                cuda_base.get_num_bdds_per_var().begin(), cuda_base.get_num_bdds_per_var().end(), 
                total_nr_bdds_per_var_.begin(),
                total_nr_bdds_per_var_.begin(),
                thrust::plus<size_t>()
                );

        // TODO: possibly initialize in parallel_mma, where it is used
        gpu_delta_ = thrust::device_vector<REAL>(2 * cuda_base.nr_variables(), 0.0);

        // TODO: same
        cpu_delta_ = std::vector<std::array<REAL, 2>>(cpu_base.nr_variables(), {0.0, 0.0});
        
        gpu_nr_bdds_per_var_ = cuda_base.get_num_bdds_per_var();

        for (size_t i=0; i<nr_variables(); ++i)
            assert(total_nr_bdds_per_var_[i] == nr_bdds(i));
    }

    template <typename REAL>
    bdd_multi_parallel_mma_base<REAL>::bdd_multi_parallel_mma_base(BDD::bdd_collection &bdd_col)
    {
        auto [cpu_bdds, gpu_bdds] = split_bdd_collection(bdd_col);
        *this = bdd_multi_parallel_mma_base<REAL>(cpu_bdds, gpu_bdds);
    }

    template <typename REAL>
    template <typename COST_ITERATOR>
    void bdd_multi_parallel_mma_base<REAL>::update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end)
    {
        std::vector<double> lo_costs_cpu(cpu_base.nr_variables(), 0.0), hi_costs_cpu(cpu_base.nr_variables(), 0.0);
        std::vector<double> lo_costs_gpu(cuda_base.nr_variables(), 0.0), hi_costs_gpu(cuda_base.nr_variables(), 0.0);

        auto get_lo_cost = [&](const size_t var) -> double
        {
            assert(nr_bdds(var) > 0);
            if (var < std::distance(cost_lo_begin, cost_lo_end) && var < nr_variables())
                return *(cost_lo_begin + var);
            else
                return 0.0;
        };

        auto get_hi_cost = [&](const size_t var) -> double
        {
            assert(nr_bdds(var) > 0);
            if (var < std::distance(cost_hi_begin, cost_hi_end) && var < nr_variables())
                return *(cost_hi_begin + var);
            else
                return 0.0;
        };

        for (size_t var = 0; var < nr_variables(); ++var)
        {
            const size_t nr_cpu_bdds = var < cpu_base.nr_variables() ? cpu_base.nr_bdds(var) : 0;
            const size_t nr_gpu_bdds = var < cuda_base.nr_variables() ? gpu_nr_bdds_per_var_[var] : 0;
            assert(nr_cpu_bdds + nr_gpu_bdds > 0);
            assert(nr_cpu_bdds + nr_gpu_bdds == nr_bdds(var));

            const double cpu_ratio = double(nr_cpu_bdds) / double(nr_cpu_bdds + nr_gpu_bdds);
            const double gpu_ratio = double(nr_gpu_bdds) / double(nr_cpu_bdds + nr_gpu_bdds);
            assert(std::abs(cpu_ratio + gpu_ratio) - 1.0 < 1e-6);

            if (var < cpu_base.nr_variables())
            {
                lo_costs_cpu[var] = cpu_ratio * get_lo_cost(var);
                hi_costs_cpu[var] = cpu_ratio * get_hi_cost(var);
            }
            if (var < cuda_base.nr_variables())
            {
                lo_costs_gpu[var] = gpu_ratio * get_lo_cost(var);
                hi_costs_gpu[var] = gpu_ratio * get_hi_cost(var);
            }
        }

        cpu_base.update_costs(lo_costs_cpu.begin(), lo_costs_cpu.end(), hi_costs_cpu.begin(), hi_costs_cpu.end());
        cuda_base.update_costs(lo_costs_gpu.begin(), lo_costs_gpu.end(), hi_costs_gpu.begin(), hi_costs_gpu.end());
    }

    template <typename REAL>
    size_t bdd_multi_parallel_mma_base<REAL>::nr_bdds() const
    {
        return cpu_base.nr_bdds() + cuda_base.nr_bdds();
    }

    template <typename REAL>
    size_t bdd_multi_parallel_mma_base<REAL>::nr_bdds(const size_t var) const
    {
        const size_t nr_cpu_bdds = var < cpu_base.nr_variables() ? cpu_base.nr_bdds(var) : 0;
        const size_t nr_gpu_bdds = var < cuda_base.nr_variables() ? gpu_nr_bdds_per_var_[var] : 0;
        return nr_cpu_bdds + nr_gpu_bdds;
    }

    template <typename REAL>
    size_t bdd_multi_parallel_mma_base<REAL>::nr_variables() const
    {
        return std::max(cpu_base.nr_variables(), cuda_base.nr_variables());
    }

    // TODO: remove?
    template <typename REAL>
    size_t bdd_multi_parallel_mma_base<REAL>::nr_variables(const size_t bdd_nr) const
    {
        throw std::runtime_error("not supported yet");
    }

    template <typename REAL>
    double bdd_multi_parallel_mma_base<REAL>::lower_bound()
    {
        return cpu_base.lower_bound() + cuda_base.lower_bound();
    }

    template <typename REAL>
    void bdd_multi_parallel_mma_base<REAL>::add_to_constant(const double c)
    {
        cpu_base.add_to_constant(c);
    }

    template <typename REAL>
    struct copy_delta_to_cpu_functor
    {
        __device__ __host__ std::array<REAL,2> operator()(thrust::tuple<REAL,REAL> x)
            {
                return {thrust::get<0>(x), thrust::get<1>(x)};
            }
    };

    // Must be executed before from_gpu?
    template<typename REAL>
        void bdd_multi_parallel_mma_base<REAL>::accumulate_delta_from_cpu(
                const std::vector<std::array<REAL,2>>& cpu_delta,
                thrust::device_vector<REAL>& accumulated)
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
            assert(cpu_delta.size() == cpu_base.nr_variables());
            assert(accumulated.size() == 2*nr_variables());

            thrust::fill(accumulated.begin() + 2*cpu_base.nr_variables(), accumulated.end(), 0.0);
            thrust::copy((REAL*)&cpu_delta[0], (REAL*)&cpu_delta[0] + 2*cpu_base.nr_variables(), accumulated.begin());
        }

    template<typename REAL>
        void bdd_multi_parallel_mma_base<REAL>::accumulate_delta_from_gpu(const thrust::device_vector<REAL>& gpu_delta, thrust::device_vector<REAL>& accumulated)
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
            assert(gpu_delta.size() == 2*cuda_base.nr_variables());
            assert(accumulated.size() == 2*nr_variables());

            thrust::transform(
                    accumulated.begin(), accumulated.begin() + 2*cuda_base.nr_variables(),
                    gpu_delta.begin(), 
                    accumulated.begin(), 
                    thrust::plus<REAL>()
                    );
        }

    template<typename REAL>
        void bdd_multi_parallel_mma_base<REAL>::split_delta_to_gpu(const thrust::device_vector<REAL>& total_delta, thrust::device_vector<REAL>& gpu_delta)
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
            assert(total_delta.size() == 2*nr_variables());
            assert(gpu_delta.size() == 2*cuda_base.nr_variables());

            thrust::copy(total_delta.begin(), total_delta.begin() + 2*cuda_base.nr_variables(), gpu_delta.begin());
            // TODO: really needed?
            // set those elements to zero where the cuda solver has no BDDs?
        }

    template<typename REAL>
        void bdd_multi_parallel_mma_base<REAL>::split_delta_to_cpu(
                const thrust::device_vector<REAL>& total_delta,
                std::vector<std::array<REAL, 2>>& cpu_delta)
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
            assert(total_delta.size() == 2*nr_variables());
            assert(cpu_delta.size() == cpu_base.nr_variables());

            thrust::copy(total_delta.begin(), total_delta.begin() + 2*cpu_base.nr_variables(), (REAL*)&cpu_delta[0]);
            // TODO: set components to zero without BDDs?
        }

    template <typename REAL>
    void bdd_multi_parallel_mma_base<REAL>::forward_mm(const REAL omega, thrust::device_vector<REAL>& delta)
    {
        assert(delta.size() == 2*nr_variables());
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("hybrid mma forward mm");
            split_delta_to_gpu(delta, gpu_delta_);
            split_delta_to_cpu(delta, cpu_delta_);
            auto cpu_fut = std::async(std::launch::async, [&]() { 
                    cpu_base.forward_mm(omega, cpu_delta_);
                    accumulate_delta_from_cpu(cpu_delta_, delta);
                    });
            cuda_base.forward_mm(omega, gpu_delta_);
            cpu_fut.wait();
            accumulate_delta_from_gpu(gpu_delta_, delta);
        }
    }

    template <typename REAL>
    void bdd_multi_parallel_mma_base<REAL>::backward_mm(const REAL omega, thrust::device_vector<REAL>& delta)
    {
        assert(delta.size() == 2*nr_variables());
        {
            MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME2("hybrid mma forward mm");
            split_delta_to_cpu(delta, cpu_delta_);
            split_delta_to_gpu(delta, gpu_delta_);
            auto cpu_fut = std::async(std::launch::async, [&]() { 
                    cpu_base.backward_mm(omega, cpu_delta_);
                    accumulate_delta_from_cpu(cpu_delta_, delta);
                    });
            cuda_base.backward_mm(omega, gpu_delta_);
            cpu_fut.wait();
            accumulate_delta_from_gpu(gpu_delta_, delta);
        }
    }

    template <typename REAL>
    void bdd_multi_parallel_mma_base<REAL>::parallel_mma()
    {
        if(total_delta_.size() == 0)
            total_delta_ = thrust::device_vector<REAL>(2*nr_variables(), 0.0);
        else
            assert(total_delta_.size() == 2*nr_variables());

        forward_mm(0.5, total_delta_);
        /*
        for(size_t i=0; i<nr_variables(); ++i)
        {
            if(nr_bdds(i) > 0)
            {
                total_delta_[2*i] /= nr_bdds(i);
                total_delta_[2*i+1] /= nr_bdds(i);
            }
            else
            {
                total_delta_[2*i] = 0.0;
                total_delta_[2*i+1] = 0.0;
            }
        }
        */
        normalize_delta(total_delta_);
        backward_mm(0.5, total_delta_);
        normalize_delta(total_delta_);
        /*
        for(size_t i=0; i<nr_variables(); ++i)
        {
            if(nr_bdds(i) > 0)
            {
                total_delta_[2*i] /= nr_bdds(i);
                total_delta_[2*i+1] /= nr_bdds(i);
            }
            else
            {
                total_delta_[2*i] = 0.0;
                total_delta_[2*i+1] = 0.0;
            }
        }
        */
    }

    template <typename REAL>
    struct normalize_delta_func
    {
        REAL* delta;
        const REAL* nr_bdds;

        __host__ __device__ void operator()(const size_t i) const
        {
            const size_t norm = nr_bdds[i];
            if(norm > 0)
            {
                delta[2*i] /= norm;
                delta[2*i+1] /= norm;
            }
            else
            {
                delta[2*i] = 0.0;
                delta[2*i+1] = 0.0;
            }
        }
    };

    template <typename REAL>
    void bdd_multi_parallel_mma_base<REAL>::normalize_delta(thrust::device_vector<REAL>& delta) const
    {
        assert(delta.size() == 2*nr_variables());
        normalize_delta_func<REAL> func {
            thrust::raw_pointer_cast(delta.data()), 
            thrust::raw_pointer_cast(total_nr_bdds_per_var_.data())
        };
        thrust::for_each_n(thrust::make_counting_iterator<size_t>(0), nr_variables(), func);
    }

    template <typename REAL>
    two_dim_variable_array<std::array<double, 2>> bdd_multi_parallel_mma_base<REAL>::min_marginals()
    {
        const auto cpu_mms = cpu_base.min_marginals();
        const auto gpu_mms = cuda_base.min_marginals();

        two_dim_variable_array<std::array<double, 2>> mms;
        std::vector<std::array<double, 2>> cur_mms;
        for (size_t i = 0; i < nr_variables(); ++i)
        {
            cur_mms.clear();
            if (i < cpu_mms.size())
                for (size_t j = 0; j < cpu_mms.size(i); ++j)
                    cur_mms.push_back(cpu_mms(i, j));
            if (i < gpu_mms.size())
                for (size_t j = 0; j < gpu_mms.size(i); ++j)
                    cur_mms.push_back(gpu_mms(i, j));
            mms.push_back(cur_mms.begin(), cur_mms.end());
        }

        return mms;
    }

    template <typename REAL>
    void bdd_multi_parallel_mma_base<REAL>::fix_variable(const size_t var, const bool val)
    {
        throw std::runtime_error("not implemented yet");
        // assert(var < nr_variables());
        // if(var < cpu_base.nr_variables())
        //     cpu_base.fix_variable(var, val);
        // if(var < cuda_base.nr_variables())
        //     cuda_base.fix_variable(var, val);
    }

    template class bdd_multi_parallel_mma_base<float>;
    template class bdd_multi_parallel_mma_base<double>;

    template void bdd_multi_parallel_mma_base<float>::update_costs(float *, float *, float *, float *);
    template void bdd_multi_parallel_mma_base<float>::update_costs(double *, double *, double *, double *);
    template void bdd_multi_parallel_mma_base<float>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_multi_parallel_mma_base<float>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_multi_parallel_mma_base<float>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);
    template void bdd_multi_parallel_mma_base<float>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_multi_parallel_mma_base<double>::update_costs(float *, float *, float *, float *);
    template void bdd_multi_parallel_mma_base<double>::update_costs(double *, double *, double *, double *);
    template void bdd_multi_parallel_mma_base<double>::update_costs(std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_multi_parallel_mma_base<double>::update_costs(std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_multi_parallel_mma_base<double>::update_costs(std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator, std::vector<float>::const_iterator);
    template void bdd_multi_parallel_mma_base<double>::update_costs(std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator, std::vector<double>::const_iterator);
}

