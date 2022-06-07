#include "bdd_multi_parallel_mma_base.h"
#include "two_dimensional_variable_array.hxx"
#include <limits>
#include <thrust/host_vector.h>

namespace LPMP {

    // return bdd indices for cpu and cuda bases
    std::array<std::vector<size_t>,2> bdd_decomposition(BDD::bdd_collection& bdd_col)
    {
        // TODO: make parallel
        std::vector<size_t> nr_bdd_nodes_per_layer;
        for(size_t bdd_nr=0; bdd_nr<bdd_col.nr_bdds(); ++bdd_nr)
        {
            assert(bdd_col.is_qbdd(bdd_nr));
            const auto bdd_vars = bdd_col.variables(bdd_nr);
            if(bdd_vars.size() > nr_bdd_nodes_per_layer.size())
                nr_bdd_nodes_per_layer.resize(bdd_vars.size(), 0);
            size_t prev_var = bdd_col.root_variable(bdd_nr);
            size_t layer = 1;
            for(auto bdd_it=bdd_col.begin(bdd_nr); bdd_it!=bdd_col.end(bdd_nr); ++bdd_it)
            {
                const auto& bdd_instr = *bdd_it;
                const size_t bdd_var = bdd_instr.index;
                if(bdd_var != prev_var)
                {
                    prev_var = bdd_var;
                    ++layer;
                }
                ++nr_bdd_nodes_per_layer[layer];
            }
            assert(layer == bdd_vars.size());
        }

        constexpr static size_t min_nr_bdd_nodes_per_layer = 1024;
        std::ptrdiff_t layer_th = nr_bdd_nodes_per_layer.size()-1;
        for(; layer_th>0; --layer_th)
        {
            if(nr_bdd_nodes_per_layer[layer_th] >= 1024)
                break;
        }

        if(layer_th + 10 > nr_bdd_nodes_per_layer.size())
        {
            std::cout << "[multi parallel mma base] Too few slim layers, optimize everything with on GPU\n";
            std::vector<size_t> gpu_bdds(bdd_col.nr_bdds());
            std::iota(gpu_bdds.begin(), gpu_bdds.end(), 0);
            return {std::vector<size_t>{}, gpu_bdds};
        }

        std::cout << "[multi parallel mma base] Optimize BDDs with more than " << layer_th << " hops on CPU\n";

        std::vector<size_t> cpu_bdds;
        std::vector<size_t> gpu_bdds;

        for(size_t bdd_nr=0; bdd_nr<bdd_col.nr_bdds(); ++bdd_nr)
        {
            const auto bdd_vars = bdd_col.variables(bdd_nr);
            if(bdd_vars.size() > layer_th)
                cpu_bdds.push_back(bdd_nr);
            else
                gpu_bdds.push_back(bdd_nr);
        }

        std::cout << "[multi parallel mma base] #CPU BDDs = " << cpu_bdds.size() << ", #GPU BDDs = " << gpu_bdds.size() << "\n";

            return {cpu_bdds, gpu_bdds};
        }

    template<typename REAL>
        bdd_multi_parallel_mma_base<REAL>::bdd_multi_parallel_mma_base(BDD::bdd_collection& bdd_col)
        {
            const auto [cpu_bdds, gpu_bdds] = bdd_decomposition(bdd_col);
            BDD::bdd_collection cpu_bdd_col = bdd_col;
            cpu_bdd_col.remove(gpu_bdds.begin(), gpu_bdds.end());
            BDD::bdd_collection cuda_bdd_col = bdd_col;
            cuda_bdd_col.remove(cpu_bdds.begin(), cpu_bdds.end());
            cpu_base = decltype(cpu_base)(cpu_bdd_col);
            cuda_base = decltype(cuda_base)(cuda_bdd_col);

            std::vector<size_t> cpu_nr_bdds_per_var;
            for(size_t i=0; i<cpu_base.nr_variables(); ++i)
                cpu_nr_bdds_per_var.push_back(cpu_base.nr_bdds(i));
            total_nr_bdds_per_var_ = thrust::device_vector<size_t>(nr_variables());
            thrust::copy(cpu_nr_bdds_per_var.begin(), cpu_nr_bdds_per_var.end(), total_nr_bdds_per_var_.begin());
            thrust::transform(cuda_base.get_num_bdds_per_var().begin(), cuda_base.get_num_bdds_per_var().end(), total_nr_bdds_per_var_.begin(), total_nr_bdds_per_var_.begin(), thrust::plus<size_t>());
        }

    template<typename REAL>
        template<typename COST_ITERATOR> 
        void bdd_multi_parallel_mma_base<REAL>::update_costs(COST_ITERATOR cost_lo_begin, COST_ITERATOR cost_lo_end, COST_ITERATOR cost_hi_begin, COST_ITERATOR cost_hi_end)
        {
            std::vector<double> lo_costs_cpu(cpu_base.nr_variables()), hi_costs_cpu(cpu_base.nr_variables());
            std::vector<double> lo_costs_gpu(cuda_base.nr_variables()), hi_costs_gpu(cuda_base.nr_variables());

            auto get_lo_cost = [&](const size_t var) {
                if(var < std::distance(cost_lo_begin, cost_lo_end) && var < nr_variables())
                    return *(cost_lo_begin+var)/double(nr_bdds(var));
                else
                    return 0.0;
            };
            auto get_hi_cost = [&](const size_t var) {
                if(var < std::distance(cost_hi_begin, cost_hi_end) && var < nr_variables())
                    return *(cost_hi_begin+var)/double(nr_bdds(var));
                else
                    return 0.0;
            };

            for(size_t var=0; var<nr_variables(); ++var)
            {
                const size_t nr_cpu_bdds = var < cpu_base.nr_variables() ? cpu_base.nr_bdds(var) : 0;
                const size_t nr_gpu_bdds = var < cuda_base.nr_variables() ? cuda_base.nr_bdds(var) : 0;

                const double cpu_ratio = double(nr_cpu_bdds)/double(nr_cpu_bdds + nr_gpu_bdds);
                const double gpu_ratio = double(nr_gpu_bdds)/double(nr_cpu_bdds + nr_gpu_bdds);

                if(var<cpu_base.nr_variables())
                {
                    lo_costs_cpu[var] = cpu_ratio * get_lo_cost(var);
                    hi_costs_cpu[var] = cpu_ratio * get_lo_cost(var);
                }
                if(var<cuda_base.nr_variables())
                {
                    lo_costs_gpu[var] = gpu_ratio * get_lo_cost(var);
                    hi_costs_gpu[var] = gpu_ratio * get_lo_cost(var);
                }
            }

            cpu_base.update_costs(lo_costs_cpu.begin(), lo_costs_cpu.end(), hi_costs_cpu.begin(), hi_costs_cpu.end());
            cuda_base.update_costs(lo_costs_gpu.begin(), lo_costs_gpu.end(), hi_costs_gpu.begin(), hi_costs_gpu.end());
        }

    template<typename REAL>
        size_t bdd_multi_parallel_mma_base<REAL>::nr_bdds() const
        {
            return cpu_base.nr_bdds() + cuda_base.nr_bdds();
        }
    template<typename REAL>
        size_t bdd_multi_parallel_mma_base<REAL>::nr_bdds(const size_t var) const
        {
            const size_t nr_cpu_bdds = var < cpu_base.nr_variables() ? cpu_base.nr_bdds(var) : 0;
            const size_t nr_gpu_bdds = var < cuda_base.nr_variables() ? cuda_base.nr_bdds(var) : 0;
            return nr_cpu_bdds + nr_gpu_bdds;
        }

    template<typename REAL>
        size_t bdd_multi_parallel_mma_base<REAL>::nr_variables() const
        {
            return std::max(cpu_base.nr_variables(), cuda_base.nr_variables());
        }

    // TODO: remove?
    template<typename REAL>
        size_t bdd_multi_parallel_mma_base<REAL>::nr_variables(const size_t bdd_nr) const
        {
            throw std::runtime_error("not supported yet");
            //if(bdd_nr < cpu_base.nr_bdds())
            //    return cpu_base.nr_variables(bdd_nr);
            //else
            //    return cuda_base.nr_variables(bdd_nr - cpu_base.nr_bdds());
        }

    template<typename REAL>
        double bdd_multi_parallel_mma_base<REAL>::lower_bound()
        {
            return cpu_base.lower_bound() + cuda_base.lower_bound();
        }

    template<typename REAL>
        void bdd_multi_parallel_mma_base<REAL>::add_to_constant(const double c)
        {
            cpu_base.add_to_constant(c);
        }

    template<typename REAL>
        void copy_delta_to_cpu(const thrust::device_vector<REAL>& delta_lo, const thrust::device_vector<REAL>& delta_hi, std::vector<std::array<REAL,2>>& delta)
        {

        }

    template<typename REAL>
        void copy_delta_to_gpu(const std::vector<std::array<REAL,2>>& delta, thrust::device_vector<REAL>& delta_lo, thrust::device_vector<REAL>& delta_hi)
        {

        }


    template<typename REAL>
        void bdd_multi_parallel_mma_base<REAL>::parallel_mma()
        {
            // TODO: hold this vector permanently without reallocating every iteration
            std::vector<std::array<REAL,2>> total_delta(nr_variables(), {0.0,0.0});
            // add cpu deltas to gpu ones
            auto accumulate_deltas = [&](const std::vector<std::array<REAL,2>>& cpu_delta, const thrust::device_vector<REAL>& gpu_delta_lo, const thrust::device_vector<REAL>& gpu_delta_hi)
            {
                copy_delta_to_cpu(gpu_delta_lo, gpu_delta_hi, total_delta);
                for(size_t i=0; i<cpu_delta.size(); ++i)
                {
                    total_delta[i][0] += cpu_delta[i][0];
                    total_delta[i][1] += cpu_delta[i][1];
                }
            };

            auto average_delta = [&]() {
                for(size_t i=0; i<nr_variables(); ++i)
                {
                    total_delta[i][0] /= REAL(nr_bdds(i));
                    total_delta[i][1] /= REAL(nr_bdds(i));
                }
            };

            auto split_delta = [&](std::vector<std::array<REAL,2>>& cpu_delta, thrust::device_vector<REAL>& gpu_delta_lo, thrust::device_vector<REAL>& gpu_delta_hi)
            {
                // to cpu
                for(size_t i=0; i<cpu_delta.size(); ++i)
                {
                    const REAL frac = REAL(cpu_base.nr_bdds(i)) / REAL(nr_bdds(i));
                    // TODO: or cpu_delta_in_? Or give parameters to lambda (best)
                    cpu_delta[i][0] = frac*total_delta[i][0];
                    cpu_delta[i][1] = frac*total_delta[i][1];
                }

                // to gpu
                thrust::host_vector<int> gpu_nr_bdds_per_var = cuda_base.get_num_bdds_per_var();
                std::vector<REAL> gpu_delta_lo_tmp(cuda_base.nr_variables());
                std::vector<REAL> gpu_delta_hi_tmp(cuda_base.nr_variables());
                for(size_t i=0; i<cuda_base.nr_variables(); ++i)
                {
                    const REAL frac = REAL(gpu_nr_bdds_per_var[i]) / REAL(total_nr_bdds_per_var_[i]);
                    gpu_delta_lo_tmp[i] = frac*total_delta[i][0];
                    gpu_delta_hi_tmp[i] = frac*total_delta[i][1];
                }
                gpu_delta_lo = gpu_delta_lo;
                gpu_delta_hi = gpu_delta_hi;
            };

            // forward pass
            cpu_base.forward_mm(0.5, cpu_delta_out_, cpu_delta_in_);
            cuda_base.forward_mm(0.5, gpu_delta_lo_, gpu_delta_hi_);

            accumulate_deltas(cpu_delta_out_, gpu_delta_lo_, gpu_delta_hi_);
            average_delta();
            split_delta(cpu_delta_out_, gpu_delta_lo_, gpu_delta_hi_);
            std::swap(cpu_delta_in_, cpu_delta_out_);
            std::fill(cpu_delta_out_.begin(), cpu_delta_out_.end(), std::array<REAL,2>({std::numeric_limits<REAL>::infinity(), std::numeric_limits<REAL>::infinity()}));

            // backward pass
            cpu_base.backward_mm(0.5, cpu_delta_out_, cpu_delta_in_);
            cuda_base.backward_mm(0.5, gpu_delta_lo_, gpu_delta_hi_);

            accumulate_deltas(cpu_delta_out_, gpu_delta_lo_, gpu_delta_hi_);
            average_delta();
            split_delta(cpu_delta_out_, gpu_delta_lo_, gpu_delta_hi_);
            std::swap(cpu_delta_in_, cpu_delta_out_);
            std::fill(cpu_delta_out_.begin(), cpu_delta_out_.end(), std::array<REAL,2>({std::numeric_limits<REAL>::infinity(), std::numeric_limits<REAL>::infinity()}));
        }

    // TODO: not currently used
    template<typename REAL>
        struct normalize_delta_func {
            __host__ __device__ void operator()(const thrust::tuple<REAL&, REAL&, int> t) const
            {
                const int norm = thrust::get<2>(t);
                REAL& hi_cost = thrust::get<0>(t);
                hi_cost /= norm;
                REAL& lo_cost = thrust::get<1>(t);
                lo_cost /= norm;
            }
        };

    template<typename REAL>
        void bdd_multi_parallel_mma_base<REAL>::normalize_delta(thrust::device_vector<REAL>& delta_lo, thrust::device_vector<REAL>& delta_hi) const
        {
            auto first = thrust::make_zip_iterator(thrust::make_tuple(delta_hi.begin(), delta_lo.begin(), total_nr_bdds_per_var_.begin()));
            auto last = thrust::make_zip_iterator(thrust::make_tuple(delta_hi.end(), delta_lo.end(), total_nr_bdds_per_var_.end()));
            thrust::for_each(first, last, normalize_delta_func<REAL>());
        }

    template<typename REAL>
        two_dim_variable_array<std::array<double,2>> bdd_multi_parallel_mma_base<REAL>::min_marginals()
        {
            const auto cpu_mms = cpu_base.min_marginals();
            const auto gpu_mms = cuda_base.min_marginals();

            two_dim_variable_array<std::array<double,2>> mms;
            std::vector<std::array<double,2>> cur_mms;
            for(size_t i=0; i<nr_variables(); ++i)
            {
                cur_mms.clear();
                if(i < cpu_mms.size())
                    for(size_t j=0; j<cpu_mms.size(i); ++j)
                        cur_mms.push_back(cpu_mms(i,j));
                if(i < gpu_mms.size())
                    for(size_t j=0; j<gpu_mms.size(i); ++j)
                        cur_mms.push_back(gpu_mms(i,j));
                mms.push_back(cur_mms.begin(), cur_mms.end());
            }

            return mms;
        }

    template<typename REAL>
        void bdd_multi_parallel_mma_base<REAL>::fix_variable(const size_t var, const bool val)
        {
            throw std::runtime_error("not implemented yet");
            //assert(var < nr_variables());
            //if(var < cpu_base.nr_variables())
            //    cpu_base.fix_variable(var, val);
            //if(var < cuda_base.nr_variables())
            //    cuda_base.fix_variable(var, val);
        }

    template class bdd_multi_parallel_mma_base<float>;
    template class bdd_multi_parallel_mma_base<double>;

    template void bdd_multi_parallel_mma_base<float>::update_costs(float*, float*, float*, float*);
    template void bdd_multi_parallel_mma_base<float>::update_costs(double*, double*, double*, double*);
    template void bdd_multi_parallel_mma_base<float>::update_costs(std::vector<float>::iterator,std::vector<float>::iterator,std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_multi_parallel_mma_base<float>::update_costs(std::vector<double>::iterator,std::vector<double>::iterator,std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_multi_parallel_mma_base<float>::update_costs(std::vector<float>::const_iterator,std::vector<float>::const_iterator,std::vector<float>::const_iterator, std::vector<float>::const_iterator);
    template void bdd_multi_parallel_mma_base<float>::update_costs(std::vector<double>::const_iterator,std::vector<double>::const_iterator,std::vector<double>::const_iterator, std::vector<double>::const_iterator);

    template void bdd_multi_parallel_mma_base<double>::update_costs(float*, float*, float*, float*);
    template void bdd_multi_parallel_mma_base<double>::update_costs(double*, double*, double*, double*);
    template void bdd_multi_parallel_mma_base<double>::update_costs(std::vector<float>::iterator,std::vector<float>::iterator,std::vector<float>::iterator, std::vector<float>::iterator);
    template void bdd_multi_parallel_mma_base<double>::update_costs(std::vector<double>::iterator,std::vector<double>::iterator,std::vector<double>::iterator, std::vector<double>::iterator);
    template void bdd_multi_parallel_mma_base<double>::update_costs(std::vector<float>::const_iterator,std::vector<float>::const_iterator,std::vector<float>::const_iterator, std::vector<float>::const_iterator);
    template void bdd_multi_parallel_mma_base<double>::update_costs(std::vector<double>::const_iterator,std::vector<double>::const_iterator,std::vector<double>::const_iterator, std::vector<double>::const_iterator);

}
