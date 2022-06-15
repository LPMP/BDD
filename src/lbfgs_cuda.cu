#include "time_measure_util.h"
#include "cuda_utils.h"
#include "lbfgs_cuda.h"
#include <thrust/for_each.h>
#include <thrust/inner_product.h>

namespace LPMP {
    // TODO: check swap, terminal nodes.

    template<typename REAL>
    lbfgs_cuda<REAL>::lbfgs_cuda(const int n, const int _m)
    {
        prev_x = thrust::device_vector<REAL>(n);
        prev_grad_f = thrust::device_vector<REAL>(n);
        m = _m;
        rho_inv_history = std::vector<REAL>(m);
        s_history = std::vector<thrust::device_vector<REAL>>(m);
        y_history = std::vector<thrust::device_vector<REAL>>(m);
    }

    template<typename REAL>
    void lbfgs_cuda<REAL>::store_next_itr(thrust::device_vector<REAL>& cur_x, thrust::device_vector<REAL>& cur_grad_f)
    {
        assert(cur_x.size() == prev_x.size());
        assert(cur_grad_f.size() == prev_x.size());
        if (!prev_states_stored)
        {
            prev_x = cur_x;
            prev_grad_f = cur_grad_f;
            prev_states_stored = true;
        }
        else
        {
            thrust::device_vector<REAL> cur_s(cur_x.size()); // compute x_k - x_{k-1}
            thrust::transform(cur_x.begin(), cur_x.end(), prev_x.begin(), cur_s.begin(), thrust::minus<REAL>());

            // compute grad_f_k - grad_f_{k-1}, but since we have maximization problem and lbfgs updates are derived for minimization so multiply gradients by -1.
            thrust::device_vector<REAL> cur_y(cur_grad_f.size());  
            thrust::transform(prev_grad_f.begin(), prev_grad_f.end(), cur_grad_f.begin(), cur_y.begin(), thrust::minus<REAL>());

            REAL rho_inv = thrust::inner_product(cur_s.begin(), cur_s.end(), cur_y.begin(), (REAL) 0.0);
            if (rho_inv > 1e-1) // otherwise, skip the iterate as curvature condition is not strongly satisfied.
            {
                rho_inv_history[next_insertion_index] = rho_inv;
                s_history[next_insertion_index] = cur_s;
                y_history[next_insertion_index] = cur_y;
                next_insertion_index = (next_insertion_index + 1) % m;
                num_history = min(num_history + 1, m);
            } //TODO: when skipping estimate of hessian will become out-of-date. However removing these updates as below gives worse results than not removing.
            // else
            // {
            //     num_history = max(num_history - 1, 0);
            // }
            prev_x = cur_x;
            prev_grad_f = cur_grad_f;
        }
    }

    template<typename REAL>
    struct update_q
    {
        const REAL alpha;
        const REAL* y;
        REAL* q;
        __host__ __device__ void operator()(const int idx)
        {
            q[idx] -= alpha * y[idx];
        }
    };

    template<typename REAL>
    struct update_r
    {
        const REAL alpha;
        const REAL beta;
        const REAL* s;
        REAL* r;
        __host__ __device__ void operator()(const int idx)
        {
            r[idx] += s[idx] * (alpha - beta);
        }
    };

    template<typename REAL>
    bool lbfgs_cuda<REAL>::project_gradient(thrust::device_ptr<REAL> grad_f)
    {
        if (num_history == 0) // can be possible if curvature condition not satisfied.
            return false;

        const int n = s_history[0].size();

        std::vector<REAL> alpha_history;
        for (int count = 0; count < num_history; count++)
        {
            int i = next_insertion_index - count - 1;
            if (i < 0)
                i = m + i;

            assert(i >= 0 && i < m);
            assert(s_history[i].size() == n);
            const REAL alpha = thrust::inner_product(s_history[i].begin(), s_history[i].end(), grad_f, (REAL) 0.0) / (1e-4 + rho_inv_history[i]);
            
            alpha_history.push_back(alpha);
            update_q<REAL> update_q_func({alpha, thrust::raw_pointer_cast(y_history[i].data()), thrust::raw_pointer_cast(grad_f)});

            thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + n, update_q_func);
        }

        REAL last_y_norm = thrust::inner_product(y_history.back().begin(), y_history.back().end(), y_history.back().begin(), (REAL) 0.0);
        REAL initial_H_diag_multiplier = rho_inv_history.back() / (1e-4 + last_y_norm);
        // Skip line 5 in Alg.1 and fuse with line 7 for first loop itr.
        for (int count = 0; count < num_history; count++)
        {
            int i = next_insertion_index - num_history + count;
            if (i < 0)
                i = m + i;

            assert(i >= 0 && i < m);
            assert(y_history[i].size() == n);

            REAL current_rho = 1 / (1e-4 + rho_inv_history[i]);
            if (count == 0)
                current_rho *= initial_H_diag_multiplier;
            const REAL beta = current_rho * thrust::inner_product(y_history[i].begin(), y_history[i].end(), grad_f, (REAL) 0.0);

            update_r<REAL> update_r_func({alpha_history[num_history - count - 1], beta, thrust::raw_pointer_cast(s_history[i].data()), thrust::raw_pointer_cast(grad_f)});

            thrust::for_each(thrust::make_counting_iterator<int>(0), thrust::make_counting_iterator<int>(0) + n, update_r_func);
        }
        return true;
    }

    template class lbfgs_cuda<float>;
    template class lbfgs_cuda<double>;
}