#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace LPMP {

template<typename REAL>
class lbfgs_cuda{
    public:
        lbfgs_cuda() {}
        lbfgs_cuda(const int n, const int _m = 10);

        // Stores history of next iteration to be used in project_gradient.
        // cur_x, cur_grad_f is undefined after the function call.
        void store_next_itr(thrust::device_vector<REAL>& cur_x, thrust::device_vector<REAL>& cur_grad_f);

        // Projects grad_f to compute updated direction using stored history.
        // Returns in-place.
        bool project_gradient(thrust::device_ptr<REAL> grad_f);

        int get_m() const { return m; }

        void flush_states(); 

        bool update_possible();

         void next_itr_without_storage();
       
    private:
        std::vector<thrust::device_vector<REAL>> s_history, y_history; // difference of x, and grad f resp.
        thrust::device_vector<REAL> prev_x, prev_grad_f;
        std::vector<REAL> rho_inv_history;
        int m = -1;
        int num_history = 0;
        int next_insertion_index = 0;
        double initial_rho_inv = 0.0;

        bool prev_states_stored = false;
        bool initial_rho_inv_valid = false;

    };
}