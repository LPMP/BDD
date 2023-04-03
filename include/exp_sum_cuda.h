#pragma once

#include <cuda_runtime.h>
#include "cuda_utils.h"

namespace LPMP {

template<typename REAL>
struct exp_sum_cuda {
    __host__ __device__ exp_sum_cuda() {}
    __host__ __device__ exp_sum_cuda(const REAL _sum, const REAL _max) : sum(_sum), max(_max) {}
    __host__ __device__ exp_sum_cuda(const REAL x);
    REAL sum = 0.0;
    REAL max = -1e30;

    __host__ __device__ void update(const exp_sum_cuda o);
    __host__ __device__ exp_sum_cuda<REAL> operator+(const exp_sum_cuda<REAL>& o) const;
    __host__ __device__ exp_sum_cuda<REAL>& operator+=(const exp_sum_cuda<REAL>& o);
    __host__ __device__ exp_sum_cuda<REAL> operator*(const exp_sum_cuda<REAL>& o) const;
    __host__ __device__ exp_sum_cuda<REAL>& operator*=(const exp_sum_cuda<REAL>& o);
    __host__ __device__ exp_sum_cuda<REAL> operator*(const REAL x) const;
    __host__ __device__ exp_sum_cuda<REAL>& operator*=(const REAL x);
    __host__ __device__ bool operator==(const exp_sum_cuda<REAL>& o) const;
    __host__ __device__ REAL logg() const;
    __host__ __device__ REAL value() const;
    // operator std::tuple<REAL,REAL>() const { return {sum, max}; }
};

template<typename REAL> __host__ __device__
    double exp_sum_cuda_diff_log(const exp_sum_cuda<REAL> a, const exp_sum_cuda<REAL> b)
    {
        return log(a.sum/b.sum) + a.max-b.max;
    }

template<typename REAL> __host__ __device__
    exp_sum_cuda<REAL> operator*(const REAL x, const exp_sum_cuda<REAL> es)
    {
        return es * x;
    }

// encapsulation in exp_sum_cuda
template<typename REAL> __host__ __device__
    exp_sum_cuda<REAL>::exp_sum_cuda(const REAL x)
    {
        sum = 1.0;
        max = x;
    }

template<typename REAL> __host__ __device__
void exp_sum_cuda<REAL>::update(const exp_sum_cuda o)
{
    assert(isfinite(sum));
    assert(sum >= 0.0);
    assert(!isnan(max));
    
    if(o.sum == 0.0)
    {
        //assert(o.max == -std::numeric_limits<REAL>::infinity());
        return;
    }
    
    if(max > o.max)
        sum += o.sum * exp(o.max - max);
    else
    {
        sum *= exp(max - o.max);
        sum += o.sum;
        max = o.max;
    }

    assert(isfinite(sum));
    assert(!isnan(max));
}

template<typename REAL> __host__ __device__
    exp_sum_cuda<REAL> exp_sum_cuda<REAL>::operator+(const exp_sum_cuda<REAL>& o) const
    {
        exp_sum_cuda<REAL> es = *this;
        es.update(o);
        return es;
    }

template<typename REAL> __host__ __device__
    exp_sum_cuda<REAL>& exp_sum_cuda<REAL>::operator+=(const exp_sum_cuda<REAL>& o)
    {
        *this = *this + o;
        return *this;
    }

template<typename REAL> __host__ __device__
    exp_sum_cuda<REAL> exp_sum_cuda<REAL>::operator*(const exp_sum_cuda<REAL>& o) const
    {
        exp_sum_cuda<REAL> es = *this;
        es.sum *= o.sum;
        es.max += o.max;
        return es;
    }

template<typename REAL> __host__ __device__
    exp_sum_cuda<REAL>& exp_sum_cuda<REAL>::operator*=(const exp_sum_cuda<REAL>& o)
    {
        *this = *this * o;
        return *this;
    }

template<typename REAL> __host__ __device__
    exp_sum_cuda<REAL> exp_sum_cuda<REAL>::operator*(const REAL x) const
    {
        assert(x > 0.0);
        exp_sum_cuda<REAL> es = *this;
        es.sum *= x;
        return es;
    }

template<typename REAL> __host__ __device__
    exp_sum_cuda<REAL>& exp_sum_cuda<REAL>::operator*=(const REAL x)
    {
        assert(x > 0.0);
        sum *= x;
        return *this;
    }

template<typename REAL> __host__ __device__
    bool exp_sum_cuda<REAL>::operator==(const exp_sum_cuda<REAL>& o) const
    {
        return sum == o.sum && max == o.max;
    }

template<typename REAL> __host__ __device__
    REAL exp_sum_cuda<REAL>::logg() const
    {
        return log(sum) + max;
    }

template<typename REAL> __host__ __device__
    REAL exp_sum_cuda<REAL>::value() const
    {
        return sum * exp(max);
    }

template<typename REAL> __host__ __device__
    void forward_add(const REAL self_max, const REAL self_sum, const REAL next_max, const REAL transition_cost, REAL* next_sum_address)
    {
        const REAL exponent = self_max + transition_cost - next_max;
        assert(exponent <= 0);
        atomicAdd(next_sum_address, exp(exponent) * self_sum);
    }

template<typename REAL> __host__ __device__
    void backward_add(const REAL prev_max, const REAL prev_sum, const REAL self_max, const REAL transition_cost, REAL& self_sum)
    {
        const REAL exponent = prev_max + transition_cost - self_max;
        assert(exponent <= 0);
        self_sum += exp(exponent) * prev_sum;
    }
}

