#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/discard_iterator.h>
#include "time_measure_util.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/random.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline float __int_as_float_host(int a)
{
    union {int a; float b;} u;
    u.a = a;
    return u.b;
}

#define CUDART_INF_F_HOST __int_as_float_host(0x7f800000)

// copied from: https://github.com/treecode/Bonsai/blob/8904dd3ebf395ccaaf0eacef38933002b49fc3ba/runtime/profiling/derived_atomic_functions.h#L186
__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ double atomicMin(double *address, double val)
{
    unsigned long long int ret = __double_as_longlong(*address);
    while(val < __longlong_as_double(ret))
    {
        unsigned long long int old = ret;
        if((ret = atomicCAS((unsigned long long int *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// float atomicMax
__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

// copied from https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val +
                    __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

inline int get_cuda_device()
{   
    return 0; // Get first possible GPU. CUDA_VISIBLE_DEVICES automatically masks the rest of GPUs.
}

inline void print_gpu_memory_stats()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    std::cout<<"Total memory(MB): "<<total / (1024 * 1024)<<", Free(MB): "<<free / (1024 * 1024)<<std::endl;
}

inline void checkCudaStatus(std::string add_info = "")
{
    cudaDeviceSynchronize();
    cudaError error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n at: %s \n", cudaGetErrorString(error), add_info.c_str());
        exit(-1);
    }
}

inline void checkCudaError(cudaError_t status, std::string errorMsg)
{
    if (status != cudaSuccess) {
        std::cout << "CUDA error: " << errorMsg << ", status" <<cudaGetErrorString(status) << std::endl;
        throw std::exception();
    }
}

inline std::tuple<thrust::device_vector<int>, thrust::device_vector<int>> get_unique_with_counts(const thrust::device_vector<int>& input)
{
    assert(thrust::is_sorted(input.begin(), input.end()));
    thrust::device_vector<int> unique_counts(input.size() + 1);
    thrust::device_vector<int> unique_values(input.size());

    auto new_end = thrust::unique_by_key_copy(input.begin(), input.end(), thrust::make_counting_iterator(0), unique_values.begin(), unique_counts.begin());
    int num_unique = std::distance(unique_values.begin(), new_end.first);
    unique_values.resize(num_unique);
    unique_counts.resize(num_unique + 1); // contains smallest index of each unique element.
    
    unique_counts[num_unique] = input.size();
    thrust::adjacent_difference(unique_counts.begin(), unique_counts.end(), unique_counts.begin());
    unique_counts = thrust::device_vector<int>(unique_counts.begin() + 1, unique_counts.end());

    return {unique_values, unique_counts};
}

template<typename T>
inline thrust::device_vector<T> repeat_values(const thrust::device_vector<T>& values, const thrust::device_vector<int>& counts)
{
    MEASURE_CUMULATIVE_FUNCTION_EXECUTION_TIME
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

template<typename T>
inline thrust::device_vector<T> concatenate(const thrust::device_vector<T>& a, const thrust::device_vector<T>& b)
{
    thrust::device_vector<T> ab(a.size() + b.size());
    thrust::copy(a.begin(), a.end(), ab.begin());
    thrust::copy(b.begin(), b.end(), ab.begin() + a.size());
    return ab;
}

inline void coo_sorting(thrust::device_vector<int>& i, thrust::device_vector<int>& j)
{
    auto first = thrust::make_zip_iterator(thrust::make_tuple(i.begin(), j.begin()));
    auto last = thrust::make_zip_iterator(thrust::make_tuple(i.end(), j.end()));
    thrust::sort(first, last);
}

template<typename T>
struct add_noise_func
{
    unsigned int seed;
    T noise_mag;
    T* vec;

    __host__ __device__ void operator()(const unsigned int n)
    {
        thrust::default_random_engine rng;
        thrust::uniform_real_distribution<T> dist(-noise_mag, noise_mag);
        rng.discard(seed + n);
        vec[n] += dist(rng);
    }
};

template<typename T>
inline void add_noise(thrust::device_ptr<T> v, const size_t num, const T noise_magnitude, const unsigned int seed)
{
    add_noise_func<T> add_noise({seed, noise_magnitude, thrust::raw_pointer_cast(v)});
    thrust::for_each(thrust::make_counting_iterator<unsigned int>(0), thrust::make_counting_iterator<unsigned int>(0) + num, add_noise);
}

template<typename T>
inline void print_vector(const thrust::device_vector<T>& v, const char* name, const int num = 0)
{
    std::cout<<name<<": ";
    if (num == 0)
        thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
    else
    {
        int size = std::distance(v.begin(), v.end());
        thrust::copy(v.begin(), v.begin() + std::min(size, num), std::ostream_iterator<T>(std::cout, " "));
    }
    std::cout<<"\n";
}

template<typename T>
inline void print_vector(const thrust::device_ptr<T>& v, const char* name, const int num)
{
    std::cout<<name<<": ";
    thrust::copy(v, v + num, std::ostream_iterator<T>(std::cout, " "));
    std::cout<<"\n";
}

template<typename T>
inline void check_finite(const thrust::device_ptr<T>& v, const size_t num)
{
    auto result = thrust::minmax_element(v, v + num);
    assert(std::isfinite(*result.first));
    assert(std::isfinite(*result.second));
}

template<typename T>
inline void print_min_max(const thrust::device_ptr<T>& v, const char* name, const size_t num)
{
    auto result = thrust::minmax_element(v, v + num);
    std::cout<<name<<": min = "<<*result.first<<", max = "<<*result.second<<"\n";
}

struct tuple_min
{
    template<typename REAL>
    __host__ __device__
    thrust::tuple<REAL, REAL> operator()(const thrust::tuple<REAL, REAL>& t0, const thrust::tuple<REAL, REAL>& t1)
    {
        return thrust::make_tuple(min(thrust::get<0>(t0), thrust::get<0>(t1)), min(thrust::get<1>(t0), thrust::get<1>(t1)));
    }
};

struct tuple_sum
{
    template<typename T>
    __host__ __device__
    thrust::tuple<T, T> operator()(const thrust::tuple<T, T>& t0, const thrust::tuple<T, T>& t1)
    {
        return thrust::make_tuple(thrust::get<0>(t0) + thrust::get<0>(t1), thrust::get<1>(t0) + thrust::get<1>(t1));
    }
};
