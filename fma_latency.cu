// FMA Latency Measurement Kernels
//
// This file contains various CUDA kernels to measure the latency of
// fused multiply-add (FMA) operations under different execution patterns:
// - Basic latency measurement
// - Interleaved execution (ILP)
// - Non-interleaved execution (sequential chains)

#include <cuda_runtime.h>
#include <iostream>

using data_type = float;

// Inline assembly macro to read GPU cycle counter
#define clock_cycle() \
    ({ \
        unsigned long long ret; \
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(ret)); \
        ret; \
    })

////////////////////////////////////////////////////////////////////////////////
// Basic FMA Latency

__global__ void
fma_latency(data_type *n, unsigned long long *d_start, unsigned long long *d_end, int *num_fmas) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();
    data_type x = *n;
    data_type y0 = 0;
    data_type y1 = 0;
    data_type y2 = 0;
    data_type y3 = 0;
    data_type y4 = 0;
    data_type y5 = 0;
    data_type y6 = 0;
    data_type y7 = 0;
    data_type y8 = 0;
    data_type y9 = 0;
    data_type y10 = 0;
    data_type y11 = 0;
    data_type y12 = 0;
    data_type y13 = 0;
    data_type y14 = 0;
    data_type y15 = 0;
    data_type y16 = 0;
    data_type y17 = 0;
    data_type y18 = 0;

    // Memory fence to ensure that the reads are done.
    __threadfence();
    start_time = clock_cycle();

    y0 += x * 4; // 1
    y1 += y0 * 4; // 2
    y2 += y1 * 4; // 3
    y3 += y2 * 4; // 4
    y4 += y3 * 4; // 5
    y5 += y4 * 4; // 6
    y6 += y5 * 4; // 7
    y7 += y6 * 4; // 8
    y8 += y7 * 4; // 9
    y9 += y8 * 4; // 10
    y10 += y9 * 4; // 11
    y11 += y10 * 4; // 12
    y12 += y11 * 4; // 13
    y13 += y12 * 4; // 14
    y14 += y13 * 4; // 15
    y15 += y14 * 4; // 16
    y16 += y15 * 4; // 17
    y17 += y16 * 4; // 18
    y18 += y17 * 4; // 19
    x += y18 * 4; // 20

    end_time = clock_cycle();

    *n = x;
    *d_start = start_time;
    *d_end =  end_time;
    *num_fmas = 20;
}

////////////////////////////////////////////////////////////////////////////////
// FMA Latency + Instruction Level Parallelism (Interleaved)

__global__ void fma_latency_interleaved(
        data_type *n,
        unsigned long long *d_start,
        unsigned long long *d_end,
        int *num_fmas) {
    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();

    data_type x = *n;
    data_type y = *n;

    data_type tx0 = 0;
    data_type tx1 = 0;
    data_type tx2 = 0;
    data_type tx3 = 0;
    data_type tx4 = 0;
    data_type tx5 = 0;
    data_type tx6 = 0;
    data_type tx7 = 0;
    data_type tx8 = 0;

    data_type ty0 = 0;
    data_type ty1 = 0;
    data_type ty2 = 0;
    data_type ty3 = 0;
    data_type ty4 = 0;
    data_type ty5 = 0;
    data_type ty6 = 0;
    data_type ty7 = 0;
    data_type ty8 = 0;

    // Memory fence to ensure that the reads are done.
    __threadfence();

    start_time = clock_cycle();

    tx0 += x * 4; // 1
    ty0 += y * 4; // 2
    tx1 += tx0 * 4; // 3
    ty1 += ty0 * 4; // 4
    tx2 += tx1 * 4; // 5
    ty2 += ty1 * 4; // 6
    tx3 += tx2 * 4; // 7
    ty3 += ty2 * 4; // 8
    tx4 += tx3 * 4; // 9
    ty4 += ty3 * 4; // 10
    tx5 += tx4 * 4; // 11
    ty5 += ty4 * 4; // 12
    tx6 += tx5 * 4; // 13
    ty6 += ty5 * 4; // 14
    tx7 += tx6 * 4; // 15
    ty7 += ty6 * 4; // 16
    tx8 += tx7 * 4; // 17
    ty8 += ty7 * 4; // 18
    x += tx8 * 4; // 19
    y += ty8 * 4; // 20

    end_time = clock_cycle();

    *n = x + y;
    *d_start = start_time;
    *d_end = end_time;
    *num_fmas = 20;
}

////////////////////////////////////////////////////////////////////////////////
// FMA Latency + Sequential Execution (No Interleaving)

__global__ void fma_latency_no_interleave(
    data_type *n,
    unsigned long long *d_start,
    unsigned long long *d_end,
    int *num_fmas) {

    unsigned long long start_time = clock_cycle();
    unsigned long long end_time = clock_cycle();

    data_type x = *n;
    data_type y = *n;

    data_type tx0 = 0;
    data_type tx1 = 0;
    data_type tx2 = 0;
    data_type tx3 = 0;
    data_type tx4 = 0;
    data_type tx5 = 0;
    data_type tx6 = 0;
    data_type tx7 = 0;
    data_type tx8 = 0;

    data_type ty0 = 0;
    data_type ty1 = 0;
    data_type ty2 = 0;
    data_type ty3 = 0;
    data_type ty4 = 0;
    data_type ty5 = 0;
    data_type ty6 = 0;
    data_type ty7 = 0;
    data_type ty8 = 0;

    // Memory fence to ensure that the reads are done.
    __threadfence();

    start_time = clock_cycle();

    tx0 += x * 4; // 1
    tx1 += tx0 * 4; // 3
    tx2 += tx1 * 4; // 5
    tx3 += tx2 * 4; // 7
    tx4 += tx3 * 4; // 9
    tx5 += tx4 * 4; // 11
    tx6 += tx5 * 4; // 13
    tx7 += tx6 * 4; // 15
    tx8 += tx7 * 4; // 17
    x += tx8 * 4; // 19

    ty0 += y * 4; // 2
    ty1 += ty0 * 4; // 4
    ty2 += ty1 * 4; // 6
    ty3 += ty2 * 4; // 8
    ty4 += ty3 * 4; // 10
    ty5 += ty4 * 4; // 12
    ty6 += ty5 * 4; // 14
    ty7 += ty6 * 4; // 16
    ty8 += ty7 * 4; // 18
    y += ty8 * 4; // 20

    end_time = clock_cycle();

    *n = x + y;
    *d_start = start_time;
    *d_end = end_time;
    *num_fmas = 20;
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

// CUDA error checking macro
#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error " << static_cast<int>(err) << " (" \
                      << cudaGetErrorString(err) << ") at " << __FILE__ << ":" \
                      << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Macro to run kernel and print timing results.
#define run_kernel_and_print(kernel, d_n, d_start, d_end, num_fmas) \
    do { \
        unsigned long long h_time_start = 0ull, h_time_end = 0ull; \
        data_type result = 0.0f; \
        int h_num_fmas = 0; \
\
        kernel<<<1, 1>>>(d_n, d_start, d_end, num_fmas); \
        CUDA_CHECK(cudaDeviceSynchronize()); \
        CUDA_CHECK(cudaMemcpy( \
            &h_time_start, \
            d_start, \
            sizeof(unsigned long long), \
            cudaMemcpyDeviceToHost)); \
        CUDA_CHECK(cudaMemcpy( \
            &h_time_end, \
            d_end, \
            sizeof(unsigned long long), \
            cudaMemcpyDeviceToHost)); \
        CUDA_CHECK(cudaMemcpy(&result, d_n, sizeof(data_type), cudaMemcpyDeviceToHost)); \
        CUDA_CHECK(cudaMemcpy(&h_num_fmas, num_fmas, sizeof(int), cudaMemcpyDeviceToHost)); \
\
        float duration = static_cast<float>(h_time_end - h_time_start); \
        duration /= h_num_fmas; \
        std::cout << "Latency of " << #kernel \
                  << " code snippet = " << duration << " cycles" \
                  << " (" << h_num_fmas << " FMAs)" \
                  << std::endl; \
    } while (0)

int main() {
    data_type *d_n = nullptr;
    unsigned long long *d_start = nullptr;
    unsigned long long *d_end = nullptr;
    int *num_fmas = nullptr;

    data_type host_val = 4.0f;
    unsigned long long host_init_time = 0ull;
    int host_num_fmas = 0;

    CUDA_CHECK(cudaMalloc(&d_n, sizeof(data_type)));
    CUDA_CHECK(cudaMalloc(&d_start, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_end, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&num_fmas, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_n, &host_val, sizeof(data_type), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_start,
        &host_init_time,
        sizeof(unsigned long long),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_end,
        &host_init_time,
        sizeof(unsigned long long),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(num_fmas, &host_num_fmas, sizeof(int), cudaMemcpyHostToDevice));

    run_kernel_and_print(fma_latency, d_n, d_start, d_end, num_fmas);
    run_kernel_and_print(fma_latency_interleaved, d_n, d_start, d_end, num_fmas);
    run_kernel_and_print(fma_latency_no_interleave, d_n, d_start, d_end, num_fmas);

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_n));
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_end));
    CUDA_CHECK(cudaFree(num_fmas));

    return 0;
}