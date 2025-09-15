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
    x += y8 * 4; // 10

    end_time = clock_cycle();

    *n = x;
    *d_start = start_time;
    *d_end =  end_time;
    *num_fmas = 10;
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
    data_type ty0 = 0;
    data_type ty1 = 0;
    data_type ty2 = 0;
    data_type ty3 = 0;

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
    x += tx3 * 4; // 9
    y += ty3 * 4; // 10

    end_time = clock_cycle();

    *n = x + y;
    *d_start = start_time;
    *d_end = end_time;
    *num_fmas = 10;
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
    // Memory fence to ensure that the reads are done.
    __threadfence();

    start_time = clock_cycle();

    /// <--- /your code here --->

    end_time = clock_cycle();

    *n = x + y;
    *d_start = start_time;
    *d_end = end_time;
    *num_fmas = 1;
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