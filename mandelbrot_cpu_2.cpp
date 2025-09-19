// Optional arguments:
//  -r <img_size>
//  -b <max iterations>
//  -i <implementation: {"scalar", "vector", "vector_ilp", "vector_multicore",
//  "vector_multicore_multithread", "vector_multicore_multithread_ilp", "all"}>

#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <pthread.h>
#include <cassert>
#include <cstdio>
#include <vector>
#include <array>

constexpr float window_zoom = 1.0 / 10000.0f;
constexpr float window_x = -0.743643887 - 0.5 * window_zoom;
constexpr float window_y = 0.131825904 - 0.5 * window_zoom;
constexpr uint32_t default_max_iters = 2000;

// CPU Scalar Mandelbrot set generation.
// Based on the "optimized escape time algorithm" in
// https://en.wikipedia.org/wiki/Plotting_algorithms_for_the_Mandelbrot_set
void mandelbrot_cpu_scalar(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    for (uint64_t i = 0; i < img_size; ++i) {
        for (uint64_t j = 0; j < img_size; ++j) {
            float cx = (float(j) / float(img_size)) * window_zoom + window_x;
            float cy = (float(i) / float(img_size)) * window_zoom + window_y;

            float x2 = 0.0f;
            float y2 = 0.0f;
            float w = 0.0f;
            uint32_t iters = 0;
            while (x2 + y2 <= 4.0f && iters < max_iters) {
                float x = x2 - y2 + cx;
                float y = w - (x2 + y2) + cy;
                x2 = x * x;
                y2 = y * y;
                float z = x + y;
                w = z * z;
                ++iters;
            }

            // Write result.
            out[i * img_size + j] = iters;
        }
    }
}

uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

/// <--- your code here --->


// OPTIONAL: Uncomment this block to include your CPU vector implementation
// from Lab 1 for easy comparison.
//
// (If you do this, you'll need to update your code to use the new constants
// 'window_zoom', 'window_x', and 'window_y'.)

#define HAS_VECTOR_IMPL // <~~ keep this line if you want to benchmark the vector kernel!

////////////////////////////////////////////////////////////////////////////////
// Vector

// void mandelbrot_cpu_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
//     uint32_t chunk_size = 16;
//     assert(img_size % chunk_size == 0);
//     uint32_t num_chunks = img_size / chunk_size;

//     __m512 img_size_vec = _mm512_set1_ps(img_size);
//     __m512 window_zoom_vec = _mm512_set1_ps(window_zoom);
//     __m512 window_x_vec = _mm512_set1_ps(window_x);
//     __m512 window_y_vec = _mm512_set1_ps(window_y);
//     __m512i max_iters_vec = _mm512_set1_epi32(max_iters);
//     for (uint64_t i = 0; i < img_size; i++) {
// 	    for (uint64_t ch = 0; ch < num_chunks; ch++) {
//             uint32_t base_j = ch * chunk_size;
// 		    __m512 j_vec = _mm512_set_ps(base_j + 15, base_j + 14, base_j + 13, base_j + 12,
//                                         base_j + 11, base_j + 10, base_j + 9, base_j + 8,
//                                         base_j + 7, base_j + 6, base_j + 5, base_j + 4,
//                                         base_j + 3, base_j + 2, base_j + 1, base_j + 0);
// 		    __m512 i_vec = _mm512_set1_ps(i);
// 		    __m512 cx_div = _mm512_div_ps(j_vec, img_size_vec);
// 		    __m512 cx_mul = _mm512_mul_ps(cx_div, window_zoom_vec);
// 		    __m512 cx = _mm512_add_ps(cx_mul, window_x_vec);
// 		    __m512 cy_div = _mm512_div_ps(i_vec, img_size_vec);
// 		    __m512 cy_mul = _mm512_mul_ps(cy_div, window_zoom_vec);
// 		    __m512 cy = _mm512_add_ps(cy_mul, window_y_vec);

// 		    __m512 x2 = _mm512_set1_ps(0);
// 		    __m512 y2 = _mm512_set1_ps(0);
// 		    __m512 w = _mm512_set1_ps(0);
// 		    __m512i iters = _mm512_set1_epi32(0);

// 		    __m512 x2_y2_sum = _mm512_add_ps(x2, y2);
// 		    __m512 four_vec = _mm512_set1_ps(4.0f);
// 		    __mmask16 less_than_four = _mm512_cmp_ps_mask(x2_y2_sum, four_vec, 2);
//             __mmask16 less_than_max_iters = _mm512_cmp_epi32_mask(iters, max_iters_vec, 1);
//             __mmask16 active_mask = _kand_mask16(less_than_four, less_than_max_iters);

//             while (active_mask) {
//                 __m512 _x = _mm512_sub_ps(x2, y2);
//                 __m512 x = _mm512_add_ps(_x, cx);
//                 __m512 _y = _mm512_sub_ps(w, x2_y2_sum);
//                 __m512 y = _mm512_add_ps(_y, cy);

//                 x2 = _mm512_mul_ps(x, x);
//                 y2 = _mm512_mul_ps(y, y);
//                 __m512 z = _mm512_add_ps(x, y);
//                 w = _mm512_mul_ps(z, z);

//                 // inc loop bounds
//                 iters = _mm512_mask_add_epi32(iters, active_mask, iters, _mm512_set1_epi32(1));

//                 // repeat the loop bound checks here
//                 x2_y2_sum = _mm512_mask_add_ps(x2_y2_sum, active_mask, x2, y2);
//                 less_than_four = _mm512_cmp_ps_mask(x2_y2_sum, four_vec, 2);
//                 less_than_max_iters = _mm512_cmp_epi32_mask(iters, max_iters_vec, 1);
//                 active_mask = _kand_mask16(less_than_four, less_than_max_iters);
//             }

//             _mm512_storeu_epi32(&(out[i * img_size + ch * chunk_size]), iters);

// 		}
//     }
// }

void mandelbrot_cpu_vector(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    uint32_t chunk_size = 16;
    assert(img_size % chunk_size == 0);
    uint32_t num_chunks = img_size / chunk_size;

    __m512 img_size_vec = _mm512_set1_ps(img_size);
    __m512 window_zoom_vec = _mm512_set1_ps(window_zoom);
    __m512 window_x_vec = _mm512_set1_ps(window_x);
    __m512 window_y_vec = _mm512_set1_ps(window_y);
    __m512i max_iters_vec = _mm512_set1_epi32(max_iters);
    __m512 four_vec = _mm512_set1_ps(4.0f);

    for (uint64_t ch = 0; ch < num_chunks; ch++) {
        uint32_t base_j = ch * chunk_size;
        __m512 j_vec = _mm512_set_ps(base_j + 15, base_j + 14, base_j + 13, base_j + 12,
                                    base_j + 11, base_j + 10, base_j + 9, base_j + 8,
                                    base_j + 7, base_j + 6, base_j + 5, base_j + 4,
                                    base_j + 3, base_j + 2, base_j + 1, base_j + 0);
        __m512 cx_div = _mm512_div_ps(j_vec, img_size_vec);
        __m512 cx_mul = _mm512_mul_ps(cx_div, window_zoom_vec);
        __m512 cx = _mm512_add_ps(cx_mul, window_x_vec);

        for (uint64_t i = 0; i < img_size; i++) {
		    __m512 i_vec = _mm512_set1_ps(i);
		    __m512 cy_div = _mm512_div_ps(i_vec, img_size_vec);
		    __m512 cy_mul = _mm512_mul_ps(cy_div, window_zoom_vec);
		    __m512 cy = _mm512_add_ps(cy_mul, window_y_vec);

            __m512 x2 = _mm512_set1_ps(0);
            __m512 y2 = _mm512_set1_ps(0);
            __m512 w = _mm512_set1_ps(0);
            __m512i iters = _mm512_set1_epi32(0);
		    __m512 x2_y2_sum = _mm512_add_ps(x2, y2);
		    __mmask16 less_than_four = _mm512_cmp_ps_mask(x2_y2_sum, four_vec, 2);

            int iter = 0;
            while (iter < max_iters) {
                __m512 _x = _mm512_sub_ps(x2, y2);
                __m512 x = _mm512_add_ps(_x, cx);
                __m512 _y = _mm512_sub_ps(w, x2_y2_sum);
                __m512 y = _mm512_add_ps(_y, cy);

                x2 = _mm512_mul_ps(x, x);
                y2 = _mm512_mul_ps(y, y);
                __m512 z = _mm512_add_ps(x, y);
                w = _mm512_mul_ps(z, z);

                // inc loop bounds
                iters = _mm512_mask_add_epi32(iters, less_than_four, iters, _mm512_set1_epi32(1));

                // repeat the loop bound checks here
                x2_y2_sum = _mm512_mask_add_ps(x2_y2_sum, less_than_four, x2, y2);
                less_than_four = _mm512_cmp_ps_mask(x2_y2_sum, four_vec, 2);
                iter++;
                if (!less_than_four) {
                    break;
                }
            }

            _mm512_storeu_epi32(&(out[i * img_size + ch * chunk_size]), iters);

		}
    }
}


////////////////////////////////////////////////////////////////////////////////
// Vector + ILP

void mandelbrot_cpu_vector_ilp(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    uint32_t chunk_size = 16;
    assert(img_size % chunk_size == 0);
    uint32_t num_chunks = img_size / chunk_size;

    constexpr int unroll_factor = 4;

    __m512 img_size_vec = _mm512_set1_ps(img_size);
    __m512 window_zoom_vec = _mm512_set1_ps(window_zoom);
    __m512 window_x_vec = _mm512_set1_ps(window_x);
    __m512 window_y_vec = _mm512_set1_ps(window_y);
    __m512i max_iters_vec = _mm512_set1_epi32(max_iters);
    __m512 four_vec = _mm512_set1_ps(4.0f);

    std::array<__m512, unroll_factor> i_vec;
    std::array<__m512, unroll_factor> cy_div;
    std::array<__m512, unroll_factor> cy_mul;
    std::array<__m512, unroll_factor> cy;
    std::array<__m512, unroll_factor> x2;
    std::array<__m512, unroll_factor> y2;
    std::array<__m512, unroll_factor> w;
    std::array<__m512i, unroll_factor> iters;
    std::array<__m512, unroll_factor> x2_y2_sum;
    std::array<__mmask16, unroll_factor> less_than_four;
    std::array<__m512, unroll_factor> _x;
    std::array<__m512, unroll_factor> x;
    std::array<__m512, unroll_factor> _y;
    std::array<__m512, unroll_factor> y;
    std::array<__m512, unroll_factor> z;
    std::array<uint64_t, unroll_factor> i;

    assert(img_size % unroll_factor == 0);
    uint64_t num_unrolled_iters = img_size / unroll_factor;

    for (uint64_t i_ch = 0; i_ch < num_unrolled_iters; i_ch++) {

        for (uint64_t ch = 0; ch < num_chunks; ch++) {

            uint32_t base_j = ch * chunk_size;
            __m512 j_vec = _mm512_set_ps(base_j + 15, base_j + 14, base_j + 13, base_j + 12,
                                        base_j + 11, base_j + 10, base_j + 9, base_j + 8,
                                        base_j + 7, base_j + 6, base_j + 5, base_j + 4,
                                        base_j + 3, base_j + 2, base_j + 1, base_j + 0);
            __m512 cx_div = _mm512_div_ps(j_vec, img_size_vec);
            __m512 cx_mul = _mm512_mul_ps(cx_div, window_zoom_vec);
            __m512 cx = _mm512_add_ps(cx_mul, window_x_vec);

            i[0] = i_ch * unroll_factor + 0;
            i[1] = i_ch * unroll_factor + 1;
            i[2] = i_ch * unroll_factor + 2;
            i[3] = i_ch * unroll_factor + 3;
            // i[4] = i_ch * unroll_factor + 4;
            // i[5] = i_ch * unroll_factor + 5;
            // i[6] = i_ch * unroll_factor + 6;
            // i[7] = i_ch * unroll_factor + 7;

            i_vec[0] = _mm512_set1_ps(i[0]);
            i_vec[1] = _mm512_set1_ps(i[1]);
            i_vec[2] = _mm512_set1_ps(i[2]);
            i_vec[3] = _mm512_set1_ps(i[3]);
            // i_vec[4] = _mm512_set1_ps(i[4]);
            // i_vec[5] = _mm512_set1_ps(i[5]);
            // i_vec[6] = _mm512_set1_ps(i[6]);
            // i_vec[7] = _mm512_set1_ps(i[7]);

            cy_div[0] = _mm512_div_ps(i_vec[0], img_size_vec);
            cy_div[1] = _mm512_div_ps(i_vec[1], img_size_vec);
            cy_div[2] = _mm512_div_ps(i_vec[2], img_size_vec);
            cy_div[3] = _mm512_div_ps(i_vec[3], img_size_vec);
            // cy_div[4] = _mm512_div_ps(i_vec[4], img_size_vec);
            // cy_div[5] = _mm512_div_ps(i_vec[5], img_size_vec);
            // cy_div[6] = _mm512_div_ps(i_vec[6], img_size_vec);
            // cy_div[7] = _mm512_div_ps(i_vec[7], img_size_vec);

            cy_mul[0] = _mm512_mul_ps(cy_div[0], window_zoom_vec);
            cy_mul[1] = _mm512_mul_ps(cy_div[1], window_zoom_vec);
            cy_mul[2] = _mm512_mul_ps(cy_div[2], window_zoom_vec);
            cy_mul[3] = _mm512_mul_ps(cy_div[3], window_zoom_vec);
            // cy_mul[4] = _mm512_mul_ps(cy_div[4], window_zoom_vec);
            // cy_mul[5] = _mm512_mul_ps(cy_div[5], window_zoom_vec);
            // cy_mul[6] = _mm512_mul_ps(cy_div[6], window_zoom_vec);
            // cy_mul[7] = _mm512_mul_ps(cy_div[7], window_zoom_vec);

            cy[0] = _mm512_add_ps(cy_mul[0], window_y_vec);
            cy[1] = _mm512_add_ps(cy_mul[1], window_y_vec);
            cy[2] = _mm512_add_ps(cy_mul[2], window_y_vec);
            cy[3] = _mm512_add_ps(cy_mul[3], window_y_vec);
            // cy[4] = _mm512_add_ps(cy_mul[4], window_y_vec);
            // cy[5] = _mm512_add_ps(cy_mul[5], window_y_vec);
            // cy[6] = _mm512_add_ps(cy_mul[6], window_y_vec);
            // cy[7] = _mm512_add_ps(cy_mul[7], window_y_vec);

            x2[0] = _mm512_set1_ps(0);
            x2[1] = _mm512_set1_ps(0);
            x2[2] = _mm512_set1_ps(0);
            x2[3] = _mm512_set1_ps(0);
            // x2[4] = _mm512_set1_ps(0);
            // x2[5] = _mm512_set1_ps(0);
            // x2[6] = _mm512_set1_ps(0);
            // x2[7] = _mm512_set1_ps(0);

            y2[0] = _mm512_set1_ps(0);
            y2[1] = _mm512_set1_ps(0);
            y2[2] = _mm512_set1_ps(0);
            y2[3] = _mm512_set1_ps(0);
            // y2[4] = _mm512_set1_ps(0);
            // y2[5] = _mm512_set1_ps(0);
            // y2[6] = _mm512_set1_ps(0);
            // y2[7] = _mm512_set1_ps(0);

            w[0] = _mm512_set1_ps(0);
            w[1] = _mm512_set1_ps(0);
            w[2] = _mm512_set1_ps(0);
            w[3] = _mm512_set1_ps(0);
            // w[4] = _mm512_set1_ps(0);
            // w[5] = _mm512_set1_ps(0);
            // w[6] = _mm512_set1_ps(0);
            // w[7] = _mm512_set1_ps(0);

            iters[0] = _mm512_set1_epi32(0);
            iters[1] = _mm512_set1_epi32(0);
            iters[2] = _mm512_set1_epi32(0);
            iters[3] = _mm512_set1_epi32(0);
            // iters[4] = _mm512_set1_epi32(0);
            // iters[5] = _mm512_set1_epi32(0);
            // iters[6] = _mm512_set1_epi32(0);
            // iters[7] = _mm512_set1_epi32(0);

            x2_y2_sum[0] = _mm512_add_ps(x2[0], y2[0]);
            x2_y2_sum[1] = _mm512_add_ps(x2[1], y2[1]);
            x2_y2_sum[2] = _mm512_add_ps(x2[2], y2[2]);
            x2_y2_sum[3] = _mm512_add_ps(x2[3], y2[3]);
            // x2_y2_sum[4] = _mm512_add_ps(x2[4], y2[4]);
            // x2_y2_sum[5] = _mm512_add_ps(x2[5], y2[5]);
            // x2_y2_sum[6] = _mm512_add_ps(x2[6], y2[6]);
            // x2_y2_sum[7] = _mm512_add_ps(x2[7], y2[7]);

            less_than_four[0] = _mm512_cmp_ps_mask(x2_y2_sum[0], four_vec, 2);
            less_than_four[1] = _mm512_cmp_ps_mask(x2_y2_sum[1], four_vec, 2);
            less_than_four[2] = _mm512_cmp_ps_mask(x2_y2_sum[2], four_vec, 2);
            less_than_four[3] = _mm512_cmp_ps_mask(x2_y2_sum[3], four_vec, 2);
            // less_than_four[4] = _mm512_cmp_ps_mask(x2_y2_sum[4], four_vec, 2);
            // less_than_four[5] = _mm512_cmp_ps_mask(x2_y2_sum[5], four_vec, 2);
            // less_than_four[6] = _mm512_cmp_ps_mask(x2_y2_sum[6], four_vec, 2);
            // less_than_four[7] = _mm512_cmp_ps_mask(x2_y2_sum[7], four_vec, 2);

            int iter = 0;
            while (iter < max_iters) {
                _x[0] = _mm512_sub_ps(x2[0], y2[0]);
                _x[1] = _mm512_sub_ps(x2[1], y2[1]);
                _x[2] = _mm512_sub_ps(x2[2], y2[2]);
                _x[3] = _mm512_sub_ps(x2[3], y2[3]);
                // _x[4] = _mm512_sub_ps(x2[4], y2[4]);
                // _x[5] = _mm512_sub_ps(x2[5], y2[5]);
                // _x[6] = _mm512_sub_ps(x2[6], y2[6]);
                // _x[7] = _mm512_sub_ps(x2[7], y2[7]);

                x[0] = _mm512_add_ps(_x[0], cx);
                x[1] = _mm512_add_ps(_x[1], cx);
                x[2] = _mm512_add_ps(_x[2], cx);
                x[3] = _mm512_add_ps(_x[3], cx);
                // x[4] = _mm512_add_ps(_x[4], cx);
                // x[5] = _mm512_add_ps(_x[5], cx);
                // x[6] = _mm512_add_ps(_x[6], cx);
                // x[7] = _mm512_add_ps(_x[7], cx);

                _y[0] = _mm512_sub_ps(w[0], x2_y2_sum[0]);
                _y[1] = _mm512_sub_ps(w[1], x2_y2_sum[1]);
                _y[2] = _mm512_sub_ps(w[2], x2_y2_sum[2]);
                _y[3] = _mm512_sub_ps(w[3], x2_y2_sum[3]);
                // _y[4] = _mm512_sub_ps(w[4], x2_y2_sum[4]);
                // _y[5] = _mm512_sub_ps(w[5], x2_y2_sum[5]);
                // _y[6] = _mm512_sub_ps(w[6], x2_y2_sum[6]);
                // _y[7] = _mm512_sub_ps(w[7], x2_y2_sum[7]);

                y[0] = _mm512_add_ps(_y[0], cy[0]);
                y[1] = _mm512_add_ps(_y[1], cy[1]);
                y[2] = _mm512_add_ps(_y[2], cy[2]);
                y[3] = _mm512_add_ps(_y[3], cy[3]);
                // y[4] = _mm512_add_ps(_y[4], cy[4]);
                // y[5] = _mm512_add_ps(_y[5], cy[5]);
                // y[6] = _mm512_add_ps(_y[6], cy[6]);
                // y[7] = _mm512_add_ps(_y[7], cy[7]);

                x2[0] = _mm512_mul_ps(x[0], x[0]);
                x2[1] = _mm512_mul_ps(x[1], x[1]);
                x2[2] = _mm512_mul_ps(x[2], x[2]);
                x2[3] = _mm512_mul_ps(x[3], x[3]);
                // x2[4] = _mm512_mul_ps(x[4], x[4]);
                // x2[5] = _mm512_mul_ps(x[5], x[5]);
                // x2[6] = _mm512_mul_ps(x[6], x[6]);
                // x2[7] = _mm512_mul_ps(x[7], x[7]);

                y2[0] = _mm512_mul_ps(y[0], y[0]);
                y2[1] = _mm512_mul_ps(y[1], y[1]);
                y2[2] = _mm512_mul_ps(y[2], y[2]);
                y2[3] = _mm512_mul_ps(y[3], y[3]);
                // y2[4] = _mm512_mul_ps(y[4], y[4]);
                // y2[5] = _mm512_mul_ps(y[5], y[5]);
                // y2[6] = _mm512_mul_ps(y[6], y[6]);
                // y2[7] = _mm512_mul_ps(y[7], y[7]);

                z[0] = _mm512_add_ps(x[0], y[0]);
                z[1] = _mm512_add_ps(x[1], y[1]);
                z[2] = _mm512_add_ps(x[2], y[2]);
                z[3] = _mm512_add_ps(x[3], y[3]);
                // z[4] = _mm512_add_ps(x[4], y[4]);
                // z[5] = _mm512_add_ps(x[5], y[5]);
                // z[6] = _mm512_add_ps(x[6], y[6]);
                // z[7] = _mm512_add_ps(x[7], y[7]);

                w[0] = _mm512_mul_ps(z[0], z[0]);
                w[1] = _mm512_mul_ps(z[1], z[1]);
                w[2] = _mm512_mul_ps(z[2], z[2]);
                w[3] = _mm512_mul_ps(z[3], z[3]);
                // w[4] = _mm512_mul_ps(z[4], z[4]);
                // w[5] = _mm512_mul_ps(z[5], z[5]);
                // w[6] = _mm512_mul_ps(z[6], z[6]);
                // w[7] = _mm512_mul_ps(z[7], z[7]);

                // inc loop bounds
                iters[0] = _mm512_mask_add_epi32(iters[0], less_than_four[0], iters[0], _mm512_set1_epi32(1));
                iters[1] = _mm512_mask_add_epi32(iters[1], less_than_four[1], iters[1], _mm512_set1_epi32(1));
                iters[2] = _mm512_mask_add_epi32(iters[2], less_than_four[2], iters[2], _mm512_set1_epi32(1));
                iters[3] = _mm512_mask_add_epi32(iters[3], less_than_four[3], iters[3], _mm512_set1_epi32(1));
                // iters[4] = _mm512_mask_add_epi32(iters[4], less_than_four[4], iters[4], _mm512_set1_epi32(1));
                // iters[5] = _mm512_mask_add_epi32(iters[5], less_than_four[5], iters[5], _mm512_set1_epi32(1));
                // iters[6] = _mm512_mask_add_epi32(iters[6], less_than_four[6], iters[6], _mm512_set1_epi32(1));
                // iters[7] = _mm512_mask_add_epi32(iters[7], less_than_four[7], iters[7], _mm512_set1_epi32(1));

                // repeat the loop bound checks here
                x2_y2_sum[0] = _mm512_mask_add_ps(x2_y2_sum[0], less_than_four[0], x2[0], y2[0]);
                x2_y2_sum[1] = _mm512_mask_add_ps(x2_y2_sum[1], less_than_four[1], x2[1], y2[1]);
                x2_y2_sum[2] = _mm512_mask_add_ps(x2_y2_sum[2], less_than_four[2], x2[2], y2[2]);
                x2_y2_sum[3] = _mm512_mask_add_ps(x2_y2_sum[3], less_than_four[3], x2[3], y2[3]);
                // x2_y2_sum[4] = _mm512_mask_add_ps(x2_y2_sum[4], less_than_four[4], x2[4], y2[4]);
                // x2_y2_sum[5] = _mm512_mask_add_ps(x2_y2_sum[5], less_than_four[5], x2[5], y2[5]);
                // x2_y2_sum[6] = _mm512_mask_add_ps(x2_y2_sum[6], less_than_four[6], x2[6], y2[6]);
                // x2_y2_sum[7] = _mm512_mask_add_ps(x2_y2_sum[7], less_than_four[7], x2[7], y2[7]);

                less_than_four[0] = _mm512_cmp_ps_mask(x2_y2_sum[0], four_vec, 2);
                less_than_four[1] = _mm512_cmp_ps_mask(x2_y2_sum[1], four_vec, 2);
                less_than_four[2] = _mm512_cmp_ps_mask(x2_y2_sum[2], four_vec, 2);
                less_than_four[3] = _mm512_cmp_ps_mask(x2_y2_sum[3], four_vec, 2);
                // less_than_four[4] = _mm512_cmp_ps_mask(x2_y2_sum[4], four_vec, 2);
                // less_than_four[5] = _mm512_cmp_ps_mask(x2_y2_sum[5], four_vec, 2);
                // less_than_four[6] = _mm512_cmp_ps_mask(x2_y2_sum[6], four_vec, 2);
                // less_than_four[7] = _mm512_cmp_ps_mask(x2_y2_sum[7], four_vec, 2);

                iter++;
                // if (!less_than_four[0] && !less_than_four[1] && !less_than_four[2] && !less_than_four[3] && !less_than_four[4] && !less_than_four[5] && !less_than_four[6] && !less_than_four[7]) {
                if (!less_than_four[0] && !less_than_four[1] && !less_than_four[2] && !less_than_four[3]) {
                // if (!less_than_four[0] && !less_than_four[1]) {
                    break;
                }
            }
            _mm512_storeu_epi32(&(out[i[0] * img_size + ch * chunk_size]), iters[0]);
            _mm512_storeu_epi32(&(out[i[1] * img_size + ch * chunk_size]), iters[1]);
            _mm512_storeu_epi32(&(out[i[2] * img_size + ch * chunk_size]), iters[2]);
            _mm512_storeu_epi32(&(out[i[3] * img_size + ch * chunk_size]), iters[3]);  
            // _mm512_storeu_epi32(&(out[i[4] * img_size + ch * chunk_size]), iters[4]);
            // _mm512_storeu_epi32(&(out[i[5] * img_size + ch * chunk_size]), iters[5]);
            // _mm512_storeu_epi32(&(out[i[6] * img_size + ch * chunk_size]), iters[6]);
            // _mm512_storeu_epi32(&(out[i[7] * img_size + ch * chunk_size]), iters[7]);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core

typedef struct {
    uint32_t img_size;
    uint32_t max_iters;
    uint32_t *out;
    uint64_t thread_idx;
    int num_threads;
} mandelbrot_thread_data_t;

void* mandelbrot_cpu_vector_chunk_helper(void* arg) {
    mandelbrot_thread_data_t* data = (mandelbrot_thread_data_t*)arg;
    uint32_t img_size = data->img_size;
    uint32_t max_iters = data->max_iters;
    uint32_t* out = data->out;
    uint64_t thread_idx = data->thread_idx;
    int num_threads = data->num_threads;

    uint32_t j_chunk_size = 16;
    assert(img_size % j_chunk_size == 0);
    uint32_t num_j_chunks = img_size / j_chunk_size;

    __m512 img_size_vec = _mm512_set1_ps(img_size);
    __m512 window_zoom_vec = _mm512_set1_ps(window_zoom);
    __m512 window_x_vec = _mm512_set1_ps(window_x);
    __m512 window_y_vec = _mm512_set1_ps(window_y);
    __m512i max_iters_vec = _mm512_set1_epi32(max_iters);

    for (uint64_t i = thread_idx; i < img_size; i += num_threads) {
        for (uint64_t j_ch = 0; j_ch < num_j_chunks; j_ch++) {
            uint32_t base_j = j_ch * j_chunk_size;
            __m512 j_vec = _mm512_set_ps(base_j + 15, base_j + 14, base_j + 13, base_j + 12,
                                        base_j + 11, base_j + 10, base_j + 9, base_j + 8,
                                        base_j + 7, base_j + 6, base_j + 5, base_j + 4,
                                        base_j + 3, base_j + 2, base_j + 1, base_j + 0);
            __m512 i_vec = _mm512_set1_ps(i);
            __m512 cx_div = _mm512_div_ps(j_vec, img_size_vec);
            __m512 cx_mul = _mm512_mul_ps(cx_div, window_zoom_vec);
            __m512 cx = _mm512_add_ps(cx_mul, window_x_vec);
            __m512 cy_div = _mm512_div_ps(i_vec, img_size_vec);
            __m512 cy_mul = _mm512_mul_ps(cy_div, window_zoom_vec);
            __m512 cy = _mm512_add_ps(cy_mul, window_y_vec);

            __m512 x2 = _mm512_set1_ps(0);
            __m512 y2 = _mm512_set1_ps(0);
            __m512 w = _mm512_set1_ps(0);
            __m512i iters = _mm512_set1_epi32(0);

            __m512 x2_y2_sum = _mm512_add_ps(x2, y2);
            __m512 four_vec = _mm512_set1_ps(4.0f);
            __mmask16 less_than_four = _mm512_cmp_ps_mask(x2_y2_sum, four_vec, 2);
            __mmask16 less_than_max_iters = _mm512_cmp_epi32_mask(iters, max_iters_vec, 1);
            __mmask16 active_mask = _kand_mask16(less_than_four, less_than_max_iters);

            while (active_mask) {
                __m512 _x = _mm512_sub_ps(x2, y2);
                __m512 x = _mm512_add_ps(_x, cx);
                __m512 _y = _mm512_sub_ps(w, x2_y2_sum);
                __m512 y = _mm512_add_ps(_y, cy);

                x2 = _mm512_mul_ps(x, x);
                y2 = _mm512_mul_ps(y, y);
                __m512 z = _mm512_add_ps(x, y);
                w = _mm512_mul_ps(z, z);

                // inc loop bounds
                iters = _mm512_mask_add_epi32(iters, active_mask, iters, _mm512_set1_epi32(1));

                // repeat the loop bound checks here
                x2_y2_sum = _mm512_mask_add_ps(x2_y2_sum, active_mask, x2, y2);
                less_than_four = _mm512_cmp_ps_mask(x2_y2_sum, four_vec, 2);
                less_than_max_iters = _mm512_cmp_epi32_mask(iters, max_iters_vec, 1);
                active_mask = _kand_mask16(less_than_four, less_than_max_iters);
            }

            _mm512_storeu_epi32(&(out[i * img_size + j_ch * j_chunk_size]), iters);
        }
    }

    return nullptr;
}

void mandelbrot_cpu_vector_multicore(uint32_t img_size, uint32_t max_iters, uint32_t *out) {
    int num_threads = 8;
    std::vector<pthread_t> threads(num_threads);
    std::vector<mandelbrot_thread_data_t> thread_data(num_threads);

    for (int idx = 0; idx < num_threads; idx++) {
        thread_data[idx].img_size = img_size;
        thread_data[idx].max_iters = max_iters;
        thread_data[idx].out = out;
        thread_data[idx].thread_idx = idx;
        thread_data[idx].num_threads = num_threads;

        pthread_create(&threads.at(idx), NULL, mandelbrot_cpu_vector_chunk_helper, &thread_data.at(idx));
    }

    for (int idx = 0; idx < num_threads; idx++) {
        pthread_join(threads.at(idx), NULL);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core

void mandelbrot_cpu_vector_multicore_multithread(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    // TODO: Implement this function.
    int num_threads = 16;
    std::vector<pthread_t> threads(num_threads);
    std::vector<mandelbrot_thread_data_t> thread_data(num_threads);

    for (int idx = 0; idx < num_threads; idx++) {
        thread_data[idx].img_size = img_size;
        thread_data[idx].max_iters = max_iters;
        thread_data[idx].out = out;
        thread_data[idx].thread_idx = idx;
        thread_data[idx].num_threads = num_threads;

        pthread_create(&threads.at(idx), NULL, mandelbrot_cpu_vector_chunk_helper, &thread_data.at(idx));
    }

    for (int idx = 0; idx < num_threads; idx++) {
        pthread_join(threads.at(idx), NULL);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Vector + Multi-core + Multi-thread-per-core + ILP

void mandelbrot_cpu_vector_multicore_multithread_ilp(
    uint32_t img_size,
    uint32_t max_iters,
    uint32_t *out) {
    // TODO: Implement this function.
}

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <vector>

// Useful functions and structures.
enum MandelbrotImpl {
    SCALAR,
    VECTOR,
    VECTOR_ILP,
    VECTOR_MULTICORE,
    VECTOR_MULTICORE_MULTITHREAD,
    VECTOR_MULTICORE_MULTITHREAD_ILP,
    ALL
};

// Command-line arguments parser.
int ParseArgsAndMakeSpec(
    int argc,
    char *argv[],
    uint32_t *img_size,
    uint32_t *max_iters,
    MandelbrotImpl *impl) {
    char *implementation_str = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0) {
            if (i + 1 < argc) {
                *img_size = atoi(argv[++i]);
                if (*img_size % 32 != 0) {
                    std::cerr << "Error: Image width must be a multiple of 32"
                              << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -r" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                *max_iters = atoi(argv[++i]);
            } else {
                std::cerr << "Error: No value specified for -b" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                implementation_str = argv[++i];
                if (strcmp(implementation_str, "scalar") == 0) {
                    *impl = SCALAR;
                } else if (strcmp(implementation_str, "vector") == 0) {
                    *impl = VECTOR;
                } else if (strcmp(implementation_str, "vector_ilp") == 0) {
                    *impl = VECTOR_ILP;
                } else if (strcmp(implementation_str, "vector_multicore") == 0) {
                    *impl = VECTOR_MULTICORE;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD;
                } else if (
                    strcmp(implementation_str, "vector_multicore_multithread_ilp") == 0) {
                    *impl = VECTOR_MULTICORE_MULTITHREAD_ILP;
                } else if (strcmp(implementation_str, "all") == 0) {
                    *impl = ALL;
                } else {
                    std::cerr << "Error: unknown implementation" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: No value specified for -i" << std::endl;
                return 1;
            }
        } else {
            std::cerr << "Unknown flag: " << argv[i] << std::endl;
            return 1;
        }
    }
    std::cout << "Testing with image size " << *img_size << "x" << *img_size << " and "
              << *max_iters << " max iterations." << std::endl;

    return 0;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void writeBMP(const char *fname, uint32_t img_size, const std::vector<uint8_t> &pixels) {
    uint32_t width = img_size;
    uint32_t height = img_size;

    BMPHeader header;
    header.width = width;
    header.height = height;
    header.imageSize = width * height * 3;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    file.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
}

std::vector<uint8_t> iters_to_colors(
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    uint32_t width = img_size;
    uint32_t height = img_size;
    uint32_t min_iters = max_iters;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            min_iters = std::min(min_iters, iters[i * img_size + j]);
        }
    }
    float log_iters_min = log2f(static_cast<float>(min_iters));
    float log_iters_range =
        log2f(static_cast<float>(max_iters) / static_cast<float>(min_iters));
    auto pixel_data = std::vector<uint8_t>(width * height * 3);
    for (uint32_t i = 0; i < height; i++) {
        for (uint32_t j = 0; j < width; j++) {
            uint32_t iter = iters[i * width + j];

            uint8_t r = 0, g = 0, b = 0;
            if (iter < max_iters) {
                auto log_iter = log2f(static_cast<float>(iter)) - log_iters_min;
                auto intensity = static_cast<uint8_t>(log_iter * 222 / log_iters_range);
                r = 32;
                g = 32 + intensity;
                b = 32;
            }

            auto index = (i * width + j) * 3;
            pixel_data[index] = b;
            pixel_data[index + 1] = g;
            pixel_data[index + 2] = r;
        }
    }
    return pixel_data;
}

// Benchmarking macros and configuration.
static constexpr size_t kNumOfOuterIterations = 10;
static constexpr size_t kNumOfInnerIterations = 1;
#define BENCHPRESS(func, ...) \
    do { \
        std::cout << std::endl << "Running " << #func << " ...\n"; \
        std::vector<double> times(kNumOfOuterIterations); \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) { \
                func(__VA_ARGS__); \
            } \
            auto end = std::chrono::high_resolution_clock::now(); \
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) \
                           .count() / \
                kNumOfInnerIterations; \
        } \
        std::sort(times.begin(), times.end()); \
        std::stringstream sstream; \
        sstream << std::fixed << std::setw(6) << std::setprecision(2) \
                << times[0] / 1'000'000; \
        std::cout << "  Runtime: " << sstream.str() << " ms" << std::endl; \
    } while (0)

double difference(
    uint32_t img_size,
    uint32_t max_iters,
    std::vector<uint32_t> &result,
    std::vector<uint32_t> &ref_result) {
    int64_t diff = 0;
    for (uint32_t i = 0; i < img_size; i++) {
        for (uint32_t j = 0; j < img_size; j++) {
            diff +=
                abs(int(result[i * img_size + j]) - int(ref_result[i * img_size + j]));
        }
    }
    return diff / double(img_size * img_size * max_iters);
}

void dump_image(
    const char *fname,
    uint32_t img_size,
    uint32_t max_iters,
    const std::vector<uint32_t> &iters) {
    // Dump result as an image.
    auto pixel_data = iters_to_colors(img_size, max_iters, iters);
    writeBMP(fname, img_size, pixel_data);
}

// Main function.
// Compile with:
//  g++ -march=native -O3 -Wall -Wextra -o mandelbrot mandelbrot_cpu.cc
int main(int argc, char *argv[]) {
    // Get Mandelbrot spec.
    uint32_t img_size = 1024;
    uint32_t max_iters = default_max_iters;
    enum MandelbrotImpl impl = ALL;
    if (ParseArgsAndMakeSpec(argc, argv, &img_size, &max_iters, &impl))
        return -1;

    // Allocate memory.
    std::vector<uint32_t> result(img_size * img_size);
    std::vector<uint32_t> ref_result(img_size * img_size);

    // Compute the reference solution
    mandelbrot_cpu_scalar(img_size, max_iters, ref_result.data());

    // Test the desired kernels.
    if (impl == SCALAR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_scalar, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_scalar.bmp", img_size, max_iters, result);
    }

#ifdef HAS_VECTOR_IMPL
    if (impl == VECTOR || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }
#endif

    if (impl == VECTOR_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_ilp, img_size, max_iters, result.data());
        dump_image("out/mandelbrot_cpu_vector_ilp.bmp", img_size, max_iters, result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(mandelbrot_cpu_vector_multicore, img_size, max_iters, result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    if (impl == VECTOR_MULTICORE_MULTITHREAD_ILP || impl == ALL) {
        memset(result.data(), 0, sizeof(uint32_t) * img_size * img_size);
        BENCHPRESS(
            mandelbrot_cpu_vector_multicore_multithread_ilp,
            img_size,
            max_iters,
            result.data());
        dump_image(
            "out/mandelbrot_cpu_vector_multicore_multithread_ilp.bmp",
            img_size,
            max_iters,
            result);

        std::cout << "  Correctness: average output difference from reference = "
                  << difference(img_size, max_iters, result, ref_result) << std::endl;
    }

    return 0;
}
