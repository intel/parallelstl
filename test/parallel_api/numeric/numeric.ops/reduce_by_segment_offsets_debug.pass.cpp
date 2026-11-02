// -*- C++ -*-
//===-- reduce_by_segment_offsets_debug.pass.cpp --------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A simple test for debugging reduce_by_segment with segments defined by offsets:
// 1M int elements split into segments of pseudo-random lengths

#include "support/test_config.h"

#include "oneapi/dpl/execution"
#include "oneapi/dpl/numeric"

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/utils_sycl.h"
#endif

#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    constexpr std::size_t n = 1 << 20;

    // Segments of lengths 0..1023 (mean ~512), with a few long ones of 64K
    std::vector<std::size_t> offsets{0};
    std::mt19937 gen(42);
    std::uniform_int_distribution<std::size_t> len_dist(0, 1023);
    while (offsets.back() < n)
    {
        const std::size_t len = offsets.size() % 500 == 0 ? 65536 : len_dist(gen);
        offsets.push_back(std::min(n, offsets.back() + len));
    }
    const std::size_t m = offsets.size() - 1;

    std::vector<int> input(n);
    for (std::size_t i = 0; i < n; ++i)
        input[i] = int(i % 7) - 3;

    // Expected results
    const int init = 1;
    std::vector<int> expected(m);
    for (std::size_t s = 0; s < m; ++s)
    {
        int acc = init;
        for (std::size_t i = offsets[s]; i < offsets[s + 1]; ++i)
            acc += input[i];
        expected[s] = acc;
    }

    sycl::queue q = TestUtils::get_test_queue();
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << ", n = " << n
              << ", segments = " << m << std::endl;

    int* d_input = sycl::malloc_device<int>(n, q);
    std::size_t* d_offsets = sycl::malloc_device<std::size_t>(m + 1, q);
    int* d_result = sycl::malloc_device<int>(m, q);
    q.copy(input.data(), d_input, n).wait();
    q.copy(offsets.data(), d_offsets, m + 1).wait();

    auto policy = TestUtils::get_dpcpp_test_policy();
    oneapi::dpl::reduce_by_segment(policy, d_input, d_input + n, d_result, m, d_offsets, init, std::plus<int>{});

    std::vector<int> result(m);
    q.copy(d_result, result.data(), m).wait();

    for (std::size_t s = 0; s < m; ++s)
    {
        if (result[s] != expected[s])
        {
            std::cout << "Mismatch at segment " << s << " [" << offsets[s] << ", " << offsets[s + 1]
                      << "): expected " << expected[s] << ", got " << result[s] << std::endl;
            break;
        }
    }
    EXPECT_EQ_N(expected.begin(), result.begin(), m, "wrong result of reduce_by_segment with offsets");

    sycl::free(d_input, q);
    sycl::free(d_offsets, q);
    sycl::free(d_result, q);
#endif // TEST_DPCPP_BACKEND_PRESENT

    return TestUtils::done(TEST_DPCPP_BACKEND_PRESENT);
}
