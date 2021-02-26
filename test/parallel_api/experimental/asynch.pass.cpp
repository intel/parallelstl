// -*- C++ -*-
//===-- async.pass.cpp ----------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "oneapi/dpl/execution"
#include "oneapi/dpl/iterator"

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#   include "oneapi/dpl/async"
#   include <CL/sycl.hpp>
#endif

#include <iostream>
#include <iomanip>
#include <numeric>

template <typename _T1, typename _T2>
void
ASSERT_EQUAL(_T1&& X, _T2&& Y)
{
    if (X != Y)
        std::cout << "CHECK CORRECTNESS (ASYNC): fail (" << X << "," << Y << ")" << std::endl;
}

#if TEST_DPCPP_BACKEND_PRESENT
void
test_with_buffers()
{
    const int n = 100;
    {
        //sycl::queue q;
        sycl::buffer<int> x{n};
        sycl::buffer<int> y{n};

        //auto my_policy = oneapi::dpl::execution::make_device_policy(q);
        auto my_policy = oneapi::dpl::execution::make_device_policy<class Copy1>(oneapi::dpl::execution::dpcpp_default);
        auto res_1a = oneapi::dpl::experimental::copy_async(my_policy, oneapi::dpl::counting_iterator<int>(0),
                                                            oneapi::dpl::counting_iterator<int>(n),
                                                            oneapi::dpl::begin(x)); // x = [0..n]
        auto my_policy1 = oneapi::dpl::execution::make_device_policy<class Fill1>(my_policy);
        auto res_1b = oneapi::dpl::experimental::fill_async(my_policy1, oneapi::dpl::begin(y), oneapi::dpl::end(y),
                                                            7); // y = [7..7]
        auto my_policy2 = oneapi::dpl::execution::make_device_policy<class ForEach1>(my_policy);
        auto res_2a = oneapi::dpl::experimental::for_each_async(
            my_policy2, oneapi::dpl::begin(x), oneapi::dpl::end(x), [](int& e) { ++e; }, res_1a); // x = [1..n]
        auto my_policy3 = oneapi::dpl::execution::make_device_policy<class Transform1>(my_policy);
        auto res_2b = oneapi::dpl::experimental::transform_async(
            my_policy3, oneapi::dpl::begin(y), oneapi::dpl::end(y), oneapi::dpl::begin(y),
            [](const int& e) { return e / 2; },
            res_1b); // y = [3..3]

        sycl::buffer<int> z{n}; //std::vector<int> z(n);
        auto my_policy4 = oneapi::dpl::execution::make_device_policy<class Transform2>(my_policy);
        auto res_3 = oneapi::dpl::experimental::transform_async(my_policy4, oneapi::dpl::begin(x), oneapi::dpl::end(x),
                                                                oneapi::dpl::begin(y), oneapi::dpl::begin(z),
                                                                std::plus<int>(), res_2a, res_2b); // z = [4..n+3]
        auto my_policy5 = oneapi::dpl::execution::make_device_policy<class Reduce1>(my_policy);
        auto alpha = oneapi::dpl::experimental::reduce_async(my_policy5, oneapi::dpl::begin(x), oneapi::dpl::end(x), 0,
                                                             std::plus<int>(),
                                                             res_2a)
                         .get(); // alpha = n*(n+1)/2
        auto my_policy6 = oneapi::dpl::execution::make_device_policy<class Reduce2>(my_policy);
        auto beta =
            oneapi::dpl::experimental::transform_reduce_async(my_policy6, oneapi::dpl::begin(z), oneapi::dpl::end(z), 0,
                                                              std::plus<int>(), [=](int e) { return alpha * e; })
                .get();
        const int expected1 = (n * (n + 1) / 2) * ((n + 3) * (n + 4) / 2 - 6);
        EXPECT_TRUE(beta == expected1, "wrong effect from async test with sycl buffer");
    }
}

void
test_with_usm()
{
    cl::sycl::queue q;
    const int n = 1024;
    const int n_small = 13;

    // ASYNC TEST USING USM //
    // TODO: Extend tests by checking true async behavior in more detail
    {
        // Allocate space for data using USM.
        uint64_t* data1 =
            static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));
        uint64_t* data2 =
            static_cast<uint64_t*>(cl::sycl::malloc_shared(n * sizeof(uint64_t), q.get_device(), q.get_context()));

        // Initialize data
        for (int i = 0; i != n - 1; ++i)
        {
            data1[i] = i % 4 + 1;
            data2[i] = data1[i] + 1;
            if (i > 3 && i != n - 2)
            {
                ++i;
                data1[i] = data1[i - 1];
                data2[i] = data2[i - 1];
            }
        }
        data1[n - 1] = 0;
        data2[n - 1] = 0;

        // compute reference values
        const uint64_t ref1 = std::inner_product(data2, data2 + n, data1, 0);
        const uint64_t ref2 = std::accumulate(data1, data1 + n_small, 0);

        // call first algorithm
        auto new_policy1 = oneapi::dpl::execution::make_device_policy<class async1>(q);
        auto fut1 = oneapi::dpl::experimental::transform_reduce_async(
            new_policy1, data2, data2 + n, data1, 0, std::plus<uint64_t>(), std::multiplies<uint64_t>());

        // call second algorithm and wait for result
        auto new_policy2 = oneapi::dpl::execution::make_device_policy<class async2>(q);
        auto res2 = oneapi::dpl::experimental::reduce_async(new_policy2, data1, data1 + n_small).get();

        // call third algorithm that has to wait for first to complete
        auto new_policy3 = oneapi::dpl::execution::make_device_policy<class async3>(q);
        oneapi::dpl::experimental::sort_async(new_policy3, data2, data2 + n, fut1);

        // check values
        auto res1 = fut1.get();
        EXPECT_TRUE(res1 == ref1, "wrong effect from async transform reduce with usm");
        EXPECT_TRUE(res2 == ref2, "wrong effect from async reduce with usm");

        sycl::free(data1, q);
        sycl::free(data2, q);
    }
}
#endif

int
main()
{
#if TEST_DPCPP_BACKEND_PRESENT
    test_with_buffers();
    test_with_usm();
#endif
    std::cout << "done" << std::endl;
    return 0;
}
