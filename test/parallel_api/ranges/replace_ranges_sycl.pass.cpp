// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include <oneapi/dpl/execution>

#include "support/pstl_test_config.h"

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    int data[max_n]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int expected[max_n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int val1 = -1, val2 = -2;

    auto lambda = [](auto i) { return i%2 == 0; };

    using namespace oneapi::dpl::experimental::ranges;

    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));

        auto view = views::all(A);
                                       
        replace_if(TestUtils::default_dpcpp_policy, view, lambda, val1);
        replace(TestUtils::default_dpcpp_policy, A, val1, val2);
    }

    //check result
    ::std::replace_if(expected, expected + max_n, lambda, val2);

    EXPECT_EQ_N(expected, data, max_n, "wrong effect from replace(_if) with sycl ranges");
#endif //_ENABLE_RANGES_TESTING
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}