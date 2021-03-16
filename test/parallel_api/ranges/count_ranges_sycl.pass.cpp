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
    int data[max_n]     = {0, 1, -1, 3, 4, 5, 6, -1, 8, 9};

    auto lambda = [](auto i) { return i%2 == 0; };

    auto res1 = 0, res2 = 0;
    using namespace oneapi::dpl::experimental::ranges;
    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));

        auto view = views::all(A);
                                       
        res1 = count_if(TestUtils::default_dpcpp_policy, view, lambda);
        res2 = count(TestUtils::default_dpcpp_policy, A, -1);
    }

    EXPECT_TRUE(res1 == 4, "wrong result from count_if with sycl ranges");
    EXPECT_TRUE(res2 == 2, "wrong result from count with sycl ranges");
#endif //_ENABLE_RANGES_TESTING
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
