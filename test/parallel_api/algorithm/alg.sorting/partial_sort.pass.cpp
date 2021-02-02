// -*- C++ -*-
//===-- partial_sort.pass.cpp ---------------------------------------------===//
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

#include "support/pstl_test_config.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#include "support/utils.h"

#include <cmath>

using namespace TestUtils;

#if !_ONEDPL_BACKEND_SYCL
static ::std::atomic<int32_t> count_val;
static ::std::atomic<int32_t> count_comp;

template <typename T>
struct Num
{
    T val;

    Num() { ++count_val; }
    Num(T v) : val(v) { ++count_val; }
    Num(const Num<T>& v) : val(v.val) { ++count_val; }
    Num(Num<T>&& v) : val(v.val) { ++count_val; }
    ~Num() { --count_val; }
    Num<T>&
    operator=(const Num<T>& v)
    {
        val = v.val;
        return *this;
    }
    operator T() const { return val; }
    bool
    operator<(const Num<T>& v) const
    {
        ++count_comp;
        return val < v.val;
    }
};
#endif

template <typename Type>
struct test_brick_partial_sort
{
    template <typename Policy, typename InputIterator, typename Compare>
    typename ::std::enable_if<is_same_iterator_category<InputIterator, ::std::random_access_iterator_tag>::value,
                            void>::type
    operator()(Policy&& exec, InputIterator first, InputIterator last, InputIterator exp_first, InputIterator exp_last,
               Compare compare)
    {

        typedef typename ::std::iterator_traits<InputIterator>::value_type T;

        // The rand()%(2*n+1) encourages generation of some duplicates.
        ::std::srand(42);
        const ::std::size_t n = last - first;
        for (::std::size_t k = 0; k < n; ++k)
        {
            first[k] = T(rand() % (2 * n + 1));
        }
        ::std::copy(first, last, exp_first);

        for (::std::size_t p = 0; p < n; p = p <= 16 ? p + 1 : ::std::size_t(31.415 * p))
        {
            auto m1 = first + p;
            auto m2 = exp_first + p;

            ::std::partial_sort(exp_first, m2, exp_last, compare);
#if !_ONEDPL_BACKEND_SYCL
            count_comp = 0;
#endif
            ::std::partial_sort(exec, first, m1, last, compare);
            EXPECT_EQ_N(exp_first, first, p, "wrong effect from partial_sort");

#if !_ONEDPL_BACKEND_SYCL
            //checking upper bound number of comparisons; O(p*(last-first)log(middle-first)); where p - number of threads;
            if (m1 - first > 1)
            {
                auto complex = ::std::ceil(n * ::std::log(float32_t(m1 - first)));
#if defined(_ONEDPL_PAR_BACKEND_TBB)
                auto p = tbb::this_task_arena::max_concurrency();
#else
                auto p = 1;
#endif

#ifdef _DEBUG
                if (count_comp > complex * p)
                {
                    ::std::cout << "complexity exceeded" << ::std::endl;
                }
#endif
            }
#endif // !_ONEDPL_BACKEND_SYCL
        }
    }

    template <typename Policy, typename InputIterator, typename Compare>
    typename ::std::enable_if<!is_same_iterator_category<InputIterator, ::std::random_access_iterator_tag>::value,
                            void>::type
    operator()(Policy&& exec, InputIterator first, InputIterator last, InputIterator exp_first, InputIterator exp_last,
               Compare compare)
    {
    }
};

template <typename T, typename Compare>
void
test_partial_sort(Compare compare)
{

    const ::std::size_t n_max = 100000;
    Sequence<T> in(n_max);
    Sequence<T> exp(n_max);
    for (::std::size_t n = 0; n < n_max; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        invoke_on_all_policies<0>()(test_brick_partial_sort<T>(), in.begin(), in.begin() + n, exp.begin(),
                                    exp.begin() + n, compare);
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        partial_sort(exec, iter, iter, iter, non_const(::std::less<T>()));
    }
};

int
main()
{
// Disable the test for SYCL as it relies on global atomic for counting number of comparisons
#if !_ONEDPL_BACKEND_SYCL
    count_val = 0;

    test_partial_sort<Num<float32_t>>([](Num<float32_t> x, Num<float32_t> y) { return x < y; });

    EXPECT_TRUE(count_val == 0, "cleanup error");
#endif

    test_partial_sort<int32_t>(
        [](int32_t x, int32_t y) { return x > y; }); // Reversed so accidental use of < will be detected.

    test_algo_basic_single<int32_t>(run_for_rnd<test_non_const<int32_t>>());

    ::std::cout << done() << ::std::endl;
    return 0;
}
