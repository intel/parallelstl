// -*- C++ -*-
//===-- replace.pass.cpp --------------------------------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
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
#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(algorithm)

#if  !defined(_PSTL_TEST_REPLACE) && !defined(_PSTL_TEST_REPLACE_IF)
#define _PSTL_TEST_REPLACE
#define _PSTL_TEST_REPLACE_IF
#endif

using namespace TestUtils;

// This class is needed to check the self-copying
struct copy_int
{
    int32_t value;
    int32_t copied_times = 0;
    explicit copy_int(int32_t val = 0) { value = val; }

    copy_int&
    operator=(const copy_int& other)
    {
#if _PSTL_BACKEND_SYCL==0
        if (&other == this)
            copied_times++;
        else
#endif
        {
            value = other.value;
            copied_times = other.copied_times;
        }
        return *this;
    }

    bool
    operator==(const copy_int& other) const
    {
        return (value == other.value);
    }
};

template <typename T1, typename T2>
struct test_replace
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename T, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 expected_b, Iterator1 expected_e, Iterator2 actual_b,
               Iterator2 actual_e, Iterator3 data_b, Iterator3 data_e, Predicate pred, const T& value, const T& old_value)
    {
        using namespace std;

        copy(data_b, data_e, expected_b);
        copy(data_b, data_e, actual_b);

        replace(expected_b, expected_e, old_value, value);
        replace(exec, actual_b, actual_e, old_value, value);

        EXPECT_TRUE((check<T, Iterator2>(actual_b, actual_e)), "wrong result of self assignment check");
        EXPECT_TRUE(equal(expected_b, expected_e, actual_b), "wrong result of replace");
    }

    template <typename T, typename Iterator1>
    bool
    check(Iterator1 b, Iterator1 e)
    {
        return true;
    }

    template <typename T, typename Iterator1>
    typename std::enable_if<std::is_same<T, copy_int>::value, bool>::type_t
    check(Iterator1 b, Iterator1 e)
    {
        return std::all_of(b, e, [](const copy_int& elem) { return elem.copied_times == 0; });
    }
};

template<typename T1, typename T2>
struct test_replace_if
{
    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename T, typename Predicate>
    void
    operator()(ExecutionPolicy&& exec, Iterator1 expected_b, Iterator1 expected_e, Iterator2 actual_b,
               Iterator2 actual_e, Iterator3 data_b, Iterator3 data_e, Predicate pred, const T& value, const T& old_value)
    {
        using namespace std;

        copy(data_b, data_e, expected_b);
        copy(data_b, data_e, actual_b);

        replace_if(expected_b, expected_e, pred, value);
        replace_if(exec, actual_b, actual_e, pred, value);
        EXPECT_TRUE(equal(expected_b, expected_e, actual_b), "wrong result of replace_if");
    }
};

template <typename T1, typename T2, typename Pred>
void
test(Pred pred)
{
    typedef typename Sequence<T2>::iterator iterator_type;

    const std::size_t max_len = 100000;

    const T1 value = T1(0);
    const T1 new_value = T1(666);

    Sequence<T2> expected(max_len);
    Sequence<T2> actual(max_len);

    Sequence<T2> data(max_len, [&value](std::size_t i) {
        if (i % 3 == 2)
        {
            return T1(i);
        }
        else
        {
            return value;
        }
    });

    for (std::size_t len = 0; len < max_len; len = len <= 16 ? len + 1 : std::size_t(3.1415 * len))
    {
#ifdef _PSTL_TEST_REPLACE
        invoke_on_all_policies<0>()(test_replace<T1, T2>{},
                                    expected.begin(), expected.begin() + len,
                                    actual.begin(), actual.begin() + len,
                                    data.begin(), data.begin() + len,
                                    pred, new_value, value);
#endif
#ifdef _PSTL_TEST_REPLACE_IF
        invoke_on_all_policies<1>()(test_replace_if<T1, T2>{},
                                    expected.begin(), expected.begin() + len,
                                    actual.begin(), actual.begin() + len,
                                    data.begin(), data.begin() + len,
                                    pred, new_value, value);
#endif
    }
}

template <typename T>
struct test_non_const
{
    template <typename Policy, typename Iterator>
    void
    operator()(Policy&& exec, Iterator iter)
    {
        auto is_even = [&](float64_t v) {
            uint32_t i = (uint32_t)v;
            return i % 2 == 0;
        };
        invoke_if(exec, [&]() { replace_if(exec, iter, iter, non_const(is_even), T(0)); });
    }
};

int
main()
{
    test<int32_t, float32_t>(dpstd::__internal::__equal_value<int32_t>(666));
    test<uint16_t, uint8_t>([](const uint16_t& elem) { return elem % 3 < 2; });
    test<float64_t, int64_t>([](const float64_t& elem) { return elem * elem - 3.5 * elem > 10; });
    //test<copy_int, copy_int>([](const copy_int& val) { return val.value / 5 > 2; });

    //test_algo_basic_single<int32_t>(run_for_rnd_fw<test_non_const<int32_t>>());
#ifdef _PSTL_TEST_REPLACE_IF
    test_algo_basic_single<int16_t>(run_for_rnd_fw<test_non_const<int16_t>>());
#endif

    std::cout << done() << std::endl;
    return 0;
}
