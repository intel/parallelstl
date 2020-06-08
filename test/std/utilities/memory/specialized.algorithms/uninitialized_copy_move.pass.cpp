// -*- C++ -*-
//===-- uninitialized_copy_move.pass.cpp ----------------------------------===//
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

// Tests for uninitialized_copy, uninitialized_copy_n, uninitialized_move, uninitialized_move_n

#include "support/pstl_test_config.h"
#include "support/utils.h"

#include _PSTL_TEST_HEADER(execution)
#include _PSTL_TEST_HEADER(memory)

#if  !defined(UNITIALIZED_COPY) && !defined(UNITIALIZED_COPY_N) &&\
     !defined(UNITIALIZED_MOVE) && !defined(UNITIALIZED_MOVE_N)
#define UNITIALIZED_COPY
#define UNITIALIZED_COPY_N
#define UNITIALIZED_MOVE
#define UNITIALIZED_MOVE_N
#endif

using namespace TestUtils;

// function of checking correctness for uninitialized.construct.value
template <typename InputIterator, typename OutputIterator, typename Size>
bool
IsCheckValueCorrectness(InputIterator first1, OutputIterator first2, Size n)
{
    for (Size i = 0; i < n; ++i, ++first1, ++first2)
    {
        if (*first1 != *first2)
        {
            return false;
        }
    }
    return true;
}

template <typename Type>
struct test_uninitialized_copy
{
    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, size_t n,
               /*is_trivial<T>=*/std::false_type)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;
        // it needs for cleaning memory that was filled by default constructors in unique_ptr<T[]> p(new T[n])
        // and for cleaning memory after last calling of uninitialized_value_construct_n.
        // It is important for non-trivial types
        std::destroy_n(dpstd::execution::seq, out_first, n);
        T::SetCount(0);

        std::uninitialized_copy(exec, first, last, out_first);
        EXPECT_TRUE(T::Count() == n, "wrong uninitialized_copy");
    }

#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN
    template <typename InputIterator, typename OutputIterator>
    void
    operator()(dpstd::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first,
               size_t n, /*is_trivial<T>=*/std::true_type)
    {
    }

    template <typename InputIterator, typename OutputIterator>
    void
    operator()(dpstd::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, size_t n, /*is_trivial<T>=*/std::true_type)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, size_t n,
               /*is_trivial<T>=*/std::true_type)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        std::uninitialized_copy(exec, first, last, out_first);
        EXPECT_TRUE(IsCheckValueCorrectness(first, out_first, n), "wrong uninitialized_copy");
    }
};

template <typename Type>
struct test_uninitialized_copy_n
{
    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, size_t n,
               /*is_trivial<T>=*/std::false_type)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;
        // it needs for cleaning memory that was filled by default constructors in unique_ptr<T[]> p(new T[n])
        // and for cleaning memory after last calling of uninitialized_value_construct_n.
        // It is important for non-trivial types
        std::destroy_n(dpstd::execution::seq, out_first, n);
        T::SetCount(0);

        std::uninitialized_copy_n(exec, first, n, out_first);
        EXPECT_TRUE(T::Count() == n, "wrong uninitialized_copy_n");
    }

#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN
    template <typename InputIterator, typename OutputIterator>
    void
    operator()(dpstd::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first,
               size_t n, /*is_trivial<T>=*/std::true_type)
    {
    }

    template <typename InputIterator, typename OutputIterator>
    void
    operator()(dpstd::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, size_t n, /*is_trivial<T>=*/std::true_type)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, size_t n,
               /*is_trivial<T>=*/std::true_type)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        std::uninitialized_copy_n(exec, first, n, out_first);
        EXPECT_TRUE(IsCheckValueCorrectness(first, out_first, n), "wrong uninitialized_copy_n");
    }
};

template <typename Type>
struct test_uninitialized_move
{
    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, size_t n,
               /*is_trivial<T>=*/std::false_type)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;
        // it needs for cleaning memory that was filled by default constructors in unique_ptr<T[]> p(new T[n])
        // and for cleaning memory after last calling of uninitialized_value_construct_n.
        // It is important for non-trivial types
        std::destroy_n(dpstd::execution::seq, out_first, n);
        T::SetCount(0);

        std::uninitialized_move(exec, first, last, out_first);
        EXPECT_TRUE(T::MoveCount() == n, "wrong uninitialized_move");
    }

#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN
    template <typename InputIterator, typename OutputIterator>
    void
    operator()(dpstd::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first,
               size_t n, /*is_trivial<T>=*/std::true_type)
    {
    }

    template <typename InputIterator, typename OutputIterator>
    void
    operator()(dpstd::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, size_t n, /*is_trivial<T>=*/std::true_type)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, size_t n,
               /*is_trivial<T>=*/std::true_type)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        std::uninitialized_move(exec, first, last, out_first);
        EXPECT_TRUE(IsCheckValueCorrectness(first, out_first, n), "wrong uninitialized_move");
    }
};

template <typename Type>
struct test_uninitialized_move_n
{
    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, size_t n,
               /*is_trivial<T>=*/std::false_type)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;
        // it needs for cleaning memory that was filled by default constructors in unique_ptr<T[]> p(new T[n])
        // and for cleaning memory after last calling of uninitialized_value_construct_n.
        // It is important for non-trivial types
        std::destroy_n(dpstd::execution::seq, out_first, n);
        T::SetCount(0);

        std::uninitialized_move_n(exec, first, n, out_first);
        EXPECT_TRUE(T::MoveCount() == n, "wrong uninitialized_move_n");
    }

#if _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN
    template <typename InputIterator, typename OutputIterator>
    void
    operator()(dpstd::execution::unsequenced_policy, InputIterator first, InputIterator last, OutputIterator out_first,
               size_t n, /*is_trivial<T>=*/std::true_type)
    {
    }

    template <typename InputIterator, typename OutputIterator>
    void
    operator()(dpstd::execution::parallel_unsequenced_policy, InputIterator first, InputIterator last,
               OutputIterator out_first, size_t n, /*is_trivial<T>=*/std::true_type)
    {
    }
#endif

    template <typename Policy, typename InputIterator, typename OutputIterator>
    void
    operator()(Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, size_t n,
               /*is_trivial<T>=*/std::true_type)
    {
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        std::uninitialized_move_n(exec, first, n, out_first);
        EXPECT_TRUE(IsCheckValueCorrectness(first, out_first, n), "wrong uninitialized_move_n");
    }
};

template <typename T>
void
test_uninitialized_copy_move_by_type()
{
    std::size_t N = 100000;
    for (size_t n = 0; n <= N; n = n <= 16 ? n + 1 : size_t(3.1415 * n))
    {
        Sequence<T> in(n, [=](size_t k) -> T { return T(k); });
#if !_PSTL_BACKEND_SYCL
        std::unique_ptr<T[]> p(new T[n]);
        auto out_begin = p.get();
#else
        // common pointers are not supported for hetero backend
        // cl::sycl::buffer<T,1> buf(n); // async nature of buffer requires sycn before EXPECT_ macro
        // auto out_begin = dpstd::begin(buf);
        Sequence<T> out(n, [=](size_t k) -> T { return T{}; });
        auto out_begin = out.begin();
        // TODO: "memory objects" should be created in another abstraction level
        //       to avoid multiple enable/disable macro for different backends
#endif
#ifdef UNITIALIZED_COPY
        invoke_on_all_policies<>()(test_uninitialized_copy<T>(), in.begin(), in.end(), out_begin, n,
                                   std::is_trivial<T>());
#endif
#ifdef UNITIALIZED_COPY_N
        invoke_on_all_policies<>()(test_uninitialized_copy_n<T>(), in.begin(), in.end(), out_begin, n,
                                   std::is_trivial<T>());
#endif
#ifdef UNITIALIZED_MOVE
        invoke_on_all_policies<>()(test_uninitialized_move<T>(), in.begin(), in.end(), out_begin, n,
                                   std::is_trivial<T>());
#endif
#ifdef UNITIALIZED_MOVE_N
        invoke_on_all_policies<>()(test_uninitialized_move_n<T>(), in.begin(), in.end(), out_begin, n,
                                   std::is_trivial<T>());
#endif
    }
}

int
main()
{

#if !_PSTL_DPCPP_TEST_UNINITIALIZED_BROKEN
    // for trivial types
    test_uninitialized_copy_move_by_type<int16_t>();
    test_uninitialized_copy_move_by_type<float64_t>();
#endif

#if !_PSTL_BACKEND_SYCL
    // for user-defined types
#if !_PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN && !_PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN &&     \
    !_PSTL_ICC_16_VC14_TEST_PAR_TBB_RT_RELEASE_64_BROKEN
    test_uninitialized_copy_move_by_type<Wrapper<int8_t>>();
#endif
#endif

    std::cout << done() << std::endl;
    return 0;
}
