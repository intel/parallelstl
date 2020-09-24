// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#ifndef _PSTL_EXECUTION_POLICY_DEFS_H
#define _PSTL_EXECUTION_POLICY_DEFS_H

#include <type_traits>
#include <iterator>

namespace oneapi
{
namespace dpl
{
namespace execution
{
inline namespace v1
{

// 2.4, Sequential execution policy
class sequenced_policy
{
  public:
    // For internal use only
    static constexpr ::std::false_type
    __allow_unsequenced()
    {
        return ::std::false_type{};
    }
    static constexpr ::std::false_type
    __allow_vector()
    {
        return ::std::false_type{};
    }
    static constexpr ::std::false_type
    __allow_parallel()
    {
        return ::std::false_type{};
    }
};

#if _PSTL_USE_PAR_POLICIES
// 2.5, Parallel execution policy
class parallel_policy
{
  public:
    // For internal use only
    static constexpr ::std::false_type
    __allow_unsequenced()
    {
        return ::std::false_type{};
    }
    static constexpr ::std::false_type
    __allow_vector()
    {
        return ::std::false_type{};
    }
    static constexpr ::std::true_type
    __allow_parallel()
    {
        return ::std::true_type{};
    }
};

// 2.6, Parallel+Vector execution policy
class parallel_unsequenced_policy
{
  public:
    // For internal use only
    static constexpr ::std::true_type
    __allow_unsequenced()
    {
        return ::std::true_type{};
    }
    static constexpr ::std::true_type
    __allow_vector()
    {
        return ::std::true_type{};
    }
    static constexpr ::std::true_type
    __allow_parallel()
    {
        return ::std::true_type{};
    }
};
#endif

class unsequenced_policy
{
  public:
    // For internal use only
    static constexpr ::std::true_type
    __allow_unsequenced()
    {
        return ::std::true_type{};
    }
    static constexpr ::std::true_type
    __allow_vector()
    {
        return ::std::true_type{};
    }
    static constexpr ::std::false_type
    __allow_parallel()
    {
        return ::std::false_type{};
    }
};

// 2.8, Execution policy objects
constexpr sequenced_policy seq{};
#if _PSTL_USE_PAR_POLICIES
constexpr parallel_policy par{};
constexpr parallel_unsequenced_policy par_unseq{};
#endif
constexpr unsequenced_policy unseq{};

// 2.3, Execution policy type trait
template <class T>
struct is_execution_policy : ::std::false_type
{
};

template <>
struct is_execution_policy<oneapi::dpl::execution::sequenced_policy> : ::std::true_type
{
};
#if _PSTL_USE_PAR_POLICIES
template <>
struct is_execution_policy<oneapi::dpl::execution::parallel_policy> : ::std::true_type
{
};
template <>
struct is_execution_policy<oneapi::dpl::execution::parallel_unsequenced_policy> : ::std::true_type
{
};
#endif
template <>
struct is_execution_policy<oneapi::dpl::execution::unsequenced_policy> : ::std::true_type
{
};

#if _PSTL_CPP14_VARIABLE_TEMPLATES_PRESENT
template <class T>
constexpr bool is_execution_policy_v = oneapi::dpl::execution::is_execution_policy<T>::value;
#endif

} // namespace v1
} // namespace execution

namespace __internal
{

// Extension: host execution policy type trait
template <class _T>
struct __is_host_execution_policy : ::std::false_type
{
};

template <>
struct __is_host_execution_policy<oneapi::dpl::execution::sequenced_policy> : ::std::true_type
{
};
#if _PSTL_USE_PAR_POLICIES
template <>
struct __is_host_execution_policy<oneapi::dpl::execution::parallel_policy> : ::std::true_type
{
};
template <>
struct __is_host_execution_policy<oneapi::dpl::execution::parallel_unsequenced_policy> : ::std::true_type
{
};
#endif
template <>
struct __is_host_execution_policy<oneapi::dpl::execution::unsequenced_policy> : ::std::true_type
{
};

template <class _ExecPolicy, class _T>
using __enable_if_execution_policy = typename ::std::enable_if<
    oneapi::dpl::execution::is_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;

template <class _ExecPolicy, class _T>
using __enable_if_host_execution_policy = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_host_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value, _T>::type;

template <class _ExecPolicy, const bool __condition, class _T>
using __enable_if_host_execution_policy_conditional = typename ::std::enable_if<
    oneapi::dpl::__internal::__is_host_execution_policy<typename ::std::decay<_ExecPolicy>::type>::value && __condition,
    _T>::type;

template <typename _ExecPolicy, typename _T>
struct __ref_or_copy_impl
{
    using type = const _T&;
};

template <typename _ExecPolicy, typename _T>
using __ref_or_copy =
    typename oneapi::dpl::__internal::__ref_or_copy_impl<typename ::std::decay<_ExecPolicy>::type, _T>::type;

template <typename _R>
using __difference_t = decltype(::std::declval<_R&>().size());

template <typename _R>
using __value_t = typename ::std::iterator_traits<decltype(::std::declval<_R&>().begin())>::value_type;

} // namespace __internal

} // namespace dpl
} // namespace oneapi

#if _PSTL_BACKEND_SYCL
#    include "hetero/dpcpp/execution_sycl_defs.h"
#endif

#endif /* _PSTL_EXECUTION_POLICY_DEFS_H */
