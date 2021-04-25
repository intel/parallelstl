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

#ifndef _ONEDPL_EXECUTION_IMPL_H
#define _ONEDPL_EXECUTION_IMPL_H

#include <iterator>
#include <type_traits>

#include "execution_defs.h"
#include "iterator_defs.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{

/* policy */
template <typename Policy>
struct __policy_traits
{
};

template <>
struct __policy_traits<oneapi::dpl::execution::sequenced_policy>
{
    typedef ::std::false_type __allow_parallel;
    typedef ::std::false_type __allow_unsequenced;
    typedef ::std::false_type __allow_vector;
};

template <>
struct __policy_traits<oneapi::dpl::execution::unsequenced_policy>
{
    typedef ::std::false_type __allow_parallel;
    typedef ::std::true_type __allow_unsequenced;
    typedef ::std::true_type __allow_vector;
};

template <>
struct __policy_traits<oneapi::dpl::execution::parallel_policy>
{
    typedef ::std::true_type __allow_parallel;
    typedef ::std::false_type __allow_unsequenced;
    typedef ::std::false_type __allow_vector;
};

template <>
struct __policy_traits<oneapi::dpl::execution::parallel_unsequenced_policy>
{
    typedef ::std::true_type __allow_parallel;
    typedef ::std::true_type __allow_unsequenced;
    typedef ::std::true_type __allow_vector;
};

template <typename _ExecutionPolicy>
using __allow_vector =
    typename __internal::__policy_traits<typename ::std::decay<_ExecutionPolicy>::type>::__allow_vector;

template <typename _ExecutionPolicy>
using __allow_unsequenced =
    typename __internal::__policy_traits<typename ::std::decay<_ExecutionPolicy>::type>::__allow_unsequenced;

template <typename _ExecutionPolicy>
using __allow_parallel =
    typename __internal::__policy_traits<typename ::std::decay<_ExecutionPolicy>::type>::__allow_parallel;

//------------------------------------------------------------------------
// backend selector with tags
//------------------------------------------------------------------------

struct __tbb_backend {};

template <class _IsVector>
struct __serial_tag
{
    using __is_vector = _IsVector;
};

template <class _IsVector>
struct __parallel_tag
{
    using __is_vector = _IsVector;
    // backend tag can be change depending on
    // TBB availability in the environment
    using __backend_tag = __tbb_backend;
};

struct __parallel_forward_tag
{
    using __is_vector = ::std::false_type;
    // backend tag can be change depending on
    // TBB availability in the environment
    using __backend_tag = __tbb_backend;
};

template <class _IsVector, class... _IteratorTypes>
using __tag_type =
    typename ::std::conditional<__internal::__is_random_access_iterator<_IteratorTypes...>::value,
                                __parallel_tag<_IsVector>,
                                typename ::std::conditional<__is_forward_iterator<_IteratorTypes...>::value,
                                                            __parallel_forward_tag,
                                                            __serial_tag<_IsVector>
                                                            >::type
                                >::type;

template <class... _IteratorTypes>
__serial_tag<std::false_type>
__select_backend(oneapi::dpl::execution::sequenced_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__serial_tag<__internal::__is_random_access_iterator<_IteratorTypes...>>
__select_backend(oneapi::dpl::execution::unsequenced_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__tag_type<std::false_type, _IteratorTypes...>
__select_backend(oneapi::dpl::execution::parallel_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__tag_type<__internal::__is_random_access_iterator<_IteratorTypes...>, _IteratorTypes...>
__select_backend(oneapi::dpl::execution::parallel_unsequenced_policy, _IteratorTypes&&...)
{
    return {};
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

#endif /* _ONEDPL_EXECUTION_IMPL_H */
