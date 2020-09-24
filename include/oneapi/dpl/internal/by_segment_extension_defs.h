/*
 *  Copyright (c) 2019-2020 Intel Corporation
 *
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef DPCPP_BY_SEGMENT_EXTENSION_DEFS_H_
#define DPCPP_BY_SEGMENT_EXTENSION_DEFS_H_

#include <oneapi/dpl/pstl/dpstd_config.h>

namespace oneapi
{
namespace dpl
{

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result, T init);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T,
          typename BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result, T init, BinaryPredicate binary_pred);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T,
          typename BinaryPredicate, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
exclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result, T init, BinaryPredicate binary_pred, BinaryOperator binary_op);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
inclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
inclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result, BinaryPredicate binary_pred);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator>
inclusive_scan_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                          OutputIterator result, BinaryPredicate binary_pred, BinaryOperator binary_op);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIterator1, OutputIterator2>>
reduce_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                  OutputIterator1 result1, OutputIterator2 result2);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIterator1, OutputIterator2>>
reduce_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                  OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, ::std::pair<OutputIterator1, OutputIterator2>>
reduce_by_segment(Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2,
                  OutputIterator1 result1, OutputIterator2 result2, BinaryPred binary_pred, BinaryOperator binary_op);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_DPSTD_DEPRECATED_MSG("Please replace use with exclusive_scan_by_segment")
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator> exclusive_scan_by_key(
    Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T>
_DPSTD_DEPRECATED_MSG("Please replace use with exclusive_scan_by_segment")
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator> exclusive_scan_by_key(
    Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, T init);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T,
          typename BinaryPredicate>
_DPSTD_DEPRECATED_MSG("Please replace use with exclusive_scan_by_segment")
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator> exclusive_scan_by_key(
    Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, T init,
    BinaryPredicate binary_pred);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename T,
          typename BinaryPredicate, typename BinaryOperator>
_DPSTD_DEPRECATED_MSG("Please replace use with exclusive_scan_by_segment")
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator> exclusive_scan_by_key(
    Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result, T init,
    BinaryPredicate binary_pred, BinaryOperator binary_op);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator>
_DPSTD_DEPRECATED_MSG("Please replace use with inclusive_scan_by_segment")
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator> inclusive_scan_by_key(
    Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate>
_DPSTD_DEPRECATED_MSG("Please replace use with inclusive_scan_by_segment")
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator> inclusive_scan_by_key(
    Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result,
    BinaryPredicate binary_pred);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator,
          typename BinaryPredicate, typename BinaryOperator>
_DPSTD_DEPRECATED_MSG("Please replace use with inclusive_scan_by_segment")
oneapi::dpl::__internal::__enable_if_execution_policy<Policy, OutputIterator> inclusive_scan_by_key(
    Policy&& policy, InputIterator1 first1, InputIterator1 last1, InputIterator2 first2, OutputIterator result,
    BinaryPredicate binary_pred, BinaryOperator binary_op);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2>
_DPSTD_DEPRECATED_MSG("Please replace use with reduce_by_segment")
oneapi::dpl::__internal::__enable_if_execution_policy<
    Policy, ::std::pair<OutputIterator1, OutputIterator2>> reduce_by_key(Policy&& policy, InputIterator1 first1,
                                                                         InputIterator1 last1, InputIterator2 first2,
                                                                         OutputIterator1 result1,
                                                                         OutputIterator2 result2);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred>
_DPSTD_DEPRECATED_MSG("Please replace use with reduce_by_segment")
oneapi::dpl::__internal::__enable_if_execution_policy<
    Policy, ::std::pair<OutputIterator1, OutputIterator2>> reduce_by_key(Policy&& policy, InputIterator1 first1,
                                                                         InputIterator1 last1, InputIterator2 first2,
                                                                         OutputIterator1 result1,
                                                                         OutputIterator2 result2,
                                                                         BinaryPred binary_pred);

template <typename Policy, typename InputIterator1, typename InputIterator2, typename OutputIterator1,
          typename OutputIterator2, typename BinaryPred, typename BinaryOperator>
_DPSTD_DEPRECATED_MSG("Please replace use with reduce_by_segment")
oneapi::dpl::__internal::__enable_if_execution_policy<
    Policy, ::std::pair<OutputIterator1, OutputIterator2>> reduce_by_key(Policy&& policy, InputIterator1 first1,
                                                                         InputIterator1 last1, InputIterator2 first2,
                                                                         OutputIterator1 result1,
                                                                         OutputIterator2 result2,
                                                                         BinaryPred binary_pred,
                                                                         BinaryOperator binary_op);
} // end namespace dpl
} // end namespace oneapi

#endif
