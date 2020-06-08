// -*- C++ -*-
//===-- pstl_test_config.h ------------------------------------------------===//
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

#ifndef _PSTL_TEST_config_H
#define _PSTL_TEST_config_H

#define _PSTL_TEST_STRING(X) _PSTL_TEST_STRING_AUX(dpstd/X)
#define _PSTL_TEST_STRING_AUX(X) #X
//to support the optional including: <algorithm>, <memory>, <numeric> or <pstl/algorithm>, <pstl/memory>, <pstl/numeric>
#define _PSTL_TEST_HEADER(HEADER_ID) _PSTL_TEST_STRING(HEADER_ID)

#if defined(_MSC_VER) && defined(_DEBUG)
#define _SCL_SECURE_NO_WARNINGS //to prevent the compilation warning. Microsoft STL implementation has specific checking of an iterator range in DEBUG mode for the containers from the standard library.
#endif

#define _PSTL_ICC_16_17_TEST_REDUCTION_BOOL_TYPE_RELEASE_64_BROKEN                                                    \
    (__x86_64 && !_DEBUG && __INTEL_COMPILER && __INTEL_COMPILER <= 1700 && !__APPLE__)
#define _PSTL_ICC_16_17_TEST_REDUCTION_RELEASE_BROKEN                                                                 \
    (!_DEBUG && __INTEL_COMPILER &&                                                                                    \
     (__INTEL_COMPILER < 1800 || (__INTEL_COMPILER == 1800 && __INTEL_COMPILER_UPDATE < 1)))
#define _PSTL_ICC_1800_TEST_MONOTONIC_RELEASE_64_BROKEN                                                               \
    (__x86_64 && !_DEBUG && __INTEL_COMPILER && __INTEL_COMPILER == 1800 && __INTEL_COMPILER_UPDATE < 1)
#define _PSTL_ICC_17_TEST_MAC_RELEASE_32_BROKEN                                                                       \
    (__i386__ && !_DEBUG && __INTEL_COMPILER >= 1700 && __INTEL_COMPILER < 1800 && __APPLE__)
#define _PSTL_ICC_18_VC141_TEST_SIMD_LAMBDA_RELEASE_BROKEN                                                            \
    (!_DEBUG && __INTEL_COMPILER >= 1800 && __INTEL_COMPILER < 1900 && _MSC_VER == 1910)
#define _PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN                                                           \
    (_M_IX86 && _DEBUG && __INTEL_COMPILER >= 1700 && __INTEL_COMPILER < 1800 && _MSC_VER >= 1900)
#define _PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN                                                            \
    (_M_IX86 && _DEBUG && __INTEL_COMPILER >= 1600 && __INTEL_COMPILER < 1700 && _MSC_VER == 1900)
#define _PSTL_ICC_16_VC14_TEST_PAR_TBB_RT_RELEASE_64_BROKEN                                                           \
    (_PSTL_USE_PAR_POLICIES && ((_M_X64 && _MSC_VER == 1900) || __x86_64) && !_DEBUG && __INTEL_COMPILER < 1700)
#define _PSTL_ICC_16_17_TEST_64_TIMEOUT (__x86_64 && __INTEL_COMPILER && __INTEL_COMPILER < 1800 && !__APPLE__)
#define _PSTL_ICC_18_TEST_EARLY_EXIT_MONOTONIC_RELEASE_BROKEN (!_DEBUG && __INTEL_COMPILER && __INTEL_COMPILER == 1800)
#define _PSTL_CLANG_TEST_BIG_OBJ_DEBUG_32_BROKEN                                                                      \
    (__i386__ && PSTL_USE_DEBUG && __clang__ && _PSTL_CLANG_VERSION <= 90000)
#define _PSTL_ICC_16_17_18_TEST_UNIQUE_MASK_RELEASE_BROKEN                                                            \
    (!_DEBUG && __INTEL_COMPILER &&                                                                                    \
     (__INTEL_COMPILER < 1800 || (__INTEL_COMPILER == 1800 && __INTEL_COMPILER_UPDATE < 3)))
#define _PSTL_ICC_18_TEST_EARLY_EXIT_AVX_RELEASE_BROKEN                                                               \
    (!_DEBUG && __INTEL_COMPILER == 1800 && __AVX__ && !__AVX2__ && !__AVX512__)
#define _PSTL_ICC_19_TEST_IS_PARTITIONED_RELEASE_BROKEN                                                               \
    (!PSTL_USE_DEBUG && (__linux__ || __APPLE__) && __INTEL_COMPILER == 1900)
#define _PSTL_ICL_19_VC14_VC141_TEST_SCAN_RELEASE_BROKEN                                                              \
    (__INTEL_COMPILER == 1900 && _MSC_VER >= 1900 && _MSC_VER <= 1910)
#define _PSTL_ICC_19_TEST_SIMD_UDS_WINDOWS_RELEASE_BROKEN (__INTEL_COMPILER == 1900 && _MSC_VER && !_DEBUG)
#define _PSTL_ICC_18_19_TEST_SIMD_MONOTONIC_WINDOWS_RELEASE_BROKEN													   \
	((__INTEL_COMPILER == 1800 || __INTEL_COMPILER == 1900) && _MSC_VER && !_DEBUG)
#define _PSTL_ICC_18_19_TEST_REVERSE_ITERATOR_WITH_STRIDE_BROKEN                                                      \
    (__i386__ && (__INTEL_COMPILER == 1800 || __INTEL_COMPILER == 1900))
#define _PSTL_STD_UNINITIALIZED_FILL_BROKEN                                                                      \
    (_MSC_VER == 1900)

#define _SYCL_ARITHMETIC_TYPES_DESTRUCTION_BROKEN (__SYCL_COMPILER_VERSION <= 20191123)

#define _PSTL_SYCL_TEST_USM 1

#endif /* _PSTL_TEST_config_H */
