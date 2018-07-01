/*
    Copyright (c) 2017-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#ifndef __PSTL_TEST_config_H
#define __PSTL_TEST_config_H

#if defined(_MSC_VER) && defined(_DEBUG)
#define _SCL_SECURE_NO_WARNINGS //to prevent the compilation warning. Microsoft STL implementation has specific checking of an iterator range in DEBUG mode for the containers from the standard library.
#endif

#define __PSTL_ICC_16_17_TEST_REDUCTION_BOOL_TYPE_RELEASE_64_BROKEN (__x86_64 && !_DEBUG && __INTEL_COMPILER  && __INTEL_COMPILER <= 1700 && !__APPLE__)
#define __PSTL_ICC_16_17_TEST_REDUCTION_RELEASE_BROKEN (!_DEBUG && __INTEL_COMPILER  && (__INTEL_COMPILER < 1800 || (__INTEL_COMPILER == 1800 && __INTEL_COMPILER_UPDATE < 1)))
#define __PSTL_ICC_1800_TEST_MONOTONIC_RELEASE_64_BROKEN (__x86_64 && !_DEBUG && __INTEL_COMPILER  && __INTEL_COMPILER == 1800 && __INTEL_COMPILER_UPDATE < 1)
#define __PSTL_ICC_17_TEST_MAC_RELEASE_32_BROKEN (__i386__ && !_DEBUG && __INTEL_COMPILER >= 1700 && __INTEL_COMPILER < 1800 && __APPLE__)
#define __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN (_M_IX86 && _DEBUG && __INTEL_COMPILER >= 1700 && __INTEL_COMPILER < 1800 && _MSC_VER == 1910)
#define __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN (_M_IX86 && _DEBUG && __INTEL_COMPILER >= 1600 && __INTEL_COMPILER < 1700 && _MSC_VER == 1900)
#define __PSTL_ICC_16_VC14_TEST_PAR_TBB_RT_RELEASE_64_BROKEN (__PSTL_USE_PAR_POLICIES && _M_X64 && !_DEBUG && __INTEL_COMPILER < 1700 && _MSC_VER == 1900)
#define __PSTL_ICC_16_17_TEST_64_TIMEOUT (__x86_64 && __INTEL_COMPILER  && __INTEL_COMPILER < 1800 && !__APPLE__)
#define __PSTL_ICC_18_TEST_EARLY_EXIT_MONOTONIC_RELEASE_BROKEN (!_DEBUG && __INTEL_COMPILER  && __INTEL_COMPILER == 1800)
#define __PSTL_CLANG_TEST_BIG_OBJ_DEBUG_32_BROKEN (__i386__ && PSTL_USE_DEBUG && __clang__ &&  __PSTL_CLANG_VERSION <= 90000)

#endif /* __PSTL_TEST_config_H */
