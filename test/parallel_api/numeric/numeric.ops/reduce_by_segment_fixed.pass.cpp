// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// reduce_by_segment with segments of a fixed length (experiment): the results against a serial reference for the
// host policies and the device policy, lengths around the work-item, work-group and tile bounds of the device paths,
// segments crossing or beyond the end of the input, empty inputs, zero lengths and invalid arguments

#include "support/test_config.h"

#include "oneapi/dpl/execution"
#include "oneapi/dpl/numeric"

#include "support/utils.h"

#if TEST_DPCPP_BACKEND_PRESENT
#    include "support/utils_sycl.h"
#endif

#include <cstdint>    // std::int64_t
#include <functional> // std::plus
#include <string>     // std::string, std::to_string
#include <vector>     // std::vector

using namespace TestUtils;

// The reference: segment k is [min(k L, n), min((k + 1) L, n)), an empty one gets the init
template <typename T, typename Op>
std::vector<T>
reference(const std::vector<T>& in, std::int64_t num_segments, std::int64_t length, T init, Op op)
{
    const std::int64_t n = std::int64_t(in.size());
    std::vector<T> out(num_segments > 0 ? num_segments : 0);
    for (std::int64_t k = 0; k < num_segments; ++k)
    {
        const std::int64_t b = length == 0 ? 0 : (k <= n / length ? std::min(n, k * length) : n);
        const std::int64_t e = length == 0 ? 0 : (k + 1 <= n / length ? std::min(n, (k + 1) * length) : n);
        T acc = init;
        for (std::int64_t i = b; i < e; ++i)
            acc = op(acc, in[i]);
        out[k] = acc;
    }
    return out;
}

template <typename Policy, typename T, typename Op>
void
check(Policy&& policy, const char* name, std::int64_t n, std::int64_t num_segments, std::int64_t length, T init, Op op)
{
    std::vector<T> in(n);
    for (std::int64_t i = 0; i < n; ++i)
        in[i] = T(i % 7) - T(3);
    const std::vector<T> expected = reference(in, num_segments, length, init, op);
    std::vector<T> out(num_segments > 0 ? num_segments : 0, T(-77));

    auto res = oneapi::dpl::reduce_by_segment(std::forward<Policy>(policy), in.begin(), in.end(), out.begin(),
                                              num_segments, length, init, op);
    const std::string what = std::string(name) + " n=" + std::to_string(n) + " L=" + std::to_string(length) +
                             " m=" + std::to_string(num_segments);
    EXPECT_TRUE(res == out.begin() + (num_segments > 0 && length >= 0 ? num_segments : 0),
                ("wrong returned iterator: " + what).c_str());
    if (length >= 0)
        EXPECT_EQ_N(expected.begin(), out.begin(), expected.size(), ("wrong result: " + what).c_str());
}

// The integral checks with one policy and the float one with another, so that the kernel names of a device policy differ
template <typename PolicyInt, typename PolicyFloat>
void
run(PolicyInt&& policy_int, PolicyFloat&& policy_float, const char* name)
{
    for (std::int64_t n : {0, 1, 31, 32, 1000, 4096, 4097, 100000, 1 << 20})
        for (std::int64_t length : {0, 1, 3, 31, 32, 33, 255, 256, 1024, 4095, 4096, 4097, 8192, 70000, 1 << 21})
        {
            const std::int64_t whole = length == 0 ? 3 : (n + length - 1) / length;
            for (std::int64_t num_segments : {std::int64_t(1), whole, whole + 5, whole / 2 + 1})
                check(policy_int, name, n, num_segments, length, std::int64_t(11), std::plus<std::int64_t>{});
        }
    // Invalid arguments write nothing
    check(policy_int, name, 100, 0, 10, std::int64_t(1), std::plus<std::int64_t>{});
    check(policy_int, name, 100, -3, 10, std::int64_t(1), std::plus<std::int64_t>{});
    check(policy_int, name, 100, 4, -1, std::int64_t(1), std::plus<std::int64_t>{});
    // A float accumulator and a user operation without a known identity
    check(policy_float, name, 50000, 13, 4000, 0.5f, [](float a, float b) { return a + b; });
}

int
main()
{
    run(oneapi::dpl::execution::seq, oneapi::dpl::execution::seq, "seq");
    run(oneapi::dpl::execution::par, oneapi::dpl::execution::par, "par");
#if TEST_DPCPP_BACKEND_PRESENT
    auto policy = TestUtils::get_dpcpp_test_policy();
    run(CLONE_TEST_POLICY_IDX(policy, 0), CLONE_TEST_POLICY_IDX(policy, 1), "device");
#endif
    return TestUtils::done();
}
