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

#include "support/pstl_test_config.h"

#ifdef PSTL_STANDALONE_TESTS
#include "pstl/execution"
#include "pstl/algorithm"
#else
#include <execution>
#include <algorithm>
#endif // PSTL_STANDALONE_TESTS

#include "support/parallel_utils.h"

using namespace Parallel_TestUtils;

struct test_one_policy {
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN//dummy specialization by policy type, in case of broken configuration
    template <typename Iterator1, typename Iterator2, typename Predicate>
    void operator()(pstl::execution::unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred) { }
    template <typename Iterator1, typename Iterator2, typename Predicate>
    void operator()(pstl::execution::parallel_unsequenced_policy, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred) { }
#endif

    template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate>
    void operator()(ExecutionPolicy&& exec, Iterator1 b, Iterator1 e, Iterator2 bsub, Iterator2 esub, Predicate pred) {
        using namespace std;
        Iterator1 expected = find_first_of(b, e, bsub, esub, pred);
        Iterator1 actual = find_first_of(exec, b, e, bsub, esub, pred);
        EXPECT_TRUE(actual == expected, "wrong return result from find_first_of with a predicate");

        expected = find_first_of(b, e, bsub, esub);
        actual = find_first_of(exec, b, e, bsub, esub);
        EXPECT_TRUE(actual == expected, "wrong return result from find_first_of");
    }
};

template <typename T, typename Predicate>
void test(Predicate pred) {

    const std::size_t max_n1 = 1000;
    const std::size_t max_n2 = (max_n1 * 10) / 8;
    Sequence<T> in1(max_n1, [](std::size_t k) {return T(1); });
    Sequence<T> in2(max_n2, [](std::size_t k) {return T(0); });
    for (std::size_t n1 = 0; n1 <= max_n1; n1 = n1 <= 16 ? n1 + 1 : size_t(3.1415 * n1)) {
        std::size_t sub_n[] = { 0, 1, n1 / 3, n1, (n1 * 10) / 8 };
        for(const auto n2 : sub_n) {
            bool not_0 = n2 != 0;
            invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + n1, in2.data(), in2.data() + n2, pred);

            in2[n2 / 2] = T(1);
            invoke_on_all_policies(test_one_policy(), in1.cbegin(), in1.cbegin() + n1, in2.data(), in2.data() + n2, pred);

            if (n2 >= 3) {
                in2[2 * n2 / 3] = T(1);
                invoke_on_all_policies(test_one_policy(), in1.cbegin(), in1.cbegin() + n1, in2.begin(), in2.begin() + n2, pred);
                in2[2 * n2 / 3] = T(0);
            }
            in2[n2 / 2] = T(0);
        }
    }
    invoke_on_all_policies(test_one_policy(), in1.begin(), in1.begin() + max_n1 / 10, in1.data(), in1.data() + max_n1 / 10, pred);
}

int32_t main() {
    test<int32_t>(std::equal_to<int32_t>());
    test<uint16_t>(std::not_equal_to<uint16_t>());
    test<float64_t>([](const float64_t x, const float64_t y) {return x*x == y*y; });

    std::cout << done() << std::endl;
    return 0;
}
