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

#include <type_traits>
#ifdef PSTL_STANDALONE_TESTS
#include "pstl/execution"
#include "pstl/algorithm"
#else
#include <execution>
#include <algorithm>
#endif // PSTL_STANDALONE_TESTS

#include "support/parallel_utils.h"

using namespace Parallel_TestUtils;

template<typename InputIterator, typename OutputIterator>
void check_and_reset(InputIterator first, InputIterator last, OutputIterator out_first) {
    typedef typename std::iterator_traits<OutputIterator>::value_type Out;
    typename std::iterator_traits<OutputIterator>::difference_type k = 0;
    for (; first != last; ++first, ++out_first, ++k ) {
        // check
        Out expected = 1 - *first;
        Out actual = *out_first;
        EXPECT_EQ(expected, actual, "wrong value in output sequence");
        // reset
        *out_first = k % 7 != 4 ? 7 * k - 5 : 0;
    }
}

struct test_one_policy {
    template <typename Policy, typename InputIterator, typename OutputIterator, typename UnaryOp>
    void operator()( Policy&& exec, InputIterator first, InputIterator last, OutputIterator out_first, OutputIterator out_last, UnaryOp op) {
        auto orr = std::transform( exec, first, last, out_first, op );
        EXPECT_TRUE(out_last == orr, "transform returned wrong iterator");
        check_and_reset( first, last, out_first );
    }
};

template <typename Tin, typename Tout>
void test() {
    for (size_t n = 0; n <= 100000; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        Sequence<Tin> in(n, [](int k) {
            return k%5!=1 ? 3*k-7 : 0;
        });

        Sequence<Tout> out( n );

        const auto flip = Complement<Tin,Tout>(1);
        invoke_on_all_policies(test_one_policy(), in.begin(), in.end(), out.begin(), out.end(), flip);
        invoke_on_all_policies(test_one_policy(), in.cbegin(), in.cend(), out.begin(), out.end(), flip);
    }
}

int32_t main( ) {
    test<int32_t, int32_t>();
    test<int32_t, float32_t>();
    test<uint16_t, float32_t>();
    test<float32_t, float64_t>();
    test<float64_t, float64_t>();
    std::cout << "done" << std::endl;
    return 0;
}
