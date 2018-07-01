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

// Tests for is_heap, is_heap_until
// Tests for copy_if and remove_copy_if
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

struct WithCmpOp {
    int32_t _first;
    int32_t _second;
    WithCmpOp() : _first(0), _second(0) { };
    explicit WithCmpOp(int32_t x) : _first(x), _second(x) { };
    bool operator < (const WithCmpOp& rhs) const {
        return this->_first < rhs._first;
    }
};

struct test_is_heap {
#if __PSTL_ICC_17_VC141_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN || __PSTL_ICC_16_VC14_TEST_SIMD_LAMBDA_DEBUG_32_BROKEN//dummy specialization by policy type, in case of broken configuration
    template <typename Iterator, typename Predicate>
    typename std::enable_if<is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value, void>::type
        operator()(pstl::execution::unsequenced_policy, Iterator first, Iterator last, Predicate pred) { }
    template <typename Iterator, typename Predicate>
    typename std::enable_if<is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value, void>::type
        operator()(pstl::execution::parallel_unsequenced_policy, Iterator first, Iterator last, Predicate pred) { }
#endif

    template <typename Policy, typename Iterator, typename Predicate>
    typename std::enable_if<is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value, void>::type
        operator()(Policy&& exec, Iterator first, Iterator last, Predicate pred) {
        using namespace std;
        // is_heap
        {
            bool expected = is_heap(first, last);
            bool actual = is_heap(exec, first, last);
            EXPECT_TRUE(expected == actual, "wrong return value from is_heap");
        }
        // is_heap with predicate
        {
            bool expected = is_heap(first, last, pred);
            bool actual = is_heap(exec, first, last, pred);
            EXPECT_TRUE(expected == actual, "wrong return value from is_heap with predicate");
        }
        // is_heap_until
        {
            Iterator expected = is_heap_until(first, last);
            Iterator actual = is_heap_until(exec, first, last);
            EXPECT_TRUE(expected == actual, "wrong return value from is_heap_until");
        }
        // is_heap_until with predicate
        {
            const Iterator expected = is_heap_until(first, last, pred);
            const auto y = std::distance(first, expected);
            const Iterator actual = is_heap_until(exec, first, last, pred);
            const auto x = std::distance(first, actual);
            EXPECT_TRUE(expected == actual, "wrong return value from is_heap_until with predicate");
        }
    }

    // is_heap, is_heap_until works only with random access iterators
    template <typename Policy, typename Iterator, typename Predicate>
    typename std::enable_if<!is_same_iterator_category<Iterator, std::random_access_iterator_tag>::value, void>::type
        operator()(Policy&& exec, Iterator first, Iterator last, Predicate pred) { }
};

template <typename T, typename Comp>
void test_is_heap_by_type(Comp comp) {
    using namespace std;

    const size_t max_size = 100000;
    for (size_t n = 0; n <= max_size; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        Sequence<T> in(n, [](size_t v)->T { return T(v); });

        invoke_on_all_policies(test_is_heap(), in.begin(), in.end(), comp);

        std::make_heap(in.begin(), in.begin() + n / 4, comp);
        invoke_on_all_policies(test_is_heap(), in.cbegin(), in.cend(), comp);

        std::make_heap(in.begin(), in.begin() + n / 3, comp);
        invoke_on_all_policies(test_is_heap(), in.begin(), in.end(), comp);

        std::make_heap(in.begin(), in.end(), comp);
        invoke_on_all_policies(test_is_heap(), in.cbegin(), in.cend(), comp);
    }

    Sequence<T> in(max_size/10, [](size_t v)->T { return T(1); });
    invoke_on_all_policies(test_is_heap(), in.begin(), in.end(), comp);
}

int32_t main() {
    test_is_heap_by_type<float32_t>(std::greater<float32_t>());
    test_is_heap_by_type<WithCmpOp>(std::less<WithCmpOp>());
    test_is_heap_by_type<uint64_t>([](uint64_t x, uint64_t y) {return x % 100 < y % 100; });

    std::cout <<done() << std::endl;
    return 0;
}
