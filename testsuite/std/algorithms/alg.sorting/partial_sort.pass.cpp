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

// Tests for partial_sort

// Tests for copy_if and remove_copy_if
#include "support/pstl_test_config.h"

#include <cmath>
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

static std::atomic<int32_t> count_val;
static std::atomic<int32_t> count_comp;

template <typename T>
struct Num {
    T val;

    Num() { ++count_val; }
    Num(T v): val(v) { ++count_val; }
    Num(const Num<T>& v): val(v.val) { ++count_val; }
    Num(Num<T>&& v): val(v.val) { ++count_val; }
    ~Num() { --count_val; }
    Num<T>& operator=(const Num<T>& v) { val = v.val;  return *this; }
    operator T() const { return val; }
    bool operator<(const Num<T>& v) const {
        ++count_comp;
        return val < v.val;
    }
};

struct test_brick_partial_sort {
    template <typename Policy, typename InputIterator, typename Compare>
    typename std::enable_if<is_same_iterator_category<InputIterator, std::random_access_iterator_tag>::value, void>::type
        operator()(Policy&& exec, InputIterator first, InputIterator last, Compare compare) {

        typedef typename std::iterator_traits<InputIterator>::value_type T;
        
        // The rand()%(2*n+1) encourages generation of some duplicates.
        std::srand(42);
        const std::size_t n = last - first;
        for (std::size_t k = 0; k < n; ++k) {
            first[k] = T(rand() % (2 * n + 1));
        }

        for (std::size_t p = 0; p < n; p = p <= 16 ? p + 1 : std::size_t(31.415 * p)) {
            auto m = first + p;

            count_comp = 0;
            std::partial_sort(exec, first, m, last, compare);

            //checking upper bound number of comparisons; O(p*(last-first)log(middle-first)); where p - number of threads;
            if (m - first > 1) {
                auto complex = std::ceil(n * std::log(float32_t(m - first)));
#if __PSTL_USE_PAR_POLICIES
                auto p = tbb::this_task_arena::max_concurrency();
#else
                auto p = 1;
#endif
                EXPECT_TRUE(count_comp < complex*p, "bad complexity");
            }

            EXPECT_TRUE(std::is_sorted(first, m, compare), "bad partial sort: first - middle");

            //10 is a kind of cut_off to avoid sorted (or occasionally sorted) the range [middle, end) at small values of n.
            EXPECT_TRUE(!std::is_sorted(m, last, compare) || last - m < 10, "bad partial sort: middle - end");
        }
    }

    template <typename Policy, typename InputIterator, typename Compare>
    typename std::enable_if<!is_same_iterator_category<InputIterator, std::random_access_iterator_tag>::value, void>::type
        operator()(Policy&& exec, InputIterator first, InputIterator end, Compare compare) {}
};

template<typename T, typename Compare>
void test_partial_sort(Compare compare) {

    const std::size_t n_max = 100000;
    Sequence<T> in(n_max);
    for (std::size_t n = 0; n < n_max; n = n <= 16 ? n + 1 : size_t(3.1415 * n)) {
        invoke_on_all_policies(test_brick_partial_sort(), in.begin(), in.begin() + n, compare);
    }
}

int32_t main() {
    count_val = 0;

    test_partial_sort<Num<float32_t>>([](Num<float32_t> x, Num<float32_t> y) {return x < y; });

    EXPECT_TRUE(count_val == 0, "cleanup error");

    test_partial_sort<int32_t>([](int32_t x, int32_t y) {return x > y; });  // Reversed so accidental use of < will be detected.

    std::cout << done() << std::endl;
    return 0;
}
