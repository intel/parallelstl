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

// Tests for is_sorted, is_sorted_until

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

struct test_is_sorted {
    template <typename Policy, typename Iterator>
    void operator()(Policy&& exec, Iterator first, Iterator last, bool exam) {
        using namespace std;
        typedef typename std::iterator_traits<Iterator>::value_type T;

        //try random-access iterator
        bool res = is_sorted(exec, first, last);
        EXPECT_TRUE(exam == res, "is_sorted wrong result for random-access iterator");
        auto iexam = is_sorted_until(first, last);
        auto ires = is_sorted_until(exec, first, last);
        EXPECT_TRUE(iexam==ires, "is_sorted_until wrong result for random-access iterator");

        //try random-access iterator with a predicate
        res = is_sorted(exec, first, last, std::less<T>());
        EXPECT_TRUE(exam==res, "is_sorted wrong result for random-access iterator");
        iexam = is_sorted_until(first, last, std::less<T>());
        ires = is_sorted_until(exec, first, last, std::less<T>());
        EXPECT_TRUE(iexam==ires, "is_sorted_until wrong result for random-access iterator");
    }
};

template <typename T>
void test_is_sorted_by_type() {

    Sequence<T> in(99999, [](size_t v)->T { return T(v); }); //fill 0..n

    invoke_on_all_policies(test_is_sorted(), in.begin(), in.end(), std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted(), in.cbegin(), in.cend(), std::is_sorted(in.begin(), in.end()));

    in[in.size()/2] = -1;
    invoke_on_all_policies(test_is_sorted(), in.begin(), in.end(), std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted(), in.cbegin(), in.cend(), std::is_sorted(in.begin(), in.end()));

    in[1] = -1;
    invoke_on_all_policies(test_is_sorted(), in.begin(), in.end(), std::is_sorted(in.begin(), in.end()));
    invoke_on_all_policies(test_is_sorted(), in.cbegin(), in.cend(), std::is_sorted(in.begin(), in.end()));

    //an empty container
    Sequence<T> in0(0);
    invoke_on_all_policies(test_is_sorted(), in0.begin(), in0.end(), std::is_sorted(in0.begin(), in0.end()));
    invoke_on_all_policies(test_is_sorted(), in0.cbegin(), in0.cend(), std::is_sorted(in0.begin(), in0.end()));

    //non-descending order
    Sequence<T> in1(9, [](size_t v)->T { return T(0); });
    invoke_on_all_policies(test_is_sorted(), in1.begin(), in1.end(), std::is_sorted(in1.begin(), in1.end()));
    invoke_on_all_policies(test_is_sorted(), in1.cbegin(), in1.cend(), std::is_sorted(in1.begin(), in1.end()));
}

int32_t main( ) {

    test_is_sorted_by_type<int32_t>();
    test_is_sorted_by_type<float64_t>();

    test_is_sorted_by_type<Wrapper<int32_t>>();

    std::cout << "done" << std::endl;
    return 0;
}
