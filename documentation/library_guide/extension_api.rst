Extension API
################################
The Extension API currently includes algorithms, iterators, and function object classes. The algorithms
include segmented reduce, segmented scan and vectorized search algorithms. The iterators provided implement
zip, transform, and permutation operations on other iterators, and also include a counting iterator
and a discard iterator. The function object classes provide minimum, maximum and identity operations
that may be passed to algorithms such as reduce or transform. The Extension API also includes an experimental
implementation of range-based algorithms and the ranges required to use them.


Parallel Algorithms
-------------------

The definitions of the algorithms provided in the Extension API are available through the ``oneapi/dpl/algorithm``
header.  All algorithms are implemented in the ``oneapi::dpl`` namespace.

- reduce_by_segment

The ``reduce_by_segment`` algorithm performs partial reductions on a sequence's values and keys. Each
reduction is computed with a given reduction operation for a contiguous subsequence of values, which are
determined by keys being equal according to a predicate. A return value is a pair of iterators holding
the end of the output sequences for keys and values.

For correct computation, the reduction operation should be associative. If no operation is specified,
the default operation for the reduction is ``std::plus``, and the default predicate is ``std::equal_to``.
The algorithm requires that the type of the elements used for values be default constructible.

    Example::

        keys:   [0,0,0,1,1,1]
        values: [1,2,3,4,5,6]
        output_keys:   [0,1]
        output_values: [1+2+3=6,4+5+6=15]

- inclusive_scan_by_segment

The ``inclusive_scan_by_segment`` algorithm performs partial prefix scans on a sequence's values. Each
scan applies to a contiguous subsequence of values, which are determined by the keys associated with the
values being equal. The return value is an iterator targeting the end of the result sequence.

For correct computation, the prefix scan operation should be associative. If no operation is specified,
the default operation is ``std::plus``, and the default predicate is ``std::equal_to``. The algorithm
requires that the type of the elements used for values be default constructible.

    Example::

        keys:   [0,0,0,1,1,1]
        values: [1,2,3,4,5,6]
        result: [1,1+2=3,1+2+3=6,4,4+5=9,4+5+6=15]

- exclusive_scan_by_segment

The ``exclusive_scan_by_segment`` algorithm performs partial prefix scans on a sequence's values. Each
scan applies to a contiguous subsequence of values that are determined by the keys associated with the values
being equal, and sets the first element to the initial value provided. The return value is an iterator
targeting the end of the result sequence.

For correct computation, the prefix scan operation should be associative. If no operation is specified,
the default operation is ``std::plus``, and the default predicate is ``std::equal_to``.

    Example::

        keys:   [0,0,0,1,1,1]
        values: [1,2,3,4,5,6]
        initial value: [0]
        result: [0,0+1=1,0+1+2=3,0,0+4=4,0+4+5=9]

- binary_search

The ``binary_search`` algorithm performs a binary search of the input sequence for each of the values in
the search sequence provided.  For each element of the search sequence the algorithm writes a boolean value
to the result sequence that indicates whether the search value was found in the input sequence. An iterator
to one past the last value in the result sequence is returned. The algorithm assumes the input sequence has
been sorted by the comparator provided. If no comparator is provided then a function object that uses
``operator<`` to compare the elements will be used.

    Example::

        input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
        search sequence: [0, 2, 4, 7, 6]
        result sequence: [true, true, false, false, true]

- lower_bound

The ``lower_bound`` algorithm performs a binary search of the input sequence for each of the values in
the search sequence provided to identify the lowest index in the input sequence where the search value could
be inserted without violating the sorted ordering of the input sequence.  The lowest index for each search
value is written to the result sequence, and the algorithm returns an iterator to one past the last value
written to the result sequence. If no comparator is provided then a function object that uses ``operator<``
to compare the elements will be used.

    Example::

        input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
        search sequence: [0, 2, 4, 7, 6]
        result sequence: [0, 1, 8, 10, 8]


- upper_bound

The ``upper_bound`` algorithm performs a binary search of the input sequence for each of the values in
the search sequence provided to identify the highest index in the input sequence where the search value could
be inserted without violating the sorted ordering of the input sequence.  The highest index for each search
value is written to the result sequence, and the algorithm returns an iterator to one past the last value
written to the result sequence. If no comparator is provided then a function object that uses ``operator<``
to compare the elements will be used.

    Example::

        input sequence:  [0, 2, 2, 2, 3, 3, 3, 3, 6, 6]
        search sequence: [0, 2, 4, 7, 6]
        result sequence: [1, 4, 8, 10, 10]

Iterators
---------

The definitions of the iterators provided in the Extension API are available through the ``oneapi/dpl/iterator``
header.  All iterators are implemented in the ``oneapi::dpl`` namespace.

- counting_iterator

``counting_iterator`` is a random access iterator-like type whose dereferenced value is an integer
counter. Instances of ``counting_iterator`` provide read-only dereference operations. The counter of an
``counting_iterator`` instance changes according to the arithmetics of the random access iterator type.

.. code:: cpp

    using namespace oneapi;
    dpl::counting_iterator<int> count_a(0);
    dpl::counting_iterator<int> count_b = count_a + 10;
    int init = count_a[0]; // OK: init == 0
    *count_b = 7; // ERROR: counting_iterator doesn't provide write operations
    auto sum = std::reduce(dpl::execution::dpcpp_default,
                           count_a, count_b, init); // sum is (0 + 0 + 1 + ... + 9) = 45


- discard_iterator

``discard_iterator`` is a random access iterator-like type that provides write-only dereference
operations that discard values passed.

The iterator is useful in the implementation of stencil algorithms where the stencil is not part of the
desired output. An example of this would be a ``copy_if`` algorithm that receives an an input iterator range
and a stencil iterator range and copies the elements of the input whose corresponding stencil value is 1. We
do not want to declare a temporary allocation to store the copy of the stencil, and thus use ``discard_iterator``.

.. code:: cpp

    using namespace oneapi;
    auto zipped_first = dpl::make_zip_iterator(first, stencil);
    std::copy_if(dpl::execution::dpcpp_default,
                 zipped_first, zipped_first + (last - first),
                 dpl::make_zip_iterator(result, dpl::discard_iterator()),
                 [](auto t){return get<1>(t) == 1;}

- transform_iterator

``transform_iterator`` is an iterator defined over another iterator whose dereferenced value is the result
of a function applied to the corresponding element of the original iterator.  Both the type of the original
iterator and the unary function applied during dereference operations are required template parameters of
the ``transform_iterator`` class. The constructor of the ``transform_iterator`` receives both the original
iterator and an instance of the unary transform operation as well.

To simplify the construction of the iterator ``oneapi::dpl::make_transform_iterator`` is provided. The
function receives the original iterator and transform operation instance as arguments, and constructs the
``transform_iterator`` instance.

.. code:: cpp

    using namespace oneapi;
    dpl::counting_iterator<int> first(0);
    dpl::counting_iterator<int> last(10);
    auto transform_first = dpl::make_transform_iterator(first, std::negate<int>());
    auto transform_last = transform_first + (last - first);
    auto sum = std::reduce(dpl::execution::dpcpp_default,
                           transform_first, transform_last); // sum is (0 + -1 + ... + -9) = -45

- permutation_iterator

``permutation_iterator`` is an iterator whose dereferenced value set is defined by the source iterator
provided, and whose iteration order over the dereferenced value set is defined by either another iterator or
a functor whose index operator defines the mapping from the ``permutation_iterator`` index to the index of the
source iterator. The ``permutation_iterator`` is useful in implementing applications where noncontiguous
elements of data represented by an iterator need to be processed by an algorithm as though they were contiguous.
An example is copying every other element to an output iterator.

``make_permutation_iterator`` is provided to simplify construction of iterator instances.  The function
receives the source iterator and the iterator or function object representing the index map.

.. code:: cpp

    struct multiply_index_by_two {
        template <typename Index>
        Index operator()(const Index& i)
        { return i*2; }
    };

    // first and last are iterators that define a contiguous range of input elements
    // compute the number of elements in the range between the first and last that are accessed
    // by the permutation iterator
    size_t num_elements = std::distance(first, last) / 2 + std::distance(first, last) % 2;
    using namespace oneapi;
    auto permutation_first = dpl::make_permutation_iterator(first, multiply_index_by_two());
    auto permutation_last = permutation_first + num_elements;
    std::copy(dpl::execution::dpcpp_default, permutation_first, permutation_last, result);

- zip_iterator

``zip_iterator`` is an iterator constructed with one or more iterators as input. The value returned by the
iterator when dereferenced is a tuple of the values returned by dereferencing the member iterators on which
the ``zip_iterator`` is defined. Arithmetic operations performed on a ``zip_iterator`` instance are also
applied to each of the member iterators.

The ``make_zip_iterator`` function is provided to simplify the construction of ``zip_iterator`` instances.
The function receives each of the iterators to be held as member iterators by the ``zip_iterator`` instance
it returns.

The example provided for ``discard_iterator`` demonstrates ``zip_iterator`` use in defining stencil
algorithms. The ``zip_iterator`` is also useful in defining "by key" algorithms where input iterators
representing keys and values are processed as key-value pairs. The example below demonstrates a stable sort
by key where only the keys are compared but both keys and values are swapped.

.. code:: cpp

    using namespace oneapi;
    auto zipped_begin = dpl::make_zip_iterator(keys_begin, vals_begin);
    std::stable_sort(dpl::execution::dpcpp_default, zipped_begin, zipped_begin + n,
        [](auto lhs, auto rhs) { return get<0>(lhs) < get<0>(rhs); });


Function Object Classes
--------------------------

The definitions of the function objects provided in the Extension API are available through the
``oneapi/dpl/functional`` header.  All function objects are implemented in the ``oneapi::dpl`` namespace.

- identity: A C++11 implementation of the C++20 ``std::identity`` function object type, where the operator() returns the
  argument unchanged.

- minimum: A function object type where the operator() applies ``std::less`` to its arguments, then returns the
  lesser argument unchanged.

- maximum: A function object type where the operator() applies ``std::greater`` to its arguments, then returns the
  greater argument unchanged.

Range-based API
--------------------------

C++20 indroduces the *Ranges* library. С++20 standard splits ranges into two categories: *factories* and *adaptors*.
A range factory doesn't have underlying data. An element is generated on success by an index or by dereferencing an iterator.
A range adaptor, from the DPC++ library perspective, is an utility that transforms *base range*, or another adapted range into 
a view with custom behavior.

The DPC++ library supports ``iota_view`` range factory.

``sycl::buffer`` wrapped with ``all_view`` can be used as the range.

The DPC++ library considers the supported factories and ``all_view`` as base ranges.
The range adaptors may be combined into a pipeline with a ``base`` range at the beginning. For example:

.. code:: cpp

    cl::sycl::buffer<int> buf(data, cl::sycl::range<1>(10));
    auto range_1 = iota_view(0, 10) | views::reverse();
    auto range_2 = all_view(buf) | views::reverse();

For the range, based on the ``all_view`` factory, data access is permitted on a device only. ``size()`` and ``empty()`` methods are allowed 
to be called on both host and device.

The following algorithms are available to use with the ranges:

``for_each``, ``copy``, ``transform``, ``find``, ``find_if``, ``find_if_not``, ``find_end``, ``find_first_of``, ``search``, ``is_sorted``,
``is_sorted_until``, ``reduce``, ``transform_reduce``, ``min_element``, ``max_element``, ``minmax_element``,
``exclusive_scan``, ``inclusive_scan``, ``transform_exclusive_scan``, ``transform_inclusive_scan``.

The signature example of the range-based algorithms looks like:

.. code:: cpp

  template <typename ExecutionPolicy, typename Range1, typename Range2>
  void copy(ExecutionPolicy&& exec, Range1&& source, Range2&& destination);

where ``source`` is used instead of two iterators to represent the input. ``destination`` represents the output.

These algorithms are declared in ``oneapi::dpl::experimental::ranges`` namespace and implemented only for DPC++ policies.
In order to make these algorithm available the ``<oneapi/dpl/ranges>`` header should be included.
Use of the range-based API requires C++17 and the C++ standard libraries coming with GCC 8.1 (or higher) or Clang 7 (or higher).

The following viewable ranges are declared in ``oneapi::dpl::experimental::ranges`` namespace. Only those are allowed to use as ranges for range-based algorithms.

* ``iota_view``. A range factory - generates a sequence of N elements which starts from an initial value and ends by final N-1.
* ``all_view``. A custom utility - represents a view of all or a part of ``sycl::buffer`` underlying elements.
* ``guard_view``. A custom utility - represents a view of USM data range defined by a two USM pointers.
* ``zip_view``. A custom range adapter - produces one ``zip_view`` from other several views.
* ``transform_view``. A range adapter - represents a view of a underlying sequence after applying a transformation to each element.
* ``reverse_view``. A range adapter - produces a reversed sequence of elements provided by another view.
* ``take_view``. A range adapter - produces a view of the first N elements from another view.
* ``drop_view``. A range adapter - produces a view excluding the first N elements from another view.

Example of Range-based API usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: cpp

    using namespace oneapi::dpl::experimental::ranges;

    {
        cl::sycl::buffer<int> A(data, cl::sycl::range<1>(max_n));
        cl::sycl::buffer<int> B(data2, cl::sycl::range<1>(max_n));

        auto view = all_view(A) | views::reverse();
        auto range_res = all_view<int, cl::sycl::access::mode::write>(B);

        copy(oneapi::dpl::execution::dpcpp_default, view, range_res);
    }
