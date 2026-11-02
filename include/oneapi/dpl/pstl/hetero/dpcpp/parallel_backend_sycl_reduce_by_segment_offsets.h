// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
#define _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <type_traits>
#include <utility>

#include "sycl_defs.h"
#include "parallel_backend_sycl.h"
#include "unseq_backend_sycl.h"
#include "utils_ranges_sycl.h"

#include "../../functional_impl.h" // oneapi::dpl::maximum
#include "../../segmented_reduce_utils.h"
#include "../../utils.h"

namespace oneapi
{
namespace dpl
{
namespace __par_backend_hetero
{

// reduce_by_segment with segments defined by offsets

// One work-item per short segment. A longer segment is left to the reduction in chunks and only reports its length,
// capped as only its square root matters, through an atomic maximum.
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_variable_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __n;         // the size of the input
    _Size __threshold; // the length of the longest short segment
    _InitType __init;
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _Offsets, typename _Result, typename _Input,
              typename _MaxLength>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Offsets __offsets, _Result __result, _Input __input,
               _MaxLength __max_length) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");

        const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__idx], __n);
        const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__idx + 1], __n);
        if (__end - __begin > __threshold)
        {
            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                __ref(__max_length[0]);
            __ref.fetch_max(static_cast<std::uint32_t>(std::min<std::uint64_t>(
                static_cast<std::uint64_t>(__end - __begin), std::numeric_limits<std::uint32_t>::max())));
            return;
        }
        std::get<0>(__result[__idx]) =
            oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(__input, __begin, __end, __init, __binary_op);
    }
};

// Writes the initial value into every result: all the segments are empty
template <typename _Acc, typename _InitType>
struct __segmented_reduce_init_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _InitType __init;

    // __result is a zip view of the result only, so that the segments are not vectorized over
    template <typename _IsFull, typename _Params, typename _Result>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _Result __result) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");
        std::get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init);
    }
};

template <typename _Name>
class __segmented_reduce_fill_wrapper;

// A short segment is reduced by one work-item. A long segment is split into chunks, each reduced by one work-item, and
// the partial results of the segment are combined by one work-item. The values below were measured on GPUs and CPUs:
// - a segment up to this length is short;
inline constexpr std::size_t __segmented_reduce_gpu_threshold = 512;
inline constexpr std::size_t __segmented_reduce_cpu_threshold = 32768;
// - on GPUs the best chunk length is close to the square root of the longest segment, which balances the serial work
//   of the two levels; on CPUs the chunk length hardly matters above 1024.
inline constexpr std::size_t __segmented_reduce_cpu_chunk_size = 4096;

inline std::size_t
__segmented_reduce_chunk_size(bool __is_gpu, std::size_t __max_length)
{
    if (__is_gpu)
    {
        const auto __root = static_cast<std::size_t>(std::sqrt(static_cast<double>(__max_length)));
        return std::clamp<std::size_t>(oneapi::dpl::__internal::__dpl_bit_ceil(__root), 64, 16384);
    }
    return __segmented_reduce_cpu_chunk_size;
}

// The length of a segment within the input; a negatively sized segment is empty
template <typename _Size>
struct __segmented_reduce_length
{
    _Size __n; // the size of the input

    template <typename _OffsetEnd, typename _OffsetBegin>
    _Size
    operator()(_OffsetEnd __end, _OffsetBegin __begin) const
    {
        const _Size __b = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__begin, __n);
        const _Size __e = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__end, __n);
        return __b < __e ? __e - __b : _Size{0};
    }
};

// The number of chunks of a segment given by the pair of its offsets
template <typename _Size>
struct __segmented_reduce_chunk_count
{
    _Size __n;          // the size of the input
    _Size __chunk_size; // the length of a chunk
    _Size __min_length; // the segments not longer than this have no chunks

    template <typename _Bounds>
    _Size
    operator()(const _Bounds& __bounds) const
    {
        const _Size __length = __segmented_reduce_length<_Size>{__n}(std::get<1>(__bounds), std::get<0>(__bounds));
        return __length > __min_length ? __length / __chunk_size + (__length % __chunk_size != 0) : _Size{0};
    }
};

// Reduces one chunk of a long segment. The first chunk of a segment takes the remainder of its length and starts
// from the operation on the initial value and its first element; every other chunk has exactly __chunk_size elements
// and starts from the operation on its first two elements.
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_chunk_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __n;            // the size of the input
    _Size __num_segments; // the number of segments
    _Size __chunk_size;   // the length of a chunk
    _InitType __init;
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Offsets, typename _Partials,
              typename _Input>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Offsets __offsets,
               _Partials __partials, _Input __input) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");

        const _Size __chunk = __idx;
        if (__chunk >= __chunk_ends[__num_segments - 1])
            return;

        // The segment of the chunk: the first one whose chunks end after it
        _Size __lo = 0, __hi = __num_segments - 1;
        while (__lo < __hi)
        {
            const _Size __mid = __lo + (__hi - __lo) / 2;
            if (__chunk_ends[__mid] > __chunk)
                __hi = __mid;
            else
                __lo = __mid + 1;
        }
        const _Size __first_chunk = __lo == 0 ? _Size{0} : _Size(__chunk_ends[__lo - 1]);
        const _Size __begin = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__lo], __n);
        const _Size __end = oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__lo + 1], __n);
        const _Size __head = (__end - __begin) - (_Size(__chunk_ends[__lo]) - __first_chunk - 1) * __chunk_size;

        // One loop for every chunk, so that the work-items of a sub-group do not diverge into two long loops
        const _Size __k = __chunk - __first_chunk;
        const _Size __chunk_begin = __k == 0 ? __begin : __begin + __head + (__k - 1) * __chunk_size;
        const _Size __chunk_end = __k == 0 ? __begin + __head : __chunk_begin + __chunk_size;
        // A chunk other than the first one starts from the operation on its first two elements, as a partial result
        // of std::reduce does: the accumulator is never made from an element alone, which may mean something else than
        // the element to the operation (such a chunk has at least 64 elements)
        _Acc __start = __k == 0 ? oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                      std::invoke(__binary_op, __init, __input[__chunk_begin]))
                                : oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(
                                      std::invoke(__binary_op, __input[__chunk_begin], __input[__chunk_begin + 1]));
        std::get<0>(__partials[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
            __input, __chunk_begin + (__k == 0 ? 1 : 2), __chunk_end, std::move(__start), __binary_op);
    }
};

// Combines the partial results of the chunks of every segment
template <typename _Acc, typename _Size, typename _InitType, typename _BinaryOperation>
struct __segmented_reduce_combine_brick
{
    static constexpr std::uint8_t __iters_per_item = 1;

    _Size __num_partials; // the number of the computed partial results
    bool __write_empty;   // whether a segment without chunks gets the initial value; otherwise it is left as is
    _InitType __init;
    mutable _BinaryOperation __binary_op;

    template <typename _IsFull, typename _Params, typename _ChunkEnds, typename _Partials, typename _Result>
    void
    operator()(_IsFull, const std::size_t __idx, _Params, _ChunkEnds __chunk_ends, _Partials __partials,
               _Result __result) const
    {
        static_assert(_Params::__vector_size == 1,
                      "The brick operates with tuples which must be excluded from vectorizable types");

        const _Size __first = __idx == 0 ? _Size{0} : _Size(__chunk_ends[__idx - 1]);
        // A segment without chunks is empty, or short and already reduced. Unordered offsets may give more chunks than
        // computed: their results are unspecified.
        const _Size __last = std::min<_Size>(__chunk_ends[__idx], __num_partials);
        if (__first >= __last)
        {
            if (__write_empty)
                std::get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init);
            return;
        }
        std::get<0>(__result[__idx]) = oneapi::dpl::__internal::__segmented_reduce_serial<_Acc>(
            __partials, __first + 1, __last, _Acc(__partials[__first]), __binary_op);
    }
};

template <typename _Name>
class __segmented_reduce_length_wrapper;
template <typename _Name>
class __segmented_reduce_scan_wrapper;
template <typename _Name>
class __segmented_reduce_chunk_wrapper;
template <typename _Name>
class __segmented_reduce_combine_wrapper;

// The two kernels of the note "Segmented Reduction: Algorithmic Design Summary", on GPUs for arithmetic types; no
// work-group waits for another. The first one reduces tiles of the input into labeled partial sums, pairs of a value
// and a segment, at least one for every segment, the initial value taken by the first tile of the segment. The second
// one reduces the partial sums of the same segment: it writes the result of every segment whose partial sums one of its
// tiles holds entirely and, for its next level, the partial sums of the segments that cross the bounds of its tiles. A
// slot with no partial sum is labeled with no segment.
struct __segmented_reduce_tiles_tuning
{
    static constexpr std::uint32_t __lanes = 32;
    static constexpr std::uint32_t __work_group = 256;
    static constexpr std::uint32_t __items_per_work_item = 16;
    static constexpr std::uint32_t __tile = __work_group * __items_per_work_item;
};

template <typename... _Name>
class __segmented_reduce_tiles_kernel;
template <typename... _Name>
class __segmented_reduce_sums_kernel;

// The state of a scan over the work-items of a tile: the last run of the work-items so far, whether it starts after
// the first of them (__resets) and whether it continues from the previous tile (__at_start)
template <typename _Acc, typename _Key>
struct __segmented_reduce_run_state
{
    _Key __key;
    _Acc __value;
    std::uint8_t __has;
    std::uint8_t __resets;
    std::uint8_t __at_start;
};

// The local memory of both kernels: a tile of values and of their segments, padded by one slot per work-item's part
// so that the parts of the work-items of a sub-group are in different banks; the bounds of the segments of a tile
// relative to it; the parts of the work-items and the scan over them
template <typename _Acc, typename _Key, typename _Size>
struct __segmented_reduce_tiles_local
{
    using _State = __segmented_reduce_run_state<_Acc, _Key>;

    __dpl_sycl::__local_accessor<_Acc> __values;
    __dpl_sycl::__local_accessor<_Key> __keys;
    __dpl_sycl::__local_accessor<std::uint32_t> __bounds;
    __dpl_sycl::__local_accessor<_Key> __first_keys;
    __dpl_sycl::__local_accessor<_Acc> __first_values;
    __dpl_sycl::__local_accessor<std::uint8_t> __single;
    __dpl_sycl::__local_accessor<std::int32_t> __next;
    __dpl_sycl::__local_accessor<_State> __states;
    __dpl_sycl::__local_accessor<_Size> __info;

    __segmented_reduce_tiles_local(sycl::handler& __cgh, std::size_t __padded_tile, std::size_t __bounds_size,
                                   std::size_t __work_group)
        : __values(sycl::range<1>(__padded_tile), __cgh), __keys(sycl::range<1>(__padded_tile), __cgh),
          __bounds(sycl::range<1>(__bounds_size), __cgh), __first_keys(sycl::range<1>(__work_group), __cgh),
          __first_values(sycl::range<1>(__work_group), __cgh), __single(sycl::range<1>(__work_group), __cgh),
          __next(sycl::range<1>(__work_group), __cgh), __states(sycl::range<1>(2 * __work_group), __cgh),
          __info(sycl::range<1>(3), __cgh)
    {
    }
};

template <typename _Acc, typename _Key, typename _TilesName, typename _SumsName, typename _InitType,
          typename _BinaryOperation>
struct __segmented_reduce_tiles_submitter;

template <typename _Acc, typename _Key, typename... _TilesName, typename... _SumsName, typename _InitType,
          typename _BinaryOperation>
struct __segmented_reduce_tiles_submitter<_Acc, _Key, __internal::__optional_kernel_name<_TilesName...>,
                                          __internal::__optional_kernel_name<_SumsName...>, _InitType, _BinaryOperation>
{
    static constexpr std::uint32_t __lanes = __segmented_reduce_tiles_tuning::__lanes;
    static constexpr std::uint32_t __work_group = __segmented_reduce_tiles_tuning::__work_group;
    static constexpr std::uint32_t __items = __segmented_reduce_tiles_tuning::__items_per_work_item;
    static constexpr std::uint32_t __tile = __segmented_reduce_tiles_tuning::__tile;
    static constexpr std::uint32_t __padded_tile = __tile + __tile / __items;
    static constexpr _Key __no_segment = std::numeric_limits<_Key>::max();

    using _State = __segmented_reduce_run_state<_Acc, _Key>;

    _InitType __init;
    mutable _BinaryOperation __binary_op;

    template <typename _InRange, typename _OffsetRange, typename _OutRange, typename _Size>
    void
    operator()(sycl::queue& __q, _InRange& __input, _OffsetRange& __offsets, _OutRange& __result, const _Size __n,
               const _Size __m) const
    {
        using _Local = __segmented_reduce_tiles_local<_Acc, _Key, _Size>;

        const std::size_t __num_tiles = oneapi::dpl::__internal::__dpl_ceiling_div(std::size_t(__n), __tile);
        // A slot for every work-group of the first kernel, at most ceil(n / tile) + m of them
        const std::size_t __num_slots = __num_tiles + std::size_t(__m);
        const std::size_t __first_level_tiles = oneapi::dpl::__internal::__dpl_ceiling_div(__num_slots, __tile);
        // The next levels: two slots for every tile of a level, the head and the tail of the tile
        const std::size_t __next_slots = __first_level_tiles > 1 ? 2 * __first_level_tiles : 1;

        __device_storage<_Acc> __sums{__q, __num_slots};
        __device_storage<_Key> __keys{__q, __num_slots};
        __device_storage<_Acc> __next_sums{__q, __next_slots};
        __device_storage<_Key> __next_keys{__q, __next_slots};

        sycl::event __event = __q.submit([&](sycl::handler& __cgh) {
            oneapi::dpl::__ranges::__require_access(__cgh, __input, __offsets);
            auto __sums_acc = __get_accessor(sycl::write_only, __sums, __cgh, __dpl_sycl::__no_init{});
            auto __keys_acc = __get_accessor(sycl::write_only, __keys, __cgh, __dpl_sycl::__no_init{});
            // Only the reduction over the work-group
            const _Local __local(__cgh, __work_group, 1, __work_group);

            __cgh.parallel_for<_TilesName...>(
                sycl::nd_range<1>(sycl::range<1>(__num_slots * __work_group), sycl::range<1>(__work_group)),
                [=, *this](sycl::nd_item<1> __item) [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(__lanes)]] {
                    __reduce_slice(__item, __input, __offsets, __n, __m, __local, __sums_acc.__data(),
                                   __keys_acc.__data());
                });
        });

        __device_storage<_Acc>* __in_sums = &__sums;
        __device_storage<_Key>* __in_keys = &__keys;
        __device_storage<_Acc>* __out_sums = &__next_sums;
        __device_storage<_Key>* __out_keys = &__next_keys;
        std::size_t __count = __num_slots;
        for (;;)
        {
            const std::size_t __sum_tiles = oneapi::dpl::__internal::__dpl_ceiling_div(__count, __tile);
            const bool __last_level = __sum_tiles == 1;

            __event = __q.submit([&](sycl::handler& __cgh) {
                oneapi::dpl::__ranges::__require_access(__cgh, __result);
                auto __in_sums_acc = __get_accessor(sycl::read_only, *__in_sums, __cgh);
                auto __in_keys_acc = __get_accessor(sycl::read_only, *__in_keys, __cgh);
                auto __out_sums_acc = __get_accessor(sycl::read_write, *__out_sums, __cgh);
                auto __out_keys_acc = __get_accessor(sycl::read_write, *__out_keys, __cgh);
                const _Local __local(__cgh, __padded_tile, 1, __work_group);

                __cgh.depends_on(__event);

                __cgh.parallel_for<_SumsName...>(
                    sycl::nd_range<1>(sycl::range<1>(__sum_tiles * __work_group), sycl::range<1>(__work_group)),
                    [=, *this](sycl::nd_item<1> __item) [[_ONEDPL_SYCL_REQD_SUB_GROUP_SIZE(__lanes)]] {
                        __reduce_sums(__item, __result, _Key(__m), __count, __last_level, __local,
                                      __in_sums_acc.__data(), __in_keys_acc.__data(), __out_sums_acc.__data(),
                                      __out_keys_acc.__data());
                    });
            });

            if (__last_level)
                break;
            std::swap(__in_sums, __out_sums);
            std::swap(__in_keys, __out_keys);
            __count = 2 * __sum_tiles;
        }

        // Waited here, as the temporary data above is released on return
        __event.wait_and_throw();
    }

  private:
    // The slot of the local memory of the value __j of a tile
    static std::uint32_t
    __pad(const std::uint32_t __j)
    {
        return __j + __j / __items;
    }

    // The offset __k clipped to the input
    template <typename _OffsetRange, typename _Size>
    static _Size
    __bound(const _OffsetRange& __offsets, const _Size __n, const _Size __k)
    {
        return oneapi::dpl::__internal::__segmented_reduce_clamp_offset(__offsets[__k], __n);
    }

    // The first index of [__lo, __hi) where __pred is false, __hi if none, for a __pred true and then false: a search
    // by __lanes probes at once
    template <typename _SubGroup, typename _Size, typename _Pred>
    static _Size
    __search(const _SubGroup& __sg, _Size __lo, _Size __hi, _Pred __pred)
    {
        const std::uint32_t __lane = __sg.get_local_linear_id();
        while (__lo < __hi)
        {
            const _Size __step = (__hi - __lo + _Size(__lanes) - 1) / _Size(__lanes);
            const _Size __probe = __lo + _Size(__lane) * __step;
            const std::uint32_t __true = __dpl_sycl::__reduce_over_group(
                __sg, std::uint32_t(__probe < __hi && __pred(__probe)), sycl::plus<std::uint32_t>());
            if (__true == 0)
                return __lo;
            __hi = std::min(__hi, __lo + _Size(__true) * __step);
            __lo = __lo + _Size(__true - 1) * __step + 1;
        }
        return __lo;
    }

    // The reduction of the values of the first __count work-items, in work-item 0: over every sub-group, then over the
    // sub-groups
    _Acc
    __reduce_group(const sycl::nd_item<1> __item, const __dpl_sycl::__local_accessor<_Acc>& __values,
                   const oneapi::dpl::__internal::__lazy_ctor_storage<_Acc>& __value, const std::uint32_t __count) const
    {
        const auto __sg = __item.get_sub_group();
        const std::uint32_t __lane = __sg.get_local_linear_id();
        const std::uint32_t __sg_id = __sg.get_group_linear_id();
        const std::uint32_t __lid = __item.get_local_linear_id();

        auto __reduce_lanes = [&](_Acc __v, const std::uint32_t __num_valid) {
            for (std::uint32_t __s = __lanes / 2; __s > 0; __s /= 2)
            {
                const _Acc __other = sycl::shift_group_left(__sg, __v, __s);
                __v = __lane + __s < __num_valid ? std::invoke(__binary_op, __v, __other) : __v;
            }
            return __v;
        };

        const std::uint32_t __sg_begin = __sg_id * __lanes;
        const std::uint32_t __sg_valid = __count > __sg_begin ? std::min(__count - __sg_begin, __lanes) : 0;
        // The lanes past the values take the value of the first lane, which never takes part
        const _Acc __first = sycl::select_from_group(__sg, __value.__v, 0);
        const _Acc __v = __reduce_lanes(__lid < __count ? __value.__v : __first, __sg_valid);
        if (__lane == 0 && __sg_valid > 0)
            __values[__sg_id] = __v;
        sycl::group_barrier(__item.get_group());

        const std::uint32_t __groups = oneapi::dpl::__internal::__dpl_ceiling_div(__count, __lanes);
        _Acc __w = __values[__lane < __groups ? __lane : 0];
        if (__sg_id == 0)
            __w = __reduce_lanes(__w, __groups);
        return __w;
    }

    // Reduces a tile of labeled values in the local memory, __len of them, the values labeled with no segment or with
    // __num_keys or more skipped: unordered offsets may leave slots unwritten, which no result may be written from. A
    // work-item takes __items consecutive values and passes the runs of one segment inside them to __emit; the runs
    // crossing work-items are joined by a scan over the work-items and passed by the work-item where they end. Every
    // run is passed with whether it continues from the previous tile, labeled __prev_key at its end, and into the next
    // one, labeled __next_key at its start.
    template <typename _Size, typename _Emit>
    void
    __reduce_keyed_tile(const sycl::nd_item<1> __item, const __segmented_reduce_tiles_local<_Acc, _Key, _Size>& __local,
                        const std::uint32_t __len, const _Key __num_keys, const _Key __prev_key, const _Key __next_key,
                        _Emit __emit) const
    {
        const auto __group = __item.get_group();
        const std::uint32_t __lid = __item.get_local_linear_id();
        const std::uint32_t __from = std::min(__len, __lid * __items);
        const std::uint32_t __to = std::min(__len, __from + __items);

        _Key __first_key = __no_segment;
        _Key __key = __no_segment;
        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __first_value, __value;
        std::uint32_t __runs = 0;
        for (std::uint32_t __j = __from; __j < __to; ++__j)
        {
            const _Key __k = __local.__keys[__pad(__j)];
            if (__k >= __num_keys)
                continue;
            if (__k == __key)
            {
                __value.__v = std::invoke(__binary_op, __value.__v, __local.__values[__pad(__j)]);
                continue;
            }
            // A run ends: the first one of the work-item, or one inside it
            if (__runs == 1)
            {
                __first_key = __key;
                __first_value.__setup(__value.__v);
            }
            if (__runs > 1)
                __emit(__key, __value.__v, false, false);
            ++__runs;
            __key = __k;
            __value.__setup(__local.__values[__pad(__j)]);
        }
        const bool __has_values = __runs > 0;
        const bool __single = __runs == 1;
        if (__single)
        {
            __first_key = __key;
            __first_value.__setup(__value.__v);
        }

        // The last work-item before this one with values; the first run of the first one may continue the previous
        // tile
        const std::int32_t __prev =
            __dpl_sycl::__exclusive_scan_over_group(__group, __has_values ? std::int32_t(__lid) : std::int32_t(-1),
                                                    std::int32_t(-1), sycl::maximum<std::int32_t>());
        const bool __first_at_start = __prev < 0 && __prev_key == __first_key;

        __local.__first_keys[__lid] = __first_key;
        __local.__single[__lid] = __single;
        if (__has_values)
            __local.__first_values[__lid] = __first_value.__v;
        // The last run of the work-item, as the element of the scan
        _State __state{__key, __value.__v, std::uint8_t(__has_values), std::uint8_t(!__single),
                       std::uint8_t(__single && __first_at_start)};
        __local.__states[__lid] = __state;
        sycl::group_barrier(__group);

        // The next work-item with values, by a scan over the work-items in the reverse order
        const std::uint32_t __rev = __work_group - 1 - __lid;
        const std::int32_t __next = __dpl_sycl::__exclusive_scan_over_group(
            __group, __local.__first_keys[__rev] != __no_segment ? std::int32_t(__rev) : std::int32_t(__work_group),
            std::int32_t(__work_group), sycl::minimum<std::int32_t>());
        __local.__next[__rev] = __next;

        // An inclusive scan of the last runs over the work-items, two buffers of the states
        auto __combine = [&](const _State& __a, const _State& __b) {
            if (!__b.__has)
                return __a;
            if (!__a.__has || __b.__resets)
                return __b;
            if (__a.__key == __b.__key)
                return _State{__b.__key, std::invoke(__binary_op, __a.__value, __b.__value), 1, __a.__resets,
                              __a.__at_start};
            return _State{__b.__key, __b.__value, 1, 1, 0};
        };
        std::uint32_t __buffer_in = 0;
        for (std::uint32_t __s = 1; __s < __work_group; __s *= 2)
        {
            if (__lid >= __s)
                __state = __combine(__local.__states[__buffer_in * __work_group + __lid - __s], __state);
            __buffer_in = 1 - __buffer_in;
            __local.__states[__buffer_in * __work_group + __lid] = __state;
            sycl::group_barrier(__group);
        }

        if (!__has_values)
            return;

        const std::int32_t __nx = __local.__next[__lid];
        const bool __at_tile_end = __nx == std::int32_t(__work_group);

        // The first run with the runs before it in the tile
        _Acc __first_total = __first_value.__v;
        bool __first_from_start = __first_at_start;
        if (__lid > 0)
        {
            const _State __before = __local.__states[__buffer_in * __work_group + __lid - 1];
            if (__before.__has && __before.__key == __first_key)
            {
                __first_total = std::invoke(__binary_op, __before.__value, __first_value.__v);
                __first_from_start = __before.__at_start;
            }
        }
        const bool __first_goes_on = !__at_tile_end && __local.__first_keys[__nx] == __first_key;
        if (!__single)
            __emit(__first_key, __first_total, __first_from_start, false);
        if (__single && !__first_goes_on)
            __emit(__first_key, __first_total, __first_from_start, __at_tile_end && __next_key == __first_key);

        // The last run, starting in this work-item
        const bool __last_goes_on = !__at_tile_end && __local.__first_keys[__nx] == __key;
        if (!__single && !__last_goes_on)
            __emit(__key, __value.__v, false, __at_tile_end && __next_key == __key);
    }

    // The number of work-groups of the segments before the segment __k, ceil((begin(__k) - start) / __tile) + __k:
    // a work-group never takes two segments, so a segment of the length l takes at most ceil(l / __tile) + 1 of them
    template <typename _OffsetRange, typename _Size>
    static _Size
    __groups_before(const _OffsetRange& __offsets, const _Size __n, const _Size __start, const _Size __k)
    {
        const _Size __from_start = std::max(_Size{0}, __bound(__offsets, __n, __k) - __start);
        return (__from_start + _Size(__tile) - 1) / _Size(__tile) + __k;
    }

    // The first kernel, variant B of the note: the work-group __wg takes a slice of one segment, found by a search of
    // its first sub-group over the numbers of the work-groups before the segments, and writes its partial sum to the
    // slot __wg, with the initial value in the first slice of the segment, which an empty segment also has; a
    // work-group with no slice labels its slot with no segment.
    template <typename _InRange, typename _OffsetRange, typename _Size>
    void
    __reduce_slice(const sycl::nd_item<1> __item, const _InRange& __input, const _OffsetRange& __offsets,
                   const _Size __n, const _Size __m, const __segmented_reduce_tiles_local<_Acc, _Key, _Size>& __local,
                   _Acc* const __sums, _Key* const __keys) const
    {
        const auto __sg = __item.get_sub_group();
        const std::uint32_t __lid = __item.get_local_linear_id();
        const std::size_t __wg = __item.get_group_linear_id();
        const _Size __group = _Size(__wg);
        const _Size __start = __bound(__offsets, __n, _Size{0});

        // The first segment with more work-groups before it than __group, less one
        if (__sg.get_group_linear_id() == 0)
        {
            const _Size __lo = __search(__sg, _Size{0}, __m + 1, [&](const _Size __j) {
                return __groups_before(__offsets, __n, __start, __j) <= __group;
            });
            if (__sg.get_local_linear_id() == 0)
                __local.__info[0] = __lo - 1;
        }
        sycl::group_barrier(__item.get_group());
        const _Size __k = __local.__info[0];

        // Unordered offsets may give no segment or a negative position in it
        const _Size __local_group =
            __k >= 0 && __k < __m ? __group - __groups_before(__offsets, __n, __start, __k) : -1;
        if (__local_group < 0)
        {
            if (__lid == 0)
                __keys[__wg] = __no_segment;
            return;
        }

        const _Size __segment_end = __bound(__offsets, __n, __k + 1);
        const _Size __begin = __bound(__offsets, __n, __k) + __local_group * _Size(__tile);
        const _Size __end = std::min(__segment_end, __begin + _Size(__tile));
        if (__begin >= __end)
        {
            if (__lid == 0 && __local_group == 0)
            {
                __sums[__wg] = oneapi::dpl::__internal::__segmented_reduce_to_acc<_Acc>(__init);
                __keys[__wg] = _Key(__k);
            }
            if (__lid == 0 && __local_group > 0)
                __keys[__wg] = __no_segment;
            return;
        }

        const std::uint32_t __count = std::uint32_t(std::min<_Size>(__end - __begin, _Size(__work_group)));
        oneapi::dpl::__internal::__lazy_ctor_storage<_Acc> __acc;
        _Size __i = __begin + _Size(__lid);
        if (__i < __end)
        {
            __acc.__setup(__input[__i]);
            for (__i += _Size(__work_group); __i < __end; __i += _Size(__work_group))
                __acc.__v = std::invoke(__binary_op, __acc.__v, __input[__i]);
        }
        // A tree in the local memory: the reduction by shifts within the sub-groups (__reduce_group) faults on PVC
        // here, unlike in the first kernel of variant A, without a found cause
        if (__lid < __count)
            __local.__values[__lid] = __acc.__v;
        for (std::uint32_t __s = __work_group / 2; __s > 0; __s /= 2)
        {
            sycl::group_barrier(__item.get_group());
            if (__lid < __s && __lid + __s < __count)
                __local.__values[__lid] =
                    std::invoke(__binary_op, __local.__values[__lid], __local.__values[__lid + __s]);
        }
        const _Acc __sum = __local.__values[0];
        if (__lid == 0)
        {
            __sums[__wg] = __local_group == 0 ? std::invoke(__binary_op, __init, __sum) : __sum;
            __keys[__wg] = _Key(__k);
        }
    }

    // The second kernel: the tile __wg of the __count labeled partial sums. The result of a segment whose partial sums
    // the tile holds entirely is written; at the next level, a segment continuing from the previous tile goes to the
    // head slot of the tile, 2 * __wg, one continuing into the next tile to its tail slot, 2 * __wg + 1, and one doing
    // both to the head slot. An unlabeled tail slot may thus be inside a segment: then the head slot before it
    // continues the segment.
    template <typename _OutRange, typename _Size>
    void
    __reduce_sums(const sycl::nd_item<1> __item, const _OutRange& __result, const _Key __num_keys,
                  const std::size_t __count, const bool __last_level,
                  const __segmented_reduce_tiles_local<_Acc, _Key, _Size>& __local, const _Acc* const __sums,
                  const _Key* const __keys, _Acc* const __next_sums, _Key* const __next_keys) const
    {
        const std::uint32_t __lid = __item.get_local_linear_id();
        const std::size_t __wg = __item.get_group_linear_id();
        const std::size_t __tile_begin = __wg * __tile;
        const std::uint32_t __len = std::uint32_t(std::min<std::size_t>(__count - __tile_begin, __tile));

        for (std::uint32_t __j = __lid; __j < __len; __j += __work_group)
        {
            __local.__keys[__pad(__j)] = __keys[__tile_begin + __j];
            __local.__values[__pad(__j)] = __sums[__tile_begin + __j];
        }
        if (__lid == 0)
        {
            __local.__info[0] = 0;
            __local.__info[1] = 0;
        }

        _Key __prev_key = __no_segment;
        if (__tile_begin >= 1)
            __prev_key = __keys[__tile_begin - 1];
        if (__prev_key == __no_segment && __tile_begin >= 2)
            __prev_key = __keys[__tile_begin - 2];
        const std::size_t __tile_end = __tile_begin + __len;
        const _Key __next_key = __tile_end < __count ? __keys[__tile_end] : __no_segment;
        sycl::group_barrier(__item.get_group());

        auto __emit = [&](const _Key __k, const _Acc __acc, const bool __at_start, const bool __at_end) {
            if (!__at_start && !__at_end)
            {
                __result[__k] = __acc;
                return;
            }
            const std::size_t __slot = 2 * __wg + (__at_start ? 0 : 1);
            __next_sums[__slot] = __acc;
            __next_keys[__slot] = __k;
            __local.__info[__at_start ? 0 : 1] = 1;
        };
        __reduce_keyed_tile(__item, __local, __len, __num_keys, __prev_key, __next_key, __emit);

        sycl::group_barrier(__item.get_group());
        if (__lid == 0 && !__last_level && __local.__info[0] == 0)
            __next_keys[2 * __wg] = __no_segment;
        if (__lid == 0 && !__last_level && __local.__info[1] == 0)
            __next_keys[2 * __wg + 1] = __no_segment;
    }
};

// Writes the initial value into every result: every segment is empty, because the input is or because the offsets
// give no element to any segment
template <typename _Acc, typename _BackendTag, typename _ExecutionPolicy, typename _Range, typename _Size,
          typename _InitType>
void
__parallel_reduce_by_segment_offsets_fill(_BackendTag, _ExecutionPolicy&& __exec, _Range&& __result, _Size __m,
                                          const _InitType& __init)
{
    // A separate kernel name: the reduction kernel of the same call is instantiated too
    __parallel_for(_BackendTag{},
                   make_wrapped_policy<__segmented_reduce_fill_wrapper>(std::forward<_ExecutionPolicy>(__exec)),
                   __segmented_reduce_init_brick<_Acc, _InitType>{__init}, __m,
                   oneapi::dpl::__ranges::make_zip_view(std::forward<_Range>(__result)))
        .event()
        .wait_and_throw();
}

// The segments of the input [0, __n) defined by the __m + 1 offsets, reduced into the __m results
template <typename _Acc, typename _BackendTag, typename _ExecutionPolicy, typename _InRange, typename _OffsetRange,
          typename _OutRange, typename _Size, typename _InitType, typename _BinaryOperation>
void
__parallel_reduce_by_segment_offsets(_BackendTag, _ExecutionPolicy&& __exec, _InRange&& __input,
                                     _OffsetRange&& __offsets, _OutRange&& __result, const _Size __n, const _Size __m,
                                     _InitType __init, _BinaryOperation __binary_op)
{
    const bool __is_gpu = __exec.queue().get_device().is_gpu();

    // On GPUs arithmetic types by the tiles of the note, labeled by segments in 32 bits where they fit
    if constexpr (std::is_arithmetic_v<_Acc> &&
                  std::is_arithmetic_v<oneapi::dpl::__internal::__value_t<std::decay_t<_InRange>>>)
    {
        sycl::queue __q = __exec.queue();
        if (__is_gpu &&
            oneapi::dpl::__internal::__supports_sub_group_size(__q, __segmented_reduce_tiles_tuning::__lanes))
        {
            using _CustomName = oneapi::dpl::__internal::__policy_kernel_name<_ExecutionPolicy>;

            auto __submit = [&](auto __key_tag) {
                using _Key = decltype(__key_tag);
                using _TilesKernel =
                    __internal::__kernel_name_provider<__segmented_reduce_tiles_kernel<_CustomName, _Key>>;
                using _SumsKernel =
                    __internal::__kernel_name_provider<__segmented_reduce_sums_kernel<_CustomName, _Key>>;

                __segmented_reduce_tiles_submitter<_Acc, _Key, _TilesKernel, _SumsKernel, _InitType, _BinaryOperation>{
                    __init, __binary_op}(__q, __input, __offsets, __result, __n, __m);
            };
            if (std::uint64_t(__m) < std::numeric_limits<std::uint32_t>::max())
                __submit(std::uint32_t{});
            else
                __submit(std::uint64_t{});
            return;
        }
    }

    const _Size __threshold = __is_gpu ? __segmented_reduce_gpu_threshold : __segmented_reduce_cpu_threshold;

    // Reduces the segments longer than __min_length in chunks; __max_segments bounds the number of such segments
    auto __reduce_in_chunks = [&](_Size __min_length, bool __write_empty, _Size __max_length, _Size __max_segments) {
        const _Size __chunk_size = __segmented_reduce_chunk_size(__is_gpu, __max_length);

        // The end of the chunks of every segment: an inclusive scan of their numbers, computed from the pairs of
        // the adjacent offsets. The chunks depend on it through __chunk_ends, so the host does not wait for it; its
        // future keeps the temporary data of the scan until the partial results are combined.
        auto __chunk_ends = __buffer<_Size>(__m).get_buffer();
        auto __scan = __parallel_transform_scan(
            _BackendTag{}, make_wrapped_policy<__segmented_reduce_scan_wrapper>(__exec),
            oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::take_view_simple(__offsets, __m),
                                                 oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1})),
            oneapi::dpl::__ranges::views::all_write(__chunk_ends), __m,
            __segmented_reduce_chunk_count<_Size>{__n, __chunk_size, __min_length},
            unseq_backend::__no_init_value<_Size>{}, std::plus<_Size>{}, /*_Inclusive=*/std::true_type{});

        // At most n / __chunk_size + 1 chunks per segment, unless the offsets are not ordered
        const _Size __num_partials = __n / __chunk_size + __max_segments + 1;
        auto __partials = __buffer<_Acc>(__num_partials).get_buffer();
        __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_chunk_wrapper>(__exec),
                       __segmented_reduce_chunk_brick<_Acc, _Size, _InitType, _BinaryOperation>{__n, __m, __chunk_size,
                                                                                                __init, __binary_op},
                       __num_partials, oneapi::dpl::__ranges::views::all_read(__chunk_ends), __offsets,
                       oneapi::dpl::__ranges::make_zip_view(oneapi::dpl::__ranges::views::all_write(__partials)),
                       __input);
        __parallel_for(_BackendTag{}, make_wrapped_policy<__segmented_reduce_combine_wrapper>(__exec),
                       __segmented_reduce_combine_brick<_Acc, _Size, _InitType, _BinaryOperation>{
                           __num_partials, __write_empty, __init, __binary_op},
                       __m, oneapi::dpl::__ranges::views::all_read(__chunk_ends),
                       oneapi::dpl::__ranges::views::all_read(__partials),
                       oneapi::dpl::__ranges::make_zip_view(__result))
            .event()
            .wait_and_throw();
        __scan.event().wait_and_throw();
    };

    if (__n / __m > __threshold)
    {
        // The average segment is long: all the segments in chunks. The longest one selects the chunk length on GPUs.
        const _Size __max_length =
            __is_gpu
                ? __parallel_transform_reduce<_Size, std::true_type>(
                      _BackendTag{}, make_wrapped_policy<__segmented_reduce_length_wrapper>(__exec),
                      oneapi::dpl::maximum<_Size>{},
                      unseq_backend::walk_n<__segmented_reduce_length<_Size>>{__segmented_reduce_length<_Size>{__n}},
                      unseq_backend::__init_value<_Size>{_Size{0}},
                      oneapi::dpl::__ranges::drop_view_simple(__offsets, _Size{1}),
                      oneapi::dpl::__ranges::take_view_simple(__offsets, __m))
                : __n;
        if (__max_length == 0)
        {
            __parallel_reduce_by_segment_offsets_fill<_Acc>(_BackendTag{}, std::forward<_ExecutionPolicy>(__exec),
                                                            std::forward<_OutRange>(__result), __m, __init);
            return;
        }
        __reduce_in_chunks(_Size{0}, /*__write_empty=*/true, __max_length, __m);
        return;
    }

    // The short segments, one work-item each; the long ones only report the longest length. The report is in a buffer,
    // which costs less on GPUs than copying it from device USM.
    auto __report_buf = __buffer<std::uint32_t>(1).get_buffer();
    // The buffer has no host data: the atomic maximum starts from the zero written here
    __report_buf.get_host_access(sycl::write_only)[0] = 0;
    __parallel_for(_BackendTag{}, __exec,
                   __segmented_reduce_variable_brick<_Acc, _Size, _InitType, _BinaryOperation>{__n, __threshold, __init,
                                                                                               __binary_op},
                   __m, __offsets, oneapi::dpl::__ranges::make_zip_view(__result), __input,
                   oneapi::dpl::__ranges::views::all(__report_buf))
        .event()
        .wait_and_throw();
    const _Size __max_length = __report_buf.get_host_access(sycl::read_only)[0];

    // The long segments in chunks: at most n / __threshold of them
    if (__max_length > 0)
        __reduce_in_chunks(__threshold, /*__write_empty=*/false, __max_length, __n / __threshold);
}

} // namespace __par_backend_hetero
} // namespace dpl
} // namespace oneapi

#endif // _ONEDPL_PARALLEL_BACKEND_SYCL_REDUCE_BY_SEGMENT_OFFSETS_H
