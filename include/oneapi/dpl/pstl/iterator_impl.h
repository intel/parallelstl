// -*- C++ -*-
//===-- iterator_impl.h ---------------------------------------------------===//
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

#ifndef _PSTL_iterator_impl_H
#define _PSTL_iterator_impl_H

#include <iterator>
#include <tuple>
#include <cassert>

#include "utils.h"
#include "tuple_impl.h"

namespace oneapi
{
namespace dpl
{
namespace __internal
{
template <size_t _Np>
struct __tuple_util
{
    template <typename _TupleType, typename _DifferenceType>
    static void
    __increment(_TupleType& __it, _DifferenceType __forward)
    {
        ::std::get<_Np - 1>(__it) = ::std::get<_Np - 1>(__it) + __forward;
        __tuple_util<_Np - 1>::__increment(__it, __forward);
    }
    template <typename _TupleType>
    static void
    __pre_increment(_TupleType& __it)
    {
        ++::std::get<_Np - 1>(__it);
        __tuple_util<_Np - 1>::__pre_increment(__it);
    }
};

template <>
struct __tuple_util<0>
{
    template <typename _TupleType, typename _DifferenceType>
    static void
    __increment(_TupleType&, _DifferenceType)
    {
    }
    template <typename _TupleType>
    static void
    __pre_increment(_TupleType&)
    {
    }
};

template <typename _TupleReturnType>
struct __make_references
{
    template <typename _TupleType, ::std::size_t... _Ip>
    _TupleReturnType
    operator()(const _TupleType& __t, oneapi::dpl::__internal::__index_sequence<_Ip...>)
    {
        return _TupleReturnType(*::std::get<_Ip>(__t)...);
    }
};

//zip_iterator version for forward iterator
//== and != comparison is performed only on the first element of the tuple
template <typename... _Types>
class zip_forward_iterator
{
    static const ::std::size_t __num_types = sizeof...(_Types);
    typedef typename ::std::tuple<_Types...> __it_types;

  public:
    typedef typename ::std::make_signed<::std::size_t>::type difference_type;
    typedef ::std::tuple<typename ::std::iterator_traits<_Types>::value_type...> value_type;
    typedef ::std::tuple<typename ::std::iterator_traits<_Types>::reference...> reference;
    typedef ::std::tuple<typename ::std::iterator_traits<_Types>::pointer...> pointer;
    typedef ::std::forward_iterator_tag iterator_category;

    zip_forward_iterator() : __my_it_() {}
    explicit zip_forward_iterator(_Types... __args) : __my_it_(::std::make_tuple(__args...)) {}
    zip_forward_iterator(const zip_forward_iterator& __input) : __my_it_(__input.__my_it_) {}
    zip_forward_iterator&
    operator=(const zip_forward_iterator& __input)
    {
        __my_it_ = __input.__my_it_;
        return *this;
    }

    reference operator*() const
    {
        return __make_references<reference>()(__my_it_, __make_index_sequence<__num_types>());
    }

    zip_forward_iterator&
    operator++()
    {
        __tuple_util<__num_types>::__pre_increment(__my_it_);
        return *this;
    }
    zip_forward_iterator
    operator++(int)
    {
        zip_forward_iterator __it(*this);
        ++(*this);
        return __it;
    }

    bool
    operator==(const zip_forward_iterator& __it) const
    {
        return ::std::get<0>(__my_it_) == ::std::get<0>(__it.__my_it_);
    }
    bool
    operator!=(const zip_forward_iterator& __it) const
    {
        return !(*this == __it);
    }

    __it_types
    base() const
    {
        return __my_it_;
    }

  private:
    __it_types __my_it_;
};

} // namespace __internal
} // namespace dpl
} // namespace oneapi

namespace oneapi
{
namespace dpl
{
template <typename _Ip>
class counting_iterator
{
    static_assert(::std::is_integral<_Ip>::value, "Cannot instantiate counting_iterator with a non-integer type");

  public:
    typedef typename ::std::make_signed<_Ip>::type difference_type;
    typedef _Ip value_type;
    typedef const _Ip* pointer;
    // There is no storage behind the iterator, so we return a value instead of reference.
    typedef _Ip reference;
    typedef ::std::random_access_iterator_tag iterator_category;
    using is_passed_directly = ::std::true_type;

    counting_iterator() : __my_counter_() {}
    explicit counting_iterator(_Ip __init) : __my_counter_(__init) {}

    reference operator*() const { return __my_counter_; }
    reference operator[](difference_type __i) const { return *(*this + __i); }

    difference_type
    operator-(const counting_iterator& __it) const
    {
        return __my_counter_ - __it.__my_counter_;
    }

    counting_iterator&
    operator+=(difference_type __forward)
    {
        __my_counter_ += __forward;
        return *this;
    }
    counting_iterator&
    operator-=(difference_type __backward)
    {
        return *this += -__backward;
    }
    counting_iterator&
    operator++()
    {
        return *this += 1;
    }
    counting_iterator&
    operator--()
    {
        return *this -= 1;
    }

    counting_iterator
    operator++(int)
    {
        counting_iterator __it(*this);
        ++(*this);
        return __it;
    }
    counting_iterator
    operator--(int)
    {
        counting_iterator __it(*this);
        --(*this);
        return __it;
    }

    counting_iterator
    operator-(difference_type __backward) const
    {
        return counting_iterator(__my_counter_ - __backward);
    }
    counting_iterator
    operator+(difference_type __forward) const
    {
        return counting_iterator(__my_counter_ + __forward);
    }
    friend counting_iterator
    operator+(difference_type __forward, const counting_iterator __it)
    {
        return __it + __forward;
    }

    bool
    operator==(const counting_iterator& __it) const
    {
        return *this - __it == 0;
    }
    bool
    operator!=(const counting_iterator& __it) const
    {
        return !(*this == __it);
    }
    bool
    operator<(const counting_iterator& __it) const
    {
        return *this - __it < 0;
    }
    bool
    operator>(const counting_iterator& __it) const
    {
        return __it < *this;
    }
    bool
    operator<=(const counting_iterator& __it) const
    {
        return !(*this > __it);
    }
    bool
    operator>=(const counting_iterator& __it) const
    {
        return !(*this < __it);
    }

  private:
    _Ip __my_counter_;
};

template <typename... _Types>
class zip_iterator
{
    static_assert(sizeof...(_Types) > 0, "Cannot instantiate zip_iterator with empty template parameter pack");
    static const ::std::size_t __num_types = sizeof...(_Types);
    typedef oneapi::dpl::__internal::tuple<_Types...> __it_types;

  public:
    typedef typename ::std::make_signed<::std::size_t>::type difference_type;
    typedef oneapi::dpl::__internal::tuple<typename ::std::iterator_traits<_Types>::value_type...> value_type;
    typedef oneapi::dpl::__internal::tuplewrapper<typename ::std::iterator_traits<_Types>::reference...> reference;
    typedef ::std::tuple<typename ::std::iterator_traits<_Types>::pointer...> pointer;
    typedef ::std::random_access_iterator_tag iterator_category;
    using is_zip = ::std::true_type;

    zip_iterator() : __my_it_() {}
    explicit zip_iterator(_Types... __args) : __my_it_(::std::make_tuple(__args...)) {}
    zip_iterator(const zip_iterator& __input) : __my_it_(__input.__my_it_) {}
    zip_iterator&
    operator=(const zip_iterator& __input)
    {
        __my_it_ = __input.__my_it_;
        return *this;
    }

    reference operator*() const
    {
        return oneapi::dpl::__internal::__make_references<reference>()(
            __my_it_, oneapi::dpl::__internal::__make_index_sequence<__num_types>());
    }
    reference operator[](difference_type __i) const { return *(*this + __i); }

    difference_type
    operator-(const zip_iterator& __it) const
    {
        return ::std::get<0>(__my_it_) - ::std::get<0>(__it.__my_it_);
    }

    zip_iterator&
    operator+=(difference_type __forward)
    {
        oneapi::dpl::__internal::__tuple_util<__num_types>::__increment(__my_it_, __forward);
        return *this;
    }
    zip_iterator&
    operator-=(difference_type __backward)
    {
        return *this += -__backward;
    }
    zip_iterator&
    operator++()
    {
        return *this += 1;
    }
    zip_iterator&
    operator--()
    {
        return *this -= 1;
    }

    zip_iterator
    operator++(int)
    {
        zip_iterator __it(*this);
        ++(*this);
        return __it;
    }
    zip_iterator
    operator--(int)
    {
        zip_iterator __it(*this);
        --(*this);
        return __it;
    }

    zip_iterator
    operator-(difference_type __backward) const
    {
        zip_iterator __it(*this);
        return __it -= __backward;
    }
    zip_iterator
    operator+(difference_type __forward) const
    {
        zip_iterator __it(*this);
        return __it += __forward;
    }
    friend zip_iterator
    operator+(difference_type __forward, const zip_iterator& __it)
    {
        return __it + __forward;
    }

    bool
    operator==(const zip_iterator& __it) const
    {
        return *this - __it == 0;
    }
    __it_types
    base() const
    {
        return __my_it_;
    }

    bool
    operator!=(const zip_iterator& __it) const
    {
        return !(*this == __it);
    }
    bool
    operator<(const zip_iterator& __it) const
    {
        return *this - __it < 0;
    }
    bool
    operator>(const zip_iterator& __it) const
    {
        return __it < *this;
    }
    bool
    operator<=(const zip_iterator& __it) const
    {
        return !(*this > __it);
    }
    bool
    operator>=(const zip_iterator& __it) const
    {
        return !(*this < __it);
    }

  private:
    __it_types __my_it_;
};

template <typename... _Tp>
zip_iterator<_Tp...>
make_zip_iterator(_Tp... __args)
{
    return zip_iterator<_Tp...>(__args...);
}

template <typename _Iter, typename _UnaryFunc>
class transform_iterator
{
  private:
    _Iter __my_it_;
    const _UnaryFunc __my_unary_func_;

  public:
    typedef typename ::std::iterator_traits<_Iter>::difference_type difference_type;
    typedef decltype(__my_unary_func_(::std::declval<typename ::std::iterator_traits<_Iter>::reference>())) reference;
    typedef typename ::std::remove_reference<reference>::type value_type;
    typedef typename ::std::iterator_traits<_Iter>::pointer pointer;
    typedef typename ::std::random_access_iterator_tag iterator_category;

    transform_iterator(_Iter __it, _UnaryFunc __unary_func) : __my_it_(__it), __my_unary_func_(__unary_func)
    {
        static_assert((::std::is_same<typename ::std::iterator_traits<_Iter>::iterator_category,
                                      ::std::random_access_iterator_tag>::value),
                      "Random access iterator required.");
    }
    transform_iterator(const transform_iterator& __input)
        : __my_it_(__input.__my_it_), __my_unary_func_(__input.__my_unary_func_)
    {
    }
    transform_iterator&
    operator=(const transform_iterator& __input)
    {
        __my_it_ = __input.__my_it_;
        return *this;
    }
    reference operator*() const { return __my_unary_func_(*__my_it_); }
    reference operator[](difference_type __i) const { return *(*this + __i); }
    transform_iterator&
    operator++()
    {
        ++__my_it_;
        return *this;
    }
    transform_iterator&
    operator--()
    {
        --__my_it_;
        return *this;
    }
    transform_iterator
    operator++(int)
    {
        transform_iterator __it(*this);
        ++(*this);
        return __it;
    }
    transform_iterator
    operator--(int)
    {
        transform_iterator __it(*this);
        --(*this);
        return __it;
    }
    transform_iterator
    operator+(difference_type __forward) const
    {
        return {__my_it_ + __forward, __my_unary_func_};
    }
    transform_iterator
    operator-(difference_type __backward) const
    {
        return {__my_it_ - __backward, __my_unary_func_};
    }
    transform_iterator&
    operator+=(difference_type __forward)
    {
        __my_it_ += __forward;
        return *this;
    }
    transform_iterator&
    operator-=(difference_type __backward)
    {
        __my_it_ -= __backward;
        return *this;
    }
    friend transform_iterator
    operator+(difference_type __forward, const transform_iterator& __it)
    {
        return __it + __forward;
    }
    difference_type
    operator-(const transform_iterator& __it) const
    {
        return __my_it_ - __it.__my_it_;
    }
    bool
    operator==(const transform_iterator& __it) const
    {
        return *this - __it == 0;
    }
    bool
    operator!=(const transform_iterator& __it) const
    {
        return !(*this == __it);
    }
    bool
    operator<(const transform_iterator& __it) const
    {
        return *this - __it < 0;
    }
    bool
    operator>(const transform_iterator& __it) const
    {
        return __it < *this;
    }
    bool
    operator<=(const transform_iterator& __it) const
    {
        return !(*this > __it);
    }
    bool
    operator>=(const transform_iterator& __it) const
    {
        return !(*this < __it);
    }

    _Iter
    base() const
    {
        return __my_it_;
    }
    _UnaryFunc
    functor() const
    {
        return __my_unary_func_;
    }
};

template <typename _Iter, typename _UnaryFunc>
transform_iterator<_Iter, _UnaryFunc>
make_transform_iterator(_Iter __it, _UnaryFunc __unary_func)
{
    return transform_iterator<_Iter, _UnaryFunc>(__it, __unary_func);
}

} // namespace dpl
} // namespace oneapi

namespace oneapi
{
namespace dpl
{
namespace __internal
{

struct make_zipiterator_functor
{
    template <typename... Args>
    auto
    operator()(Args&&... args) const -> decltype(oneapi::dpl::make_zip_iterator(::std::forward<Args>(args)...))
    {
        return oneapi::dpl::make_zip_iterator(::std::forward<Args>(args)...);
    }
};

// The functions are required because
// after applying a functor to each element of a tuple
// we may need to get a zip iterator

template <typename F, template <typename...> class TBig, typename... T, typename... RestTuples>
auto
map_zip(F f, TBig<T...> in, RestTuples... rest)
    -> decltype(map_tuple_impl(make_zipiterator_functor{},
                               oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(), in, rest...))
{
    return map_tuple_impl(make_zipiterator_functor{}, f, oneapi::dpl::__internal::__make_index_sequence<sizeof...(T)>(),
                          in, rest...);
}

} // namespace __internal
} // namespace dpl
} // namespace oneapi

namespace dpstd
{
using oneapi::dpl::counting_iterator;
using oneapi::dpl::make_transform_iterator;
using oneapi::dpl::make_zip_iterator;
using oneapi::dpl::transform_iterator;
using oneapi::dpl::zip_iterator;
} //namespace dpstd

#endif /* _PSTL_iterator_impl_H */
