/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef HELPERS_FOR_TESTING_HPP_
#define HELPERS_FOR_TESTING_HPP_

#include <tuple>
#include <array>
#include <functional>
#include <sstream>

namespace helpers_for_testing{

template<typename...> struct list_concat;
template<template<typename...> typename L, typename...Us>
struct list_concat<L<Us...>>{
    using type = L<Us...>;
};
template<template<typename...> typename L, typename...Us, typename...Vs>
struct list_concat<L<Us...>,L<Vs...>>{
    using type = L<Us...,Vs...>;
};
template<typename T1, typename T2, typename...Tail>
struct list_concat<T1,T2,Tail...>{
    using type = typename list_concat<typename list_concat<T1,T2>::type, Tail...>::type;
};

template<template <typename...> typename, typename, typename> struct cross_product;
template<template <typename...> typename PairT, template<typename...> typename L, typename U, typename...Us, typename...Vs>
struct cross_product<PairT, L<U, Us...>, L<Vs...>>{
    using cross_u_vs = L<PairT<U,Vs>...>;
    using cross_us_vs = typename cross_product<PairT, L<Us...>, L<Vs...>>::type;
    using type = typename list_concat<cross_u_vs, cross_us_vs>::type;
};
template<template <typename...> typename PairT, template<typename...> typename L, typename...Vs>
struct cross_product<PairT, L<>, L<Vs...>>{
    using type = L<>;
};

//apply f to each element of t
template<typename F, typename Tuple, std::size_t...I>
inline auto apply_by_element(F&& f, Tuple&& t, std::index_sequence<I...>){
    if constexpr(std::disjunction_v<std::is_void<decltype(std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))))>...>){
        (std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))),...);
    }else{
        return std::make_tuple(std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t)))...);
    }
}
template<typename F, typename Tuple>
inline auto apply_by_element(F&& f, Tuple&& t){
    return apply_by_element(std::forward<F>(f), std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

template<typename T>
auto type_to_str(){
    std::stringstream ss{};
    if constexpr (std::is_const_v<std::remove_reference_t<T>>){
        ss<<"const";
    }
    if constexpr (std::is_volatile_v<T>){
        ss<<"volatile";
    }
    if constexpr (std::is_lvalue_reference_v<T>){
        ss<<"&";
    }
    if constexpr (std::is_rvalue_reference_v<T>){
        ss<<"&&";
    }
    ss<<typeid(T).name();
    return ss.str();
}

template<typename It>
auto range_to_str(It first, It last){
    std::stringstream ss{};
    for (;first!=last; ++first){
        ss<<*first<<" ";
    }
    return ss.str();
}

template<std::size_t a=279470273, std::size_t m=0xfffffffb, typename It, typename UnaryF>
auto generate_lehmer(It first, It last, UnaryF unary_f, std::size_t init){
    using value_type = typename std::iterator_traits<It>::value_type;
    std::for_each(first,last,
        [unary_f,init](auto& e)mutable{
            auto e_=init*a%m;
            init=e_;
            e=static_cast<value_type>(unary_f(e_));
        }
    );
}

template<std::size_t a=279470273, std::size_t m=0xfffffffb, typename It>
auto generate_lehmer(It first, It last, std::size_t init=1){
    generate_lehmer<a,m>(first,last,[](const auto& e){return e;},init);
}


}   //end of namespace helpers_for_testing

#endif