#ifndef HELPERS_FOR_TESTING_HPP_
#define HELPERS_FOR_TESTING_HPP_

#include <tuple>
#include <array>
#include <functional>
#include "tuple_for_testing.hpp"

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
namespace details{

template<typename> struct tuple_size;
template<typename...Ts> struct tuple_size<std::tuple<Ts...>>{static constexpr std::size_t value = std::tuple_size_v<std::tuple<Ts...>>;};
template<typename...Ts> struct tuple_size<helpers_for_testing::tuple<Ts...>>{static constexpr std::size_t value = helpers_for_testing::tuple_size_v<tuple<Ts...>>;};
template<typename T> inline constexpr std::size_t tuple_size_v = tuple_size<T>::value;

template<typename> struct tuple_maker;
template<typename...Ts> struct tuple_maker<std::tuple<Ts...>>{
    template<typename...Args>
    static auto make_tuple(Args&&...args){return std::make_tuple(std::forward<Args>(args)...);}
};
template<typename...Ts> struct tuple_maker<helpers_for_testing::tuple<Ts...>>{
    template<typename...Args>
    static auto make_tuple(Args&&...args){return helpers_for_testing::create_tuple(std::forward<Args>(args)...);}
};

}   //end of namespace details

template<typename F, typename Tuple, std::size_t...I>
inline auto apply_by_element(F&& f, Tuple&& t, std::index_sequence<I...>){
    if constexpr(std::disjunction_v<std::is_void<decltype(std::invoke(std::forward<F>(f), get<I>(std::forward<Tuple>(t))))>...>){
        (std::invoke(std::forward<F>(f), get<I>(std::forward<Tuple>(t))),...);
    }else{
        return details::tuple_maker<std::decay_t<Tuple>>::make_tuple(std::invoke(std::forward<F>(f), get<I>(std::forward<Tuple>(t)))...);
    }
}
template<typename F, typename Tuple>
inline auto apply_by_element(F&& f, Tuple&& t){
    return apply_by_element(std::forward<F>(f), std::forward<Tuple>(t), std::make_index_sequence<details::tuple_size_v<std::decay_t<Tuple>>>{});
}

//safe cmp of signed,unsigned integrals
template<typename T, typename U>
inline constexpr bool cmp_equal(T t, U u){
    using UT = std::make_unsigned_t<T>;
    using UU = std::make_unsigned_t<U>;
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
        return t==u;
    else if constexpr (std::is_signed_v<T>)
        return t<0 ? false : UT(t) == u;
    else
        return u<0 ? false : t == UU(u);
}
template<typename T, typename U>
inline constexpr bool cmp_not_equal(T t, U u){
    return !cmp_equal(t, u);
}
template<typename T, typename U>
inline constexpr bool cmp_less(T t, U u){
    using UT = std::make_unsigned_t<T>;
    using UU = std::make_unsigned_t<U>;
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
        return t<u;
    else if constexpr (std::is_signed_v<T>)
        return t<0 ? true : UT(t) < u;
    else
        return u<0 ? false : t < UU(u);
}
template<typename T, typename U>
inline constexpr bool cmp_greater(T t, U u){
    return cmp_less(u, t);
}
template<typename T, typename U>
inline constexpr bool cmp_less_equal(T t, U u){
    return !cmp_greater(t, u);
}
template<typename T, typename U>
inline constexpr bool cmp_greater_equal(T t, U u){
    return !cmp_less(t, u);
}
}   //end of namespace helpers_for_testing

#endif