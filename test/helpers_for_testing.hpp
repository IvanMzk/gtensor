#ifndef HELPERS_FOR_TESTING_HPP_
#define HELPERS_FOR_TESTING_HPP_

#include <tuple>

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

template<typename F, typename Tuple>
auto apply_by_element(F&& f, Tuple&& t){
    using tuple_type = std::decay_t<Tuple>;
    return apply_by_element(std::forward<F>(f), std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size_v<tuple_type>>{});
}
template<typename F, typename Tuple, std::size_t...I>
auto apply_by_element(F&& f, Tuple&& t, std::index_sequence<I...>){
    if constexpr(std::disjunction_v<std::is_void<decltype(std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))))>...>){
        (std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))),...);
    }else{
        return std::make_tuple(std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t)))...);
    }
}

}   //end of namespace helpers_for_testing

#endif