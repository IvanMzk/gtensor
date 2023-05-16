#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <type_traits>
#include "config.hpp"

namespace gtensor{

template<typename Impl> class basic_tensor;
template<typename T, typename Config> class tensor;

namespace detail{

#define GENERATE_HAS_MEMBER_FUNCTION_SIGNATURE(function_name,function_signature,trait_name)\
template<typename T, typename = void>\
struct trait_name : std::false_type{};\
template<typename T>\
struct trait_name<T, std::void_t<std::integral_constant<function_signature,&T::function_name>>> : std::true_type{};

template<typename T, typename = void> inline constexpr bool is_container_v = false;
template<typename T> inline constexpr bool is_container_v<T, std::void_t<decltype(std::begin(std::declval<T&>())), decltype(std::size(std::declval<T&>())), typename T::value_type>> = true;

template<typename T> inline constexpr bool is_tensor_v = false;
template<typename...Ts> inline constexpr bool is_tensor_v<gtensor::tensor<Ts...>> = true;
template<typename...Ts> inline constexpr bool is_tensor_v<gtensor::basic_tensor<Ts...>> = true;

template<typename T, typename IdxT, typename = void> inline constexpr bool is_container_of_type_v = false;
template<typename T, typename IdxT> inline constexpr bool is_container_of_type_v<T, IdxT, std::void_t<std::enable_if_t<is_container_v<T>>>> = std::is_convertible_v<typename T::value_type,IdxT>;

template<typename T, typename U, typename=void> inline constexpr bool is_tensor_of_type_v = false;
template<typename T, typename U> inline constexpr bool is_tensor_of_type_v<T,U,std::void_t<std::enable_if_t<is_tensor_v<T>>>> = std::is_convertible_v<typename T::value_type, U>;

template<typename T, typename=void> inline constexpr bool is_bool_tensor_v = false;
template<typename T> inline constexpr bool is_bool_tensor_v<T,std::void_t<std::enable_if_t<is_tensor_v<T>>>> = std::is_same_v<typename T::value_type, bool>;

template<typename, typename = void> inline constexpr bool is_iterator_v = false;
template<typename T> inline constexpr bool is_iterator_v<T,std::void_t<typename std::iterator_traits<T>::iterator_category>> = true;

template<typename From, typename To, typename=void> inline constexpr bool is_static_castable_v = false;
template<typename From, typename To> inline constexpr bool is_static_castable_v<From,To,std::void_t<decltype(static_cast<To>(std::declval<From>()))>> = true;

//find first type in pack fo which is_tensor_v is true
template<typename...Ts> struct first_tensor_type;
template<typename...Ts> struct first_tensor_type_helper;
template<typename T, typename...Ts> struct first_tensor_type_helper<std::true_type,T,Ts...>{
    using type = T;
};
template<typename T, typename...Ts> struct first_tensor_type_helper<std::false_type,T,Ts...>{
    using type = typename first_tensor_type<Ts...>::type;
};
template<typename T, typename...Ts> struct first_tensor_type<T,Ts...>{
    using type = typename first_tensor_type_helper<std::bool_constant<is_tensor_v<T>>,T,Ts...>::type;
};
template<typename...Ts> using first_tensor_type_t = typename first_tensor_type<Ts...>::type;

//std::fill may require difference_type to be convertible to integral
template<typename It, typename T>
void fill(It first, It last, const T& v){
    using difference_type = typename std::iterator_traits<It>::difference_type;
    if constexpr (std::is_integral_v<difference_type>){
        std::fill(first,last,v);
    }else{
        for(;first!=last; ++first){
            *first = v;
        }
    }
}

}   //end of namespace detail
}   //end of namespace gtensor

#endif