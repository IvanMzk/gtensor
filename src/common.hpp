#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <type_traits>
#include "forward_decl.hpp"
#include "config.hpp"

namespace gtensor{
namespace detail{

template<typename T, typename = void> constexpr inline bool is_container_v = false;
template<typename T> constexpr inline bool is_container_v<T, std::void_t<decltype(std::begin(std::declval<T&>())), decltype(std::size(std::declval<T&>())), typename T::value_type>> = true;

template<typename T> constexpr inline bool is_tensor_v = false;
template<typename...Ts> constexpr inline bool is_tensor_v<gtensor::tensor<Ts...>> = true;

template<typename T, typename IdxT, typename = void> constexpr inline bool is_container_of_type_v = false;
template<typename T, typename IdxT> constexpr inline bool is_container_of_type_v<T, IdxT, std::void_t<std::enable_if_t<is_container_v<T>>>> = std::is_convertible_v<typename T::value_type,IdxT>;

template<typename T, typename U, typename=void> constexpr inline bool is_tensor_of_type_v = false;
template<typename T, typename U> constexpr inline bool is_tensor_of_type_v<T,U,std::void_t<std::enable_if_t<is_tensor_v<T>>>> = std::is_convertible_v<typename T::value_type, U>;

template<typename T, typename=void> constexpr inline bool is_bool_tensor_v = false;
template<typename T> constexpr inline bool is_bool_tensor_v<T,std::void_t<std::enable_if_t<is_tensor_v<T>>>> = std::is_same_v<typename T::value_type, bool>;

}   //end of namespace detail
}   //end of namespace gtensor

#endif