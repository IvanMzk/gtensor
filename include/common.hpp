/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <type_traits>
#include "exception.hpp"
#include "config.hpp"

namespace gtensor{

template<typename Impl> class basic_tensor;
template<typename T, typename Layout, typename Config> class tensor;

namespace detail{

//tag to indicate argument of deducible type has no value
struct no_value{};

#define ASSERT_ORDER(order) static_assert(std::is_same_v<order, gtensor::config::c_order>||std::is_same_v<order, gtensor::config::f_order>, "order must be c_order or f_order");
#define ASSERT_TENSOR(t) static_assert(detail::is_tensor_v<t>,"tensor expected");

#if defined(__clang__)
#define ALWAYS_INLINE [[clang::always_inline]]
#elif defined(__GNUC__) || defined(__GNUG__)
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE inline
#endif

#define GENERATE_HAS_METHOD_SIGNATURE(function_name,function_signature,trait_name)\
template<typename T, typename = void>\
struct trait_name : std::false_type{};\
template<typename T>\
struct trait_name<T, std::void_t<std::integral_constant<function_signature,&T::function_name>>> : std::true_type{};

#define GENERATE_HAS_CALLABLE_METHOD(call_expression,trait_name)\
template<typename,typename=void> struct trait_name : std::false_type{};\
template<typename T> struct trait_name<T,std::void_t<decltype(std::declval<T>().call_expression)>> : std::true_type{};

GENERATE_HAS_CALLABLE_METHOD(operator[](std::declval<typename T::size_type>()), has_callable_subscript_operator_size_type);
GENERATE_HAS_CALLABLE_METHOD(operator[](std::declval<typename T::difference_type>()), has_callable_subscript_operator_difference_type);
GENERATE_HAS_CALLABLE_METHOD(operator[](std::declval<typename T::index_type>()), has_callable_subscript_operator_index_type);
GENERATE_HAS_CALLABLE_METHOD(create_indexer(), has_callable_create_indexer);
GENERATE_HAS_CALLABLE_METHOD(create_walker(), has_callable_create_walker);
GENERATE_HAS_CALLABLE_METHOD(create_walker(std::declval<typename T::dim_type>()), has_callable_create_walker_dim_type);
GENERATE_HAS_CALLABLE_METHOD(begin(), has_callable_begin);
GENERATE_HAS_CALLABLE_METHOD(end(), has_callable_end);
GENERATE_HAS_CALLABLE_METHOD(rbegin(), has_callable_rbegin);
GENERATE_HAS_CALLABLE_METHOD(rend(), has_callable_rend);
GENERATE_HAS_CALLABLE_METHOD(begin_trivial(), has_callable_begin_trivial);
GENERATE_HAS_CALLABLE_METHOD(end_trivial(), has_callable_end_trivial);
GENERATE_HAS_CALLABLE_METHOD(rbegin_trivial(), has_callable_rbegin_trivial);
GENERATE_HAS_CALLABLE_METHOD(rend_trivial(), has_callable_rend_trivial);
GENERATE_HAS_CALLABLE_METHOD(create_trivial_indexer(), has_callable_create_trivial_indexer);
GENERATE_HAS_CALLABLE_METHOD(is_trivial(), has_callable_is_trivial);

template<typename T> using has_callable_subscript_operator = std::disjunction<
    has_callable_subscript_operator_difference_type<T>,
    has_callable_subscript_operator_size_type<T>,
    has_callable_subscript_operator_index_type<T>
>;

template<typename, typename = void> inline constexpr bool is_iterator_v = false;
template<typename T> inline constexpr bool is_iterator_v<T,std::void_t<typename std::iterator_traits<T>::iterator_category>> = true;
template<typename, typename = void> inline constexpr bool is_random_access_iterator_v = false;
template<typename T> inline constexpr bool is_random_access_iterator_v<T,std::void_t<std::enable_if_t<is_iterator_v<T>>>> = std::is_convertible_v<typename std::iterator_traits<T>::iterator_category,std::random_access_iterator_tag>;

template<typename T> using has_callable_iterator = std::conjunction<has_callable_begin<T>,has_callable_end<T>>;
template<typename T> using has_callable_reverse_iterator = std::conjunction<has_callable_rbegin<T>,has_callable_rend<T>>;
template<typename T> using has_callable_iterator_trivial = std::conjunction<has_callable_begin_trivial<T>,has_callable_end_trivial<T>>;
template<typename T> using has_callable_reverse_iterator_trivial = std::conjunction<has_callable_rbegin_trivial<T>,has_callable_rend_trivial<T>>;

template<typename,typename> struct has_callable_random_access_iterator_helper : std::false_type{};
template<typename T> struct has_callable_random_access_iterator_helper<T,std::true_type> : std::bool_constant<is_random_access_iterator_v<decltype(std::declval<T>().begin())>>{};
template<typename T> using has_callable_random_access_iterator = has_callable_random_access_iterator_helper<T,std::bool_constant<has_callable_iterator<T>::value>>;

template<typename...> inline constexpr bool always_false = false;

struct unused_args{template<typename...Args> unused_args(const Args&...){}};

template<typename T, typename = void> inline constexpr bool is_container_v = false;
template<typename T> inline constexpr bool is_container_v<T, std::void_t<decltype(std::begin(std::declval<T&>())), decltype(std::size(std::declval<T&>())), typename T::value_type>> = true;

template<typename T>
struct is_tensor{
    template<typename U, typename Dummy=void> struct wrapper_{using type=U;};
    template<typename Dummy> struct wrapper_<void,Dummy>{using type=wrapper_<void>;};
    static std::false_type selector_(...);
    template<typename...Ts> static std::true_type selector_(basic_tensor<Ts...>);
    using type = decltype(selector_(std::declval<typename wrapper_<T>::type>()));
    static constexpr bool value = type::value;
};
template<typename T> inline constexpr bool is_tensor_v = is_tensor<T>::value;

template<typename...Ts> inline constexpr bool has_tensor_arg_v = (is_tensor_v<Ts>||...);

template<typename T, typename IdxT, typename = void> inline constexpr bool is_container_of_type_v = false;
template<typename T, typename IdxT> inline constexpr bool is_container_of_type_v<T, IdxT, std::void_t<std::enable_if_t<is_container_v<T>>>> = std::is_convertible_v<typename T::value_type,IdxT>;

template<typename T, typename U, typename=void> inline constexpr bool is_tensor_of_type_v = false;
template<typename T, typename U> inline constexpr bool is_tensor_of_type_v<T,U,std::void_t<std::enable_if_t<is_tensor_v<T>>>> = std::is_convertible_v<typename T::value_type, U>;

template<typename T, typename IdxT, typename = void> inline constexpr bool is_container_of_tensor_of_type_v = false;
template<typename T, typename IdxT> inline constexpr bool is_container_of_tensor_of_type_v<T, IdxT, std::void_t<std::enable_if_t<is_container_v<T>>>> = is_tensor_of_type_v<typename T::value_type, IdxT>;

template<typename T, typename=void> inline constexpr bool is_bool_tensor_v = false;
template<typename T> inline constexpr bool is_bool_tensor_v<T,std::void_t<std::enable_if_t<is_tensor_v<T>>>> = std::is_same_v<typename T::value_type, bool>;

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

//common order of tensors
//if all tensors have same order than common order is that same order, Config::order otherwise
template<typename...> struct common_order;
template<typename Config>
struct common_order<Config>{
    using type = typename Config::order;
};
template<typename Config, typename Order, typename...Orders>
struct common_order<Config,Order,Orders...>{
    using type = std::conditional_t<
        std::conjunction_v<std::is_same<Order,Orders>...>,
        Order,
        typename Config::order
    >;
};
template<typename Config, typename...Orders> using common_order_t = typename common_order<Config,Orders...>::type;

//from pack of tensors and scalars make common value_type or void if no common value_type
template<typename T, typename B> struct tensor_value_type{using type = T;};
template<typename T> struct tensor_value_type<T,std::true_type>{using type = typename T::value_type;};
template<typename T> using tensor_value_type_t = typename tensor_value_type<T, std::bool_constant<is_tensor_v<T>>>::type;
template<typename, typename...Ts> struct tensor_common_value_type{using type = void;};
template<typename...Ts> struct tensor_common_value_type<std::void_t<std::common_type_t<tensor_value_type_t<Ts>...>>,Ts...>{using type = std::common_type_t<tensor_value_type_t<Ts>...>;};
template<typename...Ts> using tensor_common_value_type_t = typename tensor_common_value_type<void,Ts...>::type;

//result type of tensor's copy<T,Config,Order>() call
template<typename T, typename Order, typename Config> struct copy_result
{
    template<typename U, typename> struct selector_{using type = tensor<U,Order,config::extend_config_t<Config,U>>;};
    template<typename U> struct selector_<U,std::true_type>{
        using value_type = typename selector_<typename U::value_type,std::bool_constant<is_tensor_v<typename U::value_type>>>::type;
        using type = tensor<value_type,Order,config::extend_config_t<Config,value_type>>;
    };
    using type = typename selector_<T,std::bool_constant<is_tensor_v<T>>>::type;
};
template<typename T, typename Order, typename Config> using copy_result_t = typename copy_result<T,Order,Config>::type;

//reserve space in arbitrary container, if possible
template<typename Container, typename T>
bool reserve(Container& container, const T& n){
    using Container_ = std::remove_cv_t<Container>;
    using difference_type = typename Container_::difference_type;
    if constexpr (is_static_castable_v<T,difference_type>){
        container.reserve(static_cast<const difference_type&>(n));
        return true;
    }else{
        return false;
    }
}

//returns dimension for given shape argument
//guarantes result is signed (assuming shape container difference_type is signed, as it must be)
template<typename ShT>
inline typename ShT::difference_type make_dim(const ShT& shape_or_strides){
    return shape_or_strides.size();
}
//make_axis helper to convert and check negative axis
//positive axis is converted to dim_type and returned as is whithout any checks
template<typename DimT, typename Axis>
inline DimT make_axis_helper(const DimT& dim, const Axis& axis){
    using dim_type = DimT;
    const dim_type axis_ = static_cast<dim_type>(axis);
    if (axis_ < dim_type{0}){
        const dim_type res = dim + axis_;
        if (res < dim_type{0}){
            throw gtensor::axis_error("invalid negative axis");
        }
        return res;
    }else{
        return axis_;
    }
}
template<typename T, typename Axis>
inline auto make_axis(const T& shape_or_dim, const Axis& axis){
    if constexpr (is_container_v<T>){   //shape container
        return make_axis_helper(make_dim(shape_or_dim), axis);
    }else if constexpr (std::is_convertible_v<Axis,T>){  //dim scalar
        return make_axis_helper(shape_or_dim, axis);
    }else{
        static_assert(always_false<T>,"invalid shape_or_dim argument");
    }
}

//Container is container of tensors
//return container of shapes (or references of shapes, if possible) of tensors
template<typename Container>
auto make_shapes_container(const Container& ts){
    using tensor_type = typename Container::value_type;
    using config_type = typename tensor_type::config_type;
    using shape_type = typename tensor_type::shape_type;
    using shapes_type = std::conditional_t<
        std::is_reference_v<decltype(std::declval<tensor_type>().shape())>,
        typename config_type::template container<std::reference_wrapper<const shape_type>>,
        typename config_type::template container<shape_type>
    >;
    shapes_type shapes{};
    shapes.reserve(ts.size());
    for (const auto& t : ts){
        shapes.push_back(t.shape());
    }
    return shapes;
}

//helper to work with container of shapes
template<typename ShT>
const ShT& unwrap_shape(const ShT& shape){
    return shape;
}
template<typename ShT>
const ShT& unwrap_shape(const std::reference_wrapper<ShT>& shape){
    return shape.get();
}


}   //end of namespace detail
}   //end of namespace gtensor

#endif