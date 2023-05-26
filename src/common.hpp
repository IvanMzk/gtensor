#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <type_traits>
#include <exception>
#include "config.hpp"

namespace gtensor{

template<typename Impl> class basic_tensor;
template<typename T, typename Config> class tensor;

class dim_exception : public std::runtime_error
{
public:
    explicit dim_exception(const char* what):
        std::runtime_error(what)
    {}
};


namespace detail{

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
GENERATE_HAS_CALLABLE_METHOD(create_walker(std::declval<typename T::dim_type>()), has_callable_create_walker);
GENERATE_HAS_CALLABLE_METHOD(begin(), has_callable_begin);
GENERATE_HAS_CALLABLE_METHOD(end(), has_callable_end);
GENERATE_HAS_CALLABLE_METHOD(rbegin(), has_callable_rbegin);
GENERATE_HAS_CALLABLE_METHOD(rend(), has_callable_rend);

template<typename T> using has_callable_subscript_operator = std::disjunction<
    has_callable_subscript_operator_difference_type<T>,
    has_callable_subscript_operator_size_type<T>,
    has_callable_subscript_operator_index_type<T>
>;

template<typename T> using has_callable_iterator = std::conjunction<has_callable_begin<T>,has_callable_end<T>>;
template<typename T> using has_callable_reverse_iterator = std::conjunction<has_callable_rbegin<T>,has_callable_rend<T>>;

template<typename...> inline constexpr bool always_false = false;

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

//standart library may require difference_type to be convertible to integral
template<typename It, typename T>
inline void fill(It first, It last, const T& v){
    using difference_type = typename std::iterator_traits<It>::difference_type;
    if constexpr (std::is_integral_v<difference_type>){
        std::fill(first,last,v);
    }else{
        for(;first!=last; ++first){
            *first = v;
        }
    }
}
template<typename It, typename T>
inline void advance(It& it, T n){
    using iterator_type = std::remove_cv_t<It>;
    using difference_type = typename std::iterator_traits<iterator_type>::difference_type;
    if constexpr (std::is_integral_v<difference_type>){
        std::advance(it,n);
    }else{
        using it_cat = typename std::iterator_traits<iterator_type>::iterator_category;
        if constexpr (std::is_convertible_v<it_cat,std::random_access_iterator_tag>){
            it+=n;
        }else{
            const difference_type zero_{0};
            if (n>=zero_){
                for (;n!=zero_;--n){
                    ++it;
                }
            }else{
                for (;n!=zero_;++n){
                    --it;
                }
            }
        }
    }
}
template<typename It>
inline It next(It it, typename std::iterator_traits<It>::difference_type n = 1)
{
    detail::advance(it, n);
    return it;
}

//returns dimension for given shape argument
//guarantes result is signed (assuming shape container difference_type is signed, as it must be)
template<typename ShT>
inline typename ShT::difference_type make_dim(const ShT& shape_or_strides){
    return shape_or_strides.size();
}
//make_direction helper to convert and check negative directions
//positive direction is converted to dim_type and returned as is whithout any checks
template<typename DimT, typename Direction>
inline DimT make_direction_helper(const DimT& dim, const Direction& direction){
    using dim_type = DimT;
    const dim_type direction_ = static_cast<dim_type>(direction);
    if (direction_ < dim_type{0}){
        const dim_type res = dim + direction_;
        if (res < dim_type{0}){
            throw gtensor::dim_exception("invalid negative direction");
        }
        return res;
    }else{
        return direction_;
    }
}
template<typename T, typename Direction>
inline auto make_direction(const T& shape_or_dim, const Direction& direction){
    if constexpr (is_container_v<T>){   //shape container
        return make_direction_helper(make_dim(shape_or_dim), direction);
    }else if constexpr (std::is_convertible_v<Direction,T>){  //dim scalar
        return make_direction_helper(shape_or_dim, direction);
    }else{
        static_assert(always_false<T>,"invalid shape_or_dim argument");
    }
}

}   //end of namespace detail
}   //end of namespace gtensor

#endif