#ifndef TENSOR_OPERATORS_HPP_
#define TENSOR_OPERATORS_HPP_

#include "type_selector.hpp"
#include "operation.hpp"
#include "expression_template_operators.hpp"

#define GTENSOR_UNARY_TENSOR_OPERATOR(NAME,F)\
template<typename Impl>\
inline auto NAME(const basic_tensor<Impl>& t){\
    return n_operator(F{},t);\
}\
template<typename Impl>\
inline auto NAME(basic_tensor<Impl>&& t){\
    return n_operator(F{},std::move(t));\
}

#define GTENSOR_BINARY_TENSOR_OPERATOR(NAME,F)\
template<typename...Ts, typename Other>\
inline auto NAME(const basic_tensor<Ts...>& t, Other&& other){\
    return n_operator(F{},t,std::forward<Other>(other));\
}\
template<typename...Ts, typename Other>\
inline auto NAME(basic_tensor<Ts...>&& t, Other&& other){\
    return n_operator(F{},std::move(t),std::forward<Other>(other));\
}\
template<typename...Us, typename...Ts>\
inline auto NAME(const basic_tensor<Us...>& t1, const basic_tensor<Ts...>& t2){\
    return n_operator(F{},t1,t2);\
}\
template<typename...Us, typename...Ts>\
inline auto NAME(const basic_tensor<Us...>& t1, basic_tensor<Ts...>&& t2){\
    return n_operator(F{},t1,std::move(t2));\
}\
template<typename...Us, typename...Ts>\
inline auto NAME(basic_tensor<Us...>&& t1, const basic_tensor<Ts...>& t2){\
    return n_operator(F{},std::move(t1),t2);\
}\
template<typename...Us, typename...Ts>\
inline auto NAME(basic_tensor<Us...>&& t1, basic_tensor<Ts...>&& t2){\
    return n_operator(F{},std::move(t1),std::move(t2));\
}

//restrict left Other if  is_tensor_v<Other> || is_convertible<Other,right::value_type> || is_convertible<right::value_type, Other>

// template<typename Other, typename...Ts>\
// inline auto NAME(Other&& other, const basic_tensor<Ts...>& t){\
//     return n_operator(F{},std::forward<Other>(other),t);\
// }\
// template<typename Other, typename...Ts>\
// inline auto NAME(Other&& other, basic_tensor<Ts...>&& t){\
//     return n_operator(F{},std::forward<Other>(other),std::move(t));\
// }

namespace gtensor{

namespace detail{

template<typename F, typename E1, typename E2, typename=void> constexpr bool is_defined_operation = false;
template<typename F, typename E1, typename E2> constexpr bool is_defined_operation<F,E1,E2,std::void_t<decltype(n_operator(std::declval<F>(),std::declval<E1>(),std::declval<E2>()))>> = true;

// template<typename F, typename E1, typename E2, typename=void> constexpr bool is_defined_operation = false;
// template<typename F, typename E1, typename E2> constexpr bool is_defined_operation<F,E1,E2,std::void_t<decltype(n_operator(std::declval<F>(),std::declval<E1>(),std::declval<E2>()))>> = true;
// template<typename F, typename E1, typename E2, typename=void> constexpr bool is_defined_operation = false;
// template<typename F, typename E1, typename E2> constexpr bool is_defined_operation<F,E1,E2,std::void_t<decltype(std::declval<F>()(std::declval<E1>(),std::declval<E2>()))>> = true;

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

}

template<typename F, typename...Operands>
inline auto n_operator(F&& f, Operands&&...operands){
    using config_type = typename detail::first_tensor_type_t<Operands...>::config_type;
    using operation_type = std::decay_t<F>;
    return n_operator_selector_t<config_type, operation_type>{}(
        std::forward<F>(f),
        std::forward<Operands>(operands)...
    );
}

// template<typename F, typename Operand, typename...Operands>
// inline auto n_operator(F&& f, Operand&& operand, Operands&&...operands){
//     using config_type = typename std::decay_t<Operand>::config_type;
//     using operation_type = std::decay_t<F>;
//     return n_operator_selector_t<config_type, operation_type>{}(
//         std::forward<F>(f),
//         std::forward<Operand>(operand),
//         std::forward<Operands>(operands)...
//     );
// }

template<typename...Us, typename...Vs>
static inline auto operator==(const basic_tensor<Us...>& t1, const basic_tensor<Vs...>& t2){
    if constexpr (std::is_same_v<basic_tensor<Us...>,basic_tensor<Vs...>>){
        if (static_cast<const void*>(&t1) == static_cast<const void*>(&t2)){
            return true;
        }
    }
    return t1.shape() == t2.shape() && std::equal(t1.begin(), t1.end(), t2.begin());
}

template<typename...Ts>
std::ostream& operator<<(std::ostream& os, const basic_tensor<Ts...>& t){
    return os<<str(t);
}
template<typename...Ts>
std::ostream& operator<<(std::ostream& os, basic_tensor<Ts...>& t){
    return os<<str(t);
}
template<typename...Ts>
std::ostream& operator<<(std::ostream& os, basic_tensor<Ts...>&& t){
    return os<<str(t);
}

//arithmetic
GTENSOR_UNARY_TENSOR_OPERATOR(operator+,operations::unary_plus);
GTENSOR_UNARY_TENSOR_OPERATOR(operator-,operations::unary_minus);
GTENSOR_BINARY_TENSOR_OPERATOR(operator+,operations::add);
GTENSOR_BINARY_TENSOR_OPERATOR(operator-,operations::sub);
GTENSOR_BINARY_TENSOR_OPERATOR(operator*,operations::mul);
GTENSOR_BINARY_TENSOR_OPERATOR(operator/,operations::div);
GTENSOR_BINARY_TENSOR_OPERATOR(operator%,operations::mod);

//bitwise
GTENSOR_UNARY_TENSOR_OPERATOR(operator~,operations::bitwise_not);
GTENSOR_BINARY_TENSOR_OPERATOR(operator&,operations::bitwise_and);
GTENSOR_BINARY_TENSOR_OPERATOR(operator|,operations::bitwise_or);
GTENSOR_BINARY_TENSOR_OPERATOR(operator^,operations::bitwise_xor);
GTENSOR_BINARY_TENSOR_OPERATOR(operator<<,operations::bitwise_lshift);
GTENSOR_BINARY_TENSOR_OPERATOR(operator>>,operations::bitwise_rshift);

//comparison
GTENSOR_BINARY_TENSOR_OPERATOR(equal,operations::equal);
GTENSOR_BINARY_TENSOR_OPERATOR(not_equal,operations::not_equal);
GTENSOR_BINARY_TENSOR_OPERATOR(operator>,operations::greater);
GTENSOR_BINARY_TENSOR_OPERATOR(operator>=,operations::greater_equal);
GTENSOR_BINARY_TENSOR_OPERATOR(operator<,operations::less);
GTENSOR_BINARY_TENSOR_OPERATOR(operator<=,operations::less_equal);

//logical
GTENSOR_UNARY_TENSOR_OPERATOR(operator!,operations::logic_not);
GTENSOR_BINARY_TENSOR_OPERATOR(operator&&,operations::logic_and);
GTENSOR_BINARY_TENSOR_OPERATOR(operator||,operations::logic_or);


// GTENSOR_UNARY_TENSOR_OPERATOR(operator+,operations::unary_plus);
// GTENSOR_UNARY_TENSOR_OPERATOR(operator-,operations::unary_minus);

// GTENSOR_BINARY_TENSOR_OPERATOR(operator+,operations::add);
// GTENSOR_BINARY_TENSOR_OPERATOR(operator-,operations::sub);
// GTENSOR_BINARY_TENSOR_OPERATOR(operator*,operations::mul);
// GTENSOR_BINARY_TENSOR_OPERATOR(operator/,operations::div);
// GTENSOR_BINARY_TENSOR_OPERATOR(equal,operations::equal);
// GTENSOR_BINARY_TENSOR_OPERATOR(operator>,operations::greater);
// GTENSOR_BINARY_TENSOR_OPERATOR(operator<,operations::less);
// GTENSOR_BINARY_TENSOR_OPERATOR(operator&&,operations::logic_and);
// GTENSOR_BINARY_TENSOR_OPERATOR(operator||,operations::logic_or);


}   //end of namespace gtensor

#endif