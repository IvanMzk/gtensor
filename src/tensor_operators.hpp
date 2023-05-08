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
template<typename Other, typename...Ts, std::enable_if_t<detail::lhs_other_v<std::decay_t<Other>,typename basic_tensor<Ts...>::value_type>,int> =0>\
inline auto NAME(Other&& other, const basic_tensor<Ts...>& t){\
    return n_operator(F{},std::forward<Other>(other),t);\
}\
template<typename Other, typename...Ts, std::enable_if_t<detail::lhs_other_v<std::decay_t<Other>,typename basic_tensor<Ts...>::value_type>,int> =0>\
inline auto NAME(Other&& other, basic_tensor<Ts...>&& t){\
    return n_operator(F{},std::forward<Other>(other),std::move(t));\
}

#define GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(NAME,F)\
template<typename...Ts, typename Other>\
inline auto& NAME(basic_tensor<Ts...>& t, Other&& other){\
    return a_operator(F{},t,std::forward<Other>(other));\
}\

namespace gtensor{

namespace detail{

template<typename Other, typename T> constexpr bool lhs_other_v = std::is_convertible_v<Other,T>||std::is_convertible_v<T,Other>;

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

}   //end of namespace detail

template<typename F, typename...Operands>
inline auto n_operator(F&& f, Operands&&...operands){
    using config_type = typename detail::first_tensor_type_t<std::remove_cv_t<std::remove_reference_t<Operands>>...>::config_type;
    using operation_type = std::decay_t<F>;
    return operator_selector_t<config_type, operation_type>::n_operator(std::forward<F>(f),std::forward<Operands>(operands)...);
}

template<typename F, typename Rhs, typename...Ts>
inline auto& a_operator(F&& f, basic_tensor<Ts...>& lhs, Rhs&& rhs){
    using config_type = typename basic_tensor<Ts...>::config_type;
    using operation_type = std::decay_t<F>;
    return operator_selector_t<config_type, operation_type>::a_operator(std::forward<F>(f),lhs,std::forward<Rhs>(rhs));
}

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

//asignment
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(assign, operations::assign);
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(operator+=, operations::assign_add);
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(operator-=, operations::assign_sub);
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(operator*=, operations::assign_mul);
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(operator/=, operations::assign_div);
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(operator%=, operations::assign_mod);
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(operator&=, operations::assign_bitwise_and);
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(operator|=, operations::assign_bitwise_or);
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(operator^=, operations::assign_bitwise_xor);
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(operator<<=, operations::assign_bitwise_lshift);
GTENSOR_ASSIGNMENT_TENSOR_OPERATOR(operator>>=, operations::assign_bitwise_rshift);

}   //end of namespace gtensor

#endif