#ifndef TENSOR_OPERATORS_HPP_
#define TENSOR_OPERATORS_HPP_

#include "type_selector.hpp"
#include "operation.hpp"
#include "expression_template_operator.hpp"

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

#define GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(NAME,F)\
template<typename...Ts, typename Rhs>\
inline basic_tensor<Ts...>& NAME(basic_tensor<Ts...>& lhs, Rhs&& rhs){\
    a_operator(F{},lhs,std::forward<Rhs>(rhs));\
    return lhs;\
}\
template<typename...Ts, typename Rhs>\
inline tensor<Ts...>& NAME(tensor<Ts...>& lhs, Rhs&& rhs){\
    NAME(detail::as_basic_tensor(lhs),std::forward<Rhs>(rhs));\
    return lhs;\
}

namespace gtensor{

namespace detail{

template<typename Other, typename T> inline constexpr bool lhs_other_v = std::is_convertible_v<Other,T>||std::is_convertible_v<T,Other>;

template<typename...Ts>
inline basic_tensor<Ts...>& as_basic_tensor(basic_tensor<Ts...>& t){
    return t;
}

}   //end of namespace detail

//generalized broadcast operator
template<typename F, typename...Operands>
inline auto n_operator(F&& f, Operands&&...operands){
    using config_type = typename detail::first_tensor_type_t<std::remove_cv_t<std::remove_reference_t<Operands>>...>::config_type;
    using operation_type = std::decay_t<F>;
    return operator_selector_t<config_type, operation_type>::n_operator(std::forward<F>(f),std::forward<Operands>(operands)...);
}

//generalized broadcast assign
template<typename F, typename Rhs, typename...Ts>
inline basic_tensor<Ts...>& a_operator(F&& f, basic_tensor<Ts...>& lhs, Rhs&& rhs){
    using config_type = typename basic_tensor<Ts...>::config_type;
    using operation_type = std::decay_t<F>;
    operator_selector_t<config_type, operation_type>::a_operator(std::forward<F>(f),lhs,std::forward<Rhs>(rhs));
    return lhs;
}

template<typename...Us, typename...Vs>
static inline auto operator==(const basic_tensor<Us...>& t1, const basic_tensor<Vs...>& t2){
    if (t1.is_same(t2)){
        return true;
    }else{
        return t1.shape() == t2.shape() && std::equal(t1.begin(), t1.end(), t2.begin());
    }
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
template<typename...Ts, typename Rhs>
inline basic_tensor<Ts...>& assign(basic_tensor<Ts...>& lhs, Rhs&& rhs){
    using RhsT = std::remove_cv_t<std::remove_reference_t<Rhs>>;
    static_assert(detail::is_tensor_v<RhsT>||std::is_convertible_v<RhsT,typename basic_tensor<Ts...>::value_type>);
    if (lhs.is_same(rhs)){
        return lhs;
    }
    a_operator(operations::assign{},lhs,std::forward<Rhs>(rhs));
    return lhs;
}
template<typename...Ts, typename Rhs>
inline tensor<Ts...>& assign(tensor<Ts...>& lhs, Rhs&& rhs){
    assign(detail::as_basic_tensor(lhs),std::forward<Rhs>(rhs));
    return lhs;
}
//compound assignment
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator+=, operations::assign_add);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator-=, operations::assign_sub);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator*=, operations::assign_mul);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator/=, operations::assign_div);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator%=, operations::assign_mod);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator&=, operations::assign_bitwise_and);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator|=, operations::assign_bitwise_or);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator^=, operations::assign_bitwise_xor);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator<<=, operations::assign_bitwise_lshift);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator>>=, operations::assign_bitwise_rshift);

}   //end of namespace gtensor
#endif