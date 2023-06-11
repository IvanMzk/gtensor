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

template<typename...Ts> inline constexpr bool has_tensor_arg_v = (is_tensor_v<Ts>||...);
template<typename Other, typename T> inline constexpr bool lhs_other_v = std::is_convertible_v<Other,T>||std::is_convertible_v<T,Other>;

template<typename T, typename B> struct tensor_value_type{using type = T;};
template<typename T> struct tensor_value_type<T,std::true_type>{using type = typename T::value_type;};
template<typename T> using tensor_value_type_t = typename tensor_value_type<T, std::bool_constant<is_tensor_v<T>>>::type;
//from pack of tensors and scalars make common value_type or void if no common value_type
template<typename, typename...Ts> struct tensor_common_value_type{using type = void;};
template<typename...Ts> struct tensor_common_value_type<std::void_t<std::common_type_t<tensor_value_type_t<Ts>...>>,Ts...>{using type = std::common_type_t<tensor_value_type_t<Ts>...>;};
template<typename...Ts> using tensor_common_value_type_t = typename tensor_common_value_type<void,Ts...>::type;

template<typename...Ts>
inline basic_tensor<Ts...>& as_basic_tensor(basic_tensor<Ts...>& t){
    return t;
}

template<typename...Us, typename...Vs>
inline auto tensor_equal_helper(std::true_type, const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan){    //arithmetic value_type overload
    const bool equal_shapes = u.shape() == v.shape();
    if (equal_nan){
        return equal_shapes && std::equal(u.begin(), u.end(), v.begin(),[](auto e1, auto e2){return math::isnan(e1) ? math::isnan(e2) : e1==e2;});
    }else{
        return equal_shapes && std::equal(u.begin(), u.end(), v.begin());
    }
}
template<typename...Us, typename...Vs>
inline auto tensor_equal_helper(std::false_type, const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool){    //math::isnan defined for arithmetic types only
    const bool equal_shapes = u.shape() == v.shape();
    return equal_shapes && std::equal(u.begin(), u.end(), v.begin());
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

//return true if two tensors has same shape and elements
//if equal_nan is true nans compared as equal
template<typename...Us, typename...Vs>
inline auto tensor_equal(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false){
    using common_value_type = detail::tensor_common_value_type_t<basic_tensor<Us...>, basic_tensor<Vs...>>;
    if (u.is_same(v)){
        return true;
    }else{
        return detail::tensor_equal_helper(typename std::is_arithmetic<common_value_type>::type{}, u, v, equal_nan);
    }
}
template<typename...Us, typename...Vs>
inline auto operator==(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v){
    return tensor_equal(u,v);
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

//assignment
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