/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef TENSOR_OPERATORS_HPP_
#define TENSOR_OPERATORS_HPP_

#include "module_selector.hpp"
#include "operations.hpp"
#include "expression_template_engine/expression_template_operator.hpp"

namespace gtensor{

namespace detail{

template<typename Other, typename T> inline constexpr bool lhs_other_v = std::is_convertible_v<Other,T>||std::is_convertible_v<T,Other>;
template<typename T, typename U=void> static constexpr bool is_printable_v = false;
template<typename T> static constexpr bool is_printable_v<T,std::void_t<decltype(std::cout<<std::declval<T>())>> = true;

template<typename...Ts>
inline basic_tensor<Ts...>& as_basic_tensor(basic_tensor<Ts...>& t){
    return t;
}

template<typename Stream, typename...Ts>
void str_helper(const basic_tensor<Ts...>& t, std::string separator, Stream& stream){
    using index_type = typename basic_tensor<Ts...>::index_type;
    using value_type = typename basic_tensor<Ts...>::value_type;
    if (t.size()>0){
        if (t.dim()>1){
            auto axis_size = t.shape()[0];
            if (axis_size > 0){
                if (axis_size > 6){
                    stream<<"{",str_helper(t(0).copy(),separator,stream),stream<<"}"<<separator;
                    stream<<"{",str_helper(t(1).copy(),separator,stream),stream<<"}"<<separator;
                    stream<<"{",str_helper(t(2).copy(),separator,stream),stream<<"}"<<separator;
                    stream<<"..."<<separator;
                    stream<<"{",str_helper(t(axis_size-3).copy(),separator,stream),stream<<"}"<<separator;
                    stream<<"{",str_helper(t(axis_size-2).copy(),separator,stream),stream<<"}"<<separator;
                    stream<<"{",str_helper(t(axis_size-1).copy(),separator,stream),stream<<"}";
                }else{
                    index_type i=0;
                    for (--axis_size; i!=axis_size; ++i){
                        stream<<"{";
                        str_helper(t(i).copy(),separator,stream);
                        stream<<"}"<<separator;
                    }
                    stream<<"{";
                    str_helper(t(i).copy(),separator,stream);
                    stream<<"}";
                }
            }
        }else{
            if constexpr (is_printable_v<value_type>){
                const index_type n_max{1000};
                auto it = t.begin();
                auto last = t.end();
                const auto n = last - it;
                if (n>n_max){
                        stream<<*it++<<separator;
                        stream<<*it++<<separator;
                        stream<<*it++<<separator;
                        stream<<"..."<<separator;
                        last-=index_type{3};
                        stream<<*last++<<separator;
                        stream<<*last++<<separator;
                        stream<<*last;
                }else{
                    for (--last; it!=last; ++it){
                        stream<<*it<<separator;
                    }
                    stream<<*it;
                }
            }else{
                stream<<"...";
            }
        }
    }
}

}   //end of namespace detail

//generalized elementwise broadcast n-arity operator
//F is operation functor which perform on operands elements
//F's call operator arity must be equal to operands number,
//Operands may be scalar or tensor, operands shapes must broadcast
//result is tensor, result's value_type is f's call operator result type;
template<typename F, typename...Operands>
inline auto n_operator(F&& f, Operands&&...operands){
    using config_type = typename detail::first_tensor_type_t<std::remove_cv_t<std::remove_reference_t<Operands>>...>::config_type;
    using operation_type = std::decay_t<F>;
    return generalized_operator_selector_t<config_type, operation_type>::n_operator(std::forward<F>(f),std::forward<Operands>(operands)...);
}

//generalized elementwise broadcast assign operator
//F is assign operation functor
//F's call operator takes reference to lhs and rhs elements and should have assign semantic or compaund assign semantic, return is discarded
//Lhs is tensor, Rhs tensor or scalar, shapes of lhs and rhs must broadcast
//result is reference to lhs
template<typename F, typename Tensor, typename Rhs>
inline std::decay_t<Tensor>& a_operator(F&& f, Tensor&& lhs, Rhs&& rhs){
    using F_ = std::decay_t<F>;
    using Tensor_ = std::decay_t<Tensor>;
    using config_type = typename Tensor_::config_type;
    generalized_operator_selector_t<config_type, F_>::a_operator(std::forward<F>(f),std::forward<Tensor>(lhs),std::forward<Rhs>(rhs));
    return lhs;
}

//tensor operators and related functions implementation

#define GTENSOR_TENSOR_OPERATOR_FUNCTION(NAME,F)\
template<typename...Args>\
static auto NAME(Args&&...args){\
    static_assert(detail::has_tensor_arg_v<std::remove_cv_t<std::remove_reference_t<Args>>...>,"at least one arg must be tensor");\
    return n_operator(F{},std::forward<Args>(args)...);\
}

#define GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(NAME,F)\
template<typename Tensor, typename Rhs>\
static std::decay_t<Tensor>& NAME(Tensor&& lhs, Rhs&& rhs){\
    static_assert(detail::is_tensor_v<std::decay_t<Tensor>>,"lhs must be tensor");\
    a_operator(F{},std::forward<Tensor>(lhs),std::forward<Rhs>(rhs));\
    return lhs;\
}

struct tensor_operators
{
    //return true if two tensors has same shape and elements
    //if equal_nan is true nans compared as equal
    template<typename...Us, typename...Vs>
    static bool tensor_equal(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false){
        using common_order = detail::common_order_t<typename basic_tensor<Us...>::config_type, typename basic_tensor<Us...>::order, typename basic_tensor<Vs...>::order>;
        if (u.is_same(v)){
            return true;
        }else{
            const bool equal_shapes = u.shape() == v.shape();
            auto a_u = u.traverse_order_adapter(common_order{});
            auto a_v = v.traverse_order_adapter(common_order{});
            if (equal_nan){
                return equal_shapes && std::equal(a_u.begin(), a_u.end(), a_v.begin(), gtensor::operations::math_isequal<std::true_type>{});
            }else{
                return equal_shapes && std::equal(a_u.begin(), a_u.end(), a_v.begin(), gtensor::operations::math_isequal<std::false_type>{});
            }
        }
    }

    //return true if two tensors have same shape and close elements within specified tolerance
    template<typename...Us, typename...Vs, typename Tol>
    static bool tensor_close(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, Tol relative_tolerance, Tol absolute_tolerance, bool equal_nan = false){
        using common_order = detail::common_order_t<typename basic_tensor<Us...>::config_type, typename basic_tensor<Us...>::order, typename basic_tensor<Vs...>::order>;
        if (u.is_same(v)){
            return true;
        }else{
            const bool equal_shapes = u.shape() == v.shape();
            auto a_u = u.traverse_order_adapter(common_order{});
            auto a_v = v.traverse_order_adapter(common_order{});
            if (equal_nan){
                return equal_shapes && std::equal(a_u.begin(), a_u.end(), a_v.begin(), operations::math_isclose<Tol,std::true_type>{relative_tolerance,absolute_tolerance});
            }else{
                return equal_shapes && std::equal(a_u.begin(), a_u.end(), a_v.begin(), operations::math_isclose<Tol,std::false_type>{relative_tolerance,absolute_tolerance});
            }
        }
    }
    //return true if two tensors have same shape and close elements, use machine epsilon as tolerance
    template<typename...Us, typename...Vs>
    static bool tensor_close(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false){
        using common_value_type = detail::tensor_common_value_type_t<basic_tensor<Us...>,basic_tensor<Vs...>>;
        static constexpr common_value_type e = math::numeric_traits<common_value_type>::epsilon();
        return tensor_close(u,v,e,e,equal_nan);
    }

    //return true if two tensors have close elements within specified tolerance
    //shapes may not be equal, but must broadcast
    template<typename...Us, typename...Vs, typename Tol>
    static bool allclose(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, Tol relative_tolerance, Tol absolute_tolerance, bool equal_nan = false){
        using common_order = detail::common_order_t<typename basic_tensor<Us...>::config_type, typename basic_tensor<Us...>::order, typename basic_tensor<Vs...>::order>;
        using shape_type = typename basic_tensor<Us...>::shape_type;
        if (u.is_same(v)){
            return true;
        }else{
            auto common_shape = detail::make_broadcast_shape<shape_type>(u.shape(),v.shape());
            auto a_u = u.traverse_order_adapter(common_order{});
            auto a_v = v.traverse_order_adapter(common_order{});
            if (equal_nan){
                return std::equal(a_u.begin(common_shape), a_u.end(common_shape), a_v.begin(common_shape), operations::math_isclose<Tol,std::true_type>{relative_tolerance,absolute_tolerance});
            }else{
                return std::equal(a_u.begin(common_shape), a_u.end(common_shape), a_v.begin(common_shape), operations::math_isclose<Tol,std::false_type>{relative_tolerance,absolute_tolerance});
            }
        }
    }
    //return true if two tensors have close elements, use machine epsilon as tolerance
    template<typename...Us, typename...Vs>
    static bool allclose(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false){
        using common_value_type = detail::tensor_common_value_type_t<basic_tensor<Us...>,basic_tensor<Vs...>>;
        static constexpr common_value_type e = math::numeric_traits<common_value_type>::epsilon();
        return allclose(u,v,e,e,equal_nan);
    }

    //return string representation of tensor
    template<typename P, typename...Ts>
    static auto str(const basic_tensor<Ts...>& t, const P& precision){
        std::stringstream ss{};
        ss.precision(precision);
        std::string separator{","};
        ss<<"["<<detail::shape_to_str(t.shape())<<"{";
        detail::str_helper(t,separator,ss);
        ss<<"}]";
        return ss.str();
    }

    //cast
    template<typename To, typename T>
    static auto cast(T&& t){
        ASSERT_TENSOR(std::remove_cv_t<std::remove_reference_t<T>>);
        return n_operator(operations::cast<To>{}, std::forward<T>(t));
    }

    //like ternary operator
    GTENSOR_TENSOR_OPERATOR_FUNCTION(where,operations::where);

    //arithmetic
    GTENSOR_TENSOR_OPERATOR_FUNCTION(unary_plus,operations::unary_plus);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(unary_minus,operations::unary_minus);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(add,operations::add);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(sub,operations::sub);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(mul,operations::mul);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(div,operations::div);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(mod,operations::mod);

    //bitwise
    GTENSOR_TENSOR_OPERATOR_FUNCTION(bitwise_not,operations::bitwise_not);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(bitwise_and,operations::bitwise_and);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(bitwise_or,operations::bitwise_or);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(bitwise_xor,operations::bitwise_xor);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(bitwise_lshift,operations::bitwise_lshift);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(bitwise_rshift,operations::bitwise_rshift);

    //strict comparison
    GTENSOR_TENSOR_OPERATOR_FUNCTION(equal,operations::equal);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(not_equal,operations::not_equal);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(greater,operations::greater);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(greater_equal,operations::greater_equal);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(less,operations::less);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(less_equal,operations::less_equal);

    //close comparison
    template<typename T, typename U, typename Tol, typename EqualNan = std::false_type>
    static auto isclose(T&& t, U&& u, Tol relative_tolerance, Tol absolute_tolerance, EqualNan equal_nan = EqualNan{}){
        using T_ = std::remove_cv_t<std::remove_reference_t<T>>;
        using U_ = std::remove_cv_t<std::remove_reference_t<U>>;
        (void)equal_nan;
        static_assert(detail::has_tensor_arg_v<T_,U_>,"at least one arg must be tensor");
        return n_operator(operations::math_isclose<Tol, EqualNan>{relative_tolerance, absolute_tolerance}, std::forward<T>(t), std::forward<U>(u));
    }
    template<typename T, typename U, typename EqualNan = std::false_type>
    static auto isclose(T&& t, U&& u, EqualNan equal_nan = EqualNan{}){
        using T_ = std::remove_cv_t<std::remove_reference_t<T>>;
        using U_ = std::remove_cv_t<std::remove_reference_t<U>>;
        using common_value_type = detail::tensor_common_value_type_t<T_,U_>;
        static constexpr common_value_type e = math::numeric_traits<common_value_type>::epsilon();
        return isclose(std::forward<T>(t),std::forward<U>(u),e,e,equal_nan);
    }

    //logical
    GTENSOR_TENSOR_OPERATOR_FUNCTION(logic_not,operations::logic_not);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(logic_and,operations::logic_and);
    GTENSOR_TENSOR_OPERATOR_FUNCTION(logic_or,operations::logic_or);

    //elementwise assignment
    template<typename...Ts, typename Rhs>
    static basic_tensor<Ts...>& assign(basic_tensor<Ts...>& lhs, Rhs&& rhs){
        using RhsT = std::remove_cv_t<std::remove_reference_t<Rhs>>;
        static_assert(detail::is_tensor_v<RhsT>||std::is_convertible_v<RhsT,typename basic_tensor<Ts...>::value_type>);
        if (lhs.is_same(rhs)){
            return lhs;
        }
        a_operator(operations::assign{},lhs,std::forward<Rhs>(rhs));
        return lhs;
    }
    template<typename...Ts, typename Rhs>
    static tensor<Ts...>& assign(tensor<Ts...>& lhs, Rhs&& rhs){
        assign(detail::as_basic_tensor(lhs),std::forward<Rhs>(rhs));
        return lhs;
    }

    //elementwise compound assignment
    GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(assign_add,operations::assign_add);
    GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(assign_sub,operations::assign_sub);
    GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(assign_mul,operations::assign_mul);
    GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(assign_div,operations::assign_div);
    GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(assign_mod,operations::assign_mod);
    GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(assign_bitwise_and,operations::assign_bitwise_and);
    GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(assign_bitwise_or,operations::assign_bitwise_or);
    GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(assign_bitwise_xor,operations::assign_bitwise_xor);
    GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(assign_bitwise_lshift,operations::assign_bitwise_lshift);
    GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION(assign_bitwise_rshift,operations::assign_bitwise_rshift);
};  //end of tensor_operators

//tensor operators and related functions frontend
//frontend uses compile-time dispatch to select implementation, see module_selector.hpp

//return true if two tensors has same shape and elements
//if equal_nan is true nans compared as equal
template<typename...Us, typename...Vs>
bool tensor_equal(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false){
    using config_type = typename basic_tensor<Us...>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::tensor_equal(u,v,equal_nan);
}

template<typename...Us, typename...Vs>
bool operator==(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v){
    return tensor_equal(u,v);
}
template<typename...Us, typename...Vs>
bool operator!=(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v){
    return !(u==v);
}

//return true if two tensors have same shape and close elements within specified tolerance
template<typename...Us, typename...Vs, typename Tol>
bool tensor_close(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, Tol relative_tolerance, Tol absolute_tolerance, bool equal_nan = false){
    using config_type = typename basic_tensor<Us...>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::tensor_close(u,v,relative_tolerance,absolute_tolerance,equal_nan);
}
//return true if two tensors have same shape and close elements, use machine epsilon as tolerance
template<typename...Us, typename...Vs>
bool tensor_close(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false){
    using config_type = typename basic_tensor<Us...>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::tensor_close(u,v,equal_nan);
}

//return true if two tensors have close elements within specified tolerance
//shapes may not be equal, but must broadcast
template<typename...Us, typename...Vs, typename Tol>
bool allclose(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, Tol relative_tolerance, Tol absolute_tolerance, bool equal_nan = false){
    using config_type = typename basic_tensor<Us...>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::allclose(u,v,relative_tolerance,absolute_tolerance,equal_nan);
}
//return true if two tensors have close elements, use machine epsilon as tolerance
template<typename...Us, typename...Vs>
bool allclose(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false){
    using config_type = typename basic_tensor<Us...>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::allclose(u,v,equal_nan);
}

//return tensor's string representation
template<typename P=int, typename...Ts>
auto str(const basic_tensor<Ts...>& t, const P& precision=3){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::str(t,precision);
}

template<typename...Ts>
std::ostream& operator<<(std::ostream& os, const basic_tensor<Ts...>& t){
    return os<<str(t);
}

//elementwise tensor operators and related functions frontend

#define GTENSOR_UNARY_TENSOR_OPERATOR(NAME,F)\
template<typename...Ts>\
auto NAME(const basic_tensor<Ts...>& t){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return gtensor::tensor_operators_selector_t<config_type>::F(t);\
}\
template<typename...Ts>\
auto NAME(basic_tensor<Ts...>&& t){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return gtensor::tensor_operators_selector_t<config_type>::F(std::move(t));\
}

#define GTENSOR_BINARY_TENSOR_OPERATOR(NAME,F)\
template<typename...Ts, typename Other>\
auto NAME(const basic_tensor<Ts...>& t, Other&& other){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return gtensor::tensor_operators_selector_t<config_type>::F(t, std::forward<Other>(other));\
}\
template<typename...Ts, typename Other>\
auto NAME(basic_tensor<Ts...>&& t, Other&& other){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return gtensor::tensor_operators_selector_t<config_type>::F(std::move(t), std::forward<Other>(other));\
}\
template<typename Other, typename...Ts, std::enable_if_t<detail::lhs_other_v<std::decay_t<Other>,typename basic_tensor<Ts...>::value_type>,int> =0>\
auto NAME(Other&& other, const basic_tensor<Ts...>& t){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return gtensor::tensor_operators_selector_t<config_type>::F(std::forward<Other>(other),t);\
}\
template<typename Other, typename...Ts, std::enable_if_t<detail::lhs_other_v<std::decay_t<Other>,typename basic_tensor<Ts...>::value_type>,int> =0>\
auto NAME(Other&& other, basic_tensor<Ts...>&& t){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return gtensor::tensor_operators_selector_t<config_type>::F(std::forward<Other>(other),std::move(t));\
}

#define GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(NAME,F)\
template<typename...Ts, typename Rhs>\
basic_tensor<Ts...>& NAME(basic_tensor<Ts...>& lhs, Rhs&& rhs){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return gtensor::tensor_operators_selector_t<config_type>::F(lhs,std::forward<Rhs>(rhs));\
}\
template<typename...Ts, typename Rhs>\
basic_tensor<Ts...>& NAME(basic_tensor<Ts...>&& lhs, Rhs&& rhs){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return gtensor::tensor_operators_selector_t<config_type>::F(std::move(lhs),std::forward<Rhs>(rhs));\
}\
template<typename...Ts, typename Rhs>\
tensor<Ts...>& NAME(tensor<Ts...>& lhs, Rhs&& rhs){\
    using config_type = typename tensor<Ts...>::config_type;\
    return gtensor::tensor_operators_selector_t<config_type>::F(lhs,std::forward<Rhs>(rhs));\
}\
template<typename...Ts, typename Rhs>\
tensor<Ts...>& NAME(tensor<Ts...>&& lhs, Rhs&& rhs){\
    using config_type = typename tensor<Ts...>::config_type;\
    return gtensor::tensor_operators_selector_t<config_type>::F(std::move(lhs),std::forward<Rhs>(rhs));\
}

//cast
template<typename To, typename Tensor>
auto cast(Tensor&& t){
    using Tensor_ = std::remove_cv_t<std::remove_reference_t<Tensor>>;
    ASSERT_TENSOR(Tensor_);
    return gtensor::tensor_operators_selector_t<typename Tensor_::config_type>::template cast<To>(std::forward<Tensor>(t));
}

//elementwise ternary operator, arguments can be tensor or scalar, shapes must broadcast
//t is condition, u,v are variants
template<typename T, typename U, typename V>
auto where(T&& t, U&& u, V&& v){
    using T_ = std::remove_cv_t<std::remove_reference_t<T>>;
    using U_ = std::remove_cv_t<std::remove_reference_t<U>>;
    using V_ = std::remove_cv_t<std::remove_reference_t<V>>;
    using config_type = typename detail::first_tensor_type_t<T_,U_,V_>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::where(std::forward<T>(t),std::forward<U>(u),std::forward<V>(v));
}

//arithmetic
GTENSOR_UNARY_TENSOR_OPERATOR(operator+,unary_plus);
GTENSOR_UNARY_TENSOR_OPERATOR(operator-,unary_minus);
GTENSOR_BINARY_TENSOR_OPERATOR(operator+,add);
GTENSOR_BINARY_TENSOR_OPERATOR(operator-,sub);
GTENSOR_BINARY_TENSOR_OPERATOR(operator*,mul);
GTENSOR_BINARY_TENSOR_OPERATOR(operator/,div);
GTENSOR_BINARY_TENSOR_OPERATOR(operator%,mod);

//bitwise
GTENSOR_UNARY_TENSOR_OPERATOR(operator~,bitwise_not);
GTENSOR_BINARY_TENSOR_OPERATOR(operator&,bitwise_and);
GTENSOR_BINARY_TENSOR_OPERATOR(operator|,bitwise_or);
GTENSOR_BINARY_TENSOR_OPERATOR(operator^,bitwise_xor);
GTENSOR_BINARY_TENSOR_OPERATOR(operator<<,bitwise_lshift);
GTENSOR_BINARY_TENSOR_OPERATOR(operator>>,bitwise_rshift);

//strict comparison
GTENSOR_BINARY_TENSOR_OPERATOR(equal,equal);
GTENSOR_BINARY_TENSOR_OPERATOR(not_equal,not_equal);
GTENSOR_BINARY_TENSOR_OPERATOR(operator>,greater);
GTENSOR_BINARY_TENSOR_OPERATOR(operator>=,greater_equal);
GTENSOR_BINARY_TENSOR_OPERATOR(operator<,less);
GTENSOR_BINARY_TENSOR_OPERATOR(operator<=,less_equal);

//close comparison
template<typename T, typename U, typename Tol, typename EqualNan = std::false_type>
inline auto isclose(T&& t, U&& u, Tol relative_tolerance, Tol absolute_tolerance, EqualNan equal_nan = EqualNan{}){
    using T_ = std::remove_cv_t<std::remove_reference_t<T>>;
    using U_ = std::remove_cv_t<std::remove_reference_t<U>>;
    using config_type = typename detail::first_tensor_type_t<T_,U_>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::isclose(std::forward<T>(t),std::forward<U>(u),relative_tolerance,absolute_tolerance,equal_nan);
}
template<typename T, typename U, typename EqualNan = std::false_type>
inline auto isclose(T&& t, U&& u, EqualNan equal_nan = EqualNan{}){
    using T_ = std::remove_cv_t<std::remove_reference_t<T>>;
    using U_ = std::remove_cv_t<std::remove_reference_t<U>>;
    using config_type = typename detail::first_tensor_type_t<T_,U_>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::isclose(std::forward<T>(t),std::forward<U>(u),equal_nan);
}

//logical
GTENSOR_UNARY_TENSOR_OPERATOR(operator!,logic_not);
GTENSOR_BINARY_TENSOR_OPERATOR(operator&&,logic_and);
GTENSOR_BINARY_TENSOR_OPERATOR(operator||,logic_or);

//elementwise assignment
template<typename...Ts, typename Rhs>
inline basic_tensor<Ts...>& assign(basic_tensor<Ts...>& lhs, Rhs&& rhs){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::assign(lhs,std::forward<Rhs>(rhs));
}
template<typename...Ts, typename Rhs>
inline tensor<Ts...>& assign(tensor<Ts...>& lhs, Rhs&& rhs){
    using config_type = typename tensor<Ts...>::config_type;
    return gtensor::tensor_operators_selector_t<config_type>::assign(lhs,std::forward<Rhs>(rhs));
}

//elementwise compound assignment
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator+=,assign_add);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator-=,assign_sub);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator*=,assign_mul);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator/=,assign_div);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator%=,assign_mod);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator&=,assign_bitwise_and);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator|=,assign_bitwise_or);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator^=,assign_bitwise_xor);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator<<=,assign_bitwise_lshift);
GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR(operator>>=,assign_bitwise_rshift);

#undef GTENSOR_TENSOR_OPERATOR_FUNCTION
#undef GTENSOR_TENSOR_OPERATOR_COMPOUND_ASSIGNMENT_FUNCTION
#undef GTENSOR_UNARY_TENSOR_OPERATOR
#undef GTENSOR_BINARY_TENSOR_OPERATOR
#undef GTENSOR_COMPOUND_ASSIGNMENT_TENSOR_OPERATOR

}   //end of namespace gtensor
#endif