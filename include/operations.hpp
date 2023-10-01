/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef OPERATIONS_HPP_
#define OPERATIONS_HPP_

#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include "common.hpp"
#include "math.hpp"

#define GTENSOR_UNARY_OPERATOR_FUNCTOR(NAME, OP)\
struct NAME{\
    template<typename T>\
    auto operator()(T&& arg)const{\
        return (OP std::forward<T>(arg));\
    }\
};

#define GTENSOR_BINARY_OPERATOR_FUNCTOR(NAME, OP)\
struct NAME{\
    template<typename T1, typename T2>\
    auto operator()(T1&& arg1, T2&& arg2)const{\
        return (std::forward<T1>(arg1) OP std::forward<T2>(arg2));\
    }\
};

#define GTENSOR_ASSIGN_OPERATOR_FUNCTOR(NAME, OP)\
struct NAME{\
    template<typename T1, typename T2>\
    void operator()(T1&& arg1, T2&& arg2)const{\
        std::forward<T1>(arg1) OP std::forward<T2>(arg2);\
    }\
};

#define GTENSOR_FUNCTION_FUNCTOR(NAME, F)\
struct NAME{\
    template<typename...Args>\
    auto operator()(Args&&...args)const{\
        return F(std::forward<Args>(args)...);\
    }\
};

namespace gtensor{

namespace operations{

//cast
template<typename To>
struct cast
{
    template<typename T>
    To operator()(T t)const{
        return static_cast<To>(t);
    }
};

//ternary
struct where
{
    template<typename T, typename U, typename V>
    auto operator()(T t, U u, V v)const{
        return t ? u : v;
    }
};

//arithmetic
GTENSOR_UNARY_OPERATOR_FUNCTOR(unary_plus,+);
GTENSOR_UNARY_OPERATOR_FUNCTOR(unary_minus,-);
GTENSOR_BINARY_OPERATOR_FUNCTOR(add,+);
GTENSOR_BINARY_OPERATOR_FUNCTOR(sub,-);
GTENSOR_BINARY_OPERATOR_FUNCTOR(mul,*);
GTENSOR_BINARY_OPERATOR_FUNCTOR(div,/);
GTENSOR_BINARY_OPERATOR_FUNCTOR(mod,%);

//bitwise
GTENSOR_UNARY_OPERATOR_FUNCTOR(bitwise_not,~);
GTENSOR_BINARY_OPERATOR_FUNCTOR(bitwise_and,&);
GTENSOR_BINARY_OPERATOR_FUNCTOR(bitwise_or,|);
GTENSOR_BINARY_OPERATOR_FUNCTOR(bitwise_xor,^);
GTENSOR_BINARY_OPERATOR_FUNCTOR(bitwise_lshift,<<);
GTENSOR_BINARY_OPERATOR_FUNCTOR(bitwise_rshift,>>);

//comparison
GTENSOR_BINARY_OPERATOR_FUNCTOR(equal,==);
GTENSOR_BINARY_OPERATOR_FUNCTOR(not_equal,!=);
GTENSOR_BINARY_OPERATOR_FUNCTOR(greater,>);
GTENSOR_BINARY_OPERATOR_FUNCTOR(greater_equal,>=);
GTENSOR_BINARY_OPERATOR_FUNCTOR(less,<);
GTENSOR_BINARY_OPERATOR_FUNCTOR(less_equal,<=);

//logical
GTENSOR_UNARY_OPERATOR_FUNCTOR(logic_not,!);
GTENSOR_BINARY_OPERATOR_FUNCTOR(logic_and,&&);
GTENSOR_BINARY_OPERATOR_FUNCTOR(logic_or,||);

//asignment
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign,=);
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign_add,+=);
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign_sub,-=);
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign_mul,*=);
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign_div,/=);
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign_mod,%=);
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign_bitwise_and,&=);
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign_bitwise_or,|=);
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign_bitwise_xor,^=);
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign_bitwise_lshift,<<=);
GTENSOR_ASSIGN_OPERATOR_FUNCTOR(assign_bitwise_rshift,>>=);

//math
//basic
GTENSOR_FUNCTION_FUNCTOR(math_abs,math::abs);
GTENSOR_FUNCTION_FUNCTOR(math_fmod,math::fmod);
GTENSOR_FUNCTION_FUNCTOR(math_remainder,math::remainder);
GTENSOR_FUNCTION_FUNCTOR(math_fma,math::fma);
GTENSOR_FUNCTION_FUNCTOR(math_fmax,math::fmax);
GTENSOR_FUNCTION_FUNCTOR(math_fmin,math::fmin);
GTENSOR_FUNCTION_FUNCTOR(math_fdim,math::fdim);
GTENSOR_FUNCTION_FUNCTOR(math_clip,math::clip);
GTENSOR_FUNCTION_FUNCTOR(math_divmod,math::divmod);
//exponential
GTENSOR_FUNCTION_FUNCTOR(math_exp,math::exp);
GTENSOR_FUNCTION_FUNCTOR(math_exp2,math::exp2);
GTENSOR_FUNCTION_FUNCTOR(math_expm1,math::expm1);
GTENSOR_FUNCTION_FUNCTOR(math_log,math::log);
GTENSOR_FUNCTION_FUNCTOR(math_log10,math::log10);
GTENSOR_FUNCTION_FUNCTOR(math_log2,math::log2);
GTENSOR_FUNCTION_FUNCTOR(math_log1p,math::log1p);
//power
GTENSOR_FUNCTION_FUNCTOR(math_pow,math::pow);
GTENSOR_FUNCTION_FUNCTOR(math_sqrt,math::sqrt);
GTENSOR_FUNCTION_FUNCTOR(math_cbrt,math::cbrt);
GTENSOR_FUNCTION_FUNCTOR(math_hypot,math::hypot);
//trigonometric
GTENSOR_FUNCTION_FUNCTOR(math_sin,math::sin);
GTENSOR_FUNCTION_FUNCTOR(math_cos,math::cos);
GTENSOR_FUNCTION_FUNCTOR(math_tan,math::tan);
GTENSOR_FUNCTION_FUNCTOR(math_asin,math::asin);
GTENSOR_FUNCTION_FUNCTOR(math_acos,math::acos);
GTENSOR_FUNCTION_FUNCTOR(math_atan,math::atan);
GTENSOR_FUNCTION_FUNCTOR(math_atan2,math::atan2);
//hyperbolic
GTENSOR_FUNCTION_FUNCTOR(math_sinh,math::sinh);
GTENSOR_FUNCTION_FUNCTOR(math_cosh,math::cosh);
GTENSOR_FUNCTION_FUNCTOR(math_tanh,math::tanh);
GTENSOR_FUNCTION_FUNCTOR(math_asinh,math::asinh);
GTENSOR_FUNCTION_FUNCTOR(math_acosh,math::acosh);
GTENSOR_FUNCTION_FUNCTOR(math_atanh,math::atanh);
//nearest
GTENSOR_FUNCTION_FUNCTOR(math_ceil,math::ceil);
GTENSOR_FUNCTION_FUNCTOR(math_floor,math::floor);
GTENSOR_FUNCTION_FUNCTOR(math_trunc,math::trunc);
GTENSOR_FUNCTION_FUNCTOR(math_round,math::round);
GTENSOR_FUNCTION_FUNCTOR(math_nearbyint,math::nearbyint);
GTENSOR_FUNCTION_FUNCTOR(math_rint,math::rint);
//floating point manipulation
GTENSOR_FUNCTION_FUNCTOR(math_ldexp,math::ldexp);
GTENSOR_FUNCTION_FUNCTOR(math_nextafter,math::nextafter);
GTENSOR_FUNCTION_FUNCTOR(math_copysign,math::copysign);
GTENSOR_FUNCTION_FUNCTOR(math_frexp,math::frexp);
GTENSOR_FUNCTION_FUNCTOR(math_modf,math::modf);

template<typename T>
class math_nan_to_num
{
    T nan_;
    T pos_inf_;
    T neg_inf_;
public:
    math_nan_to_num(T nan__, T pos_inf__, T neg_inf__):
        nan_{nan__},
        pos_inf_{pos_inf__},
        neg_inf_{neg_inf__}
    {}
    T operator()(T t)const{
        if (math::isfinite(t)){
            return t;
        }else if (math::isnan(t)){
            return nan_;
        }else{
            return t > T(0) ? pos_inf_ : neg_inf_;
        }
    }
};

//classification
GTENSOR_FUNCTION_FUNCTOR(math_isfinite,math::isfinite);
GTENSOR_FUNCTION_FUNCTOR(math_isinf,math::isinf);
GTENSOR_FUNCTION_FUNCTOR(math_isnan,math::isnan);
GTENSOR_FUNCTION_FUNCTOR(math_isnormal,math::isnormal);
//comparison
GTENSOR_FUNCTION_FUNCTOR(math_isgreater,math::isgreater);
GTENSOR_FUNCTION_FUNCTOR(math_isgreaterequal,math::isgreaterequal);
GTENSOR_FUNCTION_FUNCTOR(math_isless,math::isless);
GTENSOR_FUNCTION_FUNCTOR(math_islessequal,math::islessequal);
GTENSOR_FUNCTION_FUNCTOR(math_islessgreater,math::islessgreater);

//NanEqual should be std::true_type or std::false_type
template<typename Tol,typename NanEqual = std::false_type>
class math_isclose
{
    Tol relative_tolerance_;
    Tol absolute_tolerance_;
    template<typename T, typename U>
    bool isclose_(std::true_type, T t, U u)const{
        return math::isclose_nan_equal(t,u,relative_tolerance_,absolute_tolerance_);
    }
    template<typename T, typename U>
    bool isclose_(std::false_type, T t, U u)const{
        return math::isclose(t,u,relative_tolerance_,absolute_tolerance_);
    }
public:
    math_isclose(Tol relative_tolerance__, Tol absolute_tolerance__):
        relative_tolerance_{relative_tolerance__},
        absolute_tolerance_{absolute_tolerance__}
        {}
    template<typename T, typename U>
    bool operator()(T t, U u)const{
        return isclose_(typename NanEqual::type{}, t, u);
    }
};

//NanEqual should be std::true_type or std::false_type
template<typename NanEqual = std::false_type>
class math_isequal
{
    template<typename T, typename U>
    bool isequal_(std::true_type, T t, U u)const{
        return math::isequal_nan_equal(t,u);
    }
    template<typename T, typename U>
    bool isequal_(std::false_type, T t, U u)const{
        return t==u;
    }
public:
    template<typename T, typename U>
    bool operator()(T t, U u)const{
        return isequal_(typename NanEqual::type{}, t, u);
    }
};

//routines in rational domain
GTENSOR_FUNCTION_FUNCTOR(math_gcd,math::gcd);
GTENSOR_FUNCTION_FUNCTOR(math_lcm,math::lcm);}   //end of nemespace operations

}   //end of namespace gtensor

#undef GTENSOR_UNARY_OPERATOR_FUNCTOR
#undef GTENSOR_BINARY_OPERATOR_FUNCTOR
#undef GTENSOR_ASSIGN_OPERATOR_FUNCTOR
#undef GTENSOR_FUNCTION_FUNCTOR

#endif