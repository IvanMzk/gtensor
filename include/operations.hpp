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

#define GTENSOR_FUNCTION_FUNCTOR(NAME, F, RECURSIVE_F)\
struct NAME{\
    template<typename...Args>\
    auto operator()(Args&&...args)const{\
        if constexpr (detail::has_tensor_arg_v<std::decay_t<Args>...>){\
            return RECURSIVE_F(std::forward<Args>(args)...);\
        }else{\
            return F(std::forward<Args>(args)...);\
        }\
    }\
};

namespace gtensor{

namespace operations{

//cast
template<typename> void cast();
template<typename To>
struct cast_operation
{
    template<typename T>
    auto operator()(T&& t)const{
        if constexpr (detail::is_tensor_v<std::decay_t<T>>){
            return cast<To>(std::forward<T>(t));
        }else{
            return static_cast<To>(std::forward<T>(t));
        }
    }
};

//ternary
struct where_operation
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
struct assign{
    template<typename T1, typename T2>
    void operator()(T1&& arg1, T2&& arg2)const{
        using T1_ = std::remove_cv_t<std::remove_reference_t<T1>>;
        if constexpr (detail::is_tensor_v<T1_>){
            std::forward<T1>(arg1).assign(std::forward<T2>(arg2));
        }else{
            std::forward<T1>(arg1) = std::forward<T2>(arg2);
        }
    }
};
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
GTENSOR_FUNCTION_FUNCTOR(math_abs,math::abs,abs);
GTENSOR_FUNCTION_FUNCTOR(math_fmod,math::fmod,fmod);
GTENSOR_FUNCTION_FUNCTOR(math_remainder,math::remainder,remainder);
GTENSOR_FUNCTION_FUNCTOR(math_fma,math::fma,fma);
GTENSOR_FUNCTION_FUNCTOR(math_fmax,math::fmax,fmax);
GTENSOR_FUNCTION_FUNCTOR(math_fmin,math::fmin,fmin);
GTENSOR_FUNCTION_FUNCTOR(math_fdim,math::fdim,fdim);
GTENSOR_FUNCTION_FUNCTOR(math_clip,math::clip,clip);
GTENSOR_FUNCTION_FUNCTOR(math_divmod,math::divmod,divmod);
//exponential
GTENSOR_FUNCTION_FUNCTOR(math_exp,math::exp,exp);
GTENSOR_FUNCTION_FUNCTOR(math_exp2,math::exp2,exp2);
GTENSOR_FUNCTION_FUNCTOR(math_expm1,math::expm1,expm1);
GTENSOR_FUNCTION_FUNCTOR(math_log,math::log,log);
GTENSOR_FUNCTION_FUNCTOR(math_log10,math::log10,log10);
GTENSOR_FUNCTION_FUNCTOR(math_log2,math::log2,log2);
GTENSOR_FUNCTION_FUNCTOR(math_log1p,math::log1p,log1p);
//power
GTENSOR_FUNCTION_FUNCTOR(math_pow,math::pow,pow);
GTENSOR_FUNCTION_FUNCTOR(math_sqrt,math::sqrt,sqrt);
GTENSOR_FUNCTION_FUNCTOR(math_cbrt,math::cbrt,cbrt);
GTENSOR_FUNCTION_FUNCTOR(math_hypot,math::hypot,hypot);
//trigonometric
GTENSOR_FUNCTION_FUNCTOR(math_sin,math::sin,sin);
GTENSOR_FUNCTION_FUNCTOR(math_cos,math::cos,cos);
GTENSOR_FUNCTION_FUNCTOR(math_tan,math::tan,tan);
GTENSOR_FUNCTION_FUNCTOR(math_asin,math::asin,asin);
GTENSOR_FUNCTION_FUNCTOR(math_acos,math::acos,acos);
GTENSOR_FUNCTION_FUNCTOR(math_atan,math::atan,atan);
GTENSOR_FUNCTION_FUNCTOR(math_atan2,math::atan2,atan2);
//hyperbolic
GTENSOR_FUNCTION_FUNCTOR(math_sinh,math::sinh,sinh);
GTENSOR_FUNCTION_FUNCTOR(math_cosh,math::cosh,cosh);
GTENSOR_FUNCTION_FUNCTOR(math_tanh,math::tanh,tanh);
GTENSOR_FUNCTION_FUNCTOR(math_asinh,math::asinh,asinh);
GTENSOR_FUNCTION_FUNCTOR(math_acosh,math::acosh,acosh);
GTENSOR_FUNCTION_FUNCTOR(math_atanh,math::atanh,atanh);
//nearest
GTENSOR_FUNCTION_FUNCTOR(math_ceil,math::ceil,ceil);
GTENSOR_FUNCTION_FUNCTOR(math_floor,math::floor,floor);
GTENSOR_FUNCTION_FUNCTOR(math_trunc,math::trunc,trunc);
GTENSOR_FUNCTION_FUNCTOR(math_round,math::round,round);
GTENSOR_FUNCTION_FUNCTOR(math_nearbyint,math::nearbyint,nearbyint);
GTENSOR_FUNCTION_FUNCTOR(math_rint,math::rint,rint);
//floating point manipulation
GTENSOR_FUNCTION_FUNCTOR(math_ldexp,math::ldexp,ldexp);
GTENSOR_FUNCTION_FUNCTOR(math_nextafter,math::nextafter,nextafter);
GTENSOR_FUNCTION_FUNCTOR(math_copysign,math::copysign,copysign);
GTENSOR_FUNCTION_FUNCTOR(math_frexp,math::frexp,frexp);
GTENSOR_FUNCTION_FUNCTOR(math_modf,math::modf,modf);

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
GTENSOR_FUNCTION_FUNCTOR(math_isfinite,math::isfinite,isfinite);
GTENSOR_FUNCTION_FUNCTOR(math_isinf,math::isinf,isinf);
GTENSOR_FUNCTION_FUNCTOR(math_isnan,math::isnan,isnan);
GTENSOR_FUNCTION_FUNCTOR(math_isnormal,math::isnormal,isnormal);
//comparison
GTENSOR_FUNCTION_FUNCTOR(math_isgreater,math::isgreater,isgreater);
GTENSOR_FUNCTION_FUNCTOR(math_isgreaterequal,math::isgreaterequal,isgreaterequal);
GTENSOR_FUNCTION_FUNCTOR(math_isless,math::isless,isless);
GTENSOR_FUNCTION_FUNCTOR(math_islessequal,math::islessequal,islessequal);
GTENSOR_FUNCTION_FUNCTOR(math_islessgreater,math::islessgreater,islessgreater);

//NanEqual should be std::true_type or std::false_type
template<typename Tol,typename NanEqual = std::false_type>
class math_isclose
{
    Tol relative_tolerance_;
    Tol absolute_tolerance_;
    template<typename T, typename U>
    bool isclose_(const T& t, const U& u)const{
        static constexpr bool is_t_tensor = detail::is_tensor_v<T>;
        static constexpr bool is_u_tensor = detail::is_tensor_v<U>;
        if constexpr (is_t_tensor != is_u_tensor){
            return false;
        }else{
            if constexpr (is_t_tensor){
                return tensor_close(t,u,relative_tolerance_,absolute_tolerance_,NanEqual::value);
            }else if constexpr (NanEqual::value){
                return math::isclose_nan_equal(t,u,relative_tolerance_,absolute_tolerance_);
            }else{
                return math::isclose(t,u,relative_tolerance_,absolute_tolerance_);
            }
        }
    }
public:
    math_isclose(Tol relative_tolerance__, Tol absolute_tolerance__):
        relative_tolerance_{relative_tolerance__},
        absolute_tolerance_{absolute_tolerance__}
        {}
    template<typename T, typename U>
    bool operator()(T t, U u)const{
        return isclose_(t,u);
    }
};

//NanEqual should be std::true_type or std::false_type
template<typename NanEqual = std::false_type>
class math_isequal
{
    template<typename T, typename U>
    bool isequal_(const T& t, const U& u)const{
        static constexpr bool is_t_tensor = detail::is_tensor_v<T>;
        static constexpr bool is_u_tensor = detail::is_tensor_v<U>;
        if constexpr (is_t_tensor != is_u_tensor){
            return false;
        }else{
            if constexpr (is_t_tensor){
                return tensor_equal(t,u,NanEqual::value);
            }else if constexpr (NanEqual::value){
                return math::isequal_nan_equal(t,u);
            }else{
                return t==u;
            }
        }
    }
public:
    template<typename T, typename U>
    bool operator()(T t, U u)const{
        return isequal_(t,u);
    }
};

//routines in rational domain
GTENSOR_FUNCTION_FUNCTOR(math_gcd,math::gcd,gcd);
GTENSOR_FUNCTION_FUNCTOR(math_lcm,math::lcm,lcm);

//complex numbers
GTENSOR_FUNCTION_FUNCTOR(math_real,math::real,real);
GTENSOR_FUNCTION_FUNCTOR(math_imag,math::imag,imag);
GTENSOR_FUNCTION_FUNCTOR(math_conj,math::conj,conj);
GTENSOR_FUNCTION_FUNCTOR(math_angle,math::angle,angle);

}   //end of nemespace operations
}   //end of namespace gtensor

#undef GTENSOR_UNARY_OPERATOR_FUNCTOR
#undef GTENSOR_BINARY_OPERATOR_FUNCTOR
#undef GTENSOR_ASSIGN_OPERATOR_FUNCTOR
#undef GTENSOR_FUNCTION_FUNCTOR

#endif