#ifndef OPERATION_HPP_
#define OPERATION_HPP_

#include <cmath>
#include <cstdlib>
#include <limits>
#include "common.hpp"

#define GTENSOR_UNARY_OPERATION(NAME, OP)\
struct NAME{\
    template<typename T>\
    auto operator()(T&& arg)const{\
        return (OP std::forward<T>(arg));\
    }\
};

#define GTENSOR_BINARY_OPERATION(NAME, OP)\
struct NAME{\
    template<typename T1, typename T2>\
    auto operator()(T1&& arg1, T2&& arg2)const{\
        return (std::forward<T1>(arg1) OP std::forward<T2>(arg2));\
    }\
};

#define GTENSOR_ASSIGN_OPERATION(NAME, OP)\
struct NAME{\
    template<typename T1, typename T2>\
    void operator()(T1&& arg1, T2&& arg2)const{\
        std::forward<T1>(arg1) OP std::forward<T2>(arg2);\
    }\
};

#define GTENSOR_FUNCTION(NAME, F)\
struct NAME{\
    template<typename...Args>\
    auto operator()(Args&&...args)const{\
        return F(std::forward<Args>(args)...);\
    }\
};

#define GTENSOR_OPERATION_TAG(NAME) struct NAME{};

namespace gtensor{

namespace math{
//basic
template<typename T> auto abs(T t){return std::abs(t);}
template<typename T, typename U> auto fmod(T t, U u){return std::fmod(t,u);}
template<typename T, typename U> auto remainder(T t, U u){return std::remainder(t,u);}
template<typename T, typename U> auto fmax(T t, U u){return std::fmax(t,u);}
template<typename T, typename U> auto fmin(T t, U u){return std::fmin(t,u);}
template<typename T, typename U> auto fdim(T t, U u){return std::fdim(t,u);}
//exponential
template<typename T> auto exp(T t){return std::exp(t);}
template<typename T> auto exp2(T t){return std::exp2(t);}
template<typename T> auto expm1(T t){return std::expm1(t);}
template<typename T> auto log(T t){return std::log(t);}
template<typename T> auto log10(T t){return std::log10(t);}
template<typename T> auto log2(T t){return std::log2(t);}
template<typename T> auto log1p(T t){return std::log1p(t);}
//power
template<typename T, typename U> auto pow(T t, U u){return std::pow(t,u);}
template<typename T> auto sqrt(T t){return std::sqrt(t);}
template<typename T> auto cbrt(T t){return std::cbrt(t);}
template<typename T, typename U> auto hypot(T t, U u){return std::hypot(t,u);}
//trigonometric
template<typename T> auto sin(T t){return std::sin(t);}
template<typename T> auto cos(T t){return std::cos(t);}
template<typename T> auto tan(T t){return std::tan(t);}
template<typename T> auto asin(T t){return std::asin(t);}
template<typename T> auto acos(T t){return std::acos(t);}
template<typename T> auto atan(T t){return std::atan(t);}
template<typename T, typename U> auto atan2(T t, U u){return std::atan2(t,u);}
//hyperbolic
template<typename T> auto sinh(T t){return std::sinh(t);}
template<typename T> auto cosh(T t){return std::cosh(t);}
template<typename T> auto tanh(T t){return std::tanh(t);}
template<typename T> auto asinh(T t){return std::asinh(t);}
template<typename T> auto acosh(T t){return std::acosh(t);}
template<typename T> auto atanh(T t){return std::atanh(t);}
//nearest
template<typename T> auto ceil(T t){return std::ceil(t);}
template<typename T> auto floor(T t){return std::floor(t);}
template<typename T> auto trunc(T t){return std::trunc(t);}
template<typename T> auto round(T t){return std::round(t);}
template<typename T> auto nearbyint(T t){return std::nearbyint(t);}
template<typename T> auto rint(T t){return std::rint(t);}
//comparision
template<typename T, typename U, typename Tol>
auto is_close(T t, U u, const Tol relative_tolerance, const Tol absolute_tolerance){
    using common_type = std::common_type_t<T,U>;
    static_assert(std::is_arithmetic_v<common_type>,"math::is_close defined for arithmetic types only");
    if constexpr (std::is_floating_point_v<common_type>){
        return math::abs(t-u) <= absolute_tolerance + relative_tolerance*(math::abs(t)+math::abs(u));
    }else{
        return t==u;
    }
}
template<typename T, typename U>
auto is_close(T t, U u){
    using common_type = std::common_type_t<T,U>;
    static_assert(std::is_arithmetic_v<common_type>,"math::is_close defined for arithmetic types only");
    static constexpr common_type e = std::numeric_limits<common_type>::epsilon();
    return is_close(t,u,e,e);
}

}   //end of namespace math

namespace operations{
//arithmetic
GTENSOR_UNARY_OPERATION(unary_plus,+);
GTENSOR_UNARY_OPERATION(unary_minus,-);
GTENSOR_BINARY_OPERATION(add,+);
GTENSOR_BINARY_OPERATION(sub,-);
GTENSOR_BINARY_OPERATION(mul,*);
GTENSOR_BINARY_OPERATION(div,/);
GTENSOR_BINARY_OPERATION(mod,%);

//bitwise
GTENSOR_UNARY_OPERATION(bitwise_not,~);
GTENSOR_BINARY_OPERATION(bitwise_and,&);
GTENSOR_BINARY_OPERATION(bitwise_or,|);
GTENSOR_BINARY_OPERATION(bitwise_xor,^);
GTENSOR_BINARY_OPERATION(bitwise_lshift,<<);
GTENSOR_BINARY_OPERATION(bitwise_rshift,>>);

//comparison
GTENSOR_BINARY_OPERATION(equal,==);
GTENSOR_BINARY_OPERATION(not_equal,!=);
GTENSOR_BINARY_OPERATION(greater,>);
GTENSOR_BINARY_OPERATION(greater_equal,>=);
GTENSOR_BINARY_OPERATION(less,<);
GTENSOR_BINARY_OPERATION(less_equal,<=);

//logical
GTENSOR_UNARY_OPERATION(logic_not,!);
GTENSOR_BINARY_OPERATION(logic_and,&&);
GTENSOR_BINARY_OPERATION(logic_or,||);

//asignment
GTENSOR_ASSIGN_OPERATION(assign,=);
GTENSOR_ASSIGN_OPERATION(assign_add,+=);
GTENSOR_ASSIGN_OPERATION(assign_sub,-=);
GTENSOR_ASSIGN_OPERATION(assign_mul,*=);
GTENSOR_ASSIGN_OPERATION(assign_div,/=);
GTENSOR_ASSIGN_OPERATION(assign_mod,%=);
GTENSOR_ASSIGN_OPERATION(assign_bitwise_and,&=);
GTENSOR_ASSIGN_OPERATION(assign_bitwise_or,|=);
GTENSOR_ASSIGN_OPERATION(assign_bitwise_xor,^=);
GTENSOR_ASSIGN_OPERATION(assign_bitwise_lshift,<<=);
GTENSOR_ASSIGN_OPERATION(assign_bitwise_rshift,>>=);

//math
//basic
GTENSOR_FUNCTION(math_abs,math::abs);
GTENSOR_FUNCTION(math_fmod,math::fmod);
GTENSOR_FUNCTION(math_remainder,math::remainder);
GTENSOR_FUNCTION(math_fmax,math::fmax);
GTENSOR_FUNCTION(math_fmin,math::fmin);
GTENSOR_FUNCTION(math_fdim,math::fdim);
//exponential
GTENSOR_FUNCTION(math_exp,math::exp);
GTENSOR_FUNCTION(math_exp2,math::exp2);
GTENSOR_FUNCTION(math_expm1,math::expm1);
GTENSOR_FUNCTION(math_log,math::log);
GTENSOR_FUNCTION(math_log10,math::log10);
GTENSOR_FUNCTION(math_log2,math::log2);
GTENSOR_FUNCTION(math_log1p,math::log1p);
//power
GTENSOR_FUNCTION(math_pow,math::pow);
GTENSOR_FUNCTION(math_sqrt,math::sqrt);
GTENSOR_FUNCTION(math_cbrt,math::cbrt);
GTENSOR_FUNCTION(math_hypot,math::hypot);
//trigonometric
GTENSOR_FUNCTION(math_sin,math::sin);
GTENSOR_FUNCTION(math_cos,math::cos);
GTENSOR_FUNCTION(math_tan,math::tan);
GTENSOR_FUNCTION(math_asin,math::asin);
GTENSOR_FUNCTION(math_acos,math::acos);
GTENSOR_FUNCTION(math_atan,math::atan);
GTENSOR_FUNCTION(math_atan2,math::atan2);
//hyperbolic
GTENSOR_FUNCTION(math_sinh,math::sinh);
GTENSOR_FUNCTION(math_cosh,math::cosh);
GTENSOR_FUNCTION(math_tanh,math::tanh);
GTENSOR_FUNCTION(math_asinh,math::asinh);
GTENSOR_FUNCTION(math_acosh,math::acosh);
GTENSOR_FUNCTION(math_atanh,math::atanh);
//nearest
GTENSOR_FUNCTION(math_ceil,math::ceil);
GTENSOR_FUNCTION(math_floor,math::floor);
GTENSOR_FUNCTION(math_trunc,math::trunc);
GTENSOR_FUNCTION(math_round,math::round);
GTENSOR_FUNCTION(math_nearbyint,math::nearbyint);
GTENSOR_FUNCTION(math_rint,math::rint);
//comparison
GTENSOR_FUNCTION(math_is_close,math::is_close);
template<typename Tol>
class math_is_close_tol
{
    Tol relative_tolerance_;
    Tol absolute_tolerance_;
public:
    math_is_close_tol(Tol relative_tolerance__, Tol absolute_tolerance__):
        relative_tolerance_{relative_tolerance__},
        absolute_tolerance_{absolute_tolerance__}
        {}
    template<typename T, typename U>
    bool operator()(T t, U u){
        return math::is_close(t,u,relative_tolerance_,absolute_tolerance_);
    }
};

}   //end of nemespace operations
}   //end of namespace gtensor
#endif