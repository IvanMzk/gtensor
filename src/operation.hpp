#ifndef OPERATION_HPP_
#define OPERATION_HPP_

#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
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

namespace gtensor{

namespace math{

template<typename T>
struct numeric_traits{
    static constexpr bool is_integral(){return std::is_integral_v<T>;}
    static constexpr bool is_floating_point(){return std::is_floating_point_v<T>;}
    static constexpr bool is_arithmetic(){return is_integral() || is_floating_point();}
    static constexpr bool is_iec559(){return std::numeric_limits<T>::is_iec559;}
    static constexpr bool has_nan(){return is_iec559();}
    static constexpr T lowest(){return std::numeric_limits<T>::lowest();}
    static constexpr T min(){return std::numeric_limits<T>::min();}
    static constexpr T max(){return std::numeric_limits<T>::max();}
    static constexpr T epsilon(){
        static_assert(is_floating_point(),"no sense for non floating_point types");
        return std::numeric_limits<T>::epsilon();
    }
};

template<typename T> auto floor(T t);
//basic
template<typename T> auto abs(T t){return std::abs(t);}
template<typename T, typename U> auto fmod(T t, U u){return std::fmod(t,u);}
template<typename T, typename U> auto remainder(T t, U u){return std::remainder(t,u);}
template<typename T, typename U, typename V> auto fma(T t, U u, V v){return std::fma(t,u,v);}
template<typename T, typename U> auto fmax(T t, U u){return std::fmax(t,u);}
template<typename T, typename U> auto fmin(T t, U u){return std::fmin(t,u);}
template<typename T, typename U> auto fdim(T t, U u){return std::fdim(t,u);}
template<typename T, typename U, typename V> auto clip(T t, U min, V max){
    return t > max ? max : t < min ? min : t;
}
template<typename T, typename U> auto divmod(T num, U denom){
    const auto q = floor(num/denom);
    return std::make_pair(q,num-q*denom);
}
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
//floating point manipulation
template<typename T, typename U> auto ldexp(T t, U u){return std::ldexp(t,u);}
template<typename T, typename U> auto nextafter(T t, U u){return std::nextafter(t,u);}
template<typename T, typename U> auto copysign(T t, U u){return std::copysign(t,u);}
template<typename T> auto frexp(T t){
    int exp{};
    auto m = std::frexp(t,&exp);
    return std::make_pair(m,exp);
}
template<typename T> auto modf(T t){
    T i{};
    auto f = std::modf(t,&i);
    return std::make_pair(i,f);
}
//classification
template<typename T> auto isfinite(T t){return std::isfinite(t);}
template<typename T> auto isinf(T t){return std::isinf(t);}
template<typename T> auto isnan(T t){return std::isnan(t);}
template<typename T> auto isnormal(T t){return std::isnormal(t);}
//comparision
template<typename T, typename U> auto isgreater(T t, U u){return std::isgreater(t,u);}
template<typename T, typename U> auto isgreaterequal(T t, U u){return std::isgreaterequal(t,u);}
template<typename T, typename U> auto isless(T t, U u){return std::isless(t,u);}
template<typename T, typename U> auto islessequal(T t, U u){return std::islessequal(t,u);}
template<typename T, typename U> auto islessgreater(T t, U u){return std::islessgreater(t,u);}
template<typename T, typename U, typename Tol>
auto isclose(T t, U u, const Tol relative_tolerance, const Tol absolute_tolerance){
    static_assert(numeric_traits<T>::is_floating_point() || numeric_traits<U>::is_floating_point(), "math::isclose should be used with floating point types");
    if (t==u){return true;} //infinity or exact
    return math::abs(t-u) < absolute_tolerance + relative_tolerance*(math::abs(t)+math::abs(u));
}
template<typename T, typename U, typename Tol>
auto isclose_nan_equal(T t, U u, const Tol relative_tolerance, const Tol absolute_tolerance){
    static constexpr bool t_has_nan = numeric_traits<T>::has_nan();
    static constexpr bool u_has_nan = numeric_traits<U>::has_nan();
    static_assert(t_has_nan || u_has_nan, "math::isclose_nan_equal should be used with types that has nan");
    if constexpr (t_has_nan && u_has_nan){
        const bool is_nan_u = math::isnan(u);
        return math::isnan(t) ? is_nan_u : (is_nan_u ? false : isclose(t,u,relative_tolerance,absolute_tolerance));
    }else if constexpr (t_has_nan){
        return math::isnan(t) ? false : isclose(t,u,relative_tolerance,absolute_tolerance));
    }else{
        return math::isnan(u) ? false : isclose(t,u,relative_tolerance,absolute_tolerance));
    }
}
template<typename T, typename U, typename Tol>
auto isequal_nan_equal(T t, U u, const Tol relative_tolerance, const Tol absolute_tolerance){
    static constexpr bool t_has_nan = numeric_traits<T>::has_nan();
    static constexpr bool u_has_nan = numeric_traits<U>::has_nan();
    static_assert(t_has_nan || u_has_nan, "math::isequal_nan_equal should be used with types that has nan");
    if constexpr (t_has_nan && u_has_nan){
        const bool is_nan_u = math::isnan(u);
        return math::isnan(t) ? is_nan_u : (is_nan_u ? false : t==u;
    }else if constexpr (t_has_nan){
        return math::isnan(t) ? false : t==u;
    }else{
        return math::isnan(u) ? false : t==u;
    }
}
//routines in rational domain
template<typename T, typename U> auto gcd(T t, U u){return std::gcd(t,u);}
template<typename T, typename U> auto lcm(T t, U u){return std::lcm(t,u);}
}   //end of namespace math

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
GTENSOR_FUNCTION(math_fma,math::fma);
GTENSOR_FUNCTION(math_fmax,math::fmax);
GTENSOR_FUNCTION(math_fmin,math::fmin);
GTENSOR_FUNCTION(math_fdim,math::fdim);
GTENSOR_FUNCTION(math_clip,math::clip);
GTENSOR_FUNCTION(math_divmod,math::divmod);
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
//floating point manipulation
GTENSOR_FUNCTION(math_ldexp,math::ldexp);
GTENSOR_FUNCTION(math_nextafter,math::nextafter);
GTENSOR_FUNCTION(math_copysign,math::copysign);
GTENSOR_FUNCTION(math_frexp,math::frexp);
GTENSOR_FUNCTION(math_modf,math::modf);

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
GTENSOR_FUNCTION(math_isfinite,math::isfinite);
GTENSOR_FUNCTION(math_isinf,math::isinf);
GTENSOR_FUNCTION(math_isnan,math::isnan);
GTENSOR_FUNCTION(math_isnormal,math::isnormal);
//comparison
GTENSOR_FUNCTION(math_isgreater,math::isgreater);
GTENSOR_FUNCTION(math_isgreaterequal,math::isgreaterequal);
GTENSOR_FUNCTION(math_isless,math::isless);
GTENSOR_FUNCTION(math_islessequal,math::islessequal);
GTENSOR_FUNCTION(math_islessgreater,math::islessgreater);

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
//routines in rational domain
GTENSOR_FUNCTION(math_gcd,math::gcd);
GTENSOR_FUNCTION(math_lcm,math::lcm);

}   //end of nemespace operations

}   //end of namespace gtensor
#endif