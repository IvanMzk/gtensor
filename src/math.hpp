#ifndef MATH_HPP_
#define MATH_HPP_

#include <cmath>
#include <cstdlib>
#include <limits>
#include <numeric>
#include "common.hpp"

namespace gtensor{

namespace math{

template<typename, typename> struct default_numeric_traits;

//arbitrary type
template<typename T>
struct default_numeric_traits<T,std::false_type>
{
    static constexpr bool is_specialized(){return false;}
    static constexpr bool is_integral(){return false;}
    static constexpr bool is_floating_point(){return false;}
    static constexpr bool is_iec559(){return false;}
    static constexpr bool has_nan(){return false;}
    static constexpr T lowest(){static_assert(detail::always_false<T>,"not defined");}
    static constexpr T min(){static_assert(detail::always_false<T>,"not defined");}
    static constexpr T max(){static_assert(detail::always_false<T>,"not defined");}
    static constexpr T epsilon(){static_assert(detail::always_false<T>,"not defined");}
    static constexpr T nan(){static_assert(detail::always_false<T>,"not defined");}
};
//arithmetic type
template<typename T>
struct default_numeric_traits<T,std::true_type>
{
    static constexpr bool is_specialized(){return true;}
    static constexpr bool is_integral(){return std::is_integral_v<T>;}
    static constexpr bool is_floating_point(){return std::is_floating_point_v<T>;}
    static constexpr bool is_iec559(){return std::numeric_limits<T>::is_iec559;}
    static constexpr bool has_nan(){return is_iec559();}
    static constexpr T lowest(){return std::numeric_limits<T>::lowest();}
    static constexpr T min(){return std::numeric_limits<T>::min();}
    static constexpr T max(){return std::numeric_limits<T>::max();}
    static constexpr T epsilon(){
        static_assert(is_floating_point(),"not defined for non floating point types");
        return std::numeric_limits<T>::epsilon();
    }
    static constexpr T nan(){
        static_assert(has_nan(),"not defined");
        return std::numeric_limits<T>::quiet_NaN();
    }
};

template<typename T> struct numeric_traits : default_numeric_traits<T,typename std::is_arithmetic<T>::type>
{
    //default types for numeric_traits of arithmetic types
    //may be changed in specialization when other hierarchy of data types is used
    using floating_point_type = double;
    using integral_type = long long int;
};

//floating point type corresponding to T
template<typename T> using make_floating_point_t = std::conditional_t<
    gtensor::math::numeric_traits<T>::is_floating_point(),
    T,
    typename gtensor::math::numeric_traits<T>::floating_point_type
>;
//integral type corresponding to T
template<typename T> using make_integral_t = std::conditional_t<
    gtensor::math::numeric_traits<T>::is_integral(),
    T,
    typename gtensor::math::numeric_traits<T>::integral_type
>;

template<typename T>
struct numeric_constants
{
    static constexpr T pi(){return T{3.141592653589793238463};}
    static constexpr T e(){return T{2.71828182845904523536};}
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
template<typename T> bool isfinite(T t){return std::isfinite(t);}
template<typename T> bool isinf(T t){return std::isinf(t);}
template<typename T> bool isnan(T t){
    if constexpr (numeric_traits<T>::has_nan()){
        return std::isnan(t);
    }else{
        return false;
    }
}
template<typename T> bool isnormal(T t){return std::isnormal(t);}
//comparision
template<typename T, typename U> bool isgreater(T t, U u){return std::isgreater(t,u);}
template<typename T, typename U> bool isgreaterequal(T t, U u){return std::isgreaterequal(t,u);}
template<typename T, typename U> bool isless(T t, U u){return std::isless(t,u);}
template<typename T, typename U> bool islessequal(T t, U u){return std::islessequal(t,u);}
template<typename T, typename U> bool islessgreater(T t, U u){return std::islessgreater(t,u);}
template<typename T, typename U, typename Tol>
bool isclose(T t, U u, const Tol relative_tolerance, const Tol absolute_tolerance){
    if (t==u){return true;} //exact
    return math::abs(t-u) < absolute_tolerance + relative_tolerance*(math::abs(t)+math::abs(u));
}
template<typename T, typename U, typename Tol>
bool isclose_nan_equal(T t, U u, const Tol relative_tolerance, const Tol absolute_tolerance){
    static constexpr bool t_has_nan = numeric_traits<T>::has_nan();
    static constexpr bool u_has_nan = numeric_traits<U>::has_nan();
    if constexpr (t_has_nan && u_has_nan){
        const bool is_nan_u = math::isnan(u);
        return math::isnan(t) ? is_nan_u : (is_nan_u ? false : isclose(t,u,relative_tolerance,absolute_tolerance));
    }else if constexpr (t_has_nan){
        return math::isnan(t) ? false : isclose(t,u,relative_tolerance,absolute_tolerance);
    }else if constexpr (u_has_nan){
        return math::isnan(u) ? false : isclose(t,u,relative_tolerance,absolute_tolerance);
    }else{
        return isclose(t,u,relative_tolerance,absolute_tolerance);
    }
}
template<typename T, typename U>
bool isequal_nan_equal(T t, U u){
    static constexpr bool t_has_nan = numeric_traits<T>::has_nan();
    static constexpr bool u_has_nan = numeric_traits<U>::has_nan();
    if constexpr (t_has_nan && u_has_nan){
        const bool is_nan_u = math::isnan(u);
        return math::isnan(t) ? is_nan_u : (is_nan_u ? false : t==u);
    }else if constexpr (t_has_nan){
        return math::isnan(t) ? false : t==u;
    }else if constexpr (u_has_nan){
        return math::isnan(u) ? false : t==u;
    }else{
        return t==u;
    }
}
//routines in rational domain
template<typename T, typename U> auto gcd(T t, U u){return std::gcd(t,u);}
template<typename T, typename U> auto lcm(T t, U u){return std::lcm(t,u);}
}   //end of namespace math

}   //end of namespace gtensor
#endif