#ifndef INTEGRAL_TYPE_HPP_
#define INTEGRAL_TYPE_HPP_

#include <type_traits>
#include <limits>
#include <stdexcept>
#include <string>

namespace gtensor{

template<typename T> class integral;
class integral_exception : public std::runtime_error
{
public:
    explicit integral_exception(const char* what_):
        std::runtime_error(what_)
    {}
    explicit integral_exception(const std::string& what_):
        std::runtime_error(what_)
    {}
};

#define UNARY_INTEGRAL_OPERATOR(NAME,OP)\
template<typename U>\
inline auto NAME(const integral<U>& u){\
    return detail::make_integral(OP u.value());\
}
#define BINARY_INTEGRAL_OPERATOR(NAME,OP)\
template<typename U, typename V>\
inline auto NAME(const integral<U>& u, const integral<V>& v){\
    return detail::make_integral(u.value() OP v.value());\
}
#define BINARY_CMP_INTEGRAL_OPERATOR(NAME,OP)\
template<typename U, typename V>\
inline bool NAME(const integral<U>& u, const integral<V>& v){\
    return u.value() OP v.value();\
}
#define INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(NAME,OP)\
template<typename U>\
integral& NAME(const integral<U>& u){\
    value_ OP u.value();\
    return *this;\
}
#define INTEGRAL_UNARY_PREFIX_MEMBER_OPERATOR(NAME,OP)\
integral& NAME(){\
    OP value_;\
    return *this;\
}
#define INTEGRAL_UNARY_POSTFIX_MEMBER_OPERATOR(NAME,OP)\
integral NAME(int){\
    integral tmp(*this);\
    OP value_;\
    return tmp;\
}

namespace detail{
    template<typename T>
    struct is_signed : public std::is_signed<T>{
        static_assert(std::is_integral_v<T>);
    };
    template<typename T> inline constexpr bool is_signed_v = is_signed<T>::value;

    template<typename T>
    struct make_unsigned : public std::make_unsigned<T>{
        static_assert(std::is_integral_v<T>);
    };
    template<typename T> using make_unsigned_t = typename make_unsigned<T>::type;

    template<typename T>
    struct make_signed : public std::make_signed<T>{
        static_assert(std::is_integral_v<T>);
    };
    template<typename T> using make_signed_t = typename make_signed<T>::type;

    template<typename T>
    inline integral<T> make_integral(const T& value){return integral<T>(value);}
}   //end of namespace detail

template<typename T, typename U>
inline constexpr bool cmp_equal(T t, U u){
    using UT = detail::make_unsigned_t<T>;
    using UU = detail::make_unsigned_t<U>;
    if constexpr (detail::is_signed_v<T> == detail::is_signed_v<U>)
        return t==u;
    else if constexpr (detail::is_signed_v<T>)
        return t<0 ? false : UT(t) == u;
    else
        return u<0 ? false : t == UU(u);
}
template<typename T, typename U>
inline constexpr bool cmp_not_equal(T t, U u){
    return !cmp_equal(t, u);
}
template<typename T, typename U>
inline constexpr bool cmp_less(T t, U u){
    using UT = detail::make_unsigned_t<T>;
    using UU = detail::make_unsigned_t<U>;
    if constexpr (detail::is_signed_v<T> == detail::is_signed_v<U>)
        return t<u;
    else if constexpr (detail::is_signed_v<T>)
        return t<0 ? true : UT(t) < u;
    else
        return u<0 ? false : t < UU(u);
}
template<typename T, typename U>
inline constexpr bool cmp_greater(T t, U u){
    return cmp_less(u, t);
}
template<typename T, typename U>
inline constexpr bool cmp_less_equal(T t, U u){
    return !cmp_greater(t, u);
}
template<typename T, typename U>
inline constexpr bool cmp_greater_equal(T t, U u){
    return !cmp_less(t, u);
}

template<typename T>
class integral
{
    static_assert(std::is_integral_v<T>);
public:
    using value_type = T;
    integral() = default;
    explicit integral(value_type value__):
        value_{value__}
    {}
    template<typename U>
    explicit integral(U value__):
        value_(value__)
    {
        static_assert(std::is_integral_v<U>);
        if constexpr (std::is_unsigned_v<U>){
            if constexpr (cmp_less(std::numeric_limits<value_type>::max(), std::numeric_limits<U>::max())){
                if (cmp_less(std::numeric_limits<value_type>::max(), value__)){ //max < value_
                    throw integral_exception("narrowing conversion");
                }
            }
        }else{
            if constexpr (cmp_less(std::numeric_limits<value_type>::max(), std::numeric_limits<U>::max()) && cmp_less_equal(std::numeric_limits<value_type>::min(), std::numeric_limits<U>::min())){
                if (cmp_less(std::numeric_limits<value_type>::max(), value__) ){ //max < value__
                    throw integral_exception("narrowing conversion");
                }
            }else if constexpr (cmp_greater_equal(std::numeric_limits<value_type>::max(), std::numeric_limits<U>::max()) && cmp_greater(std::numeric_limits<value_type>::min(), std::numeric_limits<U>::min())){
                if (cmp_greater(std::numeric_limits<value_type>::min(), value__)){ //min > value__
                    throw integral_exception("narrowing conversion");
                }
            }else{
                if (cmp_less(std::numeric_limits<value_type>::max(), value__) || cmp_greater(std::numeric_limits<value_type>::min(), value__)){ //max < value__ || min > value__
                    throw integral_exception("narrowing conversion");
                }
            }
        }
    }
    template<typename U>
    explicit integral(const integral<U>& other):
        integral(other.value())
    {}

    explicit operator value_type()const{return value_;}
    value_type value()const{return value_;}

    INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(operator+=,+=);
    INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(operator-=,-=);
    INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(operator*=,*=);
    INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(operator/=,/=);
    INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(operator%=,%=);
    INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(operator&=,&=);
    INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(operator|=,|=);
    INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(operator^=,^=);
    INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(operator<<=,<<=);
    INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(operator>>=,>>=);
    INTEGRAL_UNARY_PREFIX_MEMBER_OPERATOR(operator++,++);
    INTEGRAL_UNARY_PREFIX_MEMBER_OPERATOR(operator--,--);
    INTEGRAL_UNARY_POSTFIX_MEMBER_OPERATOR(operator++,++);
    INTEGRAL_UNARY_POSTFIX_MEMBER_OPERATOR(operator--,--);
private:
    value_type value_;
};

//comparison operators
BINARY_CMP_INTEGRAL_OPERATOR(operator==,==);
BINARY_CMP_INTEGRAL_OPERATOR(operator!=,!=);
BINARY_CMP_INTEGRAL_OPERATOR(operator<,<);
BINARY_CMP_INTEGRAL_OPERATOR(operator<=,<=);
BINARY_CMP_INTEGRAL_OPERATOR(operator>,>);
BINARY_CMP_INTEGRAL_OPERATOR(operator>=,>=);
//arithmetic operators
UNARY_INTEGRAL_OPERATOR(operator+,+);
UNARY_INTEGRAL_OPERATOR(operator-,-);
BINARY_INTEGRAL_OPERATOR(operator+,+);
BINARY_INTEGRAL_OPERATOR(operator-,-);
BINARY_INTEGRAL_OPERATOR(operator*,*);
BINARY_INTEGRAL_OPERATOR(operator/,/);
BINARY_INTEGRAL_OPERATOR(operator%,%);
//bitwise operators
UNARY_INTEGRAL_OPERATOR(operator~,~);
BINARY_INTEGRAL_OPERATOR(operator&,&);
BINARY_INTEGRAL_OPERATOR(operator|,|);
BINARY_INTEGRAL_OPERATOR(operator^,^);
BINARY_INTEGRAL_OPERATOR(operator<<,<<);
BINARY_INTEGRAL_OPERATOR(operator>>,>>);

}   //end of namespace gtensor
#endif