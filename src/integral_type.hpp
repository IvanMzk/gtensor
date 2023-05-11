#ifndef INTEGRAL_TYPE_HPP_
#define INTEGRAL_TYPE_HPP_

#include <type_traits>
#include <limits>
#include <stdexcept>
#include <string>

namespace integral_type{

#define INTEGRAL_UNARY_OPERATOR(NAME,OP)\
template<typename U>\
inline integral<U> NAME(const integral<U>& u){\
    return integral<U>{OP u.value()};\
}

#define INTEGRAL_BINARY_OPERATOR(NAME,OP)\
template<typename U>\
inline integral<U> NAME(const integral<U>& l, const integral<U>& r){\
    return integral<U>{l.value() OP r.value()};\
}\
template<typename U, typename Other, std::enable_if_t<detail::enable_other_v<Other,integral<U>>,int> =0>\
inline integral<U> NAME(const integral<U>& l, const Other& v){\
    return l OP integral<U>{v};\
}\
template<typename Other, typename U, std::enable_if_t<detail::enable_other_v<Other,integral<U>>,int> =0>\
inline integral<U> NAME(const Other& v, const integral<U>& r){\
    return integral<U>{v} OP r;\
}

#define INTEGRAL_BINARY_COMPARISON_OPERATOR(NAME,OP)\
template<typename U>\
inline bool NAME(const integral<U>& l, const integral<U>& r){\
    return l.value() OP r.value();\
}\
template<typename U, typename Other, std::enable_if_t<detail::enable_other_v<Other,integral<U>>,int> =0>\
inline bool NAME(const integral<U>& l, const Other& v){\
    return l OP integral<U>{v};\
}\
template<typename Other, typename U, std::enable_if_t<detail::enable_other_v<Other,integral<U>>,int> =0>\
inline bool NAME(const Other& v, const integral<U>& r){\
    return integral<U>{v} OP r;\
}

#define INTEGRAL_BINARY_ASSIGNMENT_MEMBER_OPERATOR(NAME,OP)\
integral& NAME(const integral& rhs){\
    value_ OP rhs.value();\
    return *this;\
}\
template<typename Other, std::enable_if_t<detail::enable_other_v<Other,integral>,int> =0>\
integral& NAME(const Other& rhs){\
    return *this OP integral{rhs};\
}\
template<typename U>\
integral& NAME(const integral<U>& rhs){\
    return *this OP integral{rhs};\
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

template<typename T> class integral;
namespace detail{

template<typename Other, typename Integral> inline constexpr bool enable_other_v = std::is_convertible_v<Other,Integral>;
template<typename U, typename Integral> inline constexpr bool enable_other_v<integral<U>,Integral> = false;

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
    template<typename U, std::enable_if_t<std::is_integral_v<U> && std::is_convertible_v<U,value_type>,int> =0>
    integral(U value__):
        value_(value__) //no strong guarantee
    {
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
    integral(const integral<U>& other):
        integral(other.value())
    {}

    // explicit operator T()const{
    //     return value_;
    // }
    // template<typename U>
    // explicit operator U()const{
    //     return static_cast<U>(value_);
    // }
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
INTEGRAL_BINARY_COMPARISON_OPERATOR(operator==,==);
INTEGRAL_BINARY_COMPARISON_OPERATOR(operator!=,!=);
INTEGRAL_BINARY_COMPARISON_OPERATOR(operator<,<);
INTEGRAL_BINARY_COMPARISON_OPERATOR(operator<=,<=);
INTEGRAL_BINARY_COMPARISON_OPERATOR(operator>,>);
INTEGRAL_BINARY_COMPARISON_OPERATOR(operator>=,>=);
//arithmetic operators
INTEGRAL_UNARY_OPERATOR(operator+,+);
INTEGRAL_UNARY_OPERATOR(operator-,-);
INTEGRAL_BINARY_OPERATOR(operator+,+);
INTEGRAL_BINARY_OPERATOR(operator-,-);
INTEGRAL_BINARY_OPERATOR(operator*,*);
INTEGRAL_BINARY_OPERATOR(operator/,/);
INTEGRAL_BINARY_OPERATOR(operator%,%);
//bitwise operators
INTEGRAL_UNARY_OPERATOR(operator~,~);
INTEGRAL_BINARY_OPERATOR(operator&,&);
INTEGRAL_BINARY_OPERATOR(operator|,|);
INTEGRAL_BINARY_OPERATOR(operator^,^);
INTEGRAL_BINARY_OPERATOR(operator<<,<<);
INTEGRAL_BINARY_OPERATOR(operator>>,>>);

template<typename T>
std::ostream& operator<<(std::ostream& os, const integral<T>& t){
    return os<<t.value();
}

}   //end of namespace integral_type
#endif