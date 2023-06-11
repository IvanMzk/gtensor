#ifndef MATH_HPP_
#define MATH_HPP_
#include "tensor_operators.hpp"

#define GTENSOR_TENSOR_FUNCTION(NAME,F)\
template<typename...Args>\
inline auto NAME(Args&&...args){\
    static_assert(detail::has_tensor_arg_v<std::remove_cv_t<std::remove_reference_t<Args>>...>,"at least one arg must be tensor");\
    return n_operator(F{},std::forward<Args>(args)...);\
}

namespace gtensor{

namespace detail{
}   //end of namespace detail

//return true if two tensors has same shape and close elements within specified tolerance
template<typename...Us, typename...Vs, typename Tol>
inline auto tensor_close(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, Tol relative_tolerance, Tol absolute_tolerance, bool equal_nan = false){
    using common_value_type = detail::tensor_common_value_type_t<basic_tensor<Us...>,basic_tensor<Vs...>>;
    static_assert(std::is_arithmetic_v<common_value_type>,"routine is defined for arithmetic types only");
    if (u.is_same(v)){
        return true;
    }else{
        const common_value_type relative_tolerance_ = static_cast<common_value_type>(relative_tolerance);
        const common_value_type absolute_tolerance_ = static_cast<common_value_type>(absolute_tolerance);
        const bool equal_shapes = u.shape() == v.shape();
        if (equal_nan){
            return equal_shapes && std::equal(u.begin(), u.end(), v.begin(), operations::math_is_close<common_value_type,std::true_type>{relative_tolerance_,absolute_tolerance_});
        }else{
            return equal_shapes && std::equal(u.begin(), u.end(), v.begin(), operations::math_is_close<common_value_type,std::false_type>{relative_tolerance_,absolute_tolerance_});
        }
    }
}
//return true if two tensors has same shape and close elements, use machine epsilon as tolerance
template<typename...Us, typename...Vs>
inline auto tensor_close(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false){
    using common_value_type = detail::tensor_common_value_type_t<basic_tensor<Us...>,basic_tensor<Vs...>>;
    static_assert(std::is_arithmetic_v<common_value_type>,"routine is defined for arithmetic types only");
    static constexpr common_value_type e = std::numeric_limits<common_value_type>::epsilon();
    return tensor_close(u,v,e,e);
}

//element wise math functions
//basic
GTENSOR_TENSOR_FUNCTION(abs, operations::math_abs);
GTENSOR_TENSOR_FUNCTION(fmod, operations::math_fmod);
GTENSOR_TENSOR_FUNCTION(remainder, operations::math_remainder);
GTENSOR_TENSOR_FUNCTION(fma, operations::math_fma);
GTENSOR_TENSOR_FUNCTION(fmax, operations::math_fmax);
GTENSOR_TENSOR_FUNCTION(fmin, operations::math_fmin);
GTENSOR_TENSOR_FUNCTION(fdim, operations::math_fdim);
//exponential
GTENSOR_TENSOR_FUNCTION(exp, operations::math_exp);
GTENSOR_TENSOR_FUNCTION(exp2, operations::math_exp2);
GTENSOR_TENSOR_FUNCTION(expm1, operations::math_expm1);
GTENSOR_TENSOR_FUNCTION(log, operations::math_log);
GTENSOR_TENSOR_FUNCTION(log10, operations::math_log10);
GTENSOR_TENSOR_FUNCTION(log2, operations::math_log2);
GTENSOR_TENSOR_FUNCTION(log1p, operations::math_log1p);
//power
GTENSOR_TENSOR_FUNCTION(pow, operations::math_pow);
GTENSOR_TENSOR_FUNCTION(sqrt, operations::math_sqrt);
GTENSOR_TENSOR_FUNCTION(cbrt, operations::math_cbrt);
GTENSOR_TENSOR_FUNCTION(hypot, operations::math_hypot);
//trigonometric
GTENSOR_TENSOR_FUNCTION(sin, operations::math_sin);
GTENSOR_TENSOR_FUNCTION(cos, operations::math_cos);
GTENSOR_TENSOR_FUNCTION(tan, operations::math_tan);
GTENSOR_TENSOR_FUNCTION(asin, operations::math_asin);
GTENSOR_TENSOR_FUNCTION(acos, operations::math_acos);
GTENSOR_TENSOR_FUNCTION(atan, operations::math_atan);
GTENSOR_TENSOR_FUNCTION(atan2, operations::math_atan2);
//hyperbolic
GTENSOR_TENSOR_FUNCTION(sinh, operations::math_sinh);
GTENSOR_TENSOR_FUNCTION(cosh, operations::math_cosh);
GTENSOR_TENSOR_FUNCTION(tanh, operations::math_tanh);
GTENSOR_TENSOR_FUNCTION(asinh, operations::math_asinh);
GTENSOR_TENSOR_FUNCTION(acosh, operations::math_acosh);
GTENSOR_TENSOR_FUNCTION(atanh, operations::math_atanh);
//nearest
GTENSOR_TENSOR_FUNCTION(ceil, operations::math_ceil);
GTENSOR_TENSOR_FUNCTION(floor, operations::math_floor);
GTENSOR_TENSOR_FUNCTION(trunc, operations::math_trunc);
GTENSOR_TENSOR_FUNCTION(round, operations::math_round);
GTENSOR_TENSOR_FUNCTION(nearbyint, operations::math_nearbyint);
GTENSOR_TENSOR_FUNCTION(rint, operations::math_rint);
//classification
GTENSOR_TENSOR_FUNCTION(isfinite, operations::math_isfinite);
GTENSOR_TENSOR_FUNCTION(isinf, operations::math_isinf);
GTENSOR_TENSOR_FUNCTION(isnan, operations::math_isnan);
GTENSOR_TENSOR_FUNCTION(isnormal, operations::math_isnormal);
//comparison
GTENSOR_TENSOR_FUNCTION(isgreater, operations::math_isgreater);
GTENSOR_TENSOR_FUNCTION(isgreaterequal, operations::math_isgreaterequal);
GTENSOR_TENSOR_FUNCTION(isless, operations::math_isless);
GTENSOR_TENSOR_FUNCTION(islessequal, operations::math_islessequal);
GTENSOR_TENSOR_FUNCTION(islessgreater, operations::math_islessgreater);
template<typename T, typename U, typename Tol, typename EqualNan = std::false_type>
inline auto is_close(T&& t, U&& u, Tol relative_tolerance, Tol absolute_tolerance, EqualNan equal_nan = EqualNan{}){
    using T_ = std::remove_cv_t<std::remove_reference_t<T>>;
    using U_ = std::remove_cv_t<std::remove_reference_t<U>>;
    static_assert(detail::has_tensor_arg_v<T_,U_>,"at least one arg must be tensor");
    using common_value_type = detail::tensor_common_value_type_t<T_,U_>;
    const common_value_type relative_tolerance_ = static_cast<common_value_type>(relative_tolerance);
    const common_value_type absolute_tolerance_ = static_cast<common_value_type>(absolute_tolerance);
    return n_operator(operations::math_is_close<common_value_type, EqualNan>{relative_tolerance_, absolute_tolerance_}, std::forward<T>(t), std::forward<U>(u));
}
template<typename T, typename U, typename EqualNan = std::false_type>
inline auto is_close(T&& t, U&& u, EqualNan equal_nan = EqualNan{}){
    using T_ = std::remove_cv_t<std::remove_reference_t<T>>;
    using U_ = std::remove_cv_t<std::remove_reference_t<U>>;
    static_assert(detail::has_tensor_arg_v<T_,U_>,"at least one arg must be tensor");
    using common_value_type = detail::tensor_common_value_type_t<T_,U_>;
    static_assert(std::is_arithmetic_v<common_value_type>,"routine is defined for arithmetic types only");
    static constexpr common_value_type e = std::numeric_limits<common_value_type>::epsilon();
    return is_close(std::forward<T>(t),std::forward<U>(u),e,e,equal_nan);
}

}   //end of namespace gtensor
#endif