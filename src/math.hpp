#ifndef MATH_HPP_
#define MATH_HPP_
#include "tensor_operators.hpp"
#include "reduce.hpp"

namespace gtensor{

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
            return equal_shapes && std::equal(u.begin(), u.end(), v.begin(), operations::math_isclose<common_value_type,std::true_type>{relative_tolerance_,absolute_tolerance_});
        }else{
            return equal_shapes && std::equal(u.begin(), u.end(), v.begin(), operations::math_isclose<common_value_type,std::false_type>{relative_tolerance_,absolute_tolerance_});
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
GTENSOR_TENSOR_FUNCTION(clip, operations::math_clip);
GTENSOR_TENSOR_FUNCTION(divmod, operations::math_divmod);
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
//floating point manipulation
GTENSOR_TENSOR_FUNCTION(frexp,operations::math_frexp);
GTENSOR_TENSOR_FUNCTION(ldexp,operations::math_ldexp);
GTENSOR_TENSOR_FUNCTION(modf,operations::math_modf);
GTENSOR_TENSOR_FUNCTION(nextafter,operations::math_nextafter);
GTENSOR_TENSOR_FUNCTION(copysign,operations::math_copysign);
template<typename T>
inline auto nan_to_num(
    T&& t,
    typename std::remove_cv_t<std::remove_reference_t<T>>::value_type nan = 0,
    typename std::remove_cv_t<std::remove_reference_t<T>>::value_type pos_inf = std::numeric_limits<typename std::remove_cv_t<std::remove_reference_t<T>>::value_type>::max(),
    typename std::remove_cv_t<std::remove_reference_t<T>>::value_type neg_inf = std::numeric_limits<typename std::remove_cv_t<std::remove_reference_t<T>>::value_type>::min()
)
{
    ASSERT_TENSOR(std::remove_cv_t<std::remove_reference_t<T>>);
    using value_type = typename std::remove_cv_t<std::remove_reference_t<T>>::value_type;
    return n_operator(operations::math_nan_to_num<value_type>{nan,pos_inf,neg_inf}, std::forward<T>(t));
}
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
inline auto isclose(T&& t, U&& u, Tol relative_tolerance, Tol absolute_tolerance, EqualNan equal_nan = EqualNan{}){
    using T_ = std::remove_cv_t<std::remove_reference_t<T>>;
    using U_ = std::remove_cv_t<std::remove_reference_t<U>>;
    static_assert(detail::has_tensor_arg_v<T_,U_>,"at least one arg must be tensor");
    using common_value_type = detail::tensor_common_value_type_t<T_,U_>;
    const common_value_type relative_tolerance_ = static_cast<common_value_type>(relative_tolerance);
    const common_value_type absolute_tolerance_ = static_cast<common_value_type>(absolute_tolerance);
    return n_operator(operations::math_isclose<common_value_type, EqualNan>{relative_tolerance_, absolute_tolerance_}, std::forward<T>(t), std::forward<U>(u));
}
template<typename T, typename U, typename EqualNan = std::false_type>
inline auto isclose(T&& t, U&& u, EqualNan equal_nan = EqualNan{}){
    using T_ = std::remove_cv_t<std::remove_reference_t<T>>;
    using U_ = std::remove_cv_t<std::remove_reference_t<U>>;
    static_assert(detail::has_tensor_arg_v<T_,U_>,"at least one arg must be tensor");
    using common_value_type = detail::tensor_common_value_type_t<T_,U_>;
    static_assert(std::is_arithmetic_v<common_value_type>,"routine is defined for arithmetic types only");
    static constexpr common_value_type e = std::numeric_limits<common_value_type>::epsilon();
    return isclose(std::forward<T>(t),std::forward<U>(u),e,e,equal_nan);
}
//routines in rational domain
GTENSOR_TENSOR_FUNCTION(gcd,operations::math_gcd);
GTENSOR_TENSOR_FUNCTION(lcm,operations::math_lcm);

namespace math_reduce_operations{

struct all
{
    template<typename It>
    auto operator()(It first, It last){
        return std::all_of(first,last,[](const auto& e){return static_cast<bool>(e);});
    }
};

struct any
{
    template<typename It>
    auto operator()(It first, It last){
        return std::any_of(first,last,[](const auto& e){return static_cast<bool>(e);});
    }
};

struct amin
{
    template<typename It>
    auto operator()(It first, It last){
        return *std::min_element(first,last);;
    }
};

struct amax
{
    template<typename It>
    auto operator()(It first, It last){
        return *std::max_element(first,last);;
    }
};

struct sum
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        value_type sum{0};
        for(;first!=last; ++first){
            sum+=*first;
        }
        return sum;
    }
};

struct prod
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        value_type prod{1};
        for(;first!=last; ++first){
            prod*=*first;
        }
        return prod;
    }
};

struct cumsum{
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It, DstIt dfirst, DstIt dlast, IdxT,IdxT){
        auto cumsum_ = *first;
        *dfirst = cumsum_;
        for(++dfirst,++first; dfirst!=dlast; ++dfirst,++first){
            cumsum_+=*first;
            *dfirst = cumsum_;
        }
    }
};

struct cumprod{
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It, DstIt dfirst, DstIt dlast, IdxT,IdxT){
        auto cumprod_ = *first;
        *dfirst = cumprod_;
        for(++dfirst,++first; dfirst!=dlast; ++dfirst,++first){
            cumprod_*=*first;
            *dfirst = cumprod_;
        }
    }
};

struct diff_1{
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It, DstIt dfirst, DstIt dlast, const IdxT&, const IdxT&){
        for (;dfirst!=dlast;++dfirst){
            auto prev = *first;
            *dfirst = *(++first) - prev;
        }
    }
};

struct diff_2{
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It, DstIt dfirst, DstIt dlast, const IdxT&, const IdxT&){
        for (;dfirst!=dlast;++dfirst){
            auto v0 = *first;
            auto v1 = *(++first);
            auto v2 = *(++first);
            *dfirst = v2-v1-v1+v0;
            --first;
        }
    }
};

};

//math functions along given axis or axes
//axes may be scalar or container if multiple axes permitted
//empty container means apply function along all axes

//test if all elements along given axes evaluate to true
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto all(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::all{},keep_dims);
}
template<typename...Ts>
auto all(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::all{},keep_dims);
}
//all along all axes
template<typename...Ts>
auto all(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::all{},keep_dims);
}

//test if any of elements along given axes evaluate to true
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto any(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::any{},keep_dims);
}
template<typename...Ts>
auto any(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::any{},keep_dims);
}
//any along all axes
template<typename...Ts>
auto any(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::any{},keep_dims);
}

//min element along given axes
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto amin(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::amin{},keep_dims);
}
template<typename...Ts>
auto amin(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::amin{},keep_dims);
}
//amin along all axes
template<typename...Ts>
auto amin(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::amin{},keep_dims);
}

//max element along given axes
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto amax(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::amax{},keep_dims);
}
template<typename...Ts>
auto amax(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::amax{},keep_dims);
}
//amax along all axes
template<typename...Ts>
auto amax(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::amax{},keep_dims);
}

//sum elements along given axes
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto sum(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::sum{},keep_dims);
}
template<typename...Ts>
auto sum(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::sum{},keep_dims);
}
//sum along all axes
template<typename...Ts>
auto sum(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::sum{},keep_dims);
}

//multiply elements along given axes
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto prod(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::prod{},keep_dims);
}
template<typename...Ts>
auto prod(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::prod{},keep_dims);
}
//prod along all axes
template<typename...Ts>
auto prod(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::prod{},keep_dims);
}

//cumulative sum along given axis
//axis is scalar
template<typename...Ts, typename DimT>
auto cumsum(const basic_tensor<Ts...>& t, const DimT& axis){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return slide(t,axis,math_reduce_operations::cumsum{}, window_size, window_step);
}
//cumsum along all axes
template<typename...Ts>
auto cumsum(const basic_tensor<Ts...>& t){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return slide(t,math_reduce_operations::cumsum{}, window_size, window_step);
}

//cumulative product along given axis
//axis is scalar
template<typename...Ts, typename DimT>
auto cumprod(const basic_tensor<Ts...>& t, const DimT& axis){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return slide(t,axis,math_reduce_operations::cumprod{}, window_size, window_step);
}
//cumprod along all axes
template<typename...Ts>
auto cumprod(const basic_tensor<Ts...>& t){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return slide(t,math_reduce_operations::cumprod{}, window_size, window_step);
}

//n-th difference along given axis
//axis is scalar, default is last axis
template<typename...Ts>
auto diff(const basic_tensor<Ts...>& t, std::size_t n = 1, const typename basic_tensor<Ts...>::dim_type axis = -1){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 2;
    const index_type window_step = 1;
    if (n==0){
        return t;
    }else{
        auto res = slide(t, axis, math_reduce_operations::diff_1{}, window_size, window_step);
        return diff(res, --n, axis);
    }
}
//more efficient, none recursive implementation of second differences
template<typename...Ts>
auto diff2(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::dim_type axis = -1){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 3;
    const index_type window_step = 1;
    return slide(t, axis, math_reduce_operations::diff_2{}, window_size, window_step);
}



}   //end of namespace gtensor
#endif