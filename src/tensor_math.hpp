#ifndef TENSOR_MATH_HPP_
#define TENSOR_MATH_HPP_
#include <functional>
#include <algorithm>
#include "tensor_operators.hpp"
#include "reduce.hpp"
#include "reduce_operations.hpp"


namespace gtensor{

namespace detail{

template<typename DefaultInitial, typename Initial>
auto make_initial(const DefaultInitial& default_initial, const Initial& initial){
    if constexpr (std::is_same_v<gtensor::detail::no_value,Initial>){
        return default_initial;
    }else{
        return initial;
    }
}

}   //end of namespace detail

//tensor math implementation

#define GTENSOR_TENSOR_MATH_FUNCTION(NAME,F)\
template<typename...Args>\
static auto NAME(Args&&...args){\
    static_assert(detail::has_tensor_arg_v<std::remove_cv_t<std::remove_reference_t<Args>>...>,"at least one arg must be tensor");\
    return n_operator(F{},std::forward<Args>(args)...);\
}

#define GTENSOR_TENSOR_MATH_REDUCE_FUNCTION(NAME,POLICY,BINARY_F,RANGE_F,ANY_ORDER,INITIAL)\
template<typename Policy, typename...Ts, typename Axes>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){\
    return reduce(policy,t,axes,BINARY_F{},RANGE_F{},keep_dims,ANY_ORDER,INITIAL);\
}\
template<typename Policy, typename...Ts>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims){\
    return reduce_flatten(policy,t,BINARY_F{},RANGE_F{},keep_dims,ANY_ORDER,INITIAL);\
}\
template<typename...Ts, typename Axes>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){\
    return reduce(POLICY{},t,axes,BINARY_F{},RANGE_F{},keep_dims,ANY_ORDER,INITIAL);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims){\
    return reduce_flatten(POLICY{},t,BINARY_F{},RANGE_F{},keep_dims,ANY_ORDER,INITIAL);\
}

#define GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(NAME,POLICY,BINARY_F,RANGE_F,ANY_ORDER,INITIAL)\
template<typename Policy, typename...Ts, typename Axes, typename Initial>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims, const Initial& initial){\
    return reduce(policy,t,axes,BINARY_F{},RANGE_F{},keep_dims,ANY_ORDER,detail::make_initial(INITIAL,initial));\
}\
template<typename Policy, typename...Ts, typename Initial>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims, const Initial& initial){\
    return reduce_flatten(policy,t,BINARY_F{},RANGE_F{},keep_dims,ANY_ORDER,detail::make_initial(INITIAL,initial));\
}\
template<typename...Ts, typename Axes, typename Initial>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims, const Initial& initial){\
    return reduce(POLICY{},t,axes,BINARY_F{},RANGE_F{},keep_dims,ANY_ORDER,detail::make_initial(INITIAL,initial));\
}\
template<typename...Ts, typename Initial>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims, const Initial& initial){\
    return reduce_flatten(POLICY{},t,BINARY_F{},RANGE_F{},keep_dims,ANY_ORDER,detail::make_initial(INITIAL,initial));\
}

#define GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(NAME,POLICY,F)\
template<typename Policy, typename...Ts, typename DimT>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis){\
    using index_type = typename basic_tensor<Ts...>::index_type;\
    using value_type = typename basic_tensor<Ts...>::value_type;\
    const index_type window_size = 1;\
    const index_type window_step = 1;\
    return slide<value_type>(policy,t,axis,F{}, window_size, window_step);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t){\
    using index_type = typename basic_tensor<Ts...>::index_type;\
    using value_type = typename basic_tensor<Ts...>::value_type;\
    const index_type window_size = 1;\
    const index_type window_step = 1;\
    return slide_flatten<value_type>(t,F{}, window_size, window_step);\
}\
template<typename...Ts, typename DimT>\
static auto NAME(const basic_tensor<Ts...>& t, const DimT& axis){\
    using index_type = typename basic_tensor<Ts...>::index_type;\
    using value_type = typename basic_tensor<Ts...>::value_type;\
    const index_type window_size = 1;\
    const index_type window_step = 1;\
    return slide<value_type>(POLICY{},t,axis,F{}, window_size, window_step);\
}

struct tensor_math
{
    //elementwise math functions
    //basic
    GTENSOR_TENSOR_MATH_FUNCTION(abs, operations::math_abs);
    GTENSOR_TENSOR_MATH_FUNCTION(fmod, operations::math_fmod);
    GTENSOR_TENSOR_MATH_FUNCTION(remainder, operations::math_remainder);
    GTENSOR_TENSOR_MATH_FUNCTION(fma, operations::math_fma);
    GTENSOR_TENSOR_MATH_FUNCTION(fmax, operations::math_fmax);
    GTENSOR_TENSOR_MATH_FUNCTION(fmin, operations::math_fmin);
    GTENSOR_TENSOR_MATH_FUNCTION(fdim, operations::math_fdim);
    GTENSOR_TENSOR_MATH_FUNCTION(clip, operations::math_clip);
    GTENSOR_TENSOR_MATH_FUNCTION(divmod, operations::math_divmod);
    //exponential
    GTENSOR_TENSOR_MATH_FUNCTION(exp, operations::math_exp);
    GTENSOR_TENSOR_MATH_FUNCTION(exp2, operations::math_exp2);
    GTENSOR_TENSOR_MATH_FUNCTION(expm1, operations::math_expm1);
    GTENSOR_TENSOR_MATH_FUNCTION(log, operations::math_log);
    GTENSOR_TENSOR_MATH_FUNCTION(log10, operations::math_log10);
    GTENSOR_TENSOR_MATH_FUNCTION(log2, operations::math_log2);
    GTENSOR_TENSOR_MATH_FUNCTION(log1p, operations::math_log1p);
    //power
    GTENSOR_TENSOR_MATH_FUNCTION(pow, operations::math_pow);
    GTENSOR_TENSOR_MATH_FUNCTION(sqrt, operations::math_sqrt);
    GTENSOR_TENSOR_MATH_FUNCTION(cbrt, operations::math_cbrt);
    GTENSOR_TENSOR_MATH_FUNCTION(hypot, operations::math_hypot);
    //trigonometric
    GTENSOR_TENSOR_MATH_FUNCTION(sin, operations::math_sin);
    GTENSOR_TENSOR_MATH_FUNCTION(cos, operations::math_cos);
    GTENSOR_TENSOR_MATH_FUNCTION(tan, operations::math_tan);
    GTENSOR_TENSOR_MATH_FUNCTION(asin, operations::math_asin);
    GTENSOR_TENSOR_MATH_FUNCTION(acos, operations::math_acos);
    GTENSOR_TENSOR_MATH_FUNCTION(atan, operations::math_atan);
    GTENSOR_TENSOR_MATH_FUNCTION(atan2, operations::math_atan2);
    //hyperbolic
    GTENSOR_TENSOR_MATH_FUNCTION(sinh, operations::math_sinh);
    GTENSOR_TENSOR_MATH_FUNCTION(cosh, operations::math_cosh);
    GTENSOR_TENSOR_MATH_FUNCTION(tanh, operations::math_tanh);
    GTENSOR_TENSOR_MATH_FUNCTION(asinh, operations::math_asinh);
    GTENSOR_TENSOR_MATH_FUNCTION(acosh, operations::math_acosh);
    GTENSOR_TENSOR_MATH_FUNCTION(atanh, operations::math_atanh);
    //nearest
    GTENSOR_TENSOR_MATH_FUNCTION(ceil, operations::math_ceil);
    GTENSOR_TENSOR_MATH_FUNCTION(floor, operations::math_floor);
    GTENSOR_TENSOR_MATH_FUNCTION(trunc, operations::math_trunc);
    GTENSOR_TENSOR_MATH_FUNCTION(round, operations::math_round);
    GTENSOR_TENSOR_MATH_FUNCTION(nearbyint, operations::math_nearbyint);
    GTENSOR_TENSOR_MATH_FUNCTION(rint, operations::math_rint);
    //floating point manipulation
    GTENSOR_TENSOR_MATH_FUNCTION(frexp,operations::math_frexp);
    GTENSOR_TENSOR_MATH_FUNCTION(ldexp,operations::math_ldexp);
    GTENSOR_TENSOR_MATH_FUNCTION(modf,operations::math_modf);
    GTENSOR_TENSOR_MATH_FUNCTION(nextafter,operations::math_nextafter);
    GTENSOR_TENSOR_MATH_FUNCTION(copysign,operations::math_copysign);
    template<typename T>
    static auto nan_to_num(
        T&& t,
        typename std::remove_cv_t<std::remove_reference_t<T>>::value_type nan = 0,
        typename std::remove_cv_t<std::remove_reference_t<T>>::value_type pos_inf = gtensor::math::numeric_traits<typename std::remove_cv_t<std::remove_reference_t<T>>::value_type>::max(),
        typename std::remove_cv_t<std::remove_reference_t<T>>::value_type neg_inf = gtensor::math::numeric_traits<typename std::remove_cv_t<std::remove_reference_t<T>>::value_type>::lowest()
    )
    {
        ASSERT_TENSOR(std::remove_cv_t<std::remove_reference_t<T>>);
        using value_type = typename std::remove_cv_t<std::remove_reference_t<T>>::value_type;
        return n_operator(operations::math_nan_to_num<value_type>{nan,pos_inf,neg_inf}, std::forward<T>(t));
    }
    //classification
    GTENSOR_TENSOR_MATH_FUNCTION(isfinite, operations::math_isfinite);
    GTENSOR_TENSOR_MATH_FUNCTION(isinf, operations::math_isinf);
    GTENSOR_TENSOR_MATH_FUNCTION(isnan, operations::math_isnan);
    GTENSOR_TENSOR_MATH_FUNCTION(isnormal, operations::math_isnormal);
    //comparison
    GTENSOR_TENSOR_MATH_FUNCTION(isgreater, operations::math_isgreater);
    GTENSOR_TENSOR_MATH_FUNCTION(isgreaterequal, operations::math_isgreaterequal);
    GTENSOR_TENSOR_MATH_FUNCTION(isless, operations::math_isless);
    GTENSOR_TENSOR_MATH_FUNCTION(islessequal, operations::math_islessequal);
    GTENSOR_TENSOR_MATH_FUNCTION(islessgreater, operations::math_islessgreater);
    //functions in rational domain
    GTENSOR_TENSOR_MATH_FUNCTION(gcd,operations::math_gcd);
    GTENSOR_TENSOR_MATH_FUNCTION(lcm,operations::math_lcm);

    //math functions along given axis or axes
    //axes may be scalar or container if multiple axes permitted
    //empty container means apply function along all axes

    using default_policy = reduce_auto<1>;

    //test if all elements along given axes evaluate to true
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_FUNCTION(all,default_policy,math_reduce_operations::logical_binary_operation<std::logical_and<void>>,math_reduce_operations::all,true,true);

    //test if any of elements along given axes evaluate to true
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_FUNCTION(any,default_policy,math_reduce_operations::logical_binary_operation<std::logical_or<void>>,math_reduce_operations::any,true,false);

    //min element along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(amin,default_policy,math_reduce_operations::nan_propagate_comparator<std::less<void>>,math_reduce_operations::amin,true,detail::no_value{});

    //max element along given axes
    // //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(amax,default_policy,math_reduce_operations::nan_propagate_comparator<std::greater<void>>,math_reduce_operations::amax,true,detail::no_value{});

    //sum elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(sum,default_policy,math_reduce_operations::nan_propagate_operation<std::plus<void>>,math_reduce_operations::sum,true,typename basic_tensor<Ts...>::value_type{0});

    //multiply elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(prod,default_policy,math_reduce_operations::nan_propagate_operation<std::multiplies<void>>,math_reduce_operations::prod,true,typename basic_tensor<Ts...>::value_type{1});

    //cumulative sum along given axis
    //axis is scalar
    GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(cumsum,default_policy,math_reduce_operations::cumsum);

    //cumulative product along given axis
    //axis is scalar
    GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(cumprod,default_policy,math_reduce_operations::cumprod);

    //nan versions
    //min element along given axes ignoring nan
    //axes may be scalar or container
    //GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nanmin,math_reduce_operations::nan_ignore_comparator<std::less<void>>,gtensor::detail::no_value{});
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nanmin,default_policy,math_reduce_operations::nan_ignore_comparator<std::less<void>>,math_reduce_operations::nanmin,true,detail::no_value{});

    //max element along given axes ignoring nan
    //axes may be scalar or container
    //GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nanmax,math_reduce_operations::nan_ignore_comparator<std::greater<void>>,gtensor::detail::no_value{});
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nanmax,default_policy,math_reduce_operations::nan_ignore_comparator<std::greater<void>>,math_reduce_operations::nanmax,true,detail::no_value{});

    //sum elements along given axes, treating nan as zero
    //axes may be scalar or container
    //GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nansum,math_reduce_operations::nan_ignoring_operation<std::plus<void>>,typename basic_tensor<Ts...>::value_type{0});
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nansum,default_policy,math_reduce_operations::nan_ignoring_operation<std::plus<void>>,math_reduce_operations::nansum,true,typename basic_tensor<Ts...>::value_type{0});

    //multiply elements along given axes, treating nan as one
    //axes may be scalar or container
    //GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nanprod,math_reduce_operations::nan_ignoring_operation<std::multiplies<void>>,typename basic_tensor<Ts...>::value_type{1});
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nanprod,default_policy,math_reduce_operations::nan_ignoring_operation<std::multiplies<void>>,math_reduce_operations::nanprod,true,typename basic_tensor<Ts...>::value_type{1});

    //cumulative sum along given axis, treating nan as zero
    //axis is scalar
    //GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(nancumsum,math_reduce_operations::nancumsum);
    GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(nancumsum,default_policy,math_reduce_operations::nancumsum);

    //cumulative product along given axis, treating nan as one
    //axis is scalar
    //GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(nancumprod,math_reduce_operations::nancumprod);
    GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(nancumprod,default_policy,math_reduce_operations::nancumprod);

    //n-th difference along given axis
    //axis is scalar, default is last axis
    template<typename Policy, typename...Ts, typename DimT>
    static auto diff(Policy policy, const basic_tensor<Ts...>& t, std::size_t n, const DimT& axis){
        using index_type = typename basic_tensor<Ts...>::index_type;
        using value_type = typename basic_tensor<Ts...>::value_type;
        const index_type window_size = 2;
        const index_type window_step = 1;
        if (n==0){
            return t;
        }else{
            auto res = slide<value_type>(policy, t, axis, math_reduce_operations::diff_1{}, window_size, window_step);
            return diff(policy, res, --n, axis);
        }
    }
    //none recursive implementation of second differences, more efficient than diff with n=2
    template<typename Policy, typename...Ts, typename DimT>
    static auto diff2(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis){
        using index_type = typename basic_tensor<Ts...>::index_type;
        using value_type = typename basic_tensor<Ts...>::value_type;
        const index_type window_size = 3;
        const index_type window_step = 1;
        return slide<value_type>(policy, t, axis, math_reduce_operations::diff_2{}, window_size, window_step);
    }
    template<typename...Ts, typename DimT>
    static auto diff(const basic_tensor<Ts...>& t, std::size_t n, const DimT& axis){
        return diff(default_policy{},t,n,axis);
    }
    template<typename...Ts, typename DimT>
    static auto diff2(const basic_tensor<Ts...>& t, const DimT& axis){
        return diff2(default_policy{},t,axis);
    }

    //gradient along given axis, interior points has 2-nd order accuracy approximation using central difference, boundary points has 1-st order accuracy approximation
    //axis is scalar
    //spacing is scalar or container, scalar means uniform sample distance, container specifies coordinates along dimension
    //container must be the same size as size along axis
    template<typename Policy, typename...Ts, typename DimT, typename Spacing>
    static auto gradient(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis, const Spacing& spacing){
        using index_type = typename basic_tensor<Ts...>::index_type;
        using value_type = typename basic_tensor<Ts...>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        const index_type window_size = 1;
        const index_type window_step = 1;
        return gtensor::slide<res_type>(policy, t, axis, math_reduce_operations::gradient{}, window_size, window_step, spacing);
    }
    template<typename...Ts, typename DimT, typename Spacing>
    static auto gradient(const basic_tensor<Ts...>& t, const DimT& axis, const Spacing& spacing){
        return gradient(default_policy{},t,axis,spacing);
    }

};   //end of struct tensor_math

//tensor math frontend
//frontend uses compile-time dispatch to select implementation, see module_selector.hpp

//elementwise tensor math routines
#define GTENSOR_TENSOR_MATH_ROUTINE(NAME,F)\
template<typename...Args>\
auto NAME(Args&&...args){\
    using config_type = typename detail::first_tensor_type_t<std::remove_cv_t<std::remove_reference_t<Args>>...>::config_type;\
    return tensor_math_selector_t<config_type>::F(std::forward<Args>(args)...);\
}

//basic
GTENSOR_TENSOR_MATH_ROUTINE(abs,abs);
GTENSOR_TENSOR_MATH_ROUTINE(fmod,fmod);
GTENSOR_TENSOR_MATH_ROUTINE(remainder,remainder);
GTENSOR_TENSOR_MATH_ROUTINE(fma,fma);
GTENSOR_TENSOR_MATH_ROUTINE(fmax,fmax);
GTENSOR_TENSOR_MATH_ROUTINE(fmin,fmin);
GTENSOR_TENSOR_MATH_ROUTINE(fdim,fdim);
GTENSOR_TENSOR_MATH_ROUTINE(clip,clip);
GTENSOR_TENSOR_MATH_ROUTINE(divmod,divmod);
//exponential
GTENSOR_TENSOR_MATH_ROUTINE(exp,exp);
GTENSOR_TENSOR_MATH_ROUTINE(exp2,exp2);
GTENSOR_TENSOR_MATH_ROUTINE(expm1,expm1);
GTENSOR_TENSOR_MATH_ROUTINE(log,log);
GTENSOR_TENSOR_MATH_ROUTINE(log10,log10);
GTENSOR_TENSOR_MATH_ROUTINE(log2,log2);
GTENSOR_TENSOR_MATH_ROUTINE(log1p,log1p);
//power
GTENSOR_TENSOR_MATH_ROUTINE(pow,pow);
GTENSOR_TENSOR_MATH_ROUTINE(sqrt,sqrt);
GTENSOR_TENSOR_MATH_ROUTINE(cbrt,cbrt);
GTENSOR_TENSOR_MATH_ROUTINE(hypot,hypot);
//trigonometric
GTENSOR_TENSOR_MATH_ROUTINE(sin,sin);
GTENSOR_TENSOR_MATH_ROUTINE(cos,cos);
GTENSOR_TENSOR_MATH_ROUTINE(tan,tan);
GTENSOR_TENSOR_MATH_ROUTINE(asin,asin);
GTENSOR_TENSOR_MATH_ROUTINE(acos,acos);
GTENSOR_TENSOR_MATH_ROUTINE(atan,atan);
GTENSOR_TENSOR_MATH_ROUTINE(atan2,atan2);
//hyperbolic
GTENSOR_TENSOR_MATH_ROUTINE(sinh,sinh);
GTENSOR_TENSOR_MATH_ROUTINE(cosh,cosh);
GTENSOR_TENSOR_MATH_ROUTINE(tanh,tanh);
GTENSOR_TENSOR_MATH_ROUTINE(asinh,asinh);
GTENSOR_TENSOR_MATH_ROUTINE(acosh,acosh);
GTENSOR_TENSOR_MATH_ROUTINE(atanh,atanh);
//nearest
GTENSOR_TENSOR_MATH_ROUTINE(ceil,ceil);
GTENSOR_TENSOR_MATH_ROUTINE(floor,floor);
GTENSOR_TENSOR_MATH_ROUTINE(trunc,trunc);
GTENSOR_TENSOR_MATH_ROUTINE(round,round);
GTENSOR_TENSOR_MATH_ROUTINE(nearbyint,nearbyint);
GTENSOR_TENSOR_MATH_ROUTINE(rint,rint);
//floating point manipulation
GTENSOR_TENSOR_MATH_ROUTINE(frexp,frexp);
GTENSOR_TENSOR_MATH_ROUTINE(ldexp,ldexp);
GTENSOR_TENSOR_MATH_ROUTINE(modf,modf);
GTENSOR_TENSOR_MATH_ROUTINE(nextafter,nextafter);
GTENSOR_TENSOR_MATH_ROUTINE(copysign,copysign);
GTENSOR_TENSOR_MATH_ROUTINE(nan_to_num,nan_to_num);
//classification
GTENSOR_TENSOR_MATH_ROUTINE(isfinite,isfinite);
GTENSOR_TENSOR_MATH_ROUTINE(isinf,isinf);
GTENSOR_TENSOR_MATH_ROUTINE(isnan,isnan);
GTENSOR_TENSOR_MATH_ROUTINE(isnormal,isnormal);
//comparison
GTENSOR_TENSOR_MATH_ROUTINE(isgreater,isgreater);
GTENSOR_TENSOR_MATH_ROUTINE(isgreaterequal,isgreaterequal);
GTENSOR_TENSOR_MATH_ROUTINE(isless,isless);
GTENSOR_TENSOR_MATH_ROUTINE(islessequal,islessequal);
GTENSOR_TENSOR_MATH_ROUTINE(islessgreater,islessgreater);
//functions in rational domain
GTENSOR_TENSOR_MATH_ROUTINE(gcd,gcd);
GTENSOR_TENSOR_MATH_ROUTINE(lcm,lcm);

//tensor math routines along given axis or axes
//axes may be scalar or container if multiple axes permitted

#define GTENSOR_TENSOR_MATH_REDUCE_ROUTINE(NAME,F)\
template<typename...Ts, typename Axes>\
auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(t,axes,keep_dims);\
}\
template<typename...Ts, typename DimT>\
auto NAME(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(t,axes,keep_dims);\
}\
template<typename...Ts>\
auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(t,keep_dims);\
}

#define GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(NAME,F)\
template<typename...Ts, typename Axes, typename Initial = gtensor::detail::no_value>\
auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false, const Initial& initial = Initial{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(t,axes,keep_dims,initial);\
}\
template<typename...Ts, typename DimT, typename Initial = gtensor::detail::no_value>\
auto NAME(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false, const Initial& initial = Initial{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(t,axes,keep_dims,initial);\
}\
template<typename...Ts, typename Initial = gtensor::detail::no_value>\
auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false, const Initial& initial = Initial{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(t,keep_dims,initial);\
}

#define GTENSOR_TENSOR_MATH_CUMULATE_ROUTINE(NAME,F)\
template<typename...Ts, typename DimT>\
auto NAME(const basic_tensor<Ts...>& t, const DimT& axis){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(t,axis);\
}\
template<typename...Ts>\
auto NAME(const basic_tensor<Ts...>& t){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(t);\
}

//test if all elements along given axes evaluate to true
//axes may be scalar or container
GTENSOR_TENSOR_MATH_REDUCE_ROUTINE(all,all);

//test if any of elements along given axes evaluate to true
//axes may be scalar or container
GTENSOR_TENSOR_MATH_REDUCE_ROUTINE(any,any);

//min element along given axes
//axes may be scalar or container
GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(amin,amin);
GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(min,amin);

// //max element along given axes
// //axes may be scalar or container
GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(amax,amax);
GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(max,amax);

//sum elements along given axes
//axes may be scalar or container
GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(sum,sum);

//multiply elements along given axes
//axes may be scalar or container
GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(prod,prod);

//cumulative sum along given axis
//axis is scalar
GTENSOR_TENSOR_MATH_CUMULATE_ROUTINE(cumsum,cumsum);

//cumulative product along given axis
//axis is scalar
GTENSOR_TENSOR_MATH_CUMULATE_ROUTINE(cumprod,cumprod);

//nan versions
//min element along given axes ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(nanmin,nanmin);

//max element along given axes ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(nanmax,nanmax);

//sum elements along given axes, treating nan as zero
//axes may be scalar or container
GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(nansum,nansum);

//multiply elements along given axes, treating nan as one
//axes may be scalar or container
GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE(nanprod,nanprod);

//cumulative sum along given axis, treating nan as zero
//axis is scalar
GTENSOR_TENSOR_MATH_CUMULATE_ROUTINE(nancumsum,nancumsum);

//cumulative product along given axis, treating nan as one
//axis is scalar
GTENSOR_TENSOR_MATH_CUMULATE_ROUTINE(nancumprod,nancumprod);

//n-th difference along given axis
//axis is scalar, default is last axis
template<typename...Ts, typename DimT=int>
auto diff(const basic_tensor<Ts...>& t, std::size_t n = 1, const DimT& axis = -1){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return tensor_math_selector_t<config_type>::diff(t,n,axis);
}
//none recursive implementation of second differences, more efficient than diff with n=2
template<typename...Ts, typename DimT=int>
auto diff2(const basic_tensor<Ts...>& t, const DimT& axis = -1){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return tensor_math_selector_t<config_type>::diff2(t,axis);
}

//gradient along given axis, interior points has 2-nd order accuracy approximation using central difference, boundary points has 1-st order accuracy approximation
//axis is scalar
//spacing is scalar or container, scalar means uniform sample distance, container specifies coordinates along dimension
//container must be the same size as size along axis
template<typename...Ts, typename DimT, typename Spacing>
auto gradient(const basic_tensor<Ts...>& t, const DimT& axis, const Spacing& spacing){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return tensor_math_selector_t<config_type>::gradient(t,axis,spacing);
}
template<typename...Ts, typename DimT>
auto gradient(const basic_tensor<Ts...>& t, const DimT& axis){
    using config_type = typename basic_tensor<Ts...>::config_type;
    using value_type = typename basic_tensor<Ts...>::value_type;
    return tensor_math_selector_t<config_type>::gradient(t,axis,value_type{1});
}

#undef GTENSOR_TENSOR_MATH_FUNCTION
#undef GTENSOR_TENSOR_MATH_REDUCE_FUNCTION
#undef GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION
#undef GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION
#undef GTENSOR_TENSOR_MATH_ROUTINE
#undef GTENSOR_TENSOR_MATH_REDUCE_ROUTINE
#undef GTENSOR_TENSOR_MATH_REDUCE_INITIAL_ROUTINE
#undef GTENSOR_TENSOR_MATH_CUMULATE_ROUTINE

}   //end of namespace gtensor
#endif