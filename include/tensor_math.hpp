/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef TENSOR_MATH_HPP_
#define TENSOR_MATH_HPP_
#include <functional>
#include <algorithm>
#include <numeric>
#include <immintrin.h>
#include "allocation.hpp"
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

constexpr auto sqrti_helper(std::size_t n, std::size_t l, std::size_t r){
    if (l==r){
        return l;
    }
    const auto mid = ((l+r+1)/2);
    const auto res = mid*mid;
    if (res==n){
        return mid;
    }else if (res > n){
        return sqrti_helper(n,l,mid-1);
    }else{
        return sqrti_helper(n,mid,r);
    }
}
constexpr auto sqrti(std::size_t n){
    return sqrti_helper(n,1,n);
}

//non zero multiple of A, nearest to n, A must be pow of 2
template<std::size_t A>
constexpr std::size_t nearest_multiple(std::size_t n){
    static_assert(A != 0);
    static_assert((A&(A-1))  == 0);
    auto n_ = n & ~(A-1);
    return n%A==0 ? (n==0 ? A : n) : (n_==0 ? A : n_);
}

}   //end of namespace detail

//tensor math implementation

#define GTENSOR_TENSOR_MATH_FUNCTION(NAME,F)\
template<typename...Args>\
static auto NAME(Args&&...args){\
    static_assert(detail::has_tensor_arg_v<std::remove_cv_t<std::remove_reference_t<Args>>...>,"at least one arg must be tensor");\
    return n_operator(F{},std::forward<Args>(args)...);\
}

#define GTENSOR_TENSOR_MATH_REDUCE_FUNCTION(NAME,BINARY_F,RANGE_F,INITIAL)\
template<typename Policy, typename...Ts, typename Axes>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){\
    if constexpr (multithreading::exec_policy_traits<Policy>::is_seq::value){\
        return reduce_binary(policy,t,axes,BINARY_F{},keep_dims,INITIAL);\
    }else{\
        return reduce_range(policy,t,axes,RANGE_F{},keep_dims,true);\
    }\
}\
template<typename Policy, typename...Ts>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims){\
    return reduce_binary(policy,t,detail::no_value{},BINARY_F{},keep_dims,INITIAL);\
}\
template<typename...Ts, typename Axes>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){\
    return NAME(multithreading::exec_pol<1>{},t,axes,keep_dims);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims){\
    return NAME(multithreading::exec_pol<1>{},t,keep_dims);\
}

#define GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(NAME,BINARY_F,RANGE_F,INITIAL)\
template<typename Policy, typename...Ts, typename Axes, typename Initial>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims, const Initial& initial){\
    if constexpr (multithreading::exec_policy_traits<Policy>::is_seq::value){\
        return reduce_binary(policy,t,axes,BINARY_F{},keep_dims,detail::make_initial(INITIAL,initial));\
    }else{\
        return reduce_range(policy,t,axes,RANGE_F{},keep_dims,true,detail::make_initial(INITIAL,initial));\
    }\
}\
template<typename Policy, typename...Ts, typename Initial>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims, const Initial& initial){\
    return reduce_binary(policy,t,detail::no_value{},BINARY_F{},keep_dims,detail::make_initial(INITIAL,initial));\
}\
template<typename...Ts, typename Axes, typename Initial>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims, const Initial& initial){\
    return NAME(multithreading::exec_pol<1>{},t,axes,keep_dims,initial);\
}\
template<typename...Ts, typename Initial>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims, const Initial& initial){\
    return NAME(multithreading::exec_pol<1>{},t,keep_dims,initial);\
}

#define GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(NAME,F)\
template<typename Policy, typename...Ts, typename Axis>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axis& axis){\
    const auto window_size = 1;\
    const auto window_step = 1;\
    return slide<typename gtensor::detail::copy_type_t<basic_tensor<Ts...>>::value_type>(policy,t,axis,F{},window_size,window_step);\
}\
template<typename...Ts, typename Policy>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t){\
    return NAME(policy,t,detail::no_value{});\
}\
template<typename...Ts, typename Axis>\
static auto NAME(const basic_tensor<Ts...>& t, const Axis& axis){\
    return NAME(multithreading::exec_pol<1>{},t,axis);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t){\
    return NAME(multithreading::exec_pol<1>{},t);\
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
    //complex numbers
    GTENSOR_TENSOR_MATH_FUNCTION(real,operations::math_real);
    GTENSOR_TENSOR_MATH_FUNCTION(imag,operations::math_imag);
    GTENSOR_TENSOR_MATH_FUNCTION(conj,operations::math_conj);
    GTENSOR_TENSOR_MATH_FUNCTION(angle,operations::math_angle);

    //math functions along given axis or axes
    //axes may be scalar or container if multiple axes permitted
    //empty container means apply function along all axes

    //test if all elements along given axes evaluate to true
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_FUNCTION(all,math_reduce_operations::logical_binary_operation<std::logical_and<void>>,math_reduce_operations::all,true);

    //test if any of elements along given axes evaluate to true
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_FUNCTION(any,math_reduce_operations::logical_binary_operation<std::logical_or<void>>,math_reduce_operations::any,false);

    //min element along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(amin,math_reduce_operations::nan_propagate_extremum<std::less<void>>,math_reduce_operations::amin,detail::no_value{});

    //max element along given axes
    // //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(amax,math_reduce_operations::nan_propagate_extremum<std::greater<void>>,math_reduce_operations::amax,detail::no_value{});

    //sum elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(sum,math_reduce_operations::nan_propagate_operation<std::plus<void>>,math_reduce_operations::sum,detail::tensor_copy_value_type_t<basic_tensor<Ts...>>(0));

    //multiply elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(prod,math_reduce_operations::nan_propagate_operation<std::multiplies<void>>,math_reduce_operations::prod,detail::tensor_copy_value_type_t<basic_tensor<Ts...>>(1));

    //cumulative sum along given axis
    //axis is scalar
    GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(cumsum,math_reduce_operations::cumsum);

    //cumulative product along given axis
    //axis is scalar
    GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(cumprod,math_reduce_operations::cumprod);

    //nan versions
    //min element along given axes ignoring nan
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nanmin,math_reduce_operations::nan_ignore_extremum<std::less<void>>,math_reduce_operations::nanmin,detail::no_value{});

    //max element along given axes ignoring nan
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nanmax,math_reduce_operations::nan_ignore_extremum<std::greater<void>>,math_reduce_operations::nanmax,detail::no_value{});

    //sum elements along given axes, treating nan as zero
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nansum,math_reduce_operations::nan_ignoring_operation<std::plus<void>>,math_reduce_operations::nansum,detail::tensor_copy_value_type_t<basic_tensor<Ts...>>(0));

    //multiply elements along given axes, treating nan as one
    //axes may be scalar or container
    GTENSOR_TENSOR_MATH_REDUCE_INITIAL_FUNCTION(nanprod,math_reduce_operations::nan_ignoring_operation<std::multiplies<void>>,math_reduce_operations::nanprod,detail::tensor_copy_value_type_t<basic_tensor<Ts...>>(1));

    //cumulative sum along given axis, treating nan as zero
    //axis is scalar
    GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(nancumsum,math_reduce_operations::nancumsum);

    //cumulative product along given axis, treating nan as one
    //axis is scalar
    GTENSOR_TENSOR_MATH_CUMULATE_FUNCTION(nancumprod,math_reduce_operations::nancumprod);

    //n-th difference along given axis
    //axis is scalar, default is last axis
    template<typename Policy, typename...Ts, typename DimT>
    static auto diff(Policy policy, const basic_tensor<Ts...>& t, std::size_t n, const DimT& axis){
        if (n==0){
            return t.copy();
        }else{
            const auto window_size = 2;
            const auto window_step = 1;
            auto res = slide<typename detail::copy_type_t<basic_tensor<Ts...>>::value_type>(policy, t, axis, math_reduce_operations::diff_1{}, window_size, window_step);
            return diff(policy, res, --n, axis);
        }
    }
    //none recursive implementation of second differences, more efficient than diff with n=2
    template<typename Policy, typename...Ts, typename DimT>
    static auto diff2(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis){
        const auto window_size = 3;
        const auto window_step = 1;
        return slide<typename detail::copy_type_t<basic_tensor<Ts...>>::value_type>(policy, t, axis, math_reduce_operations::diff_2{}, window_size, window_step);
    }
    template<typename...Ts, typename DimT>
    static auto diff(const basic_tensor<Ts...>& t, std::size_t n, const DimT& axis){
        return diff(multithreading::exec_pol<1>{},t,n,axis);
    }
    template<typename...Ts, typename DimT>
    static auto diff2(const basic_tensor<Ts...>& t, const DimT& axis){
        return diff2(multithreading::exec_pol<1>{},t,axis);
    }

    //gradient along given axis, interior points has 2-nd order accuracy approximation using central difference, boundary points has 1-st order accuracy approximation
    //axis is scalar
    //spacing is scalar or container, scalar means uniform sample distance, container specifies coordinates along dimension
    //container must be the same size as size along axis
    template<typename Policy, typename...Ts, typename DimT, typename Spacing>
    static auto gradient(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis, const Spacing& spacing){
        using index_type = typename basic_tensor<Ts...>::index_type;
        using value_type = typename basic_tensor<Ts...>::value_type;
        using res_type = gtensor::math::make_floating_point_like_t<value_type>;
        const index_type window_size = 1;
        const index_type window_step = 1;
        return gtensor::slide<res_type>(policy, t, axis, math_reduce_operations::gradient{}, window_size, window_step, spacing);
    }
    template<typename...Ts, typename DimT, typename Spacing>
    static auto gradient(const basic_tensor<Ts...>& t, const DimT& axis, const Spacing& spacing){
        return gradient(multithreading::exec_pol<1>{},t,axis,spacing);
    }

    //matmul
    template<typename Policy, typename...Ts,typename...Us>
    static auto matmul(Policy policy, const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
        using tensor_type1 = basic_tensor<Ts...>;
        using tensor_type2 = basic_tensor<Us...>;
        using value_type1 = typename tensor_type1::value_type;
        using value_type2 = typename tensor_type2::value_type;
        using order1 = typename tensor_type1::order;
        using order2 = typename tensor_type2::order;
        using res_order = std::conditional_t<std::is_same_v<order1,order2>,order1,gtensor::config::c_order>;
        using config_type = typename tensor_type1::config_type;
        using res_type = detail::tensor_copy_type_t<std::decay_t<decltype(std::declval<value_type1>()*std::declval<value_type2>())>,res_order,config_type>;
        using res_value_type = typename res_type::value_type;

        const auto& shape1 = t1.shape();
        const auto& shape2 = t2.shape();
        check_matmul_args(shape1,shape2);
        const auto dim1 = detail::make_dim(shape1);
        const auto dim2 = detail::make_dim(shape2);
        if (dim1==1){
            if (dim2==1){   //(n,) x (n,)
                auto a1 = t1.traverse_order_adapter(order1{});
                auto a2 = t2.traverse_order_adapter(order2{});
                return res_type(std::inner_product(a1.begin(),a1.end(),a2.begin(),res_value_type{0}));
            }else{  //(n,) x (...,n,m)
                return matmul_1d_helper<res_type>(t1,t2,true);
            }
        }else{
            if (dim2==1){   //(...,n,m) x (m,)
                return matmul_1d_helper<res_type>(t2,t1,false);
            }else{  //(...,n,m) x (...,m,k)
                return matmul_nd_helper<res_type>(policy,t1,t2);
            }
        }
    }

    template<typename...Ts,typename...Us>
    static auto matmul(const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
        return matmul(multithreading::exec_pol<1>{},t1,t2);
    }
private:

    template<typename ShT>
    static void check_matmul_args(const ShT& shape1, const ShT& shape2){
        const auto dim1 = detail::make_dim(shape1);
        const auto dim2 = detail::make_dim(shape2);
        if (dim1==0 || dim2==0){
            throw value_error("matmul doesn't support scalar arguments");
        }
        const auto k1 = dim1>1 ? shape1[dim1-1] : shape1[0];
        const auto k2 = dim2>1 ? shape2[dim2-2] : shape2[0];
        if (k1!=k2){
            throw value_error("matmul: tensors shapes not compatible");
        }
    }

    //t_1d.dim()==1, t_nd.dim()>1
    template<typename ResT, typename...Ts, typename...Us>
    static auto matmul_1d_helper(const basic_tensor<Ts...>& t_1d, const basic_tensor<Us...>& t_nd, const bool is_1d_left){
        using res_type = ResT;
        using order = typename ResT::order;
        using config_type = typename ResT::config_type;
        using shape_type = typename ResT::shape_type;
        using order_nd = typename basic_tensor<Us...>::order;

        const auto& t_nd_shape = t_nd.shape();
        const auto t_nd_dim = t_nd.dim();

        shape_type res_shape_(t_nd_dim-1);
        std::copy(t_nd_shape.begin(),t_nd_shape.end()-2,res_shape_.begin());
        *(res_shape_.end()-1) = is_1d_left ? *(t_nd_shape.end()-1) : *(t_nd_shape.end()-2);
        res_type res(std::move(res_shape_),0);

        auto nd_tr = walker_forward_range_traverser<config_type,decltype(t_nd.create_walker())>{t_nd_shape,t_nd.create_walker(),0,t_nd_dim-2};
        auto res_tr = walker_forward_range_traverser<config_type,decltype(res.create_walker())>{res.shape(),res.create_walker(),0,res.dim()-1};
        auto w_1d = t_1d.create_walker();

        const auto res_axis = res.dim()-1;
        const auto inner_axis = is_1d_left ? t_nd_dim-2 : t_nd_dim-1;
        const auto outer_axis = is_1d_left ? t_nd_dim-1 : t_nd_dim-2;
        const auto inner_size = t_nd_shape[inner_axis];
        const auto outer_size = t_nd_shape[outer_axis];

        auto matmul_nd_right_c = [res_axis](auto& res_tr, auto& nd_tr, auto& w_1d, const auto& inner_axis, const auto& outer_axis, const auto& inner_size, const auto& outer_size){
            do{
                auto w_nd = nd_tr.walker();
                auto w_res = res_tr.walker();

                for (auto i=inner_size;;--i){
                    const auto e_1d = *w_1d;
                    for (auto j=outer_size;;--j){
                        decltype(auto) res = *w_res;
                        res=res+*w_nd*e_1d;
                        if (j==1) break;
                        w_nd.step(outer_axis);
                        w_res.step(res_axis);
                    }
                    w_nd.reset_back(outer_axis);
                    w_res.reset_back(res_axis);
                    if (i==1) break;
                    w_nd.step(inner_axis);
                    w_1d.step(0);
                }
                w_1d.reset_back();
                nd_tr.template next<order>();
            }while(res_tr.template next<order>());
        };

        auto matmul_nd_right_f = [res_axis](auto& res_tr, auto& nd_tr, auto& w_1d, const auto& inner_axis, const auto& outer_axis, const auto& inner_size, const auto& outer_size){
            do{
                auto w_nd = nd_tr.walker();
                auto w_res = res_tr.walker();

                for (auto i=outer_size;;--i){
                    decltype(auto) res = *w_res;
                    for (auto j=inner_size;;--j){
                        res=res+*w_nd**w_1d;
                        if (j==1) break;
                        w_nd.step(inner_axis);
                        w_1d.step(0);
                    }
                    w_nd.reset_back(inner_axis);
                    w_1d.reset_back(0);
                    if (i==1) break;
                    w_nd.step(outer_axis);
                    w_res.step(res_axis);
                }
                nd_tr.template next<order>();
            }while(res_tr.template next<order>());
        };

        if constexpr (std::is_same_v<order_nd,gtensor::config::c_order>){
            if (is_1d_left){
                matmul_nd_right_c(res_tr,nd_tr,w_1d,inner_axis,outer_axis,inner_size,outer_size);
            }else{
                matmul_nd_right_f(res_tr,nd_tr,w_1d,inner_axis,outer_axis,inner_size,outer_size);
            }
        }else{
            if (is_1d_left){
                matmul_nd_right_f(res_tr,nd_tr,w_1d,inner_axis,outer_axis,inner_size,outer_size);
            }else{
                matmul_nd_right_c(res_tr,nd_tr,w_1d,inner_axis,outer_axis,inner_size,outer_size);

            }
        }
        return res;
    }

    template<typename T, typename T1, typename T2, typename Config>
    class matmul_2d_blis2
    {
        using index_type = typename Config::index_type;
        using dim_type = typename Config::dim_type;

        static constexpr bool use_common_type = std::is_arithmetic_v<T> && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2> ||
            gtensor::math::is_complex_v<T> &&gtensor::math::is_complex_v<T1> && gtensor::math::is_complex_v<T2>;
        using value_type1 = std::conditional_t<use_common_type,T,T1>;
        using value_type2 = std::conditional_t<use_common_type,T,T2>;
        using res_value_type = T;

        static constexpr bool is_buffer_on_stack = false;
        static constexpr std::size_t alignment = 64;
        static constexpr std::size_t L1_size = 32768;
        static constexpr std::size_t L2_size = 262144;
        static constexpr std::size_t Mc = detail::nearest_multiple<alignment>(detail::sqrti(L2_size/(2*sizeof(value_type1))));
        static constexpr std::size_t Nc = Mc;
        static constexpr std::size_t Kc = L2_size/(Mc*sizeof(value_type1));
        static constexpr std::size_t Nr = L1_size/(Kc*sizeof(value_type2));
        static constexpr std::size_t Mr = Mc;
        static_assert(Mr%alignment==0);

        const index_type k;
        const dim_type i_axis;
        const dim_type j_axis;

        ALWAYS_INLINE auto adjust_block_size(const index_type& idx, const index_type& block_size, const index_type& max){
            return idx+block_size>max ? max-idx : block_size;
        }

        template<typename W, typename U, typename DimT>
        ALWAYS_INLINE void fill_buf(W& w, U* dst, const DimT& inner_axis, const DimT& outer_axis, const index_type& inner_size, const index_type& outer_size, const index_type& block_size){
            index_type ii=0;
            for (;ii<inner_size; ii+=block_size,w.walk(inner_axis,block_size)){
                const auto block_size_ = adjust_block_size(ii,block_size,inner_size);
                for (auto i=outer_size; i!=0; --i,w.step(outer_axis)){
                    const auto dst_last = dst+static_cast<std::ptrdiff_t>(block_size_);
                    if (block_size_ > 3){
                        for (const auto dst_last_=dst_last-3; dst<dst_last_; dst+=4){
                            *dst = *w;
                            w.step(inner_axis);
                            *(dst+1) = *w;
                            w.step(inner_axis);
                            *(dst+2) = *w;
                            w.step(inner_axis);
                            *(dst+3) = *w;
                            w.step(inner_axis);
                        }
                    }
                    for (;dst!=dst_last; ++dst,w.step(inner_axis)){
                        *dst = *w;
                    }
                    w.walk_back(inner_axis,block_size_);
                }
                w.walk_back(outer_axis,outer_size);
            }
            w.walk_back(inner_axis,ii);
        }

        template<typename ResW>
        ALWAYS_INLINE void fill_res(const res_value_type* buf, ResW& res_w, const index_type& mr_, const index_type& nr_){
            for (auto i=nr_; i!=0; --i,res_w.step(j_axis)){
                const auto buf_last = buf+static_cast<std::ptrdiff_t>(mr_);
                if (mr_ > 3){
                    for (const auto buf_last_=buf_last-3; buf<buf_last_; buf+=4){
                        *res_w = *res_w + *buf;
                        res_w.step(i_axis);
                        *res_w = *res_w + *(buf+1);
                        res_w.step(i_axis);
                        *res_w = *res_w + *(buf+2);
                        res_w.step(i_axis);
                        *res_w = *res_w + *(buf+3);
                        res_w.step(i_axis);
                    }
                }
                for (;buf!=buf_last; ++buf,res_w.step(i_axis)){
                    *res_w = *res_w + *buf;
                }
                res_w.walk_back(i_axis,mr_);
            }
            res_w.walk_back(j_axis,nr_);
        }

        template<typename T_, typename T1_, typename T2_>
        ALWAYS_INLINE void micro_kernel_generic(T_* res_buf, const T1_* const a_buf, const T2_* b_buf, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            auto res_buf_ = res_buf;
            for (const auto b_last=b_buf+nr_; b_buf!=b_last; ++b_buf){
                const auto e = *b_buf;
                for (std::ptrdiff_t ir=0; ir!=mr_; ++ir,++res_buf_){
                    *res_buf_=a_buf[ir]*e;
                }
            }
            for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                const auto a_buf_ = a_buf+kk*mr_;
                auto res_buf_ = res_buf;
                for (const auto b_last=b_buf+nr_; b_buf!=b_last; ++b_buf){
                    const auto e = *b_buf;
                    for (std::ptrdiff_t ir=0; ir!=mr_; ++ir,++res_buf_){
                        *res_buf_=*res_buf_+a_buf_[ir]*e;
                    }
                }
            }
        }

        template<typename T_, typename T1_, typename T2_>
        ALWAYS_INLINE void micro_kernel(T_* res_buf, const T1_* const a_buf, const T2_* b_buf, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            micro_kernel_generic(res_buf,a_buf,b_buf,mr_,nr_,kc_);
        }

        template<typename U=T, std::enable_if_t<std::is_same_v<U,U> && HAS_FMA && HAS_AVX && sizeof(double)==8, int> =0>
        ALWAYS_INLINE void micro_kernel(double* res_buf, const double* const a_buf, const double* b_buf, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            micro_kernel_sd(res_buf,a_buf,b_buf,mr_,nr_,kc_);
        }

        template<typename U=T, std::enable_if_t<std::is_same_v<U,U> && HAS_FMA && HAS_AVX && sizeof(float)==4, int> =0>
        ALWAYS_INLINE void micro_kernel(float* res_buf, const float* const a_buf, const float* b_buf, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            micro_kernel_sd(res_buf,a_buf,b_buf,mr_,nr_,kc_);
        }

        template<typename ResW, typename T_, typename T1_, typename T2_>
        ALWAYS_INLINE void macro_kernel(ResW& res_w, T_* res_buf, const T1_* const a_buf, const T2_* b_buf, const std::ptrdiff_t& mc_, const std::ptrdiff_t& nc_, const std::ptrdiff_t& kc_){
            const std::ptrdiff_t mr{Mr};
            const std::ptrdiff_t nr{Nr};
            std::ptrdiff_t i=0;
            for (;i<nc_; i+=nr,res_w.walk(j_axis,nr)){
                const auto nr_ = adjust_block_size(i,nr,nc_);
                auto a_buf_ = a_buf;
                std::ptrdiff_t j=0;
                for (;j<mc_; j+=mr,res_w.walk(i_axis,mr)){
                    const auto mr_ = adjust_block_size(j,mr,mc_);
                    micro_kernel(res_buf,a_buf_,b_buf,mr_,nr_,kc_);
                    fill_res(res_buf,res_w,mr_,nr_);
                    a_buf_+=mr_*kc_;
                }
                b_buf+=kc_*nr_;
                res_w.walk_back(i_axis,j);
            }
            res_w.walk_back(j_axis,i);
        }

        // #define AVX_MATMUL_BROADCAST(Ptr) asm ("vbroadcastsd %0,%%ymm0" ::"m"(*Ptr));

        // #define AVX_MATMUL_MUL(OFFSET)\
        // asm ("vmovapd %1,%%ymm1\n\t"\
        //     "vmulpd %%ymm1,%%ymm0,%%ymm2\n\t"\
        //     "vmovapd %%ymm2,%0"\
        //     :"=m"(*(res_buf_+OFFSET)):"m"(*(a_data+OFFSET))\
        // );

        // #define AVX_MATMUL_FMA(OFFSET)\
        // asm ("vmovapd %2,%%ymm2\n\t"\
        //     "vfmadd231pd %1,%%ymm0,%%ymm2\n\t"\
        //     "vmovapd %%ymm2,%0"\
        //     :"=m"(*(res_buf_+OFFSET)):"m"(*(a_data_+OFFSET)),"m"(*(res_buf_+OFFSET))\
        // );

        //#define AVX_MATMUL_BROADCAST(Ptr) auto b_y = _mm256_broadcast_sd(Ptr);
        // #define AVX_MATMUL_MUL(OFFSET) _mm256_store_pd(res_buf_+OFFSET,_mm256_mul_pd(_mm256_load_pd(a_data+OFFSET),b_y));
        // #define AVX_MATMUL_FMA(OFFSET) _mm256_store_pd(res_buf_+OFFSET,_mm256_fmadd_pd(_mm256_load_pd(a_data_+OFFSET),b_y,_mm256_load_pd(res_buf_+OFFSET)));

        ALWAYS_INLINE auto avx_broadcast(const double* buf){
            return _mm256_broadcast_sd(buf);
        }
        ALWAYS_INLINE auto avx_broadcast(const float* buf){
            return _mm256_broadcast_ss(buf);
        }
        ALWAYS_INLINE auto avx_element(const __m256d& y){
            return _mm256_cvtsd_f64(y);
        }
        ALWAYS_INLINE auto avx_element(const __m256& y){
            return _mm256_cvtss_f32(y);
        }

        ALWAYS_INLINE void avx_mul(double* res_buf, const double* a_buf, const __m256d& b_y,  const std::size_t& offset){
            _mm256_store_pd(res_buf+offset,_mm256_mul_pd(_mm256_load_pd(a_buf+offset),b_y));
        }
        ALWAYS_INLINE void avx_mul(float* res_buf, const float* a_buf, const __m256& b_y,  const std::size_t& offset){
            _mm256_store_ps(res_buf+offset,_mm256_mul_ps(_mm256_load_ps(a_buf+offset),b_y));
        }
        template<std::size_t NPacked, std::size_t...I, typename U, typename P, typename V>
        ALWAYS_INLINE void avx_mul_n(std::index_sequence<I...>, U* res_buf, P* a_buf, const V& b_y){
            (avx_mul(res_buf,a_buf,b_y,I*NPacked),...);
        }

        ALWAYS_INLINE void avx_fma(double* res_buf, const double* a_buf, const __m256d& b_y,  const std::size_t& offset){
            _mm256_store_pd(res_buf+offset,_mm256_fmadd_pd(_mm256_load_pd(a_buf+offset),b_y,_mm256_load_pd(res_buf+offset)));
        }
        ALWAYS_INLINE void avx_fma(float* res_buf, const float* a_buf, const __m256& b_y,  const std::size_t& offset){
            _mm256_store_ps(res_buf+offset,_mm256_fmadd_ps(_mm256_load_ps(a_buf+offset),b_y,_mm256_load_ps(res_buf+offset)));
        }
        template<std::size_t NPacked, std::size_t...I, typename U, typename P, typename V>
        ALWAYS_INLINE void avx_fma_n(std::index_sequence<I...>, U* res_buf, P* a_buf, const V& b_y){
            (avx_fma(res_buf,a_buf,b_y,I*NPacked),...);
        }

        template<typename U>
        ALWAYS_INLINE void micro_kernel_sd(U* res_buf, const U* const a_data, const U* b_data, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            static_assert(std::is_floating_point_v<U>);
            static_assert(sizeof(U)==4 || sizeof(U)==8);
            static constexpr std::size_t n_packed = 32/sizeof(U);
            if (mr_==Mr){   //Mr is guaranteed to be multiple of alignment
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nr_; b_data!=b_last; ++b_data){
                    const auto b_y = avx_broadcast(b_data);
                    avx_mul_n<n_packed>(std::make_index_sequence<Mr/n_packed>{},res_buf_,a_data,b_y);
                    res_buf_+=Mr;
                }
                for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                    const auto a_data_ = a_data+kk*mr_;
                    auto res_buf_ = res_buf;
                    for (const auto b_last=b_data+nr_; b_data!=b_last; ++b_data){
                        const auto b_y = avx_broadcast(b_data);
                        avx_fma_n<n_packed>(std::make_index_sequence<Mr/n_packed>{},res_buf_,a_data_,b_y);
                        res_buf_+=Mr;
                    }
                }
            }else if (mr_>n_packed-1){
                std::ptrdiff_t mm{0};
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nr_; b_data!=b_last; ++b_data){
                    auto a_data_=a_data;
                    const auto a_last = a_data_+mr_;
                    const auto b_y = avx_broadcast(b_data);
                    for (;mm!=0; --mm,++a_data_,++res_buf_){
                        *res_buf_=*a_data_*avx_element(b_y);
                    }
                    for (const auto a_last_=a_last-(n_packed-1); a_data_<a_last_; a_data_+=n_packed,res_buf_+=n_packed){
                        if constexpr (n_packed==4){
                            _mm256_store_pd(res_buf_,_mm256_mul_pd(_mm256_loadu_pd(a_data_),b_y));
                        }else{
                            _mm256_store_ps(res_buf_,_mm256_mul_ps(_mm256_loadu_ps(a_data_),b_y));
                        }
                    }
                    for (mm=n_packed; a_data_!=a_last; ++a_data_,++res_buf_,--mm){
                        *res_buf_=*a_data_*avx_element(b_y);
                    }
                }
                for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                    const auto a_data_ = a_data+kk*mr_;
                    res_buf_ = res_buf;
                    std::ptrdiff_t mm{0};
                    for (const auto b_last=b_data+nr_; b_data!=b_last; ++b_data){
                        auto a_data__=a_data_;
                        const auto a_last = a_data_+mr_;
                        const auto b_y = avx_broadcast(b_data);
                        for (;mm!=0; --mm,++a_data__,++res_buf_){
                            *res_buf_+=*a_data__*avx_element(b_y);
                        }
                        for (const auto a_last__=a_last-(n_packed-1); a_data__<a_last__; a_data__+=n_packed,res_buf_+=n_packed){
                            if constexpr (n_packed==4){
                                _mm256_store_pd(res_buf_,_mm256_fmadd_pd(_mm256_loadu_pd(a_data__),b_y,_mm256_load_pd(res_buf_)));
                            }else{
                                _mm256_store_ps(res_buf_,_mm256_fmadd_ps(_mm256_loadu_ps(a_data__),b_y,_mm256_load_ps(res_buf_)));
                            }
                        }
                        for (mm=n_packed; a_data__!=a_last; ++a_data__,++res_buf_,--mm){
                            *res_buf_+=*a_data__*avx_element(b_y);
                        }
                    }
                }
            }else{
                micro_kernel_generic(res_buf,a_data,b_data,mr_,nr_,kc_);
            }
        }

    public:
        matmul_2d_blis2(const index_type& k_, const dim_type& i_axis_, const dim_type& j_axis_):
            k{k_},
            i_axis{i_axis_},
            j_axis{j_axis_}
        {}

        template<typename ResW, typename W1, typename W2>
        ALWAYS_INLINE void operator()(ResW res_w, W1 w1, W2 w2, const index_type& ic_min, const index_type& ic_max, const index_type& jc_min, const index_type& jc_max, res_value_type* res_buf, value_type1* a_buf, value_type2* b_buf){
            const index_type mc{Mc};
            const index_type nc{Nc};
            const index_type kc{Kc};
            const index_type mr{Mr};
            const index_type nr{Nr};
            for (index_type jc=jc_min; jc<jc_max; jc+=nc){
                w2.walk(j_axis,jc);
                res_w.walk(j_axis,jc);
                const auto nc_ = adjust_block_size(jc,nc,jc_max);
                for (index_type pc=0; pc<k; pc+=kc){
                    w1.walk(j_axis,pc);
                    w2.walk(i_axis,pc);
                    const auto kc_ = adjust_block_size(pc,kc,k);
                    fill_buf(w2,b_buf,j_axis,i_axis,nc_,kc_,nr);
                    for (index_type ic=ic_min; ic<ic_max; ic+=mc){
                        w1.walk(i_axis,ic);
                        res_w.walk(i_axis,ic);
                        const auto mc_ = adjust_block_size(ic,mc,ic_max);
                        fill_buf(w1,a_buf,i_axis,j_axis,mc_,kc_,mr);
                        macro_kernel(res_w,res_buf,a_buf,b_buf,static_cast<std::ptrdiff_t>(mc_),static_cast<std::ptrdiff_t>(nc_),static_cast<std::ptrdiff_t>(kc_));
                        w1.walk_back(i_axis,ic);
                        res_w.walk_back(i_axis,ic);
                    }
                    w1.walk_back(j_axis,pc);
                    w2.walk_back(i_axis,pc);
                }
                w2.walk_back(j_axis,jc);
                res_w.walk_back(j_axis,jc);
            }
        }

        template<typename ResW, typename W1, typename W2>
        ALWAYS_INLINE void operator()(ResW res_w, W1 w1, W2 w2, const index_type& ic_min, const index_type& ic_max, const index_type& jc_min, const index_type& jc_max){
            if constexpr (is_buffer_on_stack){
                alignas(alignment) std::array<res_value_type,Mr*Nr> res_buf;
                alignas(alignment) std::array<value_type1,Mc*Kc> a_buf;
                alignas(alignment) std::array<value_type2,Kc*Nc> b_buf;
                this->operator()(res_w,w1,w2,ic_min,ic_max,jc_min,jc_max,res_buf.data(),a_buf.data(),b_buf.data());
            }else{
                auto make_buf_size = [](auto i_size, auto j_size, auto t_size){
                    return alignment*(i_size*j_size*t_size/alignment+1);
                };
                const auto res_buf_size = make_buf_size(Mr,Nr,sizeof(res_value_type));
                const auto a_buf_size = make_buf_size(Mc,Kc,sizeof(value_type1));
                const auto b_buf_size = make_buf_size(Kc,Nc,sizeof(value_type2));
                if constexpr (std::is_same_v<res_value_type,value_type1> && std::is_same_v<res_value_type,value_type2>){
                    gtensor::basic_storage<res_value_type,allocation::aligned_allocator<res_value_type,alignment>> buf(res_buf_size+a_buf_size+b_buf_size);
                    this->operator()(res_w,w1,w2,ic_min,ic_max,jc_min,jc_max,buf.data(),buf.data()+res_buf_size,buf.data()+res_buf_size+a_buf_size);

                }else{
                    gtensor::basic_storage<res_value_type,allocation::aligned_allocator<res_value_type,alignment>> res_buf(res_buf_size);
                    gtensor::basic_storage<value_type1,allocation::aligned_allocator<value_type1,alignment>> a_buf(a_buf_size);
                    gtensor::basic_storage<value_type2,allocation::aligned_allocator<value_type2,alignment>> b_buf(b_buf_size);
                    this->operator()(res_w,w1,w2,ic_min,ic_max,jc_min,jc_max,res_buf.data(),a_buf.data(),b_buf.data());
                }
            }
        }
    };

    template<typename T, typename T1, typename T2, typename Config>
    class matmul_2d_blis1
    {
        using index_type = typename Config::index_type;
        using dim_type = typename Config::dim_type;

        static constexpr bool use_common_type = std::is_arithmetic_v<T> && std::is_arithmetic_v<T1> && std::is_arithmetic_v<T2> ||
            gtensor::math::is_complex_v<T> &&gtensor::math::is_complex_v<T1> && gtensor::math::is_complex_v<T2>;
        using value_type1 = std::conditional_t<use_common_type,T,T1>;
        using value_type2 = std::conditional_t<use_common_type,T,T2>;
        using res_value_type = T;

        static constexpr bool is_buffer_on_stack = false;
        static constexpr std::size_t alignment = 64;
        static constexpr std::size_t L1_size = 32768;
        static constexpr std::size_t L2_size = 262144;
        static constexpr std::size_t Mc = detail::nearest_multiple<alignment>(detail::sqrti(L2_size/(2*sizeof(value_type1))));
        static constexpr std::size_t Nc = Mc;
        static constexpr std::size_t Kc = L2_size/(Mc*sizeof(value_type1));
        static constexpr std::size_t Nr = L1_size/(Kc*sizeof(value_type2));
        static constexpr std::size_t Mr = Mc;
        static_assert(Mr%alignment==0);


        const index_type k;
        const dim_type i_axis;
        const dim_type j_axis;

        ALWAYS_INLINE auto adjust_block_size(const index_type& idx, const index_type& block_size, const index_type& max){
            return idx+block_size>max ? max-idx : block_size;
        }

        template<typename W, typename U, typename DimT>
        ALWAYS_INLINE void fill_buf(W& w, U* dst, const DimT& inner_axis, const DimT& outer_axis, const index_type& inner_size, const index_type& outer_size, const index_type& block_size){
            index_type ii=0;
            for (;ii<inner_size; ii+=block_size,w.walk(inner_axis,block_size)){
                const auto block_size_ = adjust_block_size(ii,block_size,inner_size);
                for (auto i=outer_size; i!=0; --i,w.step(outer_axis)){
                    const auto dst_last = dst+static_cast<std::ptrdiff_t>(block_size_);
                    if (block_size_ > 3){
                        for (const auto dst_last_=dst_last-3; dst<dst_last_; dst+=4){
                            *dst = *w;
                            w.step(inner_axis);
                            *(dst+1) = *w;
                            w.step(inner_axis);
                            *(dst+2) = *w;
                            w.step(inner_axis);
                            *(dst+3) = *w;
                            w.step(inner_axis);
                        }
                    }
                    for (;dst!=dst_last; ++dst,w.step(inner_axis)){
                        *dst = *w;
                    }
                    w.walk_back(inner_axis,block_size_);
                }
                w.walk_back(outer_axis,outer_size);
            }
            w.walk_back(inner_axis,ii);
        }

        template<typename ResW>
        ALWAYS_INLINE void fill_res(const T* buf, ResW& res_w, const index_type& mc_, const index_type& nc_){
            const index_type mr{Mr};
            const index_type nr{Nr};
            index_type ii=0;
            for (;ii<nc_; ii+=nr,res_w.walk(j_axis,nr)){
                const auto nr_ = adjust_block_size(ii,nr,nc_);
                index_type jj=0;
                for (;jj<mc_; jj+=mr,res_w.walk(i_axis,mr)){
                    const auto mr_ = adjust_block_size(jj,mr,mc_);
                    for (auto i=nr_; i!=0; --i,res_w.step(j_axis)){
                        const auto buf_last = buf+static_cast<std::ptrdiff_t>(mr_);
                        if (mr_ > 3){
                            for (const auto buf_last_=buf_last-3; buf<buf_last_; buf+=4){
                                *res_w = *res_w + *buf;
                                res_w.step(i_axis);
                                *res_w = *res_w + *(buf+1);
                                res_w.step(i_axis);
                                *res_w = *res_w + *(buf+2);
                                res_w.step(i_axis);
                                *res_w = *res_w + *(buf+3);
                                res_w.step(i_axis);
                            }
                        }
                        for (;buf!=buf_last; ++buf,res_w.step(i_axis)){
                            *res_w = *res_w + *buf;
                        }
                        res_w.walk_back(i_axis,mr_);
                    }
                    res_w.walk_back(j_axis,nr_);

                }
                res_w.walk_back(i_axis,jj);
            }
            res_w.walk_back(j_axis,ii);
        }

        template<typename T_, typename T1_, typename T2_>
        ALWAYS_INLINE void micro_kernel_generic(T_* res_buf, const T1_* const a_buf, const T2_* b_buf, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            auto res_buf_ = res_buf;
            for (const auto b_last=b_buf+nr_; b_buf!=b_last; ++b_buf){
                const auto e = *b_buf;
                for (std::ptrdiff_t ir=0; ir!=mr_; ++ir,++res_buf_){
                    *res_buf_=a_buf[ir]*e;
                }
            }
            for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                const auto a_buf_ = a_buf+kk*mr_;
                auto res_buf_ = res_buf;
                for (const auto b_last=b_buf+nr_; b_buf!=b_last; ++b_buf){
                    const auto e = *b_buf;
                    for (std::ptrdiff_t ir=0; ir!=mr_; ++ir,++res_buf_){
                        *res_buf_=*res_buf_+a_buf_[ir]*e;
                    }
                }
            }
        }

        template<typename T_, typename T1_, typename T2_>
        ALWAYS_INLINE void micro_kernel(T_* res_buf, const T1_* const a_buf, const T2_* b_buf, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            micro_kernel_generic(res_buf,a_buf,b_buf,mr_,nr_,kc_);
        }

        template<typename U=T, std::enable_if_t<std::is_same_v<U,U> && HAS_FMA && HAS_AVX, int> =0>
        ALWAYS_INLINE void micro_kernel(double* res_buf, const double* const a_buf, const double* b_buf, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            micro_kernel_sd(res_buf,a_buf,b_buf,mr_,nr_,kc_);
        }

        template<typename U=T, std::enable_if_t<std::is_same_v<U,U> && HAS_FMA && HAS_AVX, int> =0>
        ALWAYS_INLINE void micro_kernel(float* res_buf, const float* const a_buf, const float* b_buf, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            micro_kernel_sd(res_buf,a_buf,b_buf,mr_,nr_,kc_);
        }

        template<typename T_, typename T1_, typename T2_>
        ALWAYS_INLINE void macro_kernel(T_* res_buf, const T1_* const a_buf, const T2_* b_buf, const std::ptrdiff_t& mc_, const std::ptrdiff_t& nc_, const std::ptrdiff_t& kc_){
            const std::ptrdiff_t mr{Mr};
            const std::ptrdiff_t nr{Nr};
            for (std::ptrdiff_t i=0; i<nc_; i+=nr){
                const auto nr_ = adjust_block_size(i,nr,nc_);
                auto a_buf_ = a_buf;
                for (std::ptrdiff_t j=0; j<mc_; j+=mr){
                    const auto mr_ = adjust_block_size(j,mr,mc_);
                    //std::cout<<std::endl<<mr_<<" "<<nr_<<" "<<detail::alignment(res_buf)<<" "<<detail::alignment(res_buf+mr_*nr_);
                    micro_kernel(res_buf,a_buf_,b_buf,mr_,nr_,kc_);
                    res_buf+=mr_*nr_;
                    a_buf_+=mr_*kc_;
                }
                b_buf+=kc_*nr_;
            }
        }

        // #define AVX_MATMUL_BROADCAST(Ptr) asm ("vbroadcastsd %0,%%ymm0" ::"m"(*Ptr));

        // #define AVX_MATMUL_MUL(OFFSET)\
        // asm ("vmovapd %1,%%ymm1\n\t"\
        //     "vmulpd %%ymm1,%%ymm0,%%ymm2\n\t"\
        //     "vmovapd %%ymm2,%0"\
        //     :"=m"(*(res_buf_+OFFSET)):"m"(*(a_data+OFFSET))\
        // );

        // #define AVX_MATMUL_FMA(OFFSET)\
        // asm ("vmovapd %2,%%ymm2\n\t"\
        //     "vfmadd231pd %1,%%ymm0,%%ymm2\n\t"\
        //     "vmovapd %%ymm2,%0"\
        //     :"=m"(*(res_buf_+OFFSET)):"m"(*(a_data_+OFFSET)),"m"(*(res_buf_+OFFSET))\
        // );

        //#define AVX_MATMUL_BROADCAST(Ptr) auto b_y = _mm256_broadcast_sd(Ptr);
        // #define AVX_MATMUL_MUL(OFFSET) _mm256_store_pd(res_buf_+OFFSET,_mm256_mul_pd(_mm256_load_pd(a_data+OFFSET),b_y));
        // #define AVX_MATMUL_FMA(OFFSET) _mm256_store_pd(res_buf_+OFFSET,_mm256_fmadd_pd(_mm256_load_pd(a_data_+OFFSET),b_y,_mm256_load_pd(res_buf_+OFFSET)));

        ALWAYS_INLINE auto avx_broadcast(const double* buf){
            return _mm256_broadcast_sd(buf);
        }
        ALWAYS_INLINE auto avx_broadcast(const float* buf){
            return _mm256_broadcast_ss(buf);
        }
        ALWAYS_INLINE auto avx_element(const __m256d& y){
            return _mm256_cvtsd_f64(y);
        }
        ALWAYS_INLINE auto avx_element(const __m256& y){
            return _mm256_cvtss_f32(y);
        }


        ALWAYS_INLINE void avx_mul(double* res_buf, const double* a_buf, const __m256d& b_y,  const std::size_t& offset){
            _mm256_store_pd(res_buf+offset,_mm256_mul_pd(_mm256_load_pd(a_buf+offset),b_y));
        }
        ALWAYS_INLINE void avx_mul(float* res_buf, const float* a_buf, const __m256& b_y,  const std::size_t& offset){
            //std::cout<<std::endl<<"avx_mul "<<detail::alignment(res_buf+offset);
            _mm256_store_ps(res_buf+offset,_mm256_mul_ps(_mm256_load_ps(a_buf+offset),b_y));
        }
        template<std::size_t NPacked, std::size_t...I, typename U, typename P, typename V>
        ALWAYS_INLINE void avx_mul_n(std::index_sequence<I...>, U* res_buf, P* a_buf, const V& b_y){
            (avx_mul(res_buf,a_buf,b_y,I*NPacked),...);
        }

        ALWAYS_INLINE void avx_fma(double* res_buf, const double* a_buf, const __m256d& b_y,  const std::size_t& offset){
            _mm256_store_pd(res_buf+offset,_mm256_fmadd_pd(_mm256_load_pd(a_buf+offset),b_y,_mm256_load_pd(res_buf+offset)));
        }
        ALWAYS_INLINE void avx_fma(float* res_buf, const float* a_buf, const __m256& b_y,  const std::size_t& offset){
            _mm256_store_ps(res_buf+offset,_mm256_fmadd_ps(_mm256_load_ps(a_buf+offset),b_y,_mm256_load_ps(res_buf+offset)));
        }
        template<std::size_t NPacked, std::size_t...I, typename U, typename P, typename V>
        ALWAYS_INLINE void avx_fma_n(std::index_sequence<I...>, U* res_buf, P* a_buf, const V& b_y){
            (avx_fma(res_buf,a_buf,b_y,I*NPacked),...);
        }

        template<typename U>
        ALWAYS_INLINE void micro_kernel_sd(U* res_buf, const U* const a_data, const U* b_data, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            static_assert(std::is_floating_point_v<U>);
            static_assert(sizeof(U)==4 || sizeof(U)==8);
            static constexpr std::size_t n_packed = 32/sizeof(U);
            //std::cout<<std::endl<<"res_buf alignment "<<detail::alignment(res_buf);
            if (Mr%n_packed==0 && mr_==Mr){
                //std::cout<<std::endl<<"if (Mrn_packed==0 && mr_==Mr){";
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nr_; b_data!=b_last; ++b_data){
                    const auto b_y = avx_broadcast(b_data);
                    avx_mul_n<n_packed>(std::make_index_sequence<Mr/n_packed>{},res_buf_,a_data,b_y);
                    res_buf_+=Mr;
                }
                for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                    const auto a_data_ = a_data+kk*mr_;
                    auto res_buf_ = res_buf;
                    for (const auto b_last=b_data+nr_; b_data!=b_last; ++b_data){
                        const auto b_y = avx_broadcast(b_data);
                        avx_fma_n<n_packed>(std::make_index_sequence<Mr/n_packed>{},res_buf_,a_data_,b_y);
                        res_buf_+=Mr;
                    }
                }
            }else if (mr_>n_packed-1){
                //std::cout<<std::endl<<"}else if (mr_>n_packed-1){ "<<mr_<<" "<<nr_<<" "<<detail::alignment(res_buf);
                std::ptrdiff_t mm{0};
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nr_; b_data!=b_last; ++b_data){
                    auto a_data_=a_data;
                    const auto a_last = a_data_+mr_;
                    const auto b_y = avx_broadcast(b_data);
                    for (;mm!=0; --mm,++a_data_,++res_buf_){
                        *res_buf_=*a_data_*avx_element(b_y);
                    }
                    for (const auto a_last_=a_last-(n_packed-1); a_data_<a_last_; a_data_+=n_packed,res_buf_+=n_packed){
                        if constexpr (n_packed==4){
                            _mm256_store_pd(res_buf_,_mm256_mul_pd(_mm256_loadu_pd(a_data_),b_y));
                        }else{
                            _mm256_store_ps(res_buf_,_mm256_mul_ps(_mm256_loadu_ps(a_data_),b_y));
                        }
                        //avx_mul(res_buf_,a_data_,b_y,0);
                    }
                    for (mm=n_packed; a_data_!=a_last; ++a_data_,++res_buf_,--mm){
                        *res_buf_=*a_data_*avx_element(b_y);
                    }
                }
                for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                    const auto a_data_ = a_data+kk*mr_;
                    res_buf_ = res_buf;
                    std::ptrdiff_t mm{0};
                    for (const auto b_last=b_data+nr_; b_data!=b_last; ++b_data){
                        auto a_data__=a_data_;
                        const auto a_last = a_data_+mr_;
                        const auto b_y = avx_broadcast(b_data);
                        for (;mm!=0; --mm,++a_data__,++res_buf_){
                            *res_buf_+=*a_data__*avx_element(b_y);
                        }
                        for (const auto a_last__=a_last-(n_packed-1); a_data__<a_last__; a_data__+=n_packed,res_buf_+=n_packed){
                            if constexpr (n_packed==4){
                                _mm256_store_pd(res_buf_,_mm256_fmadd_pd(_mm256_loadu_pd(a_data__),b_y,_mm256_load_pd(res_buf_)));
                            }else{
                                _mm256_store_ps(res_buf_,_mm256_fmadd_ps(_mm256_loadu_ps(a_data__),b_y,_mm256_load_ps(res_buf_)));
                            }
                            //avx_fma(res_buf_,a_data__,b_y,0);
                        }
                        for (mm=n_packed; a_data__!=a_last; ++a_data__,++res_buf_,--mm){
                            *res_buf_+=*a_data__*avx_element(b_y);
                        }
                    }
                }
            }else{
                micro_kernel_generic(res_buf,a_data,b_data,mr_,nr_,kc_);
            }
        }

    public:
        matmul_2d_blis1(const index_type& k_, const dim_type& i_axis_, const dim_type& j_axis_):
            k{k_},
            i_axis{i_axis_},
            j_axis{j_axis_}
        {}

        template<typename ResW, typename W1, typename W2>
        ALWAYS_INLINE void operator()(ResW res_w, W1 w1, W2 w2, const index_type& ic_min, const index_type& ic_max, const index_type& jc_min, const index_type& jc_max, T* res_buf, T1* a_buf, T2* b_buf){
            const index_type mc{Mc};
            const index_type nc{Nc};
            const index_type kc{Kc};
            const index_type mr{Mr};
            const index_type nr{Nr};
            for (index_type jc=jc_min; jc<jc_max; jc+=nc){
                w2.walk(j_axis,jc);
                res_w.walk(j_axis,jc);
                const auto nc_ = adjust_block_size(jc,nc,jc_max);
                for (index_type pc=0; pc<k; pc+=kc){
                    w1.walk(j_axis,pc);
                    w2.walk(i_axis,pc);
                    const auto kc_ = adjust_block_size(pc,kc,k);
                    fill_buf(w2,b_buf,j_axis,i_axis,nc_,kc_,nr);
                    for (index_type ic=ic_min; ic<ic_max; ic+=mc){
                        w1.walk(i_axis,ic);
                        res_w.walk(i_axis,ic);
                        const auto mc_ = adjust_block_size(ic,mc,ic_max);
                        fill_buf(w1,a_buf,i_axis,j_axis,mc_,kc_,mr);
                        macro_kernel(res_buf,a_buf,b_buf,static_cast<std::ptrdiff_t>(mc_),static_cast<std::ptrdiff_t>(nc_),static_cast<std::ptrdiff_t>(kc_));
                        fill_res(res_buf,res_w,mc_,nc_);
                        w1.walk_back(i_axis,ic);
                        res_w.walk_back(i_axis,ic);
                    }
                    w1.walk_back(j_axis,pc);
                    w2.walk_back(i_axis,pc);
                }
                w2.walk_back(j_axis,jc);
                res_w.walk_back(j_axis,jc);
            }
        }

        template<typename ResW, typename W1, typename W2>
        ALWAYS_INLINE void operator()(ResW res_w, W1 w1, W2 w2, const index_type& ic_min, const index_type& ic_max, const index_type& jc_min, const index_type& jc_max){
            if constexpr (is_buffer_on_stack){
                alignas(alignment) std::array<res_value_type,Mc*Nc> res_buf;
                alignas(alignment) std::array<value_type1,Mc*Kc> a_buf;
                alignas(alignment) std::array<value_type2,Kc*Nc> b_buf;
                this->operator()(res_w,w1,w2,ic_min,ic_max,jc_min,jc_max,res_buf.data(),a_buf.data(),b_buf.data());
            }else{
                auto make_buf_size = [](auto i_size, auto j_size, auto t_size){
                    return alignment*(i_size*j_size*t_size/alignment+1);
                };
                const auto res_buf_size = make_buf_size(Mc,Nc,sizeof(res_value_type));
                const auto a_buf_size = make_buf_size(Mc,Kc,sizeof(value_type1));
                const auto b_buf_size = make_buf_size(Kc,Nc,sizeof(value_type2));
                if constexpr (std::is_same_v<res_value_type,value_type1> && std::is_same_v<res_value_type,value_type2>){
                    gtensor::basic_storage<res_value_type,allocation::aligned_allocator<res_value_type,alignment>> buf(res_buf_size+a_buf_size+b_buf_size);
                    this->operator()(res_w,w1,w2,ic_min,ic_max,jc_min,jc_max,buf.data(),buf.data()+res_buf_size,buf.data()+res_buf_size+a_buf_size);

                }else{
                    gtensor::basic_storage<res_value_type,allocation::aligned_allocator<res_value_type,alignment>> res_buf(res_buf_size);
                    gtensor::basic_storage<value_type1,allocation::aligned_allocator<value_type1,alignment>> a_buf(a_buf_size);
                    gtensor::basic_storage<value_type2,allocation::aligned_allocator<value_type2,alignment>> b_buf(b_buf_size);
                    this->operator()(res_w,w1,w2,ic_min,ic_max,jc_min,jc_max,res_buf.data(),a_buf.data(),b_buf.data());
                }
            }
        }
    };

    template<typename T, typename T1, typename T2, typename Order, typename Config, std::size_t Mc, std::size_t Nc, std::size_t Kc, std::size_t Mr, std::size_t Nr, typename BufferOnStack=std::false_type>
    class matmul_2d_blis
    {
        using index_type = typename Config::index_type;
        using dim_type = typename Config::dim_type;
        const index_type k;
        const dim_type i_axis;
        const dim_type j_axis;

        ALWAYS_INLINE auto adjust_block_size(const index_type& idx, const index_type& block_size, const index_type& max){
            return idx+block_size>max ? max-idx : block_size;
        }

        template<typename W, typename U, typename DimT>
        ALWAYS_INLINE void fill_buf(W& w, U* dst, const DimT& inner_axis, const DimT& outer_axis, const index_type& inner_size, const index_type& outer_size, const index_type& block_size){
            index_type ii=0;
            for (;ii<inner_size; ii+=block_size,w.walk(inner_axis,block_size)){
                const auto block_size_ = adjust_block_size(ii,block_size,inner_size);
                for (auto i=outer_size; i!=0; --i,w.step(outer_axis)){
                    const auto dst_last = dst+static_cast<std::ptrdiff_t>(block_size_);
                    if (block_size_ > 3){
                        for (const auto dst_last_=dst_last-3; dst<dst_last_; dst+=4){
                            *dst = *w;
                            w.step(inner_axis);
                            *(dst+1) = *w;
                            w.step(inner_axis);
                            *(dst+2) = *w;
                            w.step(inner_axis);
                            *(dst+3) = *w;
                            w.step(inner_axis);
                        }
                    }
                    for (;dst!=dst_last; ++dst,w.step(inner_axis)){
                        *dst = *w;
                    }
                    w.walk_back(inner_axis,block_size_);
                }
                w.walk_back(outer_axis,outer_size);
            }
            w.walk_back(inner_axis,ii);
        }

        template<typename ResW, typename DimT>
        ALWAYS_INLINE void fill_res(const T* buf, ResW& res_w, const DimT& inner_axis, const DimT& outer_axis, const index_type& inner_size, const index_type& outer_size){
            for (auto i=outer_size; i!=0; --i,res_w.step(outer_axis)){
                const auto buf_last = buf+static_cast<std::ptrdiff_t>(inner_size);
                if (inner_size > 3){
                    for (const auto buf_last_=buf_last-3; buf<buf_last_; buf+=4){
                        *res_w = *res_w + *buf;
                        res_w.step(inner_axis);
                        *res_w = *res_w + *(buf+1);
                        res_w.step(inner_axis);
                        *res_w = *res_w + *(buf+2);
                        res_w.step(inner_axis);
                        *res_w = *res_w + *(buf+3);
                        res_w.step(inner_axis);
                    }
                }
                for (;buf!=buf_last; ++buf,res_w.step(inner_axis)){
                    *res_w = *res_w + *buf;
                }
                res_w.walk_back(inner_axis,inner_size);
            }
            res_w.walk_back(outer_axis,outer_size);
        }

        // template<typename ResW, typename DimT>
        // ALWAYS_INLINE void fill_res(const T* buf, ResW& res_w, const DimT& inner_axis, const DimT& outer_axis, const index_type& i_axis_size, const index_type& j_axis_size, const index_type& block_inner_size, const index_type& block_outer_size){
        //     const index_type mr{Mr};
        //     const index_type nr{Nr};
        //     index_type jj=0;
        //     for (;jj<j_axis_size; jj+=nr,res_w.walk(j_axis,nr)){
        //         //const auto block_outer_size_ = adjust_block_size(jj,nr,j_axis_size);
        //         const auto j_block_size_ = adjust_block_size(jj,nr,j_axis_size);
        //         index_type ii=0;
        //         for (;ii<i_axis_size; ii+=mr,res_w.walk(i_axis,mr)){
        //             //const auto block_inner_size_ = adjust_block_size(ii,mr,i_axis_size);
        //             const auto i_block_size_ = adjust_block_size(ii,mr,i_axis_size);

        //             for (auto i=block_outer_size_; i!=0; --i,res_w.step(outer_axis)){
        //                 const auto buf_last = buf+static_cast<std::ptrdiff_t>(block_inner_size_);
        //                 if (block_inner_size_ > 3){
        //                     for (const auto buf_last_=buf_last-3; buf<buf_last_; buf+=4){
        //                         *res_w = *res_w + *buf;
        //                         res_w.step(inner_axis);
        //                         *res_w = *res_w + *(buf+1);
        //                         res_w.step(inner_axis);
        //                         *res_w = *res_w + *(buf+2);
        //                         res_w.step(inner_axis);
        //                         *res_w = *res_w + *(buf+3);
        //                         res_w.step(inner_axis);
        //                     }
        //                 }
        //                 for (;buf!=buf_last; ++buf,res_w.step(inner_axis)){
        //                     *res_w = *res_w + *buf;
        //                 }
        //                 res_w.walk_back(inner_axis,block_inner_size_);
        //             }
        //             res_w.walk_back(outer_axis,block_outer_size_);

        //         }
        //         res_w.walk_back(i_axis,ii);
        //     }
        //     res_w.walk_back(j_axis,jj);
        // }

        template<typename T_, typename T1_, typename T2_>
        ALWAYS_INLINE void kernel(T_* res_buf, const T1_* const a_data, const T2_* b_data, const std::ptrdiff_t& mr_, const std::ptrdiff_t& nr_, const std::ptrdiff_t& kc_){
            auto res_buf_ = res_buf;
            for (const auto b_last=b_data+nr_; b_data!=b_last; ++b_data){
                const auto e = *b_data;
                for (std::ptrdiff_t ir=0; ir!=mr_; ++ir,++res_buf_){
                    *res_buf_=a_data[ir]*e;
                }
            }
            for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                const auto a_data_ = a_data+kk*mr_;
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nr_; b_data!=b_last; ++b_data){
                    const auto e = *b_data;
                    for (std::ptrdiff_t ir=0; ir!=mr_; ++ir,++res_buf_){
                        *res_buf_=*res_buf_+a_data_[ir]*e;
                    }
                }
            }
        }

        #define AVX_MATMUL_BROADCAST(V) asm ("vbroadcastsd %0,%%ymm0" ::"m"(V));

        #define AVX_MATMUL_MUL(OFFSET)\
        asm ("vmovapd %1,%%ymm1\n\t"\
            "vmulpd %%ymm1,%%ymm0,%%ymm2\n\t"\
            "vmovapd %%ymm2,%0"\
            :"=m"(*(res_buf_+OFFSET)):"m"(*(a_data+OFFSET))\
        );

        #define AVX_MATMUL_FMA(OFFSET)\
        asm ("vmovapd %2,%%ymm2\n\t"\
            "vfmadd231pd %1,%%ymm0,%%ymm2\n\t"\
            "vmovapd %%ymm2,%0"\
            :"=m"(*(res_buf_+OFFSET)):"m"(*(a_data_+OFFSET)),"m"(*(res_buf_+OFFSET))\
        );

        template<typename T_ = T, std::enable_if_t<std::is_same_v<T_,T_> && HAS_FMA && HAS_AVX && false, int> =0>
        ALWAYS_INLINE void kernel(double* res_buf, const double* const a_data, const double* b_data, const std::ptrdiff_t& mc_, const std::ptrdiff_t& nc_, const std::ptrdiff_t& kc_){
            if (mc_==Mc){
                //std::cout<<std::endl<<"if (mc_==Mc && false){";
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                    AVX_MATMUL_BROADCAST(*b_data);
                    AVX_MATMUL_MUL(0);
                    AVX_MATMUL_MUL(4);
                    AVX_MATMUL_MUL(8);
                    AVX_MATMUL_MUL(12);
                    AVX_MATMUL_MUL(16);
                    AVX_MATMUL_MUL(20);
                    AVX_MATMUL_MUL(24);
                    AVX_MATMUL_MUL(28);
                    AVX_MATMUL_MUL(32);
                    AVX_MATMUL_MUL(36);
                    AVX_MATMUL_MUL(40);
                    AVX_MATMUL_MUL(44);
                    AVX_MATMUL_MUL(48);
                    AVX_MATMUL_MUL(52);
                    AVX_MATMUL_MUL(56);
                    AVX_MATMUL_MUL(60);
                    AVX_MATMUL_MUL(64);
                    AVX_MATMUL_MUL(68);
                    AVX_MATMUL_MUL(72);
                    AVX_MATMUL_MUL(76);
                    AVX_MATMUL_MUL(80);
                    AVX_MATMUL_MUL(84);
                    AVX_MATMUL_MUL(88);
                    AVX_MATMUL_MUL(92);
                    AVX_MATMUL_MUL(96);
                    AVX_MATMUL_MUL(100);
                    AVX_MATMUL_MUL(104);
                    AVX_MATMUL_MUL(108);
                    AVX_MATMUL_MUL(112);
                    AVX_MATMUL_MUL(116);
                    AVX_MATMUL_MUL(120);
                    AVX_MATMUL_MUL(124);
                    res_buf_+=128;
                }
                for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                    const auto a_data_ = a_data+kk*mc_;
                    auto res_buf_ = res_buf;
                    for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                        AVX_MATMUL_BROADCAST(*b_data);
                        AVX_MATMUL_FMA(0);
                        AVX_MATMUL_FMA(4);
                        AVX_MATMUL_FMA(8);
                        AVX_MATMUL_FMA(12);
                        AVX_MATMUL_FMA(16);
                        AVX_MATMUL_FMA(20);
                        AVX_MATMUL_FMA(24);
                        AVX_MATMUL_FMA(28);
                        AVX_MATMUL_FMA(32);
                        AVX_MATMUL_FMA(36);
                        AVX_MATMUL_FMA(40);
                        AVX_MATMUL_FMA(44);
                        AVX_MATMUL_FMA(48);
                        AVX_MATMUL_FMA(52);
                        AVX_MATMUL_FMA(56);
                        AVX_MATMUL_FMA(60);
                        AVX_MATMUL_FMA(64);
                        AVX_MATMUL_FMA(68);
                        AVX_MATMUL_FMA(72);
                        AVX_MATMUL_FMA(76);
                        AVX_MATMUL_FMA(80);
                        AVX_MATMUL_FMA(84);
                        AVX_MATMUL_FMA(88);
                        AVX_MATMUL_FMA(92);
                        AVX_MATMUL_FMA(96);
                        AVX_MATMUL_FMA(100);
                        AVX_MATMUL_FMA(104);
                        AVX_MATMUL_FMA(108);
                        AVX_MATMUL_FMA(112);
                        AVX_MATMUL_FMA(116);
                        AVX_MATMUL_FMA(120);
                        AVX_MATMUL_FMA(124);
                        res_buf_+=128;
                    }
                }
            }else if (mc_>3){
                //std::cout<<std::endl<<"if }else if (mc_>3){";
                std::ptrdiff_t mm{0};
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                    auto a_data_=a_data;
                    const auto a_last = a_data_+mc_;
                    const auto b_y =_mm256_broadcast_sd(b_data);
                    for (;mm!=0; --mm,++a_data_,++res_buf_){
                        *res_buf_=*a_data_*_mm256_cvtsd_f64(b_y);
                    }
                    for (const auto a_last_=a_last-3; a_data_<a_last_; a_data_+=4,res_buf_+=4){
                        _mm256_store_pd(res_buf_,_mm256_mul_pd(_mm256_loadu_pd(a_data_),b_y));
                    }
                    for (mm=4; a_data_!=a_last; ++a_data_,++res_buf_,--mm){
                        *res_buf_=*a_data_*_mm256_cvtsd_f64(b_y);
                    }
                }
                for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                    const auto a_data_ = a_data+kk*mc_;
                    res_buf_ = res_buf;
                    std::ptrdiff_t mm{0};
                    for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                        auto a_data__=a_data_;
                        const auto a_last = a_data_+mc_;
                        const auto b_y = _mm256_broadcast_sd(b_data);
                        for (;mm!=0; --mm,++a_data__,++res_buf_){
                            *res_buf_+=*a_data__*_mm256_cvtsd_f64(b_y);
                        }
                        for (const auto a_last__=a_last-3; a_data__<a_last__; a_data__+=4,res_buf_+=4){
                            _mm256_store_pd(res_buf_,_mm256_fmadd_pd(_mm256_loadu_pd(a_data__),b_y,_mm256_load_pd(res_buf_)));
                        }
                        for (mm=4; a_data__!=a_last; ++a_data__,++res_buf_,--mm){
                            *res_buf_+=*a_data__*_mm256_cvtsd_f64(b_y);
                        }
                    }
                }
            }else{
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                    const auto e = *b_data;
                    for (std::ptrdiff_t ir=0; ir!=mc_; ++ir,++res_buf_){
                        *res_buf_=a_data[ir]*e;
                    }
                }
                for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                    const auto a_data_ = a_data+kk*mc_;
                    auto res_buf_ = res_buf;
                    for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                        auto a_data__=a_data_;
                        const auto b_e = *b_data;
                        for (const auto a_last = a_data_+mc_; a_data__!=a_last; ++a_data__,++res_buf_){
                            *res_buf_+=*a_data__*b_e;
                        }
                    }
                }
            }
        }


        // template<typename T_ = T, std::enable_if_t<std::is_same_v<T_,T_> && HAS_FMA && HAS_AVX && Mc==128, int> =0>
        // ALWAYS_INLINE void kernel(double* res_buf, const double* const a_data, const double* b_data, const std::ptrdiff_t& mc_, const std::ptrdiff_t& nc_, const std::ptrdiff_t& kc_){
        //     if (mc_==Mc){
        //         //std::cout<<std::endl<<"if (mc_==Mc && false){";
        //         auto res_buf_ = res_buf;
        //         for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //             const auto b_y =_mm256_broadcast_sd(b_data);
        //             _mm256_store_pd(res_buf_,_mm256_mul_pd(_mm256_load_pd(a_data),b_y));
        //             _mm256_store_pd(res_buf_+4,_mm256_mul_pd(_mm256_load_pd(a_data+4),b_y));
        //             _mm256_store_pd(res_buf_+8,_mm256_mul_pd(_mm256_load_pd(a_data+8),b_y));
        //             _mm256_store_pd(res_buf_+12,_mm256_mul_pd(_mm256_load_pd(a_data+12),b_y));
        //             _mm256_store_pd(res_buf_+16,_mm256_mul_pd(_mm256_load_pd(a_data+16),b_y));
        //             _mm256_store_pd(res_buf_+20,_mm256_mul_pd(_mm256_load_pd(a_data+20),b_y));
        //             _mm256_store_pd(res_buf_+24,_mm256_mul_pd(_mm256_load_pd(a_data+24),b_y));
        //             _mm256_store_pd(res_buf_+28,_mm256_mul_pd(_mm256_load_pd(a_data+28),b_y));
        //             _mm256_store_pd(res_buf_+32,_mm256_mul_pd(_mm256_load_pd(a_data+32),b_y));
        //             _mm256_store_pd(res_buf_+36,_mm256_mul_pd(_mm256_load_pd(a_data+36),b_y));
        //             _mm256_store_pd(res_buf_+40,_mm256_mul_pd(_mm256_load_pd(a_data+40),b_y));
        //             _mm256_store_pd(res_buf_+44,_mm256_mul_pd(_mm256_load_pd(a_data+44),b_y));
        //             _mm256_store_pd(res_buf_+48,_mm256_mul_pd(_mm256_load_pd(a_data+48),b_y));
        //             _mm256_store_pd(res_buf_+52,_mm256_mul_pd(_mm256_load_pd(a_data+52),b_y));
        //             _mm256_store_pd(res_buf_+56,_mm256_mul_pd(_mm256_load_pd(a_data+56),b_y));
        //             _mm256_store_pd(res_buf_+60,_mm256_mul_pd(_mm256_load_pd(a_data+60),b_y));
        //             _mm256_store_pd(res_buf_+64,_mm256_mul_pd(_mm256_load_pd(a_data+64),b_y));
        //             _mm256_store_pd(res_buf_+68,_mm256_mul_pd(_mm256_load_pd(a_data+68),b_y));
        //             _mm256_store_pd(res_buf_+72,_mm256_mul_pd(_mm256_load_pd(a_data+72),b_y));
        //             _mm256_store_pd(res_buf_+76,_mm256_mul_pd(_mm256_load_pd(a_data+76),b_y));
        //             _mm256_store_pd(res_buf_+80,_mm256_mul_pd(_mm256_load_pd(a_data+80),b_y));
        //             _mm256_store_pd(res_buf_+84,_mm256_mul_pd(_mm256_load_pd(a_data+84),b_y));
        //             _mm256_store_pd(res_buf_+88,_mm256_mul_pd(_mm256_load_pd(a_data+88),b_y));
        //             _mm256_store_pd(res_buf_+92,_mm256_mul_pd(_mm256_load_pd(a_data+92),b_y));
        //             _mm256_store_pd(res_buf_+96,_mm256_mul_pd(_mm256_load_pd(a_data+96),b_y));
        //             _mm256_store_pd(res_buf_+100,_mm256_mul_pd(_mm256_load_pd(a_data+100),b_y));
        //             _mm256_store_pd(res_buf_+104,_mm256_mul_pd(_mm256_load_pd(a_data+104),b_y));
        //             _mm256_store_pd(res_buf_+108,_mm256_mul_pd(_mm256_load_pd(a_data+108),b_y));
        //             _mm256_store_pd(res_buf_+112,_mm256_mul_pd(_mm256_load_pd(a_data+112),b_y));
        //             _mm256_store_pd(res_buf_+116,_mm256_mul_pd(_mm256_load_pd(a_data+116),b_y));
        //             _mm256_store_pd(res_buf_+120,_mm256_mul_pd(_mm256_load_pd(a_data+120),b_y));
        //             _mm256_store_pd(res_buf_+124,_mm256_mul_pd(_mm256_load_pd(a_data+124),b_y));
        //             res_buf_+=128;
        //         }
        //         for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
        //             const auto a_data_ = a_data+kk*mc_;
        //             auto res_buf_ = res_buf;
        //             for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //                 const auto b_y = _mm256_broadcast_sd(b_data);
        //                 _mm256_store_pd(res_buf_,_mm256_fmadd_pd(_mm256_load_pd(a_data_),b_y,_mm256_load_pd(res_buf_)));
        //                 _mm256_store_pd(res_buf_+4,_mm256_fmadd_pd(_mm256_load_pd(a_data_+4),b_y,_mm256_load_pd(res_buf_+4)));
        //                 _mm256_store_pd(res_buf_+8,_mm256_fmadd_pd(_mm256_load_pd(a_data_+8),b_y,_mm256_load_pd(res_buf_+8)));
        //                 _mm256_store_pd(res_buf_+12,_mm256_fmadd_pd(_mm256_load_pd(a_data_+12),b_y,_mm256_load_pd(res_buf_+12)));
        //                 _mm256_store_pd(res_buf_+16,_mm256_fmadd_pd(_mm256_load_pd(a_data_+16),b_y,_mm256_load_pd(res_buf_+16)));
        //                 _mm256_store_pd(res_buf_+20,_mm256_fmadd_pd(_mm256_load_pd(a_data_+20),b_y,_mm256_load_pd(res_buf_+20)));
        //                 _mm256_store_pd(res_buf_+24,_mm256_fmadd_pd(_mm256_load_pd(a_data_+24),b_y,_mm256_load_pd(res_buf_+24)));
        //                 _mm256_store_pd(res_buf_+28,_mm256_fmadd_pd(_mm256_load_pd(a_data_+28),b_y,_mm256_load_pd(res_buf_+28)));
        //                 _mm256_store_pd(res_buf_+32,_mm256_fmadd_pd(_mm256_load_pd(a_data_+32),b_y,_mm256_load_pd(res_buf_+32)));
        //                 _mm256_store_pd(res_buf_+36,_mm256_fmadd_pd(_mm256_load_pd(a_data_+36),b_y,_mm256_load_pd(res_buf_+36)));
        //                 _mm256_store_pd(res_buf_+40,_mm256_fmadd_pd(_mm256_load_pd(a_data_+40),b_y,_mm256_load_pd(res_buf_+40)));
        //                 _mm256_store_pd(res_buf_+44,_mm256_fmadd_pd(_mm256_load_pd(a_data_+44),b_y,_mm256_load_pd(res_buf_+44)));
        //                 _mm256_store_pd(res_buf_+48,_mm256_fmadd_pd(_mm256_load_pd(a_data_+48),b_y,_mm256_load_pd(res_buf_+48)));
        //                 _mm256_store_pd(res_buf_+52,_mm256_fmadd_pd(_mm256_load_pd(a_data_+52),b_y,_mm256_load_pd(res_buf_+52)));
        //                 _mm256_store_pd(res_buf_+56,_mm256_fmadd_pd(_mm256_load_pd(a_data_+56),b_y,_mm256_load_pd(res_buf_+56)));
        //                 _mm256_store_pd(res_buf_+60,_mm256_fmadd_pd(_mm256_load_pd(a_data_+60),b_y,_mm256_load_pd(res_buf_+60)));
        //                 _mm256_store_pd(res_buf_+64,_mm256_fmadd_pd(_mm256_load_pd(a_data_+64),b_y,_mm256_load_pd(res_buf_+64)));
        //                 _mm256_store_pd(res_buf_+68,_mm256_fmadd_pd(_mm256_load_pd(a_data_+68),b_y,_mm256_load_pd(res_buf_+68)));
        //                 _mm256_store_pd(res_buf_+72,_mm256_fmadd_pd(_mm256_load_pd(a_data_+72),b_y,_mm256_load_pd(res_buf_+72)));
        //                 _mm256_store_pd(res_buf_+76,_mm256_fmadd_pd(_mm256_load_pd(a_data_+76),b_y,_mm256_load_pd(res_buf_+76)));
        //                 _mm256_store_pd(res_buf_+80,_mm256_fmadd_pd(_mm256_load_pd(a_data_+80),b_y,_mm256_load_pd(res_buf_+80)));
        //                 _mm256_store_pd(res_buf_+84,_mm256_fmadd_pd(_mm256_load_pd(a_data_+84),b_y,_mm256_load_pd(res_buf_+84)));
        //                 _mm256_store_pd(res_buf_+88,_mm256_fmadd_pd(_mm256_load_pd(a_data_+88),b_y,_mm256_load_pd(res_buf_+88)));
        //                 _mm256_store_pd(res_buf_+92,_mm256_fmadd_pd(_mm256_load_pd(a_data_+92),b_y,_mm256_load_pd(res_buf_+92)));
        //                 _mm256_store_pd(res_buf_+96,_mm256_fmadd_pd(_mm256_load_pd(a_data_+96),b_y,_mm256_load_pd(res_buf_+96)));
        //                 _mm256_store_pd(res_buf_+100,_mm256_fmadd_pd(_mm256_load_pd(a_data_+100),b_y,_mm256_load_pd(res_buf_+100)));
        //                 _mm256_store_pd(res_buf_+104,_mm256_fmadd_pd(_mm256_load_pd(a_data_+104),b_y,_mm256_load_pd(res_buf_+104)));
        //                 _mm256_store_pd(res_buf_+108,_mm256_fmadd_pd(_mm256_load_pd(a_data_+108),b_y,_mm256_load_pd(res_buf_+108)));
        //                 _mm256_store_pd(res_buf_+112,_mm256_fmadd_pd(_mm256_load_pd(a_data_+112),b_y,_mm256_load_pd(res_buf_+112)));
        //                 _mm256_store_pd(res_buf_+116,_mm256_fmadd_pd(_mm256_load_pd(a_data_+116),b_y,_mm256_load_pd(res_buf_+116)));
        //                 _mm256_store_pd(res_buf_+120,_mm256_fmadd_pd(_mm256_load_pd(a_data_+120),b_y,_mm256_load_pd(res_buf_+120)));
        //                 _mm256_store_pd(res_buf_+124,_mm256_fmadd_pd(_mm256_load_pd(a_data_+124),b_y,_mm256_load_pd(res_buf_+124)));
        //                 res_buf_+=128;
        //             }
        //         }
        //     }else if (mc_>3){
        //         //std::cout<<std::endl<<"if }else if (mc_>3){";
        //         std::ptrdiff_t mm{0};
        //         auto res_buf_ = res_buf;
        //         for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //             auto a_data_=a_data;
        //             const auto a_last = a_data_+mc_;
        //             const auto b_y =_mm256_broadcast_sd(b_data);
        //             for (;mm!=0; --mm,++a_data_,++res_buf_){
        //                 *res_buf_=*a_data_*_mm256_cvtsd_f64(b_y);
        //             }
        //             for (const auto a_last_=a_last-3; a_data_<a_last_; a_data_+=4,res_buf_+=4){
        //                 _mm256_store_pd(res_buf_,_mm256_mul_pd(_mm256_loadu_pd(a_data_),b_y));
        //             }
        //             for (mm=4; a_data_!=a_last; ++a_data_,++res_buf_,--mm){
        //                 *res_buf_=*a_data_*_mm256_cvtsd_f64(b_y);
        //             }
        //         }
        //         for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
        //             const auto a_data_ = a_data+kk*mc_;
        //             res_buf_ = res_buf;
        //             std::ptrdiff_t mm{0};
        //             for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //                 auto a_data__=a_data_;
        //                 const auto a_last = a_data_+mc_;
        //                 const auto b_y = _mm256_broadcast_sd(b_data);
        //                 for (;mm!=0; --mm,++a_data__,++res_buf_){
        //                     *res_buf_+=*a_data__*_mm256_cvtsd_f64(b_y);
        //                 }
        //                 for (const auto a_last__=a_last-3; a_data__<a_last__; a_data__+=4,res_buf_+=4){
        //                     _mm256_store_pd(res_buf_,_mm256_fmadd_pd(_mm256_loadu_pd(a_data__),b_y,_mm256_load_pd(res_buf_)));
        //                 }
        //                 for (mm=4; a_data__!=a_last; ++a_data__,++res_buf_,--mm){
        //                     *res_buf_+=*a_data__*_mm256_cvtsd_f64(b_y);
        //                 }
        //             }
        //         }
        //     }else{
        //         auto res_buf_ = res_buf;
        //         for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //             const auto e = *b_data;
        //             for (std::ptrdiff_t ir=0; ir!=mc_; ++ir,++res_buf_){
        //                 *res_buf_=a_data[ir]*e;
        //             }
        //         }
        //         for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
        //             const auto a_data_ = a_data+kk*mc_;
        //             auto res_buf_ = res_buf;
        //             for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //                 auto a_data__=a_data_;
        //                 const auto b_e = *b_data;
        //                 for (const auto a_last = a_data_+mc_; a_data__!=a_last; ++a_data__,++res_buf_){
        //                     *res_buf_+=*a_data__*b_e;
        //                 }
        //             }
        //         }
        //     }
        // }

    public:
        matmul_2d_blis(const index_type& k_, const dim_type& i_axis_, const dim_type& j_axis_):
            k{k_},
            i_axis{i_axis_},
            j_axis{j_axis_}
        {}

        template<typename ResW, typename W1, typename W2>
        ALWAYS_INLINE void operator()(ResW res_w, W1 w1, W2 w2, const index_type& ic_min, const index_type& ic_max, const index_type& jc_min, const index_type& jc_max, T* res_buf, T1* a_buf, T2* b_buf){
            const index_type mc{Mc};
            const index_type nc{Nc};
            const index_type kc{Kc};
            const index_type mr{Mr};
            const index_type nr{Nr};
            for (index_type jc=jc_min; jc<jc_max; jc+=nc){
                w2.walk(j_axis,jc);
                res_w.walk(j_axis,jc);
                const auto nc_ = adjust_block_size(jc,nc,jc_max);
                for (index_type pc=0; pc<k; pc+=kc){
                    w1.walk(j_axis,pc);
                    w2.walk(i_axis,pc);
                    const auto kc_ = adjust_block_size(pc,kc,k);
                    fill_buf(w2,b_buf,j_axis,i_axis,nc_,kc_,nr);
                    for (index_type ic=ic_min; ic<ic_max; ic+=mc){
                        w1.walk(i_axis,ic);
                        res_w.walk(i_axis,ic);
                        const auto mc_ = adjust_block_size(ic,mc,ic_max);
                        fill_buf(w1,a_buf,i_axis,j_axis,mc_,kc_,mr);

                        auto res_buf_ = res_buf;
                        auto b_buf_ = b_buf;
                        index_type jr=0;
                        for (;jr<nc_; jr+=nr,res_w.walk(j_axis,nr)){
                            const auto nr_ = adjust_block_size(jr,nr,nc_);
                            auto a_buf_ = a_buf;
                            index_type ir=0;
                            for (;ir<mc_; ir+=mr,res_w.walk(i_axis,mr)){
                                const auto mr_ = adjust_block_size(ir,mr,mc_);
                                if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
                                    kernel(res_buf_,b_buf_,a_buf_,static_cast<std::ptrdiff_t>(nr_),static_cast<std::ptrdiff_t>(mr_),static_cast<std::ptrdiff_t>(kc_));
                                    fill_res(res_buf_,res_w,j_axis,i_axis,nr_,mr_);
                                }else{
                                    kernel(res_buf_,a_buf_,b_buf_,static_cast<std::ptrdiff_t>(mr_),static_cast<std::ptrdiff_t>(nr_),static_cast<std::ptrdiff_t>(kc_));
                                    fill_res(res_buf_,res_w,i_axis,j_axis,mr_,nr_);
                                }
                                res_buf_+=mr_*nr_;
                                a_buf_+=mr_*kc_;
                            }
                            res_w.walk_back(i_axis,ir);
                            b_buf_+=kc_*nr_;
                        }
                        res_w.walk_back(j_axis,jr);

                        w1.walk_back(i_axis,ic);
                        res_w.walk_back(i_axis,ic);
                    }
                    w1.walk_back(j_axis,pc);
                    w2.walk_back(i_axis,pc);
                }
                w2.walk_back(j_axis,jc);
                res_w.walk_back(j_axis,jc);
            }
        }

        template<typename ResW, typename W1, typename W2>
        ALWAYS_INLINE void operator()(ResW res_w, W1 w1, W2 w2, const index_type& ic_min, const index_type& ic_max, const index_type& jc_min, const index_type& jc_max){
            static constexpr std::size_t alignment = 64;
            if constexpr (BufferOnStack::value){
                alignas(alignment) std::array<T,Mc*Nc> res_buf;
                alignas(alignment) std::array<T1,Mc*Kc> a_buf;
                alignas(alignment) std::array<T2,Kc*Nc> b_buf;
                this->operator()(res_w,w1,w2,ic_min,ic_max,jc_min,jc_max,res_buf.data(),a_buf.data(),b_buf.data());
            }else{
                auto make_buf_size = [](auto i_size, auto j_size, auto t_size){
                    return alignment*(i_size*j_size*t_size/alignment+1);
                };
                const auto res_buf_size = make_buf_size(Mc,Nc,sizeof(T));
                const auto a_buf_size = make_buf_size(Mc,Kc,sizeof(T1));
                const auto b_buf_size = make_buf_size(Kc,Nc,sizeof(T2));
                gtensor::basic_storage<T,allocation::aligned_allocator<T,alignment>> res_buf(res_buf_size);
                gtensor::basic_storage<T1,allocation::aligned_allocator<T1,alignment>> a_buf(a_buf_size);
                gtensor::basic_storage<T2,allocation::aligned_allocator<T2,alignment>> b_buf(b_buf_size);
                this->operator()(res_w,w1,w2,ic_min,ic_max,jc_min,jc_max,res_buf.data(),a_buf.data(),b_buf.data());
            }
        }
    };

    template<typename T, typename T1, typename T2, typename Order, typename Config, std::size_t Mc, std::size_t Nc, std::size_t Kc, typename BufferOnStack=std::false_type>
    class matmul_2d
    {
        using index_type = typename Config::index_type;
        using dim_type = typename Config::dim_type;
        const index_type k;
        const dim_type i_axis;
        const dim_type j_axis;

        template<typename W, typename U, typename DimT>
        ALWAYS_INLINE void fill_buf(W& w, U* dst, const DimT& inner_axis, const DimT& outer_axis, const index_type& inner_size, const index_type& outer_size){
            for (auto i=outer_size; i!=0; --i,w.step(outer_axis)){
                const auto dst_last = dst+static_cast<std::ptrdiff_t>(inner_size);
                if (inner_size > 3){
                    for (const auto dst_last_=dst_last-3; dst<dst_last_; dst+=4){
                        *dst = *w;
                        w.step(inner_axis);
                        *(dst+1) = *w;
                        w.step(inner_axis);
                        *(dst+2) = *w;
                        w.step(inner_axis);
                        *(dst+3) = *w;
                        w.step(inner_axis);
                    }
                }
                for (;dst!=dst_last; ++dst,w.step(inner_axis)){
                    *dst = *w;
                }
                w.walk_back(inner_axis,inner_size);
            }
            w.walk_back(outer_axis,outer_size);
        };

        template<typename ResW, typename DimT>
        ALWAYS_INLINE void fill_res(const T* buf, ResW& res_w, const DimT& inner_axis, const DimT& outer_axis, const index_type& inner_size, const index_type& outer_size){
            for (auto i=outer_size; i!=0; --i,res_w.step(outer_axis)){
                const auto buf_last = buf+static_cast<std::ptrdiff_t>(inner_size);
                if (inner_size > 3){
                    for (const auto buf_last_=buf_last-3; buf<buf_last_; buf+=4){
                        *res_w = *res_w + *buf;
                        res_w.step(inner_axis);
                        *res_w = *res_w + *(buf+1);
                        res_w.step(inner_axis);
                        *res_w = *res_w + *(buf+2);
                        res_w.step(inner_axis);
                        *res_w = *res_w + *(buf+3);
                        res_w.step(inner_axis);
                    }
                }
                for (;buf!=buf_last; ++buf,res_w.step(inner_axis)){
                    *res_w = *res_w + *buf;
                }
                res_w.walk_back(inner_axis,inner_size);
            }
            res_w.walk_back(outer_axis,outer_size);
        };

        template<typename T_, typename T1_, typename T2_>
        ALWAYS_INLINE void kernel(T_* res_buf, const T1_* const a_data, const T2_* b_data, const std::ptrdiff_t& mc_, const std::ptrdiff_t& nc_, const std::ptrdiff_t& kc_){
            auto res_buf_ = res_buf;
            for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                const auto e = *b_data;
                for (std::ptrdiff_t ir=0; ir!=mc_; ++ir,++res_buf_){
                    *res_buf_=a_data[ir]*e;
                }
            }
            for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                const auto a_data_ = a_data+kk*mc_;
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                    const auto e = *b_data;
                    for (std::ptrdiff_t ir=0; ir!=mc_; ++ir,++res_buf_){
                        *res_buf_=*res_buf_+a_data_[ir]*e;
                    }
                }
            }
        }

        #define AVX_MATMUL_BROADCAST(V) asm ("vbroadcastsd %0,%%ymm0" ::"m"(V));

        #define AVX_MATMUL_MUL(OFFSET)\
        asm ("vmovapd %1,%%ymm1\n\t"\
            "vmulpd %%ymm1,%%ymm0,%%ymm2\n\t"\
            "vmovapd %%ymm2,%0"\
            :"=m"(*(res_buf_+OFFSET)):"m"(*(a_data+OFFSET))\
        );

        #define AVX_MATMUL_FMA(OFFSET)\
        asm ("vmovapd %2,%%ymm2\n\t"\
            "vfmadd231pd %1,%%ymm0,%%ymm2\n\t"\
            "vmovapd %%ymm2,%0"\
            :"=m"(*(res_buf_+OFFSET)):"m"(*(a_data_+OFFSET)),"m"(*(res_buf_+OFFSET))\
        );

        template<typename T_ = T, std::enable_if_t<std::is_same_v<T_,T_> && HAS_FMA && HAS_AVX && false, int> =0>
        ALWAYS_INLINE void kernel(double* res_buf, const double* const a_data, const double* b_data, const std::ptrdiff_t& mc_, const std::ptrdiff_t& nc_, const std::ptrdiff_t& kc_){
            if (mc_==Mc){
                //std::cout<<std::endl<<"if (mc_==Mc && false){";
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                    AVX_MATMUL_BROADCAST(*b_data);
                    AVX_MATMUL_MUL(0);
                    AVX_MATMUL_MUL(4);
                    AVX_MATMUL_MUL(8);
                    AVX_MATMUL_MUL(12);
                    AVX_MATMUL_MUL(16);
                    AVX_MATMUL_MUL(20);
                    AVX_MATMUL_MUL(24);
                    AVX_MATMUL_MUL(28);
                    AVX_MATMUL_MUL(32);
                    AVX_MATMUL_MUL(36);
                    AVX_MATMUL_MUL(40);
                    AVX_MATMUL_MUL(44);
                    AVX_MATMUL_MUL(48);
                    AVX_MATMUL_MUL(52);
                    AVX_MATMUL_MUL(56);
                    AVX_MATMUL_MUL(60);
                    AVX_MATMUL_MUL(64);
                    AVX_MATMUL_MUL(68);
                    AVX_MATMUL_MUL(72);
                    AVX_MATMUL_MUL(76);
                    AVX_MATMUL_MUL(80);
                    AVX_MATMUL_MUL(84);
                    AVX_MATMUL_MUL(88);
                    AVX_MATMUL_MUL(92);
                    AVX_MATMUL_MUL(96);
                    AVX_MATMUL_MUL(100);
                    AVX_MATMUL_MUL(104);
                    AVX_MATMUL_MUL(108);
                    AVX_MATMUL_MUL(112);
                    AVX_MATMUL_MUL(116);
                    AVX_MATMUL_MUL(120);
                    AVX_MATMUL_MUL(124);
                    res_buf_+=128;
                }
                for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                    const auto a_data_ = a_data+kk*mc_;
                    auto res_buf_ = res_buf;
                    for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                        AVX_MATMUL_BROADCAST(*b_data);
                        AVX_MATMUL_FMA(0);
                        AVX_MATMUL_FMA(4);
                        AVX_MATMUL_FMA(8);
                        AVX_MATMUL_FMA(12);
                        AVX_MATMUL_FMA(16);
                        AVX_MATMUL_FMA(20);
                        AVX_MATMUL_FMA(24);
                        AVX_MATMUL_FMA(28);
                        AVX_MATMUL_FMA(32);
                        AVX_MATMUL_FMA(36);
                        AVX_MATMUL_FMA(40);
                        AVX_MATMUL_FMA(44);
                        AVX_MATMUL_FMA(48);
                        AVX_MATMUL_FMA(52);
                        AVX_MATMUL_FMA(56);
                        AVX_MATMUL_FMA(60);
                        AVX_MATMUL_FMA(64);
                        AVX_MATMUL_FMA(68);
                        AVX_MATMUL_FMA(72);
                        AVX_MATMUL_FMA(76);
                        AVX_MATMUL_FMA(80);
                        AVX_MATMUL_FMA(84);
                        AVX_MATMUL_FMA(88);
                        AVX_MATMUL_FMA(92);
                        AVX_MATMUL_FMA(96);
                        AVX_MATMUL_FMA(100);
                        AVX_MATMUL_FMA(104);
                        AVX_MATMUL_FMA(108);
                        AVX_MATMUL_FMA(112);
                        AVX_MATMUL_FMA(116);
                        AVX_MATMUL_FMA(120);
                        AVX_MATMUL_FMA(124);
                        res_buf_+=128;
                    }
                }
            }else if (mc_>3){
                //std::cout<<std::endl<<"if }else if (mc_>3){";
                std::ptrdiff_t mm{0};
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                    auto a_data_=a_data;
                    const auto a_last = a_data_+mc_;
                    const auto b_y =_mm256_broadcast_sd(b_data);
                    for (;mm!=0; --mm,++a_data_,++res_buf_){
                        *res_buf_=*a_data_*_mm256_cvtsd_f64(b_y);
                    }
                    for (const auto a_last_=a_last-3; a_data_<a_last_; a_data_+=4,res_buf_+=4){
                        _mm256_store_pd(res_buf_,_mm256_mul_pd(_mm256_loadu_pd(a_data_),b_y));
                    }
                    for (mm=4; a_data_!=a_last; ++a_data_,++res_buf_,--mm){
                        *res_buf_=*a_data_*_mm256_cvtsd_f64(b_y);
                    }
                }
                for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                    const auto a_data_ = a_data+kk*mc_;
                    res_buf_ = res_buf;
                    std::ptrdiff_t mm{0};
                    for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                        auto a_data__=a_data_;
                        const auto a_last = a_data_+mc_;
                        const auto b_y = _mm256_broadcast_sd(b_data);
                        for (;mm!=0; --mm,++a_data__,++res_buf_){
                            *res_buf_+=*a_data__*_mm256_cvtsd_f64(b_y);
                        }
                        for (const auto a_last__=a_last-3; a_data__<a_last__; a_data__+=4,res_buf_+=4){
                            _mm256_store_pd(res_buf_,_mm256_fmadd_pd(_mm256_loadu_pd(a_data__),b_y,_mm256_load_pd(res_buf_)));
                        }
                        for (mm=4; a_data__!=a_last; ++a_data__,++res_buf_,--mm){
                            *res_buf_+=*a_data__*_mm256_cvtsd_f64(b_y);
                        }
                    }
                }
            }else{
                auto res_buf_ = res_buf;
                for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                    const auto e = *b_data;
                    for (std::ptrdiff_t ir=0; ir!=mc_; ++ir,++res_buf_){
                        *res_buf_=a_data[ir]*e;
                    }
                }
                for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
                    const auto a_data_ = a_data+kk*mc_;
                    auto res_buf_ = res_buf;
                    for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
                        auto a_data__=a_data_;
                        const auto b_e = *b_data;
                        for (const auto a_last = a_data_+mc_; a_data__!=a_last; ++a_data__,++res_buf_){
                            *res_buf_+=*a_data__*b_e;
                        }
                    }
                }
            }
        }


        // template<typename T_ = T, std::enable_if_t<std::is_same_v<T_,T_> && HAS_FMA && HAS_AVX && Mc==128, int> =0>
        // ALWAYS_INLINE void kernel(double* res_buf, const double* const a_data, const double* b_data, const std::ptrdiff_t& mc_, const std::ptrdiff_t& nc_, const std::ptrdiff_t& kc_){
        //     if (mc_==Mc){
        //         //std::cout<<std::endl<<"if (mc_==Mc && false){";
        //         auto res_buf_ = res_buf;
        //         for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //             const auto b_y =_mm256_broadcast_sd(b_data);
        //             _mm256_store_pd(res_buf_,_mm256_mul_pd(_mm256_load_pd(a_data),b_y));
        //             _mm256_store_pd(res_buf_+4,_mm256_mul_pd(_mm256_load_pd(a_data+4),b_y));
        //             _mm256_store_pd(res_buf_+8,_mm256_mul_pd(_mm256_load_pd(a_data+8),b_y));
        //             _mm256_store_pd(res_buf_+12,_mm256_mul_pd(_mm256_load_pd(a_data+12),b_y));
        //             _mm256_store_pd(res_buf_+16,_mm256_mul_pd(_mm256_load_pd(a_data+16),b_y));
        //             _mm256_store_pd(res_buf_+20,_mm256_mul_pd(_mm256_load_pd(a_data+20),b_y));
        //             _mm256_store_pd(res_buf_+24,_mm256_mul_pd(_mm256_load_pd(a_data+24),b_y));
        //             _mm256_store_pd(res_buf_+28,_mm256_mul_pd(_mm256_load_pd(a_data+28),b_y));
        //             _mm256_store_pd(res_buf_+32,_mm256_mul_pd(_mm256_load_pd(a_data+32),b_y));
        //             _mm256_store_pd(res_buf_+36,_mm256_mul_pd(_mm256_load_pd(a_data+36),b_y));
        //             _mm256_store_pd(res_buf_+40,_mm256_mul_pd(_mm256_load_pd(a_data+40),b_y));
        //             _mm256_store_pd(res_buf_+44,_mm256_mul_pd(_mm256_load_pd(a_data+44),b_y));
        //             _mm256_store_pd(res_buf_+48,_mm256_mul_pd(_mm256_load_pd(a_data+48),b_y));
        //             _mm256_store_pd(res_buf_+52,_mm256_mul_pd(_mm256_load_pd(a_data+52),b_y));
        //             _mm256_store_pd(res_buf_+56,_mm256_mul_pd(_mm256_load_pd(a_data+56),b_y));
        //             _mm256_store_pd(res_buf_+60,_mm256_mul_pd(_mm256_load_pd(a_data+60),b_y));
        //             _mm256_store_pd(res_buf_+64,_mm256_mul_pd(_mm256_load_pd(a_data+64),b_y));
        //             _mm256_store_pd(res_buf_+68,_mm256_mul_pd(_mm256_load_pd(a_data+68),b_y));
        //             _mm256_store_pd(res_buf_+72,_mm256_mul_pd(_mm256_load_pd(a_data+72),b_y));
        //             _mm256_store_pd(res_buf_+76,_mm256_mul_pd(_mm256_load_pd(a_data+76),b_y));
        //             _mm256_store_pd(res_buf_+80,_mm256_mul_pd(_mm256_load_pd(a_data+80),b_y));
        //             _mm256_store_pd(res_buf_+84,_mm256_mul_pd(_mm256_load_pd(a_data+84),b_y));
        //             _mm256_store_pd(res_buf_+88,_mm256_mul_pd(_mm256_load_pd(a_data+88),b_y));
        //             _mm256_store_pd(res_buf_+92,_mm256_mul_pd(_mm256_load_pd(a_data+92),b_y));
        //             _mm256_store_pd(res_buf_+96,_mm256_mul_pd(_mm256_load_pd(a_data+96),b_y));
        //             _mm256_store_pd(res_buf_+100,_mm256_mul_pd(_mm256_load_pd(a_data+100),b_y));
        //             _mm256_store_pd(res_buf_+104,_mm256_mul_pd(_mm256_load_pd(a_data+104),b_y));
        //             _mm256_store_pd(res_buf_+108,_mm256_mul_pd(_mm256_load_pd(a_data+108),b_y));
        //             _mm256_store_pd(res_buf_+112,_mm256_mul_pd(_mm256_load_pd(a_data+112),b_y));
        //             _mm256_store_pd(res_buf_+116,_mm256_mul_pd(_mm256_load_pd(a_data+116),b_y));
        //             _mm256_store_pd(res_buf_+120,_mm256_mul_pd(_mm256_load_pd(a_data+120),b_y));
        //             _mm256_store_pd(res_buf_+124,_mm256_mul_pd(_mm256_load_pd(a_data+124),b_y));
        //             res_buf_+=128;
        //         }
        //         for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
        //             const auto a_data_ = a_data+kk*mc_;
        //             auto res_buf_ = res_buf;
        //             for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //                 const auto b_y = _mm256_broadcast_sd(b_data);
        //                 _mm256_store_pd(res_buf_,_mm256_fmadd_pd(_mm256_load_pd(a_data_),b_y,_mm256_load_pd(res_buf_)));
        //                 _mm256_store_pd(res_buf_+4,_mm256_fmadd_pd(_mm256_load_pd(a_data_+4),b_y,_mm256_load_pd(res_buf_+4)));
        //                 _mm256_store_pd(res_buf_+8,_mm256_fmadd_pd(_mm256_load_pd(a_data_+8),b_y,_mm256_load_pd(res_buf_+8)));
        //                 _mm256_store_pd(res_buf_+12,_mm256_fmadd_pd(_mm256_load_pd(a_data_+12),b_y,_mm256_load_pd(res_buf_+12)));
        //                 _mm256_store_pd(res_buf_+16,_mm256_fmadd_pd(_mm256_load_pd(a_data_+16),b_y,_mm256_load_pd(res_buf_+16)));
        //                 _mm256_store_pd(res_buf_+20,_mm256_fmadd_pd(_mm256_load_pd(a_data_+20),b_y,_mm256_load_pd(res_buf_+20)));
        //                 _mm256_store_pd(res_buf_+24,_mm256_fmadd_pd(_mm256_load_pd(a_data_+24),b_y,_mm256_load_pd(res_buf_+24)));
        //                 _mm256_store_pd(res_buf_+28,_mm256_fmadd_pd(_mm256_load_pd(a_data_+28),b_y,_mm256_load_pd(res_buf_+28)));
        //                 _mm256_store_pd(res_buf_+32,_mm256_fmadd_pd(_mm256_load_pd(a_data_+32),b_y,_mm256_load_pd(res_buf_+32)));
        //                 _mm256_store_pd(res_buf_+36,_mm256_fmadd_pd(_mm256_load_pd(a_data_+36),b_y,_mm256_load_pd(res_buf_+36)));
        //                 _mm256_store_pd(res_buf_+40,_mm256_fmadd_pd(_mm256_load_pd(a_data_+40),b_y,_mm256_load_pd(res_buf_+40)));
        //                 _mm256_store_pd(res_buf_+44,_mm256_fmadd_pd(_mm256_load_pd(a_data_+44),b_y,_mm256_load_pd(res_buf_+44)));
        //                 _mm256_store_pd(res_buf_+48,_mm256_fmadd_pd(_mm256_load_pd(a_data_+48),b_y,_mm256_load_pd(res_buf_+48)));
        //                 _mm256_store_pd(res_buf_+52,_mm256_fmadd_pd(_mm256_load_pd(a_data_+52),b_y,_mm256_load_pd(res_buf_+52)));
        //                 _mm256_store_pd(res_buf_+56,_mm256_fmadd_pd(_mm256_load_pd(a_data_+56),b_y,_mm256_load_pd(res_buf_+56)));
        //                 _mm256_store_pd(res_buf_+60,_mm256_fmadd_pd(_mm256_load_pd(a_data_+60),b_y,_mm256_load_pd(res_buf_+60)));
        //                 _mm256_store_pd(res_buf_+64,_mm256_fmadd_pd(_mm256_load_pd(a_data_+64),b_y,_mm256_load_pd(res_buf_+64)));
        //                 _mm256_store_pd(res_buf_+68,_mm256_fmadd_pd(_mm256_load_pd(a_data_+68),b_y,_mm256_load_pd(res_buf_+68)));
        //                 _mm256_store_pd(res_buf_+72,_mm256_fmadd_pd(_mm256_load_pd(a_data_+72),b_y,_mm256_load_pd(res_buf_+72)));
        //                 _mm256_store_pd(res_buf_+76,_mm256_fmadd_pd(_mm256_load_pd(a_data_+76),b_y,_mm256_load_pd(res_buf_+76)));
        //                 _mm256_store_pd(res_buf_+80,_mm256_fmadd_pd(_mm256_load_pd(a_data_+80),b_y,_mm256_load_pd(res_buf_+80)));
        //                 _mm256_store_pd(res_buf_+84,_mm256_fmadd_pd(_mm256_load_pd(a_data_+84),b_y,_mm256_load_pd(res_buf_+84)));
        //                 _mm256_store_pd(res_buf_+88,_mm256_fmadd_pd(_mm256_load_pd(a_data_+88),b_y,_mm256_load_pd(res_buf_+88)));
        //                 _mm256_store_pd(res_buf_+92,_mm256_fmadd_pd(_mm256_load_pd(a_data_+92),b_y,_mm256_load_pd(res_buf_+92)));
        //                 _mm256_store_pd(res_buf_+96,_mm256_fmadd_pd(_mm256_load_pd(a_data_+96),b_y,_mm256_load_pd(res_buf_+96)));
        //                 _mm256_store_pd(res_buf_+100,_mm256_fmadd_pd(_mm256_load_pd(a_data_+100),b_y,_mm256_load_pd(res_buf_+100)));
        //                 _mm256_store_pd(res_buf_+104,_mm256_fmadd_pd(_mm256_load_pd(a_data_+104),b_y,_mm256_load_pd(res_buf_+104)));
        //                 _mm256_store_pd(res_buf_+108,_mm256_fmadd_pd(_mm256_load_pd(a_data_+108),b_y,_mm256_load_pd(res_buf_+108)));
        //                 _mm256_store_pd(res_buf_+112,_mm256_fmadd_pd(_mm256_load_pd(a_data_+112),b_y,_mm256_load_pd(res_buf_+112)));
        //                 _mm256_store_pd(res_buf_+116,_mm256_fmadd_pd(_mm256_load_pd(a_data_+116),b_y,_mm256_load_pd(res_buf_+116)));
        //                 _mm256_store_pd(res_buf_+120,_mm256_fmadd_pd(_mm256_load_pd(a_data_+120),b_y,_mm256_load_pd(res_buf_+120)));
        //                 _mm256_store_pd(res_buf_+124,_mm256_fmadd_pd(_mm256_load_pd(a_data_+124),b_y,_mm256_load_pd(res_buf_+124)));
        //                 res_buf_+=128;
        //             }
        //         }
        //     }else if (mc_>3){
        //         //std::cout<<std::endl<<"if }else if (mc_>3){";
        //         std::ptrdiff_t mm{0};
        //         auto res_buf_ = res_buf;
        //         for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //             auto a_data_=a_data;
        //             const auto a_last = a_data_+mc_;
        //             const auto b_y =_mm256_broadcast_sd(b_data);
        //             for (;mm!=0; --mm,++a_data_,++res_buf_){
        //                 *res_buf_=*a_data_*_mm256_cvtsd_f64(b_y);
        //             }
        //             for (const auto a_last_=a_last-3; a_data_<a_last_; a_data_+=4,res_buf_+=4){
        //                 _mm256_store_pd(res_buf_,_mm256_mul_pd(_mm256_loadu_pd(a_data_),b_y));
        //             }
        //             for (mm=4; a_data_!=a_last; ++a_data_,++res_buf_,--mm){
        //                 *res_buf_=*a_data_*_mm256_cvtsd_f64(b_y);
        //             }
        //         }
        //         for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
        //             const auto a_data_ = a_data+kk*mc_;
        //             res_buf_ = res_buf;
        //             std::ptrdiff_t mm{0};
        //             for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //                 auto a_data__=a_data_;
        //                 const auto a_last = a_data_+mc_;
        //                 const auto b_y = _mm256_broadcast_sd(b_data);
        //                 for (;mm!=0; --mm,++a_data__,++res_buf_){
        //                     *res_buf_+=*a_data__*_mm256_cvtsd_f64(b_y);
        //                 }
        //                 for (const auto a_last__=a_last-3; a_data__<a_last__; a_data__+=4,res_buf_+=4){
        //                     _mm256_store_pd(res_buf_,_mm256_fmadd_pd(_mm256_loadu_pd(a_data__),b_y,_mm256_load_pd(res_buf_)));
        //                 }
        //                 for (mm=4; a_data__!=a_last; ++a_data__,++res_buf_,--mm){
        //                     *res_buf_+=*a_data__*_mm256_cvtsd_f64(b_y);
        //                 }
        //             }
        //         }
        //     }else{
        //         auto res_buf_ = res_buf;
        //         for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //             const auto e = *b_data;
        //             for (std::ptrdiff_t ir=0; ir!=mc_; ++ir,++res_buf_){
        //                 *res_buf_=a_data[ir]*e;
        //             }
        //         }
        //         for (std::ptrdiff_t kk=1; kk!=kc_; ++kk){
        //             const auto a_data_ = a_data+kk*mc_;
        //             auto res_buf_ = res_buf;
        //             for (const auto b_last=b_data+nc_; b_data!=b_last; ++b_data){
        //                 auto a_data__=a_data_;
        //                 const auto b_e = *b_data;
        //                 for (const auto a_last = a_data_+mc_; a_data__!=a_last; ++a_data__,++res_buf_){
        //                     *res_buf_+=*a_data__*b_e;
        //                 }
        //             }
        //         }
        //     }
        // }

        ALWAYS_INLINE auto adjust_block_size(const index_type& idx, const index_type& block_size, const index_type& max){
            return idx+block_size>max ? max-idx : block_size;
        }

    public:
        matmul_2d(const index_type& k_, const dim_type& i_axis_, const dim_type& j_axis_):
            k{k_},
            i_axis{i_axis_},
            j_axis{j_axis_}
        {}

        template<typename ResW, typename W1, typename W2>
        ALWAYS_INLINE void operator()(ResW res_w, W1 w1, W2 w2, const index_type& ic_min, const index_type& ic_max, const index_type& jc_min, const index_type& jc_max, T* res_buf, T1* a_buf, T2* b_buf){
            const index_type mc{Mc};
            const index_type nc{Nc};
            const index_type kc{Kc};
            for (index_type jc=jc_min; jc<jc_max; jc+=nc){
                w2.walk(j_axis,jc);
                res_w.walk(j_axis,jc);
                const auto nc_ = adjust_block_size(jc,nc,jc_max);
                for (index_type pc=0; pc<k; pc+=kc){
                    w1.walk(j_axis,pc);
                    w2.walk(i_axis,pc);
                    const auto kc_ = adjust_block_size(pc,kc,k);
                    fill_buf(w2,b_buf,j_axis,i_axis,nc_,kc_);
                    for (index_type ic=ic_min; ic<ic_max; ic+=mc){
                        w1.walk(i_axis,ic);
                        res_w.walk(i_axis,ic);
                        const auto mc_ = adjust_block_size(ic,mc,ic_max);
                        fill_buf(w1,a_buf,i_axis,j_axis,mc_,kc_);
                        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
                            kernel(res_buf,b_buf,a_buf,static_cast<std::ptrdiff_t>(nc_),static_cast<std::ptrdiff_t>(mc_),static_cast<std::ptrdiff_t>(kc_));
                            fill_res(res_buf,res_w,j_axis,i_axis,nc_,mc_);
                        }else{
                            kernel(res_buf,a_buf,b_buf,static_cast<std::ptrdiff_t>(mc_),static_cast<std::ptrdiff_t>(nc_),static_cast<std::ptrdiff_t>(kc_));
                            fill_res(res_buf,res_w,i_axis,j_axis,mc_,nc_);
                        }
                        w1.walk_back(i_axis,ic);
                        res_w.walk_back(i_axis,ic);
                    }
                    w1.walk_back(j_axis,pc);
                    w2.walk_back(i_axis,pc);
                }
                w2.walk_back(j_axis,jc);
                res_w.walk_back(j_axis,jc);
            }
        }

        template<typename ResW, typename W1, typename W2>
        ALWAYS_INLINE void operator()(ResW res_w, W1 w1, W2 w2, const index_type& ic_min, const index_type& ic_max, const index_type& jc_min, const index_type& jc_max){
            static constexpr std::size_t alignment = 64;
            if constexpr (BufferOnStack::value){
                alignas(alignment) std::array<T,Mc*Nc> res_buf;
                alignas(alignment) std::array<T1,Mc*Kc> a_buf;
                alignas(alignment) std::array<T2,Kc*Nc> b_buf;
                this->operator()(res_w,w1,w2,ic_min,ic_max,jc_min,jc_max,res_buf.data(),a_buf.data(),b_buf.data());
            }else{
                auto make_buf_size = [](auto i_size, auto j_size, auto t_size){
                    return alignment*(i_size*j_size*t_size/alignment+1);
                };
                const auto res_buf_size = make_buf_size(Mc,Nc,sizeof(T));
                const auto a_buf_size = make_buf_size(Mc,Kc,sizeof(T1));
                const auto b_buf_size = make_buf_size(Kc,Nc,sizeof(T2));
                gtensor::basic_storage<T,allocation::aligned_allocator<T,alignment>> res_buf(res_buf_size);
                gtensor::basic_storage<T1,allocation::aligned_allocator<T1,alignment>> a_buf(a_buf_size);
                gtensor::basic_storage<T2,allocation::aligned_allocator<T2,alignment>> b_buf(b_buf_size);
                this->operator()(res_w,w1,w2,ic_min,ic_max,jc_min,jc_max,res_buf.data(),a_buf.data(),b_buf.data());
            }
        }
    };

    //t1,t2,res are at least 2d
    template<typename ResT, typename Policy, typename...Ts, typename...Us>
    static auto matmul_nd_helper(Policy policy, const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
        using res_type = ResT;
        using order = typename res_type::order;
        using value_type = typename res_type::value_type;
        using config_type = typename res_type::config_type;
        using index_type = typename res_type::index_type;
        using shape_type = typename res_type::shape_type;
        using order = typename res_type::order;
        using value_type1 = detail::copy_type_t<typename basic_tensor<Ts...>::value_type>;
        using value_type2 = detail::copy_type_t<typename basic_tensor<Us...>::value_type>;
        using gtensor::config::c_order;
        using gtensor::config::f_order;

        const auto& shape1 = t1.shape();
        const auto& shape2 = t2.shape();
        auto br_shape = detail::make_broadcast_shape<shape_type>(shape_type(shape1.begin(),shape1.end()-2), shape_type(shape2.begin(),shape2.end()-2));
        shape_type res_shape_(br_shape.size()+2);
        std::copy(br_shape.begin(),br_shape.end(),res_shape_.begin());
        *(res_shape_.end()-2) = *(shape1.end()-2);
        *(res_shape_.end()-1) = *(shape2.end()-1);
        res_type res(std::move(res_shape_),0);
        if (res.empty()){
            return res;
        }
        const auto& res_shape = res.shape();
        const auto res_dim = detail::make_dim(res_shape);
        auto tr1 = walker_forward_range_traverser<config_type,decltype(t1.create_walker(res_dim))>{res_shape,t1.create_walker(res_dim),0,res_dim-2};
        auto tr2 = walker_forward_range_traverser<config_type,decltype(t2.create_walker(res_dim))>{res_shape,t2.create_walker(res_dim),0,res_dim-2};
        auto res_tr = walker_forward_range_traverser<config_type,decltype(res.create_walker(res_dim))>{res_shape,res.create_walker(res_dim),0,res_dim-2};

        const auto i_axis = res_dim-2;
        const auto j_axis = res_dim-1;
        const auto m = res_shape[i_axis];
        const auto n = res_shape[j_axis];
        const auto k = *(shape1.end()-1);

        using matmul_type = matmul_2d_blis2<value_type,value_type1,value_type2,config_type>;
        //using matmul_type = matmul_2d_blis1<value_type,value_type1,value_type2,config_type>;
        //using matmul_type = matmul_2d_blis<value_type,value_type1,value_type2,order,config_type,mc_size,nc_size,kc_size,mr_size,nr_size,buffer_on_stack>;
        //using matmul_type = matmul_2d<value_type,value_type1,value_type2,order,config_type,mc_size,nc_size,kc_size,buffer_on_stack>;
        matmul_type mm(k,i_axis,j_axis);

        if constexpr (multithreading::exec_policy_traits<Policy>::is_seq::value){
            do{
                mm(res_tr.walker(),tr1.walker(),tr2.walker(),0,m,0,n);
                tr1.template next<order>();
                tr2.template next<order>();
            }while(res_tr.template next<order>());
        }else{
            const auto n_tasks = multithreading::exec_policy_traits<Policy>::par_tasks::value;
            auto ti = static_cast<std::size_t>(std::sqrt(n_tasks*static_cast<std::size_t>(m)/static_cast<double>(static_cast<std::size_t>(n))));
            auto tj = static_cast<std::size_t>(std::sqrt(n_tasks*static_cast<std::size_t>(n)/static_cast<double>(static_cast<std::size_t>(m))));
            if (ti == 0){
                ti=1;
                tj = n_tasks;
            }else if (tj == 0){
                tj=1;
                ti = n_tasks;
            }else{
                const auto ti_ = n_tasks/tj;
                const auto tj_ = n_tasks/ti;
                const auto n_tasks1 = ti_*tj;
                const auto n_tasks2 = ti*tj_;
                if (n_tasks-n_tasks1 < n_tasks-n_tasks2){
                    ti=ti_;
                }else{
                    tj=tj_;
                }
            }
            //here we have ti and tj - number of parts to split corresponding dimension and to split mxn result matrix into rect parts
            //each such rect part will be assigned to separate task, ti*tj is total number of tasks
            //ti*tj <= n_tasks, if matrix can't be splitted into rect parts by n_tasks, ti*tj will be less than requested tasks number

            const index_type i_parts = static_cast<index_type>(ti);
            const index_type j_parts = static_cast<index_type>(tj);
            const index_type i_step = m > i_parts ? m/i_parts : 1;
            const index_type j_step = n > j_parts ? n/j_parts : 1;
            multithreading::task_group group{};
            std::size_t task_counter{0};
            do{
                for (index_type j = 0; j<n; j+=j_step){
                    for (index_type i = 0; i<m; i+=i_step,++task_counter){
                        multithreading::get_pool().push_group(group,mm,res_tr.walker(),tr1.walker(),tr2.walker(),i,std::min(i+i_step,m),j,std::min(j+j_step,n));
                    }
                }
                group.wait();
                tr1.template next<order>();
                tr2.template next<order>();
            }while(res_tr.template next<order>());
        }

        return res;
    }

};   //end of struct tensor_math

//tensor math frontend
//frontend uses compile-time dispatch to select implementation, see module_selector.hpp

//elementwise tensor math routines
#define GTENSOR_TENSOR_MATH_ROUTINE(NAME,F)\
template<typename...Args>\
auto NAME(Args&&...args){\
    using config_type = typename detail::common_config_type_t<std::remove_cv_t<std::remove_reference_t<Args>>...>;\
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
GTENSOR_TENSOR_MATH_ROUTINE(power,pow);
GTENSOR_TENSOR_MATH_ROUTINE(sqrt,sqrt);
GTENSOR_TENSOR_MATH_ROUTINE(cbrt,cbrt);
GTENSOR_TENSOR_MATH_ROUTINE(hypot,hypot);
//trigonometric
GTENSOR_TENSOR_MATH_ROUTINE(sin,sin);
GTENSOR_TENSOR_MATH_ROUTINE(cos,cos);
GTENSOR_TENSOR_MATH_ROUTINE(tan,tan);
GTENSOR_TENSOR_MATH_ROUTINE(asin,asin);
GTENSOR_TENSOR_MATH_ROUTINE(arcsin,asin);
GTENSOR_TENSOR_MATH_ROUTINE(acos,acos);
GTENSOR_TENSOR_MATH_ROUTINE(arccos,acos);
GTENSOR_TENSOR_MATH_ROUTINE(atan,atan);
GTENSOR_TENSOR_MATH_ROUTINE(arctan,atan);
GTENSOR_TENSOR_MATH_ROUTINE(atan2,atan2);
GTENSOR_TENSOR_MATH_ROUTINE(arctan2,atan2);
//hyperbolic
GTENSOR_TENSOR_MATH_ROUTINE(sinh,sinh);
GTENSOR_TENSOR_MATH_ROUTINE(cosh,cosh);
GTENSOR_TENSOR_MATH_ROUTINE(tanh,tanh);
GTENSOR_TENSOR_MATH_ROUTINE(asinh,asinh);
GTENSOR_TENSOR_MATH_ROUTINE(arcsinh,asinh);
GTENSOR_TENSOR_MATH_ROUTINE(acosh,acosh);
GTENSOR_TENSOR_MATH_ROUTINE(arccosh,acosh);
GTENSOR_TENSOR_MATH_ROUTINE(atanh,atanh);
GTENSOR_TENSOR_MATH_ROUTINE(arctanh,atanh);
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
//complex numbers
GTENSOR_TENSOR_MATH_ROUTINE(real,real);
GTENSOR_TENSOR_MATH_ROUTINE(imag,imag);
GTENSOR_TENSOR_MATH_ROUTINE(conj,conj);
GTENSOR_TENSOR_MATH_ROUTINE(conjugate,conj);
GTENSOR_TENSOR_MATH_ROUTINE(angle,angle);

//tensor math routines along given axis or axes
//axes may be scalar or container if multiple axes permitted

#define GTENSOR_TENSOR_MATH_REDUCE_ROUTINE(NAME,F)\
template<typename Policy, typename...Ts, typename Axes>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(policy,t,axes,keep_dims);\
}\
template<typename Policy, typename...Ts, typename DimT>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(policy,t,axes,keep_dims);\
}\
template<typename Policy, typename...Ts>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(policy,t,keep_dims);\
}\
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
template<typename Policy, typename...Ts, typename Axes, typename Initial = gtensor::detail::no_value>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false, const Initial& initial = Initial{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(policy,t,axes,keep_dims,initial);\
}\
template<typename Policy, typename...Ts, typename DimT, typename Initial = gtensor::detail::no_value>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false, const Initial& initial = Initial{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(policy,t,axes,keep_dims,initial);\
}\
template<typename Policy, typename...Ts, typename Initial = gtensor::detail::no_value>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims = false, const Initial& initial = Initial{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(policy,t,keep_dims,initial);\
}\
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
template<typename Policy, typename...Ts, typename DimT>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(policy,t,axis);\
}\
template<typename Policy, typename...Ts>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return tensor_math_selector_t<config_type>::F(policy,t);\
}\
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
template<typename Policy, typename...Ts, typename DimT=int>
auto diff(Policy policy, const basic_tensor<Ts...>& t, std::size_t n = 1, const DimT& axis = -1){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return tensor_math_selector_t<config_type>::diff(policy,t,n,axis);
}
template<typename...Ts, typename DimT=int>
auto diff(const basic_tensor<Ts...>& t, std::size_t n = 1, const DimT& axis = -1){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return tensor_math_selector_t<config_type>::diff(t,n,axis);
}
//none recursive implementation of second differences, more efficient than diff with n=2
template<typename Policy, typename...Ts, typename DimT=int>
auto diff2(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis = -1){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return tensor_math_selector_t<config_type>::diff2(policy,t,axis);
}
template<typename...Ts, typename DimT=int>
auto diff2(const basic_tensor<Ts...>& t, const DimT& axis = -1){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return tensor_math_selector_t<config_type>::diff2(t,axis);
}

//gradient along given axis, interior points has 2-nd order accuracy approximation using central difference, boundary points has 1-st order accuracy approximation
//axis is scalar
//spacing is scalar or container, scalar means uniform sample distance, container specifies coordinates along dimension
//container must be the same size as size along axis
template<typename Policy, typename...Ts, typename DimT, typename Spacing>
auto gradient(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis, const Spacing& spacing){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return tensor_math_selector_t<config_type>::gradient(policy,t,axis,spacing);
}
template<typename Policy, typename...Ts, typename DimT>
auto gradient(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis){
    using config_type = typename basic_tensor<Ts...>::config_type;
    using value_type = typename basic_tensor<Ts...>::value_type;
    return tensor_math_selector_t<config_type>::gradient(policy,t,axis,value_type{1});
}
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

//matmul
//if both arguments are 2d they are multiplied like conventional matrices.
//if either argument is nd, n>2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
//if the first argument is 1d, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
//if the second argument is 1d, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
template<typename Policy, typename...Ts,typename...Us>
auto matmul(Policy policy, const basic_tensor<Ts...>& a, const basic_tensor<Us...>& b){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return tensor_math_selector_t<config_type>::matmul(policy,a,b);
}
template<typename...Ts,typename...Us>
auto matmul(const basic_tensor<Ts...>& a, const basic_tensor<Us...>& b){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return tensor_math_selector_t<config_type>::matmul(a,b);
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