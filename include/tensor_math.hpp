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
    template<typename...Ts,typename...Us>
    static auto matmul(const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
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
                return matmul_nd_helper<res_type>(t1,t2);
            }
        }
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

    //t1,t2,res are at least 2d
    template<typename ResT, typename...Ts, typename...Us>
    static auto matmul_nd_helper(const basic_tensor<Ts...>& t1, const basic_tensor<Us...>& t2){
        using res_type = ResT;
        using order = typename res_type::order;
        using config_type = typename res_type::config_type;
        using shape_type = typename res_type::shape_type;
        using index_type = typename res_type::index_type;
        using order = typename res_type::order;
        using value_type1 = typename basic_tensor<Ts...>::value_type;
        using value_type2 = typename basic_tensor<Us...>::value_type;
        using order1 = typename basic_tensor<Ts...>::order;
        using order2 = typename basic_tensor<Us...>::order;
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

        const auto& res_shape = res.shape();
        const auto res_dim = detail::make_dim(res_shape);
        auto tr1 = walker_forward_range_traverser<config_type,decltype(t1.create_walker(res_dim))>{res_shape,t1.create_walker(res_dim),0,res_dim-2};
        auto tr2 = walker_forward_range_traverser<config_type,decltype(t2.create_walker(res_dim))>{res_shape,t2.create_walker(res_dim),0,res_dim-2};
        auto res_tr = walker_forward_range_traverser<config_type,decltype(res.create_walker(res_dim))>{res_shape,res.create_walker(res_dim),0,res_dim-2};
        const auto k = *(shape1.end()-1);
        const auto k_ = 16;
        const auto n_ = 128;

        static constexpr std::size_t mc_size = 256;
        static constexpr std::size_t nc_size = 256;
        static constexpr std::size_t kc_size = 128;
        const index_type mc = mc_size;
        const index_type nc = nc_size;
        const index_type kc = kc_size;

        auto make_submatrix_size = [](const auto& idx, const auto& block_size, const auto& max){return idx+block_size>max ? max-idx : block_size;};


        auto core_2d_cc_ff = [k](auto res_w, auto w1, auto w2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n){
            for (auto i=m;;--i){
                for (auto j=k;;--j){
                    const auto e1 = *w1;
                    for(auto r=n;;--r){
                        decltype(auto) res = *res_w;
                        res=res+e1**w2;
                        if (r==1) break;
                        w2.step(j_axis);
                        res_w.step(j_axis);
                    }
                    w2.reset_back(j_axis);
                    res_w.reset_back(j_axis);
                    if (j==1) break;
                    w2.step(i_axis);
                    w1.step(j_axis);
                }
                w2.reset_back(i_axis);
                w1.reset_back(j_axis);
                if (i==1) break;
                res_w.step(i_axis);
                w1.step(i_axis);
            }
        };

        auto core_2d_cc_ff_tiled = [k,k_](auto res_w, auto w1, auto w2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n, const auto& n_){
            auto make_size_ = [](const auto& i, const auto& b_, const auto& b){return i+b_>b ? b-i : b_;};
            for (auto kk=0; kk<k; kk+=k_){
                for (auto jj=0; jj<n; jj+=n_){

                    const auto k__ = make_size_(kk,k_,k);
                    const auto n__ = make_size_(jj,n_,n);
                    res_w.walk(j_axis,jj);
                    w1.walk(j_axis,kk);
                    w2.walk(i_axis,kk);
                    w2.walk(j_axis,jj);
                    for (auto i=m;;--i){
                        for (auto j=k__;;--j){    //kk
                            const auto e1 = *w1;
                            for(auto r=n__;;--r){     //jj
                                decltype(auto) res = *res_w;
                                res=res+e1**w2;
                                if (r==1) break;
                                w2.step(j_axis);
                                res_w.step(j_axis);
                            }
                            w2.walk_back(j_axis,n__-1);
                            res_w.walk_back(j_axis,n__-1);
                            if (j==1) break;
                            w2.step(i_axis);
                            w1.step(j_axis);
                        }
                        w2.walk_back(i_axis,k__-1);
                        w1.walk_back(j_axis,k__-1);
                        if (i==1) break;
                        res_w.step(i_axis);
                        w1.step(i_axis);
                    }
                    res_w.reset_back();
                    w1.reset_back();
                    w2.reset_back();

                }
            }

        };

        auto core_2d_cc_ff_mt = [k](auto res_w, auto w1, auto w2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n){
            for (auto i=m;;--i){
                for (auto j=k;;--j){
                    const auto e1 = *w1;
                    for(auto r=n;;--r){
                        decltype(auto) res = *res_w;
                        res=res+e1**w2;
                        if (r==1) break;
                        w2.step(j_axis);
                        res_w.step(j_axis);
                    }
                    w2.walk_back(j_axis,n-1);
                    res_w.walk_back(j_axis,n-1);
                    if (j==1) break;
                    w2.step(i_axis);
                    w1.step(j_axis);
                }
                w2.walk_back(i_axis,k-1);
                w1.walk_back(j_axis,k-1);
                if (i==1) break;
                res_w.step(i_axis);
                w1.step(i_axis);
            }
        };

        auto core_2d_cf = [k](auto res_w, auto w1, auto w2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n){
            for (auto i=m;;--i){
                for (auto j=n;;--j){
                    decltype(auto) res = *res_w;
                    for(auto r=k;;--r){
                        res=res+*w1**w2;
                        if (r==1) break;
                        w1.step(j_axis);
                        w2.step(i_axis);
                    }
                    w1.reset_back(j_axis);
                    w2.reset_back(i_axis);
                    if (j==1) break;
                    w2.step(j_axis);
                    res_w.step(j_axis);
                }
                w2.reset_back(j_axis);
                res_w.reset_back(j_axis);
                if (i==1) break;
                w1.step(i_axis);
                res_w.step(i_axis);
            }
        };

        auto core_2d_fc = [k](auto res_w, auto w1, auto w2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n){
            for (auto i=k;;--i){
                for (auto j=m;;--j){
                    const auto e1 = *w1;
                    for(auto r=n;;--r){
                        decltype(auto) res = *res_w;
                        res=res+e1**w2;
                        if (r==1) break;
                        w2.step(j_axis);
                        res_w.step(j_axis);
                    }
                    w2.reset_back(j_axis);
                    res_w.reset_back(j_axis);
                    if (j==1) break;
                    w1.step(i_axis);
                    res_w.step(i_axis);
                }
                w1.reset_back(i_axis);
                res_w.reset_back(i_axis);
                if (i==1) break;
                w1.step(j_axis);
                w2.step(i_axis);
            }
        };

        auto core_nd = [](auto core_2d, auto& res_tr, auto& tr1, auto& tr2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n){
            do{
                core_2d(res_tr.walker(),tr1.walker(),tr2.walker(),i_axis,j_axis,m,n);
                tr1.template next<order>();
                tr2.template next<order>();
            }while(res_tr.template next<order>());
        };

        auto core_nd_tiled = [](auto core_2d, auto& res_tr, auto& tr1, auto& tr2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n, const auto& n_){
            do{
                core_2d(res_tr.walker(),tr1.walker(),tr2.walker(),i_axis,j_axis,m,n,n_);
                tr1.template next<order>();
                tr2.template next<order>();
            }while(res_tr.template next<order>());
        };

        auto core_nd_tiled1 = [k,make_submatrix_size](auto& res_tr, auto& tr1, auto& tr2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n, const auto& mc, const auto& nc, const auto& kc){
            std::array<value_type1,mc_size*kc_size> a_buf;
            std::array<value_type2,kc_size*nc_size> b_buf;

            auto fill_buf = [i_axis,j_axis](auto& w, auto dit, const auto& mc_, const auto& kc_){
                for (auto i=mc_; i!=0; --i,w.step(i_axis)){
                    for (auto j=kc_; j!=0; --j,w.step(j_axis),++dit){
                        *dit = *w;
                    }
                    w.walk_back(j_axis,kc_);
                }
                w.walk_back(i_axis,mc_);
            };

            auto kernel = [](auto& res_w, auto a_it, auto b_it, const auto& i_axis, const auto& j_axis, const auto& mc_, const auto& kc_, const auto& nc_){
                for (index_type ir=0; ir!=mc_; ++ir,res_w.step(i_axis)){
                    const auto ii = ir*kc_;
                    for (index_type kk=0; kk!=kc_; ++kk){
                        const auto e = a_it[kk+ii];
                        const auto jj = kk*nc_;
                        for (index_type jr=0; jr!=nc_; ++jr, res_w.step(j_axis)){
                            *res_w+=e*b_it[jr+jj];
                        }
                        res_w.walk_back(j_axis,nc_);
                    }
                }
                res_w.walk_back(i_axis,mc_);
            };

            do{
                auto res_w = res_tr.walker();
                auto w1 = tr1.walker();
                auto w2 = tr2.walker();

                for (index_type jc=0; jc<n; jc+=nc){
                    w2.walk(j_axis,jc);
                    res_w.walk(j_axis,jc);
                    for (index_type pc=0; pc<k; pc+=kc){
                        w1.walk(j_axis,pc);
                        w2.walk(i_axis,pc);
                        const auto kc_ = make_submatrix_size(pc,kc,k);
                        const auto nc_ = make_submatrix_size(jc,nc,n);
                        fill_buf(w2,b_buf.begin(),kc_,nc_);
                        for (index_type ic=0; ic<m; ic+=mc){
                            w1.walk(i_axis,ic);
                            res_w.walk(i_axis,ic);
                            const auto mc_ = make_submatrix_size(ic,mc,m);
                            fill_buf(w1,a_buf.begin(),mc_,kc_);
                            kernel(res_w,a_buf.cbegin(),b_buf.cbegin(),i_axis,j_axis,mc_,kc_,nc_);
                            w1.walk_back(i_axis,ic);
                            res_w.walk_back(i_axis,ic);
                        }
                        w1.walk_back(j_axis,pc);
                        w2.walk_back(i_axis,pc);
                    }
                    w2.walk_back(j_axis,jc);
                    res_w.walk_back(j_axis,jc);
                }

                tr1.template next<order>();
                tr2.template next<order>();
            }while(res_tr.template next<order>());
        };

        auto core_nd_tiled2 = [k,make_submatrix_size](auto& res_tr, auto& tr1, auto& tr2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n, const auto& mc, const auto& nc, const auto& kc){
            std::array<value_type1,mc_size*kc_size> a_buf;
            std::array<value_type2,kc_size*nc_size> b_buf;

            auto fill_buf = [](auto& w, auto dit, const auto& inner_axis, const auto& outer_axis, const auto& inner_size, const auto& outer_size){
                for (auto i=outer_size; i!=0; --i,w.step(outer_axis)){
                    for (auto j=inner_size; j!=0; --j,w.step(inner_axis),++dit){
                        *dit = *w;
                    }
                    w.walk_back(inner_axis,inner_size);
                }
                w.walk_back(outer_axis,outer_size);
            };

            auto kernel_c = [](auto& res_w, auto a_it, auto b_it, const auto& i_axis, const auto& j_axis, const auto& mc_, const auto& kc_, const auto& nc_){
                for (index_type kk=0; kk!=kc_; ++kk){
                    const auto jj = kk*mc_;
                    const auto ii = kk*nc_;
                    for (index_type ir=0; ir!=mc_; ++ir,res_w.step(i_axis)){
                        const auto e = a_it[ir+jj];
                        for (index_type jr=0; jr!=nc_; ++jr,res_w.step(j_axis)){
                            *res_w+=e*b_it[jr+ii];
                        }
                        res_w.walk_back(j_axis,nc_);
                    }
                    res_w.walk_back(i_axis,mc_);
                }
            };
            auto kernel_f = [](auto& res_w, auto a_it, auto b_it, const auto& i_axis, const auto& j_axis, const auto& mc_, const auto& kc_, const auto& nc_){
                for (index_type kk=0; kk!=kc_; ++kk){
                    const auto jj = kk*mc_;
                    const auto ii = kk*nc_;
                    for (index_type jr=0; jr!=nc_; ++jr, res_w.step(j_axis)){
                        const auto e = b_it[ii+jr];
                        for (index_type ir=0; ir!=mc_; ++ir,res_w.step(i_axis)){
                            *res_w+=a_it[ir+jj]*e;
                        }
                        res_w.walk_back(i_axis,mc_);
                    }
                    res_w.walk_back(j_axis,nc_);
                }
            };

            do{
                auto res_w = res_tr.walker();
                auto w1 = tr1.walker();
                auto w2 = tr2.walker();

                for (index_type jc=0; jc<n; jc+=nc){
                    w2.walk(j_axis,jc);
                    res_w.walk(j_axis,jc);
                    for (index_type pc=0; pc<k; pc+=kc){
                        w1.walk(j_axis,pc);
                        w2.walk(i_axis,pc);
                        const auto kc_ = make_submatrix_size(pc,kc,k);
                        const auto nc_ = make_submatrix_size(jc,nc,n);
                        fill_buf(w2,b_buf.begin(),j_axis,i_axis,nc_,kc_);
                        for (index_type ic=0; ic<m; ic+=mc){
                            w1.walk(i_axis,ic);
                            res_w.walk(i_axis,ic);
                            const auto mc_ = make_submatrix_size(ic,mc,m);
                            fill_buf(w1,a_buf.begin(),i_axis,j_axis,mc_,kc_);
                            if constexpr (std::is_same_v<order,c_order>){
                                kernel_c(res_w,a_buf.cbegin(),b_buf.cbegin(),i_axis,j_axis,mc_,kc_,nc_);
                            }else{
                                kernel_f(res_w,a_buf.cbegin(),b_buf.cbegin(),i_axis,j_axis,mc_,kc_,nc_);
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

                tr1.template next<order>();
                tr2.template next<order>();
            }while(res_tr.template next<order>());
        };

        auto core_nd_tiled1_mt = [k,make_submatrix_size](auto& res_tr, auto& tr1, auto& tr2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n, const auto& mc, const auto& nc, const auto& kc){
            std::array<value_type1,mc_size*kc_size> a_buf;
            std::array<value_type2,kc_size*nc_size> b_buf;

            auto fill_buf = [i_axis,j_axis](auto& w, auto dit, const auto& mc_, const auto& kc_){
                for (auto i=mc_; i!=0; --i,w.step(i_axis)){
                    for (auto j=kc_; j!=0; --j,w.step(j_axis),++dit){
                        *dit = *w;
                    }
                    w.walk_back(j_axis,kc_);
                }
                w.walk_back(i_axis,mc_);
            };

            auto kernel = [i_axis,j_axis](auto& res_w, auto a_it, auto b_it, const auto& mc_, const auto& kc_, const auto& nc_){
                for (index_type ir=0; ir!=mc_; ++ir,res_w.step(i_axis)){
                    const auto ii = ir*kc_;
                    for (index_type kk=0; kk!=kc_; ++kk){
                        const auto e = a_it[kk+ii];
                        const auto jj = kk*nc_;
                        for (index_type jr=0; jr!=nc_; ++jr, res_w.step(j_axis)){
                            *res_w+=e*b_it[jr+jj];
                        }
                        res_w.walk_back(j_axis,nc_);
                    }
                }
                res_w.walk_back(i_axis,mc_);
            };

            auto ic_body = [fill_buf,kernel](auto res_w, auto w1, auto b_it, const auto& mc_, const auto& kc_, const auto& nc_){
                std::array<value_type1,mc_size*kc_size> a_buf;
                fill_buf(w1,a_buf.begin(),mc_,kc_);
                kernel(res_w,a_buf.cbegin(),b_it,mc_,kc_,nc_);
            };

            multithreading::task_group group{};

            do{
                auto res_w = res_tr.walker();
                auto w1 = tr1.walker();
                auto w2 = tr2.walker();

                for (index_type jc=0; jc<n; jc+=nc){
                    w2.walk(j_axis,jc);
                    res_w.walk(j_axis,jc);
                    for (index_type pc=0; pc<k; pc+=kc){
                        w1.walk(j_axis,pc);
                        w2.walk(i_axis,pc);
                        const auto kc_ = make_submatrix_size(pc,kc,k);
                        const auto nc_ = make_submatrix_size(jc,nc,n);
                        fill_buf(w2,b_buf.begin(),kc_,nc_);
                        for (index_type ic=0; ic<m; ic+=mc){
                            w1.walk(i_axis,ic);
                            res_w.walk(i_axis,ic);
                            const auto mc_ = make_submatrix_size(ic,mc,m);

                            multithreading::get_pool().push_group(group,ic_body,res_w,w1,b_buf.cbegin(),mc_,kc_,nc_);
                            // fill_buf(w1,a_buf.begin(),mc_,kc_);
                            // kernel(res_w,a_buf.cbegin(),b_buf.cbegin(),mc_,kc_,nc_);

                            w1.walk_back(i_axis,ic);
                            res_w.walk_back(i_axis,ic);
                        }
                        group.wait();
                        w1.walk_back(j_axis,pc);
                        w2.walk_back(i_axis,pc);
                    }
                    w2.walk_back(j_axis,jc);
                    res_w.walk_back(j_axis,jc);
                }

                tr1.template next<order>();
                tr2.template next<order>();
            }while(res_tr.template next<order>());
        };

        auto core_nd_mt = [](auto core_2d, auto& res_tr, auto& tr1, auto& tr2, const auto& i_axis, const auto& j_axis, const auto& m, const auto& n, const auto& m_, const auto& n_){
            auto make_size_ = [](const auto& i, const auto& b_, const auto& b){return i+b_>b ? b-i : b_;};
            multithreading::task_group group{};
            do{
                auto res_w = res_tr.walker();
                auto w1 = tr1.walker();
                auto w2 = tr2.walker();
                for (index_type i=0; i<m; i+=m_){
                    for (index_type j=0; j<n; j+=n_){
                        res_w.walk(i_axis,i);
                        res_w.walk(j_axis,j);
                        w1.walk(i_axis,i);
                        w2.walk(j_axis,j);
                        multithreading::get_pool().push_group(group,core_2d,res_w,w1,w2,i_axis,j_axis,make_size_(i,m_,m),make_size_(j,n_,n));
                        res_w.reset_back();
                        w1.reset_back();
                        w2.reset_back();
                    }
                }
                tr1.template next<order>();
                tr2.template next<order>();
            }while(res_tr.template next<order>());
            group.wait();
        };

        if constexpr (std::is_same_v<order1,order2>){
            // const auto i_axis = res_dim-2;
            // const auto j_axis = res_dim-1;
            // core_nd_tiled2(res_tr,tr1,tr2,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis],mc,nc,kc);
            if constexpr (std::is_same_v<order1,c_order>){
                const auto i_axis = res_dim-2;
                const auto j_axis = res_dim-1;
                //core_nd_mt(core_2d_cc_ff_mt,res_tr,tr1,tr2,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis],m_,n_);
                //core_nd_tiled(core_2d_cc_ff_tiled,res_tr,tr1,tr2,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis],n_);
                //core_nd(core_2d_cc_ff,res_tr,tr1,tr2,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis]);
                core_nd_tiled1(res_tr,tr1,tr2,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis],mc,nc,kc);
                //core_nd_tiled1_mt(res_tr,tr1,tr2,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis],mc,nc,kc);
            }else{
                const auto i_axis = res_dim-1;
                const auto j_axis = res_dim-2;
                //core_nd_mt(core_2d_cc_ff_mt,res_tr,tr2,tr1,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis],n_,m_);
                //core_nd_tiled(core_2d_cc_ff_tiled,res_tr,tr2,tr1,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis],n_);
                //core_nd(core_2d_cc_ff,res_tr,tr2,tr1,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis]);
                core_nd_tiled1(res_tr,tr2,tr1,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis],nc,mc,kc);
                //core_nd_tiled1_mt(res_tr,tr2,tr1,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis],nc,mc,kc);
            }
        }else{
            const auto i_axis = res_dim-2;
            const auto j_axis = res_dim-1;
            core_nd_tiled2(res_tr,tr1,tr2,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis],mc,nc,kc);
            if constexpr (std::is_same_v<order1,c_order>){
                // const auto i_axis = res_dim-2;
                // const auto j_axis = res_dim-1;
                // core_nd(core_2d_cf,res_tr,tr1,tr2,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis]);
            }else{
                // const auto i_axis = res_dim-2;
                // const auto j_axis = res_dim-1;
                // core_nd(core_2d_fc,res_tr,tr1,tr2,i_axis,j_axis,res_shape[i_axis],res_shape[j_axis]);
            }
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