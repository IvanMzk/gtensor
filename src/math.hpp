#ifndef MATH_HPP_
#define MATH_HPP_
#include <functional>
#include "tensor_operators.hpp"
#include "reduce.hpp"

namespace gtensor{

//return true if two tensors have same shape and close elements within specified tolerance
template<typename...Us, typename...Vs, typename Tol>
inline auto tensor_close(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, Tol relative_tolerance, Tol absolute_tolerance, bool equal_nan = false){
    if (u.is_same(v)){
        return true;
    }else{
        const bool equal_shapes = u.shape() == v.shape();
        if (equal_nan){
            return equal_shapes && std::equal(u.begin(), u.end(), v.begin(), operations::math_isclose<Tol,std::true_type>{relative_tolerance,absolute_tolerance});
        }else{
            return equal_shapes && std::equal(u.begin(), u.end(), v.begin(), operations::math_isclose<Tol,std::false_type>{relative_tolerance,absolute_tolerance});
        }
    }
}
//return true if two tensors have same shape and close elements, use machine epsilon as tolerance
template<typename...Us, typename...Vs>
inline auto tensor_close(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false){
    using common_value_type = detail::tensor_common_value_type_t<basic_tensor<Us...>,basic_tensor<Vs...>>;
    static constexpr common_value_type e = math::numeric_traits<common_value_type>::epsilon();
    return tensor_close(u,v,e,e,equal_nan);
}

//return true if two tensors have close elements within specified tolerance
//shapes may not be equal, but must broadcast
template<typename...Us, typename...Vs, typename Tol>
inline auto allclose(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, Tol relative_tolerance, Tol absolute_tolerance, bool equal_nan = false){
    using shape_type = typename basic_tensor<Us...>::shape_type;
    if (u.is_same(v)){
        return true;
    }else{
        auto common_shape = detail::make_broadcast_shape<shape_type>(u.shape(),v.shape());
        if (equal_nan){
            return std::equal(u.begin(common_shape), u.end(common_shape), v.begin(common_shape), operations::math_isclose<Tol,std::true_type>{relative_tolerance,absolute_tolerance});
        }else{
            return std::equal(u.begin(common_shape), u.end(common_shape), v.begin(common_shape), operations::math_isclose<Tol,std::false_type>{relative_tolerance,absolute_tolerance});
        }
    }
}
//return true if two tensors have close elements, use machine epsilon as tolerance
template<typename...Us, typename...Vs>
inline auto allclose(const basic_tensor<Us...>& u, const basic_tensor<Vs...>& v, bool equal_nan = false){
    using common_value_type = detail::tensor_common_value_type_t<basic_tensor<Us...>,basic_tensor<Vs...>>;
    static constexpr common_value_type e = math::numeric_traits<common_value_type>::epsilon();
    return allclose(u,v,e,e,equal_nan);
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
    typename std::remove_cv_t<std::remove_reference_t<T>>::value_type neg_inf = std::numeric_limits<typename std::remove_cv_t<std::remove_reference_t<T>>::value_type>::lowest()
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
    return n_operator(operations::math_isclose<Tol, EqualNan>{relative_tolerance, absolute_tolerance}, std::forward<T>(t), std::forward<U>(u));
}
template<typename T, typename U, typename EqualNan = std::false_type>
inline auto isclose(T&& t, U&& u, EqualNan equal_nan = EqualNan{}){
    using T_ = std::remove_cv_t<std::remove_reference_t<T>>;
    using U_ = std::remove_cv_t<std::remove_reference_t<U>>;
    using common_value_type = detail::tensor_common_value_type_t<T_,U_>;
    static constexpr common_value_type e = math::numeric_traits<common_value_type>::epsilon();
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

//find extremum propagating nan
//Comparator should be like std::less for minimum and like std::greater for maximum
//Comparator result must be false when any argument is nan
template<typename Comparator>
struct extremum
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        auto res = *first;
        Comparator comparator{};
        if constexpr (gtensor::math::numeric_traits<value_type>::has_nan()){
            if (gtensor::math::isnan(res)){
                return res;
            }
            for(++first; first!=last; ++first){
                const auto& e = *first;
                if (gtensor::math::isnan(e)){
                    return e;
                }else{
                    if (comparator(e,res)){
                        res = e;
                    }
                }
            }
        }else{
            for(++first; first!=last; ++first){
                const auto& e = *first;
                if (comparator(e,res)){
                        res = e;
                }
            }
        }
        return res;
    }
};

//find extremum ignoring nan
//Comparator should be like std::less for minimum and like std::greater for maximum
//Comparator result must be false when any argument is nan
template<typename Comparator>
struct nanextremum
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        if constexpr (gtensor::math::numeric_traits<value_type>::has_nan()){
            //find first not nan
            for(;first!=last; ++first){
                if (!gtensor::math::isnan(*first)){
                    break;
                }
            }
            if (first == last){ //all nans, return last nan
                return --first,*first;
            }
        }
        Comparator comparator{};
        auto res = *first;
        for(++first; first!=last; ++first){
            const auto& e = *first;
            if (comparator(e,res)){   //must be false if e is nan, res always not nan
                res = e;
            }
        }
        return res;
    }
};

using amin = extremum<std::less<void>>;
using amax = extremum<std::greater<void>>;
using nanmin = nanextremum<std::less<void>>;
using nanmax = nanextremum<std::greater<void>>;

//accumulate propagating nan
//Operation should be like std::plus for sum and like std::multiplies for prod
//Operation result must be nan when any argument is nan
template<typename Operation>
struct accumulate
{
    accumulate() = default;
    template<typename Operation_, std::enable_if_t<!std::is_same_v<accumulate,std::remove_cv_t<std::remove_reference_t<Operation_>>> ,int> =0>
    explicit accumulate(Operation_&& operation__):
        operation_{std::forward<Operation_>(operation__)}
    {}

    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        auto res = *first;
        if constexpr (gtensor::math::numeric_traits<value_type>::has_nan()){
            if (gtensor::math::isnan(res)){
                return res;
            }
        }
        for(++first; first!=last; ++first){
            res = operation_(res,*first);   //must return nan if any of arguments is nan
        }
        return res;
    }

    const auto& operation()const{return operation_;}
private:
    Operation operation_{};
};

//accumulate ignoring nan (like treating nan as zero for sum and as one for prod)
//Operation should be like std::plus for sum and like std::multiplies for prod
//Operation result must be nan when any argument is nan
template<typename Operation>
struct nanaccumulate
{
    nanaccumulate() = default;
    template<typename Operation_, std::enable_if_t<!std::is_same_v<nanaccumulate,std::remove_cv_t<std::remove_reference_t<Operation_>>> ,int> =0>
    explicit nanaccumulate(Operation_&& operation__):
        operation_{std::forward<Operation_>(operation__)}
    {}

    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        if constexpr (gtensor::math::numeric_traits<value_type>::has_nan()){
            const auto& e = *first;
            auto res = gtensor::math::isnan(e) ? Operation::template value<value_type> : e;
            for(++first; first!=last; ++first){
                const auto& e = *first;
                if (!gtensor::math::isnan(e)){
                    res = operation_(res,e);
                }
            }
            return res;
        }else{
            auto res = *first;
            for(++first; first!=last; ++first){
                res = operation_(res,*first);
            }
            return res;
        }
    }

    const auto& operation()const{return operation_;}
private:
    Operation operation_{};
};

struct plus : public std::plus<void>{template<typename T> static constexpr T value = T(0);};
struct multiplies : public std::multiplies<void>{template<typename T> static constexpr T value = T(1);};

using sum = accumulate<plus>;
using prod = accumulate<multiplies>;
using nansum = nanaccumulate<plus>;
using nanprod = nanaccumulate<multiplies>;

//cumulate propagating nan
//Operation should be like std::plus for cumsum and like std::multiplies for cumprod
//Operation result must be nan when any argument is nan
template<typename Operation>
struct cumulate
{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        Operation operation{};
        auto res = *first;
        *dfirst = res;
        for(++dfirst,++first; dfirst!=dlast; ++dfirst,++first){
            res=operation(res,*first);
            *dfirst = res;
        }
    }
};

//cumulate ignoring nan (like treating nan as zero for cumsum and as one for cumprod)
//Operation should be like std::plus for cumsum and like std::multiplies for cumprod and has value static member (zero for plus, one for multiplies)
//Operation result must be nan when any argument is nan
template<typename Operation>
struct nancumulate
{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        using value_type = typename std::iterator_traits<It>::value_type;
        Operation operation{};
        if constexpr (gtensor::math::numeric_traits<value_type>::has_nan()){
            const auto& e = *first;
            auto res = gtensor::math::isnan(e) ? Operation::template value<value_type> : e;
            *dfirst = res;
            for(++dfirst,++first; dfirst!=dlast; ++dfirst,++first){
                const auto& e = *first;
                if (!gtensor::math::isnan(e)){
                    res=operation(res,e);
                }
                *dfirst = res;
            }
        }
        else{
            auto res = *first;
            *dfirst = res;
            for(++dfirst,++first; dfirst!=dlast; ++dfirst,++first){
                res=operation(res,*first);
                *dfirst = res;
            }
        }
    }
};

using cumsum = cumulate<plus>;
using cumprod = cumulate<multiplies>;
using nancumsum = nancumulate<plus>;
using nancumprod = nancumulate<multiplies>;

struct mean
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using res_type = std::conditional_t<
            gtensor::math::numeric_traits<value_type>::is_floating_point(),
            value_type,
            typename gtensor::math::numeric_traits<value_type>::floating_point_type
        >;
        const auto n = last - first;
        return static_cast<res_type>(sum{}(first,last)) / static_cast<res_type>(n);
    }
};

struct nanmean
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using res_type = std::conditional_t<
            gtensor::math::numeric_traits<value_type>::is_floating_point(),
            value_type,
            typename gtensor::math::numeric_traits<value_type>::floating_point_type
        >;
        if constexpr (gtensor::math::numeric_traits<value_type>::has_nan()){
            //find first not nan
            for(;first!=last; ++first){
                if (!gtensor::math::isnan(*first)){
                    break;
                }
            }
            if (first == last){ //all nans, return last nan
                return --first,*first;
            }
            auto sum = *first;
            difference_type counter{1};
            for(++first; first!=last; ++first){
                const auto& e = *first;
                if (!gtensor::math::isnan(e)){
                    sum+=e;
                    ++counter;
                }
            }
            return static_cast<res_type>(sum) / static_cast<res_type>(counter);
        }else{
            return mean{}(first,last);
        }
    }
};

struct var
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using res_type = std::conditional_t<
            gtensor::math::numeric_traits<value_type>::is_floating_point(),
            value_type,
            typename gtensor::math::numeric_traits<value_type>::floating_point_type
        >;
        const auto n = static_cast<res_type>(last-first);
        auto sum = static_cast<res_type>(*first);
        auto sum_2 = sum*sum;
        for(++first; first!=last; ++first){
            const auto& e = static_cast<const res_type&>(*first);
            sum+=e;
            sum_2+=e*e;
        }
        return (n*sum_2 - sum*sum)/(n*n);
    }
};

struct nanvar
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using res_type = std::conditional_t<
            gtensor::math::numeric_traits<value_type>::is_floating_point(),
            value_type,
            typename gtensor::math::numeric_traits<value_type>::floating_point_type
        >;
        if constexpr (gtensor::math::numeric_traits<value_type>::has_nan()){
            //find first not nan
            for(;first!=last; ++first){
                if (!gtensor::math::isnan(*first)){
                    break;
                }
            }
            if (first == last){ //all nans, return last nan
                return --first,*first;
            }
            auto sum = static_cast<res_type>(*first);
            auto sum_2 = sum*sum;
            difference_type counter{1};
            for(++first; first!=last; ++first){
                const auto& e = static_cast<const res_type&>(*first);
                if (!gtensor::math::isnan(e)){
                    sum+=e;
                    sum_2+=e*e;
                    ++counter;
                }
            }
            const auto n = static_cast<res_type>(counter);
            return (n*sum_2 - sum*sum)/(n*n);
        }else{
            return var{}(first,last);
        }
    }
};

struct diff_1
{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        for (;dfirst!=dlast;++dfirst){
            auto prev = *first;
            *dfirst = *(++first) - prev;
        }
    }
};

struct diff_2
{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        for (;dfirst!=dlast;++dfirst){
            auto v0 = *first;
            auto v1 = *(++first);
            auto v2 = *(++first);
            *dfirst = v2-v1-v1+v0;
            --first;
        }
    }
};

}   //end of namespace math_reduce_operations

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

//mean of elements along given axes
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto mean(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::mean{},keep_dims);
}
template<typename...Ts>
auto mean(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::mean{},keep_dims);
}
//mean along all axes
template<typename...Ts>
auto mean(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::mean{},keep_dims);
}

//variance of elements along given axes
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto var(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::var{},keep_dims);
}
template<typename...Ts>
auto var(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::var{},keep_dims);
}
//var along all axes
template<typename...Ts>
auto var(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::var{},keep_dims);
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
//none recursive implementation of second differences, more efficient than diff with n=2
template<typename...Ts>
auto diff2(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::dim_type axis = -1){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 3;
    const index_type window_step = 1;
    return slide(t, axis, math_reduce_operations::diff_2{}, window_size, window_step);
}


//nan versions

//min element along given axes ignoring nan
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto nanmin(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nanmin{},keep_dims);
}
template<typename...Ts>
auto nanmin(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nanmin{},keep_dims);
}
//nanmin along all axes
template<typename...Ts>
auto nanmin(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::nanmin{},keep_dims);
}

//max element along given axes ignoring nan
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto nanmax(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nanmax{},keep_dims);
}
template<typename...Ts>
auto nanmax(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nanmax{},keep_dims);
}
//nanmax along all axes
template<typename...Ts>
auto nanmax(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::nanmax{},keep_dims);
}

//sum elements along given axes, treating nan as zero
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto nansum(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nansum{},keep_dims);
}
template<typename...Ts>
auto nansum(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nansum{},keep_dims);
}
//nansum along all axes
template<typename...Ts>
auto nansum(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::nansum{},keep_dims);
}

//multiply elements along given axes, treating nan as one
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto nanprod(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nanprod{},keep_dims);
}
template<typename...Ts>
auto nanprod(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nanprod{},keep_dims);
}
//nanprod along all axes
template<typename...Ts>
auto nanprod(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::nanprod{},keep_dims);
}

//cumulative sum along given axis, treating nan as zero
//axis is scalar
template<typename...Ts, typename DimT>
auto nancumsum(const basic_tensor<Ts...>& t, const DimT& axis){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return slide(t,axis,math_reduce_operations::nancumsum{}, window_size, window_step);
}
//nancumsum along all axes
template<typename...Ts>
auto nancumsum(const basic_tensor<Ts...>& t){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return slide(t,math_reduce_operations::nancumsum{}, window_size, window_step);
}

//cumulative product along given axis, treating nan as one
//axis is scalar
template<typename...Ts, typename DimT>
auto nancumprod(const basic_tensor<Ts...>& t, const DimT& axis){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return slide(t,axis,math_reduce_operations::nancumprod{}, window_size, window_step);
}
//nancumprod along all axes
template<typename...Ts>
auto nancumprod(const basic_tensor<Ts...>& t){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return slide(t,math_reduce_operations::nancumprod{}, window_size, window_step);
}

//mean of elements along given axes, ignoring nan
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto nanmean(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nanmean{},keep_dims);
}
template<typename...Ts>
auto nanmean(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nanmean{},keep_dims);
}
//nanmean along all axes
template<typename...Ts>
auto nanmean(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::nanmean{},keep_dims);
}

//variance of elements along given axes, ignoring nan
//axes may be scalar or container
template<typename...Ts, typename Axes>
auto nanvar(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nanvar{},keep_dims);
}
template<typename...Ts>
auto nanvar(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::dim_type> axes, bool keep_dims = false){
    return reduce(t,axes,math_reduce_operations::nanvar{},keep_dims);
}
//nanvar along all axes
template<typename...Ts>
auto nanvar(const basic_tensor<Ts...>& t, bool keep_dims = false){
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},math_reduce_operations::nanvar{},keep_dims);
}

}   //end of namespace gtensor
#endif