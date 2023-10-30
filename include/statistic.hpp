/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef STATISTIC_HPP_
#define STATISTIC_HPP_

#include <functional>
#include <algorithm>
#include "math.hpp"
#include "reduce.hpp"
#include "tensor_math.hpp"
#include "reduce_operations.hpp"

namespace gtensor{

namespace detail{

template<typename P, typename U=void> constexpr bool is_pair_v = false;
template<typename P> constexpr bool is_pair_v<P,std::void_t<decltype(std::declval<P>().first),decltype(std::declval<P>().second)>> = true;

template<typename P, typename T, typename U=void> constexpr bool is_pair_of_type_v = false;
template<typename P, typename T> constexpr bool is_pair_of_type_v<P,T,std::void_t<std::enable_if_t<is_pair_v<P>>>> =
    std::is_convertible_v<decltype(std::declval<P>().first),T> && std::is_convertible_v<decltype(std::declval<P>().second),T>;

}

enum class histogram_algorithm : std::size_t {automatic,fd,scott,rice,sturges,sqrt};

//tensor statistic functions implementation

#define GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(NAME,RANGE_F,BINARY_F)\
template<typename Policy, typename...Ts, typename Axes>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    if constexpr (multithreading::exec_policy_traits<Policy>::is_seq::value){\
        return BINARY_F{}(policy,t,axes,keep_dims);\
    }else{\
        return reduce_range(policy,t,axes,RANGE_F{},keep_dims,true);\
    }\
}\
template<typename Policy, typename...Ts>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return BINARY_F{}(policy,t,keep_dims);\
}\
template<typename...Ts, typename Axes>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    return NAME(multithreading::exec_pol<1>{},t,axes,keep_dims);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return NAME(multithreading::exec_pol<1>{},t,keep_dims);\
}

#define GTENSOR_TENSOR_STATISTIC_QUANTILE_NANQUANTILE_FUNCTION(NAME,RANGE_F)\
template<typename Policy, typename...Ts, typename Axes, typename Q>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    static_assert(math::numeric_traits<Q>::is_floating_point(),"q must be of floating point type");\
    return reduce_range(policy,t,axes,RANGE_F{},keep_dims,false,q,config_type{});\
}\
template<typename...Ts, typename Axes, typename Q>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){\
    return NAME(multithreading::exec_pol<1>{},t,axes,q,keep_dims);\
}\
template<typename Policy, typename...Ts, typename Q>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Q& q, bool keep_dims = false){\
    return NAME(policy,t,detail::no_value{},q,keep_dims);\
}\
template<typename...Ts, typename Q>\
static auto NAME(const basic_tensor<Ts...>& t, const Q& q, bool keep_dims = false){\
    return NAME(multithreading::exec_pol<1>{},t,q,keep_dims);\
}\

#define GTENSOR_TENSOR_STATISTIC_MEDIAN_NANMEDIAN_FUNCTION(NAME,QUANTILE)\
template<typename Policy, typename...Ts, typename Axes>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    using value_type = typename basic_tensor<Ts...>::value_type;\
    return QUANTILE(policy,t,axes,gtensor::math::make_floating_point_like_t<value_type>{0.5},keep_dims);\
}\
template<typename...Ts, typename Axes>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    return NAME(multithreading::exec_pol<1>{},t,axes,keep_dims);\
}\
template<typename Policy, typename...Ts>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return NAME(policy,t,detail::no_value{},keep_dims);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return NAME(multithreading::exec_pol<1>{},t,keep_dims);\
}


struct statistic
{
private:
    struct ptp_binary{
        template<typename Policy, typename...Ts,typename Axes>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){
            return (max(policy,t,axes,keep_dims) - min(policy,t,axes,keep_dims)).copy();
        }
        template<typename Policy, typename...Ts>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims){
            return (max(policy,t,keep_dims) - min(policy,t,keep_dims)).copy();
        }
    };
    struct mean_binary{
        template<typename Policy, typename...Ts,typename Axes>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){
            using element_type = typename basic_tensor<Ts...>::element_type;
            using integral_type = gtensor::math::make_integral_t<element_type>;
            using fp_type = gtensor::math::make_floating_point_like_t<element_type>;
            using res_value_type = typename detail::copy_type_t<basic_tensor<Ts...>,fp_type>::value_type;
            using f_type = gtensor::math_reduce_operations::nan_propagate_operation<gtensor::math_reduce_operations::plus<void>>;
            auto tmp = reduce_binary(policy,t,axes,f_type{},keep_dims,res_value_type(0));
            if (!tmp.empty()){
                if (t.empty()){ //reduce zero size dimension
                    if constexpr (gtensor::math::numeric_traits<res_value_type>::has_nan()){
                        auto a = tmp.traverse_order_adapter(typename decltype(tmp)::order{});
                        std::fill(a.begin(),a.end(),gtensor::math::numeric_traits<res_value_type>::nan());
                    }else{
                        throw value_error("cant reduce zero size dimension without initial value");
                    }
                }else{
                    const auto axes_size = t.size()/tmp.size();
                    tmp/=static_cast<const fp_type&>(static_cast<const integral_type&>(axes_size));
                }
            }
            return tmp;
        }
        template<typename Policy, typename...Ts>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims){
            return this->operator()(policy,t,detail::no_value{},keep_dims);
        }
    };
    struct nanmean_binary{
        template<typename Policy, typename...Ts,typename Axes>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){
            using order = typename basic_tensor<Ts...>::order;
            using value_type = typename basic_tensor<Ts...>::value_type;
            using config_type = typename basic_tensor<Ts...>::config_type;
            using integral_type = gtensor::math::make_integral_t<value_type>;
            using res_type = gtensor::math::make_floating_point_like_t<value_type>;
            using f_type = gtensor::statistic_reduce_operations::nan_ignoring_counting_operation<gtensor::math_reduce_operations::plus<res_type>,integral_type>;
            auto tmp = reduce_binary(policy,t,axes,f_type{},keep_dims,std::make_pair(res_type{0},integral_type{0}));
            tensor<res_type,order,config_type> res(tmp.shape());
            std::transform(tmp.begin(),tmp.end(),res.begin(),
                [](const auto& r){
                    if constexpr (gtensor::math::numeric_traits<res_type>::has_nan()){
                        return r.first/static_cast<const res_type&>(r.second);
                    }else{
                        if (r.second==0){
                            throw value_error("cant reduce zero size dimension without initial value");
                        }
                    }
                }
            );
            return res;
        }
        template<typename Policy, typename...Ts>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims){
            return this->operator()(policy,t,detail::no_value{},keep_dims);
        }
    };
    struct var_binary{
        template<typename Policy, typename...Ts,typename Axes>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){
            using element_type = typename basic_tensor<Ts...>::element_type;
            auto squared_diff = [](const auto& e, const auto& m){
                if constexpr (math::is_complex_v<element_type>){
                    const auto d=statistic_reduce_operations::abs_helper(e-m);
                    return d*d;
                }else{
                    const auto d=e-m;
                    return d*d;
                }
            };
            auto mean_ = mean_binary{}(policy,t,axes,true);
            auto tmp = gtensor::n_operator(squared_diff,t,std::move(mean_));
            return mean_binary{}(policy,tmp,axes,keep_dims);
        }
        template<typename Policy, typename...Ts>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims){
            return this->operator()(policy,t,detail::no_value{},keep_dims);
        }
    };
    struct nanvar_binary{
        template<typename Size>
        struct nan_ignoring_counting_plus
        {
            template<typename T, typename R>
            auto operator()(const std::tuple<T,R>& e1, const std::tuple<T,R>& e2){
                const auto e1_not_nan = gtensor::math::isnan(std::get<0>(e1));
                const auto e2_not_nan = gtensor::math::isnan(std::get<0>(e2));
                if (e1_not_nan && e2_not_nan){
                    return std::make_pair(std::get<1>(e1)+std::get<1>(e2),Size{2});
                }else if (e1_not_nan){
                    return std::make_pair(std::get<1>(e1),Size{1});
                }else if (e2_not_nan){
                    return std::make_pair(std::get<1>(e2),Size{1});
                }else{
                    return std::make_pair(R(0),Size{0});
                }
            }
            template<typename R>
            auto operator()(const std::pair<R,Size>& r1, const std::pair<R,Size>& r2){
                return std::make_pair(r1.first+r2.first,r1.second+r2.second);
            }
            template<typename R, typename T>
            auto operator()(const std::pair<R,Size>& r, const std::tuple<T,R>& e){
                return gtensor::math::isnan(std::get<0>(e)) ? r : std::make_pair(r.first+std::get<1>(e),r.second+Size{1});
            }
            template<typename R, typename T>
            auto operator()(const std::tuple<T,R>& e, const std::pair<R,Size>& r){
                return this->operator()(r,e);
            }
        };

        template<typename Policy, typename...Ts,typename Axes>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){
            using order = typename basic_tensor<Ts...>::order;
            using value_type = typename basic_tensor<Ts...>::value_type;
            using config_type = typename basic_tensor<Ts...>::config_type;
            using integral_type = gtensor::math::make_integral_t<value_type>;
            using res_type = gtensor::math::make_floating_point_like_t<value_type>;
            auto mean_ = nanmean_binary{}(policy,t,axes,true);
            auto squared_diff = [](const auto& e, const auto& m){const auto d=e-m; return d*d;};
            auto make_tuple = [](const auto& e1,const auto& e2){return std::make_tuple(e1,e2);};
            auto tmp = gtensor::n_operator(make_tuple,t,gtensor::n_operator(squared_diff,t,std::move(mean_)));
            using f_type = nan_ignoring_counting_plus<integral_type>;
            auto sum = reduce_binary(tmp,axes,f_type{},keep_dims,std::make_pair(res_type{0},integral_type{0}));
            tensor<res_type,order,config_type> res(sum.shape());
            std::transform(sum.begin(),sum.end(),res.begin(),
                [](const auto& r){
                    if constexpr (gtensor::math::numeric_traits<res_type>::has_nan()){
                        return r.first/static_cast<const res_type&>(r.second);
                    }else{
                        if (r.second==0){
                            throw value_error("cant reduce zero size dimension without initial value");
                        }
                    }
                }
            );
            return res;
        }
        template<typename Policy, typename...Ts>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims){
            return this->operator()(policy,t,detail::no_value{},keep_dims);
        }
    };
    struct stdev_binary{
        template<typename Policy, typename...Ts,typename Axes>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){
            return statistic_reduce_operations::sqrt_helper(var_binary{}(policy,t,axes,keep_dims));
        }
        template<typename Policy, typename...Ts>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims){
            return this->operator()(policy,t,detail::no_value{},keep_dims);
        }
    };
    struct nanstdev_binary{
        template<typename Policy, typename...Ts,typename Axes>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims){
            return sqrt(nanvar_binary{}(policy,t,axes,keep_dims));
        }
        template<typename Policy, typename...Ts>
        auto operator()(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims){
            return this->operator()(policy,t,detail::no_value{},keep_dims);
        }
    };

public:
    //statistic functions along given axis or axes
    //axes may be scalar or container if multiple axes permitted
    //empty container means apply function along all axes

    //peak-to-peak of elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(ptp,statistic_reduce_operations::ptp,ptp_binary);

    //mean of elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(mean,statistic_reduce_operations::mean,mean_binary);

    //variance of elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(var,statistic_reduce_operations::var,var_binary);

    //standart deviation of elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(stdev,statistic_reduce_operations::stdev,stdev_binary);

    //quantile of elements along given axes
    //axes may be scalar or container
    //q must be of floating point type in range [0,1]
    GTENSOR_TENSOR_STATISTIC_QUANTILE_NANQUANTILE_FUNCTION(quantile,statistic_reduce_operations::quantile);

    //median of elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_MEDIAN_NANMEDIAN_FUNCTION(median,quantile);

    //nan versions
    //mean of elements along given axes, ignoring nan
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(nanmean,statistic_reduce_operations::nanmean,nanmean_binary);

    //variance of elements along given axes, ignoring nan
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(nanvar,statistic_reduce_operations::nanvar,nanvar_binary);

    //standart deviation of elements along given axes, ignoring nan
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(nanstdev,statistic_reduce_operations::nanstdev,nanstdev_binary);

    //quantile of elements along given axes, ignoring nan
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_QUANTILE_NANQUANTILE_FUNCTION(nanquantile,statistic_reduce_operations::nanquantile);

    //median of elements along given axes, ignoring nan
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_MEDIAN_NANMEDIAN_FUNCTION(nanmedian,nanquantile);

    //average along given axes
    //axes may be scalar or container
    //weights is container, size of weights must be size along given axes, weights must not sum to zero
    template<typename Policy, typename...Ts, typename Axes, typename Container>
    static auto average(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, const Container& weights, bool keep_dims=false){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return reduce_range(policy,t,axes,statistic_reduce_operations::average<value_type>{},keep_dims,false,weights);
    }
    template<typename...Ts, typename Axes, typename Container>
    static auto average(const basic_tensor<Ts...>& t, const Axes& axes, const Container& weights, bool keep_dims=false){
        return average(multithreading::exec_pol<1>{},t,axes,weights,keep_dims);
    }
    //like over flatten
    template<typename Policy, typename...Ts, typename Container>
    static auto average(Policy policy, const basic_tensor<Ts...>& t, const Container& weights, bool keep_dims=false){
        return average(policy,t,detail::no_value{},weights,keep_dims);
    }
    template<typename...Ts, typename Container>
    static auto average(const basic_tensor<Ts...>& t, const Container& weights, bool keep_dims=false){
        return average(multithreading::exec_pol<1>{},t,weights,keep_dims);
    }

    //moving average along given axis, axis is scalar
    //weights is container, moving window size is weights size, weights must not sum to zero
    //result axis size will be (n - window_size)/step + 1, where n is source axis size
    template<typename Policy, typename...Ts, typename Axis, typename Container, typename IdxT>
    static auto moving_average(Policy policy, const basic_tensor<Ts...>& t, const Axis& axis, const Container& weights, const IdxT& step){
        using index_type = typename basic_tensor<Ts...>::index_type;
        using value_type = typename basic_tensor<Ts...>::value_type;
        using res_type = gtensor::math::make_floating_point_like_t<value_type>;
        const auto window_size = static_cast<index_type>(weights.size());
        const auto window_step = static_cast<index_type>(step);
        return slide<res_type>(policy,t,axis,statistic_reduce_operations::moving_average<value_type>{},window_size,window_step,weights,window_step);
    }
    template<typename...Ts, typename Axis, typename Container, typename IdxT>
    static auto moving_average(const basic_tensor<Ts...>& t, const Axis& axis, const Container& weights, const IdxT& step){
        return moving_average(multithreading::exec_pol<1>{},t,axis,weights,step);
    }
    //like over flatten
    template<typename Policy, typename...Ts, typename Container, typename IdxT>
    static auto moving_average(Policy policy, const basic_tensor<Ts...>& t, const Container& weights, const IdxT& step){
        return moving_average(policy,t,detail::no_value{},weights,step);
    }
    template<typename...Ts, typename Container, typename IdxT>
    static auto moving_average(const basic_tensor<Ts...>& t, const Container& weights, const IdxT& step){
        return moving_average(multithreading::exec_pol<1>{},t,weights,step);
    }

    //moving mean along given axis, axis is scalar
    //window_size must be greater zero and less_equal than axis size
    //result axis size will be (n - window_size)/step + 1, where n is source axis size
    template<typename Policy, typename...Ts, typename Axis, typename IdxT>
    static auto moving_mean(Policy policy, const basic_tensor<Ts...>& t, const Axis& axis, const IdxT& window_size, const IdxT& step){
        using value_type = typename basic_tensor<Ts...>::value_type;
        using res_type = gtensor::math::make_floating_point_like_t<value_type>;
        return slide<res_type>(policy,t,axis,statistic_reduce_operations::moving_mean{},window_size,step,window_size,step);
    }
    template<typename...Ts, typename Axis, typename IdxT>
    static auto moving_mean(const basic_tensor<Ts...>& t, const Axis& axis, const IdxT& window_size, const IdxT& step){
        return moving_mean(multithreading::exec_pol<1>{},t,axis,window_size,step);
    }
    //like over flatten
    template<typename Policy, typename...Ts, typename IdxT>
    static auto moving_mean(Policy policy, const basic_tensor<Ts...>& t, const IdxT& window_size, const IdxT& step){
        return moving_mean(policy,t,detail::no_value{},window_size,step);
    }
    template<typename...Ts, typename IdxT>
    static auto moving_mean(const basic_tensor<Ts...>& t, const IdxT& window_size, const IdxT& step){
        return moving_mean(multithreading::exec_pol<1>{},t,window_size,step);
    }

    //histogram
    //Bins can be integral type or histogram_algorithm enum or container with at least bidirectional iterator
    //when bins is of integral type it means equal width bins number, bins must be > 0
    //when bins is of histogram_algorithm type equal width bins number is calculated according to algorithm
    //when bins is container its elements mean bins edges and must increase monotonically
    //Range is pair like type, means numeric range [first,second], first must be less or equal than second
    //weights is tensor of same shape as t
    template<typename...Ts, typename Bins, typename Range, typename Weights>
    static auto histogram(const basic_tensor<Ts...>& t, const Bins& bins, const Range& range, bool density, const Weights& weights){
        using config_type = typename basic_tensor<Ts...>::config_type;
        using value_type = typename basic_tensor<Ts...>::value_type;
        using fp_type = gtensor::math::make_floating_point_like_t<value_type>;
        using index_type = typename basic_tensor<Ts...>::index_type;
        using order = typename basic_tensor<Ts...>::order;
        using res_value_type = fp_type;
        using res_config_type = config::extend_config_t<config_type,res_value_type>;
        using res_type = tensor<res_value_type,order,res_config_type>;
        using container_type = typename config_type::template container<fp_type>;
        using container_pair_type = typename config_type::template container<std::pair<fp_type,fp_type>>;
        static constexpr bool has_weights = detail::is_tensor_of_type_v<Weights,fp_type>;
        static_assert(has_weights || std::is_same_v<Weights,detail::no_value>,"invalid weights argument");
        static_assert(math::numeric_traits<Bins>::is_integral() || std::is_same_v<Bins,histogram_algorithm> || detail::is_container_of_type_v<Bins,fp_type>, "Bins must be integral or histogram_algorithm or container type");
        static_assert(detail::is_pair_of_type_v<Range,fp_type> || std::is_same_v<Range,detail::no_value>,"invalid range argument");

        auto weights_begin = [&weights]{
            (void)weights;
            if constexpr (has_weights){
                return weights.traverse_order_adapter(order{}).begin();
            }else{
                return detail::no_value{};
            }
        };
        auto weights_end = [&weights]{
            (void)weights;
            if constexpr (has_weights){
                return weights.traverse_order_adapter(order{}).end();
            }else{
                return detail::no_value{};
            }
        };

        check_histogram_arguments(t,bins,range,weights);
        auto a = t.traverse_order_adapter(order{});
        if constexpr (detail::is_container_of_type_v<Bins, value_type>){    //not uniform bin width, need copy and sort data, range doesnt matter
            const auto edges_number = static_cast<index_type>(bins.size());
            auto edges_first = bins.begin();
            auto edges_last = bins.end();
            if (edges_number < 2){  //too few edges
                return std::make_pair(res_type{},res_type({edges_number},edges_first,edges_last));
            }
            const auto& rmin = static_cast<const fp_type&>(*edges_first);
            const auto& rmax = static_cast<const fp_type&>(*--edges_last);
            auto is_in_range = [rmin,rmax](const auto& e){return e>=rmin && e<=rmax;};
            if constexpr (has_weights){
                container_pair_type elements_weights{};
                detail::reserve(elements_weights,t.size());
                auto weights_it = weights_begin();
                for (auto it=a.begin(),last=a.end(); it!=last; ++it,++weights_it){
                    const auto& e = static_cast<const fp_type&>(*it);
                    if (is_in_range(e)){
                        elements_weights.emplace_back(e,static_cast<const fp_type&>(*weights_it));
                    }
                }
                return make_non_uniform_histogram<res_type>(elements_weights.begin(),elements_weights.end(),bins,density);
            }else{  //no weights
                container_type elements{};
                detail::reserve(elements,t.size());
                std::copy_if(a.begin(),a.end(),std::back_inserter(elements),is_in_range);
                return make_non_uniform_histogram<res_type>(elements.begin(),elements.end(),bins,density);
            }
        }else{  //uniform, bins is integral or enum
            fp_type min{0};
            fp_type max{1};
            if constexpr (std::is_same_v<Range,detail::no_value>){  //range is t.min(), t.max(),
                if (t.empty()){
                    return make_uniform_histogram<res_type>(a.begin(), a.end(), weights_begin(), weights_end(), bins, min, max, min, max, density);
                }
                const auto min_max = statistic_reduce_operations::min_max{}(a.begin(),a.end());
                min = static_cast<fp_type>(min_max.first);
                max = static_cast<fp_type>(min_max.second);
                auto rmin = min;
                auto rmax = max;
                if (rmin == rmax){
                    rmin-=0.5;
                    rmax+=0.5;
                }
                return make_uniform_histogram<res_type>(a.begin(), a.end(), weights_begin(), weights_end(), bins, min, max, rmin, rmax, density);
            }else{
                auto rmin = static_cast<fp_type>(range.first);
                auto rmax = static_cast<fp_type>(range.second);
                if (rmin == rmax){
                    rmin-=0.5;
                    rmax+=0.5;
                }
                if (t.empty()){
                    return make_uniform_histogram<res_type>(a.begin(), a.end(), weights_begin(), weights_end(), bins, min, max, rmin, rmax, density);
                }
                auto is_in_range = [rmin,rmax](const auto& e){return e>=rmin && e<=rmax;};
                if constexpr (has_weights){
                    container_type elements_{};
                    container_type weights_{};
                    detail::reserve(elements_,t.size());
                    detail::reserve(weights_,t.size());
                    auto it = a.begin();
                    auto weights_it = weights_begin();
                    min = *it;
                    max = min;
                    for (auto last=a.end(); it!=last; ++it,++weights_it){
                        const auto& e = static_cast<const fp_type&>(*it);
                        if (is_in_range(e)){
                            elements_.push_back(e);
                            weights_.push_back(static_cast<const fp_type&>(*weights_it));
                            if (e<min){
                                min=e;
                            }else if (e>max){
                                max=e;
                            }
                        }
                    }
                    return make_uniform_histogram<res_type>(elements_.begin(), elements_.end(), weights_.begin(), weights_.end(), bins, min, max, rmin, rmax, density);
                }else{  //no weights
                    container_type elements_{};
                    detail::reserve(elements_,t.size());
                    auto it = a.begin();
                    min = *it;
                    max = min;
                    for (auto last=a.end(); it!=last; ++it){
                        const auto& e = static_cast<const fp_type&>(*it);
                        if (is_in_range(e)){
                            elements_.push_back(e);
                            if (e<min){
                                min=e;
                            }else if (e>max){
                                max=e;
                            }
                        }
                    }
                    return make_uniform_histogram<res_type>(elements_.begin(), elements_.end(), weights_begin(), weights_end(), bins, min, max, rmin, rmax, density);
                }
            }
        }

    }

    template<typename...Ts, typename Bins>
    static auto histogram(const basic_tensor<Ts...>& t, const Bins& bins = 10, bool density = false){
        return histogram(t,bins,detail::no_value{},density,detail::no_value{});
    }
private:

    template<typename... Ts, typename Bins, typename Range, typename Weights>
    static void check_histogram_arguments(const basic_tensor<Ts...>& t, const Bins& bins, const Range& range, const Weights& weights){
        (void)t;
        (void)bins;
        (void)range;
        (void)weights;
        if constexpr (math::numeric_traits<Bins>::is_integral()){
            if (bins <= 0){
                throw value_error("bins must be positive when an integral");
            }
        }
        if constexpr (detail::is_tensor_v<Bins>){
            if (bins.dim() != 1){
                throw value_error("bins must be 1d when a tensor");
            }
        }
        if constexpr (detail::is_pair_v<Range>){
            if (range.first > range.second){
                throw value_error("second must be larger or equal than first in a range");
            }
        }
        if constexpr (detail::is_tensor_v<Weights>){
            if (t.shape() != weights.shape()){
                throw value_error("weights must have the same shape as t");
            }
        }
        if constexpr (detail::is_container_v<Bins>){
            if (bins.size() > 0){
                auto it=bins.begin();
                auto last=bins.end();
                auto prev = *it;
                for (++it; it!=last; ++it){
                    const auto& next = *it;
                    if (prev > next){
                        throw value_error("bins must increase monotonically when a tensor");
                    }
                    prev = next;
                }
            }
        }
    }

    //first,last - range of elements or element,weight pairs within specified edges
    //edges is container of bins edges, must have at least two edges
    template<typename ResultT, typename It, typename Container>
    static auto make_non_uniform_histogram(It first, It last, const Container& edges, bool density){
        using res_type = ResultT;
        using order = typename res_type::order;
        using index_type = typename res_type::index_type;
        using res_value_type = typename res_type::value_type;
        using integral_type = gtensor::math::make_integral_t<res_value_type>;
        using it_value_type = typename std::iterator_traits<It>::value_type;
        static constexpr bool has_weights = detail::is_pair_v<it_value_type>;

        const auto n = static_cast<res_value_type>(static_cast<integral_type>(last - first));
        std::sort(
            first,
            last,
            [](const auto& l, const auto& r){
                if constexpr (has_weights){
                    return l.first < r.first;
                }else{
                    return l < r;
                }
            }
        );
        const auto edges_number = static_cast<index_type>(edges.size());
        res_type res_bins({edges_number-1},res_value_type{0});
        auto a = res_bins.traverse_order_adapter(order{});
        {
            auto res_bins_it = a.begin();
            auto edges_it=edges.begin();
            auto edges_last=edges.end();
            auto edge_min = static_cast<res_value_type>(*edges_it);
            for (++edges_it,--edges_last; edges_it!=edges_last; ++edges_it,++res_bins_it){
                auto edge_max = static_cast<res_value_type>(*edges_it);
                for (;first!=last; ++first){
                    if constexpr (has_weights){
                        const auto& e = (*first).first;
                        if (e>=edge_min && e<edge_max){
                            *res_bins_it+=(*first).second;
                        }else{
                            break;
                        }
                    }else{
                        const auto& e = *first;
                        if (e>=edge_min && e<edge_max){
                            *res_bins_it+=res_value_type{1};
                        }else{
                            break;
                        }
                    }
                }
                edge_min = edge_max;
            }
            //last bin, include upper edge
            const auto edge_max = static_cast<res_value_type>(*edges_it);
            for (;first!=last; ++first){
                if constexpr (has_weights){
                    const auto& e = (*first).first;
                    if (e>=edge_min && e<=edge_max){
                        *res_bins_it+=(*first).second;
                    }else{
                        break;
                    }
                }else{
                    const auto& e = *first;
                    if (e>=edge_min && e<=edge_max){
                        *res_bins_it+=res_value_type{1};
                    }else{
                        break;
                    }
                }
            }
        }
        if (density){
            auto edges_it=edges.begin();
            auto edge_min = static_cast<res_value_type>(*edges_it);
            ++edges_it;
            for (auto res_bins_it=a.begin(),res_bins_last=a.end(); res_bins_it!=res_bins_last; ++res_bins_it,++edges_it){
                auto edge_max = static_cast<res_value_type>(*edges_it);
                *res_bins_it/=n*(edge_max-edge_min);
                edge_min = edge_max;
            }
        }
        return std::make_pair(std::move(res_bins),res_type({edges_number},edges.begin(),edges.end()));
    }

    //first,last - elements range that are within histogram range argument, all elements range if histogram range argument is no_value
    template<typename ResultT, typename It, typename WeightsIt, typename Bins, typename T>
    static auto make_uniform_histogram(It first, It last, WeightsIt wfirst, WeightsIt, const Bins& bins, const T& min, const T& max, const T& rmin, const T& rmax, bool density){

        using res_type = ResultT;
        using order = typename res_type::order;
        using config_type = typename res_type::config_type;
        using index_type = typename res_type::index_type;
        using res_value_type = typename res_type::value_type;
        using integral_type = gtensor::math::make_integral_t<res_value_type>;

        static constexpr bool has_weights = !std::is_same_v<WeightsIt,detail::no_value>;

        const auto n = static_cast<res_value_type>(static_cast<integral_type>(last - first));
        const auto bins_ = make_bins<config_type>(first,last,bins,min,max,rmin,rmax);
        const auto bin_width = bins_.first;
        const auto bins_number = static_cast<index_type>(static_cast<integral_type>(bins_.second));

        res_type res_bins({bins_number},res_value_type{0});
        auto a = res_bins.traverse_order_adapter(order{});
        auto res_bins_indexer = a.create_indexer();
        if constexpr (has_weights){
            for (;first!=last; ++first,++wfirst){
                auto i = static_cast<index_type>(static_cast<integral_type>((*first - rmin)/bin_width));
                if (i == bins_number){  //add elements that equal to range upper boundary to last bin (last bin includes upper boundary)
                    --i;
                }
                res_bins_indexer[i]+=*wfirst;
            }
        }else{
            for (;first!=last; ++first){
                auto i = static_cast<index_type>(static_cast<integral_type>((*first - rmin)/bin_width));
                if (i == bins_number){  //add elements that equal to range upper boundary to last bin (last bin includes upper boundary)
                    --i;
                }
                res_bins_indexer[i]+=res_value_type{1};
            }
        }
        //normalize res_bins if density is true
        if (density){
            const auto normalizer = 1/(n*bin_width);
            std::for_each(a.begin(),a.end(),[normalizer](auto& e){e*=normalizer;});
        }
        //make intervals
        res_type res_intervals({bins_number+index_type{1}},res_value_type{rmin});
        auto res_intervals_a = res_intervals.traverse_order_adapter(order{});
        auto res_intervals_it = res_intervals_a.begin();
        auto res_intervals_last = res_intervals_a.end();
        auto edge = *res_intervals_it;
        for(++res_intervals_it,--res_intervals_last; res_intervals_it!=res_intervals_last; ++res_intervals_it){
            edge+=bin_width;
            *res_intervals_it = edge;
        }
        *res_intervals_it = rmax;
        return std::make_pair(std::move(res_bins),std::move(res_intervals));
    }

    template<typename Config, typename It, typename Bins, typename T>
    static auto make_bins(It first, It last, const Bins& bins, const T& min, const T& max, const T& rmin, const T& rmax){
        using value_type = typename std::iterator_traits<It>::value_type;
        using fp_type = gtensor::math::make_floating_point_like_t<value_type>;

        if constexpr (std::is_same_v<Bins,histogram_algorithm>){
            if (first == last){ //cant use bin width estimators
                const auto nh = static_cast<fp_type>(1);
                const auto h = (rmax-rmin)/nh;
                return std::make_pair(h,nh);
            }
            const auto h_ = make_bin_width<Config>(first,last,bins,min,max);
            const auto nh = gtensor::math::ceil((rmax-rmin)/h_);
            const auto h = (rmax-rmin)/nh;
            return std::make_pair(h,nh);
        }else{
            const auto nh = static_cast<fp_type>(bins);
            const auto h = (rmax-rmin)/nh;
            return std::make_pair(h,nh);
        }
    }

    template<typename Config, typename It, typename T>
    static auto make_bin_width(It first, It last, histogram_algorithm bins, const T& min, const T& max){
        using value_type = typename std::iterator_traits<It>::value_type;
        using integral_type = gtensor::math::make_integral_t<value_type>;
        using fp_type = gtensor::math::make_floating_point_like_t<value_type>;

        const auto n = static_cast<fp_type>(static_cast<integral_type>(last - first));
        auto make_sturges = [n,min,max]{return (max-min)/(gtensor::math::log2(n)+1);};
        auto make_fd = [n,first,last]{
            statistic_reduce_operations::quantile quantile_maker{};
            const auto q1 = quantile_maker(first,last,0.25,Config{});
            const auto q3 = quantile_maker(first,last,0.75,Config{});
            return 2*(q3-q1)/gtensor::math::cbrt(n);
        };
        switch (bins){
            case histogram_algorithm::automatic:
            {
                const auto h_fd = make_fd();
                const auto h_sturges = make_sturges();
                if (h_fd!=fp_type{0}){
                    return std::min(h_fd, h_sturges);
                }else{
                    return h_sturges;
                }
            }
            case histogram_algorithm::fd:
                return make_fd();
            case histogram_algorithm::scott:
            {
                statistic_reduce_operations::stdev stdev_maker{};
                const auto stddev = stdev_maker(first,last);
                return stddev*gtensor::math::cbrt(24*gtensor::math::sqrt(gtensor::math::numeric_constants<fp_type>::pi())/n);
            }
            case histogram_algorithm::rice:
                return (max-min)/(gtensor::math::cbrt(n)*2);
            case histogram_algorithm::sqrt:
                return (max-min)/gtensor::math::sqrt(n);
            case histogram_algorithm::sturges:
                return make_sturges();
            default:
                throw value_error("invalid histogram_algorithm argument");
        };
    }

};  //end of struct statistic

//tensor statistic frontend
//frontend uses compile-time dispatch to select implementation, see module_selector.hpp

#define GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(NAME,F)\
template<typename Policy, typename...Ts, typename Axes>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(policy,t,axes,keep_dims);\
}\
template<typename Policy, typename...Ts, typename DimT>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(policy,t,axes,keep_dims);\
}\
template<typename Policy, typename...Ts>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(policy,t,keep_dims);\
}\
template<typename...Ts, typename Axes>\
auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(t,axes,keep_dims);\
}\
template<typename...Ts, typename DimT>\
auto NAME(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(t,axes,keep_dims);\
}\
template<typename...Ts>\
auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(t,keep_dims);\
}

#define GTENSOR_TENSOR_STATISTIC_QUANTILE_NANQUANTILE_ROUTINE(NAME,F)\
template<typename Policy, typename...Ts, typename Axes, typename Q>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(policy,t,axes,q,keep_dims);\
}\
template<typename Policy, typename...Ts, typename DimT, typename Q>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Q& q, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(policy,t,axes,q,keep_dims);\
}\
template<typename Policy, typename...Ts, typename Q>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Q& q, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(policy,t,q,keep_dims);\
}\
template<typename...Ts, typename Axes, typename Q>\
auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(t,axes,q,keep_dims);\
}\
template<typename...Ts, typename DimT, typename Q>\
auto NAME(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Q& q, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(t,axes,q,keep_dims);\
}\
template<typename...Ts, typename Q>\
auto NAME(const basic_tensor<Ts...>& t, const Q& q, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return statistic_selector_t<config_type>::F(t,q,keep_dims);\
}

//peak-to-peak of elements along given axes
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(ptp,ptp);

//mean of elements along given axes
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(mean,mean);

//variance of elements along given axes
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(var,var);

//standart deviation of elements along given axes
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(stdev,stdev);

//median of elements along given axes
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(median,median);

//quantile of elements along given axes
//axes may be scalar or container
//q must be of floatin point type in range [0,1]
GTENSOR_TENSOR_STATISTIC_QUANTILE_NANQUANTILE_ROUTINE(quantile,quantile);

//nan versions
//mean of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(nanmean,nanmean);

//variance of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(nanvar,nanvar);

//standart deviation of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(nanstdev,nanstdev);

//median of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(nanmedian,nanmedian);

//quantile of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_QUANTILE_NANQUANTILE_ROUTINE(nanquantile,nanquantile);

//average along given axes
//axes may be scalar or container
//weights is container, size of weights must be size along given axes, weights must not sum to zero
template<typename Policy, typename...Ts, typename Axes, typename Container, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto average(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, const Container& weights, bool keep_dims=false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::average(policy,t,axes,weights,keep_dims);
}
template<typename Policy, typename...Ts, typename DimT, typename Container, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto average(Policy policy, const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Container& weights, bool keep_dims=false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::average(policy,t,axes,weights,keep_dims);
}
template<typename Policy, typename...Ts, typename Container, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto average(Policy policy, const basic_tensor<Ts...>& t, const Container& weights, bool keep_dims=false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::average(policy,t,weights,keep_dims);
}
template<typename...Ts, typename Axes, typename Container, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto average(const basic_tensor<Ts...>& t, const Axes& axes, const Container& weights, bool keep_dims=false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::average(t,axes,weights,keep_dims);
}
template<typename...Ts, typename DimT, typename Container, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto average(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Container& weights, bool keep_dims=false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::average(t,axes,weights,keep_dims);
}
template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto average(const basic_tensor<Ts...>& t, const Container& weights, bool keep_dims=false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::average(t,weights,keep_dims);
}

//moving average along given axis, axis is scalar
//weights is container, moving window size is weights size, weights must not sum to zero
//result axis size will be (n - window_size)/step + 1, where n is source axis size
template<typename Policy, typename...Ts, typename DimT, typename Container, typename IdxT>
auto moving_average(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis, const Container& weights, const IdxT& step){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_average(policy,t,axis,weights,step);
}
template<typename Policy, typename...Ts, typename Container, typename IdxT>
auto moving_average(Policy policy, const basic_tensor<Ts...>& t, const Container& weights, const IdxT& step){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_average(policy,t,weights,step);
}
template<typename...Ts, typename DimT, typename Container, typename IdxT>
auto moving_average(const basic_tensor<Ts...>& t, const DimT& axis, const Container& weights, const IdxT& step){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_average(t,axis,weights,step);
}
template<typename...Ts, typename Container, typename IdxT>
auto moving_average(const basic_tensor<Ts...>& t, const Container& weights, const IdxT& step){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_average(t,weights,step);
}

//moving mean along given axis, axis is scalar
//window_size must be greater zero and less_equal than axis size
//result axis size will be (n - window_size)/step + 1, where n is source axis size
template<typename Policy, typename...Ts, typename DimT, typename IdxT>
auto moving_mean(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis, const IdxT& window_size, const IdxT& step){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_mean(policy,t,axis,window_size,step);
}
template<typename Policy, typename...Ts, typename IdxT>
auto moving_mean(Policy policy, const basic_tensor<Ts...>& t, const IdxT& window_size, const IdxT& step){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_mean(policy,t,window_size,step);
}
template<typename...Ts, typename DimT, typename IdxT>
auto moving_mean(const basic_tensor<Ts...>& t, const DimT& axis, const IdxT& window_size, const IdxT& step){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_mean(t,axis,window_size,step);
}
template<typename...Ts, typename IdxT>
auto moving_mean(const basic_tensor<Ts...>& t, const IdxT& window_size, const IdxT& step){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_mean(t,window_size,step);
}

//histogram
//Bins can be integral type or histogram_algorithm enum or container with at least bidirectional iterator
//when bins is of integral type it means equal width bins number, bins must be > 0
//when bins is of histogram_algorithm type equal width bins number is calculated according to algorithm
//when bins is container its elements mean bins edges and must increase monotonically
//Range is pair like type, means numeric range [first,second], first must be less or equal than second
//weights is tensor of same shape as t
template<typename...Ts, typename Bins, typename Range, typename Weights>
auto histogram(const basic_tensor<Ts...>& t, const Bins& bins, const Range& range, bool density, const Weights& weights){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::histogram(t,bins,range,density,weights);
}
template<typename...Ts, typename Bins=int>
auto histogram(const basic_tensor<Ts...>& t, const Bins& bins=10, bool density=false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::histogram(t,bins,detail::no_value{},density,detail::no_value{});
}

#undef GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION
#undef GTENSOR_TENSOR_STATISTIC_QUANTILE_NANQUANTILE_FUNCTION
#undef GTENSOR_TENSOR_STATISTIC_MEDIAN_NANMEDIAN_FUNCTION
#undef GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE
#undef GTENSOR_TENSOR_STATISTIC_QUANTILE_NANQUANTILE_ROUTINE

}   //end of namespace gtensor
#endif