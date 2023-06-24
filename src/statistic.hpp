#ifndef STATISTIC_HPP_
#define STATISTIC_HPP_

#include <functional>
#include <algorithm>
#include "math.hpp"
#include "reduce_operations.hpp"
#include "reduce.hpp"

namespace gtensor{

//tensor statistic functions implementation

#define GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(NAME,F)\
template<typename...Ts, typename Axes>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    return reduce(t,axes,F{},keep_dims);\
}

struct statistic
{
    //statistic functions along given axis or axes
    //axes may be scalar or container if multiple axes permitted
    //empty container means apply function along all axes

    //peak-to-peak of elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(ptp,statistic_reduce_operations::ptp);

    //mean of elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(mean,statistic_reduce_operations::mean);

    //variance of elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(var,statistic_reduce_operations::var);

    //standart deviation of elements along given axes
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(std,statistic_reduce_operations::stdev);

    //quantile of elements along given axes
    //axes may be scalar or container
    //q must be in range [0,1]
    template<typename...Ts, typename Axes, typename Q>
    static auto quantile(const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){
        using config_type = typename basic_tensor<Ts...>::config_type;
        return reduce(t,axes,statistic_reduce_operations::quantile{},keep_dims,q,config_type{});
    }

    //median of elements along given axes
    //axes may be scalar or container
    template<typename...Ts, typename Axes>
    static auto median(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return quantile(t,axes,gtensor::math::make_floating_point_t<value_type>{0.5},keep_dims);
    }

    //nan versions
    //mean of elements along given axes, ignoring nan
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(nanmean,statistic_reduce_operations::nanmean);

    //variance of elements along given axes, ignoring nan
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(nanvar,statistic_reduce_operations::nanvar);

    //standart deviation of elements along given axes, ignoring nan
    //axes may be scalar or container
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(nanstd,statistic_reduce_operations::nanstdev);

    //quantile of elements along given axes, ignoring nan
    //axes may be scalar or container
    template<typename...Ts, typename Axes, typename Q>
    static auto nanquantile(const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){
        using config_type = typename basic_tensor<Ts...>::config_type;
        return reduce(t,axes,statistic_reduce_operations::nanquantile{},keep_dims,q,config_type{});
    }

    //median of elements along given axes, ignoring nan
    //axes may be scalar or container
    template<typename...Ts, typename Axes>
    static auto nanmedian(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return nanquantile(t,axes,gtensor::math::make_floating_point_t<value_type>{0.5},keep_dims);
    }

    //average along given axes
    //axes may be scalar or container
    //weights is container, size of weights must be size along given axes, weights must not sum to zero
    template<typename...Ts, typename Axes, typename Container>
    static auto average(const basic_tensor<Ts...>& t, const Axes& axes, const Container& weights, bool keep_dims=false){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return reduce(t,axes,statistic_reduce_operations::average<value_type>{},keep_dims,weights);
    }

    //moving average along given axis, axis is scalar
    //weights is container, moving window size is weights size, weights must not sum to zero
    //result axis size will be (n - window_size)/step + 1, where n is source axis size
    template<typename...Ts, typename DimT, typename Container, typename IdxT>
    static auto moving_average(const basic_tensor<Ts...>& t, const DimT& axis, const Container& weights, const IdxT& step = 1){
        using value_type = typename basic_tensor<Ts...>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        const IdxT window_size = weights.size();
        return slide<res_type>(t,axis,statistic_reduce_operations::moving_average<value_type>{},window_size,step,weights,step);
    }

    //moving mean along given axis, axis is scalar
    //window_size must be greater zero and less_equal than axis size
    //result axis size will be (n - window_size)/step + 1, where n is source axis size
    template<typename...Ts, typename DimT, typename IdxT>
    static auto moving_mean(const basic_tensor<Ts...>& t, const DimT& axis, const IdxT& window_size, const IdxT& step = 1){
        using value_type = typename basic_tensor<Ts...>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        return slide<res_type>(t,axis,statistic_reduce_operations::moving_mean{},window_size,step,window_size,step);
    }

};  //end of struct statistic

//tensor statistic frontend
//frontend uses compile-time dispatch to select implementation, see module_selector.hpp

#define GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(NAME,F)\
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
    return statistic_selector_t<config_type>::F(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},keep_dims);\
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
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(std,std);

//median of elements along given axes
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(median,median);

//quantile of elements along given axes
//axes may be scalar or container
//q must be in range [0,1]
template<typename...Ts, typename Axes, typename Q>
auto quantile(const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::quantile(t,axes,q,keep_dims);
}
template<typename...Ts, typename DimT, typename Q>
auto quantile(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::quantile(t,axes,q,keep_dims);
}
template<typename...Ts, typename Q>
auto quantile(const basic_tensor<Ts...>& t, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::quantile(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},q,keep_dims);
}

//nan versions
//mean of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(nanmean,nanmean);

//variance of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(nanvar,nanvar);

//standart deviation of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(nanstd,nanstd);

//median of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE(nanmedian,nanmedian);

//quantile of elements along given axes, ignoring nan
//axes may be scalar or container
template<typename...Ts, typename Axes, typename Q>
auto nanquantile(const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::nanquantile(t,axes,q,keep_dims);
}
template<typename...Ts, typename DimT, typename Q>
auto nanquantile(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::nanquantile(t,axes,q,keep_dims);
}
template<typename...Ts, typename Q>
auto nanquantile(const basic_tensor<Ts...>& t, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::nanquantile(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},q,keep_dims);
}

//average along given axes
//axes may be scalar or container
//weights is container, size of weights must be size along given axes, weights must not sum to zero
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
//average over all axes
template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto average(const basic_tensor<Ts...>& t, const Container& weights, bool keep_dims=false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::average(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},weights,keep_dims);
}

//moving average along given axis, axis is scalar
//weights is container, moving window size is weights size, weights must not sum to zero
//result axis size will be (n - window_size)/step + 1, where n is source axis size
template<typename...Ts, typename DimT, typename Container, typename IdxT>
auto moving_average(const basic_tensor<Ts...>& t, const DimT& axis, const Container& weights, const IdxT& step = 1){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_average(t,axis,weights,step);
}

//moving mean along given axis, axis is scalar
//window_size must be greater zero and less_equal than axis size
//result axis size will be (n - window_size)/step + 1, where n is source axis size
template<typename...Ts, typename DimT, typename IdxT>
auto moving_mean(const basic_tensor<Ts...>& t, const DimT& axis, const IdxT& window_size, const IdxT& step = 1){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_mean(t,axis,window_size,step);
}


#undef GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION
#undef GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE

}   //end of namespace gtensor
#endif