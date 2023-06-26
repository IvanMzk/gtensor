#ifndef STATISTIC_HPP_
#define STATISTIC_HPP_

#include <functional>
#include <algorithm>
#include "math.hpp"
#include "reduce_operations.hpp"
#include "reduce.hpp"

namespace gtensor{

namespace detail{

template<typename P, typename T, typename U=void> constexpr bool is_pair_of_type_v = false;
template<typename P, typename T> constexpr bool is_pair_of_type_v<P,T,std::void_t<decltype(std::declval<P>().first),decltype(std::declval<P>().second)>> =
    std::is_convertible_v<decltype(std::declval<P>().first),T> && std::is_convertible_v<decltype(std::declval<P>().second),T>;

}

enum class histogram_algorithm : std::size_t {automatic,fd,scott,rice,sturges,sqrt};

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

    //histogram
    //Bins can be integral type or container or histogram_algorithm enum
    //Range is pair like type
    template<typename...Ts, typename Bins, typename Range>
    static auto histogram(const basic_tensor<Ts...>& t, const Bins& bins, const Range& range, bool density = false){
        using config_type = typename basic_tensor<Ts...>::config_type;
        using value_type = typename basic_tensor<Ts...>::value_type;
        using fp_type = gtensor::math::make_floating_point_t<value_type>;
        using index_type = typename basic_tensor<Ts...>::index_type;
        using order = typename basic_tensor<Ts...>::order;
        using container_type = typename config_type::template container<fp_type>;
        using container_difference_type = typename container_type::difference_type;

        //check_range(range);   //is range no_value or pair and first<=second
        //check_bins(bins);     //is bins integral or container or enum

        auto a = t.template traverse_order_adapter<order>();
        if constexpr (detail::is_container_of_type_v<Bins, value_type>){    //not uniform bin width, need copy and sort data
            container_type elements_{};
            if constexpr (detail::is_static_castable_v<index_type,container_difference_type>){
                const auto n = static_cast<container_difference_type>(t.size());
                elements_.reserve(n);
            }
            if constexpr (std::is_same_v<Range,detail::no_value>){
                std::copy(a.begin(),a.end(),std::back_inserter(elements_));
            }else{
                const auto rmin = static_cast<value_type>(range.first);
                const auto rmax = static_cast<value_type>(range.second);
                auto is_in_range = [rmin,rmax](const auto& e){return e>=rmin && e<=rmax;};
                std::copy_if(a.begin(),a.end(),std::back_inserter(elements_),is_in_range);
            }
            return make_non_uniform_histogram(elements_.begin(),elements_.end(),bins,density);
        }else{  //uniform, bin is integral or enum
            fp_type min{0};
            fp_type max{1};
            if constexpr (std::is_same_v<Range,detail::no_value>){  //range is t.min(), t.max(),
                if (t.empty()){
                    return make_uniform_histogram<config_type>(a.begin(), a.end(), bins, min, max, min, max, density);
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
                return make_uniform_histogram<config_type>(a.begin(), a.end(), bins, min, max, rmin, rmax, density);
            }else{
                auto rmin = static_cast<fp_type>(range.first);
                auto rmax = static_cast<fp_type>(range.second);
                if (rmin == rmax){
                    rmin-=0.5;
                    rmax+=0.5;
                }
                if (t.empty()){
                    return make_uniform_histogram<config_type>(a.begin(), a.end(), bins, min, max, rmin, rmax, density);
                }
                container_type elements_{};
                if constexpr (detail::is_static_castable_v<index_type,container_difference_type>){
                    const auto n = static_cast<container_difference_type>(t.size());
                    elements_.reserve(n);
                }
                auto it = a.begin();
                min = *it;
                max = min;
                for (auto last=a.end(); it!=last; ++it){
                    const auto& e = static_cast<const fp_type&>(*it);
                    if (e>=rmin && e<=rmax){
                        elements_.push_back(e);
                        if (e<min){
                            min=e;
                        }else if (e>max){
                            max=e;
                        }
                    }
                }
                return make_uniform_histogram<config_type>(elements_.begin(), elements_.end(), bins, min, max, rmin, rmax, density);
            }
        }

    }

    template<typename...Ts, typename Bins>
    static auto histogram(const basic_tensor<Ts...>& t, const Bins& bins, bool density = false){
        return histogram(t,bins,detail::no_value{},density);
    }
private:

    //first,last - elements range that are within histogram range argument, all elements range if histogram range argument is no_value
    template<typename Config, typename It, typename Bins, typename T>
    static auto make_uniform_histogram(It first, It last, const Bins& bins, const T& min, const T& max, const T& rmin, const T& rmax, bool density){
        using index_type = typename Config::index_type;
        using order = typename Config::order;
        using value_type = typename std::iterator_traits<It>::value_type;
        using integral_type = gtensor::math::make_integral_t<value_type>;
        using fp_type = gtensor::math::make_floating_point_t<value_type>;
        using res_value_type = fp_type;
        using res_type = tensor<res_value_type,order,Config>;

        const auto n = static_cast<fp_type>(static_cast<integral_type>(last - first));
        const auto bins_ = make_bins<Config>(first,last,bins,min,max,rmin,rmax);
        const auto bin_width = bins_.first;
        const auto bins_number = static_cast<index_type>(static_cast<integral_type>(bins_.second));

        // std::cout<<std::endl<<"rmin,rmax "<<rmin<<" "<<rmax;
        // std::cout<<std::endl<<"bins_number "<<bins_number;
        // std::cout<<std::endl<<"bin_width "<<bin_width;

        res_type res_bins({bins_number},fp_type{0});
        auto a = res_bins.template traverse_order_adapter<order>();
        auto res_bins_indexer = a.create_indexer();
        for (;first!=last; ++first){
            auto i = static_cast<index_type>(static_cast<integral_type>((*first - rmin)/bin_width));
            if (i == bins_number){  //add elements that equal to range upper boundary to last bin (last bin includes upper boundary)
                --i;
            }
            res_bins_indexer[i]+=res_value_type{1};
        }
        if (density){
            const auto normalizer = 1/(n*bin_width);
            std::for_each(a.begin(),a.end(),[normalizer](auto& e){
                e*=normalizer;
            });
        }
        res_type res_intervals({bins_number+index_type{1}},fp_type{rmin});
        auto res_intervals_a = res_intervals.template traverse_order_adapter<order>();
        auto res_intervals_it = res_intervals_a.begin();
        auto res_intervals_last = res_intervals_a.end();
        auto edge = *res_intervals_it;
        for(++res_intervals_it,--res_intervals_last; res_intervals_it!=res_intervals_last; ++res_intervals_it){
            edge+=bin_width;
            *res_intervals_it = edge;
        }
        *res_intervals_it = rmax;
        return std::make_pair(res_bins,res_intervals);
    }

    template<typename Config, typename It, typename Bins, typename T>
    static auto make_bins(It first, It last, const Bins& bins, const T& min, const T& max, const T& rmin, const T& rmax){
        using value_type = typename std::iterator_traits<It>::value_type;
        using fp_type = gtensor::math::make_floating_point_t<value_type>;

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
        using fp_type = gtensor::math::make_floating_point_t<value_type>;

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
        };
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

//histogram
//Bins can be integral type or container or histogram_algorithm enum
//Range is pair like type
template<typename...Ts, typename Bins, typename Range>
auto histogram(const basic_tensor<Ts...>& t, const Bins& bins, const Range& range, bool density){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::histogram(t,bins,range,density);
}

#undef GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION
#undef GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE

}   //end of namespace gtensor
#endif