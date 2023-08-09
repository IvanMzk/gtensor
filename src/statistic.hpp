#ifndef STATISTIC_HPP_
#define STATISTIC_HPP_

#include <functional>
#include <algorithm>
#include "math.hpp"
#include "reduce.hpp"
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

#define GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(NAME,F)\
template<typename...Ts, typename Axes>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    return reduce(t,axes,F{},keep_dims);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return reduce_flatten(t,F{},keep_dims);\
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
    GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION(stdev,statistic_reduce_operations::stdev);

    //quantile of elements along given axes
    //axes may be scalar or container
    //q must be of floating point type in range [0,1]
    template<typename...Ts, typename Axes, typename Q>
    static auto quantile(const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){
        using config_type = typename basic_tensor<Ts...>::config_type;
        static_assert(math::numeric_traits<Q>::is_floating_point(),"q must be of floating point type");
        if constexpr (std::is_same_v<Axes,detail::no_value>){
            return reduce_flatten(t,statistic_reduce_operations::quantile{},keep_dims,q,config_type{});
        }else{
            return reduce(t,axes,statistic_reduce_operations::quantile{},keep_dims,q,config_type{});
        }
    }
    //like over flatten
    template<typename...Ts, typename Q>
    static auto quantile(const basic_tensor<Ts...>& t, const Q& q, bool keep_dims = false){
        return quantile(t,detail::no_value{},q,keep_dims);
    }

    //median of elements along given axes
    //axes may be scalar or container
    template<typename...Ts, typename Axes>
    static auto median(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return quantile(t,axes,gtensor::math::make_floating_point_t<value_type>{0.5},keep_dims);
    }
    //like over flatten
    template<typename...Ts>
    static auto median(const basic_tensor<Ts...>& t, bool keep_dims = false){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return quantile(t,gtensor::math::make_floating_point_t<value_type>{0.5},keep_dims);
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
        static_assert(math::numeric_traits<Q>::is_floating_point(),"q must be of floating point type");
        if constexpr (std::is_same_v<Axes,detail::no_value>){
            return reduce_flatten(t,statistic_reduce_operations::nanquantile{},keep_dims,q,config_type{});
        }else{
            return reduce(t,axes,statistic_reduce_operations::nanquantile{},keep_dims,q,config_type{});
        }
    }
    //like over flatten, ignoring nan
    template<typename...Ts, typename Q>
    static auto nanquantile(const basic_tensor<Ts...>& t, const Q& q, bool keep_dims = false){
        return nanquantile(t,detail::no_value{},q,keep_dims);
    }

    //median of elements along given axes, ignoring nan
    //axes may be scalar or container
    template<typename...Ts, typename Axes>
    static auto nanmedian(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return nanquantile(t,axes,gtensor::math::make_floating_point_t<value_type>{0.5},keep_dims);
    }
    //like over flatten, ignoring nan
    template<typename...Ts>
    static auto nanmedian(const basic_tensor<Ts...>& t, bool keep_dims = false){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return nanquantile(t,gtensor::math::make_floating_point_t<value_type>{0.5},keep_dims);
    }

    //average along given axes
    //axes may be scalar or container
    //weights is container, size of weights must be size along given axes, weights must not sum to zero
    template<typename...Ts, typename Axes, typename Container>
    static auto average(const basic_tensor<Ts...>& t, const Axes& axes, const Container& weights, bool keep_dims=false){
        using value_type = typename basic_tensor<Ts...>::value_type;
        if constexpr (std::is_same_v<Axes,detail::no_value>){
            return reduce_flatten(t,statistic_reduce_operations::average<value_type>{},keep_dims,weights);
        }else{
            return reduce(t,axes,statistic_reduce_operations::average<value_type>{},keep_dims,weights);
        }
    }
    template<typename...Ts, typename Container>
    static auto average(const basic_tensor<Ts...>& t, const Container& weights, bool keep_dims=false){
        return average(t,detail::no_value{},weights,keep_dims);
    }

    //moving average along given axis, axis is scalar
    //weights is container, moving window size is weights size, weights must not sum to zero
    //result axis size will be (n - window_size)/step + 1, where n is source axis size
    template<typename...Ts, typename DimT, typename Container, typename IdxT>
    static auto moving_average(const basic_tensor<Ts...>& t, const DimT& axis, const Container& weights, const IdxT& step){
        using index_type = typename basic_tensor<Ts...>::index_type;
        using value_type = typename basic_tensor<Ts...>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        const auto window_size = static_cast<index_type>(weights.size());
        const auto window_step = static_cast<index_type>(step);
        return slide<res_type>(t,axis,statistic_reduce_operations::moving_average<value_type>{},window_size,window_step,weights,window_step);
    }
    //like over flatten
    template<typename...Ts, typename Container, typename IdxT>
    static auto moving_average(const basic_tensor<Ts...>& t, const Container& weights, const IdxT& step){
        using index_type = typename basic_tensor<Ts...>::index_type;
        using value_type = typename basic_tensor<Ts...>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        const auto window_size = static_cast<index_type>(weights.size());
        const auto window_step = static_cast<index_type>(step);
        return slide_flatten<res_type>(t,statistic_reduce_operations::moving_average<value_type>{},window_size,window_step,weights,window_step);
    }

    //moving mean along given axis, axis is scalar
    //window_size must be greater zero and less_equal than axis size
    //result axis size will be (n - window_size)/step + 1, where n is source axis size
    template<typename...Ts, typename DimT, typename IdxT>
    static auto moving_mean(const basic_tensor<Ts...>& t, const DimT& axis, const IdxT& window_size, const IdxT& step){
        using value_type = typename basic_tensor<Ts...>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        return slide<res_type>(t,axis,statistic_reduce_operations::moving_mean{},window_size,step,window_size,step);
    }
    //like over flatten
    template<typename...Ts, typename IdxT>
    static auto moving_mean(const basic_tensor<Ts...>& t, const IdxT& window_size, const IdxT& step){
        using value_type = typename basic_tensor<Ts...>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        return slide_flatten<res_type>(t,statistic_reduce_operations::moving_mean{},window_size,step,window_size,step);
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
        using fp_type = gtensor::math::make_floating_point_t<value_type>;
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
        return std::make_pair(res_bins,res_type({edges_number},edges.begin(),edges.end()));
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
            default:
                throw value_error("invalid histogram_algorithm argument");
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
    return statistic_selector_t<config_type>::F(t,keep_dims);\
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
    return statistic_selector_t<config_type>::quantile(t,q,keep_dims);
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
    return statistic_selector_t<config_type>::nanquantile(t,q,keep_dims);
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
    return statistic_selector_t<config_type>::average(t,weights,keep_dims);
}

//moving average along given axis, axis is scalar
//weights is container, moving window size is weights size, weights must not sum to zero
//result axis size will be (n - window_size)/step + 1, where n is source axis size
template<typename...Ts, typename DimT, typename Container, typename IdxT>
auto moving_average(const basic_tensor<Ts...>& t, const DimT& axis, const Container& weights, const IdxT& step){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_average(t,axis,weights,step);
}
//like over flatten
template<typename...Ts, typename Container, typename IdxT>
auto moving_average(const basic_tensor<Ts...>& t, const Container& weights, const IdxT& step){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::moving_average(t,weights,step);
}

//moving mean along given axis, axis is scalar
//window_size must be greater zero and less_equal than axis size
//result axis size will be (n - window_size)/step + 1, where n is source axis size
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
template<typename...Ts, typename Bins>
auto histogram(const basic_tensor<Ts...>& t, const Bins& bins=10, bool density=false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return statistic_selector_t<config_type>::histogram(t,bins,detail::no_value{},density,detail::no_value{});
}

#undef GTENSOR_TENSOR_STATISTIC_REDUCE_FUNCTION
#undef GTENSOR_TENSOR_STATISTIC_REDUCE_ROUTINE

}   //end of namespace gtensor
#endif