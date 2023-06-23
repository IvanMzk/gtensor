#ifndef STATISTIC_HPP_
#define STATISTIC_HPP_

#include <functional>
#include <algorithm>
#include "math.hpp"
#include "reduce.hpp"

namespace gtensor{

namespace statistic_reduce_operations{

//result floating point type for routines where floating point result is mandatory (mean,nanmean,var,nanvar,...)
//T is value_type of source
template<typename T> using result_floating_point_t = std::conditional_t<
    gtensor::math::numeric_traits<T>::is_floating_point(),
    T,
    typename gtensor::math::numeric_traits<T>::floating_point_type
>;

//peak-to-peak
struct ptp
{
    template<typename It>
    auto operator()(It first, It last){
        if (first==last){
            throw reduce_exception("cant reduce zero size dimension");
        }
        auto min = *first;
        auto max = min;
        for (++first; first!=last; ++first){
            const auto& e = *first;
            if(e<min){
                min = e;
            }
            if(e>max){
                max = e;
            }
        }
        return max-min;
    }
};

template<typename T>
auto reduce_empty(){
    if constexpr (gtensor::math::numeric_traits<T>::has_nan()){
        return gtensor::math::numeric_traits<T>::nan();
    }else{
        throw reduce_exception("cant reduce zero size dimension without initial value");
        return T{};    //need same return type
    }
}

struct mean
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using res_type = result_floating_point_t<value_type>;
        if (first == last){
            return reduce_empty<res_type>();
        }
        const auto n = static_cast<res_type>(last-first);
        auto sum_ = static_cast<res_type>(*first);
        for(++first; first!=last; ++first){
            sum_ += static_cast<const res_type&>(*first);
        }
        return sum_ / n;
    }
};

struct nanmean
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using res_type = result_floating_point_t<value_type>;
        if (first == last){
            return reduce_empty<res_type>();
        }
        if constexpr (gtensor::math::numeric_traits<value_type>::has_nan() && gtensor::math::numeric_traits<res_type>::has_nan()){
            //find first not nan
            for(;first!=last; ++first){
                if (!gtensor::math::isnan(static_cast<const res_type&>(*first))){
                    break;
                }
            }
            if (first == last){ //all nans, return last nan
                return --first,*first;
            }
            auto sum = static_cast<res_type>(*first);
            difference_type counter{1};
            for(++first; first!=last; ++first){
                const auto& e = static_cast<const res_type&>(*first);
                if (!gtensor::math::isnan(e)){
                    sum+=e;
                    ++counter;
                }
            }
            const auto n = static_cast<res_type>(counter);
            return sum / n;
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
        using res_type = result_floating_point_t<value_type>;
        if (first == last){
            return reduce_empty<res_type>();
        }
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
        using res_type = result_floating_point_t<value_type>;
        if (first == last){
            return reduce_empty<res_type>();
        }
        if constexpr (gtensor::math::numeric_traits<value_type>::has_nan() && gtensor::math::numeric_traits<res_type>::has_nan()){
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

struct stdev
{
    template<typename It>
    auto operator()(It first, It last){
        return gtensor::math::sqrt(var{}(first,last));
    }
};

struct nanstdev
{
    template<typename It>
    auto operator()(It first, It last){
        return gtensor::math::sqrt(nanvar{}(first,last));
    }
};

template<typename Q>
void check_quantile(const Q& q){
    if (q>Q{1} || q<Q{0}){
        throw reduce_exception("quntile must be in range [0,1]");
    }
}

//Predicate is use in copy_if that copy elements to temporary storage
//quantile: Predicate must throw reduce_exception on nan element, true otherwise
//nanquantile: Predicate must return false on nan element, true otherwise
template<typename Predicate>
struct quantile_nanquantile
{
    //quantile must be in range from 0 to 1 incluseve
    template<typename It, typename Q, typename Config>
    auto operator()(It first, It last, const Q& quantile, Config){
        using value_type = typename std::iterator_traits<It>::value_type;
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using container_type = typename Config::template container<value_type>;
        using container_difference_type = typename container_type::difference_type;
        using res_type = result_floating_point_t<value_type>;
        if (first == last){
            return reduce_empty<res_type>();
        }
        check_quantile(quantile);
        container_type elements_{};
        if constexpr (detail::is_static_castable_v<difference_type,container_difference_type>){
            elements_.reserve(static_cast<container_difference_type>(last - first));
        }
        if constexpr (gtensor::math::numeric_traits<value_type>::has_nan()){
            try{
                std::copy_if(first,last,std::back_inserter(elements_),Predicate{});
            }catch(reduce_exception){
                return gtensor::math::numeric_traits<res_type>::nan();
            }
        }else{
            std::copy(first,last,std::back_inserter(elements_));
        }
        const container_difference_type n_elements = elements_.size();
        if (n_elements == container_difference_type{0}){    //all nan
            return gtensor::math::numeric_traits<res_type>::nan();
        }

        const auto& q = static_cast<const res_type&>(quantile);
        const auto q_index = q*static_cast<res_type>(n_elements-1);
        const auto q_index_decomposed = gtensor::math::modf(q_index);
        const auto g = q_index_decomposed.second; //q_index fractional part
        auto i = static_cast<container_difference_type>(q_index_decomposed.first);  //q_index integral part

        std::nth_element(elements_.begin(),elements_.begin()+i,elements_.end());
        auto res = static_cast<res_type>(elements_[i]);
        if (g != res_type{0}){  //interpolation needed
            const auto next_after_i = *std::min_element(elements_.begin()+ ++i,elements_.end());
            res+=(static_cast<res_type>(next_after_i)-res)*g;
        }
        return res;
    }
};

struct quantile_predicate
{
    template<typename T>
    bool operator()(const T& t){
        if (gtensor::math::isnan(t)){
            throw reduce_exception("");
        }
        return true;
    }
};

struct nanquantile_predicate
{
    template<typename T>
    bool operator()(const T& t){
        if (gtensor::math::isnan(t)){
            return false;
        }
        return true;
    }
};

using quantile = quantile_nanquantile<quantile_predicate>;
using nanquantile = quantile_nanquantile<nanquantile_predicate>;

struct median
{
    template<typename It, typename Config>
    auto operator()(It first, It last, Config config){
        return quantile{}(first, last, 0.5, config);
    }
};

struct nanmedian
{
    template<typename It, typename Config>
    auto operator()(It first, It last, Config config){
        return nanquantile{}(first, last, 0.5, config);
    }
};

template<typename T, typename U>
void check_weights_size(const T& n, const U& n_weights){
    auto test = [](const auto& n_, const auto& n_weights_){
        if (n_!=n_weights_){
            throw reduce_exception("length of weights not compatible with specified axes");
        }
    };
    if constexpr (detail::is_static_castable_v<U,T>){
        test(n,static_cast<T>(n_weights));
    }else if constexpr (detail::is_static_castable_v<T,U>){
        test(static_cast<U>(n),n_weights);
    }else{
        static_assert(detail::always_false<T,U>, "types not convertible to each other");
    }
}

template<typename T>
void check_weights_sum(const T& sum){
    if (sum == T{0}){
        throw reduce_exception("weights sum to zero");
    }
}

//average
//T is source value_type
template<typename T>
class average
{
    using res_type = result_floating_point_t<T>;
    res_type weights_sum{0};
    res_type normalizer{1};
public:
    //Weights is container, weights size must be equal to last-first
    template<typename It, typename Container>
    auto operator()(It first, It last, const Container& weights){
        using value_type = typename std::iterator_traits<It>::value_type;
        static_assert(std::is_same_v<T,value_type>,"invalid average template T type argument");
        static_assert(detail::is_container_of_type_v<Container,res_type>,"invalid weights argument");

        const auto n = last - first;
        const auto n_weights = weights.size();
        check_weights_size(n,n_weights);

        res_type res{0};
        auto weights_it = weights.begin();
        if (weights_sum == res_type{0}){    //additional compute weights sum
            for (;first!=last; ++first,++weights_it){
                const auto& w = static_cast<const res_type&>(*weights_it);
                weights_sum+=w;
                res+=static_cast<const res_type&>(*first)*w;
            }
            check_weights_sum(weights_sum);
            normalizer/=weights_sum;
        }else{
            for (;first!=last; ++first,++weights_it){
                res+=static_cast<const res_type&>(*first)*static_cast<const res_type&>(*weights_it);
            }
        }
        return res*normalizer;
    }
};

//moving average
//T is source value_type
template<typename T>
class moving_average
{
    using res_type = result_floating_point_t<T>;
    res_type weights_sum{0};
    res_type normalizer{1};
public:
    template<typename It, typename DstIt, typename Container, typename IdxT>
    auto operator()(It first, It, DstIt dfirst, DstIt dlast, const Container& weights, const IdxT& step){
        using value_type = typename std::iterator_traits<It>::value_type;
        using dst_value_type = typename std::iterator_traits<DstIt>::value_type;
        using difference_type = typename std::iterator_traits<It>::difference_type;
        static_assert(std::is_same_v<res_type,dst_value_type>,"invalid average template T type argument");
        static_assert(detail::is_container_of_type_v<Container,res_type>,"invalid weights argument");

        const auto n_weights = weights.size();
        const auto window_size = static_cast<difference_type>(n_weights);
        const auto window_step = static_cast<difference_type>(step);
        average<value_type> average_maker{};
        for (;dfirst!=dlast; ++dfirst,first+=window_step){
            *dfirst = average_maker(first,first+window_size,weights);
        }
    }
};

//moving mean
struct moving_mean
{
    template<typename It, typename DstIt, typename IdxT>
    auto operator()(It first, It, DstIt dfirst, DstIt dlast, const IdxT& win_size, const IdxT& step){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using res_type = typename std::iterator_traits<DstIt>::value_type;

        const auto window_size = static_cast<difference_type>(win_size);
        const auto window_step = static_cast<difference_type>(step);
        const auto normalizer = res_type{1}/static_cast<res_type>(win_size);
        if (window_size > difference_type{2}*window_step){  //not recalculate overlapping region
            auto window_it = first;
            auto window_last = first+window_size;
            res_type res{0};
            for (;window_it!=window_last; ++window_it){
                res+=static_cast<const res_type&>(*window_it);
            }
            for(;dfirst!=dlast; ++dfirst){
                *dfirst = res*normalizer;
                for (difference_type i{0}; i!=window_step; ++i,++first){
                    res-=static_cast<const res_type&>(*first);
                }
                for (difference_type i{0}; i!=window_step; ++i,++window_it){
                    res+=static_cast<const res_type&>(*window_it);
                }
            }
        }else{  //calculate full window every iteration
            mean mean_maker{};
            for(;dfirst!=dlast; ++dfirst,first+=window_step){
                *dfirst = mean_maker(first,first+window_size);
            }
        }
    }
};

}   //end of namespace statistic_reduce_operations

//statistic functions along given axis or axes
//axes may be scalar or container if multiple axes permitted
//empty container means apply function along all axes

#define GTENSOR_STATISTIC_REDUCE_ROUTINE(name,functor,...)\
template<typename...Ts, typename Axes>\
auto name(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    return reduce(t,axes,functor{},keep_dims __VA_OPT__(,) __VA_ARGS__);\
}\
template<typename...Ts, typename DimT>\
auto name(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false){\
    return reduce(t,axes,functor{},keep_dims __VA_OPT__(,) __VA_ARGS__);\
}\
template<typename...Ts>\
auto name(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},functor{},keep_dims __VA_OPT__(,) __VA_ARGS__);\
}

//peak-to-peak of elements along given axes
//axes may be scalar or container
GTENSOR_STATISTIC_REDUCE_ROUTINE(ptp,statistic_reduce_operations::ptp);

//mean of elements along given axes
//axes may be scalar or container
GTENSOR_STATISTIC_REDUCE_ROUTINE(mean,statistic_reduce_operations::mean);

//variance of elements along given axes
//axes may be scalar or container
GTENSOR_STATISTIC_REDUCE_ROUTINE(var,statistic_reduce_operations::var);

//standart deviation of elements along given axes
//axes may be scalar or container
GTENSOR_STATISTIC_REDUCE_ROUTINE(std,statistic_reduce_operations::stdev);

//median of elements along given axes
//axes may be scalar or container
GTENSOR_STATISTIC_REDUCE_ROUTINE(median,statistic_reduce_operations::median,typename basic_tensor<Ts...>::config_type{});

//quantile of elements along given axes
//axes may be scalar or container
//q must be in range [0,1]
template<typename...Ts, typename Axes, typename Q>
auto quantile(const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reduce(t,axes,statistic_reduce_operations::quantile{},keep_dims,q,config_type{});
}
template<typename...Ts, typename DimT, typename Q>
auto quantile(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reduce(t,axes,statistic_reduce_operations::quantile{},keep_dims,q,config_type{});
}
template<typename...Ts, typename Q>
auto quantile(const basic_tensor<Ts...>& t, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},statistic_reduce_operations::quantile{},keep_dims,q,config_type{});
}

//nan versions
//mean of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_STATISTIC_REDUCE_ROUTINE(nanmean,statistic_reduce_operations::nanmean);

//variance of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_STATISTIC_REDUCE_ROUTINE(nanvar,statistic_reduce_operations::nanvar);

//standart deviation of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_STATISTIC_REDUCE_ROUTINE(nanstd,statistic_reduce_operations::nanstdev);

//median of elements along given axes, ignoring nan
//axes may be scalar or container
GTENSOR_STATISTIC_REDUCE_ROUTINE(nanmedian,statistic_reduce_operations::nanmedian,typename basic_tensor<Ts...>::config_type{});

//quantile of elements along given axes, ignoring nan
//axes may be scalar or container
template<typename...Ts, typename Axes, typename Q>
auto nanquantile(const basic_tensor<Ts...>& t, const Axes& axes, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reduce(t,axes,statistic_reduce_operations::nanquantile{},keep_dims,q,config_type{});
}
template<typename...Ts, typename DimT, typename Q>
auto nanquantile(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reduce(t,axes,statistic_reduce_operations::nanquantile{},keep_dims,q,config_type{});
}
template<typename...Ts, typename Q>
auto nanquantile(const basic_tensor<Ts...>& t, const Q& q, bool keep_dims = false){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reduce(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},statistic_reduce_operations::nanquantile{},keep_dims,q,config_type{});
}

//average along given axes
//axes may be scalar or container
//weights is container, size of weights must be size along given axes, weights must not sum to zero
template<typename...Ts, typename Axes, typename Container, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto average(const basic_tensor<Ts...>& t, const Axes& axes, const Container& weights, bool keep_dims=false){
    using value_type = typename basic_tensor<Ts...>::value_type;
    return reduce(t,axes,statistic_reduce_operations::average<value_type>{},keep_dims,weights);
}
template<typename...Ts, typename DimT, typename Container, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto average(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, const Container& weights, bool keep_dims=false){
    using value_type = typename basic_tensor<Ts...>::value_type;
    return reduce(t,axes,statistic_reduce_operations::average<value_type>{},keep_dims,weights);
}
//average over all axes
template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto average(const basic_tensor<Ts...>& t, const Container& weights, bool keep_dims=false){
    using value_type = typename basic_tensor<Ts...>::value_type;
    using dim_type = typename basic_tensor<Ts...>::dim_type;
    return reduce(t,std::initializer_list<dim_type>{},statistic_reduce_operations::average<value_type>{},keep_dims,weights);
}

//moving average along given axis, axis is scalar
//weights is container, moving window size is weights size, weights must not sum to zero
//result axis size will be (n - window_size)/step + 1, where n is source axis size
template<typename...Ts, typename DimT, typename Container, typename IdxT, std::enable_if_t<detail::is_container_v<Container>,int> =0>
auto moving_average(const basic_tensor<Ts...>& t, const DimT& axis, const Container& weights, const IdxT& step = 1){
    using value_type = typename basic_tensor<Ts...>::value_type;
    using res_type = statistic_reduce_operations::result_floating_point_t<value_type>;
    const IdxT window_size = weights.size();
    return slide<res_type>(t,axis,statistic_reduce_operations::moving_average<value_type>{},window_size,step,weights,step);
}

//moving mean along given axis, axis is scalar
//window_size must be greater zero and less_equal than axis size
//result axis size will be (n - window_size)/step + 1, where n is source axis size
template<typename...Ts, typename DimT, typename IdxT>
auto moving_mean(const basic_tensor<Ts...>& t, const DimT& axis, const IdxT& window_size, const IdxT& step = 1){
    using value_type = typename basic_tensor<Ts...>::value_type;
    using res_type = statistic_reduce_operations::result_floating_point_t<value_type>;
    return slide<res_type>(t,axis,statistic_reduce_operations::moving_mean{},window_size,step,window_size,step);
}


#undef GTENSOR_STATISTIC_REDUCE_ROUTINE

}   //end of namespace gtensor
#endif