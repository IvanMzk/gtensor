#ifndef REDUCE_OPERATIONS_HPP_
#define REDUCE_OPERATIONS_HPP_
#include <functional>
#include <algorithm>
#include <numeric>
#include "common.hpp"
#include "exception.hpp"
#include "math.hpp"

namespace gtensor{

//functors to use in tensor_math reduce functions
namespace math_reduce_operations{

template<typename Operation>
struct logical_binary_operation{
    const Operation op{};
    template<typename U, typename V>
    bool operator()(const U& u, const V& v)const{
        return op(static_cast<const bool&>(u),static_cast<const bool&>(v));
    }
};

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

//test if initial argument contain valid initial value
template<typename Initial, typename T> constexpr bool is_initial_v = !std::is_same_v<Initial,gtensor::detail::no_value> && detail::is_static_castable_v<Initial,T>;
//test if functor F has initial value
template<typename F, typename T, typename=void> constexpr bool has_initial_v = false;
template<typename F, typename T> constexpr bool has_initial_v<F,T,std::void_t<decltype(F::template value<T>())>> = true;

template<typename Functor, typename It, typename Initial>
typename std::iterator_traits<It>::value_type reduce_empty(const Initial& initial){
    using value_type = typename std::iterator_traits<It>::value_type;
    if constexpr (is_initial_v<Initial,value_type>){
        return static_cast<value_type>(initial);
    }else if constexpr(has_initial_v<Functor,value_type>){
        return Functor::template value<value_type>();
    }else{  //no initial, throw
        throw value_error("cant reduce zero size dimension without initial value");
    }
}

template<typename Functor, typename It, typename Initial>
auto make_initial(It& first, const Initial& initial){
    using value_type = typename std::iterator_traits<It>::value_type;
    if constexpr (is_initial_v<Initial,value_type>){
        return static_cast<value_type>(initial);
    }else if constexpr(has_initial_v<Functor,value_type>){
        return Functor::template value<value_type>();
    }else{  //use first element as initial and inc first
        auto res = *first;
        return ++first,res;
    }
}

template<typename Comparator>
struct extremum_nanextremum
{
    template<typename It, typename Initial = gtensor::detail::no_value>
    auto operator()(It first, It last, const Initial& initial = Initial{}){
        if (first == last){
            return reduce_empty<Comparator,It>(initial);
        }
        auto init = make_initial<Comparator>(first,initial);
        return std::accumulate(first,last,init,Comparator{});
    }
};

template<typename Comparator>
struct nan_propagate_comparator
{
    const Comparator comparator{};
    template<typename R, typename E>
    auto operator()(const R& r, const E& e){
        return gtensor::math::isnan(e) ? e : comparator(e,r) ? e : r;
    }
};

template<typename Comparator>
struct nan_ignore_comparator
{
    const Comparator comparator{};
    template<typename R, typename E>
    auto operator()(const R& r, const E& e){
        return gtensor::math::isnan(r) ? e : comparator(e,r) ? e : r;
    }
};

using amin = extremum_nanextremum<nan_propagate_comparator<std::less<void>>>;
using amax = extremum_nanextremum<nan_propagate_comparator<std::greater<void>>>;
using nanmin = extremum_nanextremum<nan_ignore_comparator<std::less<void>>>;
using nanmax = extremum_nanextremum<nan_ignore_comparator<std::greater<void>>>;

template<typename Operation>
struct accumulate_nanaccumulate
{
    template<typename It, typename Initial = gtensor::detail::no_value>
    auto operator()(It first, It last, const Initial& initial = Initial{}){
        if (first == last){
            return reduce_empty<Operation,It>(initial);
        }
        auto init = make_initial<Operation>(first,initial);
        return std::accumulate(first,last,init,Operation{});
    }
};

template<typename Operation>
struct nan_propagate_operation : Operation
{
    template<typename R, typename E>
    auto operator()(const R& r, const E& e){
        return Operation::operator()(r,e);
    }
};

template<typename Operation>
struct nan_ignoring_operation : Operation
{
    bool is_r{false};
    template<typename R, typename E>
    auto operator()(const R& r, const E& e)->decltype(Operation::operator()(r,e)){
        if (is_r){
            return gtensor::math::isnan(e) ? r : Operation::operator()(r,e);
        }else{
            if (gtensor::math::isnan(r)){
                return e;
            }else{
                is_r=true;
                if(gtensor::math::isnan(e)){
                    return r;
                }else{
                    return Operation::operator()(r,e);
                }
            }
        }
    }
};

template<typename T> struct plus : public std::plus<T>{template<typename U> inline static constexpr U value(){return U(0);}};
template<typename T> struct multiplies : public std::multiplies<T>{template<typename U> inline static constexpr U value(){return U(1);}};

using sum = accumulate_nanaccumulate<nan_propagate_operation<plus<void>>>;
using prod = accumulate_nanaccumulate<nan_propagate_operation<multiplies<void>>>;
using nansum = accumulate_nanaccumulate<nan_ignoring_operation<plus<void>>>;
using nanprod = accumulate_nanaccumulate<nan_ignoring_operation<multiplies<void>>>;

template<typename Operation>
struct cumulate_nancumulate
{
    template<typename It, typename DstIt>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast){
        using value_type = typename std::iterator_traits<It>::value_type;
        Operation operation{};
        auto res = operation(Operation::template value<value_type>(), *first);
        *dfirst = res;
        while(++first!=last){
            res = operation(res, *first);
            *++dfirst = res;
        }
    }
};

using cumsum = cumulate_nancumulate<nan_propagate_operation<plus<void>>>;
using cumprod = cumulate_nancumulate<nan_propagate_operation<multiplies<void>>>;
using nancumsum = cumulate_nancumulate<nan_ignoring_operation<plus<void>>>;
using nancumprod = cumulate_nancumulate<nan_ignoring_operation<multiplies<void>>>;

//first finite difference
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

//second finite difference
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

template<typename T, typename U>
void check_spacing_size(const T& n, const U& n_spacing){
    auto test = [](const auto& n_, const auto& n_spacing_){
        if (n_!=n_spacing_){
            throw value_error("length of spacing not compatible with specified axes");
        }
    };
    if constexpr (detail::is_static_castable_v<U,T>){
        test(n,static_cast<T>(n_spacing));
    }else if constexpr (detail::is_static_castable_v<T,U>){
        test(static_cast<U>(n),n_spacing);
    }else{
        static_assert(detail::always_false<T,U>, "types not convertible to each other");
    }
}

//e0,e1,e2 are stencil elements
//hs is bakward delta (from e1 to e0), hd is forward delta (from e1 to e2)
template<typename T>
auto make_gradient(const T& e0, const T& e1, const T& e2, const T& hs, const T& hd){
    const auto hs_sqr = hs*hs;
    const auto hd_sqr = hd*hd;
    return (e1*(hd_sqr-hs_sqr) + e2*hs_sqr -e0*hd_sqr)/(hd*hs*(hd+hs));
}

//gradient
//2-nd order accuracy gradient approximation
struct gradient
{
    //Spacing may be scalar (uniform) or container (not uniform)
    //if container, its size must equal to size along axis that is last-first
    template<typename It, typename DstIt, typename Spacing>
    auto operator()(It first, It last, DstIt dfirst, DstIt, const Spacing& spacing){
        using value_type = typename std::iterator_traits<It>::value_type;
        using dst_value_type = typename std::iterator_traits<DstIt>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        static_assert(std::is_same_v<dst_value_type,res_type>,"invalid DstIt value_type");
        const auto n = last-first;
        if (n<2){
            throw value_error("gradient requires at least 2 points");
        }
        if constexpr (detail::is_container_of_type_v<Spacing,res_type>){    //spacing is coordinates, not uniform
            check_spacing_size(n,spacing.size());
            auto spacing_it = spacing.begin();
            auto t0 = *spacing_it;
            auto t1 = *++spacing_it;
            auto hs = static_cast<res_type>(t1-t0);
            auto e0 = static_cast<res_type>(*first);
            auto e1 = static_cast<res_type>(*++first);
            *dfirst = (e1-e0) / hs;
            for(++first,++dfirst,++spacing_it; first!=last; ++first,++dfirst,++spacing_it){
                const auto& e2 = static_cast<const res_type&>(*first);
                const auto& t2 = *spacing_it;
                const auto& hd = static_cast<const res_type&>(t2-t1);
                *dfirst = make_gradient(e0,e1,e2,hs,hd);
                hs = hd;
                t1 = t2;
                e0 = e1;
                e1 = e2;
            }
            *dfirst = (e1-e0) / hs;
        }else if constexpr (detail::is_static_castable_v<Spacing,res_type>){    //spacing is delta, uniform
            const auto spacing_inv = res_type{1}/static_cast<const res_type&>(spacing);
            const auto spacing_2_inv = res_type{0.5}/static_cast<const res_type&>(spacing);
            auto e0 = *first;
            auto e1 = *++first;
            *dfirst = static_cast<const res_type&>(e1-e0)*spacing_inv;
            for(++first,++dfirst; first!=last; ++first,++dfirst){
                auto e2 = *first;
                *dfirst = static_cast<const res_type&>(e2-e0)*spacing_2_inv;
                e0 = e1;
                e1 = e2;
            }
            *dfirst = static_cast<const res_type&>(e1-e0)*spacing_inv;
        }else{
            static_assert(detail::always_false<Spacing>, "invalid spacing argument");
        }
    }
};

}   //end of namespace math_reduce_operations

//functors to use in statistic reduce functions
namespace statistic_reduce_operations{

//peak-to-peak
struct min_max
{
    template<typename It>
    auto operator()(It first, It last){
        if (first==last){
            throw value_error("cant reduce zero size dimension");
        }
        auto min = *first;
        auto max = min;
        for (++first; first!=last; ++first){
            const auto& e = *first;
            if(e<min){
                min = e;
                continue;
            }
            if(e>max){
                max = e;
            }
        }
        return std::make_pair(min,max);
    }
};
struct ptp
{
    template<typename It>
    auto operator()(It first, It last){
        auto mm = min_max{}(first,last);
        return mm.second - mm.first;
    }
};

template<typename T>
T reduce_empty(){
    if constexpr (gtensor::math::numeric_traits<T>::has_nan()){
        return gtensor::math::numeric_traits<T>::nan();
    }else{
        throw value_error("cant reduce zero size dimension without initial value");
    }
}

template<typename Operation, typename Size>
struct nan_ignoring_counting_operation : Operation
{
    template<typename E>
    auto operator()(const E& e1, const E& e2)->decltype(std::make_pair(Operation::operator()(e1,e2),std::declval<Size>())){
        const auto e1_not_nan = !gtensor::math::isnan(e1);
        const auto e2_not_nan = !gtensor::math::isnan(e2);
        if (e1_not_nan && e2_not_nan){
            return std::make_pair(Operation::operator()(e1,e2),Size{2});
        }else if (e1_not_nan){
            return std::make_pair(e1,Size{1});
        }else if (e2_not_nan){
            return std::make_pair(e2,Size{1});
        }else{
            return std::make_pair(Operation::template value<E>(),Size{0});
        }
    }
    template<typename R, typename S>
    auto operator()(const std::pair<R,S>& r1, const std::pair<R,S>& r2)->decltype(std::make_pair(Operation::operator()(r1.first,r2.first),std::declval<Size>())){
        return std::make_pair(Operation::operator()(r1.first,r2.first),r1.second+r2.second);
    }
    template<typename R, typename E, typename S>
    auto operator()(const std::pair<R,S>& r, const E& e)->decltype(std::make_pair(Operation::operator()(r.first,e),std::declval<Size>())){
        return gtensor::math::isnan(e) ? r : std::make_pair(Operation::operator()(r.first,e),r.second+Size{1});
    }
    template<typename R, typename E, typename S>
    auto operator()(const E& e, const std::pair<R,S>& r)->decltype(std::make_pair(Operation::operator()(r.first,e),std::declval<Size>())){
        return this->operator()(r,e);
    }
};

struct mean
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        if (first == last){
            return reduce_empty<res_type>();
        }
        const auto n = static_cast<res_type>(last-first);
        const auto init = static_cast<res_type>(*first);
        return std::accumulate(++first,last,init,std::plus<void>{}) / n;
    }
};

struct nanmean
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        if (first == last){
            return reduce_empty<res_type>();
        }
        auto res = std::accumulate(first,last,std::make_pair(res_type{0},difference_type{0}),
            [](const auto& r, const auto& e){
                if (gtensor::math::isnan(e)){
                    return r;
                }else{
                    return std::make_pair(r.first+e,r.second+1);
                }
            }
        );
        if (res.second == 0){
            return gtensor::math::numeric_traits<res_type>::nan();
        }else{
            return res.first / static_cast<const res_type&>(res.second);
        }
    }
};

struct var
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        if (first == last){
            return reduce_empty<res_type>();
        }
        const auto res = std::accumulate(first,last,std::make_pair(res_type{0},res_type{0}),
            [](const auto& r, const auto& e){
                return std::make_pair(r.first+e,r.second+e*e);
            }
        );
        const auto n = static_cast<const res_type&>(last-first);
        return (n*res.second - res.first*res.first)/(n*n);
    }
};

struct nanvar
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        if (first == last){
            return reduce_empty<res_type>();
        }
        auto res = std::accumulate(first,last,std::make_tuple(res_type{0},res_type{0},difference_type{0}),
            [](const auto& r, const auto& e)
            {
                if (gtensor::math::isnan(e)){
                    return r;
                }else{
                    return std::make_tuple(std::get<0>(r)+e,std::get<1>(r)+e*e,std::get<2>(r)+1);
                }
            }
        );
        const auto n = std::get<2>(res);
        if (n==0){
            return gtensor::math::numeric_traits<res_type>::nan();
        }else{
            const auto n_ = static_cast<const res_type&>(n);
            return (n_*std::get<1>(res) - std::get<0>(res)*std::get<0>(res))/(n_*n_);
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
        throw value_error("quntile must be in range [0,1]");
    }
}

//Predicate is use in copy_if that copy elements to temporary storage
//quantile: Predicate must throw value_error on nan element, true otherwise
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
        using res_type = gtensor::math::make_floating_point_t<value_type>;
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
            }catch(const value_error&){
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
            throw value_error("");
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
            throw value_error("length of weights not compatible with specified axes");
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
        throw value_error("weights sum to zero");
    }
}

//average
//T is source value_type
template<typename T>
class average
{
    using res_type = gtensor::math::make_floating_point_t<T>;
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
    using res_type = gtensor::math::make_floating_point_t<T>;
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

//functors to use in sort,search reduce functions
namespace sort_search_reduce_operations{

//slide operation to make sorted copy
struct sort
{
    //Comparator can be binary predicate functor or no_value
    template<typename It, typename DstIt, typename Comparator>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, const Comparator& comparator){
        std::copy(first,last,dfirst);
        if constexpr (std::is_same_v<Comparator,detail::no_value>){
            std::sort(dfirst,dlast);
        }else{
            std::sort(dfirst,dlast,comparator);
        }
    }
};

//slide operation to make indexes of elements of sorted tensor
struct argsort
{
    //Comparator can be binary predicate functor or no_value
    template<typename It, typename DstIt, typename Comparator, typename Config>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, const Comparator& comparator, Config){
        using value_type = typename std::iterator_traits<It>::value_type;
        using container_type = typename Config::template container<value_type>;
        using container_size_type = typename container_type::size_type;
        container_type elements(first,last);
        std::iota(dfirst,dlast,0);
        if constexpr (std::is_same_v<Comparator,detail::no_value>){
            std::sort(
                dfirst,
                dlast,
                [&elements](const auto& l, const auto& r){
                    return elements[static_cast<const container_size_type&>(l)] < elements[static_cast<const container_size_type&>(r)];
                }
            );
        }else{
            std::sort(
                dfirst,
                dlast,
                [&elements,comparator](const auto& l, const auto& r){
                    return comparator(elements[static_cast<const container_size_type&>(l)],elements[static_cast<const container_size_type&>(r)]);
                }
            );
        }
    }
};

template<typename It, typename Comparator>
void nth_element_partition_helper(It first, It nth, It last, const Comparator& comparator){
    if constexpr (std::is_same_v<Comparator,detail::no_value>){
        std::nth_element(first,nth,last);
    }else{
        std::nth_element(first,nth,last,comparator);
    }
}

template<typename T, typename Nth>
void check_nth(const T& n, const Nth& nth){
    if constexpr (detail::is_container_v<Nth>){
        if (nth.size() == 0){
            throw value_error("empty nth container");
        }
    }else{
        if (nth>=n || nth<0){
            throw value_error("nth out of bounds");
        }
    }
}

//slide operation to make partially sorted copy
struct nth_element_partition
{
    //Comparator can be binary predicate functor or no_value
    //Nth can be container or scalar
    template<typename It, typename DstIt, typename Nth, typename Comparator, typename Config>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, const Nth& nth, const Comparator& comparator, Config){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        static constexpr bool is_nth_container = detail::is_container_of_type_v<Nth,difference_type>;
        static_assert(is_nth_container || detail::is_static_castable_v<Nth,difference_type>,"invalid nth argument");
        const auto n = last-first;
        std::copy(first,last,dfirst);
        if constexpr (is_nth_container){
            using nth_container_type = typename Config::template container<difference_type>;
            check_nth(n,nth);
            nth_container_type nth_{nth.begin(),nth.end()};
            std::sort(nth_.begin(),nth_.end());
            auto prev = difference_type{-1};
            for (auto nth_it = nth_.begin(),nth_last = nth_.end(); nth_it!=nth_last; ++nth_it){
                const auto& next = static_cast<const difference_type&>(*nth_it);
                check_nth(n,next);
                nth_element_partition_helper(dfirst+(prev+1),dfirst+next,dlast,comparator);
                prev = next;
            }
        }else{  //nth scalar
            const auto nth_ = static_cast<difference_type>(nth);
            check_nth(n,nth_);
            nth_element_partition_helper(dfirst,dfirst+nth_,dlast,comparator);
        }
    }
};

template<typename It, typename Container, typename Comparator>
void nth_element_argpartition_helper(It first, It nth, It last, const Container& elements, const Comparator& comparator){
    using container_size_type = typename Container::size_type;
    if constexpr (std::is_same_v<Comparator,detail::no_value>){
        std::nth_element(first,nth,last,
            [&elements](const auto& l, const auto& r){
                return elements[static_cast<const container_size_type&>(l)] < elements[static_cast<const container_size_type&>(r)];
            }
        );
    }else{
        std::nth_element(first,nth,last,
            [&elements,comparator](const auto& l, const auto& r){
                return comparator(elements[static_cast<const container_size_type&>(l)],elements[static_cast<const container_size_type&>(r)]);
            }
        );
    }
}

//slide operation to make partially sorted copy
struct nth_element_argpartition
{
    //Comparator can be binary predicate functor or no_value
    //Nth can be container or scalar
    template<typename It, typename DstIt, typename Nth, typename Comparator, typename Config>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, const Nth& nth, const Comparator& comparator, Config){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using value_type = typename std::iterator_traits<It>::value_type;
        using elements_container_type = typename Config::template container<value_type>;
        static constexpr bool is_nth_container = detail::is_container_of_type_v<Nth,difference_type>;
        static_assert(is_nth_container || detail::is_static_castable_v<Nth,difference_type>,"invalid nth argument");
        const auto n = last-first;
        elements_container_type elements(first,last);
        std::iota(dfirst,dlast,0);
        if constexpr (is_nth_container){
            using nth_container_type = typename Config::template container<difference_type>;
            check_nth(n,nth);
            nth_container_type nth_{nth.begin(),nth.end()};
            std::sort(nth_.begin(),nth_.end());
            auto prev = difference_type{-1};
            for (auto nth_it = nth_.begin(),nth_last = nth_.end(); nth_it!=nth_last; ++nth_it){
                const auto& next = static_cast<const difference_type&>(*nth_it);
                check_nth(n,next);
                nth_element_argpartition_helper(dfirst+(prev+1),dfirst+next,dlast,elements,comparator);
                prev = next;
            }
        }else{  //nth scalar
            const auto nth_ = static_cast<difference_type>(nth);
            check_nth(n,nth_);
            nth_element_argpartition_helper(dfirst,dfirst+nth_,dlast,elements,comparator);
        }
    }
};


//reduce operation, return extremum index
template<typename Comparator, typename ThrowNanResult = std::false_type>
struct argextremum_nanargextremum
{
    template<typename It>
    auto operator()(It first, It last){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        if (first == last){
            throw value_error("cant reduce zero size dimension");
        }
        Comparator comparator{};
        auto init = *first;
        difference_type res{0};
        difference_type i{1};
        for(++first; first!=last; ++first,++i){
            const auto& e = *first;
            if (comparator(init,e)){
                init = e;
                res = i;
            }
        }
        if constexpr (ThrowNanResult::value){
            if (gtensor::math::isnan(init)){
                throw value_error("all nan slice");
            }
        }
        return res;
    }
};

template<typename Comparator>
struct nan_propagate_comparator
{
    const Comparator comparator{};
    template<typename R, typename E>
    auto operator()(const R& r, const E& e){
        return gtensor::math::isnan(r) ? false : gtensor::math::isnan(e) ? true : comparator(e,r);
    }
};

template<typename Comparator>
struct nan_ignore_comparator
{
    const Comparator comparator{};
    template<typename R, typename E>
    auto operator()(const R& r, const E& e){
        return gtensor::math::isnan(e) ? false : gtensor::math::isnan(r) ? true : comparator(e,r);
    }
};

using argmin = argextremum_nanargextremum<nan_propagate_comparator<std::less<void>>>;
using argmax = argextremum_nanargextremum<nan_propagate_comparator<std::greater<void>>>;
using nanargmin = argextremum_nanargextremum<nan_ignore_comparator<std::less<void>>,std::true_type>;
using nanargmax = argextremum_nanargextremum<nan_ignore_comparator<std::greater<void>>,std::true_type>;

struct count_nonzero
{
    template<typename It>
    auto operator()(It first, It last){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        auto counter = [](const auto& r, const auto& e){
            if (static_cast<bool>(e)){
                return r+1;
            }else{
                return r;
            }
        };
        return std::accumulate(first,last,difference_type{0},counter);
    }
};

}

}   //end of namespace gtensor
#endif