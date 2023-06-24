#ifndef REDUCE_OPERATIONS_HPP_
#define REDUCE_OPERATIONS_HPP_
#include <functional>
#include <algorithm>
#include "common.hpp"
#include "math.hpp"


namespace gtensor{

//tensor functor operations implementation to use in tensor math reduce functions
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

//test if initial argument contain valid initial value
template<typename Initial, typename T> constexpr bool is_initial_v = !std::is_same_v<Initial,gtensor::detail::no_value> && detail::is_static_castable_v<Initial,T>;
//test if functor F has initial value
template<typename F, typename T, typename=void> constexpr bool has_initial_v = false;
template<typename F, typename T> constexpr bool has_initial_v<F,T,std::void_t<decltype(F::template value<T>)>> = true;

template<typename Functor, typename It, typename Initial>
auto reduce_empty(const It&, const Initial& initial){
    using value_type = typename std::iterator_traits<It>::value_type;
    if constexpr (is_initial_v<Initial,value_type>){
        return static_cast<value_type>(initial);
    }else if constexpr(has_initial_v<Functor,value_type>){
        return Functor::template value<value_type>;
    }else{  //no initial, throw
        throw reduce_exception("cant reduce zero size dimension without initial value");
        return value_type{};    //need same return type
    }
}

template<typename Functor, typename It, typename Initial>
auto make_initial(It& first, const Initial& initial){
    using value_type = typename std::iterator_traits<It>::value_type;
    if constexpr (is_initial_v<Initial,value_type>){
        return static_cast<value_type>(initial);
    }else if constexpr(has_initial_v<Functor,value_type>){
        return Functor::template value<value_type>;
    }else{  //use first element as initial and inc first
        auto res = *first;
        return ++first,res;
    }
}

//find extremum propagating nan
//Comparator should be like std::less for minimum and like std::greater for maximum
//Comparator result must be false when any argument is nan
template<typename Comparator>
struct extremum
{
    template<typename It, typename Initial = gtensor::detail::no_value>
    auto operator()(It first, It last, const Initial& initial = Initial{}){
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first == last){
            return reduce_empty<Comparator>(first,initial);
        }
        Comparator comparator{};
        auto res = make_initial<Comparator>(first,initial);
        if (gtensor::math::isnan(res)){
            return res;
        }
        for(;first!=last; ++first){
            const auto& e = *first;
            if constexpr (gtensor::math::numeric_traits<value_type>::has_nan()){
                if (gtensor::math::isnan(e)){
                    return e;
                }
            }
            if (comparator(e,res)){
                res = e;
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
    template<typename It, typename Initial = gtensor::detail::no_value>
    auto operator()(It first, It last, const Initial& initial = Initial{}){
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first == last){
            return reduce_empty<Comparator>(first,initial);
        }
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
        auto res = make_initial<Comparator>(first,initial);
        for(;first!=last; ++first){
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
    template<typename It, typename Initial>
    auto operator()(It first, It last, const Initial& initial){
        if (first == last){
            return reduce_empty<Operation>(first,initial);
        }
        Operation operation{};
        auto res = make_initial<Operation>(first,initial);
        for(;first!=last; ++first){
            res = operation(res,*first);   //must return nan if any of arguments is nan
        }
        return res;
    }
};

//accumulate ignoring nan (like treating nan as zero for sum and as one for prod)
//Operation should be like std::plus for sum and like std::multiplies for prod
//Operation result must be nan when any argument is nan
template<typename Operation>
struct nanaccumulate
{
    template<typename It, typename Initial>
    auto operator()(It first, It last, const Initial& initial){
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first == last){
            return reduce_empty<Operation>(first,initial);
        }
        Operation operation{};
        auto res = make_initial<Operation>(first,initial);
        for(;first!=last; ++first){
            if constexpr (gtensor::math::numeric_traits<value_type>::has_nan()){
                const auto& e = *first;
                if (!gtensor::math::isnan(e)){
                    res = operation(res,e);
                }
            }else{
                res = operation(res,*first);
            }
        }
        return res;
    }
};

struct plus : public std::plus<void>{template<typename T> inline static const T value = T(0);};
struct multiplies : public std::multiplies<void>{template<typename T> inline static const T value = T(1);};

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
    void operator()(It first, It last, DstIt dfirst, DstIt dlast){
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
            cumulate<Operation>{}(first,last,dfirst,dlast);
        }
    }
};

using cumsum = cumulate<plus>;
using cumprod = cumulate<multiplies>;
using nancumsum = nancumulate<plus>;
using nancumprod = nancumulate<multiplies>;

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
            throw reduce_exception("length of spacing not compatible with specified axes");
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
    auto operator()(It first, It last, DstIt dfirst, DstIt dlast, const Spacing& spacing){
        using value_type = typename std::iterator_traits<It>::value_type;
        using dst_value_type = typename std::iterator_traits<DstIt>::value_type;
        using res_type = gtensor::math::make_floating_point_t<value_type>;
        static_assert(std::is_same_v<dst_value_type,res_type>,"invalid DstIt value_type");
        const auto n = last-first;
        if (n<2){
            throw reduce_exception("gradient requires at least 2 points");
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

}   //end of namespace gtensor
#endif