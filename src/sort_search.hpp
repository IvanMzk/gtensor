#ifndef SORT_SEARCH_HPP_
#define SORT_SEARCH_HPP_

#include "reduce_operations.hpp"
#include "reduce.hpp"

namespace gtensor
{

#define GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(NAME,F)\
template<typename...Ts, typename Axes>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    return reduce(t,axes,F{},keep_dims);\
}

//tensor sort,search functions implementation
struct sort_search
{
    //sort,search functions along given axis or axes
    //axes may be scalar or container if multiple axes permitted
    //empty container means apply function along all axes

    //return sorted copy of tensor, axis is scalar
    //Comparator is binary predicate functor, like std::less<void> or std::greater<void>
    template<typename...Ts, typename DimT, typename Comparator>
    static auto sort(const basic_tensor<Ts...>& t, const DimT& axis, const Comparator& comparator){
        using index_type = typename basic_tensor<Ts...>::index_type;
        const index_type window_size = 1;
        const index_type window_step = 1;
        return slide(t,axis,sort_search_reduce_operations::sort{}, window_size, window_step, comparator);
    }

    //return indexes that sort tensor along axis, axis is scalar
    //Comparator is binary predicate functor, like std::less<void> or std::greater<void>
    template<typename...Ts, typename DimT, typename Comparator>
    static auto argsort(const basic_tensor<Ts...>& t, const DimT& axis, const Comparator& comparator){
        using config_type = typename basic_tensor<Ts...>::config_type;
        using index_type = typename basic_tensor<Ts...>::index_type;
        const index_type window_size = 1;
        const index_type window_step = 1;
        return slide<index_type>(t,axis,sort_search_reduce_operations::argsort{}, window_size, window_step, comparator, config_type{});
    }

    //return partially sorted copy of tensor, axis is scalar
    //Nth can be container or scalar
    //Comparator is binary predicate functor, like std::less<void> or std::greater<void>
    template<typename...Ts, typename Nth, typename DimT, typename Comparator>
    static auto partition(const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis, const Comparator& comparator){
        using config_type = typename basic_tensor<Ts...>::config_type;
        using index_type = typename basic_tensor<Ts...>::index_type;
        const index_type window_size = 1;
        const index_type window_step = 1;
        return slide(t,axis,sort_search_reduce_operations::nth_element_partition{}, window_size, window_step, nth, comparator, config_type{});
    }

    //return indexes that partially sort tensor along axis, axis is scalar
    //Nth can be container or scalar
    //Comparator is binary predicate functor, like std::less<void> or std::greater<void>
    template<typename...Ts, typename Nth, typename DimT, typename Comparator>
    static auto argpartition(const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis, const Comparator& comparator){
        using config_type = typename basic_tensor<Ts...>::config_type;
        using index_type = typename basic_tensor<Ts...>::index_type;
        const index_type window_size = 1;
        const index_type window_step = 1;
        return slide<index_type>(t,axis,sort_search_reduce_operations::nth_element_argpartition{}, window_size, window_step, nth, comparator, config_type{});
    }


    GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(argmin,sort_search_reduce_operations::argmin);
    GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(argmax,sort_search_reduce_operations::argmax);
    GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(nanargmin,sort_search_reduce_operations::nanargmin);
    GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(nanargmax,sort_search_reduce_operations::nanargmax);

    GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(count_nonzero,sort_search_reduce_operations::count_nonzero);

    template<typename...Ts>
    static auto nonzero(const basic_tensor<Ts...>& t){
        using order = typename basic_tensor<Ts...>::order;
        using config_type = typename basic_tensor<Ts...>::config_type;
        using index_type = typename basic_tensor<Ts...>::index_type;
        using container_type = typename config_type::template container<index_type>;
        using container_difference_type = typename container_type::difference_type;
        using result_config_type = config::extend_config_t<config_type,index_type>;
        using result_tensor_type = tensor<index_type,order,result_config_type>;
        using result_container_type = typename config_type::template container<result_tensor_type>;
        const auto dim = static_cast<container_difference_type>(t.dim());
        if (t.empty()){
            result_container_type res{};
            detail::reserve(res,dim);
            for (container_difference_type i=0; i!=dim; ++i){
                res.emplace_back();
            }
            return res;
        }else{
            container_type indexes{};
            const auto n = t.size()*static_cast<index_type>(t.dim());
            detail::reserve(indexes,n);
            walker_forward_traverser<config_type,decltype(t.create_walker())> traverser{t.shape(),t.create_walker()};
            do{
                if (static_cast<bool>(*traverser.walker())){
                    std::copy(traverser.index().begin(),traverser.index().end(),std::back_inserter(indexes));
                }
            }while(traverser.template next<config::c_order>());
            const auto nonzero_n = static_cast<index_type>(indexes.size() / dim);
            result_container_type res{};
            detail::reserve(res,dim);
            for (container_difference_type i=0; i!=dim; ++i){
                res.emplace_back(std::initializer_list<index_type>{nonzero_n},0);
            }
            if (!indexes.empty()){
                auto indexes_first = indexes.begin();
                for (auto res_it=res.begin(),res_last=res.end(); res_it!=res_last; ++res_it,++indexes_first){
                    auto& e = *res_it;
                    auto a = e.template traverse_order_adapter<order>();
                    auto indexes_it = indexes_first;
                    for (auto it=a.begin(),last=a.end(); it!=last; ++it,indexes_it+=dim){
                        *it = *indexes_it;
                    }
                }
            }
            return res;
        }
    }

    template<typename...Ts>
    static auto argwhere(const basic_tensor<Ts...>& t){
        using order = typename basic_tensor<Ts...>::order;
        using config_type = typename basic_tensor<Ts...>::config_type;
        using index_type = typename basic_tensor<Ts...>::index_type;
        using result_config_type = config::extend_config_t<config_type,index_type>;
        using result_tensor_type = tensor<index_type,order,result_config_type>;
        using container_type = typename config_type::template container<index_type>;
        using container_difference_type = typename container_type::difference_type;
        const auto dim = static_cast<index_type>(t.dim());
        if (t.empty()){
            return result_tensor_type({0,dim},0);
        }else{
            container_type indexes{};
            const auto n = t.size()*dim;
            detail::reserve(indexes,n);
            walker_forward_traverser<config_type,decltype(t.create_walker())> traverser{t.shape(),t.create_walker()};
            do{
                if (static_cast<bool>(*traverser.walker())){
                    std::copy(traverser.index().begin(),traverser.index().end(),std::back_inserter(indexes));
                }
            }while(traverser.template next<config::c_order>());
            const auto nonzero_n = static_cast<index_type>(indexes.size()) / dim;
            if constexpr (std::is_same_v<order,config::c_order>){
                return result_tensor_type({nonzero_n,dim},indexes.begin(),indexes.end());
            }else{
                result_tensor_type res({nonzero_n,dim},0);
                if (!indexes.empty()){
                    auto dim_ = static_cast<container_difference_type>(dim);
                    auto res_it = res.template traverse_order_adapter<order>().begin();
                    auto indexes_first = indexes.begin();
                    for(index_type i=0; i!=dim; ++i,++indexes_first){
                        auto indexes_it = indexes_first;
                        for (index_type j=0; j!=nonzero_n; ++j,indexes_it+=dim_,++res_it){
                            *res_it = *indexes_it;
                        }
                    }
                }
                return res;
            }
        }
    }

};  //end of struct sort_search

//tensor sort_search frontend
//frontend uses compile-time dispatch to select implementation, see module_selector.hpp

#define GTENSOR_TENSOR_SORT_ROUTINE(NAME,F)\
template<typename...Ts, typename DimT, typename Comparator>\
static auto NAME(const basic_tensor<Ts...>& t, const DimT& axis, const Comparator& comparator){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,axis,comparator);\
}\
template<typename...Ts, typename DimT>\
static auto NAME(const basic_tensor<Ts...>& t, const DimT& axis){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,axis,detail::no_value{});\
}

#define GTENSOR_TENSOR_PARTITION_ROUTINE(NAME,F)\
template<typename...Ts, typename Nth, typename DimT, typename Comparator>\
static auto NAME(const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis, const Comparator& comparator){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,nth,axis,comparator);\
}\
template<typename...Ts, typename Nth, typename DimT>\
static auto NAME(const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,nth,axis,detail::no_value{});\
}

#define GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(NAME,F)\
template<typename...Ts, typename Axes>\
auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,axes,keep_dims);\
}\
template<typename...Ts, typename DimT>\
auto NAME(const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,axes,keep_dims);\
}\
template<typename...Ts>\
auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,std::initializer_list<typename basic_tensor<Ts...>::dim_type>{},keep_dims);\
}

//return sorted copy of tensor, axis is scalar
//Comparator is binary predicate functor, like std::less<void> or std::greater<void>
//if Comparator not given operator< is used
GTENSOR_TENSOR_SORT_ROUTINE(sort,sort);

//return indexes that sort tensor along axis, axis is scalar
//Comparator is binary predicate functor, like std::less<void> or std::greater<void>
//if Comparator not given operator< is used
GTENSOR_TENSOR_SORT_ROUTINE(argsort,argsort);

//return partially sorted copy of tensor, axis is scalar
//Nth can be container or scalar
//Comparator is binary predicate functor, like std::less<void> or std::greater<void>, if Comparator not given operator< is used
GTENSOR_TENSOR_PARTITION_ROUTINE(partition,partition);

//return indexes that partially sort tensor along axis, axis is scalar
//Nth can be container or scalar
//Comparator is binary predicate functor, like std::less<void> or std::greater<void>, if Comparator not given operator< is used
GTENSOR_TENSOR_PARTITION_ROUTINE(argpartition,argpartition);

//index of min element along given axes, propagating nan
//axes can be container or scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(argmin,argmin);

//index of max element along given axes, propagating nan
//axes can be container or scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(argmax,argmax);

//index of min element along given axes, ignoring nan
//axes can be container or scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(nanargmin,nanargmin);

//index of max element along given axes, ignoring nan
//axes can be container or scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(nanargmax,nanargmax);

//count number of values for which static_cast<bool>(e) evaluates to true
//axes can be container or scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(count_nonzero,count_nonzero);

template<typename...Ts>
auto nonzero(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return sort_search_selector_t<config_type>::nonzero(t);
}

template<typename...Ts>
auto argwhere(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return sort_search_selector_t<config_type>::argwhere(t);
}

}   //end of namespace gtensor
#endif