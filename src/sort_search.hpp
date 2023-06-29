#ifndef SORT_SEARCH_HPP_
#define SORT_SEARCH_HPP_

#include "reduce_operations.hpp"
#include "reduce.hpp"

namespace gtensor
{

//tensor sort,search functions implementation

struct sort_search
{
    //sort,search functions along given axis or axes
    //axes may be scalar or container if multiple axes permitted
    //empty container means apply function along all axes

    //return sorted copy of tensor, axis is scalar
    template<typename...Ts, typename DimT, typename Comparator>
    static auto sort(const basic_tensor<Ts...>& t, const DimT& axis, const Comparator& comparator){
        using index_type = typename basic_tensor<Ts...>::index_type;
        const index_type window_size = 1;
        const index_type window_step = 1;
        return slide(t,axis,sort_search_reduce_operations::sort{}, window_size, window_step, comparator);
    }

    //return indexes that sort tensor along axis, axis is scalar
    template<typename...Ts, typename DimT, typename Comparator>
    static auto argsort(const basic_tensor<Ts...>& t, const DimT& axis, const Comparator& comparator){
        using config_type = typename basic_tensor<Ts...>::config_type;
        using index_type = typename basic_tensor<Ts...>::index_type;
        const index_type window_size = 1;
        const index_type window_step = 1;
        return slide<index_type>(t,axis,sort_search_reduce_operations::argsort{}, window_size, window_step, comparator, config_type{});
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

//return sorted copy of tensor, axis is scalar
//Comparator is binary predicate to compare elements
//if Comparator not given operator< is used
GTENSOR_TENSOR_SORT_ROUTINE(sort,sort);

//return indexes that sort tensor along axis, axis is scalar
//Comparator is binary predicate to compare elements
//if Comparator not given operator< is used
GTENSOR_TENSOR_SORT_ROUTINE(argsort,argsort);


}   //end of namespace gtensor
#endif