#ifndef SORT_SEARCH_HPP_
#define SORT_SEARCH_HPP_

#include "indexing.hpp"
#include "reduce.hpp"
#include "reduce_operations.hpp"

namespace gtensor
{

#define GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(NAME,F)\
template<typename...Ts, typename Axes>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    return reduce(t,axes,F{},keep_dims);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return reduce_flatten(t,F{},keep_dims);\
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

    template<typename FlattenOrder, typename...Ts>
    static auto unique_flatten(const basic_tensor<Ts...>& t){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using value_type = typename tensor_type::value_type;
        using config_type = typename tensor_type::config_type;
        using shape_type = typename tensor_type::shape_type;
        using container_type = typename config_type::template container<value_type>;
        auto a = t.template traverse_order_adapter<FlattenOrder>();
        container_type tmp(a.begin(),a.end());
        std::sort(tmp.begin(),tmp.end());
        const auto unique_last = std::unique(tmp.begin(),tmp.end());
        const auto n_unique = unique_last - tmp.begin();
        tensor<value_type,order,config_type> res(detail::make_shape_of_type<shape_type>(n_unique),tmp.begin(),unique_last);
        return res;
    }

    template<typename...Ts, typename Axis=detail::no_value>
    static auto unique(const basic_tensor<Ts...>& t, const Axis& axis_=Axis{}){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using value_type = typename tensor_type::value_type;
        using config_type = typename tensor_type::config_type;
        using index_type = typename tensor_type::index_type;
        using shape_type = typename tensor_type::shape_type;
        if (t.dim() == 0 || t.dim() == 1){
            return unique_flatten<order>(t);
        }
        if constexpr (std::is_same_v<Axis,detail::no_value>){
            return unique_flatten<config::c_order>(t);
        }else{
            const auto& shape = t.shape();
            const auto axis = detail::make_axis(shape,axis_);
            const auto axis_size = shape[axis];
            const auto chunk_size = t.size()/axis_size;
            const auto predicate = detail::make_traverse_predicate(axis,std::true_type{});  //inverse, traverse all but axis
            const auto strides = detail::make_strides_div_predicate<config_type>(shape,predicate,config::c_order{});
            auto traverser = detail::make_forward_traverser(t.shape(),t.create_walker(),detail::make_traverse_predicate(axis)); //traverse along axis
            using walker_type = std::remove_cv_t<std::remove_reference_t<decltype(traverser.walker())>>;
            auto make_iterator = [&shape,&strides,&predicate,&chunk_size](const auto& w, const auto& pos){
                return pos == 0 ?
                    detail::make_axes_iterator<config::c_order>(shape,strides,w,0,predicate):
                    detail::make_axes_iterator<config::c_order>(shape,strides,w,chunk_size,predicate);
            };
            struct range{
                using make_iterator_type = decltype(make_iterator);
                walker_type walker_;
                const make_iterator_type* make_iterator_;
                range(const walker_type& walker__, const make_iterator_type& make_iterator__):
                    walker_{walker__},
                    make_iterator_{&make_iterator__}
                {}
                auto begin()const{return (*make_iterator_)(walker_,0);}
                auto end()const{return (*make_iterator_)(walker_,1);}
                bool operator<(const range& other)const{
                    return std::lexicographical_compare(
                        begin(),
                        end(),
                        other.begin(),
                        other.end()
                    );
                }
                bool operator==(const range& other)const{
                    return std::equal(
                        begin(),
                        end(),
                        other.begin()
                    );
                }
            };
            // auto range_to_str = [](auto range){
            //     std::stringstream ss{};
            //     for (auto i : range){
            //         ss<<i<<" ";
            //     }
            //     return ss.str();
            // };
            using container_type = typename config_type::template container<range>;
            container_type chunks{};
            do{
                chunks.emplace_back(traverser.walker(),make_iterator);
            }while(traverser.template next<order>());
            std::sort(chunks.begin(),chunks.end());
            const auto unique_last = std::unique(chunks.begin(),chunks.end());
            const auto n_unique = unique_last - chunks.begin();
            shape_type res_shape_{shape};
            res_shape_[axis] = static_cast<const index_type&>(n_unique);
            tensor<value_type,order,config_type> res(std::move(res_shape_));
            const auto& res_shape = res.shape();
            const auto res_strides = detail::make_strides_div_predicate<config_type>(res_shape,predicate,config::c_order{});
            auto res_traverser = detail::make_forward_traverser(res_shape,res.create_walker(),detail::make_traverse_predicate(axis)); //traverse along axis
            for (auto chunks_it=chunks.begin(); chunks_it!=unique_last; ++chunks_it,res_traverser.template next<order>()){
                const auto& chunk = *chunks_it;
                std::copy(
                    chunk.begin(),
                    chunk.end(),
                    detail::make_axes_iterator<config::c_order>(res_shape,res_strides,res_traverser.walker(),0,predicate)
                );
            }
            return res;
        }
    }

};  //end of struct sort_search

//tensor sort_search frontend
//frontend uses compile-time dispatch to select implementation, see module_selector.hpp

#define GTENSOR_TENSOR_SORT_ROUTINE(NAME,F)\
template<typename...Ts, typename DimT=int, typename Comparator=std::less<void>>\
static auto NAME(const basic_tensor<Ts...>& t, const DimT& axis=-1, const Comparator& comparator=std::less<void>{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,axis,comparator);\
}

#define GTENSOR_TENSOR_PARTITION_ROUTINE(NAME,F)\
template<typename...Ts, typename Nth, typename DimT=int, typename Comparator=std::less<void>>\
static auto NAME(const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis=-1, const Comparator& comparator=std::less<void>{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,nth,axis,comparator);\
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
    return sort_search_selector_t<config_type>::F(t,keep_dims);\
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

//returns the sorted unique elements of a tensor
template<typename...Ts, typename Axis=detail::no_value>
auto unique(const basic_tensor<Ts...>& t, const Axis& axis=Axis{}){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return sort_search_selector_t<config_type>::unique(t,axis);
}


#undef GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION
#undef GTENSOR_TENSOR_SORT_ROUTINE
#undef GTENSOR_TENSOR_PARTITION_ROUTINE
#undef GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE
}   //end of namespace gtensor
#endif