#ifndef SORT_SEARCH_HPP_
#define SORT_SEARCH_HPP_

#include "indexing.hpp"
#include "reduce.hpp"
#include "reduce_operations.hpp"

namespace gtensor
{

namespace detail{

template<typename T> inline constexpr bool is_tuple_v = false;
template<typename...Us> inline constexpr bool is_tuple_v<std::tuple<Us...>> = true;

template<typename Tuple, typename V>
auto make_tuple_or_add(Tuple&& t, V&& v){
    using Tuple_ = std::remove_cv_t<std::remove_reference_t<Tuple>>;
    if constexpr (is_tuple_v<Tuple_>){
        return std::tuple_cat(std::forward<Tuple>(t),std::make_tuple(std::forward<V>(v)));
    }else{
        return std::make_tuple(std::forward<Tuple>(t),std::forward<V>(v));
    }
}

template<typename DimT, typename Axis>
void check_unique_args(const DimT& dim, const Axis& axis_){
    if constexpr (!std::is_same_v<Axis,no_value>){
        auto axis = make_axis(dim,axis_);
        if (dim==0){
            if (axis != 0){
                throw axis_error("axis out of bounds");
            }
        }else{
            if (axis >= dim){
                throw axis_error("axis out of bounds");
            }
        }
    }
}

template<typename DimT, typename Sorter>
void check_searchsorted_args(const DimT& dim, const Sorter& sorter){
    if (dim!=1){
        throw value_error("t must be 1d");
    }
    if constexpr (detail::is_tensor_v<Sorter>){
        if (sorter.dim()!=1){
            throw value_error("sorter must be 1d");
        }
    }
}

}


#define GTENSOR_TENSOR_ARG_SEARCH_REDUCE_FUNCTION(NAME,F)\
template<typename...Ts, typename Axis>\
static auto NAME(const basic_tensor<Ts...>& t, const Axis& axis, bool keep_dims = false){\
    return reduce(t,axis,F{},keep_dims,false);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return reduce_flatten(t,F{},keep_dims,false);\
}

#define GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(NAME,F)\
template<typename...Ts, typename Axes>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    return reduce(t,axes,F{},keep_dims,true);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return reduce_flatten(t,F{},keep_dims,true);\
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
        using value_type = typename basic_tensor<Ts...>::value_type;
        const index_type window_size = 1;
        const index_type window_step = 1;
        return slide<value_type>(t,axis,sort_search_reduce_operations::sort{}, window_size, window_step, comparator);
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
        using value_type = typename basic_tensor<Ts...>::value_type;
        const index_type window_size = 1;
        const index_type window_step = 1;
        return slide<value_type>(t,axis,sort_search_reduce_operations::nth_element_partition{}, window_size, window_step, nth, comparator, config_type{});
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


    GTENSOR_TENSOR_ARG_SEARCH_REDUCE_FUNCTION(argmin,sort_search_reduce_operations::argmin);
    GTENSOR_TENSOR_ARG_SEARCH_REDUCE_FUNCTION(argmax,sort_search_reduce_operations::argmax);
    GTENSOR_TENSOR_ARG_SEARCH_REDUCE_FUNCTION(nanargmin,sort_search_reduce_operations::nanargmin);
    GTENSOR_TENSOR_ARG_SEARCH_REDUCE_FUNCTION(nanargmax,sort_search_reduce_operations::nanargmax);

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
                    auto a = e.traverse_order_adapter(order{});
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
            return result_tensor_type({index_type{0},dim},0);
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
                    auto res_it = res.traverse_order_adapter(order{}).begin();
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

    template<typename...Ts, typename ReturnIndex=std::false_type, typename ReturnInverse=std::false_type, typename ReturnCounts=std::false_type, typename Axis=detail::no_value>
    static auto unique(const basic_tensor<Ts...>& t, ReturnIndex return_index=ReturnIndex{}, ReturnInverse return_inverse=ReturnInverse{}, ReturnCounts return_counts=ReturnCounts{}, const Axis& axis_=Axis{}){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using value_type = typename tensor_type::value_type;
        using config_type = typename tensor_type::config_type;
        using index_type = typename tensor_type::index_type;
        using shape_type = typename tensor_type::shape_type;
        detail::check_unique_args(t.dim(),axis_);
        if (t.dim() == 0 || t.dim() == 1){
            return unique_flatten<order>(t,return_index,return_inverse,return_counts);
        }
        if constexpr (std::is_same_v<Axis,detail::no_value>){
            return unique_flatten<config::c_order>(t,return_index,return_inverse,return_counts);
        }else{
            const auto& shape = t.shape();
            const auto axis = detail::make_axis(shape,axis_);
            const auto axis_size = shape[axis];
            const auto chunk_size = axis_size==0 ? index_type{0} : t.size()/axis_size;
            auto axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(shape,axis,config::c_order{});
            auto traverser = axes_iterator_maker.create_forward_traverser(t.create_walker(),std::false_type{});
            using walker_type = std::remove_cv_t<std::remove_reference_t<decltype(traverser.walker())>>;
            auto make_iterator = [&axes_iterator_maker](const auto& w, const auto& pos){
                return pos == 0 ? axes_iterator_maker.begin_complement(w,std::true_type{}) : axes_iterator_maker.end_complement(w,std::true_type{});
            };
            using make_iterator_type = decltype(make_iterator);

            struct range{
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
            struct range_index : range{
                index_type idx_;
                range_index(const walker_type& walker__, const make_iterator_type& make_iterator__, const index_type& idx__):
                    range(walker__,make_iterator__),
                    idx_{idx__}
                {}
                auto index()const{return idx_;}
            };

            static constexpr bool need_index = return_index.value || return_inverse.value;
            using range_type = std::conditional_t<need_index,range_index,range> ;
            using container_type = typename config_type::template container<range_type>;
            container_type chunks{};
            index_type i{0};
            if (chunk_size!=0){
                detail::reserve(chunks,axis_size);
                do{
                    if constexpr (need_index){
                        chunks.emplace_back(traverser.walker(),make_iterator,i);
                        ++i;
                    }else{
                        chunks.emplace_back(traverser.walker(),make_iterator);
                    }
                }while(traverser.template next<order>());
            }
            std::sort(chunks.begin(),chunks.end());
            using index_container_type = typename config_type::template container<index_type>;
            using container_difference_type = typename index_container_type::difference_type;
            index_container_type inverse{};
            index_container_type counts{};
            const auto chunks_size = chunks.size();
            if constexpr(return_inverse.value){
                inverse.assign(static_cast<const container_difference_type&>(chunks_size),0);
            }
            if constexpr(return_counts.value){
                detail::reserve(counts,chunks_size);
            }
            const auto unique_last = unique_helper(chunks.begin(),chunks.end(),inverse,counts,return_inverse,return_counts);
            const auto n_unique = static_cast<const index_type&>(unique_last - chunks.begin());
            shape_type res_shape_{shape};
            res_shape_[axis] = n_unique;
            tensor<value_type,order,config_type> res(std::move(res_shape_));
            const auto& res_shape = res.shape();
            auto res_axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(res_shape,axis,config::c_order{});
            auto res_traverser = res_axes_iterator_maker.create_forward_traverser(res.create_walker(),std::false_type{});
            for (auto chunks_it=chunks.begin(); chunks_it!=unique_last; ++chunks_it,res_traverser.template next<order>()){
                const auto& chunk = *chunks_it;
                std::copy(
                    chunk.begin(),
                    chunk.end(),
                    res_axes_iterator_maker.begin_complement(res_traverser.walker(),std::true_type{})
                );
            }
            using index_tensor_type = tensor<index_type,order,config_type>;
            if constexpr (return_index.value){
                index_tensor_type unique_index(detail::make_shape_of_type<shape_type>(n_unique));
                std::transform(chunks.begin(),unique_last,unique_index.begin(),[](const auto& chunk){return chunk.index();});
                return make_unique_return<index_tensor_type>(std::make_tuple(res,unique_index),inverse,counts,return_inverse,return_counts);
            }else{
                return make_unique_return<index_tensor_type>(std::move(res),inverse,counts,return_inverse,return_counts);
            }
        }
    }

    //t must be 1d tensor
    //v may be tensor or scalar
    //side should be like std::false_type for left side (lower bound)
    //side should be like std::true_type for right side (upper bound)
    //sorter may be 1d tensor of indexes to sort t or no_value, if no_value t is considered sorted in ascending order
    template<typename...Ts, typename V, typename Side=std::false_type, typename Sorter=detail::no_value>
    static auto searchsorted(const basic_tensor<Ts...>& t, const V& v, Side side=Side{}, const Sorter& sorter=Sorter{}){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using value_type = typename tensor_type::value_type;
        using config_type = typename tensor_type::config_type;
        using index_type = typename tensor_type::index_type;
        static constexpr bool is_v_tensor = detail::is_tensor_of_type_v<V,value_type>;
        static_assert(is_v_tensor || std::is_convertible_v<V,value_type>,"v must be tensor or scalar");
        static_assert(std::is_same_v<Sorter,detail::no_value> || detail::is_tensor_of_type_v<Sorter,index_type>,"sorter must be tensor or no_value");

        detail::check_searchsorted_args(t.dim(),sorter);
        auto make_sorted = [&t,&sorter](){
            (void)sorter;
            if constexpr (std::is_same_v<Sorter,detail::no_value>){
                return t;
            }else{
                return t(sorter);
            }
        };

        auto find_index = [side](auto first, auto last, const auto& val){
            if constexpr (side.value){
                return std::upper_bound(first,last,val) - first;
            }else{
                return std::lower_bound(first,last,val) - first;
            }
        };
        auto sorted = make_sorted();
        auto a = sorted.traverse_order_adapter(order{});
        if constexpr (detail::is_tensor_v<V>){
            using res_type = tensor<index_type,order,config_type>;
            res_type res(v.shape());
            auto a_v = v.traverse_order_adapter(order{});
            std::transform(a_v.begin(),a_v.end(),res.traverse_order_adapter(order{}).begin(),[&a,&find_index](const auto& val){return find_index(a.begin(),a.end(),val);});
            return res;
        }else{
            return find_index(a.begin(),a.end(),v);
        }
    }

private:

    template<typename FlattenOrder, typename...Ts, typename ReturnIndex, typename ReturnInverse, typename ReturnCounts>
    static auto unique_flatten(const basic_tensor<Ts...>& t, ReturnIndex return_index, ReturnInverse return_inverse, ReturnCounts return_counts){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using value_type = typename tensor_type::value_type;
        using config_type = typename tensor_type::config_type;
        using index_type = typename tensor_type::index_type;
        using shape_type = typename tensor_type::shape_type;
        auto a = t.traverse_order_adapter(FlattenOrder{});
        const auto size = t.size();
        struct element{
            value_type v_;
            index_type idx_;
            element(const value_type& v__, const index_type& idx__):
                v_{v__},
                idx_{idx__}
            {}
            value_type value()const{return v_;}
            index_type index()const{return idx_;}
            bool operator<(const element& other)const{
                return v_<other.v_;
            }
            bool operator==(const element& other)const{
                return v_==other.v_;
            }
        };
        static constexpr bool need_index = return_index.value || return_inverse.value;
        using element_type = std::conditional_t<need_index,element,value_type>;;
        using container_type = typename config_type::template container<element_type>;
        container_type tmp{};
        index_type i{0};
        if constexpr (need_index){
            detail::reserve(tmp,size);
            for (auto it=a.begin(),last=a.end(); it!=last; ++it,++i){
                tmp.emplace_back(*it,i);
            }
        }else{
            tmp.assign(a.begin(),a.end());
        }
        std::sort(tmp.begin(),tmp.end());

        using index_container_type = typename config_type::template container<index_type>;
        using container_difference_type = typename index_container_type::difference_type;
        index_container_type inverse{};
        index_container_type counts{};
        if constexpr(return_inverse.value){
            inverse.assign(static_cast<const container_difference_type&>(size),0);
        }
        if constexpr(return_counts.value){
            detail::reserve(counts,size);
        }
        const auto unique_last = unique_helper(tmp.begin(),tmp.end(),inverse,counts,return_inverse,return_counts);
        const auto n_unique = unique_last - tmp.begin();
        tensor<value_type,order,config_type> res(detail::make_shape_of_type<shape_type>(n_unique));
        if constexpr (need_index){
            std::transform(tmp.begin(),unique_last,res.begin(),[](const auto& e){return e.value();});
        }else{
            std::copy(tmp.begin(),unique_last,res.begin());
        }
        using index_tensor_type = tensor<index_type,order,config_type>;
        if constexpr (return_index.value){
            index_tensor_type unique_index(detail::make_shape_of_type<shape_type>(n_unique));
            std::transform(tmp.begin(),unique_last,unique_index.begin(),[](const auto& e){return e.index();});
            return make_unique_return<index_tensor_type>(std::make_tuple(std::move(res),unique_index),inverse,counts,return_inverse,return_counts);
        }else{
            return make_unique_return<index_tensor_type>(std::move(res),inverse,counts,return_inverse,return_counts);
        }
    }

    //counts should be reserved
    //inverse should be of size last-first
    template<typename It, typename Container, typename ReturnInverse, typename ReturnCounts>
    static auto unique_helper(It first, const It last, Container& inverse, Container& counts, ReturnInverse return_inverse, ReturnCounts return_counts){
        using difference_type = typename Container::difference_type;
        using index_type = typename Container::value_type;
        if (first==last){
            return last;
        }
        auto res = first;
        if constexpr (return_counts.value){
            counts.push_back(1);
        }
        index_type i{0};
        while (++first!=last){
            if (!(*first==*res)){
                if constexpr (return_counts.value){
                    counts.push_back(1);
                }
                if constexpr (return_inverse.value){
                    ++i;
                }
                ++res;
                if (res != first){
                    *res = std::move(*first);
                }
            }else{
                if constexpr (return_counts.value){
                    ++counts.back();
                }
            }
            if constexpr (return_inverse.value){
                inverse[static_cast<const difference_type&>((*first).index())] = i;
            }
        }
        return ++res;
    }

    template<typename TenT, std::size_t I=0, typename R, typename Container, typename B, typename...Bs>
    static auto make_unique_return(R&& r, const Container& inverse, const Container& counts, B b, Bs...bs){
        if constexpr (b.value){
            if constexpr (I==0){    //inverse
                return make_unique_return<TenT,I+1>(detail::make_tuple_or_add(std::forward<R>(r),TenT({inverse.size()},inverse.begin(),inverse.end())),inverse,counts,bs...);
            }else{  //counts
                return detail::make_tuple_or_add(std::forward<R>(r),TenT({counts.size()},counts.begin(),counts.end()));
            }
        }else{
            if constexpr (I==0){
                return make_unique_return<TenT,I+1>(std::forward<R>(r),inverse,counts,bs...);
            }else{
                return r;
            }

        }
    }

};  //end of struct sort_search

//tensor sort_search frontend
//frontend uses compile-time dispatch to select implementation, see module_selector.hpp

#define GTENSOR_TENSOR_SORT_ROUTINE(NAME,F)\
template<typename...Ts, typename DimT=int, typename Comparator=std::less<void>>\
auto NAME(const basic_tensor<Ts...>& t, const DimT& axis=-1, const Comparator& comparator=Comparator{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,axis,comparator);\
}

#define GTENSOR_TENSOR_PARTITION_ROUTINE(NAME,F)\
template<typename...Ts, typename Nth, typename DimT=int, typename Comparator=std::less<void>>\
auto NAME(const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis=-1, const Comparator& comparator=Comparator{}){\
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

//index of min element along given axis, propagating nan, axes is scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(argmin,argmin);

//index of max element along given axis, propagating nan, axes is scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(argmax,argmax);

//index of min element along given axis, ignoring nan, axes is scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(nanargmin,nanargmin);

//index of max element along given axis, ignoring nan, axes is scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(nanargmax,nanargmax);

//count number of values for which static_cast<bool>(e) evaluates to true
//axes can be container or scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(count_nonzero,count_nonzero);

//returns a container of tensors, one for each dimension of t, containing the indices of the non-zero elements in that dimension
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

//returns the sorted unique elements of a tensor.
//There are three optional outputs in addition to the unique elements:
//the indices of the input tensor that give the unique values
//the indices of the unique tensor that reconstruct the input tensor
//the number of times each unique value comes up in the input tensor
template<typename...Ts, typename ReturnIndex=std::false_type, typename ReturnInverse=std::false_type, typename ReturnCounts=std::false_type, typename Axis=detail::no_value>
auto unique(const basic_tensor<Ts...>& t, ReturnIndex return_index=ReturnIndex{}, ReturnInverse return_inverse=ReturnInverse{}, ReturnCounts return_counts=ReturnCounts{}, const Axis& axis=Axis{}){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return sort_search_selector_t<config_type>::unique(t,return_index,return_inverse,return_counts,axis);
}

//find the indices into a sorted tensor t such that, if the corresponding elements in v were inserted before the indices, the order of t would be preserved.
//t must be 1d tensor
//v may be tensor or scalar
//side should be like std::false_type for left side (lower bound)
//side should be like std::true_type for right side (upper bound)
//sorter may be 1d tensor of indexes to sort t or no_value, if no_value t is considered sorted in ascending order
template<typename...Ts, typename V, typename Side=std::false_type, typename Sorter=detail::no_value>
auto searchsorted(const basic_tensor<Ts...>& t, const V& v, Side side=Side{}, const Sorter& sorter=Sorter{}){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return sort_search_selector_t<config_type>::searchsorted(t,v,side,sorter);
}

#undef GTENSOR_TENSOR_ARG_SEARCH_REDUCE_FUNCTION
#undef GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION
#undef GTENSOR_TENSOR_SORT_ROUTINE
#undef GTENSOR_TENSOR_PARTITION_ROUTINE
#undef GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE
}   //end of namespace gtensor
#endif