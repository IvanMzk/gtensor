#ifndef SORT_SEARCH_HPP_
#define SORT_SEARCH_HPP_

#include "indexing.hpp"
#include "reduce.hpp"
#include "reduce_operations.hpp"

namespace gtensor
{

namespace detail{

template<typename DimT, typename Axis>
void check_unique_args(const DimT& dim, const Axis& axis_){
    if constexpr (!std::is_same_v<Axis,no_value>){
        const auto axis = make_axis(dim,axis_);
        if (axis >= dim){
            throw axis_error("axis out of bounds");
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

//Bs sequence of std::bool_constant
template<typename...Bs>
struct bool_seq_to_index_seq{
    template<std::size_t N=0, std::size_t...I>
    struct accumulator_{
        using type = std::index_sequence<I...>;
        template<typename B>
        auto operator+(B){
            if constexpr (B::value){
                return accumulator_<N+1,I...,N>{};
            }else{
                return accumulator_<N+1,I...>{};
            }
        }
    };

    using type = typename decltype((accumulator_{} + ... + Bs{}))::type;
};

}

#define GTENSOR_TENSOR_SORT_ARGSORT_FUNCTION(NAME,F,RES_TYPE,...)\
template<typename Policy, typename...Ts, typename DimT, typename Comparator>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis, const Comparator& comparator){\
    using index_type = typename basic_tensor<Ts...>::index_type;\
    const index_type window_size = 1;\
    const index_type window_step = 1;\
    return slide<RES_TYPE>(policy,t,axis,F{},window_size,window_step,comparator __VA_OPT__(,) __VA_ARGS__);\
}\
template<typename...Ts, typename DimT, typename Comparator>\
static auto NAME(const basic_tensor<Ts...>& t, const DimT& axis, const Comparator& comparator){\
    return NAME(multithreading::exec_pol<1>{},t,axis,comparator);\
}

#define GTENSOR_TENSOR_PARTITION_ARGPARTITION_FUNCTION(NAME,F,RES_TYPE)\
template<typename Policy, typename...Ts, typename Nth, typename DimT, typename Comparator>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis, const Comparator& comparator){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    using index_type = typename basic_tensor<Ts...>::index_type;\
    const index_type window_size = 1;\
    const index_type window_step = 1;\
    return slide<RES_TYPE>(policy,t,axis,F{},window_size,window_step,nth,comparator,config_type{});\
}\
template<typename...Ts, typename Nth, typename DimT, typename Comparator>\
static auto NAME(const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis, const Comparator& comparator){\
    return NAME(multithreading::exec_pol<1>{},t,nth,axis,comparator);\
}

#define GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(NAME,F,ANY_ORDER)\
template<typename Policy, typename...Ts, typename Axes>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    return reduce_range(policy,t,axes,F{},keep_dims,ANY_ORDER);\
}\
template<typename Policy, typename...Ts>\
static auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return NAME(policy,t,detail::no_value{},keep_dims);\
}\
template<typename...Ts, typename Axes>\
static auto NAME(const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    return NAME(multithreading::exec_pol<1>{},t,axes,keep_dims);\
}\
template<typename...Ts>\
static auto NAME(const basic_tensor<Ts...>& t, bool keep_dims = false){\
    return NAME(multithreading::exec_pol<1>{},t,keep_dims);\
}

//tensor sort,search functions implementation
struct sort_search
{
    //sort,search functions along given axis or axes
    //axes may be scalar or container if multiple axes permitted

    //return sorted copy of tensor, axis is scalar
    //Comparator is binary predicate functor, like std::less<void> or std::greater<void>
    GTENSOR_TENSOR_SORT_ARGSORT_FUNCTION(sort,sort_search_reduce_operations::sort,typename basic_tensor<Ts...>::value_type);

    //return indexes that sort tensor along axis, axis is scalar
    //Comparator is binary predicate functor, like std::less<void> or std::greater<void>
    GTENSOR_TENSOR_SORT_ARGSORT_FUNCTION(argsort,sort_search_reduce_operations::argsort,index_type,typename basic_tensor<Ts...>::config_type{});

    //return partially sorted copy of tensor, axis is scalar
    //Nth can be container or scalar
    //Comparator is binary predicate functor, like std::less<void> or std::greater<void>
    GTENSOR_TENSOR_PARTITION_ARGPARTITION_FUNCTION(partition,sort_search_reduce_operations::nth_element_partition,typename basic_tensor<Ts...>::value_type);

    //return indexes that partially sort tensor along axis, axis is scalar
    //Nth can be container or scalar
    //Comparator is binary predicate functor, like std::less<void> or std::greater<void>
    GTENSOR_TENSOR_PARTITION_ARGPARTITION_FUNCTION(argpartition,sort_search_reduce_operations::nth_element_argpartition,index_type);

    //return tensor of indeces of extremum elements along given axis
    GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(argmin,sort_search_reduce_operations::argmin,false);
    GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(argmax,sort_search_reduce_operations::argmax,false);
    //return tensor of indeces of extremum elements along given axis, ignoring nan elements
    GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(nanargmin,sort_search_reduce_operations::nanargmin,false);
    GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(nanargmax,sort_search_reduce_operations::nanargmax,false);

    //return tensor of counts of non-zero elements along given axes
    GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION(count_nonzero,sort_search_reduce_operations::count_nonzero,true);

    //return container of tensors, one for each dimension of t, containing indeces of non-zero elements
    //element is non-zero if static_cast<const bool&>(element) evaluates to true
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
            const auto n = t.size()*static_cast<const index_type&>(t.dim());
            detail::reserve(indexes,n);
            auto nonzero_helper = [&indexes](const auto& shape, auto walker){
                walker_forward_traverser<config_type,decltype(walker)> traverser{shape,walker};
                do{
                    if (static_cast<const bool&>(*traverser)){
                        std::copy(traverser.index().begin(),traverser.index().end(),std::back_inserter(indexes));
                    }
                }while(traverser.template next<config::c_order>());
            };
            if (t.is_trivial()){
                nonzero_helper(t.shape(),t.create_trivial_walker());
            }else{
                nonzero_helper(t.shape(),t.create_walker());
            }
            const auto nonzero_n = static_cast<const index_type&>(indexes.size() / dim);
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

    //return tensor of indeces of non-zero elements, result shape is (N,t.dim()) where N is number of non-zero elements
    template<typename...Ts>
    static auto argwhere(const basic_tensor<Ts...>& t){
        using order = typename basic_tensor<Ts...>::order;
        using config_type = typename basic_tensor<Ts...>::config_type;
        using index_type = typename basic_tensor<Ts...>::index_type;
        using result_config_type = config::extend_config_t<config_type,index_type>;
        using result_tensor_type = tensor<index_type,order,result_config_type>;
        using container_type = typename config_type::template container<index_type>;
        using container_difference_type = typename container_type::difference_type;
        const auto dim = static_cast<const index_type&>(t.dim());
        if (t.empty()){
            return result_tensor_type({index_type{0},dim},0);
        }else{
            container_type indexes{};
            const auto n = t.size()*dim;
            detail::reserve(indexes,n);
            auto argwhere_helper = [&indexes](const auto& shape, auto walker){
                walker_forward_traverser<config_type,decltype(walker)> traverser{shape,walker};
                do{
                    if (static_cast<const bool&>(*traverser)){
                        std::copy(traverser.index().begin(),traverser.index().end(),std::back_inserter(indexes));
                    }
                }while(traverser.template next<config::c_order>());
            };
            if (t.is_trivial()){
                argwhere_helper(t.shape(),t.create_trivial_walker());
            }else{
                argwhere_helper(t.shape(),t.create_walker());
            }
            const auto nonzero_n = static_cast<const index_type&>(indexes.size()) / dim;
            if constexpr (std::is_same_v<order,config::c_order>){
                return result_tensor_type({nonzero_n,dim},indexes.begin(),indexes.end());
            }else{
                result_tensor_type res({nonzero_n,dim},0);
                if (!indexes.empty()){
                    auto dim_ = static_cast<const container_difference_type&>(dim);
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

    template<typename...Ts, typename ReturnIndex=std::false_type, typename ReturnInverse=std::false_type, typename ReturnCounts=std::false_type, typename Axis=detail::no_value>
    static auto unique(const basic_tensor<Ts...>& t, ReturnIndex return_index=ReturnIndex{}, ReturnInverse return_inverse=ReturnInverse{}, ReturnCounts return_counts=ReturnCounts{}, const Axis& axis_=Axis{}){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using value_type = typename tensor_type::value_type;
        using config_type = typename tensor_type::config_type;
        using index_type = typename tensor_type::index_type;
        using shape_type = typename tensor_type::shape_type;
        using res_tensor_type = tensor<value_type,order,config_type>;
        using index_tensor_type = tensor<index_type,order,config_type>;
        using index_container_type = typename config_type::template container<index_type>;
        using index_container_difference_type = typename index_container_type::difference_type;

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
            // if (t.empty()){
            //     shape_type res_shape(shape);
            //     const auto res_axis_size = axis_size==0 ? index_type{0} : index_type{1};
            //     res_shape[axis] = res_axis_size;
            //     if constexpr (return_index.value){
            //         return make_unique_return<index_tensor_type>(
            //             std::make_tuple(res_tensor_type(std::move(res_shape)),index_tensor_type(res_axis_size,0)),
            //             index_container_type(axis_size,0),
            //             index_container_type(res_axis_size,axis_size),
            //             return_inverse,
            //             return_counts
            //         );
            //     }else{
            //         return make_unique_return<index_tensor_type>(
            //             res_tensor_type(std::move(res_shape)),
            //             index_container_type(axis_size,0),
            //             index_container_type(res_axis_size,axis_size),
            //             return_inverse,
            //             return_counts
            //         );
            //     }
            // }

            const auto chunk_size = axis_size==0 ? index_type{0} : t.size()/axis_size;
            auto axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(shape,axis,config::c_order{});
            auto traverser = axes_iterator_maker.create_forward_traverser(t.create_walker(),std::false_type{});
            using walker_type = std::remove_cv_t<std::remove_reference_t<decltype(traverser.walker())>>;
            using axes_iterator_maker_type = decltype(axes_iterator_maker);

            struct range{
                walker_type walker_;
                const axes_iterator_maker_type* axes_iterator_maker_;
                index_type idx_;
                range(const walker_type& walker__, const axes_iterator_maker_type& axes_iterator_maker__, const index_type& idx__):
                    walker_{walker__},
                    axes_iterator_maker_{&axes_iterator_maker__},
                    idx_{idx__}
                {}
                auto index()const{return idx_;}
                auto begin()const{return axes_iterator_maker_->begin_complement(walker_,std::true_type{});}
                auto end()const{return axes_iterator_maker_->end_complement(walker_,std::true_type{});}
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

            typename config_type::template container<range> chunks{};
            index_type i{0};
            if (chunk_size!=0){
                detail::reserve(chunks,axis_size);
                do{
                    chunks.emplace_back(traverser.walker(),axes_iterator_maker,i);
                    ++i;
                }while(traverser.template next<order>());
            }
            std::sort(chunks.begin(),chunks.end());
            index_container_type inverse{};
            index_container_type counts{};
            const auto chunks_size = chunks.size();
            if constexpr(return_inverse.value){
                inverse.assign(static_cast<const index_container_difference_type&>(chunks_size),0);
            }
            if constexpr(return_counts.value){
                detail::reserve(counts,chunks_size);
            }
            const auto unique_last = unique_helper(chunks.begin(),chunks.end(),inverse,counts,return_inverse,return_counts);
            const auto n_unique = static_cast<const index_type&>(unique_last - chunks.begin());
            shape_type res_shape_{shape};
            res_shape_[axis] = n_unique;
            res_tensor_type res(std::move(res_shape_));
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
            return make_unique_result(
                res,
                std::make_tuple(
                    [&chunks,&n_unique,&unique_last](){
                        auto res_index = index_tensor_type(detail::make_shape_of_type<shape_type>(n_unique));
                        std::transform(chunks.begin(),unique_last,res_index.begin(),[](const auto& chunk){return chunk.index();});
                        return res_index;
                    },
                    [&inverse](){return index_tensor_type({inverse.size()},inverse.begin(),inverse.end());},
                    [&counts](){return index_tensor_type({counts.size()},counts.begin(),counts.end());}
                ),
                return_index,return_inverse,return_counts
            );
        }
    }

private:

    template<typename UniqueTensor, typename...Ts, std::size_t...I>
    static auto make_unique_result_helper(const UniqueTensor& unique_tensor, const std::tuple<Ts...>& makers, std::index_sequence<I...>){
        return std::make_tuple(unique_tensor,std::get<I>(makers)()...);
    }

    template<typename UniqueTensor, typename...Ts, typename...Bs>
    static auto make_unique_result(const UniqueTensor& unique_tensor, const std::tuple<Ts...>& makers, Bs...bs){
        using index_seq = typename detail::bool_seq_to_index_seq<Bs...>::type;
        if constexpr (index_seq::size()==0){
            return unique_tensor;
        }else{
            return make_unique_result_helper(unique_tensor,makers,index_seq{});
        }
    }

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
        return make_unique_result(
            res,
            std::make_tuple(
                [&tmp,&n_unique,&unique_last](){
                    auto res_index = index_tensor_type(detail::make_shape_of_type<shape_type>(n_unique));
                    if constexpr (need_index){
                        std::transform(tmp.begin(),unique_last,res_index.begin(),[](const auto& e){return e.index();});
                    }
                    return res_index;
                },
                [&inverse](){return index_tensor_type({inverse.size()},inverse.begin(),inverse.end());},
                [&counts](){return index_tensor_type({counts.size()},counts.begin(),counts.end());}
            ),
            return_index,return_inverse,return_counts
        );
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

};  //end of struct sort_search

//tensor sort_search frontend
//frontend uses compile-time dispatch to select implementation, see module_selector.hpp

#define GTENSOR_TENSOR_SORT_ROUTINE(NAME,F)\
template<typename Policy, typename...Ts, typename DimT=int, typename Comparator=std::less<void>>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, const DimT& axis=-1, const Comparator& comparator=Comparator{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(policy,t,axis,comparator);\
}\
template<typename...Ts, typename DimT=int, typename Comparator=std::less<void>>\
auto NAME(const basic_tensor<Ts...>& t, const DimT& axis=-1, const Comparator& comparator=Comparator{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,axis,comparator);\
}

#define GTENSOR_TENSOR_PARTITION_ROUTINE(NAME,F)\
template<typename Policy, typename...Ts, typename Nth, typename DimT=int, typename Comparator=std::less<void>>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis=-1, const Comparator& comparator=Comparator{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(policy,t,nth,axis,comparator);\
}\
template<typename...Ts, typename Nth, typename DimT=int, typename Comparator=std::less<void>>\
auto NAME(const basic_tensor<Ts...>& t, const Nth& nth, const DimT& axis=-1, const Comparator& comparator=Comparator{}){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(t,nth,axis,comparator);\
}

#define GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(NAME,F)\
template<typename Policy, typename...Ts, typename Axes>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(policy,t,axes,keep_dims);\
}\
template<typename Policy, typename...Ts, typename DimT>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, std::initializer_list<DimT> axes, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(policy,t,axes,keep_dims);\
}\
template<typename Policy, typename...Ts>\
auto NAME(Policy policy, const basic_tensor<Ts...>& t, bool keep_dims = false){\
    using config_type = typename basic_tensor<Ts...>::config_type;\
    return sort_search_selector_t<config_type>::F(policy,t,keep_dims);\
}\
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

//count number of non zero elements, element is non-zero if static_cast<const bool&>(element) evaluates to true
//axes can be container or scalar
GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE(count_nonzero,count_nonzero);

//return container of tensors, one for each dimension of t, containing indeces of non-zero elements
//element is non-zero if static_cast<const bool&>(element) evaluates to true
template<typename...Ts>
auto nonzero(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return sort_search_selector_t<config_type>::nonzero(t);
}

//return tensor of indeces of non-zero elements, result shape is (N,t.dim()) where N is number of non-zero elements
template<typename...Ts>
auto argwhere(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return sort_search_selector_t<config_type>::argwhere(t);
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


#undef GTENSOR_TENSOR_SORT_ARGSORT_FUNCTION
#undef GTENSOR_TENSOR_PARTITION_ARGPARTITION_FUNCTION
#undef GTENSOR_TENSOR_SORT_SEARCH_REDUCE_FUNCTION
#undef GTENSOR_TENSOR_SORT_ROUTINE
#undef GTENSOR_TENSOR_PARTITION_ROUTINE
#undef GTENSOR_TENSOR_SORT_SEARCH_REDUCE_ROUTINE
}   //end of namespace gtensor
#endif