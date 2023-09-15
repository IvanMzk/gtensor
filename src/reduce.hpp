#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include <functional>
#include "module_selector.hpp"
#include "common.hpp"
#include "exception.hpp"
#include "math.hpp"
#include "iterator.hpp"
#include "indexing.hpp"
#include "multithreading.hpp"

namespace gtensor{

namespace detail{

template<typename ShT>
auto check_reduce_args(const ShT& shape, const typename ShT::difference_type& axis){
    using dim_type = typename ShT::difference_type;
    const dim_type dim = detail::make_dim(shape);
    if (axis >= dim){
        throw axis_error("invalid reduce axis: axis is out of bounds");
    }
}
template<typename ShT, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename ShT::difference_type>,int> =0>
auto check_reduce_args(const ShT& shape, const Container& axes){
    using dim_type = typename ShT::difference_type;
    const dim_type dim = detail::make_dim(shape);
    const dim_type axes_number = static_cast<dim_type>(axes.size());
    if (axes_number > dim){
        throw axis_error("invalid reduce axes: too many axes");
    }
    auto it=axes.begin();
    auto last=axes.end();
    while(it!=last){
        const dim_type& axis = static_cast<dim_type>(*it);
        if (axis >= dim || axis < dim_type{0}){
            throw axis_error("invalid reduce axes: axis is out of bounds");
        }
        ++it;
        if (std::find(it, last, axis) != last){
            throw axis_error("invalid reduce axes: duplicates in axes");
        }
    }
}

template<typename ShT>
auto check_transform_args(const ShT& shape, const typename ShT::difference_type& axis){
    using dim_type = typename ShT::difference_type;
    const dim_type dim = detail::make_dim(shape);
    if (axis >= dim){
        throw axis_error("invalid transform axis: axis is out of bounds");
    }
}

template<typename ShT>
auto make_reduce_shape(const ShT& shape, const typename ShT::difference_type& axis, bool keep_dims){
    using shape_type = ShT;
    using dim_type = typename ShT::difference_type;
    using index_type = typename ShT::value_type;
    if (keep_dims){
        shape_type res(shape);
        res[axis] = index_type{1};
        return res;
    }else{
        dim_type dim = detail::make_dim(shape);
        shape_type res(--dim);
        auto shape_stop = shape.begin()+axis;
        std::copy(shape.begin(), shape_stop, res.begin());
        std::copy(++shape_stop, shape.end(), res.begin()+axis);
        return res;
    }
}
template<typename ShT, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename ShT::difference_type>,int> =0>
auto make_reduce_shape(const ShT& shape, const Container& axes, bool keep_dims){
    using shape_type = ShT;
    using dim_type = typename ShT::difference_type;
    using index_type = typename ShT::value_type;
    using axes_value_type = typename Container::value_type;
    const dim_type dim = detail::make_dim(shape);
    const dim_type axes_number = static_cast<dim_type>(axes.size());
    if (axes_number == dim_type{0}){
        return shape;
    }else{
        if (keep_dims){
            shape_type res(shape);
            std::for_each(axes.begin(),axes.end(),[&res](const auto& a)mutable{res[a]=index_type{1};});
            return res;
        }else{
            shape_type res{};
            res.reserve(dim - axes_number);
            auto axes_first = axes.begin();
            auto axes_last = axes.end();
            for(dim_type d{0}; d!=dim; ++d){
                if (std::find(axes_first, axes_last, static_cast<axes_value_type>(d)) == axes_last){
                    res.push_back(shape[d]);
                }
            }
            return res;
        }
    }



}
template<typename ShT>
auto make_reduce_shape(const ShT& shape, bool keep_dims){
    using shape_type = ShT;
    const auto dim = detail::make_dim(shape);
    if (keep_dims){
        return shape_type(dim,1);
    }else{
        return shape_type{};
    }
}


template<typename IdxT>
auto check_slide_args(const IdxT& axis_size, const IdxT& window_size, const IdxT& window_step){
    using index_type = IdxT;
    if (axis_size!=0){
        if (window_size > axis_size || window_size <= index_type{0}){
            throw value_error("invalid sliding window size");
        }
        if (window_step < index_type{1}){
            throw value_error("invalid sliding window step");
        }
    }
}
template<typename IdxT, typename ShT, typename DimT>
auto check_slide_args(const IdxT& size, const ShT& shape, const DimT& axis, const IdxT& window_size, const IdxT& window_step){
    using dim_type = DimT;
    using index_type = IdxT;
    const dim_type dim = detail::make_dim(shape);
    if (axis >= dim){
        throw axis_error("invalid slide axis");
    }
    if (size!=0){
        index_type axis_size = shape[axis];
        check_slide_args(axis_size,window_size,window_step);
    }
}
template<typename IdxT>
auto make_slide_size(const IdxT& size, const IdxT& window_size, const IdxT& window_step){
    return (size - window_size)/window_step + IdxT{1};
}
template<typename IdxT,typename ShT, typename DimT>
auto make_slide_shape(const IdxT& size, const ShT& shape, const DimT& axis, const IdxT& window_size, const IdxT& window_step){
    using index_type = IdxT;
    using shape_type = ShT;
    shape_type res(shape);
    if (size!=index_type{0}){
        const index_type axis_size = shape[axis];
        const index_type result_axis_size = make_slide_size(axis_size, window_size, window_step);
        res[axis] = result_axis_size;
    }
    return res;
}

template<typename Axes>
void sort_axes(Axes& axes){
    if constexpr (detail::is_container_v<Axes>){
        if (!std::is_sorted(axes.begin(),axes.end())){
            std::sort(axes.begin(),axes.end());
        }
    }
}

template<typename Axes, typename Order>
auto make_leading_axes(const Axes& sorted_axes, Order){
    if constexpr (detail::is_container_v<Axes>){
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            auto axes_it = sorted_axes.end();
            const auto a_start = *--axes_it;
            auto a_stop = a_start;
            for(const auto axes_first=sorted_axes.begin(); axes_it!=axes_first;){
                auto a_ = *--axes_it;
                if (a_+1 == a_stop){
                    a_stop = a_;
                }else{
                    break;
                }
            }
            return std::make_pair(a_start,a_stop);
        }else{
            auto axes_it = sorted_axes.begin();
            const auto a_start = *axes_it;
            auto a_stop = a_start;
            for(const auto axes_last=std::prev(sorted_axes.end()); axes_it!=axes_last;){
                auto a_ = *++axes_it;
                if (a_ == a_stop+1){
                    a_stop = a_;
                }else{
                    break;
                }
            }
            return std::make_pair(a_start,a_stop);
        }
    }else{
        return std::make_pair(sorted_axes,sorted_axes);
    }
}

template<typename ShT, typename Pair>
auto make_inner_size(const ShT& strides, const Pair& leading_axes){
    return strides[leading_axes.first];
}

template<typename ShT, typename Pair>
auto make_outer_size(const ShT& shape, const Pair& leading_axes){
    const auto a_minmax = std::minmax(leading_axes.first,leading_axes.second);
    auto res = shape[a_minmax.first];
    for (auto a=a_minmax.first+1, a_last=a_minmax.second+1; a!=a_last; ++a){
        res*=shape[a];
    }
    return res;
}

template<typename ShT, typename Pair, typename Order>
auto make_traverse_index_shape(const ShT& shape, const Pair& leading_axes, Order){
    if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
        return ShT{shape.begin(), shape.begin()+leading_axes.second};
    }else{
        return ShT{shape.begin()+leading_axes.second+1,shape.end()};
    }
}

template<typename Config, typename DimT, typename Axes>
auto make_reduce_axes_map(const DimT& dim, const Axes& sorted_axes, bool keep_dims){
    using dim_type = typename Config::dim_type;
    using res_type = typename Config::template shape<dim_type>;
    if (keep_dims){
        res_type res(dim,dim_type{0});
        std::iota(res.begin(),res.end(),dim_type{0});
        return res;
    }
    if constexpr (detail::is_container_v<Axes>){
        const auto axes_size = static_cast<const dim_type&>(sorted_axes.size());
        res_type res{};
        res.reserve(dim-axes_size);
        for (dim_type a=0, a_last=dim; a!=a_last; ++a){
            if (!std::binary_search(sorted_axes.begin(),sorted_axes.end(),a)){
                res.push_back(a);
            }
        }
        return res;
    }else{
        res_type res(dim-1,dim_type{0});
        std::iota(res.begin()+sorted_axes,res.end(),sorted_axes+1);
        std::iota(res.begin(),res.begin()+sorted_axes,0);
        return res;
    }
}

template<typename ShT, typename Pair, typename AxesMap, typename Order>
auto make_traverse_index_strides(const ShT& traverse_shape, const ShT& res_strides, const Pair& leading_axes, const AxesMap& map, Order){
    using dim_type = typename ShT::difference_type;
    const auto traverse_index_dim = detail::make_dim(traverse_shape);
    ShT res(traverse_index_dim,0);
    if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
        for (dim_type a=0, a_last=res_strides.size(); a!=a_last; ++a){
            auto a_ = map[a];
            if (a_ < leading_axes.second){
                res[a_] = res_strides[a];
            }
        }
    }else{
        for (dim_type a=0, a_last=res_strides.size(); a!=a_last; ++a){
            auto a_ = map[a];
            if (a_ > leading_axes.second){
                res[a_-leading_axes.second-1] = res_strides[a];
            }
        }
    }
    return res;
}

template<typename DstIt, typename It, typename F>
ALWAYS_INLINE void transform(DstIt first1, DstIt last1, It& first2, F f){
    for (;first1!=last1; ++first1,++first2){
        *first1 = f(*first1,*first2);
    }
}

template<typename It, typename IdxT, typename Initial, typename F>
ALWAYS_INLINE auto accumulate_n(It& first, IdxT n, Initial initial, F f){
    for (;n!=0; --n,++first){
        initial = f(initial,*first);
    }
    return initial;
}

template<typename It, typename IdxT, typename F>
ALWAYS_INLINE auto accumulate_n(It& first, IdxT n, F f){
    auto initial = *first;
    for (--n,++first; n!=0; --n,++first){
        initial = f(initial,*first);
    }
    return initial;
}

}   //end of namespace detail

//reduce policy
template<std::size_t N> struct reduce_auto : std::integral_constant<std::size_t,N>{};
template<std::size_t N> struct reduce_bin : std::integral_constant<std::size_t,N>{};
template<std::size_t N> struct reduce_rng : std::integral_constant<std::size_t,N>{};

template<typename...> struct reduce_policy_traits;
template<template<std::size_t> typename P, std::size_t V>
struct reduce_policy_traits<P<V>>{
    using policy_type = P<V>;
    using exec_policy = multithreading::exec_pol<V>;
    using is_reduce_auto = std::is_same<policy_type,reduce_auto<V>>;
    using is_reduce_bin = std::is_same<policy_type,reduce_bin<V>>;
    using is_reduce_rng = std::is_same<policy_type,reduce_rng<V>>;
};

class reducer
{

    //reduce like over flatten helper, binary functor
    //return scalar result
    template<typename Policy, typename BinaryF, typename...Ts, typename Initial>
    static auto reduce_binary_flatten_helper(Policy policy, const basic_tensor<Ts...>& parent, BinaryF reduce_f, const Initial& initial){
        using order = typename basic_tensor<Ts...>::order;
        static constexpr bool has_initial = !std::is_same_v<Initial,detail::no_value>;
        if (!has_initial && parent.empty()){
            throw value_error("cant reduce zero size dimension without initial value");
        }
        auto reduce_binary_flatten_helper_ = [&policy,&reduce_f,&initial](auto first, auto last){
            (void)initial;
            if constexpr (has_initial){
                return multithreading::reduce(policy,first,last,initial,reduce_f);
            }else{
                const auto& initial_ = *first;
                return multithreading::reduce(policy,++first,last,initial_,reduce_f);
            }
        };
        auto a = parent.traverse_order_adapter(order{});
        if (parent.is_trivial()){
            return reduce_binary_flatten_helper_(a.begin_trivial(),a.end_trivial());
        }else{
            return reduce_binary_flatten_helper_(a.begin(),a.end());
        }
    }

    //reduce like over flatten
    template<typename Policy, typename BinaryF, typename...Ts, typename Initial>
    static auto reduce_binary_flatten_(Policy policy, const basic_tensor<Ts...>& parent, BinaryF reduce_f, bool keep_dims, const Initial& initial){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using res_value_type = decltype(reduce_binary_flatten_helper(policy,parent,reduce_f,initial));
        using res_config_type = config::extend_config_t<config_type,res_value_type>;
        using res_type = tensor<res_value_type,order,res_config_type>;
        return res_type(
            detail::make_reduce_shape(parent.shape(),keep_dims),
            reduce_binary_flatten_helper(policy,parent,reduce_f,initial)
        );
    }

    //axes can be container or scalar
    //F is binary functor, takes elements, return reduce result, like std::plus
    //initial must be such that expression reduce_f(initial,element) be valid or no_value
    //traverse input countigous, i.e. traverse order dependes on layout
    template<typename Policy, typename BinaryF, typename Axes, typename...Ts, typename Initial>
    static auto reduce_binary_(Policy policy, const basic_tensor<Ts...>& parent, const Axes& axes_, BinaryF reduce_f, bool keep_dims, const Initial& initial){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using value_type = typename parent_type::value_type;
        using index_type = typename config_type::index_type;
        static constexpr bool has_initial = !std::is_same_v<Initial,detail::no_value>;
        using initial_type = std::conditional_t<has_initial, Initial, value_type>;
        using result_type =  decltype(reduce_f(std::declval<initial_type>(),std::declval<value_type>()));
        using res_value_type = std::remove_cv_t<std::remove_reference_t<result_type>>;
        using res_config_type = config::extend_config_t<config_type,res_value_type>;

        const auto pdim = parent.dim();
        const auto& pshape = parent.shape();
        auto axes = detail::make_axes<config_type>(pdim,axes_);
        detail::check_reduce_args(pshape, axes);
        auto res = tensor<res_value_type,order,res_config_type>(detail::make_reduce_shape(pshape, axes, keep_dims));
        if (!res.empty()){
            auto a_parent = parent.traverse_order_adapter(order{});
            auto a_res = res.traverse_order_adapter(order{});

            //like tensor-scalar result
            if (res.size() == index_type{1}){
                *a_res.begin() = reduce_binary_flatten_helper(policy,parent,reduce_f,initial);
                return res;
            }

            auto reduce_binary_corner_case_helper = [&parent,&reduce_f,&initial,&axes,&a_res](auto parent_first, auto parent_last){
                (void)reduce_f;
                (void)initial;
                (void)axes;
                (void)a_res;
                //empty axes container
                if constexpr (detail::is_container_v<Axes>){
                    if (axes.empty()){
                        if constexpr (has_initial){
                            std::transform(a_res.begin(),a_res.end(),parent_first,a_res.begin(),[&reduce_f,&initial](auto&&, auto&& r){return reduce_f(initial,r);});
                        }else{
                            //assuming reduction of scalar without initial is identity - nothing to reduce
                            std::transform(a_res.begin(),a_res.end(),parent_first,a_res.begin(),[](auto&&, auto&& r){return static_cast<const res_value_type&>(r);});
                        }
                        return true;
                    }
                }
                //reduce zero size
                if (parent.empty()){
                    if constexpr (has_initial){
                        std::fill(a_res.begin(),a_res.end(),initial);
                        return true;
                    }else{
                        throw value_error("cant reduce zero size dimension without initial value");
                    }
                }
                return false;
            };

            bool is_corner_case{false};
            if (parent.is_trivial()){
                is_corner_case = reduce_binary_corner_case_helper(a_parent.begin_trivial(),a_parent.end_trivial());
            }else{
                is_corner_case = reduce_binary_corner_case_helper(a_parent.begin(),a_parent.end());
            }
            if (is_corner_case){
                return res;
            }

            detail::sort_axes(axes);
            const auto leading_axes = detail::make_leading_axes(axes,order{});
            const auto inner_size = detail::make_inner_size(parent.strides(),leading_axes);
            const auto outer_size = detail::make_outer_size(pshape,leading_axes);
            const auto traverse_index_shape = detail::make_traverse_index_shape(pshape,leading_axes,order{});
            const auto axes_map = detail::make_reduce_axes_map<config_type>(pdim,axes,keep_dims);
            const auto traverse_index_strides = detail::make_traverse_index_strides(traverse_index_shape,res.descriptor().adapted_strides(),leading_axes,axes_map,order{});
            const auto traverse_index_reset_strides = detail::make_traverse_index_strides(traverse_index_shape,res.descriptor().reset_strides(),leading_axes,axes_map,order{});
            using walker_type = cursor_walker<config_type,index_type,order>;
            using traverser_type = walker_forward_traverser<config_type,walker_type>;
            traverser_type traverser{traverse_index_shape,walker_type{traverse_index_strides,traverse_index_reset_strides,0}};

            auto reduce_binary_helper_ = [&traverser,&reduce_f,inner_size,outer_size,initial](auto it, auto res_it){
                (void)initial;
                const auto res_first = res_it;
                bool init = true;
                auto offset = *traverser;
                if (inner_size == 1){
                    do{
                        auto& e = *res_it;
                        if (init){
                            if constexpr (has_initial){
                                e = detail::accumulate_n(it,outer_size,initial,reduce_f);
                            }else{  //no initial
                                e = detail::accumulate_n(it,outer_size,reduce_f);
                            }
                        }else{  //e initialized
                            e = detail::accumulate_n(it,outer_size,e,reduce_f);
                        }
                        if (traverser.template next<order>()){
                            const auto new_offset = *traverser;
                            if (new_offset > offset){   //reach uninitialized
                                init=true;
                                offset = new_offset;
                            }else{
                                init=false;
                            }
                            res_it=res_first + new_offset;
                        }else{
                            break;
                        }
                    }while(true);
                }else{
                    do{
                        if (init){
                            if constexpr (has_initial){
                                detail::transform(res_it,res_it+inner_size,it,[&reduce_f,&initial](auto&&, auto&& r){return reduce_f(initial,r);});
                            }else{  //no initial
                                detail::transform(res_it,res_it+inner_size,it,[](auto&&, auto&& r){return static_cast<const res_value_type&>(r);});
                            }
                        }
                        const auto i_stop=init?1:0;
                        for (auto i=outer_size; i!=i_stop; --i){
                            detail::transform(res_it,res_it+inner_size,it,reduce_f);
                        }
                        if (traverser.template next<order>()){
                            const auto new_offset = *traverser;
                            if (new_offset > offset){   //reach uninitialized
                                init=true;
                                offset = new_offset;
                            }else{
                                init=false;
                            }
                            res_it=res_first + new_offset;
                        }else{
                            break;
                        }
                    }while(true);
                }
            };
            if (parent.is_trivial()){
                reduce_binary_helper_(a_parent.begin_trivial(),a_res.begin());
            }else{
                reduce_binary_helper_(a_parent.begin(),a_res.begin());
            }
        }
        return res;
    }

    //reduce like over flatten helper, iterators range functor
    //return scalar result
    template<typename RangeF, typename...Ts, typename...Args>
    static auto reduce_range_flatten_helper(const basic_tensor<Ts...>& t, RangeF reduce_f, bool any_order, const Args&...args){
        using order = typename basic_tensor<Ts...>::order;
        auto reduce_flatten_helper_ = [&t,&reduce_f,&any_order](auto begin_maker, auto end_maker, const auto&...args_){
            if (t.dim()==1 || any_order){   //1d or any_order, can use native order
                auto a = t.traverse_order_adapter(order{});
                return  reduce_f(begin_maker(a), end_maker(a), args_...);
            }else{  //traverse like over flatten
                auto a = t.traverse_order_adapter(config::c_order{});
                return  reduce_f(begin_maker(a), end_maker(a), args_...);
            }
        };
        if (t.is_trivial()){
            return reduce_flatten_helper_([](auto& a){return a.begin_trivial();},[](auto& a){return a.end_trivial();},args...);
        }else{
            return reduce_flatten_helper_([](auto& a){return a.begin();},[](auto& a){return a.end();},args...);
        }
    }

    //reduce like over flatten
    template<typename RangeF, typename...Ts, typename...Args>
    static auto reduce_range_flatten_(const basic_tensor<Ts...>& t, RangeF reduce_f, bool keep_dims, bool any_order, const Args&...args){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using config_type = typename tensor_type::config_type;
        using res_value_type = decltype(reduce_range_flatten_helper(t,reduce_f,any_order,args...));
        using res_config_type = config::extend_config_t<config_type,res_value_type>;
        using res_type = tensor<res_value_type,order,res_config_type>;
        return res_type(
            detail::make_reduce_shape(t.shape(),keep_dims),
            reduce_range_flatten_helper(t,reduce_f,any_order,args...)
        );
    }

    //F takes iterators range to be reduces and additional args
    template<typename Policy, typename RangeF, typename Axes, typename...Ts, typename...Args>
    static auto reduce_range_helper(Policy, const basic_tensor<Ts...>& parent, const Axes& axes_, RangeF reduce_f, bool keep_dims, bool any_order, const Args&...args){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using traverse_order = config::c_order; //order to traverse along axes
        using index_type = typename config_type::index_type;
        using shape_type = typename config_type::shape_type;
        using axes_iterator_maker_type = decltype(detail::make_axes_iterator_maker<config_type>(std::declval<shape_type>(),axes_,traverse_order{}));
        using axes_iterator_type = decltype(std::declval<axes_iterator_maker_type>().begin(parent.create_walker(),std::false_type{}));
        using result_type = decltype(reduce_f(std::declval<axes_iterator_type>(),std::declval<axes_iterator_type>(),std::declval<const Args&>()...));
        using res_value_type = std::remove_cv_t<std::remove_reference_t<result_type>>;
        using res_config_type = config::extend_config_t<config_type,res_value_type>;

        const auto pdim = parent.dim();
        const auto& pshape = parent.shape();
        auto axes = detail::make_axes<config_type>(pdim,axes_);
        detail::check_reduce_args(pshape, axes);
        auto res = tensor<res_value_type,order,res_config_type>{detail::make_reduce_shape(pshape, axes, keep_dims)};
        if (!res.empty()){
            auto a = res.traverse_order_adapter(traverse_order{});
            //zero size axis is reduced
            if (parent.empty()){
                auto a = parent.traverse_order_adapter(order{});
                const auto e = reduce_f(a.begin(), a.end(), args...);
                std::fill(res.begin(), res.end(), e);
                return res;
            }
            const auto res_size = res.size();
            //like tensor-scalar result
            if (res_size == index_type{1}){
                *res.begin() = reduce_range_flatten_helper(parent,reduce_f,any_order,args...);
                return res;
            }
            auto axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(pshape,axes,traverse_order{});
            auto reduce_helper = [&reduce_f,&res_size,&axes_iterator_maker](auto walker, auto res_it, const auto&...args_){
                auto body = [&axes_iterator_maker](auto f, auto res_first, auto res_last, auto traverser, const auto&...args__){
                    for (;res_first!=res_last; ++res_first,traverser.next()){
                        *res_first = f(
                            axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{}),
                            axes_iterator_maker.end_complement(traverser.walker(),std::false_type{}),
                            args__...
                        );
                    }
                };
                auto traverser = axes_iterator_maker.create_random_access_traverser(walker,std::true_type{});
                if constexpr (multithreading::exec_policy_traits<Policy>::is_seq::value){
                    body(reduce_f,res_it,res_it+res_size,traverser,args_...);
                }else{  //parallelize
                    constexpr std::size_t max_par_tasks = multithreading::exec_policy_traits<Policy>::par_tasks::value;
                    constexpr std::size_t min_tasks_per_par_task = 1;
                    multithreading::par_task_size<index_type> par_sizes{res_size,max_par_tasks,min_tasks_per_par_task};

                    using future_type = decltype(
                        std::declval<decltype(multithreading::get_pool())>().push(
                            body,
                            reduce_f,
                            res_it,
                            res_it,
                            traverser,
                            std::cref(args_)...
                        )
                    );
                    std::array<future_type,max_par_tasks> futures{};
                    index_type pos{0};
                    for (std::size_t i{0}; i!=par_sizes.size(); ++i){
                        const auto par_task_size = par_sizes[i];
                        futures[i] = multithreading::get_pool().push(
                            body,
                            reduce_f,
                            res_it,
                            res_it+par_task_size,
                            traverser,
                            std::cref(args_)...
                        );
                        res_it+=par_task_size;
                        traverser.to(pos+=par_task_size);
                    }
                }
            };

            // auto reduce_helper = [&parent,&reduce_f,&any_order,&res,&pdim,&pshape,&axes](auto walker_maker, auto begin_maker, auto end_maker,auto&&...args_){
            //     if (res.size() == index_type{1}){
            //         if (pdim == dim_type{1} || any_order){   //1d or any_order, can use native order
            //             auto a = parent.traverse_order_adapter(order{});
            //             *res.begin() = reduce_f(begin_maker(a), end_maker(a), std::forward<decltype(args_)>(args_)...);
            //         }else{  //traverse like over flatten
            //             auto a = parent.traverse_order_adapter(traverse_order{});
            //             *res.begin() = reduce_f(begin_maker(a), end_maker(a), std::forward<decltype(args_)>(args_)...);
            //         }
            //     }else{
            //         //auto axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(pshape,axes,traverse_order{});
            //         //auto traverser = axes_iterator_maker.create_forward_traverser(walker_maker(parent),std::true_type{});
            //         auto axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(pshape,axes,traverse_order{});
            //         auto traverser = axes_iterator_maker.create_random_access_traverser(walker_maker(parent),std::true_type{});
            //         auto a = res.traverse_order_adapter(traverse_order{});
            //         auto res_it = a.begin();
            //         do{
            //             *res_it = reduce_f(
            //                 axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{}),
            //                 axes_iterator_maker.end_complement(traverser.walker(),std::false_type{}),
            //                 std::forward<decltype(args_)>(args_)...
            //             );
            //             ++res_it;
            //         }while(traverser.next());
            //         //}while(traverser.template next<order>());
            //     }
            // };
            // if (parent.is_trivial()){
            //     reduce_helper([](auto& p){return p.create_trivial_walker();},[](auto& a){return a.begin_trivial();},[](auto& a){return a.end_trivial();},args...);
            // }else{
            //     reduce_helper([](auto& p){return p.create_walker();},[](auto& a){return a.begin();},[](auto& a){return a.end();},args...);
            // }
            if (parent.is_trivial()){
                reduce_helper(parent.create_trivial_walker(),a.begin(),args...);
            }else{
                reduce_helper(parent.create_walker(),a.begin(),args...);
            }
        }
        return res;
    }

    template<typename Policy, typename F, typename Axes, typename...Ts, typename...Args>
    static auto reduce_range_(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, bool any_order, const Args&...args){
        using dim_type = typename basic_tensor<Ts...>::dim_type;
        if constexpr (detail::is_container_of_type_v<Axes,dim_type>){
            if (axes.size() == 1){
                return reduce_range_helper(policy,t,*axes.begin(),f,keep_dims,any_order,args...);
            }
        }
        return reduce_range_helper(policy,t,axes,f,keep_dims,any_order,args...);
    }

    //do reduce according to specified reduce policy
    template<typename Policy, typename BinaryF, typename RangeF, typename Axes, typename Initial, typename...Ts, typename...Args>
    static auto reduce_(Policy, const basic_tensor<Ts...>& parent, const Axes& axes, BinaryF binary_f, RangeF range_f, bool keep_dims, bool any_order, const Initial initial, const Args&...args){
        detail::unused_args{binary_f,range_f,any_order,initial,args...};
        using exec_policy = typename reduce_policy_traits<Policy>::exec_policy;
        if constexpr (reduce_policy_traits<Policy>::is_reduce_bin::value){
            static_assert(!std::is_same_v<BinaryF,detail::no_value>,"invalid reduce functor");
            return reduce_binary_(exec_policy{},parent,axes,binary_f,keep_dims,initial);
        }else if constexpr (reduce_policy_traits<Policy>::is_reduce_rng::value){
            static_assert(!std::is_same_v<RangeF,detail::no_value>,"invalid reduce functor");
            return reduce_range_(exec_policy{},parent,axes,range_f,keep_dims,any_order,args...);
        }else if constexpr (reduce_policy_traits<Policy>::is_reduce_auto::value){
            if constexpr (!std::is_same_v<BinaryF,detail::no_value>){
                return reduce_binary_(exec_policy{},parent,axes,binary_f,keep_dims,initial);
            }else if constexpr (!std::is_same_v<RangeF,detail::no_value>){
                return reduce_range_(exec_policy{},parent,axes,range_f,keep_dims,any_order,args...);
            }else{
                static_assert(detail::always_false<Policy>,"invalid reduce functor");
            }
        }else{
            static_assert(detail::always_false<Policy>,"invalid reduce policy");
        }
    }
    template<typename Policy, typename BinaryF, typename RangeF, typename Initial, typename...Ts, typename...Args>
    static auto reduce_flatten_(Policy, const basic_tensor<Ts...>& parent, BinaryF binary_f, RangeF range_f, bool keep_dims, bool any_order, const Initial initial, const Args&...args){
        detail::unused_args{binary_f,range_f,any_order,initial,args...};
        using exec_policy = typename reduce_policy_traits<Policy>::exec_policy;
        if constexpr (reduce_policy_traits<Policy>::is_reduce_bin::value){
            static_assert(!std::is_same_v<BinaryF,detail::no_value>,"invalid reduce functor");
            return reduce_binary_flatten_(exec_policy{},parent,binary_f,keep_dims,initial);
        }else if constexpr (reduce_policy_traits<Policy>::is_reduce_rng::value){
            static_assert(!std::is_same_v<RangeF,detail::no_value>,"invalid reduce functor");
            return reduce_range_flatten_(exec_policy{},parent,range_f,keep_dims,any_order,args...);
        }else if constexpr (reduce_policy_traits<Policy>::is_reduce_auto::value){
            if constexpr (!std::is_same_v<BinaryF,detail::no_value>){
                return reduce_binary_flatten_(exec_policy{},parent,binary_f,keep_dims,initial);
            }else if constexpr (!std::is_same_v<RangeF,detail::no_value>){
                return reduce_range_flatten_(exec_policy{},parent,range_f,keep_dims,any_order,args...);
            }else{
                static_assert(detail::always_false<Policy>,"invalid reduce functor");
            }
        }else{
            static_assert(detail::always_false<Policy>,"invalid reduce policy");
        }
    }

    template<typename ResultT, typename...Ts, typename DimT, typename F, typename IdxT, typename...Args>
    static auto slide_(const basic_tensor<Ts...>& parent, const DimT& axis_, F slide_f, const IdxT& window_size_, const IdxT& window_step_, const Args&...args)
    {
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using index_type = typename config_type::index_type;
        using dim_type = typename config_type::dim_type;
        using res_value_type = ResultT;
        using res_config_type = gtensor::config::extend_config_t<config_type,res_value_type>;

        const auto& pshape = parent.shape();
        const auto psize = parent.size();
        const dim_type axis = detail::make_axis(pshape,axis_);
        const index_type window_size = static_cast<index_type>(window_size_);
        const index_type window_step = static_cast<index_type>(window_step_);
        detail::check_slide_args(psize, pshape, axis, window_size, window_step);
        auto res = tensor<res_value_type,order,res_config_type>{detail::make_slide_shape(psize, pshape, axis, window_size, window_step)};
        if (!res.empty()){

            auto slide_helper = [&parent,&slide_f,&pshape,&axis,&res](auto walker_maker, auto begin_maker, auto end_maker, const auto&...args_){
                const auto pdim = parent.dim();
                if (pdim == dim_type{1}){
                    auto parent_a = parent.traverse_order_adapter(order{});
                    auto res_a = res.traverse_order_adapter(order{});
                    slide_f(begin_maker(parent_a), end_maker(parent_a), res_a.begin(), res_a.end(), args_...);
                }else{
                    const auto& res_shape = res.shape();
                    auto parent_axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(pshape,axis,order{});
                    auto res_axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(res_shape,axis,order{});
                    auto parent_traverser = parent_axes_iterator_maker.create_forward_traverser(walker_maker(parent),std::true_type{});
                    auto res_traverser = res_axes_iterator_maker.create_forward_traverser(res.create_walker(),std::true_type{});
                    do{
                        //0first,1last,2dst_first,3dst_last,4args
                        slide_f(
                            parent_axes_iterator_maker.begin_complement(parent_traverser.walker(),std::false_type{}),
                            parent_axes_iterator_maker.end_complement(parent_traverser.walker(),std::false_type{}),
                            res_axes_iterator_maker.begin_complement(res_traverser.walker(),std::false_type{}),
                            res_axes_iterator_maker.end_complement(res_traverser.walker(),std::false_type{}),
                            args_...
                        );
                        res_traverser.template next<order>();
                    }while(parent_traverser.template next<order>());
                }
            };
            if (parent.is_trivial()){
                slide_helper([](auto& p){return p.create_trivial_walker();},[](auto& a){return a.begin_trivial();},[](auto& a){return a.end_trivial();},args...);
            }else{
                slide_helper([](auto& p){return p.create_walker();},[](auto& a){return a.begin();},[](auto& a){return a.end();},args...);
            }
        }
        return res;
    }

    template<typename ResultT, typename...Ts, typename F, typename IdxT, typename...Args>
    static auto slide_flatten_(const basic_tensor<Ts...>& parent, F slide_f, const IdxT& window_size_, const IdxT& window_step_, const Args&...args)
    {
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using dim_type = typename config_type::dim_type;
        using index_type = typename config_type::index_type;
        using shape_type = typename config_type::shape_type;
        using res_value_type = ResultT;
        using res_config_type = gtensor::config::extend_config_t<config_type,res_value_type>;

        const auto psize = parent.size();
        const index_type window_size = static_cast<index_type>(window_size_);
        const index_type window_step = static_cast<index_type>(window_step_);
        detail::check_slide_args(psize, window_size, window_step);
        auto res = tensor<res_value_type,order,res_config_type>{shape_type{detail::make_slide_size(psize, window_size, window_step)}};
        if (!res.empty()){
            auto slide_flatten_helper = [&parent,&slide_f,&res](auto begin_maker, auto end_maker, const auto&...args_){
                const auto pdim = parent.dim();
                auto res_a = res.traverse_order_adapter(order{});
                if (pdim == dim_type{1}){
                    auto parent_a = parent.traverse_order_adapter(order{});
                    slide_f(begin_maker(parent_a), end_maker(parent_a), res_a.begin(), res_a.end(), args_...);
                }else{
                    auto parent_a = parent.traverse_order_adapter(config::c_order{});
                    slide_f(begin_maker(parent_a), end_maker(parent_a), res_a.begin(), res_a.end(), args_...);
                }
            };

            if (parent.is_trivial()){
                slide_flatten_helper([](auto& a){return a.begin_trivial();},[](auto& a){return a.end_trivial();},args...);
            }else{
                slide_flatten_helper([](auto& a){return a.begin();},[](auto& a){return a.end();},args...);
            }
        }
        return res;
    }

    template<typename F, typename DimT, typename...Ts, typename...Args>
    static void transform_(basic_tensor<Ts...>& parent, const DimT& axis_, F transform_f, const Args&...args){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using dim_type = typename config_type::dim_type;

        const auto& pshape = parent.shape();
        const dim_type axis = detail::make_axis(pshape,axis_);
        detail::check_transform_args(pshape, axis);
        const auto pdim = parent.dim();
        if (pdim == dim_type{1}){
            auto a = parent.traverse_order_adapter(order{});
            transform_f(a.begin(), a.end(), args...);
        }else{
            auto axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(pshape,axis,order{});
            auto traverser = axes_iterator_maker.create_forward_traverser(parent.create_walker(),std::true_type{});
            do{
                //0first,1last,2args
                transform_f(
                    axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{}),
                    axes_iterator_maker.end_complement(traverser.walker(),std::false_type{}),
                    args...
                );
            }while(traverser.template next<order>());
        }
    }

public:
    //interface
    template<typename Policy, typename F, typename...Ts, typename Initial>
    static auto reduce_binary_flatten(Policy policy, const basic_tensor<Ts...>& t, F f, bool keep_dims, const Initial& initial){
        return reduce_binary_flatten_(policy,t,f,keep_dims,initial);
    }

    template<typename Policy, typename F, typename Axes, typename...Ts, typename Initial>
    static auto reduce_binary(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, const Initial& initial){
        return reduce_binary_(policy,t,axes,f,keep_dims,initial);
    }

    template<typename F, typename...Ts, typename...Args>
    static auto reduce_range_flatten(const basic_tensor<Ts...>& t, F f, bool keep_dims, bool any_order, const Args&...args){
        return reduce_range_flatten_(t,f,keep_dims,any_order,args...);
    }

    template<typename Policy, typename F, typename Axes, typename...Ts, typename...Args>
    static auto reduce_range(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, bool any_order, const Args&...args){
        return reduce_range_(policy,t,axes,f,keep_dims,any_order,args...);
    }

    template<typename Policy, typename BinaryF, typename RangeF, typename Axes, typename Initial, typename...Ts, typename...Args>
    static auto reduce(Policy policy, const basic_tensor<Ts...>& parent, const Axes& axes, BinaryF binary_f, RangeF range_f, bool keep_dims, bool any_order, const Initial initial, const Args&...args){
        return reduce_(policy,parent,axes,binary_f,range_f,keep_dims,any_order,initial,args...);
    }

    template<typename Policy, typename BinaryF, typename RangeF, typename Initial, typename...Ts, typename...Args>
    static auto reduce_flatten(Policy policy, const basic_tensor<Ts...>& parent, BinaryF binary_f, RangeF range_f, bool keep_dims, bool any_order, const Initial initial, const Args&...args){
        return reduce_flatten_(policy,parent,binary_f,range_f,keep_dims,any_order,initial,args...);
    }

    template<typename ResultT, typename...Ts, typename DimT, typename F, typename IdxT, typename...Args>
    static auto slide(const basic_tensor<Ts...>& t, const DimT& axis, F f, const IdxT& window_size, const IdxT& window_step, const Args&...args){
        return slide_<ResultT>(t,axis,f,window_size,window_step,args...);
    }

    template<typename ResultT, typename...Ts, typename F, typename IdxT, typename...Args>
    static auto slide_flatten(const basic_tensor<Ts...>& t, F f, const IdxT& window_size, const IdxT& window_step, const Args&...args){
        return slide_flatten_<ResultT>(t,f,window_size,window_step,args...);
    }

    template<typename...Ts, typename DimT, typename F, typename...Args>
    static void transform(basic_tensor<Ts...>& t, const DimT& axis, F f, const Args&...args){
        transform_(t,axis,f,args...);
    }

};


//reduce like over flatten, F is binary functor
//policy is specialization of multithreading::exec_pol
template<typename Policy, typename F, typename...Ts, typename Initial=detail::no_value>
auto reduce_binary_flatten(Policy policy, const basic_tensor<Ts...>& t, F f, bool keep_dims, const Initial& initial=Initial{}){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce_binary_flatten(policy, t, f, keep_dims, initial);
}

//make tensor reduction along axes, axes can be scalar or container,
//F is binary reduce functor that operates on tensor's elements
//result tensor has value_type that is return type of F
//initial must be such that expression f(initial,element) be valid or no_value
//policy is specialization of multithreading::exec_pol
template<typename Policy, typename F, typename Axes, typename...Ts, typename Initial=detail::no_value>
auto reduce_binary(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, const Initial& initial=Initial{}){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce_binary(policy, t, axes, f, keep_dims, initial);
}

//reduce like over flatten, F takes iterators range to be reduced
//if any_order true traverse order unspecified, c_order otherwise
template<typename F, typename...Ts, typename...Args>
auto reduce_range_flatten(const basic_tensor<Ts...>& t, F f, bool keep_dims, bool any_order, const Args&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce_range_flatten(t, f, keep_dims, any_order, args...);
}

//make tensor reduction along axes, axes can be scalar or container,
//F is reduce functor with parameters: iterators range of data to be reduced, optional parameters; must return scalar - reduction result
//F call operator must be defined like this: template<typename It,typename...Args> Ret operator()(It first, It last, Args...){...}, Args is optional parameters
//result tensor has value_type that is return type of F
//if any_order true traverse order unspecified, c_order otherwise
//policy is specialization of multithreading::exec_pol
template<typename Policy, typename F, typename Axes, typename...Ts, typename...Args>
auto reduce_range(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, bool any_order, const Args&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce_range(policy, t, axes, f, keep_dims, any_order, args...);
}

//reduce according to specified policy
template<typename Policy, typename BinaryF, typename RangeF, typename Axes, typename Initial, typename...Ts, typename...Args>
auto reduce(Policy policy, const basic_tensor<Ts...>& t, const Axes& axes, BinaryF binary_f, RangeF range_f, bool keep_dims, bool any_order, const Initial initial, const Args&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce(policy, t, axes, binary_f, range_f, keep_dims, any_order, initial, args...);
}
template<typename Policy, typename BinaryF, typename RangeF, typename Initial, typename...Ts, typename...Args>
auto reduce_flatten(Policy policy, const basic_tensor<Ts...>& t, BinaryF binary_f, RangeF range_f, bool keep_dims, bool any_order, const Initial initial, const Args&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce_flatten(policy, t, binary_f, range_f, keep_dims, any_order, initial, args...);
}

//make tensor that is result of applying F to sliding window over axis, axis is scalar
//F is slide functor that takes iterators range of data to be slided, dst iterators range, optional parameters
//F call operator must be defined like this: template<typename It,typename DstIt,typename...Args> void operator()(It first, It last, DstIt dfirst, DstIt dlast, Args...){...}
//where Args is optional parameters
//result tensor's has value_type should be specialized explicitly
template<typename ResultT, typename DimT, typename...Ts, typename F, typename IdxT, typename...Args>
auto slide(const basic_tensor<Ts...>& t, const DimT& axis, F f, const IdxT& window_size, const IdxT& window_step, const Args&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::template slide<ResultT>(t, axis, f, window_size, window_step,args...);
}
template<typename ResultT, typename F, typename...Ts, typename IdxT, typename...Args>
auto slide_flatten(const basic_tensor<Ts...>& t, F f, const IdxT& window_size, const IdxT& window_step, const Args&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::template slide_flatten<ResultT>(t, f, window_size, window_step,args...);
}

//transform tensor inplace along specified axis
//F is transform functor that takes iterators range of data to be transformed
//F call operator must be defined like this: template<typename It, typename...Args> void operator()(It first, It last, Args..){...} Args is optional parameters
template<typename...Ts, typename DimT, typename F, typename...Args>
void transform(basic_tensor<Ts...>& t, const DimT& axis, F f, const Args&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    reducer_selector_t<config_type>::transform(t, axis, f, args...);
}

}   //end of namespace gtensor
#endif