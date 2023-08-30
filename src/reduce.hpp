#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include "module_selector.hpp"
#include "common.hpp"
#include "exception.hpp"
#include "math.hpp"
#include "iterator.hpp"
#include "indexing.hpp"
#include "queue.hpp"
#include "thread_pool.hpp"

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


inline constexpr std::size_t pool_workers_n = 10;
inline constexpr std::size_t pool_queue_size = 400;
inline auto& get_pool(){
    static thread_pool::thread_pool_v3 pool_{pool_workers_n, pool_queue_size};
    return pool_;
}

}   //end of namespace detail

class reducer
{
    //axes can be container or scalar
    //F is binary functor, takes elements, return reduce result, like std::plus
    //initial must be such that expression reduce_f(initial,element) be valid or no_value
    //traverse input countigous, i.e. traverse order dependes on layout
    template<typename F, typename Axes, typename...Ts, typename Initial>
    static auto reduce_binary_(const basic_tensor<Ts...>& parent, const Axes& axes_, F reduce_f, bool keep_dims, const Initial& initial){
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
        auto res = tensor<res_value_type,order,res_config_type>{detail::make_reduce_shape(pshape, axes, keep_dims)};
        if (!res.empty()){
            auto a_parent = parent.traverse_order_adapter(order{});
            auto a_res = res.traverse_order_adapter(order{});
            //empty axes container
            if constexpr (detail::is_container_v<Axes>){
                if (axes.empty()){
                    if constexpr (has_initial){
                        std::transform(a_res.begin(),a_res.end(),a_parent.begin(),a_res.begin(),[&reduce_f,&initial](auto&&, auto&& r){return reduce_f(initial,r);});
                    }else{
                        //assuming reduction of scalar without initial is identity - nothing to reduce
                        std::transform(a_res.begin(),a_res.end(),a_parent.begin(),a_res.begin(),[](auto&&, auto&& r){return static_cast<const res_value_type&>(r);});
                    }
                    return res;
                }
            }
            //reduce zero size
            if (parent.empty()){
                if constexpr (has_initial){
                    std::fill(a_res.begin(),a_res.end(),initial);
                    return res;
                }else{
                    throw value_error("cant reduce zero size dimension without initial value");
                }
            }
            //0dim result
            if (res.size() == index_type{1}){
                if constexpr (has_initial){
                    *a_res.begin() = std::accumulate(a_parent.begin(),a_parent.end(),initial,reduce_f);
                }else{
                    auto it=a_parent.begin();
                    const auto& initial_ = *it;
                    *a_res.begin() = std::accumulate(++it,a_parent.end(),initial_,reduce_f);
                }
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
            auto it = a_parent.begin();
            const auto res_first = a_res.begin();
            auto res_it = res_first;
            bool init = true;
            auto offset = *traverser;
            if (inner_size == 1){
                do{
                    auto& e = *res_it;
                    if (init){
                        if constexpr (has_initial){
                            e = std::accumulate(it,it+outer_size,initial,reduce_f);
                        }else{  //no initial
                            e = std::accumulate(it+1,it+outer_size,*it,reduce_f);
                        }
                    }else{  //e initialized
                        e = std::accumulate(it,it+outer_size,e,reduce_f);
                    }
                    it+=outer_size;

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
                            std::transform(res_it,res_it+inner_size,it,res_it,[&reduce_f,&initial](auto&&, auto&& r){return reduce_f(initial,r);});
                        }else{  //no initial
                            std::transform(res_it,res_it+inner_size,it,res_it,[](auto&&, auto&& r){return static_cast<const res_value_type&>(r);});
                        }
                        it+=inner_size;
                    }
                    const auto i_stop=init?1:0;
                    for (auto i=outer_size; i!=i_stop; --i){
                        std::transform(res_it,res_it+inner_size,it,res_it,reduce_f);
                        it+=inner_size;
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
        }
        return res;
    }

    //F takes iterators range to be reduces and additional args
    template<typename F, typename Axes, typename...Ts, typename...Args>
    static auto reduce_(const basic_tensor<Ts...>& parent, const Axes& axes_, F reduce_f, bool keep_dims, bool any_order, Args&&...args){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using traverse_order = config::c_order; //order to traverse along axes
        using dim_type = typename config_type::dim_type;
        using index_type = typename config_type::index_type;
        using shape_type = typename config_type::shape_type;
        using axes_iterator_maker_type = decltype(detail::make_axes_iterator_maker<config_type>(std::declval<shape_type>(),axes_,traverse_order{}));
        using axes_iterator_type = decltype(std::declval<axes_iterator_maker_type>().begin(parent.create_walker(),std::false_type{}));
        using result_type = decltype(reduce_f(std::declval<axes_iterator_type>(),std::declval<axes_iterator_type>(),std::declval<Args>()...));
        using res_value_type = std::remove_cv_t<std::remove_reference_t<result_type>>;
        using res_config_type = config::extend_config_t<config_type,res_value_type>;

        const auto pdim = parent.dim();
        const auto& pshape = parent.shape();
        auto axes = detail::make_axes<config_type>(pdim,axes_);
        detail::check_reduce_args(pshape, axes);
        auto res = tensor<res_value_type,order,res_config_type>{detail::make_reduce_shape(pshape, axes, keep_dims)};
        if (!res.empty()){
            if (parent.empty()){    //zero size axis is reduced
                auto a = parent.traverse_order_adapter(order{});
                const auto e = reduce_f(a.begin(), a.end(), std::forward<Args>(args)...);
                std::fill(res.begin(), res.end(), e);
            }else{
                if (res.size() == index_type{1}){
                    if (pdim == dim_type{1} || any_order){   //1d or any_order, can use native order
                        auto a = parent.traverse_order_adapter(order{});
                        *res.begin() = reduce_f(a.begin(), a.end(), std::forward<Args>(args)...);
                    }else{  //traverse like over flatten
                        auto a = parent.traverse_order_adapter(traverse_order{});
                        *res.begin() = reduce_f(a.begin(), a.end(), std::forward<Args>(args)...);
                    }
                }else{
                    auto axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(pshape,axes,traverse_order{});
                    auto traverser = axes_iterator_maker.create_forward_traverser(parent.create_walker(),std::true_type{});
                    auto a = res.traverse_order_adapter(order{});
                    auto res_it = a.begin();

                    // using future_type = decltype(
                    //     std::declval<decltype(detail::get_pool())>().push(
                    //         reduce_f,
                    //         axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{}),
                    //         axes_iterator_maker.end_complement(traverser.walker(),std::false_type{}),
                    //         std::forward<Args>(args)...
                    //     )
                    // );
                    //static constexpr int max_par_tasks = 100;
                    //queue::st_bounded_queue<future_type> futures{max_par_tasks};
                    do{
                        // auto f = detail::get_pool().push(
                        //     reduce_f,
                        //     axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{}),
                        //     axes_iterator_maker.end_complement(traverser.walker(),std::false_type{}),
                        //     std::forward<Args>(args)...
                        // );
                        // if (!futures.try_push(std::move(f))){    //par task limit
                        //     future_type f_{};
                        //     futures.try_pop(f_);
                        //     *res_it = f_.get();
                        //     ++res_it;
                        //     futures.try_push(std::move(f));
                        // }

                        *res_it = reduce_f(
                            axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{}),
                            axes_iterator_maker.end_complement(traverser.walker(),std::false_type{}),
                            std::forward<Args>(args)...
                        );
                        ++res_it;

                    }while(traverser.template next<order>());
                    // future_type f_{};
                    // while(futures.try_pop(f_)){
                    //     *res_it = f_.get();
                    //     ++res_it;
                    // }
                }
            }
        }
        return res;
    }

    template<typename F, typename...Ts, typename...Args>
    static auto reduce_flatten_(const basic_tensor<Ts...>& t, F reduce_f, bool keep_dims, bool any_order, Args&&...args){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using config_type = typename tensor_type::config_type;
        using result_type = decltype(reduce_f(t.begin(),t.end(),std::declval<Args>()...));  //assuming traverse order not affected result type
        using res_value_type = std::remove_cv_t<std::remove_reference_t<result_type>>;
        using res_config_type = config::extend_config_t<config_type,res_value_type>;
        using res_type = tensor<res_value_type,order,res_config_type>;
        if (t.dim()==1 || any_order){   //1d or any_order, can use native order
            auto a = t.traverse_order_adapter(order{});
            return  res_type(detail::make_reduce_shape(t.shape(),keep_dims), reduce_f(a.begin(), a.end(), std::forward<Args>(args)...));
        }else{  //traverse like over flatten
            auto a = t.traverse_order_adapter(config::c_order{});
            return  res_type(detail::make_reduce_shape(t.shape(),keep_dims), reduce_f(a.begin(), a.end(), std::forward<Args>(args)...));
        }
    }

    template<typename ResultT, typename...Ts, typename DimT, typename F, typename IdxT, typename...Args>
    static auto slide_(const basic_tensor<Ts...>& parent, const DimT& axis_, F slide_f, const IdxT& window_size_, const IdxT& window_step_, Args&&...args)
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
            const auto pdim = parent.dim();
            if (pdim == dim_type{1}){
                auto parent_a = parent.traverse_order_adapter(order{});
                auto res_a = res.traverse_order_adapter(order{});
                slide_f(parent_a.begin(), parent_a.end(), res_a.begin(), res_a.end(), std::forward<Args>(args)...);
            }else{
                const auto& res_shape = res.shape();
                auto parent_axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(pshape,axis,order{});
                auto res_axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(res_shape,axis,order{});
                auto parent_traverser = parent_axes_iterator_maker.create_forward_traverser(parent.create_walker(),std::true_type{});
                auto res_traverser = res_axes_iterator_maker.create_forward_traverser(res.create_walker(),std::true_type{});
                do{
                    //0first,1last,2dst_first,3dst_last,4args
                    slide_f(
                        parent_axes_iterator_maker.begin_complement(parent_traverser.walker(),std::false_type{}),
                        parent_axes_iterator_maker.end_complement(parent_traverser.walker(),std::false_type{}),
                        res_axes_iterator_maker.begin_complement(res_traverser.walker(),std::false_type{}),
                        res_axes_iterator_maker.end_complement(res_traverser.walker(),std::false_type{}),
                        std::forward<Args>(args)...
                    );
                    res_traverser.template next<order>();
                }while(parent_traverser.template next<order>());
            }
        }
        return res;
    }

    template<typename ResultT, typename...Ts, typename F, typename IdxT, typename...Args>
    static auto slide_flatten_(const basic_tensor<Ts...>& parent, F slide_f, const IdxT& window_size_, const IdxT& window_step_, Args&&...args)
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
            const auto pdim = parent.dim();
            auto res_a = res.traverse_order_adapter(order{});
            if (pdim == dim_type{1}){
                auto parent_a = parent.traverse_order_adapter(order{});
                slide_f(parent_a.begin(), parent_a.end(), res_a.begin(), res_a.end(), std::forward<Args>(args)...);
            }else{
                auto parent_a = parent.traverse_order_adapter(config::c_order{});
                slide_f(parent_a.begin(), parent_a.end(), res_a.begin(), res_a.end(), std::forward<Args>(args)...);
            }
        }
        return res;
    }

    template<typename F, typename DimT, typename...Ts, typename...Args>
    static void transform_(basic_tensor<Ts...>& parent, const DimT& axis_, F transform_f, Args&&...args){
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
            transform_f(a.begin(), a.end(), std::forward<Args>(args)...);
        }else{
            auto axes_iterator_maker = detail::make_axes_iterator_maker<config_type>(pshape,axis,order{});
            auto traverser = axes_iterator_maker.create_forward_traverser(parent.create_walker(),std::true_type{});
            do{
                //0first,1last,2args
                transform_f(
                    axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{}),
                    axes_iterator_maker.end_complement(traverser.walker(),std::false_type{}),
                    std::forward<Args>(args)...
                );
            }while(traverser.template next<order>());
        }
    }

public:
    //interface
    template<typename F, typename Axes, typename...Ts, typename...Args>
    static auto reduce(const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, bool any_order, Args&&...args){
        using dim_type = typename basic_tensor<Ts...>::dim_type;
        if constexpr (detail::is_container_of_type_v<Axes,dim_type>){
            if (axes.size() == 1){
                return reduce_(t,*axes.begin(),f,keep_dims,any_order,std::forward<Args>(args)...);
            }
        }
        return reduce_(t,axes,f,keep_dims,any_order,std::forward<Args>(args)...);
    }

    template<typename F, typename Axes, typename...Ts, typename Initial>
    static auto reduce_binary(const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, const Initial& initial){
        return reduce_binary_(t,axes,f,keep_dims,initial);
    }

    template<typename F, typename...Ts, typename...Args>
    static auto reduce_flatten(const basic_tensor<Ts...>& t, F f, bool keep_dims, bool any_order, Args&&...args){
        return reduce_flatten_(t,f,keep_dims,any_order,std::forward<Args>(args)...);
    }

    template<typename ResultT, typename...Ts, typename DimT, typename F, typename IdxT, typename...Args>
    static auto slide(const basic_tensor<Ts...>& t, const DimT& axis, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
        return slide_<ResultT>(t,axis,f,window_size,window_step,std::forward<Args>(args)...);
    }

    template<typename ResultT, typename...Ts, typename F, typename IdxT, typename...Args>
    static auto slide_flatten(const basic_tensor<Ts...>& t, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
        return slide_flatten_<ResultT>(t,f,window_size,window_step,std::forward<Args>(args)...);
    }

    template<typename...Ts, typename DimT, typename F, typename...Args>
    static void transform(basic_tensor<Ts...>& t, const DimT& axis, F f, Args&&...args){
        transform_(t,axis,f,std::forward<Args>(args)...);
    }

};

//make tensor reduction along axes, axes can be scalar or container,
//F is reduce functor with parameters: iterators range of data to be reduced, optional parameters; must return scalar - reduction result
//F call operator must be defined like this: template<typename It,typename...Args> Ret operator()(It first, It last, Args...){...}, Args is optional parameters
//result tensor has value_type that is return type of F
//if any_order true traverse order unspecified, c_order otherwise
template<typename F, typename Axes, typename...Ts, typename...Args>
auto reduce(const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, bool any_order, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce(t, axes, f, keep_dims, any_order, std::forward<Args>(args)...);
}

//make tensor reduction along axes, axes can be scalar or container,
//F is binary reduce functor that operates on tensor's elements
//result tensor has value_type that is return type of F
//initial must be such that expression f(initial,element) be valid or no_value
template<typename F, typename Axes, typename...Ts, typename Initial=detail::no_value>
auto reduce_binary(const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, const Initial& initial=Initial{}){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce_binary(t, axes, f, keep_dims, initial);
}

//reduce like over flatten
//if any_order true traverse order unspecified, c_order otherwise
template<typename F, typename...Ts, typename...Args>
auto reduce_flatten(const basic_tensor<Ts...>& t, F f, bool keep_dims, bool any_order, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce_flatten(t, f, keep_dims, any_order, std::forward<Args>(args)...);
}

//make tensor that is result of applying F to sliding window over axis, axis is scalar
//F is slide functor that takes iterators range of data to be slided, dst iterators range, optional parameters
//F call operator must be defined like this: template<typename It,typename DstIt,typename...Args> void operator()(It first, It last, DstIt dfirst, DstIt dlast, Args...){...}
//where Args is optional parameters
//result tensor's has value_type should be specialized explicitly
template<typename ResultT, typename DimT, typename...Ts, typename F, typename IdxT, typename...Args>
auto slide(const basic_tensor<Ts...>& t, const DimT& axis, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::template slide<ResultT>(t, axis, f, window_size, window_step,std::forward<Args>(args)...);
}
template<typename ResultT, typename F, typename...Ts, typename IdxT, typename...Args>
auto slide_flatten(const basic_tensor<Ts...>& t, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::template slide_flatten<ResultT>(t, f, window_size, window_step,std::forward<Args>(args)...);
}

//transform tensor inplace along specified axis
//F is transform functor that takes iterators range of data to be transformed
//F call operator must be defined like this: template<typename It, typename...Args> void operator()(It first, It last, Args..){...} Args is optional parameters
template<typename...Ts, typename DimT, typename F, typename...Args>
void transform(basic_tensor<Ts...>& t, const DimT& axis, F f, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    reducer_selector_t<config_type>::transform(t, axis, f, std::forward<Args>(args)...);
}

}   //end of namespace gtensor
#endif