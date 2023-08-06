#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include "module_selector.hpp"
#include "common.hpp"
#include "exception.hpp"
#include "math.hpp"
#include "iterator.hpp"
#include "indexing.hpp"

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

template<typename ShT>
auto make_axis_size(const ShT& shape, const typename ShT::difference_type& axis){
    return shape[axis];
}
template<typename ShT, typename Axes>
auto make_axes_size(const ShT& shape, const typename ShT::value_type& size, const Axes& axes){
    using index_type = typename ShT::value_type;
    if constexpr (detail::is_container_v<Axes>){
        return std::accumulate(axes.begin(),axes.end(),index_type{1},[&shape](const auto& r, const auto& a){return r*shape[a];});
    }else{
        return make_axis_size(shape,axes);
    }
}

}   //end of namespace detail

class reducer
{
    template<typename F, typename Axes, typename...Ts, typename...Args>
    static auto reduce_(const basic_tensor<Ts...>& parent, const Axes& axes_, F reduce_f, bool keep_dims, Args&&...args){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using traverse_order = typename parent_type::traverse_order;
        using dim_type = typename config_type::dim_type;
        using index_type = typename config_type::index_type;
        using shape_type = typename config_type::shape_type;
        using strides_div_type = detail::strides_div_t<config_type>;
        using axes_container_type = typename config_type::template container<dim_type>;
        using axes_type = decltype(detail::make_axes<axes_container_type>(std::declval<dim_type>(),axes_));
        using predicate_type = decltype(detail::make_traverse_predicate(std::declval<axes_type&>(),std::false_type{}));
        using walker_type = decltype(parent.create_walker());
        using iterator_type = decltype(
            detail::make_axes_iterator<traverse_order>(
                std::declval<shape_type&>(),
                std::declval<strides_div_type&>(),
                std::declval<walker_type>(),
                std::declval<index_type>(),
                std::declval<predicate_type>()
            )
        );
        using result_type = decltype(reduce_f(std::declval<iterator_type>(),std::declval<iterator_type>(),std::declval<Args>()...));
        using res_value_type = std::remove_cv_t<std::remove_reference_t<result_type>>;
        using res_config_type = config::extend_config_t<config_type,res_value_type>;
        using axis_type = detail::axis_type_t<Axes>;
        static_assert(math::numeric_traits<axis_type>::is_integral() && !std::is_same_v<bool,axis_type>,"Axes must be container of integrals or integral");

        auto axes = detail::make_axes<axes_container_type>(parent.dim(),axes_);
        const auto& pshape = parent.shape();
        detail::check_reduce_args(pshape, axes);
        auto res = tensor<res_value_type,order,res_config_type>{detail::make_reduce_shape(pshape, axes, keep_dims)};
        if (!res.empty()){
            if (parent.empty()){    //zero size axis is reduced
                auto a = parent.traverse_order_adapter(order{});
                const auto e = reduce_f(a.begin(), a.end(), std::forward<Args>(args)...);
                std::fill(res.begin(), res.end(), e);
            }else{
                const auto res_size = res.size();
                if (res_size == index_type{1}){
                    const auto pdim = parent.dim();
                    if (pdim == dim_type{1}){   //1d, can use native order
                        auto a = parent.traverse_order_adapter(order{});
                        *res.begin() = reduce_f(a.begin(), a.end(), std::forward<Args>(args)...);
                    }else{  //traverse like over flatten
                        auto a = parent.traverse_order_adapter(config::c_order{});
                        *res.begin() = reduce_f(a.begin(), a.end(), std::forward<Args>(args)...);
                    }
                }else{
                    const auto axes_size = detail::make_axes_size(pshape,parent.size(),axes);
                    auto predicate = detail::make_traverse_predicate(axes,std::false_type{});
                    const auto strides = detail::make_strides_div_predicate<config_type>(pshape,predicate,traverse_order{});
                    auto traverser = detail::make_forward_traverser(pshape,parent.create_walker(),detail::make_traverse_predicate(axes,std::true_type{}));  //traverse all but axes
                    auto a = res.traverse_order_adapter(order{});
                    auto res_it = a.begin();
                    do{
                        *res_it = reduce_f(
                            detail::make_axes_iterator<traverse_order>(pshape,strides,traverser.walker(),0,predicate),
                            detail::make_axes_iterator<traverse_order>(pshape,strides,traverser.walker(),axes_size,predicate),
                            std::forward<Args>(args)...
                        );
                        ++res_it;
                    }while(traverser.template next<order>());
                }
            }
        }
        return res;
    }

    template<typename F, typename...Ts, typename...Args>
    static auto reduce_flatten_(const basic_tensor<Ts...>& t, F reduce_f, bool keep_dims, Args&&...args){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using config_type = typename tensor_type::config_type;
        using result_type = decltype(reduce_f(t.begin(),t.end(),std::declval<Args>()...));
        using res_value_type = std::remove_cv_t<std::remove_reference_t<result_type>>;
        using res_config_type = config::extend_config_t<config_type,res_value_type>;
        using res_type = tensor<res_value_type,order,res_config_type>;
        //assuming iterator traverse order doesn't change result type, may not compile otherwise
        if (t.dim() == 1){   //1d, can use native order
            auto a = t.traverse_order_adapter(order{});
            return  res_type(detail::make_reduce_shape(t.shape(),keep_dims), reduce_f(a.begin(), a.end(), std::forward<Args>(args)...));
        }else{  //traverse like over flatten
            //assuming changing traverse order when traverse like over flatten is not logic error (i.e. reduce functor not expected particular order)
            //logic error for argmax like functions
            //auto a = t.traverse_order_adapter(order{});
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
                const auto parent_axis_size = detail::make_axis_size(pshape,axis);
                const auto res_axis_size = detail::make_axis_size(res_shape,axis);
                auto predicate = detail::make_traverse_predicate(axis,std::true_type{});    //inverse, to traverse all but axis
                auto parent_traverser = detail::make_forward_traverser(pshape,parent.create_walker(),predicate);
                auto res_traverser = detail::make_forward_traverser(res_shape,res.create_walker(),predicate);
                do{
                    //0first,1last,2dst_first,3dst_last,4args
                    slide_f(
                        detail::make_axis_iterator(parent_traverser.walker(),axis,index_type{0}),
                        detail::make_axis_iterator(parent_traverser.walker(),axis,parent_axis_size),
                        detail::make_axis_iterator(res_traverser.walker(),axis,index_type{0}),
                        detail::make_axis_iterator(res_traverser.walker(),axis,res_axis_size),
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
        using index_type = typename config_type::index_type;

        const auto& pshape = parent.shape();
        const dim_type axis = detail::make_axis(pshape,axis_);
        detail::check_transform_args(pshape, axis);
        const auto pdim = parent.dim();
        if (pdim == dim_type{1}){
            auto a = parent.traverse_order_adapter(order{});
            transform_f(a.begin(), a.end(), std::forward<Args>(args)...);
        }else{
            const auto parent_axis_size = detail::make_axis_size(pshape,axis);
            auto traverser = detail::make_forward_traverser(pshape,parent.create_walker(),detail::make_traverse_predicate(axis,std::true_type{}));
            do{
                //0first,1last,2args
                transform_f(
                    detail::make_axis_iterator(traverser.walker(),axis,index_type{0}),
                    detail::make_axis_iterator(traverser.walker(),axis,parent_axis_size),
                    std::forward<Args>(args)...
                );
            }while(traverser.template next<order>());
        }
    }

public:
    //interface
    template<typename F, typename Axes, typename...Ts, typename...Args>
    static auto reduce(const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, Args&&...args){
        using dim_type = typename basic_tensor<Ts...>::dim_type;
        if constexpr (detail::is_container_of_type_v<Axes,dim_type>){
            if (axes.size() == 1){
                return reduce_(t,*axes.begin(),f,keep_dims,std::forward<Args>(args)...);
            }
        }
        return reduce_(t,axes,f,keep_dims,std::forward<Args>(args)...);
    }

    template<typename F, typename...Ts, typename...Args>
    static auto reduce_flatten(const basic_tensor<Ts...>& t, F f, bool keep_dims, Args&&...args){
        return reduce_flatten_(t,f,keep_dims,std::forward<Args>(args)...);
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
template<typename F, typename Axes, typename...Ts, typename...Args>
auto reduce(const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce(t, axes, f, keep_dims, std::forward<Args>(args)...);
}
//reduce like over flatten
template<typename F, typename...Ts, typename...Args>
auto reduce_flatten(const basic_tensor<Ts...>& t, F f, bool keep_dims, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce_flatten(t, f, keep_dims, std::forward<Args>(args)...);
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