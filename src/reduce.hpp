#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include "module_selector.hpp"
#include "common.hpp"
#include "iterator.hpp"
#include "tensor.hpp"

namespace gtensor{

class reduce_exception : public std::runtime_error{
public:
    explicit reduce_exception(const char* what):
        runtime_error(what)
    {}
};

namespace detail{

template<typename Container, typename DimT, typename Axes>
auto make_axes(const DimT& dim, const Axes& axes_){
    using dim_type = DimT;
    if constexpr (detail::is_container_of_type_v<Axes,dim_type>){
        using container_size_type = typename Container::size_type;
        using axes_size_type = typename Axes::size_type;
        Container res{};
        if constexpr (std::is_convertible_v<axes_size_type, container_size_type>){
            res.reserve(static_cast<container_size_type>(axes_.size()));
        }
        for (auto it=axes_.begin(), last=axes_.end(); it!=last; ++it){
            res.push_back(make_axis(dim,*it));
        }
        return res;
    }else{
        return make_axis(dim,axes_);
    }
}

template<typename ShT>
auto check_reduce_args(const ShT& shape, const typename ShT::difference_type& axis){
    using dim_type = typename ShT::difference_type;
    const dim_type dim = detail::make_dim(shape);
    if (axis >= dim){
        throw reduce_exception("invalid reduce axis: axis is out of bounds");
    }
}
template<typename ShT, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename ShT::difference_type>,int> =0>
auto check_reduce_args(const ShT& shape, const Container& axes){
    using dim_type = typename ShT::difference_type;
    const dim_type dim = detail::make_dim(shape);
    const dim_type axes_number = static_cast<dim_type>(axes.size());
    if (axes_number > dim){
        throw reduce_exception("invalid reduce axes: too many axes");
    }
    auto it=axes.begin();
    auto last=axes.end();
    while(it!=last){
        const dim_type& axis = static_cast<dim_type>(*it);
        if (axis >= dim || axis < dim_type{0}){
            throw reduce_exception("invalid reduce axes: axis is out of bounds");
        }
        ++it;
        if (std::find(it, last, axis) != last){
            throw reduce_exception("invalid reduce axes: duplicates in axes");
        }
    }
}

template<typename ShT>
auto check_transform_args(const ShT& shape, const typename ShT::difference_type& axis){
    using dim_type = typename ShT::difference_type;
    const dim_type dim = detail::make_dim(shape);
    if (axis >= dim){
        throw reduce_exception("invalid transform axis: axis is out of bounds");
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
    if (keep_dims){
        if (axes_number == dim_type{0}){  //all axes
            return shape_type(dim, index_type{1});
        }else{
            shape_type res(shape);
            for (auto it=axes.begin(), last=axes.end(); it!=last; ++it){
                res[*it] = index_type{1};
            }
            return res;
        }
    }else{
        if (axes_number == dim_type{0}){  //all axes
            return shape_type{};
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

template<typename IdxT>
auto check_slide_args(const IdxT& size, const IdxT& window_size, const IdxT& window_step){
    using index_type = IdxT;
    if (window_size > size || window_size <= index_type{0}){
        throw reduce_exception("invalid sliding window size");
    }
    if (window_step < index_type{1}){
        throw reduce_exception("invalid sliding window step");
    }
}
template<typename ShT, typename DimT, typename IdxT>
auto check_slide_args(const ShT& shape, const DimT& axis, const IdxT& window_size, const IdxT& window_step){
    using dim_type = DimT;
    using index_type = IdxT;
    const dim_type dim = detail::make_dim(shape);
    if (axis >= dim){
        throw reduce_exception("invalid slide axis");
    }
    index_type axis_size = shape[axis];
    check_slide_args(axis_size,window_size,window_step);
}
template<typename IdxT>
auto make_slide_size(const IdxT& size, const IdxT& window_size, const IdxT& window_step){
    return (size - window_size)/window_step + IdxT{1};
}
template<typename ShT, typename DimT, typename IdxT>
auto make_slide_shape(const ShT& shape, const DimT& axis, const IdxT& window_size, const IdxT& window_step){
    using index_type = IdxT;
    using shape_type = ShT;
    shape_type res(shape);
    const index_type axis_size = shape[axis];
    if (axis_size != index_type{0}){
        const index_type result_axis_size = make_slide_size(axis_size, window_size, window_step);
        res[axis] = result_axis_size;
    }
    return res;
}

template<typename ShT>
auto make_slide_axis_size(const ShT& shape, const typename ShT::difference_type& axis){
    return shape[axis];
}

template<typename ShT, typename Axes>
auto make_reduce_axes_size(const ShT& shape, const typename ShT::value_type& size, const Axes& axes){
    using dim_type = typename ShT::difference_type;
    using index_type = typename ShT::value_type;
    if constexpr (detail::is_container_of_type_v<Axes,dim_type>){
        if (axes.size()==0){
            return size;
        }else{
            index_type axes_size{1};
            for (auto it=axes.begin(), last=axes.end(); it!=last; ++it){
                axes_size*=shape[static_cast<dim_type>(*it)];
            }
            return axes_size;
        }
    }else{
        return make_slide_axis_size(shape,axes);
    }
}

template<typename Config, typename Walker>
class axis_iterator
{
protected:
    using walker_type = Walker;
    using index_type = typename Config::index_type;
    using dim_type = typename Config::dim_type;
    using result_type = decltype(*std::declval<walker_type>());
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = index_type;
    using value_type = typename detail::iterator_internals_selector<result_type>::value_type;
    using pointer = typename detail::iterator_internals_selector<result_type>::pointer;
    using reference = typename detail::iterator_internals_selector<result_type>::reference;
    using const_reference = typename detail::iterator_internals_selector<result_type>::const_reference;

    //assuming usual stoarge subscript operator semantic i.e. subscript index in range [0,size()-1]:
    //begin should be constructed with zero flat_index_ argument, end with size() flat_index_argument
    template<typename Walker_>
    axis_iterator(Walker_&& walker_, const dim_type& reduce_axis_, const difference_type& flat_index_):
        walker{std::forward<Walker_>(walker_)},
        reduce_axis{reduce_axis_},
        flat_index{flat_index_}
    {
        if (flat_index_ > 0){
            walker.walk(reduce_axis_, flat_index_ - difference_type{1});
            walker.step(reduce_axis_);
        }
    }
    axis_iterator& operator+=(difference_type n){
        advance(n);
        return *this;
    }
    axis_iterator& operator++(){
        walker.step(reduce_axis);
        ++flat_index;
        return *this;
    }
    axis_iterator& operator--(){
        walker.step_back(reduce_axis);
        --flat_index;
        return *this;
    }
    result_type operator[](difference_type n)const{return *(*this+n);}
    result_type operator*() const{return *walker;}
    inline difference_type friend operator-(const axis_iterator& lhs, const axis_iterator& rhs){return lhs.flat_index - rhs.flat_index;}
private:
    void advance(difference_type n){
        walker.walk(reduce_axis, n);
        flat_index+=n;
    }
    walker_type walker;
    dim_type reduce_axis;
    difference_type flat_index;
};

GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_PLUS(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_MINUS(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_EQUAL(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_NOT_EQUAL(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_GREATER(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_LESS(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_GREATER_EQUAL(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_LESS_EQUAL(axis_iterator);

template<typename Config, typename Axes>
class reduce_traverse_predicate
{
    using config_type = Config;
    using axes_type = Axes;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    static_assert(detail::is_container_of_type_v<axes_type,dim_type> || std::is_convertible_v<axes_type,dim_type>);

    const axes_type* axes_;
    bool inverse_;

    bool is_in_axes(const dim_type& d)const{
        if constexpr (detail::is_container_of_type_v<axes_type,dim_type>){
            if (axes_->size()==0){
                return true;
            }else{
                const auto last = axes_->end();
                return std::find_if(axes_->begin(),last,[&d](const auto& dir){return d == static_cast<dim_type>(dir);}) != last;
            }
        }else{
            return d == static_cast<dim_type>(*axes_);
        }
    }

    bool apply_inverse(bool b)const{
        return inverse_ != b;
    }

public:
    reduce_traverse_predicate(const axes_type& axes__, bool inverse__):
        axes_{&axes__},
        inverse_{inverse__}
    {}

    bool operator()(const dim_type& d)const{
        return apply_inverse(is_in_axes(d));
    }
};

template<typename Walker, typename DimT, typename IdxT>
auto make_axis_iterator(const Walker& walker, const DimT& axis, const IdxT& pos){
    using config_type = typename Walker::config_type;
    using iterator_type = axis_iterator<config_type,Walker>;
    return iterator_type{walker,axis,pos};
}

template<typename ShT, typename Strides, typename Walker, typename Axes, typename Predicate, typename IdxT>
auto make_axes_iterator(const ShT& shape, const Strides& strides, const Walker& walker, const Axes& axes, const Predicate& predicate, const IdxT& pos){
    using config_type = typename Walker::config_type;
    using traverse_order = typename config_type::order;
    using dim_type = typename config_type::dim_type;
    using predicate_type = std::remove_cv_t<std::remove_reference_t<Predicate>>;
    using iterator_type = walker_iterator<config_type,Walker,traverse_order,predicate_type>;

    if constexpr (detail::is_container_of_type_v<Axes,dim_type>){
        return iterator_type{walker, shape, strides, pos, predicate};
    }else{
        return make_axis_iterator(walker, axes, pos);
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
        using predicate_type = detail::reduce_traverse_predicate<config_type, axes_type>;
        using walker_type = decltype(parent.create_walker());
        using iterator_type = decltype(
            detail::make_axes_iterator(
                std::declval<shape_type>(),
                std::declval<strides_div_type>(),
                std::declval<walker_type>(),
                std::declval<axes_type>(),
                std::declval<predicate_type>(),
                std::declval<index_type>()
            )
        );
        using result_type = decltype(reduce_f(std::declval<iterator_type>(),std::declval<iterator_type>(),std::declval<Args>()...));
        using res_value_type = std::remove_cv_t<std::remove_reference_t<result_type>>;
        using res_config_type = config::extend_config_t<config_type,res_value_type>;


        auto axes = detail::make_axes<axes_container_type>(parent.dim(),axes_);
        const auto& pshape = parent.shape();
        detail::check_reduce_args(pshape, axes);
        auto res = tensor<res_value_type,order,res_config_type>{detail::make_reduce_shape(pshape, axes, keep_dims)};
        if (!res.empty()){
            if (parent.empty()){    //zero size axis is reduced
                auto a = parent.template traverse_order_adapter<order>();
                const auto e = reduce_f(a.begin(), a.end(), std::forward<Args>(args)...);
                detail::fill(res.begin(), res.end(), e);
            }else{
                const auto res_size = res.size();
                if (res_size == index_type{1}){
                    const auto pdim = parent.dim();
                    if (pdim == dim_type{1}){
                        auto a = parent.template traverse_order_adapter<order>();
                        *res.begin() = reduce_f(a.begin(), a.end(), std::forward<Args>(args)...);
                    }else{
                        *res.begin() = reduce_f(parent.begin(), parent.end(), std::forward<Args>(args)...);
                    }
                }else{
                    using traverser_type = walker_forward_traverser<config_type, walker_type, predicate_type>;
                    const predicate_type predicate{axes, true};
                    const predicate_type inverse_predicate{axes, false};
                    traverser_type traverser{pshape,parent.create_walker(),predicate};
                    const auto axes_size = detail::make_reduce_axes_size(pshape,parent.size(),axes);
                    const auto strides = detail::make_strides_div_predicate<config_type>(pshape,inverse_predicate,traverse_order{});
                    auto a = res.template traverse_order_adapter<order>();
                    auto res_it = a.begin();
                    do{
                        *res_it = reduce_f(
                            detail::make_axes_iterator(pshape,strides,traverser.walker(),axes,inverse_predicate,index_type{0}),
                            detail::make_axes_iterator(pshape,strides,traverser.walker(),axes,inverse_predicate,axes_size),
                            std::forward<Args>(args)...
                        );
                        ++res_it;
                    }while(traverser.template next<order>());
                }
            }
        }
        return res;
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
        const dim_type axis = detail::make_axis(pshape,axis_);
        const index_type window_size = static_cast<index_type>(window_size_);
        const index_type window_step = static_cast<index_type>(window_step_);
        detail::check_slide_args(pshape, axis, window_size, window_step);
        auto res = tensor<res_value_type,order,res_config_type>{detail::make_slide_shape(pshape, axis, window_size, window_step)};
        if (!res.empty()){
            const auto pdim = parent.dim();
            if (pdim == dim_type{1}){
                auto parent_a = parent.template traverse_order_adapter<order>();
                auto res_a = res.template traverse_order_adapter<order>();
                slide_f(parent_a.begin(), parent_a.end(), res_a.begin(), res_a.end(), std::forward<Args>(args)...);
            }else{
                using predicate_type = detail::reduce_traverse_predicate<config_type, dim_type>;
                using parent_traverser_type = walker_forward_traverser<config_type, decltype(parent.create_walker()), predicate_type>;
                using res_traverser_type = walker_forward_traverser<config_type, decltype(res.create_walker()), predicate_type>;
                predicate_type predicate{axis, true};
                const auto& res_shape = res.shape();
                parent_traverser_type parent_traverser{pshape, parent.create_walker(), predicate};
                res_traverser_type res_traverser{res_shape, res.create_walker(), predicate};
                const auto parent_axis_size = detail::make_slide_axis_size(pshape,axis);
                const auto res_axis_size = detail::make_slide_axis_size(res_shape,axis);
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
    static auto slide_(const basic_tensor<Ts...>& parent, F slide_f, const IdxT& window_size_, const IdxT& window_step_, Args&&...args)
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
            auto res_a = res.template traverse_order_adapter<order>();
            if (pdim == dim_type{1}){
                auto parent_a = parent.template traverse_order_adapter<order>();
                slide_f(parent_a.begin(), parent_a.end(), res_a.begin(), res_a.end(), std::forward<Args>(args)...);
            }else{
                auto parent_a = parent.template traverse_order_adapter<config::c_order>();
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
            transform_f(parent.begin(), parent.end(), std::forward<Args>(args)...);
        }else{
            using predicate_type = detail::reduce_traverse_predicate<config_type, dim_type>;
            using traverser_type = walker_forward_traverser<config_type, decltype(parent.create_walker()), predicate_type>;
            predicate_type predicate{axis, true};
            traverser_type traverser{pshape, parent.create_walker(), predicate};
            const auto parent_axis_size = detail::make_slide_axis_size(pshape,axis);
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

    template<typename ResultT, typename...Ts, typename DimT, typename F, typename IdxT, typename...Args>
    static auto slide(const basic_tensor<Ts...>& t, const DimT& axis, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
        return slide_<ResultT>(t,axis,f,window_size,window_step,std::forward<Args>(args)...);
    }

    template<typename ResultT, typename...Ts, typename F, typename IdxT, typename...Args>
    static auto slide(const basic_tensor<Ts...>& t, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
        return slide_<ResultT>(t,f,window_size,window_step,std::forward<Args>(args)...);
    }

    template<typename...Ts, typename DimT, typename F, typename...Args>
    static void transform(basic_tensor<Ts...>& t, const DimT& axis, F f, Args&&...args){
        transform_(t,axis,f,std::forward<Args>(args)...);
    }
};

//make tensor reduction along axis or axes
//axes is scalar or container, if axes is empty container reduce like over flatten (all axes)
//F is reduce functor with parameters: iterators range of data to be reduced, optional parameters; must return scalar - reduction result
//F call operator must be defined like this: template<typename It,typename...Args> Ret operator()(It first, It last, Args...){...}, Args is optional parameters
//result tensor has value_type that is return type of F
template<typename F, typename Axes, typename...Ts, typename...Args>
auto reduce(const basic_tensor<Ts...>& t, const Axes& axes, F f, bool keep_dims, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce(t, axes, f, keep_dims, std::forward<Args>(args)...);
}

//make tensor that is result of applying F to sliding window over axis, axis is scalar
//F is slide functor that takes iterators range of data to be slided, dst iterators range, optional parameters
//F call operator must be defined like this: template<typename It,typename DstIt,typename...Args> void operator()(It first, It last, DstIt dfirst, DstIt dlast, Args...){...}
//where Args is optional parameters
//result tensor has value_type that is same as source tensor value_type
template<typename DimT, typename...Ts, typename F, typename IdxT, typename...Args>
auto slide(const basic_tensor<Ts...>& t, const DimT& axis, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    using value_type = typename basic_tensor<Ts...>::value_type;
    return reducer_selector_t<config_type>::template slide<value_type>(t, axis, f, window_size, window_step,std::forward<Args>(args)...);
}
//slide like over flatten in c_order
template<typename F, typename...Ts, typename IdxT, typename...Args>
auto slide(const basic_tensor<Ts...>& t, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    using value_type = typename basic_tensor<Ts...>::value_type;
    return reducer_selector_t<config_type>::template slide<value_type>(t, f, window_size, window_step,std::forward<Args>(args)...);
}
//as above, but with explicit result value_type specialization
template<typename ResultT, typename DimT, typename...Ts, typename F, typename IdxT, typename...Args>
auto slide(const basic_tensor<Ts...>& t, const DimT& axis, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::template slide<ResultT>(t, axis, f, window_size, window_step,std::forward<Args>(args)...);
}
template<typename ResultT, typename F, typename...Ts, typename IdxT, typename...Args>
auto slide(const basic_tensor<Ts...>& t, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::template slide<ResultT>(t, f, window_size, window_step,std::forward<Args>(args)...);
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