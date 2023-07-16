#ifndef INDEXING_HPP_
#define INDEXING_HPP_

#include "module_selector.hpp"
#include "math.hpp"
#include "data_accessor.hpp"
#include "iterator.hpp"

namespace gtensor{

class indexing_exception : public std::runtime_error
{
public:
    explicit indexing_exception(const char* what):
        std::runtime_error(what)
    {}
};

namespace detail{

template<typename IdxT, typename DimT, typename Axis>
void check_take_args(const IdxT& input_size, const DimT& input_dim, const IdxT& indexes_size, const Axis& axis_){
    if constexpr (!std::is_same_v<Axis,no_value>){
        auto axis = make_axis(input_dim,axis_);
        if (input_dim==0){
            if (axis != 0){
                throw indexing_exception("axis out of bounds");
            }
        }else{
            if (axis >= input_dim){
                throw indexing_exception("axis out of bounds");
            }
        }
    }
    if (input_size==0 && indexes_size!=0){
        throw indexing_exception("cannot do a non-empty take from an empty input");
    }
}

template<typename ShT, typename IdxShT, typename DimT>
auto make_take_shape(const ShT& shape, const IdxShT& indexes_shape, const DimT& axis){
    using shape_type = ShT;
    using dim_type = typename shape_type::difference_type;

    const auto dim = detail::make_dim(shape);
    const auto indexes_dim = static_cast<dim_type>(indexes_shape.size());
    const auto res_dim = dim==0 ? indexes_dim : dim - dim_type{1} + indexes_dim;
    shape_type res(res_dim);
    std::copy(shape.begin(),shape.begin()+axis,res.begin());
    std::copy(indexes_shape.begin(),indexes_shape.end(),res.begin()+axis);
    std::copy(shape.begin()+(axis+dim_type{1}),shape.end(),res.begin()+(axis+indexes_dim));
    return res;
}

template<typename Axes>
struct axis_type
{
    template<typename Dummy, typename B>
    struct selector_{
        using type = typename Axes::value_type;
    };
    template<typename Dummy>
    struct selector_<Dummy,std::false_type>{
        using type = Axes;
    };
    using type = typename selector_<void,std::bool_constant<detail::is_container_v<Axes>>>::type;
};
template<typename Axes> using axis_type_t = typename axis_type<Axes>::type;

//Axes should be integral or container of integrals
//Inverse should be like std::bool_constant
template<typename Axes, typename Inverse>
class traverse_predicate
{
    using axis_type = axis_type_t<Axes>;
    static_assert(math::numeric_traits<axis_type>::is_integral(),"Axes must be container of integrals or integral");

    const Axes* axes_;

    template<typename U>
    bool is_in_axes(const U& u_)const{
        const auto& u = static_cast<const axis_type&>(u_);
        if constexpr (detail::is_container_v<Axes>){
            if (axes_->size()==0){
                return false;
            }else{
                const auto last = axes_->end();
                return std::find_if(axes_->begin(),last,[u](const auto& e){return e == u;}) != last;
            }
        }else{
            return u == *axes_;
        }
    }

    bool apply_inverse(bool b)const{
        if constexpr (Inverse::value){  //inverse of b
            return !b;
        }else{
            return b;
        }

    }

public:

    template<typename Axes_> struct enable_ : std::conjunction<
        std::is_lvalue_reference<Axes_>,
        std::negation<std::is_same<std::remove_cv_t<std::remove_reference_t<Axes_>>, traverse_predicate>>
    >{};

    template<typename Axes_, std::enable_if_t<enable_<Axes_>::value,int> =0>
    explicit traverse_predicate(Axes_&& axes__):
        axes_{&axes__}
    {}

    const Axes& axes()const{return *axes_;}

    template<typename U>
    bool operator()(const U& u)const{
        static_assert(math::numeric_traits<U>::is_integral(),"axis must be of integral type");
        return apply_inverse(is_in_axes(u));
    }
};

template<typename Inverse=std::false_type, typename Axes>
auto make_traverse_predicate(Axes&& axes, Inverse inverse=Inverse{}){
    (void)inverse;
    using Axes_ = std::remove_cv_t<std::remove_reference_t<Axes>>;
    return traverse_predicate<Axes_,Inverse>{std::forward<Axes>(axes)};
}

template<typename ShT, typename Walker, typename Axes, typename Inverse>
auto make_forward_traverser(ShT&& shape, Walker&& walker, const traverse_predicate<Axes,Inverse>& predicate){
    using Walker_ = std::remove_cv_t<std::remove_reference_t<Walker>>;
    using config_type = typename Walker_::config_type;
    static_assert(std::is_lvalue_reference_v<ShT>,"shape must outlive traverser");
    using traverser_type = walker_forward_traverser<config_type, Walker_, traverse_predicate<Axes,Inverse>>;
    return traverser_type{shape,std::forward<Walker>(walker),predicate};
}

template<typename Walker, typename DimT, typename IdxT>
auto make_axis_iterator(Walker&& walker, const DimT& axis, const IdxT& pos){
    using Walker_ = std::remove_cv_t<std::remove_reference_t<Walker>>;
    using config_type = typename Walker_::config_type;
    using iterator_type = axis_iterator<config_type,Walker_>;
    return iterator_type{std::forward<Walker>(walker),axis,pos};
}

template<typename Order, typename ShT, typename StT, typename Walker, typename IdxT, typename Axes, typename Inverse>
auto make_axes_iterator(ShT&& shape, StT&& strides, Walker&& walker, const IdxT& pos, const traverse_predicate<Axes,Inverse>& predicate){
    using Walker_ = std::remove_cv_t<std::remove_reference_t<Walker>>;
    using config_type = typename Walker_::config_type;
    static_assert(std::is_lvalue_reference_v<ShT> && std::is_lvalue_reference_v<StT>,"shape and strides must outlive iterator");

    if constexpr (detail::is_container_v<Axes> || Inverse::value){
        using iterator_type = walker_iterator<config_type,Walker_,Order,traverse_predicate<Axes,Inverse>>;
        return iterator_type{std::forward<Walker>(walker), shape, strides, pos, predicate};
    }else{  //can use axis_iterator
        return make_axis_iterator(std::forward<Walker>(walker),predicate.axes(),pos);
    }
}


// template<typename Config, typename ShT, typename Predicate, typename Order>
// auto make_strides_div_predicate(const ShT& shape, const Predicate& predicate, Order order){
//     using dim_type = typename ShT::difference_type;
//     const dim_type dim = detail::make_dim(shape);
//     ShT tmp{};
//     tmp.reserve(dim);
//     for (dim_type i{0}; i!=dim; ++i){
//         if (predicate(i)){
//             tmp.push_back(shape[i]);
//         }
//     }
//     return detail::make_strides_div<Config>(tmp, order);
// }



// template<typename Config, typename Axes>
// class traverse_predicate
// {
//     using config_type = Config;
//     using axes_type = Axes;
//     using dim_type = typename config_type::dim_type;
//     static_assert(detail::is_container_of_type_v<axes_type,dim_type> || std::is_convertible_v<axes_type,dim_type>);

//     const axes_type* axes_;
//     bool inverse_;

//     bool is_in_axes(const dim_type& d)const{
//         if constexpr (detail::is_container_of_type_v<axes_type,dim_type>){
//             if (axes_->size()==0){
//                 return false;
//             }else{
//                 const auto last = axes_->end();
//                 return std::find_if(axes_->begin(),last,[&d](const auto& dir){return d == static_cast<dim_type>(dir);}) != last;
//             }
//         }else{
//             return d == static_cast<dim_type>(*axes_);
//         }
//     }

//     bool apply_inverse(bool b)const{
//         return inverse_ != b;
//     }

// public:
//     template<typename Axes_, std::enable_if_t<std::is_lvalue_reference_v<Axes_>,int> =0>
//     traverse_predicate(Axes_&& axes__, bool inverse__):
//         axes_{&axes__},
//         inverse_{inverse__}
//     {}

//     bool operator()(const dim_type& d)const{
//         return apply_inverse(is_in_axes(d));
//     }
// };

// //make iterator to traverse along specified axes
// //Order can be c_order or f_order
// //Walker argument must be created using t.create_walker() but may be not in its initial state e.g.: iterate over disjoint subsets of axes
// //Axes can be scalar or container
// //if axes_inverse is true traverse along axes that is not in axes argument
// //pos should be zero for begin and size along axes (or its inverse) for end
// template<typename Order, typename Walker, typename Tensor, typename Axes, typename IdxT>
// auto make_walker_axes_iterator(Walker&& walker, Tensor& t, Axes&& axes, const IdxT& pos, bool axes_inverse=false){
//     using config_type = typename Tensor::config_type;
//     using dim_type = typename Tensor::dim_type;
//     using Axes_ = std::remove_cv_t<std::remove_reference_t<Axes>>;
//     using Walker_ = std::remove_cv_t<std::remove_reference_t<Walker>>;
//     static constexpr bool is_axes_container = detail::is_container_of_type_v<Axes_,dim_type>;
//     static_assert(is_axes_container || std::is_convertible_v<Axes_,dim_type>);
//     if constexpr (is_axes_container){
//         using predicate_type = traverse_predicate<config_type,Axes_>;
//         using iterator_type = walker_iterator<config_type,Walker_,Order,predicate_type>;
//         //return iterator_type{std::forward<Walker>(walker), t.shape(), detail::make_strides_div_predicate(), pos, predicate}
//     }else{

//     }
// }
// //the same as above but use t.create_walker() for walker argument
// template<typename Order, typename Tensor, typename Axes, typename IdxT>
// auto make_walker_axes_iterator(Tensor& t, Axes&& axes, const IdxT& pos, bool axes_inverse=false){
//     return make_walker_axes_iterator(t,std::forward<Axes>(axes),pos,axes_inverse);
// }

// template<typename Walker, typename DimT, typename IdxT>
// auto make_axis_iterator(Walker&& walker, const DimT& axis, const IdxT& pos){
//     using config_type = typename Walker::config_type;
//     using iterator_type = axis_iterator<config_type,Walker>;
//     return iterator_type{walker,axis,pos};
// }

//strides should be created using make_strides_div_predicate routine
//shape,strides,axes must have lifetime longer then iterator
//axes can be scalar or container
//if inverse_axes is true traverse along axes that is not in axes argument
//pos should be zero for begin and size along axes (or its inverse) for end
// template<typename Order, typename ShT, typename StT, typename Walker, typename IdxT, typename Predicate>
// auto make_axes_iterator(ShT&& shape, StT&& strides, Walker&& walker, const IdxT& pos, const Predicate& predicate){
//     using config_type = typename Walker::config_type;
//     using dim_type = typename config_type::dim_type;
//     using Walker_ = std::remove_cv_t<std::remove_reference_t<Walker>>;
//     static_assert(std::is_lvalue_reference_v<ShT> && std::is_lvalue_reference_v<StT>,"shape and strides must outlive iterator");
//     static constexpr bool is_axes_container = detail::is_container_of_type_v<Axes_,dim_type>;
//     static_assert(is_axes_container || std::is_convertible_v<Axes_,dim_type>,"axes must be container or scalar");

//     if constexpr (is_axes_container){
//         using predicate_type = traverse_predicate<config_type,Axes_>;
//         using iterator_type = walker_iterator<config_type,Walker_,Order,predicate_type>;
//         return iterator_type{std::forward<Walker>(walker), shape, strides, pos, predicate_type{axes,inverse_axes}};
//     }else{
//         return make_axis_iterator(walker, axes, pos);
//     }
// }


}

//indexing module implementation

struct indexing{

private:

template<typename FlattenOrder, typename...Ts, typename...Us>
static auto take_flatten(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes){
    using tensor_type = basic_tensor<Ts...>;
    using order = typename tensor_type::order;
    using value_type = typename tensor_type::value_type;
    using config_type = typename tensor_type::config_type;
    using index_type = typename tensor_type::index_type;
    tensor<value_type, order, config_type> res(indexes.shape());
    if (!res.empty()){
        const auto size = t.size();
        auto indexer = t.template traverse_order_adapter<FlattenOrder>().create_indexer();
        auto indexes_it = indexes.template traverse_order_adapter<order>().begin();
        auto a_res = res.template traverse_order_adapter<order>();
        for (auto res_it=a_res.begin(),res_last=a_res.end(); res_it!=res_last; ++res_it,++indexes_it){
            const auto& idx = static_cast<const index_type&>(*indexes_it);
            if (idx < size){
                *res_it = indexer[idx];
            }else{
                throw indexing_exception("indexes is out of bounds");
            }
        }
    }
    return res;
}

public:

//take elements of tensor along axis
template<typename Axis=gtensor::detail::no_value, typename...Ts, typename...Us>
static auto take(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes, const Axis& axis_=Axis{}){
    using tensor_type = basic_tensor<Ts...>;
    using order = typename tensor_type::order;
    using value_type = typename tensor_type::value_type;
    using config_type = typename tensor_type::config_type;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using axes_type = typename config_type::template container<dim_type>;

    detail::check_take_args(t.size(),t.dim(),indexes.size(),axis_);
    if (t.dim()==1 || t.dim()==0){
        return take_flatten<order>(t,indexes);
    }
    if constexpr (std::is_same_v<Axis,gtensor::detail::no_value>){
        return take_flatten<config::c_order>(t,indexes);
    }else{
        const auto axis = detail::make_axis(t.dim(),axis_);
        tensor<value_type, order, config_type> res(detail::make_take_shape(t.shape(),indexes.shape(),axis));
        if (!res.empty()){
            const auto& input_shape = t.shape();
            const auto axis_size = input_shape[axis];
            const auto chunk_size = t.size() / axis_size;
            auto input_predicate = detail::make_traverse_predicate(axis,std::true_type{});  //inverse, traverse all but axis
            auto input_strides = detail::make_strides_div_predicate<config_type>(input_shape,input_predicate,order{});
            auto input_walker = t.create_walker();

            const auto indexes_dim = indexes.dim();
            axes_type res_axes(indexes_dim);
            std::iota(res_axes.begin(),res_axes.end(),axis);
            const auto& res_shape = res.shape();
            auto res_predicate = detail::make_traverse_predicate(res_axes,std::true_type{});    //inverse, traverse all but res_axes
            auto res_strides = detail::make_strides_div_predicate<config_type>(res_shape,res_predicate,order{});
            auto res_traverser = detail::make_forward_traverser(res_shape,res.create_walker(),detail::make_traverse_predicate(res_axes,std::false_type{}));
            auto a_indexes = indexes.template traverse_order_adapter<order>();
            for (auto indexes_it=a_indexes.begin(),indexes_last=a_indexes.end(); indexes_it!=indexes_last; ++indexes_it,res_traverser.template next<order>()){
                const auto& idx = static_cast<const index_type&>(*indexes_it);
                if (idx < axis_size){
                    input_walker.walk(axis,idx);
                    std::copy(
                        detail::make_axes_iterator<order>(input_shape,input_strides,input_walker,0,input_predicate),
                        detail::make_axes_iterator<order>(input_shape,input_strides,input_walker,chunk_size,input_predicate),
                        detail::make_axes_iterator<order>(res_shape,res_strides,res_traverser.walker(),0,res_predicate)
                    );
                    input_walker.walk(axis,-idx);
                }else{
                    throw indexing_exception("indexes is out of bounds");
                }
            }
        }
        return res;
    }
}


// //take elements of tensor along axis
// template<typename Axis=gtensor::detail::no_value, typename...Ts, typename...Us>
// static auto take(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes, const Axis& axis_=Axis{}){
//     using tensor_type = basic_tensor<Ts...>;
//     using order = typename tensor_type::order;
//     using config_type = typename tensor_type::config_type;
//     using dim_type = typename tensor_type::dim_type;
//     using index_type = typename tensor_type::index_type;
//     using input_predicate_type = detail::reduce_traverse_predicate<config_type, dim_type>;
//     using input_walker_type = decltype(t.create_walker());
//     using input_iterator_type = walker_iterator<config_type,input_walker_type,order,input_predicate_type>;
//     //check args: axis vs t.dim , non empty take from empty input
//     detail::check_take_args(t.size(),t.dim(),indexes.size(),axis_);
//     auto a_indexes = indexes.template traverse_order_adapter<order>();
//     if (t.dim()==1 || t.dim()==0){
//         return take_flatten<order>(t,indexes);
//     }
//     if constexpr (std::is_same_v<Axis,gtensor::detail::no_value>){
//         return take_flatten<config::c_order>(t,indexes);
//     }else{
//         const auto axis = detail::make_axis(t.dim(),axis_);
//         auto res = empty_like(t,detail::make_take_shape(t.shape(),indexes.shape(),axis));
//         if (!res.empty()){
//             using axes_type = typename config_type::template container<dim_type>;
//             using res_predicate_type = detail::reduce_traverse_predicate<config_type, axes_type>;
//             using res_walker_type = decltype(res.create_walker());
//             using res_traverser_type = walker_forward_traverser<config_type,res_walker_type,res_predicate_type>;
//             using res_iterator_type = walker_iterator<config_type,res_walker_type,order,res_predicate_type>;
//             const auto indexes_dim = indexes.dim();
//             const auto& input_shape = t.shape();
//             const auto axis_size = input_shape[axis];
//             const auto chunk_size = t.size() / axis_size;
//             input_predicate_type input_predicate{axis,true};    //inverse, to traverse all but axis
//             axes_type res_axes(indexes_dim);
//             std::iota(res_axes.begin(),res_axes.end(),axis);
//             const auto& res_shape = res.shape();
//             res_traverser_type res_traverser{res_shape,res.create_walker(),res_predicate_type{res_axes,false}};    //to traverse indexes axes
//             auto input_walker = t.create_walker();
//             for (auto indexes_it=a_indexes.begin(),indexes_last=a_indexes.end(); indexes_it!=indexes_last; ++indexes_it,res_traverser.template next<order>()){
//                 const auto& idx = static_cast<const index_type&>(*indexes_it);
//                 if (idx < axis_size){
//                     input_walker.walk(axis,idx);
//                     std::copy(
//                         input_iterator_type{input_walker,input_shape,t.descriptor().strides_div(),0,input_predicate},
//                         input_iterator_type{input_walker,input_shape,t.descriptor().strides_div(),chunk_size,input_predicate},
//                         res_iterator_type{res_traverser.walker(),res_shape,res.descriptor().strides_div(),0,res_predicate_type{res_axes,true}}
//                     );
//                     input_walker.walk(axis,-idx);
//                 }else{
//                     throw indexing_exception("indexes is out of bounds");
//                 }
//             }
//         }
//         return res;
//     }
// }

};

//indexing module frontend

//take elements of tensor along axis
template<typename DimT, typename...Ts, typename...Us>
auto take(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes, const DimT& axis){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return indexing_selector_t<config_type>::take(t,indexes,axis);
}
//take like over flatten
template<typename...Ts, typename...Us>
auto take(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return indexing_selector_t<config_type>::take(t,indexes);
}

}   //end of namespace gtensor
#endif