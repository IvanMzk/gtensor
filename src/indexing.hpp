#ifndef INDEXING_HPP_
#define INDEXING_HPP_

#include "module_selector.hpp"
#include "exception.hpp"
#include "math.hpp"
#include "data_accessor.hpp"
#include "iterator.hpp"

namespace gtensor{

namespace detail{

template<typename IdxT, typename DimT, typename Axis>
void check_take_args(const IdxT& input_size, const DimT& input_dim, const IdxT& indexes_size, const Axis& axis_){
    if constexpr (!std::is_same_v<Axis,no_value>){
        auto axis = make_axis(input_dim,axis_);
        if (input_dim==0){
            if (axis != 0){
                throw axis_error("axis out of bounds");
            }
        }else{
            if (axis >= input_dim){
                throw axis_error("axis out of bounds");
            }
        }
    }
    if (input_size==0 && indexes_size!=0){
        throw value_error("cannot do a non-empty take from an empty input");
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

template<typename DimT, typename Axis>
void check_take_along_axis_args(const DimT& input_dim_, const DimT& indexes_dim, const Axis& axis_){
    static constexpr bool is_axis = !std::is_same_v<Axis,no_value>;
    const auto input_dim = is_axis ? input_dim_ : DimT{1};
    if (input_dim!=indexes_dim){
        throw value_error("t and indexes must have the same number of dimensions");
    }
    if constexpr (is_axis){
        auto axis = make_axis(input_dim,axis_);
        if (axis >= input_dim){
            throw axis_error("axis out of bounds");
        }
    }
}

template<typename ShT, typename IdxShT, typename DimT>
auto make_take_along_axis_shape(const ShT& shape, const IdxShT& indexes_shape, const DimT& axis){
    using shape_type = ShT;
    shape_type tmp(shape);
    tmp[axis] = indexes_shape[axis];
    return detail::make_broadcast_shape<shape_type>(tmp,indexes_shape);
}

// template<typename Axes>
// struct axis_type
// {
//     template<typename Dummy, typename B>
//     struct selector_{
//         using type = typename Axes::value_type;
//     };
//     template<typename Dummy>
//     struct selector_<Dummy,std::false_type>{
//         using type = Axes;
//     };
//     using type = typename selector_<void,std::bool_constant<detail::is_container_v<Axes>>>::type;
// };
// template<typename Axes> using axis_type_t = typename axis_type<Axes>::type;

// //Axes should be integral or container of integrals
// //Inverse should be like std::bool_constant
// template<typename Axes, typename Inverse>
// class traverse_predicate
// {
//     using axis_type = axis_type_t<Axes>;
//     static_assert(math::numeric_traits<axis_type>::is_integral(),"Axes must be container of integrals or integral");

//     const Axes* axes_;

//     template<typename U>
//     bool is_in_axes(const U& u_)const{
//         const auto& u = static_cast<const axis_type&>(u_);
//         if constexpr (detail::is_container_v<Axes>){
//             if (axes_->size()==0){
//                 return false;
//             }else{
//                 const auto last = axes_->end();
//                 return std::find_if(axes_->begin(),last,[u](const auto& e){return e == u;}) != last;
//             }
//         }else{
//             return u == *axes_;
//         }
//     }

//     bool apply_inverse(bool b)const{
//         if constexpr (Inverse::value){  //inverse of b
//             return !b;
//         }else{
//             return b;
//         }

//     }

// public:

//     template<typename Axes_> struct enable_ : std::conjunction<
//         std::is_lvalue_reference<Axes_>,
//         std::negation<std::is_same<std::remove_cv_t<std::remove_reference_t<Axes_>>, traverse_predicate>>
//     >{};

//     template<typename Axes_, std::enable_if_t<enable_<Axes_>::value,int> =0>
//     explicit traverse_predicate(Axes_&& axes__):
//         axes_{&axes__}
//     {}

//     const Axes& axes()const{return *axes_;}

//     template<typename U>
//     bool operator()(const U& u)const{
//         static_assert(math::numeric_traits<U>::is_integral(),"axis must be of integral type");
//         return apply_inverse(is_in_axes(u));
//     }
// };

// template<typename Inverse=std::false_type, typename Axes>
// auto make_traverse_predicate(Axes&& axes, Inverse inverse=Inverse{}){
//     (void)inverse;
//     using Axes_ = std::remove_cv_t<std::remove_reference_t<Axes>>;
//     return traverse_predicate<Axes_,Inverse>{std::forward<Axes>(axes)};
// }

// template<typename ShT, typename Walker, typename Axes, typename Inverse>
// auto make_forward_traverser(ShT&& shape, Walker&& walker, const traverse_predicate<Axes,Inverse>& predicate){
//     using Walker_ = std::remove_cv_t<std::remove_reference_t<Walker>>;
//     using config_type = typename Walker_::config_type;
//     static_assert(std::is_lvalue_reference_v<ShT>,"shape must outlive traverser");
//     using traverser_type = walker_forward_traverser<config_type, Walker_, traverse_predicate<Axes,Inverse>>;
//     return traverser_type{shape,std::forward<Walker>(walker),predicate};
// }

// template<typename Walker, typename DimT, typename IdxT>
// auto make_axis_iterator(Walker&& walker, const DimT& axis, const IdxT& pos){
//     using Walker_ = std::remove_cv_t<std::remove_reference_t<Walker>>;
//     using config_type = typename Walker_::config_type;
//     using iterator_type = axis_iterator<config_type,Walker_>;
//     return iterator_type{std::forward<Walker>(walker),axis,pos};
// }

// //make iterator to traverse along specified axes
// //Order can be c_order or f_order
// //shape,strides,axes must have lifetime longer then iterator, strides should be created with detail::make_strides_div_predicate routine
// //Walker argument must be created using t.create_walker() but may be not in its initial state e.g.: iterate over disjoint subsets of axes
// //pos should be zero for begin and size along axes (or its inverse) for end
// //predicate specifies which axes to traverse
// template<typename Order, typename ShT, typename StT, typename Walker, typename IdxT, typename Axes, typename Inverse>
// auto make_axes_iterator(ShT&& shape, StT&& strides, Walker&& walker, const IdxT& pos, const traverse_predicate<Axes,Inverse>& predicate){
//     using Walker_ = std::remove_cv_t<std::remove_reference_t<Walker>>;
//     using config_type = typename Walker_::config_type;
//     static_assert(std::is_lvalue_reference_v<ShT> && std::is_lvalue_reference_v<StT>,"shape and strides must outlive iterator");

//     if constexpr (detail::is_container_v<Axes> || Inverse::value){
//         using iterator_type = walker_iterator<config_type,Walker_,Order,traverse_predicate<Axes,Inverse>>;
//         return iterator_type{std::forward<Walker>(walker), shape, strides, pos, predicate};
//     }else{  //can use axis_iterator
//         return make_axis_iterator(std::forward<Walker>(walker),predicate.axes(),pos);
//     }
// }


//Axes is scalar or container
template<typename Config, typename Axes, typename Order>
class axes_iterator_maker{
    using config_type = Config;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using strides_div_type = strides_div_t<config_type>;
    using axes_container_type = typename config_type::template shape<dim_type>;
    //using axes_type = std::conditional_t<is_container_v<Axes>,axes_container_type,dim_type>;
    using axes_type = decltype(make_axes<config_type>(std::declval<dim_type>(),std::declval<Axes>()));
    using axes_map_type = decltype(make_range_traverser_axes_map<config_type>(std::declval<dim_type>(),std::declval<axes_type>()));

    const shape_type* shape_;
    axes_type axes_;
    axes_map_type axes_map_;
    shape_type traverse_shape_;
    strides_div_type traverse_strides_;

public:
    axes_iterator_maker(const shape_type& shape__, const Axes& axes__):
        shape_{&shape__},
        axes_{make_axes<config_type>(make_dim(shape__),axes__)},
        axes_map_{make_range_traverser_axes_map<config_type>(make_dim(shape__),axes_)},
        traverse_shape_{make_range_traverser_shape(*shape_,axes_map_)},
        traverse_strides_{make_range_traverser_strides_div<config_type>(traverse_shape_,axes_number(),Order{})}
    {}

    const auto& axes()const{
        return axes_;
    }

    template<typename Walker, typename Inverse=std::false_type>
    auto create_forward_traverser(Walker&& walker, Inverse inverse=Inverse{})const{
        using Walker_ = std::remove_cv_t<std::remove_reference_t<Walker>>;
        using walker_type = gtensor::mapping_axes_walker<Walker_>;
        using traverser_type = gtensor::walker_forward_range_traverser<config_type,walker_type>;
        if constexpr (inverse.value){   //traverse all but axes
            return traverser_type{traverse_shape_,walker_type{axes_map_,std::forward<Walker>(walker)},axes_number(),make_dim(shape_)};
        }else{  //traverse axes
            return traverser_type{traverse_shape_,walker_type{axes_map_,std::forward<Walker>(walker)},0,axes_number()};
        }
    }

    template<typename Walker, typename Inverse=std::false_type>
    auto begin(Walker&& walker, Inverse inverse=Inverse{})const{
        return create_walker_iterator(std::forward<Walker>(walker),0,inverse);
    }
    template<typename Walker, typename Inverse=std::false_type>
    auto end(Walker&& walker, Inverse inverse=Inverse{})const{
        return create_walker_iterator(std::forward<Walker>(walker),axes_size(inverse),inverse);
    }

private:

    dim_type axes_number()const{
        if constexpr (detail::is_container_v<axes_type>){
            return axes_.size();
        }else{
            return 1;
        }
    }

    template<typename Inverse>
    index_type axes_size(Inverse inverse)const{
        if constexpr (inverse.value){
            return std::accumulate(traverse_shape_.begin()+axes_number(),traverse_shape_.end(),index_type{1},std::multiplies<void>{});
        }else{
            return std::accumulate(traverse_shape_.begin(),traverse_shape_.begin()+axes_number(),index_type{1},std::multiplies<void>{});
        }
    }


    template<typename Walker, typename Inverse>
    auto create_walker_iterator(Walker&& walker, const index_type& pos, Inverse inverse)const{
        using Walker_ = std::remove_cv_t<std::remove_reference_t<Walker>>;
        using walker_type = gtensor::mapping_axes_walker<Walker_>;
        using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_range_traverser<config_type,walker_type>>,Order>;
        using iterator_type = gtensor::walker_iterator<config_type,traverser_type>;
        if constexpr (inverse.value){   //traverse all but axes
            return iterator_type{walker_type{axes_map_,std::forward<Walker>(walker)},traverse_shape_,traverse_strides_,pos,axes_number(),make_dim(traverse_shape_)};
        }else{  //traverse axes
            if constexpr (detail::is_container_v<axes_type>){
                return iterator_type{walker_type{axes_map_,std::forward<Walker>(walker)},traverse_shape_,traverse_strides_,pos,0,axes_number()};
            }else{  //axes scalar
                return gtensor::axis_iterator<config_type,Walker_>{std::forward<Walker>(walker),axes_,pos};
            }
        }
    }
};

template<typename Config, typename ShT, typename Axes, typename Order>
auto make_axes_iterator_maker(const ShT& shape, const Axes& axes_, Order){
    return axes_iterator_maker<Config,Axes,Order>{shape,axes_};
}

}   //end of namespace detail

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
    tensor<value_type,order,config_type> res(indexes.shape());
    if (!res.empty()){
        const auto size = t.size();
        auto indexer = t.traverse_order_adapter(FlattenOrder{}).create_indexer();
        auto indexes_it = indexes.traverse_order_adapter(order{}).begin();
        auto a_res = res.traverse_order_adapter(order{});
        for (auto res_it=a_res.begin(),res_last=a_res.end(); res_it!=res_last; ++res_it,++indexes_it){
            const auto& idx = static_cast<const index_type&>(*indexes_it);
            if (idx < size){
                *res_it = indexer[idx];
            }else{
                throw index_error("indexes is out of bounds");
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
    using axes_type = typename config_type::template shape<dim_type>;

    detail::check_take_args(t.size(),t.dim(),indexes.size(),axis_);
    if (t.dim()==1 || t.dim()==0){
        return take_flatten<order>(t,indexes);
    }
    if constexpr (std::is_same_v<Axis,gtensor::detail::no_value>){
        return take_flatten<config::c_order>(t,indexes);
    }else{
        const auto axis = detail::make_axis(t.dim(),axis_);
        tensor<value_type,order,config_type> res(detail::make_take_shape(t.shape(),indexes.shape(),axis));
        if (!res.empty()){
            const auto& input_shape = t.shape();
            const auto axis_size = input_shape[axis];
            const auto chunk_size = t.size() / axis_size;
            //auto input_predicate = detail::make_traverse_predicate(axis,std::true_type{});  //inverse, traverse all but axis
            //auto input_strides = detail::make_strides_div_predicate<config_type>(input_shape,input_predicate,order{});
            //auto input_walker = t.create_walker();

            auto input_iterator_maker = detail::make_axes_iterator_maker(input_shape,axis,order{});
            auto input_walker = t.create_walker();

            const auto indexes_dim = indexes.dim();
            axes_type res_axes(indexes_dim);
            std::iota(res_axes.begin(),res_axes.end(),axis);
            const auto& res_shape = res.shape();
            // auto res_predicate = detail::make_traverse_predicate(res_axes,std::true_type{});    //inverse, traverse all but res_axes
            // auto res_strides = detail::make_strides_div_predicate<config_type>(res_shape,res_predicate,order{});
            // auto res_traverser = detail::make_forward_traverser(res_shape,res.create_walker(),detail::make_traverse_predicate(res_axes,std::false_type{}));
            auto res_iterator_maker = detail::make_axes_iterator_maker(res_shape,res_axes,order{});
            auto res_traverser = res_iterator_maker.create_forward_traverser(res.create_walker(),std::false_type{});

            auto a_indexes = indexes.traverse_order_adapter(order{});
            for (auto indexes_it=a_indexes.begin(),indexes_last=a_indexes.end(); indexes_it!=indexes_last; ++indexes_it,res_traverser.template next<order>()){
                const auto& idx = static_cast<const index_type&>(*indexes_it);
                if (idx < axis_size){
                    input_walker.walk(axis,idx);
                    std::copy(
                        // detail::make_axes_iterator<order>(input_shape,input_strides,input_walker,0,input_predicate),
                        // detail::make_axes_iterator<order>(input_shape,input_strides,input_walker,chunk_size,input_predicate),
                        // detail::make_axes_iterator<order>(res_shape,res_strides,res_traverser.walker(),0,res_predicate)
                        input_iterator_maker.begin(input_walker,std::true_type{}),
                        input_iterator_maker.end(input_walker,std::true_type{}),
                        res_iterator_maker.begin(res_traverser.walker(),std::true_type{})
                    );
                    input_walker.walk(axis,-idx);
                }else{
                    throw index_error("indexes is out of bounds");
                }
            }
        }
        return res;
    }
}


// template<typename Axis=gtensor::detail::no_value, typename...Ts, typename...Us>
// static auto take(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes, const Axis& axis_=Axis{}){
//     using tensor_type = basic_tensor<Ts...>;
//     using order = typename tensor_type::order;
//     using value_type = typename tensor_type::value_type;
//     using config_type = typename tensor_type::config_type;
//     using dim_type = typename tensor_type::dim_type;
//     using index_type = typename tensor_type::index_type;
//     using axes_type = typename config_type::template container<dim_type>;

//     detail::check_take_args(t.size(),t.dim(),indexes.size(),axis_);
//     if (t.dim()==1 || t.dim()==0){
//         return take_flatten<order>(t,indexes);
//     }
//     if constexpr (std::is_same_v<Axis,gtensor::detail::no_value>){
//         return take_flatten<config::c_order>(t,indexes);
//     }else{
//         const auto axis = detail::make_axis(t.dim(),axis_);
//         tensor<value_type,order,config_type> res(detail::make_take_shape(t.shape(),indexes.shape(),axis));
//         if (!res.empty()){
//             const auto& input_shape = t.shape();
//             const auto axis_size = input_shape[axis];
//             const auto chunk_size = t.size() / axis_size;
//             auto input_predicate = detail::make_traverse_predicate(axis,std::true_type{});  //inverse, traverse all but axis
//             auto input_strides = detail::make_strides_div_predicate<config_type>(input_shape,input_predicate,order{});
//             auto input_walker = t.create_walker();

//             const auto indexes_dim = indexes.dim();
//             axes_type res_axes(indexes_dim);
//             std::iota(res_axes.begin(),res_axes.end(),axis);
//             const auto& res_shape = res.shape();
//             auto res_predicate = detail::make_traverse_predicate(res_axes,std::true_type{});    //inverse, traverse all but res_axes
//             auto res_strides = detail::make_strides_div_predicate<config_type>(res_shape,res_predicate,order{});
//             auto res_traverser = detail::make_forward_traverser(res_shape,res.create_walker(),detail::make_traverse_predicate(res_axes,std::false_type{}));
//             auto a_indexes = indexes.traverse_order_adapter(order{});
//             for (auto indexes_it=a_indexes.begin(),indexes_last=a_indexes.end(); indexes_it!=indexes_last; ++indexes_it,res_traverser.template next<order>()){
//                 const auto& idx = static_cast<const index_type&>(*indexes_it);
//                 if (idx < axis_size){
//                     input_walker.walk(axis,idx);
//                     std::copy(
//                         detail::make_axes_iterator<order>(input_shape,input_strides,input_walker,0,input_predicate),
//                         detail::make_axes_iterator<order>(input_shape,input_strides,input_walker,chunk_size,input_predicate),
//                         detail::make_axes_iterator<order>(res_shape,res_strides,res_traverser.walker(),0,res_predicate)
//                     );
//                     input_walker.walk(axis,-idx);
//                 }else{
//                     throw index_error("indexes is out of bounds");
//                 }
//             }
//         }
//         return res;
//     }
// }

// template<typename Axis=gtensor::detail::no_value, typename...Ts, typename...Us>
// static auto take_along_axis(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes, const Axis& axis_=Axis{}){
//     using tensor_type = basic_tensor<Ts...>;
//     using order = typename tensor_type::order;
//     using value_type = typename tensor_type::value_type;
//     using config_type = typename tensor_type::config_type;
//     using index_type = typename tensor_type::index_type;

//     detail::check_take_along_axis_args(t.dim(),indexes.dim(),axis_);
//     if (t.dim()==1 || t.dim()==0){
//         return take_flatten<order>(t,indexes);
//     }
//     if constexpr (std::is_same_v<Axis,gtensor::detail::no_value>){
//         return take_flatten<config::c_order>(t,indexes);
//     }else{
//         const auto axis = detail::make_axis(t.dim(),axis_);
//         const auto& input_shape = t.shape();
//         const auto shape = detail::make_take_along_axis_shape(input_shape,indexes.shape(),axis);
//         const auto input_axis_size = input_shape[axis];
//         const auto indexes_axis_size = indexes.shape()[axis];
//         tensor<value_type,order,config_type> res(shape);
//         if (!res.empty()){
//             const auto predicate = detail::make_traverse_predicate(axis,std::true_type{});  //inverse, traverse all but axis
//             auto input_traverser = detail::make_forward_traverser(shape,t.create_walker(),predicate);
//             auto indexes_traverser = detail::make_forward_traverser(shape,indexes.create_walker(),predicate);
//             auto res_traverser = detail::make_forward_traverser(shape,res.create_walker(),predicate);
//             do{
//                 auto indexes_it = detail::make_axis_iterator(indexes_traverser.walker(),axis,0);
//                 auto indexes_last = detail::make_axis_iterator(indexes_traverser.walker(),axis,indexes_axis_size);
//                 auto res_it = detail::make_axis_iterator(res_traverser.walker(),axis,0);
//                 auto input_walker = input_traverser.walker();
//                 for (;indexes_it!=indexes_last; ++indexes_it,++res_it){
//                     const auto& idx = static_cast<const index_type&>(*indexes_it);
//                     if (idx < input_axis_size){
//                         input_walker.walk(axis,idx);
//                         *res_it = *input_walker;
//                         input_walker.walk(axis,-idx);
//                     }else{
//                         throw index_error("indexes is out of bounds");
//                     }
//                 }
//                 input_traverser.template next<order>();
//                 indexes_traverser.template next<order>();
//             }while(res_traverser.template next<order>());
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

// //take values from the input array by matching 1d index and data slices
// template<typename DimT, typename...Ts, typename...Us>
// auto take_along_axis(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes, const DimT& axis){
//     using config_type = typename basic_tensor<Ts...>::config_type;
//     return indexing_selector_t<config_type>::take_along_axis(t,indexes,axis);
// }
// //take_along_axis like over flatten
// template<typename...Ts, typename...Us>
// auto take_along_axis(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes){
//     using config_type = typename basic_tensor<Ts...>::config_type;
//     return indexing_selector_t<config_type>::take_along_axis(t,indexes);
// }

}   //end of namespace gtensor
#endif