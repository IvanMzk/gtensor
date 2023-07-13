#ifndef INDEXING_HPP_
#define INDEXING_HPP_

#include "module_selector.hpp"
#include "tensor.hpp"
#include "builder.hpp"

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

}

//indexing module implementation

struct indexing{

private:

template<typename FlattenOrder, typename...Ts, typename...Us>
static auto take_flatten(const basic_tensor<Ts...>& t, const basic_tensor<Us...>& indexes){
    using tensor_type = basic_tensor<Ts...>;
    using order = typename tensor_type::order;
    using index_type = typename tensor_type::index_type;
    auto res = empty_like(t,indexes.shape());
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
    using config_type = typename tensor_type::config_type;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using input_predicate_type = detail::reduce_traverse_predicate<config_type, dim_type>;
    using input_walker_type = decltype(t.create_walker());
    using input_iterator_type = walker_iterator<config_type,input_walker_type,order,input_predicate_type>;
    //check args: axis vs t.dim , non empty take from empty input
    detail::check_take_args(t.size(),t.dim(),indexes.size(),axis_);
    auto a_indexes = indexes.template traverse_order_adapter<order>();
    if (t.dim()==1 || t.dim()==0){
        return take_flatten<order>(t,indexes);
    }
    if constexpr (std::is_same_v<Axis,gtensor::detail::no_value>){
        return take_flatten<config::c_order>(t,indexes);
    }else{
        const auto axis = detail::make_axis(t.dim(),axis_);
        auto res = empty_like(t,detail::make_take_shape(t.shape(),indexes.shape(),axis));
        if (!res.empty()){
            using axes_type = typename config_type::template container<dim_type>;
            using res_predicate_type = detail::reduce_traverse_predicate<config_type, axes_type>;
            using res_walker_type = decltype(res.create_walker());
            using res_traverser_type = walker_forward_traverser<config_type,res_walker_type,res_predicate_type>;
            using res_iterator_type = walker_iterator<config_type,res_walker_type,order,res_predicate_type>;
            const auto indexes_dim = indexes.dim();
            const auto& input_shape = t.shape();
            const auto axis_size = input_shape[axis];
            const auto chunk_size = t.size() / axis_size;
            input_predicate_type input_predicate{axis,true};    //inverse, to traverse all but axis
            axes_type res_axes(indexes_dim);
            std::iota(res_axes.begin(),res_axes.end(),axis);
            const auto& res_shape = res.shape();
            res_traverser_type res_traverser{res_shape,res.create_walker(),res_predicate_type{res_axes,false}};    //to traverse indexes axes
            auto input_walker = t.create_walker();
            for (auto indexes_it=a_indexes.begin(),indexes_last=a_indexes.end(); indexes_it!=indexes_last; ++indexes_it,res_traverser.template next<order>()){
                const auto& idx = static_cast<const index_type&>(*indexes_it);
                if (idx < axis_size){
                    input_walker.walk(axis,idx);
                    std::copy(
                        input_iterator_type{input_walker,input_shape,t.descriptor().strides_div(),0,input_predicate},
                        input_iterator_type{input_walker,input_shape,t.descriptor().strides_div(),chunk_size,input_predicate},
                        res_iterator_type{res_traverser.walker(),res_shape,res.descriptor().strides_div(),0,res_predicate_type{res_axes,true}}
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