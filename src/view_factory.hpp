#ifndef VIEW_FACTORY_HPP_
#define VIEW_FACTORY_HPP_

#include "engine_traits.hpp"
#include "descriptor.hpp"
#include "broadcast.hpp"
#include "tensor.hpp"
#include "slice.hpp"


namespace gtensor{

namespace detail{

template<typename T>
inline auto make_view_shape_element(const T& sub){
    using difference_type = typename T::difference_type;
    difference_type step_ = sub.step > difference_type(0) ? sub.step : -sub.step;
    return sub.start == sub.stop ?
        typename difference_type(0) :
        sub.start < sub.stop ?
            (sub.stop - sub.start-difference_type(1))/step_ + difference_type(1) :
            (sub.start - sub.stop-difference_type(1))/step_ + difference_type(1);
}
/*make view slice shape*/
template<typename ShT, typename SubsT>
inline ShT make_view_slice_shape(const ShT& pshape, const SubsT& subs){
    ShT res{};
    res.reserve(pshape.size());
    std::for_each(subs.begin(), subs.end(), [&res](const auto& sub){res.push_back(make_view_shape_element(sub));});
    std::for_each(pshape.data()+subs.size(), pshape.data()+pshape.size(), [&res](const auto& elem){res.push_back(elem);});
    return res;
}
/*make view slice offset*/
template<typename ShT, typename SubsT>
inline typename ShT::value_type make_view_slice_offset(const ShT& pstrides, const SubsT& subs){
    using index_type = typename ShT::value_type;
    index_type res{0};
    std::for_each(subs.begin(), subs.end(), [&res, pstrides_it = pstrides.begin()](const auto& sub)mutable{res+=sub.start*(*pstrides_it); ++pstrides_it;});
    return res;
}
/*make view slice cstrides*/
template<typename ShT, typename SubsT>
inline ShT make_view_slice_cstrides(const ShT& pstrides, const SubsT& subs){
    using index_type = typename ShT::value_type;
    ShT res{};
    res.reserve(pstrides.size());
    auto pstrides_it = pstrides.begin();
    std::for_each(subs.begin(), subs.end(), [&res, &pstrides_it](const auto& sub){res.push_back(sub.step*(*pstrides_it)); ++pstrides_it;});
    std::for_each(pstrides_it, pstrides.end(), [&res](const auto& elem){res.push_back(elem);});
    return res;
}

/*make transposed
* indeces is positions of source shape elements in transposed shape
* if indeces is empty transposed shape is reverse of source
*/
template<typename ShT>
ShT transpose(const ShT& src, const ShT& indeces){
    ShT res{};
    res.reserve(src.size());
    if (indeces.empty()){
        res.assign(src.rbegin(), src.rend());
    }else{
        std::for_each(indeces.begin(), indeces.end(), [&src, &res](const auto& pos){res.push_back(src[pos]);});
    }
    return res;
}

/*make view subdim shape*/
template<typename ShT>
inline ShT make_view_subdim_shape(const ShT& pshape, const ShT& subs){
    if (subs.empty()){
        return pshape;
    }else{
        return ShT(pshape.data()+subs.size(), pshape.data()+pshape.size());
    }
}
/*make view subdim offset*/
template<typename ShT>
inline typename ShT::value_type make_view_subdim_offset(const ShT& pstrides, const ShT& subs){
    using index_type = typename ShT::value_type;
    return std::inner_product(subs.begin(),subs.end(),pstrides.begin(),index_type(0));
}

/*make view reshape shape*/
template<typename ShT>
inline ShT make_view_reshape_shape(const ShT& pshape, const ShT& subs){
    if (subs.empty()){
        return pshape;
    }else{
        return subs;
    }
}


/*
* make mapping view shape
*/
template<typename ShT>
inline ShT make_shape_index_tensor(const ShT& pshape, const ShT& index_shape, const typename ShT::value_type& index_dim){
    using shape_type = ShT;
    auto pdim = pshape.size();
    if (index_dim > pdim){
        throw subscript_exception("invalid index tensor subscript");
    }
    auto res = shape_type(pdim - index_dim + index_shape.size());
    std::copy(index_shape.begin(), index_shape.end(), res.begin());
    std::copy(pshape.begin()+index_dim, pshape.end(), res.begin()+index_shape.size());
    return res;
}
/*
* make mapping view map
* params should be tensors with indexes, shapes of tensors must broadcast
* MapT is type of map container with interface like std::vector and must be spesialized explicitly
*/
template<typename MapT, typename ShT, typename...It>
MapT make_map_index_tensor(const ShT& pshape, const ShT& pstrides, const typename ShT::value_type& index_size, It&&...index_iters){
    using map_type = MapT;
    using index_type = typename map_type::value_type;
    using shape_type = ShT;
    constexpr std::size_t index_dim = sizeof...(It);

    auto block_size = std::accumulate(pshape.begin()+index_dim,pshape.end(),index_type(1),std::multiplies<index_type>{});
    auto res = map_type(block_size*index_size, index_type{0});
    auto i = std::size_t{0};
    auto j = std::size_t{0};
    while(i!=index_size){
        auto n = std::size_t{0};
        auto block_first = index_type{0};
        ((block_first+=check_index(*index_iters,pshape[n])*pstrides[n],++n),...);
        auto block_end = j+block_size;
        while(j!=block_end){
            res[j] = block_first;
            ++j;
            ++block_first;
        }
        ++i;
        ((++index_iters),...);
    }
    return res;
}

template<typename IdxT>
inline auto check_index(const IdxT& idx, const IdxT& shape_element){
    if (idx < shape_element){
        return idx;
    }else{
        throw subscript_exception("invalid index tensor subscript");
    }
}

}   //end of namespace detail

// template<typename ValT, template<typename> typename Cfg>
// class view_factory_base{
//     using tensor_base_type = tensor_base<ValT,Cfg>;
//     using config_type = Cfg<ValT>;
//     using shape_type = typename config_type::shape_type;
//     using slices_collection_type = typename config_type::slices_collection_type;
// public:
//     virtual ~view_factory_base(){}
//     virtual std::shared_ptr<tensor_base_type> create_view_slice(const std::shared_ptr<tensor_base_type>&, const slices_collection_type&)const = 0;
//     virtual std::shared_ptr<tensor_base_type> create_view_transpose(const std::shared_ptr<tensor_base_type>&, const shape_type&)const = 0;
//     virtual std::shared_ptr<tensor_base_type> create_view_subdim(const std::shared_ptr<tensor_base_type>&, const shape_type&)const = 0;
//     virtual std::shared_ptr<tensor_base_type> create_view_reshape(const std::shared_ptr<tensor_base_type>&, const shape_type&)const = 0;

//     static std::shared_ptr<view_factory_base> create_factory();
// };

template<typename ValT, typename CfgT>
class view_factory
{
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using slices_collection_type = typename CfgT::slices_collection_type;
    using view_reshape_descriptor_type = descriptor_with_libdivide<CfgT>;
    using view_subdim_descriptor_type = descriptor_with_offset<CfgT>;
    using view_slice_descriptor_type = converting_descriptor<CfgT>;
    using mapping_view_descriptor_type = mapping_descriptor<CfgT>;
    template<typename EngineT> using view_reshape = gtensor::viewing_tensor<view_reshape_descriptor_type, EngineT>;
    template<typename EngineT> using view_subdim = gtensor::viewing_tensor<view_subdim_descriptor_type, EngineT>;
    template<typename EngineT> using view_slice = gtensor::viewing_tensor<view_slice_descriptor_type, EngineT>;
    template<typename EngineT> using mapping_view = gtensor::viewing_tensor<mapping_view_descriptor_type, EngineT>;

    static auto create_view_slice_descriptor(const shape_type& shape, const shape_type& strides, const slices_collection_type& subs){
        return view_slice_descriptor_type{
            detail::make_view_slice_shape(shape,subs),
            detail::make_view_slice_cstrides(strides,subs),
            detail::make_view_slice_offset(strides,subs)
        };
    }
    static auto create_view_transpose_descriptor(const shape_type& shape, const shape_type& strides, const shape_type& subs){
        return view_slice_descriptor_type{
            detail::transpose(shape,subs),
            detail::transpose(strides,subs),
            index_type(0)
        };
    }
    static auto create_view_subdim_descriptor(const shape_type& shape, const shape_type& strides, const shape_type& subs){
        return view_subdim_descriptor_type{
            detail::make_view_subdim_shape(shape,subs),
            detail::make_view_subdim_offset(strides,subs)
        };
    }
    static auto create_view_reshape_descriptor(const shape_type& shape, const shape_type& subs){
        return view_reshape_descriptor_type{
            detail::make_view_reshape_shape(shape,subs)
        };
    }
    template<typename...Subs>
    static auto create_mapping_view_descriptor_index_tensor(const shape_type& shape, const shape_type& strides, const Subs&...subs){
        auto index_shape = detail::broadcast_shape<shape_type>(subs.impl()->shape()...);
        return mapping_view_descriptor_type{
            index_shape,

        };
    }
public:
    template<typename ImplT>
    static auto create_view_slice(const std::shared_ptr<ImplT>& parent, const slices_collection_type& subs){
        using impl_type = view_slice<typename detail::viewing_engine_traits<typename CfgT::engine, ValT, CfgT, view_slice_descriptor_type, ImplT>::type>;
        return tensor<ValT,CfgT,impl_type>{std::make_shared<impl_type>(create_view_slice_descriptor(parent->shape(), parent->strides(), subs),parent)};
    }
    template<typename ImplT>
    static auto create_view_transpose(const std::shared_ptr<ImplT>& parent, const shape_type& subs){
        using impl_type = view_slice<typename detail::viewing_engine_traits<typename CfgT::engine, ValT, CfgT, view_slice_descriptor_type, ImplT>::type>;
        return tensor<ValT,CfgT,impl_type>{std::make_shared<impl_type>(create_view_transpose_descriptor(parent->shape(), parent->strides(), subs),parent)};
    }
    template<typename ImplT>
    static auto create_view_subdim(const std::shared_ptr<ImplT>& parent, const shape_type& subs){
        using impl_type = view_subdim<typename detail::viewing_engine_traits<typename CfgT::engine, ValT, CfgT, view_subdim_descriptor_type, ImplT>::type>;
        return tensor<ValT,CfgT,impl_type>{std::make_shared<impl_type>(create_view_subdim_descriptor(parent->shape(), parent->strides(), subs),parent)};
    }
    template<typename ImplT>
    static auto create_view_reshape(const std::shared_ptr<ImplT>& parent, const shape_type& subs){
        using impl_type = view_reshape<typename detail::viewing_engine_traits<typename CfgT::engine, ValT, CfgT, view_reshape_descriptor_type, ImplT>::type>;
        return tensor<ValT,CfgT,impl_type>{std::make_shared<impl_type>(create_view_reshape_descriptor(parent->shape(), subs),parent)};
    }
    template<typename ImplT, typename...Subs>
    static auto create_mapping_view_index_tensor(const std::shared_ptr<ImplT>& parent, const Subs&...subs){
        using impl_type = mapping_view<typename detail::viewing_engine_traits<typename CfgT::engine, ValT, CfgT, mapping_view_descriptor_type, ImplT>::type>;
    }
};

}   //end of namespace gtensor



#endif