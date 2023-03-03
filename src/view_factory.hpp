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
    using index_type = typename T::index_type;
    index_type step_ = sub.step > index_type(0) ? sub.step : -sub.step;
    return sub.start == sub.stop ?
        index_type(0) :
        sub.start < sub.stop ?
            (sub.stop - sub.start-index_type(1))/step_ + index_type(1) :
            (sub.start - sub.stop-index_type(1))/step_ + index_type(1);
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
inline auto mapping_view_block_size(const ShT& pshape, const typename ShT::value_type& subs_dim_or_subs_number){
    using index_type = typename ShT::value_type;
    auto pdim = pshape.size();
    if (subs_dim_or_subs_number > index_type{0} && pdim > index_type{0}){
        return std::accumulate(pshape.begin()+subs_dim_or_subs_number,pshape.end(),index_type(1),std::multiplies<index_type>{});
    }else{
        return index_type{0};
    }
}

template<typename ShT>
inline ShT make_shape_index_tensor(const ShT& pshape, const ShT& index_shape, const std::size_t& index_dim){
    using shape_type = ShT;
    auto pdim = static_cast<std::size_t>(pshape.size());
    if (index_dim > pdim){
        throw subscript_exception("invalid index tensor subscript");
    }
    auto res = shape_type(pdim - index_dim + index_shape.size());
    std::copy(index_shape.begin(), index_shape.end(), res.begin());
    std::copy(pshape.begin()+index_dim, pshape.end(), res.begin()+index_shape.size());
    return res;
}
template<typename ShT, typename It>
inline ShT make_shape_bool_tensor(const ShT& pshape, const ShT& index_shape, It&& index_begin, It&& index_end){
    using shape_type = ShT;
    auto pdim = pshape.size();
    auto index_dim = index_shape.size();
    if (index_dim > pdim){
        throw subscript_exception("invalid bool tensor subscript");
    }
    auto pshape_it = pshape.begin();
    for (auto index_shape_it = index_shape.begin(), index_shape_end = index_shape.end(); index_shape_it!=index_shape_end; ++index_shape_it, ++pshape_it){
        if (*index_shape_it > *pshape_it){
            throw subscript_exception("invalid bool tensor subscript");
        }
    }
    if (auto trues_number = std::count(index_begin, index_end, true)){
        auto res = shape_type(pdim - index_dim + 1);
        res[0] = trues_number;
        std::copy(pshape.begin()+index_dim, pshape.end(), res.begin()+1);
        return res;
    }else{
        return shape_type{};
    }
}
/*
* make mapping view map from index tensors
* MapT is type of map container with interface like std::vector and must be spesialized explicitly
*/
template<typename IdxT>
inline auto check_index(const IdxT& idx, const IdxT& shape_element){
    if (idx < shape_element){
        return idx;
    }else{
        throw subscript_exception("invalid index tensor subscript");
    }
}
template<typename MapT, typename ShT, typename...It>
MapT make_map_index_tensor(const ShT& pshape, const ShT& pstrides, const std::size_t& index_size, It&&...index_begin){
    using map_type = MapT;
    using index_type = typename map_type::value_type;
    constexpr std::size_t index_dim = sizeof...(It);

    auto block_size = mapping_view_block_size(pshape, index_dim);
    auto res = map_type(block_size*index_size, index_type{0});
    auto i = std::size_t{0};
    auto j = std::size_t{0};
    while(i!=index_size){
        auto n = std::size_t{0};
        auto block_first = index_type{0};
        ((block_first+=check_index(static_cast<index_type>(*index_begin),static_cast<index_type>(pshape[n]))*static_cast<index_type>(pstrides[n]),++n),...);
        auto block_end = j+block_size;
        while(j!=block_end){
            res[j] = block_first;
            ++j;
            ++block_first;
        }
        ++i;
        ((++index_begin),...);
    }
    return res;
}
/*
* make mapping view map from bool tensor
*/
template<typename MapT, typename ShT, typename It>
MapT make_map_bool_tensor(const ShT& pshape, const ShT& pstrides, const typename ShT::value_type& view_size, const typename ShT::value_type& index_dim, It&& index_begin, It&& index_end){
    using map_type = MapT;
    using index_type = typename map_type::value_type;
    auto block_size = mapping_view_block_size(pshape, index_dim);
    auto stride = pstrides[index_dim-1];
    auto res = map_type(view_size, index_type{0});
    auto i = std::size_t{0};
    auto j = std::size_t{0};
    while(index_begin != index_end){
        if (*index_begin){
            auto block_first = i*stride;
            auto block_end = j+block_size;
            while(j!=block_end){
                res[j] = block_first;
                ++j;
                ++block_first;
            }
        }
        ++i;
        ++index_begin;
    }
    return res;
}

template<typename ShT>
auto check_bool_mapping_view_subs(const ShT& pshape, const ShT& subs_shape){
    using index_type = typename ShT::value_type;
    index_type pdim = pshape.size();
    index_type subs_dim = subs_shape.size();
    if (pdim > index_type{0})
    {
        if (subs_dim > pdim){
            throw subscript_exception("invalid bool tensor subscript");
        }
        auto pshape_it = pshape.begin();
        for (auto subs_shape_it = subs_shape.begin(), subs_shape_end = subs_shape.end(); subs_shape_it!=subs_shape_end; ++subs_shape_it, ++pshape_it){
            if (*subs_shape_it > *pshape_it){
                throw subscript_exception("invalid bool tensor subscript");
            }
        }
    }
}

template<typename ShT, typename Subs>
inline auto bool_mapping_view_block_size(const ShT& pshape, const Subs& subs){
    return mapping_view_block_size(pshape,subs.dim());
}

template<typename MapT, typename ShT>
MapT make_bool_mapping_view_map(const ShT& pshape, typename ShT::value_type& block_size, typename ShT::value_type& subs_size){
    using index_type = typename ShT::value_type;
    index_type map_size = block_size*subs_size;
    return MapT(map_size);
}

template<typename MapT, typename ShT, typename WalkerAdapter>
auto fill_bool_mapping_view_map(MapT& map, const ShT& pstrides, const typename ShT::value_type& block_size, const typename ShT::value_type& subs_size, WalkerAdapter subs_it){
    using index_type = typename ShT::value_type;
    if (subs_size > index_type{0} && map.size() > index_type{0}){
        index_type map_size{0};
        if (block_size == 1){
            do{
                if(*subs_it.walker()){
                    map[map_size] = std::inner_product(subs_it.index().begin(), subs_it.index().end(), pstrides.begin(), index_type{0});
                    ++map_size;
                }
            }while(subs_it.next());
        }else{
            do{
                if(*subs_it.walker()){
                    auto block_first = std::inner_product(subs_it.index().begin(), subs_it.index().end(), pstrides.begin(), index_type{0});
                    for(index_type i{0}; i!=block_size; ++i){
                        map[map_size] = block_first+i;
                        ++map_size;
                    }
                }
            }while(subs_it.next());
        }
        map.resize(map_size);
        return map_size;
    }else{
        return index_type{0};
    }
}

template<typename ShT>
inline ShT make_bool_mapping_view_shape(const ShT& pshape, const ShT& subs_shape, const typename ShT::value_type& block_size, const typename ShT::value_type& filled_map_size){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    index_type pdim = pshape.size();
    index_type subs_dim = subs_shape.size();
    if (filled_map_size > index_type{0}){
        index_type trues_number = filled_map_size / block_size;
        auto res = shape_type(pdim - subs_dim + index_type{1});
        auto res_it = res.begin();
        *res_it = trues_number;
        ++res_it;
        std::copy(pshape.begin()+subs_dim, pshape.end(), res_it);
        return res;
    }else{
        return shape_type{};
    }
}


}   //end of namespace detail

template<typename ValT, typename CfgT>
class view_factory
{
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using slices_collection_type = typename slice_traits<CfgT>::slices_collection_type;
    using view_reshape_descriptor_type = basic_descriptor<CfgT>;
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
    static auto create_index_mapping_view_descriptor(const shape_type& shape, const shape_type& strides, const Subs&...subs){
        using map_type = typename mapping_view_descriptor_type::map_type;
        auto index_shape = detail::broadcast_shape<shape_type>(subs.shape()...);
        return mapping_view_descriptor_type{
            detail::make_shape_index_tensor(shape,index_shape,sizeof...(Subs)),
            detail::make_map_index_tensor<map_type>(shape,strides,detail::make_size(index_shape),subs.begin_broadcast(index_shape)...)
        };
    }
    template<typename Subs>
    static auto create_bool_mapping_view_descriptor(const shape_type& shape, const shape_type& strides, const Subs& subs){
        using map_type = typename mapping_view_descriptor_type::map_type;
        using walker_adapter_type = walker_iterator_adapter<CfgT, decltype(subs.engine().create_walker())>;
        auto subs_size = subs.size();
        auto block_size = detail::bool_mapping_view_block_size(shape, subs);
        auto map = detail::make_bool_mapping_view_map<map_type>(shape, block_size, subs_size);
        auto map_size = detail::fill_bool_mapping_view_map(map, strides, block_size, subs_size, walker_adapter_type{subs.descriptor().shape(),subs.descriptor().strides_div(), subs.engine().create_walker()});
        return mapping_view_descriptor_type{
            detail::make_bool_mapping_view_shape(shape, subs.descriptor().shape(), block_size, map_size),
            std::move(map)
        };
    }
    // template<typename Sub>
    // static auto create_mapping_view_descriptor_bool_tensor(const shape_type& shape, const shape_type& strides, const Sub& sub){
    //     using map_type = typename mapping_view_descriptor_type::map_type;
    //     auto view_shape = detail::make_shape_bool_tensor(shape,sub.shape(),sub.begin(),sub.end());
    //     auto view_size = detail::make_size(view_shape);
    //     return mapping_view_descriptor_type{
    //         std::move(view_shape),
    //         detail::make_map_bool_tensor<map_type>(shape,strides,view_size,sub.dim(),sub.begin(),sub.end())
    //     };
    // }
public:
    template<typename ImplT>
    static auto create_view_slice(const std::shared_ptr<ImplT>& parent, const slices_collection_type& subs){
        using impl_type = view_slice<typename detail::viewing_engine_traits<typename CfgT::host_engine, CfgT, view_slice_descriptor_type, ImplT>::type>;
        return tensor<ValT,CfgT,impl_type>::make_tensor(create_view_slice_descriptor(parent->shape(), parent->strides(), subs),parent);
    }
    template<typename ImplT>
    static auto create_view_transpose(const std::shared_ptr<ImplT>& parent, const shape_type& subs){
        using impl_type = view_slice<typename detail::viewing_engine_traits<typename CfgT::host_engine, CfgT, view_slice_descriptor_type, ImplT>::type>;
        return tensor<ValT,CfgT,impl_type>::make_tensor(create_view_transpose_descriptor(parent->shape(), parent->strides(), subs),parent);
    }
    template<typename ImplT>
    static auto create_view_subdim(const std::shared_ptr<ImplT>& parent, const shape_type& subs){
        using impl_type = view_subdim<typename detail::viewing_engine_traits<typename CfgT::host_engine, CfgT, view_subdim_descriptor_type, ImplT>::type>;
        return tensor<ValT,CfgT,impl_type>::make_tensor(create_view_subdim_descriptor(parent->shape(), parent->strides(), subs),parent);
    }
    template<typename ImplT>
    static auto create_view_reshape(const std::shared_ptr<ImplT>& parent, const shape_type& subs){
        using impl_type = view_reshape<typename detail::viewing_engine_traits<typename CfgT::host_engine, CfgT, view_reshape_descriptor_type, ImplT>::type>;
        return tensor<ValT,CfgT,impl_type>::make_tensor(create_view_reshape_descriptor(parent->shape(), subs),parent);
    }
    template<typename ImplT, typename...Subs>
    static auto create_mapping_view_index_tensor(const std::shared_ptr<ImplT>& parent, const Subs&...subs){
        using impl_type = mapping_view<typename detail::viewing_engine_traits<typename CfgT::host_engine, CfgT, mapping_view_descriptor_type, ImplT>::type>;
        return tensor<ValT,CfgT,impl_type>::make_tensor(create_index_mapping_view_descriptor(parent->shape(), parent->strides(), subs...),parent);
    }
    template<typename ImplT, typename Sub>
    static auto create_mapping_view_bool_tensor(const std::shared_ptr<ImplT>& parent, const Sub& sub){
        using impl_type = mapping_view<typename detail::viewing_engine_traits<typename CfgT::host_engine, CfgT, mapping_view_descriptor_type, ImplT>::type>;
        return tensor<ValT,CfgT,impl_type>::make_tensor(create_bool_mapping_view_descriptor(parent->shape(), parent->strides(), sub),parent);
    }
};

}   //end of namespace gtensor



#endif