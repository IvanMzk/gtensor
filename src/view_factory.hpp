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


//helpers for making index_mapping_view
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

template<typename ShT, typename SizeT>
inline ShT make_index_mapping_view_shape(const ShT& pshape, const ShT& subs_shape, const SizeT& subs_number){
    using shape_type = ShT;
    using size_type = SizeT;
    auto pdim = static_cast<size_type>(pshape.size());
    auto subs_dim = static_cast<size_type>(subs_shape.size());
    if (pdim > size_type{0}){
        if (subs_number > pdim){
            throw subscript_exception("invalid index tensor subscript");
        }
        auto res = shape_type(pdim - subs_number + subs_dim);
        std::copy(subs_shape.begin(), subs_shape.end(), res.begin());
        std::copy(pshape.begin()+subs_number, pshape.end(), res.begin()+subs_dim);
        return res;
    }else{
        return shape_type{};
    }
}

template<typename IdxT>
inline auto check_index(const IdxT& idx, const IdxT& shape_element){
    if (idx < shape_element){
        return idx;
    }else{
        throw subscript_exception("invalid index tensor subscript");
    }
}

template<typename It, typename ParentIndexer, typename ShT, typename...WalkerAdapter>
auto fill_index_mapping_view(It it, ParentIndexer pindexer, const ShT& pshape, const ShT& pstrides, const typename ShT::value_type& subs_size, WalkerAdapter...subs_it){
    using index_type = typename ShT::value_type;
    constexpr std::size_t subs_number = sizeof...(WalkerAdapter);

    const auto block_size = mapping_view_block_size(pshape, subs_number);
    const auto view_size = block_size*subs_size;
    if (view_size > index_type{0}){
        auto block_first = index_type{0};
        if (block_size == index_type{1}){
            for(index_type i{0}; i!=subs_size; ++i,((subs_it.next()),...)){
                block_first = index_type{0};
                auto n = std::size_t{0};
                ((block_first+=check_index(static_cast<index_type>(*subs_it.walker()),static_cast<index_type>(pshape[n]))*static_cast<index_type>(pstrides[n]),++n),...);
                *it = pindexer[block_first];
                ++it;
            }
        }else{
            index_type j{0};
            for(index_type i{0}; i!=subs_size; ++i,((subs_it.next()),...)){
                block_first = index_type{0};
                auto n = std::size_t{0};
                ((block_first+=check_index(static_cast<index_type>(*subs_it.walker()),static_cast<index_type>(pshape[n]))*static_cast<index_type>(pstrides[n]),++n),...);
                auto block_index_end = j+block_size;
                for(; j!=block_index_end; ++j){
                    *it = pindexer[block_first];
                    ++block_first;
                    ++it;
                }
            }
        }
    }
    return view_size;
}

//helpers for making bool_mapping_view
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

template<typename It, typename ParentIndexer, typename ShT, typename WalkerAdapter>
auto fill_bool_mapping_view(It it, ParentIndexer pindexer, const ShT& pstrides, const typename ShT::value_type& block_size, const typename ShT::value_type& subs_size, WalkerAdapter subs_it){
    using index_type = typename ShT::value_type;
    if (subs_size > index_type{0} && block_size > index_type{0}){
        index_type view_size{0};
        if (block_size == index_type{1}){
            do{
                if(*subs_it.walker()){
                    ++view_size;
                    index_type pindex = std::inner_product(subs_it.index().begin(), subs_it.index().end(), pstrides.begin(), index_type{0});
                    *it = pindexer[pindex];
                    ++it;
                }
            }while(subs_it.next());
        }else{
            do{
                if(*subs_it.walker()){
                    view_size+=block_size;
                    auto block_first = std::inner_product(subs_it.index().begin(), subs_it.index().end(), pstrides.begin(), index_type{0});
                    for(index_type i{0}; i!=block_size; ++i){
                        *it = pindexer[block_first+i];
                        ++it;
                    }
                }
            }while(subs_it.next());
        }
        return view_size;
    }else{
        return index_type{0};
    }
}


}   //end of namespace detail

template<typename ValT, typename CfgT>
class view_factory
{
    using size_type = typename CfgT::size_type;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using slices_collection_type = typename slice_traits<CfgT>::slices_collection_type;
    using view_reshape_descriptor_type = basic_descriptor<CfgT>;
    using view_subdim_descriptor_type = descriptor_with_offset<CfgT>;
    using view_slice_descriptor_type = converting_descriptor<CfgT>;
    template<typename EngineT> using view_reshape = gtensor::viewing_tensor<view_reshape_descriptor_type, EngineT>;
    template<typename EngineT> using view_subdim = gtensor::viewing_tensor<view_subdim_descriptor_type, EngineT>;
    template<typename EngineT> using view_slice = gtensor::viewing_tensor<view_slice_descriptor_type, EngineT>;

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
    static auto create_index_mapping_view(const std::shared_ptr<ImplT>& parent, const Subs&...subs){
        using impl_type = storage_tensor<typename detail::storage_engine_traits<typename CfgT::host_engine,CfgT,typename CfgT::template storage<ValT>>::type>;
        auto subs_shape = detail::broadcast_shape<shape_type>(subs.descriptor().shape()...);
        auto subs_size = detail::make_size(subs_shape);
        auto res = tensor<ValT,CfgT,impl_type>::make_tensor(detail::make_index_mapping_view_shape(parent->shape(), subs_shape, sizeof...(Subs)), ValT{});
        detail::fill_index_mapping_view(
            res.begin(),
            parent->engine().create_indexer(),
            parent->shape(),
            parent->strides(),
            subs_size,
            walker_bidirectional_adapter<CfgT, decltype(subs.engine().create_walker())>{subs_shape, subs.engine().create_walker()}...
        );
        return res;
    }
    template<typename ImplT, typename Subs>
    static auto create_bool_mapping_view(const std::shared_ptr<ImplT>& parent, const Subs& subs){
        using impl_type = storage_tensor<typename detail::storage_engine_traits<typename CfgT::host_engine,CfgT,typename CfgT::template storage<ValT>>::type>;
        detail::check_bool_mapping_view_subs(parent->shape(), subs.descriptor().shape());
        auto block_size = detail::bool_mapping_view_block_size(parent->shape(), subs);
        auto subs_size = subs.size();
        auto res = tensor<ValT,CfgT,impl_type>::make_tensor(parent->shape(), ValT{});
        auto view_size = detail::fill_bool_mapping_view(
            res.begin(),
            parent->engine().create_indexer(),
            parent->strides(),
            block_size,
            subs_size,
            walker_bidirectional_adapter<CfgT, decltype(subs.engine().create_walker())>{subs.descriptor().shape(), subs.engine().create_walker()}
        );
        res.impl()->resize(detail::make_bool_mapping_view_shape(parent->shape(), subs.descriptor().shape(), block_size, view_size));
        return res;

    }
};

}   //end of namespace gtensor



#endif