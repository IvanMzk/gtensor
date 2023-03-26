#ifndef VIEW_FACTORY_HPP_
#define VIEW_FACTORY_HPP_

#include "tensor_factory.hpp"
#include "descriptor.hpp"
#include "broadcast.hpp"
#include "tensor.hpp"
#include "slice.hpp"

namespace gtensor{

namespace detail{

template<typename SliceT>
inline auto make_view_slice_shape_element(const SliceT& subs){
    using index_type = typename SliceT::index_type;
    index_type step_ = subs.step > index_type(0) ? subs.step : -subs.step;
    return subs.start == subs.stop ?
        index_type(0) :
        subs.start < subs.stop ?
            (subs.stop - subs.start-index_type(1))/step_ + index_type(1) :
            (subs.start - subs.stop-index_type(1))/step_ + index_type(1);
}
/*make view slice shape*/
template<typename ShT, typename SliceT, typename SizeT>
inline ShT make_view_slice_shape(const ShT& pshape, const SliceT& subs, const SizeT& direction){
    ShT res{pshape};
    res[direction] = make_view_slice_shape_element(subs);
    return res;
}
template<typename ShT, typename SubsT>
inline ShT make_view_slice_shape(const ShT& pshape, const SubsT& subs){
    ShT res{};
    res.reserve(pshape.size());
    std::for_each(subs.begin(), subs.end(), [&res](const auto& sub){res.push_back(make_view_slice_shape_element(sub));});
    std::for_each(pshape.data()+subs.size(), pshape.data()+pshape.size(), [&res](const auto& elem){res.push_back(elem);});
    return res;
}
/*make view slice offset*/
template<typename ShT, typename SliceT, typename SizeT>
inline typename ShT::value_type make_view_slice_offset(const ShT& pstrides, const SliceT& subs, const SizeT& direction){
    return subs.start*pstrides[direction];
}
template<typename ShT, typename SubsT>
inline typename ShT::value_type make_view_slice_offset(const ShT& pstrides, const SubsT& subs){
    using index_type = typename ShT::value_type;
    index_type res{0};
    std::for_each(subs.begin(), subs.end(), [&res, pstrides_it = pstrides.begin()](const auto& subs_)mutable{res+=subs_.start*(*pstrides_it); ++pstrides_it;});
    return res;
}
/*make view slice cstrides*/
template<typename ShT, typename SliceT, typename SizeT>
inline ShT make_view_slice_cstrides(const ShT& pstrides, const SliceT& subs, const SizeT& direction){
    ShT res{pstrides};
    res[direction] = subs.step*pstrides[direction];
    return res;
}
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
template<typename ShT, typename...Indeces>
ShT transpose(const ShT& src, const Indeces&...indeces){
    ShT res{};
    res.reserve(src.size());
    if constexpr (sizeof...(Indeces) == 0){
        res.assign(src.rbegin(), src.rend());
    }else{
        (res.push_back(src[indeces]),...);
    }
    return res;
}

template<typename T>
inline void check_transpose_subs(const T&){}
template<typename SizeT, typename...Subs>
inline void check_transpose_subs(const SizeT& dim, const Subs&...subs){
    using size_type = SizeT;
    size_type subs_number = sizeof...(Subs);
    if (dim!=subs_number){
        throw subscript_exception("transpose must have no or dim subscripts");
    }
    std::array<bool, sizeof...(Subs)> check_buffer;
    check_buffer.fill(false);
    ([&subs_number, &check_buffer](const auto& sub){
        if (static_cast<size_type>(sub)>=subs_number || check_buffer[static_cast<std::size_t>(sub)]){
            throw subscript_exception("invalid transpose subscript");
        }else{
            check_buffer[static_cast<std::size_t>(sub)]=true;
        }
    }(subs),...);
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

template<typename ShT>
inline void check_subdim_subs(const ShT& shape, const ShT& subs){
    using index_type = typename ShT::value_type;
    if (subs.size() >= shape.size()){
        throw subscript_exception("subdim subscripts number must be less than dim");
    }
    const index_type zero_index(0);
    auto shape_it = shape.begin();
    for (auto subs_it = subs.begin(), subs_end = subs.end(); subs_it != subs_end; ++subs_it, ++shape_it){
        auto sub = *subs_it;
        if (sub < zero_index || sub >= *shape_it){
            throw subscript_exception("invalid subdim subscript");
        }
    }
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

template<typename IdxT>
inline void check_reshape_subs(const IdxT&){}
template<typename IdxT, typename...Subs>
inline void check_reshape_subs(const IdxT& size, const Subs&...subs){
    using index_type = IdxT;
    index_type vsize{1};
    ([&vsize](const auto& sub){vsize*=sub;}(subs),...);
    if (size != vsize){throw subscript_exception("invalid new shape; size of reshape view must be equal to size of its parent");}
}


//helpers for making index_mapping_view
template<typename ShT, typename SizeT>
inline auto mapping_view_chunk_size(const ShT& pshape, const SizeT& subs_dim_or_subs_number){
    using index_type = typename ShT::value_type;
    return std::accumulate(pshape.begin()+subs_dim_or_subs_number,pshape.end(),index_type(1),std::multiplies<index_type>{});
}

//check subscripts number and directions sizes to be valid
//check of subscripts indeces defered to place where subscripts should be iterated (result fill or mapping descriptor making)
template<typename ShT, typename...ShTs>
inline void check_index_mapping_view_subs(const ShT& pshape, const ShTs&...subs_shapes){
    using size_type = typename ShT::size_type;
    using index_type = typename ShT::value_type;
    auto pdim = static_cast<size_type>(pshape.size());
    auto subs_number = static_cast<size_type>(sizeof...(ShTs));
    //check subs number not exceed parent dim
    if (subs_number > pdim){
        throw subscript_exception("invalid index tensor subscript");
    }
    //check zero size parent direction not subscripted with not empty subs
    auto pshape_it = pshape.begin();
    bool exception_flag = false;
    if((((exception_flag=exception_flag||(*pshape_it == index_type{0} && detail::make_size(subs_shapes) != index_type{0})),++pshape_it,exception_flag)||...)){
        throw subscript_exception("invalid index tensor subscript");
    }
}

template<typename ShT, typename SizeT>
inline ShT make_index_mapping_view_shape(const ShT& pshape, const ShT& subs_shape, const SizeT& subs_number){
    using shape_type = ShT;
    using size_type = SizeT;
    auto pdim = static_cast<size_type>(pshape.size());
    auto subs_dim = static_cast<size_type>(subs_shape.size());
    auto res = shape_type(pdim - subs_number + subs_dim);
    std::copy(subs_shape.begin(), subs_shape.end(), res.begin());
    std::copy(pshape.begin()+subs_number, pshape.end(), res.begin()+subs_dim);
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

template<typename ShT, typename ParentIndexer, typename ResIt,  typename...SubsIt>
auto fill_index_mapping_view(const ShT& pshape, const ShT& pstrides, ParentIndexer pindexer, ResIt res_it, const ShT& subs_shape, SubsIt...subs_it){
    using size_type = typename ShT::size_type;
    using index_type = typename ShT::value_type;

    const index_type subs_size = detail::make_size(subs_shape);
    if (subs_size != index_type{0}){
        const size_type subs_number = sizeof...(SubsIt);
        const index_type chunk_size = mapping_view_chunk_size(pshape, subs_number);
        if (chunk_size == index_type{1}){
            do{
                index_type block_first{0};
                size_type n{0};
                ((block_first+=check_index(static_cast<index_type>(*subs_it.walker()),pshape[n])*pstrides[n],++n),...);
                *res_it = pindexer[block_first];
                ++res_it;
            }while(((subs_it.next()),...));
        }else{
            do{
                index_type block_first{0};
                size_type n{0};
                ((block_first+=check_index(static_cast<index_type>(*subs_it.walker()),pshape[n])*pstrides[n],++n),...);
                for(index_type i{0}; i!=chunk_size; ++i){
                    *res_it = pindexer[block_first+i];
                    ++res_it;
                }
            }while(((subs_it.next()),...));
        }
    }
}

//helpers for making bool_mapping_view
template<typename ShT>
auto check_bool_mapping_view_subs(const ShT& pshape, const ShT& subs_shape){
    using size_type = typename ShT::size_type;
    size_type pdim = pshape.size();
    size_type subs_dim = subs_shape.size();
    if (subs_dim > pdim){
        throw subscript_exception("invalid bool tensor subscript");
    }
    for (auto subs_shape_it = subs_shape.begin(), pshape_it = pshape.begin(); subs_shape_it!=subs_shape.end(); ++subs_shape_it, ++pshape_it){
        if (*subs_shape_it > *pshape_it){
            throw subscript_exception("invalid bool tensor subscript");
        }
    }
}

template<typename ShT, typename SizeT>
inline ShT make_bool_mapping_view_shape(const ShT& pshape, const typename ShT::value_type& subs_trues_number, const SizeT& subs_dim){
    using shape_type = ShT;
    using size_type = SizeT;
    size_type pdim = pshape.size();
    auto res = shape_type(pdim - subs_dim + size_type{1});
    auto res_it = res.begin();
    *res_it = subs_trues_number;
    ++res_it;
    std::copy(pshape.begin()+subs_dim, pshape.end(), res_it);
    return res;
}

template<typename ShT, typename ParentIndexer, typename ResIt, typename Subs, typename SubsIt>
auto fill_bool_mapping_view(const ShT& pshape, const ShT& pstrides, ParentIndexer pindexer, ResIt res_it, const Subs& subs, SubsIt subs_it){
    using index_type = typename ShT::value_type;
    using size_type = typename ShT::size_type;

    index_type trues_number{0};
    if (!subs.empty()){
        size_type subs_dim = subs.dim();
        index_type chunk_size = mapping_view_chunk_size(pshape, subs_dim);

        if (chunk_size == index_type{1}){
            do{
                if(*subs_it.walker()){
                    ++trues_number;
                    index_type pindex = std::inner_product(subs_it.index().begin(), subs_it.index().end(), pstrides.begin(), index_type{0});
                    *res_it = pindexer[pindex];
                    ++res_it;
                }
            }while(subs_it.next());
        }else{
            do{
                if(*subs_it.walker()){
                    ++trues_number;
                    auto block_first = std::inner_product(subs_it.index().begin(), subs_it.index().end(), pstrides.begin(), index_type{0});
                    for(index_type i{0}; i!=chunk_size; ++i){
                        *res_it = pindexer[block_first+i];
                        ++res_it;
                    }
                }
            }while(subs_it.next());
        }
    }
    return trues_number;
}

}   //end of namespace detail

template<typename ValT, typename CfgT>
class view_factory
{
    using size_type = typename CfgT::size_type;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using slice_type = typename slice_traits<CfgT>::slice_type;
    using slices_container_type = typename slice_traits<CfgT>::slices_container_type;
    using view_reshape_descriptor_type = basic_descriptor<CfgT>;
    using view_subdim_descriptor_type = descriptor_with_offset<CfgT>;
    using view_slice_descriptor_type = converting_descriptor<CfgT>;
    template<typename EngineT> using view_reshape = gtensor::viewing_tensor<view_reshape_descriptor_type, EngineT>;
    template<typename EngineT> using view_subdim = gtensor::viewing_tensor<view_subdim_descriptor_type, EngineT>;
    template<typename EngineT> using view_slice = gtensor::viewing_tensor<view_slice_descriptor_type, EngineT>;

    static auto create_view_slice_descriptor(const shape_type& shape, const shape_type& strides, const slice_type& subs, const size_type& direction){
        return view_slice_descriptor_type{
            detail::make_view_slice_shape(shape,subs,direction),
            detail::make_view_slice_cstrides(strides,subs,direction),
            detail::make_view_slice_offset(strides,subs,direction)
        };
    }
    static auto create_view_slice_descriptor(const shape_type& shape, const shape_type& strides, const slices_container_type& subs){
        return view_slice_descriptor_type{
            detail::make_view_slice_shape(shape,subs),
            detail::make_view_slice_cstrides(strides,subs),
            detail::make_view_slice_offset(strides,subs)
        };
    }
    template<typename...Subs>
    static auto create_view_transpose_descriptor(const shape_type& shape, const shape_type& strides, const Subs&...subs){
        return view_slice_descriptor_type{
            detail::transpose(shape,subs...),
            detail::transpose(strides,subs...),
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
    static auto create_view_slice(const std::shared_ptr<ImplT>& parent, const slice_type& subs, const size_type& direction){
        return viewing_tensor_factory<CfgT,view_slice_descriptor_type,ImplT>::make(create_view_slice_descriptor(parent->shape(), parent->strides(), subs, direction),parent);
    }
    template<typename ImplT>
    static auto create_view_slice(const std::shared_ptr<ImplT>& parent, const slices_container_type& subs){
        return viewing_tensor_factory<CfgT,view_slice_descriptor_type,ImplT>::make(create_view_slice_descriptor(parent->shape(), parent->strides(), subs),parent);
    }
    template<typename ImplT, typename...Subs>
    static auto create_view_transpose(const std::shared_ptr<ImplT>& parent, const Subs&...subs){
        return viewing_tensor_factory<CfgT,view_slice_descriptor_type,ImplT>::make(create_view_transpose_descriptor(parent->shape(), parent->strides(), subs...),parent);
    }
    template<typename ImplT>
    static auto create_view_subdim(const std::shared_ptr<ImplT>& parent, const shape_type& subs){
        return viewing_tensor_factory<CfgT,view_subdim_descriptor_type,ImplT>::make(create_view_subdim_descriptor(parent->shape(), parent->strides(), subs),parent);
    }
    template<typename ImplT>
    static auto create_view_reshape(const std::shared_ptr<ImplT>& parent, const shape_type& subs){
        return viewing_tensor_factory<CfgT,view_reshape_descriptor_type,ImplT>::make(create_view_reshape_descriptor(parent->shape(), subs),parent);
    }
    template<typename ImplT, typename...Subs>
    static auto create_index_mapping_view(const std::shared_ptr<ImplT>& parent, const Subs&...subs){
        const auto& pshape = parent->shape();
        detail::check_index_mapping_view_subs(pshape, subs.descriptor().shape()...);
        auto subs_shape = detail::broadcast_shape<shape_type>(subs.descriptor().shape()...);
        size_type subs_number = sizeof...(Subs);
        auto res = storage_tensor_factory<CfgT,ValT>::make(detail::make_index_mapping_view_shape(pshape, subs_shape, subs_number), ValT{});
        if (!res.empty()){
            detail::fill_index_mapping_view(
                pshape,
                parent->strides(),
                parent->engine().create_indexer(),
                res.begin(),
                subs_shape,
                walker_forward_adapter<CfgT, decltype(subs.engine().create_walker())>{subs_shape, subs.engine().create_walker()}...
            );
        }
        return res;
    }
    template<typename ImplT, typename Subs>
    static auto create_bool_mapping_view(const std::shared_ptr<ImplT>& parent, const Subs& subs){
        const auto& pshape = parent->shape();
        const auto& subs_shape = subs.shape();
        detail::check_bool_mapping_view_subs(pshape, subs_shape);
        auto res = gtensor::storage_tensor_factory<CfgT,ValT>::make(pshape, ValT{});
        index_type subs_trues_number{0};
        if (!res.empty()){
            subs_trues_number = detail::fill_bool_mapping_view(
                pshape,
                parent->descriptor().strides(),
                parent->engine().create_indexer(),
                res.begin(),
                subs,
                walker_forward_adapter<CfgT, decltype(subs.engine().create_walker())>{subs_shape, subs.engine().create_walker()}
            );
        }else{
            subs_trues_number = static_cast<index_type>(std::count(subs.begin(),subs.end(),true));
        }
        res.impl()->resize(detail::make_bool_mapping_view_shape(pshape, subs_trues_number, subs.dim()));
        return res;
    }
};

}   //end of namespace gtensor



#endif