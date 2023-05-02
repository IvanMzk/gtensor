#ifndef VIEW_FACTORY_HPP_
#define VIEW_FACTORY_HPP_

#include "tensor_factory.hpp"
#include "descriptor.hpp"
#include "broadcast.hpp"
#include "tensor.hpp"
#include "slice.hpp"
#include "common.hpp"
#include "type_selector.hpp"

namespace gtensor{

class subscript_exception : public std::runtime_error{
    public: subscript_exception(const char* what):runtime_error(what){}
};

namespace detail{
//slice view helpers
template<typename IdxT>
inline IdxT bound_low(const IdxT& min, const IdxT& i){
    return i<min ? min:i;
}
template<typename IdxT>
inline IdxT bound_high(const IdxT& max, const IdxT& i){
    return i>max ? max:i;
}
template<typename SliceT, typename IdxT>
inline IdxT make_slice_start(const IdxT& pshape_element, const SliceT& subs){
    using index_type = IdxT;
    const index_type zero_index{0};
    index_type start = subs.start();
    if (!subs.is_start()){
        return subs.step()>zero_index ? zero_index:pshape_element-index_type{1};  //negative corrected defaults
    }
    return start<zero_index ? pshape_element+start:start;   //negative correction
}
template<typename SliceT, typename IdxT>
inline IdxT make_slice_stop(const IdxT& pshape_element, const SliceT& subs){
    using index_type = IdxT;
    const index_type zero_index{0};
    index_type stop = subs.stop();
    if (!subs.is_stop()){
        return subs.step()>zero_index ? pshape_element:-index_type{1};  //negative corrected defaults
    }
    return stop<zero_index ? pshape_element+stop:stop;   //negative correction
}
template<typename SliceT, typename IdxT>
inline IdxT make_slice_view_shape_element(const IdxT& pshape_element, const SliceT& subs){
    using index_type = IdxT;
    const index_type zero_index{0};
    index_type step = subs.step();
    index_type d{0};
    const index_type start = make_slice_start(pshape_element, subs);
    const index_type stop = subs.is_reduce() ? start+index_type{1} : make_slice_stop(pshape_element, subs);
    if (step > zero_index){
        if (start >= pshape_element || stop<=zero_index){
            return zero_index;
        }else{
            d = bound_high(pshape_element,stop) - bound_low(zero_index,start);
        }
    }else{
        if (start < zero_index || stop>=pshape_element-index_type{1}){
            return zero_index;
        }else{
            d = bound_high(pshape_element-index_type{1},start) - bound_low(index_type{-1},stop);
        }
        step = -step;
    }
    return d<=zero_index ? zero_index:(d-index_type{1})/step+index_type{1};
}
template<typename SliceT, typename IdxT>
inline IdxT make_slice_view_cstride_element(const IdxT& pstride_element, const SliceT& subs){
    return pstride_element*subs.step();
}
template<typename ShT, typename Container>
inline typename ShT::size_type make_slice_view_dim(const ShT& pshape, const Container& subs){
    using dim_type = typename ShT::size_type;
    dim_type pdim = pshape.size();
    dim_type reduce_number = std::count_if(subs.begin(),subs.end(),[](const auto& subs_){return subs_.is_reduce();});
    return pdim - reduce_number;
}
template<typename ShT, typename SizeT, typename Container, typename ElementMaker>
inline ShT make_slice_view_shape_cstrides(const ShT& pshape_or_pstrides, const SizeT& res_dim, const Container& subs, ElementMaker element_maker){
    using index_type = typename ShT::value_type;
    ShT res(res_dim, index_type{});
    auto it = pshape_or_pstrides.begin();
    auto res_it = res.begin();
    for (auto subs_it = subs.begin(); subs_it!=subs.end(); ++subs_it,++it){
        const auto& subs_ = *subs_it;
        if (!subs_.is_reduce()){
            *res_it = element_maker(*it, subs_);
            ++res_it;
        }
    }
    for(;it!=pshape_or_pstrides.end();++it,++res_it){
        *res_it = *it;
    }
    return res;
}
template<typename ShT, typename SizeT, typename SliceT, typename ElementMaker>
inline ShT make_slice_view_shape_cstrides_direction(const ShT& pshape_pstrides, const SizeT& direction, const SliceT& subs, ElementMaker element_maker){
    using dim_type = SizeT;
    using index_type = typename ShT::value_type;
    if (subs.is_reduce()){
        const dim_type pdim = pshape_pstrides.size();
        const dim_type res_dim = pdim-1;
        ShT res(res_dim,index_type{});
        auto pshape_pstrides_it = pshape_pstrides.begin();
        const auto pshape_pstrides_direction_it = pshape_pstrides_it+direction;
        auto res_it = res.begin();
        for(;pshape_pstrides_it!=pshape_pstrides_direction_it;++pshape_pstrides_it,++res_it){
            *res_it=*pshape_pstrides_it;
        }
        for(++pshape_pstrides_it;pshape_pstrides_it!=pshape_pstrides.end();++pshape_pstrides_it,++res_it){
            *res_it=*pshape_pstrides_it;
        }
        return res;
    }else{
        ShT res{pshape_pstrides};
        res[direction] = element_maker(pshape_pstrides[direction], subs);
        return res;
    }
}
//check args
template<typename ShT, typename Container>
inline void check_slice_view_args(const ShT& pshape, const Container& subs){
    using dim_type = typename ShT::size_type;
    using index_type = typename ShT::value_type;
    const dim_type pdim = pshape.size();
    const dim_type subs_number = subs.size();
    if (subs_number > pdim){
        throw subscript_exception("invalid subscripts number");
    }
    auto pshape_it = pshape.begin();
    for (auto subs_it = subs.begin(); subs_it!=subs.end(); ++subs_it,++pshape_it){
        const auto& subs_ = *subs_it;
        if (subs_.is_reduce()){
            const auto& pshape_element = *pshape_it;
            const auto& start = make_slice_start(pshape_element, subs_);
            if (start<index_type{0} || start>=pshape_element){
                throw subscript_exception("invalid subscripts");
            }
        }
    }
}
//slice view shape
template<typename ShT, typename SizeT, typename Container>
inline ShT make_slice_view_shape(const ShT& pshape, const SizeT& res_dim, const Container& subs){
    return make_slice_view_shape_cstrides(pshape, res_dim, subs, [](const auto& pelement, const auto& subs_){return make_slice_view_shape_element(pelement,subs_);});
}
template<typename ShT, typename SizeT, typename SliceT>
inline ShT make_slice_view_shape_direction(const ShT& pshape, const SizeT& direction, const SliceT& subs){
    return make_slice_view_shape_cstrides_direction(pshape, direction, subs, [](const auto& pelement, const auto& subs_){return make_slice_view_shape_element(pelement,subs_);});
}
//slice view cstrides
template<typename ShT, typename SizeT, typename Container>
inline ShT make_slice_view_cstrides(const ShT& pstrides, const SizeT& res_dim, const Container& subs){
    return make_slice_view_shape_cstrides(pstrides, res_dim, subs, [](const auto& pelement, const auto& subs_){return make_slice_view_cstride_element(pelement,subs_);});
}
template<typename ShT, typename SizeT, typename SliceT>
inline ShT make_slice_view_cstrides_direction(const ShT& pstrides, const SizeT& direction, const SliceT& subs){
    return make_slice_view_shape_cstrides_direction(pstrides, direction, subs, [](const auto& pelement, const auto& subs_){return make_slice_view_cstride_element(pelement,subs_);});
}
//slice view offset
template<typename ShT, typename Container>
inline typename ShT::value_type make_slice_view_offset(const ShT& pshape, const ShT& pstrides, const Container& subs){
    using index_type = typename ShT::value_type;
    const index_type zero_index{0};
    index_type res{0};
    auto pshape_it = pshape.begin();
    auto pstrides_it = pstrides.begin();
    for (auto subs_it = subs.begin(); subs_it!=subs.end(); ++subs_it,++pshape_it,++pstrides_it){
        const auto& pshape_element = *pshape_it;
        const auto& subs_ = *subs_it;
        index_type start_ = make_slice_start(pshape_element, subs_);
        start_ = subs_.step()>zero_index ? bound_low(zero_index, start_):bound_high(pshape_element-index_type{1}, start_);
        res+=*pstrides_it*start_;
    }
    return res;
}
template<typename ShT, typename SizeT, typename SliceT>
inline typename ShT::value_type make_slice_view_offset_direction(const ShT& pshape, const ShT& pstrides, const SizeT& direction, const SliceT& subs){
    return pstrides[direction]*make_slice_start(pshape[direction],subs);
}
//transpose view helpers
template<typename ShT, typename Container>
ShT make_transpose_view_shape(const ShT& pshape, const Container& subs){
    using shape_type = ShT;
    using dim_type = typename shape_type::size_type;
    if (std::empty(subs)){
        return shape_type(pshape.rbegin(), pshape.rend());
    }else{
        shape_type res{};
        res.reserve(pshape.size());
        for(auto it=subs.begin(); it!=subs.end(); ++it){
            res.push_back(pshape[static_cast<dim_type>(*it)]);
        }
        return res;
    }
}
template<typename ShT, typename Container>
ShT make_view_transpose_strides(const ShT& pstrides, const Container& subs){
    return make_transpose_view_shape(pstrides, subs);
}
template<typename SizeT, typename Container>
inline void check_transpose_args(const SizeT& dim, const Container& subs){
    using dim_type = SizeT;
    if (!std::empty(subs)){
        const dim_type subs_number = subs.size();
        if (dim!=subs_number){
            throw subscript_exception("transpose must have no or dim subscripts");
        }
        auto subs_end = subs.end();
        auto subs_it = subs.begin();
        using sub_type = typename Container::value_type;
        while(subs_it!=subs_end){
            const auto& sub = *subs_it;
            if (sub < sub_type{0}){
                throw subscript_exception("invalid transpose argument");
            }
            if (static_cast<const dim_type&>(sub) >= dim){
                throw subscript_exception("invalid transpose argument");
            }
            ++subs_it;
            if (std::find(subs_it, subs_end, sub) != subs_end){
                throw subscript_exception("invalid transpose argument");
            }
        }
    }
}
template<typename...Subs>
inline void check_transpose_args_variadic(const Subs&...subs){
    auto is_less_zero = [](const auto& sub){
        using sub_type = std::remove_reference_t<decltype(sub)>;
        return sub < sub_type{0} ? true : false;
    };
    if ((is_less_zero(subs)||...)){
        throw subscript_exception("invalid transpose argument");
    }
}
//subdim view helpers
template<typename IdxT, typename Subs>
IdxT make_subdim_index(const IdxT& pshape_element, const Subs& subs){
    using index_type = IdxT;
    index_type subs_{subs};
    return subs_ < index_type{0} ? pshape_element+subs_:subs_;
}
template<typename ShT, typename SizeT>
inline ShT make_subdim_view_shape(const ShT& pshape, const SizeT& subs_number){
    return ShT(pshape.begin()+subs_number, pshape.end());
}
template<typename ShT, typename Container>
inline typename ShT::value_type make_subdim_view_offset(const ShT& pshape, const ShT& pstrides, const Container& subs){
    using index_type = typename ShT::value_type;
    index_type res{0};
    auto pstrides_it = pstrides.begin();
    auto pshape_it = pshape.begin();
    for (auto subs_it = subs.begin(); subs_it!=subs.end(); ++subs_it,++pshape_it,++pstrides_it){
        res+=*pstrides_it*make_subdim_index(*pshape_it,*subs_it);
    }
    return res;
}
template<typename ShT, typename Container>
inline void check_subdim_args(const ShT& pshape, const Container& subs){
    using index_type = typename ShT::value_type;
    using dim_type = typename ShT::size_type;
    const dim_type& subs_number = subs.size();
    const dim_type& pdim = pshape.size();
    if (subs_number > pdim){
        throw subscript_exception("subscripts number exceeds dim");
    }
    auto pshape_it = pshape.begin();
    for (auto subs_it = subs.begin(), subs_end = subs.end(); subs_it != subs_end; ++subs_it, ++pshape_it){
        const index_type& pshape_element = *pshape_it;
        const index_type& sub = make_subdim_index(pshape_element ,*subs_it);
        if (sub < index_type{0} || sub >= *pshape_it){
            throw subscript_exception("invalid subdim subscript");
        }
    }
}
//reshape view helpers
template<typename ShT, typename Container>
inline ShT make_reshape_view_shape(const ShT& pshape, const typename ShT::value_type& psize, const Container& subs){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    if (std::empty(subs)){
        return pshape;
    }else{
        shape_type res(subs.begin(), subs.end());
        index_type vsize{1};
        auto res_it = res.begin();
        auto res_end = res.end();
        for(;res_it!=res_end; ++res_it){
            const index_type& res_element = *res_it;
            if (res_element >= index_type{0}){
                vsize*=res_element;
            }else{
                break;
            }
        }
        if (res_it!=res_end){   //unknown dimension
            while(--res_end!=res_it){
                vsize*=*res_end;
            }
            *res_it = psize / vsize;
        }
        return res;
    }
}
template<typename IdxT, typename Container>
inline auto check_reshape_args(const IdxT& psize, const Container& subs){
    using index_type = IdxT;
    if (!std::empty(subs)){
        index_type vsize{1};
        bool new_direction{false};
        for(auto it=subs.begin(); it!=subs.end(); ++it){
            const index_type& sub = *it;
            if (sub < index_type{0}){
                if (new_direction){
                    throw subscript_exception("reshape arguments can only specify one unknown dimension");
                }
                new_direction = true;
            }else{
                vsize*=sub;
            }
        }
        if (new_direction){
            if (vsize == index_type{0} || psize % vsize != index_type{0}){
                throw subscript_exception("invalid reshape arguments");
            }
        }else if (psize != vsize){
            throw subscript_exception("invalid reshape arguments");
        }
    }
}
//index_mapping_view helpers
template<typename ShT, typename SizeT>
inline auto mapping_view_chunk_size(const ShT& pshape, const SizeT& subs_dim_or_subs_number){
    using index_type = typename ShT::value_type;
    return std::accumulate(pshape.begin()+subs_dim_or_subs_number,pshape.end(),index_type(1),std::multiplies<index_type>{});
}

//check subscripts number and directions sizes to be valid
//check of subscripts indeces defered to place where subscripts should be iterated (result fill or mapping descriptor making)
template<typename ShT, typename...ShTs>
inline void check_index_mapping_view_subs(const ShT& pshape, const ShTs&...subs_shapes){
    using dim_type = typename ShT::size_type;
    using index_type = typename ShT::value_type;
    auto pdim = static_cast<dim_type>(pshape.size());
    auto subs_number = static_cast<dim_type>(sizeof...(ShTs));
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
    using dim_type = SizeT;
    auto pdim = static_cast<dim_type>(pshape.size());
    auto subs_dim = static_cast<dim_type>(subs_shape.size());
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
    using dim_type = typename ShT::size_type;
    using index_type = typename ShT::value_type;

    const index_type subs_size = detail::make_size(subs_shape);
    if (subs_size != index_type{0}){
        const dim_type subs_number = sizeof...(SubsIt);
        const index_type chunk_size = mapping_view_chunk_size(pshape, subs_number);
        if (chunk_size == index_type{1}){
            do{
                index_type block_first{0};
                dim_type n{0};
                ((block_first+=check_index(static_cast<index_type>(*subs_it.walker()),pshape[n])*pstrides[n],++n),...);
                *res_it = pindexer[block_first];
                ++res_it;
            }while(((subs_it.next()),...));
        }else{
            do{
                index_type block_first{0};
                dim_type n{0};
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
    using dim_type = typename ShT::size_type;
    dim_type pdim = pshape.size();
    dim_type subs_dim = subs_shape.size();
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
    using dim_type = SizeT;
    dim_type pdim = pshape.size();
    auto res = shape_type(pdim - subs_dim + dim_type{1});
    auto res_it = res.begin();
    *res_it = subs_trues_number;
    ++res_it;
    std::copy(pshape.begin()+subs_dim, pshape.end(), res_it);
    return res;
}

template<typename ShT, typename ParentIndexer, typename ResIt, typename Subs, typename SubsIt>
auto fill_bool_mapping_view(const ShT& pshape, const ShT& pstrides, ParentIndexer pindexer, ResIt res_it, const Subs& subs, SubsIt subs_it){
    using index_type = typename ShT::value_type;
    using dim_type = typename ShT::size_type;

    index_type trues_number{0};
    if (!subs.empty()){
        dim_type subs_dim = subs.dim();
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

class view_factory
{
    template<typename CfgT> using view_reshape_descriptor = basic_descriptor<CfgT>;
    template<typename CfgT> using view_subdim_descriptor = descriptor_with_offset<CfgT>;
    template<typename CfgT> using slice_view_descriptor = converting_descriptor<CfgT>;
    template<typename CfgT> using view_transpose_descriptor = converting_descriptor<CfgT>;
    //subdim view
    template<typename...Ts, typename Container>
    static auto create_subdim_view_container(const tensor<Ts...>& parent, const Container& subs){
        using tensor_type = tensor<Ts...>;
        using impl_type = typename tensor_type::impl_type;
        using config_type = typename tensor_type::config_type;
        using dim_type = typename tensor_type::dim_type;
        using descriptor_type = view_subdim_descriptor<config_type>;
        auto parent_impl = parent.impl();
        const auto& pshape = parent_impl->shape();
        detail::check_subdim_args(pshape,subs);
        const dim_type subs_number = subs.size();
        return viewing_tensor_factory<config_type,descriptor_type,impl_type>::make(
            descriptor_type{
                detail::make_subdim_view_shape(pshape,subs_number),
                detail::make_subdim_view_offset(pshape,parent_impl->strides(),subs)
            },
            parent_impl
        );
    }
    template<typename...Ts, typename...Subs>
    static auto create_subdim_view_variadic(const tensor<Ts...>& parent, const Subs&...subs){
        using config_type = typename tensor<Ts...>::config_type;
        using index_type = typename config_type::index_type;
        return create_subdim_view_container(parent, typename config_type::template container<index_type>{subs...});
    }
    //reshape view
    template<typename...Ts, typename Container>
    static auto create_reshape_view_container(const tensor<Ts...>& parent, const Container& subs){
        using tensor_type = tensor<Ts...>;
        using impl_type = typename tensor_type::impl_type;
        using config_type = typename tensor_type::config_type;
        using descriptor_type = view_reshape_descriptor<config_type>;
        auto parent_impl = parent.impl();
        const auto& psize = parent_impl->size();
        detail::check_reshape_args(psize,subs);
        return viewing_tensor_factory<config_type,descriptor_type,impl_type>::make(
            descriptor_type{detail::make_reshape_view_shape(parent_impl->shape(), psize,subs)},
            parent_impl
        );
    }
    template<typename...Ts, typename...Subs>
    static auto create_reshape_view_variadic(const tensor<Ts...>& parent, const Subs&...subs){
        using config_type = typename tensor<Ts...>::config_type;
        using index_type = typename config_type::index_type;
        return create_reshape_view_container(parent, typename config_type::template container<index_type>{subs...});
    }
    //transpose view
    template<typename...Ts, typename Container>
    static auto create_transpose_view_container(const tensor<Ts...>& parent, const Container& subs){
        using tensor_type = tensor<Ts...>;
        using impl_type = typename tensor_type::impl_type;
        using config_type = typename tensor_type::config_type;
        using index_type = typename tensor_type::index_type;
        using descriptor_type = view_transpose_descriptor<config_type>;
        auto parent_impl = parent.impl();
        detail::check_transpose_args(parent_impl->dim(),subs);
        return viewing_tensor_factory<config_type,descriptor_type,impl_type>::make(
            descriptor_type{
                detail::make_transpose_view_shape(parent_impl->shape(),subs),
                detail::make_view_transpose_strides(parent_impl->strides(),subs),
                index_type{0}
            },
            parent_impl
        );
    }
    template<typename...Ts, typename...Subs>
    static auto create_transpose_view_variadic(const tensor<Ts...>& parent, const Subs&...subs){
        using config_type = typename tensor<Ts...>::config_type;
        using dim_type = typename config_type::dim_type;
        detail::check_transpose_args_variadic(subs...);
        return create_transpose_view_container(parent, typename config_type::template container<dim_type>{static_cast<dim_type>(subs)...});
    }
    //slice view
    template<typename...Ts, typename Container>
    static auto create_slice_view_container(const tensor<Ts...>& parent, const Container& subs){
        using tensor_type = tensor<Ts...>;
        using impl_type = typename tensor_type::impl_type;
        using config_type = typename tensor_type::config_type;
        using dim_type = typename tensor_type::dim_type;
        using slice_type = typename tensor_type::slice_type;
        using descriptor_type = slice_view_descriptor<config_type>;
        static_assert(std::is_same_v<typename Container::value_type,slice_type>);
        auto parent_impl = parent.impl();
        const auto& pshape = parent_impl->shape();
        detail::check_slice_view_args(pshape,subs);
        const auto& pstrides = parent_impl->strides();
        dim_type res_dim = detail::make_slice_view_dim(pshape, subs);
        return viewing_tensor_factory<config_type,descriptor_type,impl_type>::make(
            descriptor_type{
                detail::make_slice_view_shape(pshape,res_dim,subs),
                detail::make_slice_view_cstrides(pstrides,res_dim,subs),
                detail::make_slice_view_offset(pshape,pstrides,subs)
            },
            parent_impl
        );
    }
    template<typename...Ts>
    static auto create_slice_view_container(const tensor<Ts...>& parent, std::initializer_list<std::initializer_list<typename tensor<Ts...>::slice_item_type>> subs){
        using config_type = typename tensor<Ts...>::config_type;
        using slice_type = typename tensor<Ts...>::slice_type;
        return create_slice_view_container(parent, typename config_type::template container<slice_type>(subs.begin(),subs.end()));
    }
    template<typename...Ts,typename...Subs>
    static auto create_slice_view_variadic(const tensor<Ts...>& parent, const Subs&...subs){
        using tensor_type = tensor<Ts...>;
        using config_type = typename tensor_type::config_type;
        using slice_type = typename tensor_type::slice_type;
        using slice_item_type = typename slice_type::slice_item_type;
        using reduce_tag_type = typename slice_type::reduce_tag_type;
        using index_type = typename tensor_type::index_type;
        struct slice_maker{
            auto operator()(const index_type& start){return slice_type{start,reduce_tag_type{}};}
            auto operator()(std::initializer_list<slice_item_type> slice_init_list){return slice_type{slice_init_list};}
            const auto& operator()(const slice_type& slice){return slice;}
        };
        if constexpr((std::is_convertible_v<Subs,index_type>&&...)){    //can make subdim view
            return create_subdim_view_variadic(parent, subs...);
        }else{
            slice_maker maker{};
            return create_slice_view_container(parent, typename config_type::template container<slice_type>{maker(subs)...});
        }
    }
    //index mapping view
    template<typename...Ts, typename...Subs>
    static auto create_index_mapping_view_(const tensor<Ts...>& parent, const Subs&...subs){
        using tensor_type = tensor<Ts...>;
        using value_type = typename tensor_type::value_type;
        using config_type = typename tensor_type::config_type;
        using shape_type = typename tensor_type::shape_type;
        using dim_type = typename tensor_type::dim_type;
        auto parent_impl = parent.impl();
        const auto& pshape = parent_impl->shape();
        detail::check_index_mapping_view_subs(pshape, subs.descriptor().shape()...);
        auto subs_shape = detail::broadcast_shape<shape_type>(subs.descriptor().shape()...);
        dim_type subs_number = sizeof...(Subs);
        auto res = storage_tensor_factory<config_type,value_type>::make(detail::make_index_mapping_view_shape(pshape, subs_shape, subs_number), value_type{});
        if (!res.empty()){
            detail::fill_index_mapping_view(
                pshape,
                parent_impl->strides(),
                parent_impl->engine().create_indexer(),
                res.begin(),
                subs_shape,
                walker_forward_adapter<config_type, decltype(subs.engine().create_walker())>{subs_shape, subs.engine().create_walker()}...
            );
        }
        return res;
    }
    //bool mapping view
    template<typename...Ts, typename Subs>
    static auto create_bool_mapping_view_(const tensor<Ts...>& parent, const Subs& subs){
        using tensor_type = tensor<Ts...>;
        using value_type = typename tensor_type::value_type;
        using config_type = typename tensor_type::config_type;
        using index_type = typename tensor_type::index_type;
        auto parent_impl = parent.impl();
        const auto& pshape = parent_impl->shape();
        const auto& subs_shape = subs.shape();
        detail::check_bool_mapping_view_subs(pshape, subs_shape);
        auto res = gtensor::storage_tensor_factory<config_type,value_type>::make(pshape, value_type{});
        index_type subs_trues_number{0};
        if (!res.empty()){
            subs_trues_number = detail::fill_bool_mapping_view(
                pshape,
                parent_impl->descriptor().strides(),
                parent_impl->engine().create_indexer(),
                res.begin(),
                subs,
                walker_forward_adapter<config_type, decltype(subs.engine().create_walker())>{subs_shape, subs.engine().create_walker()}
            );
        }else{
            subs_trues_number = static_cast<index_type>(std::count(subs.begin(),subs.end(),true));
        }
        res.impl()->resize(detail::make_bool_mapping_view_shape(pshape, subs_trues_number, subs.dim()));
        return res;
    }

public:
    //view_factory interface
    //reshape view
    template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename tensor<Ts...>::index_type>,int> = 0>
    static auto create_reshape_view(const tensor<Ts...>& parent, const Container& subs){
        return create_reshape_view_container(parent, subs);
    }
    template<typename...Ts, typename...Subs, std::enable_if_t<(std::is_convertible_v<Subs, typename tensor<Ts...>::index_type>&&...),int> = 0>
    static auto create_reshape_view(const tensor<Ts...>& parent, const Subs&...subs){
        return create_reshape_view_variadic(parent, subs...);
    }
    //transpose view
    template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename tensor<Ts...>::dim_type>,int> = 0>
    static auto create_transpose_view(const tensor<Ts...>& parent, const Container& subs){
        return create_transpose_view_container(parent, subs);
    }
    template<typename...Ts, typename...Subs, std::enable_if_t<(std::is_convertible_v<Subs, typename tensor<Ts...>::dim_type>&&...),int> = 0>
    static auto create_transpose_view(const tensor<Ts...>& parent, const Subs&...subs){
        return create_transpose_view_variadic(parent, subs...);
    }
    //slice view
    template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename tensor<Ts...>::slice_type>,int> = 0>
    static auto create_slice_view(const tensor<Ts...>& parent, const Container& subs){
        return create_slice_view_container(parent, subs);
    }
    template<
        typename...Ts,
        typename...Subs,
        std::enable_if_t<((std::is_convertible_v<Subs, typename tensor<Ts...>::index_type> || std::is_convertible_v<Subs, typename tensor<Ts...>::slice_type>)&&...),int> = 0
    >
    static auto create_slice_view(const tensor<Ts...>& parent, const Subs&...subs){
        return create_slice_view_variadic(parent, subs...);
    }
    //index mapping view
    template<typename...Ts, typename...Subs>
    static auto create_index_mapping_view(const tensor<Ts...>& parent, const Subs&...subs){
        return create_index_mapping_view_(parent, subs...);
    }
    //bool mapping view
    template<typename...Ts, typename Subs>
    static auto create_bool_mapping_view(const tensor<Ts...>& parent, const Subs& subs){
        return create_bool_mapping_view_(parent, subs);
    }

};  //end of class view_factory

//view_factory module interface
//slice view
template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename tensor<Ts...>::slice_type>,int> = 0>
inline auto create_slice_view(const tensor<Ts...>& parent, const Container& subs){
    using config_type = typename tensor<Ts...>::config_type;
    return view_factory_selector<config_type>::type::create_slice_view(parent, subs);
}
template<
    typename...Ts,
    typename...Subs,
    std::enable_if_t<((std::is_convertible_v<Subs, typename tensor<Ts...>::index_type> || std::is_convertible_v<Subs, typename tensor<Ts...>::slice_type>)&&...),int> = 0
>
inline auto create_slice_view(const tensor<Ts...>& parent, const Subs&...subs){
    using config_type = typename tensor<Ts...>::config_type;
    return view_factory_selector<config_type>::type::create_slice_view(parent, subs...);
}
//reshape view
template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename tensor<Ts...>::index_type>,int> = 0>
inline auto create_reshape_view(const tensor<Ts...>& parent, const Container& subs){
    using config_type = typename tensor<Ts...>::config_type;
    return view_factory_selector<config_type>::type::create_reshape_view(parent, subs);
}
template<typename...Ts, typename...Subs, std::enable_if_t<(std::is_convertible_v<Subs, typename tensor<Ts...>::index_type>&&...),int> = 0>
inline auto create_reshape_view(const tensor<Ts...>& parent, const Subs&...subs){
    using config_type = typename tensor<Ts...>::config_type;
    return view_factory_selector<config_type>::type::create_reshape_view(parent, subs...);
}
//transpose view
template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename tensor<Ts...>::dim_type>,int> = 0>
inline auto create_transpose_view(const tensor<Ts...>& parent, const Container& subs){
    using config_type = typename tensor<Ts...>::config_type;
    return view_factory_selector<config_type>::type::create_transpose_view(parent, subs);
}
template<typename...Ts, typename...Subs, std::enable_if_t<(std::is_convertible_v<Subs, typename tensor<Ts...>::dim_type>&&...),int> = 0>
inline auto create_transpose_view(const tensor<Ts...>& parent, const Subs&...subs){
    using config_type = typename tensor<Ts...>::config_type;
    return view_factory_selector<config_type>::type::create_transpose_view(parent, subs...);
}
//index mapping view
template<typename...Ts, typename...Subs>
inline auto create_index_mapping_view(const tensor<Ts...>& parent, const Subs&...subs){
    using config_type = typename tensor<Ts...>::config_type;
    using index_type = typename config_type::index_type;
    static_assert((detail::is_tensor_of_type_v<Subs,index_type>&&...));
    return view_factory_selector<config_type>::type::create_index_mapping_view(parent, subs...);
}
//bool mapping view
template<typename...Ts, typename Subs>
inline auto create_bool_mapping_view(const tensor<Ts...>& parent, const Subs& subs){
    using config_type = typename tensor<Ts...>::config_type;
    static_assert(detail::is_bool_tensor_v<Subs>);
    return view_factory_selector<config_type>::type::create_bool_mapping_view(parent, subs);
}

}   //end of namespace gtensor



#endif