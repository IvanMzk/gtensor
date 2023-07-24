#ifndef VIEW_FACTORY_HPP_
#define VIEW_FACTORY_HPP_

#include "module_selector.hpp"
#include "common.hpp"
#include "exception.hpp"
#include "slice.hpp"
#include "descriptor.hpp"
#include "tensor_factory.hpp"
#include "tensor_implementation.hpp"

namespace gtensor{

// class subscript_exception : public std::runtime_error{
//     public: subscript_exception(const char* what):runtime_error(what){}
// };

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
inline typename ShT::difference_type make_slice_view_dim(const ShT& pshape, const Container& subs){
    using dim_type = typename ShT::difference_type;
    dim_type pdim = detail::make_dim(pshape);
    dim_type reduce_number = std::count_if(subs.begin(),subs.end(),[](const auto& subs_){return subs_.is_reduce();});
    return pdim - reduce_number;
}
template<typename ShT, typename DimT, typename Container, typename ElementMaker>
inline ShT make_slice_view_shape_cstrides(const ShT& pshape_or_pstrides, const DimT& res_dim, const Container& subs, ElementMaker element_maker){
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
//check args
template<typename ShT, typename Container>
inline void check_slice_view_args(const ShT& pshape, const Container& subs){
    using dim_type = typename ShT::difference_type;
    using index_type = typename ShT::value_type;
    const dim_type pdim = detail::make_dim(pshape);
    const dim_type subs_number = subs.size();
    if (subs_number > pdim){
        throw index_error("invalid subscripts number");
    }
    auto pshape_it = pshape.begin();
    for (auto subs_it = subs.begin(); subs_it!=subs.end(); ++subs_it,++pshape_it){
        const auto& subs_ = *subs_it;
        if (subs_.is_reduce()){
            const auto& pshape_element = *pshape_it;
            const auto& start = make_slice_start(pshape_element, subs_);
            if (start<index_type{0} || start>=pshape_element){
                throw index_error("invalid subscripts");
            }
        }
    }
}
//slice view shape
template<typename ShT, typename DimT, typename Container>
inline ShT make_slice_view_shape(const ShT& pshape, const DimT& res_dim, const Container& subs){
    return make_slice_view_shape_cstrides(pshape, res_dim, subs, [](const auto& pelement, const auto& subs_){return make_slice_view_shape_element(pelement,subs_);});
}
//slice view cstrides
template<typename ShT, typename DimT, typename Container>
inline ShT make_slice_view_cstrides(const ShT& pstrides, const DimT& res_dim, const Container& subs){
    return make_slice_view_shape_cstrides(pstrides, res_dim, subs, [](const auto& pelement, const auto& subs_){return make_slice_view_cstride_element(pelement,subs_);});
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
template<typename ShT, typename DimT, typename SliceT>
inline typename ShT::value_type make_slice_view_offset_direction(const ShT& pshape, const ShT& pstrides, const DimT& direction, const SliceT& subs){
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
template<typename DimT, typename Container>
inline void check_transpose_args(const DimT& dim, const Container& subs){
    using dim_type = DimT;
    if (!std::empty(subs)){
        const dim_type subs_number = subs.size();
        if (dim!=subs_number){
            throw value_error("transpose must have no or dim subscripts");
        }
        auto subs_end = subs.end();
        auto subs_it = subs.begin();
        using sub_type = typename Container::value_type;
        while(subs_it!=subs_end){
            const auto& sub = *subs_it;
            if (sub < sub_type{0}){
                throw value_error("invalid transpose argument");
            }
            if (static_cast<const dim_type&>(sub) >= dim){
                throw value_error("invalid transpose argument");
            }
            ++subs_it;
            if (std::find(subs_it, subs_end, sub) != subs_end){
                throw value_error("invalid transpose argument");
            }
        }
    }
}
template<typename...Subs>
inline void check_transpose_args_variadic(const Subs&...subs){
    if constexpr (sizeof...(Subs) > 0){
        auto is_less_zero = [](const auto& sub){
            using sub_type = std::remove_reference_t<decltype(sub)>;
            return sub < sub_type{0} ? true : false;
        };
        if ((is_less_zero(subs)||...)){
            throw value_error("invalid transpose argument");
        }
    }
}
//subdim view helpers
template<typename IdxT, typename Subs>
IdxT make_subdim_index(const IdxT& pshape_element, const Subs& subs){
    using index_type = IdxT;
    index_type subs_{subs};
    return subs_ < index_type{0} ? pshape_element+subs_:subs_;
}
template<typename ShT, typename DimT>
inline ShT make_subdim_view_shape(const ShT& pshape, const DimT& subs_number){
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
    using dim_type = typename ShT::difference_type;
    const dim_type& subs_number = subs.size();
    const dim_type& pdim = detail::make_dim(pshape);
    if (subs_number > pdim){
        throw index_error("subscripts number exceeds dim");
    }
    auto pshape_it = pshape.begin();
    for (auto subs_it = subs.begin(), subs_end = subs.end(); subs_it != subs_end; ++subs_it, ++pshape_it){
        const index_type& pshape_element = *pshape_it;
        const index_type& sub = make_subdim_index(pshape_element ,*subs_it);
        if (sub < index_type{0} || sub >= *pshape_it){
            throw index_error("invalid subdim subscript");
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
                    throw value_error("reshape arguments can only specify one unknown dimension");
                }
                new_direction = true;
            }else{
                vsize*=sub;
            }
        }
        if (new_direction){
            if (vsize == index_type{0} || psize % vsize != index_type{0}){
                throw value_error("invalid reshape arguments");
            }
        }else if (psize != vsize){
            throw value_error("invalid reshape arguments");
        }
    }
}
//index_mapping_view helpers
template<typename ShT, typename DimT>
inline auto mapping_view_chunk_size(const ShT& pshape, const DimT& subs_dim_or_subs_number){
    using index_type = typename ShT::value_type;
    return std::accumulate(pshape.begin()+subs_dim_or_subs_number,pshape.end(),index_type(1),std::multiplies<index_type>{});
}

//check subscripts number and directions sizes to be valid
//check of subscripts indeces defered to place where subscripts should be iterated (result fill or mapping descriptor making)
template<typename ShT, typename...ShTs>
inline void check_index_mapping_view_subs_variadic(const ShT& pshape, const ShTs&...subs_shapes){
    using dim_type = typename ShT::difference_type;
    using index_type = typename ShT::value_type;
    const dim_type pdim = detail::make_dim(pshape);
    const dim_type subs_number = sizeof...(ShTs);
    if (pdim == dim_type{0}){
        throw index_error("can't subscript 0-dim tensor");
    }
    //check subs number not exceed parent dim
    if (subs_number > pdim){
        throw index_error("invalid subscripts number");
    }
    //check zero size parent direction not subscripted with not empty subs
    auto pshape_it = pshape.begin();
    bool exception_flag = false;
    if((((exception_flag=exception_flag||(*pshape_it == index_type{0} && detail::make_size(subs_shapes) != index_type{0})),++pshape_it,exception_flag)||...)){
        throw index_error("invalid index tensor subscript");
    }
}

template<typename ShT, typename Container>
inline void check_index_mapping_view_subs_container(const ShT& pshape, const Container& shapes){
    using dim_type = typename ShT::difference_type;
    using index_type = typename ShT::value_type;
    const dim_type pdim = detail::make_dim(pshape);
    const dim_type subs_number = shapes.size();
    if (pdim == dim_type{0}){
        throw index_error("can't subscript 0-dim tensor");
    }
    //check subs number not exceed parent dim
    if (subs_number > pdim){
        throw index_error("invalid subscripts number");
    }
    //check zero size parent direction not subscripted with not empty subs
    auto pshape_it = pshape.begin();
    for (auto shapes_it = shapes.begin(), shapes_last = shapes.end(); shapes_it!=shapes_last; ++shapes_it,++pshape_it){
        const auto& shape = unwrap_shape(*shapes_it);
        if (*pshape_it == index_type{0} && detail::make_size(shape) != index_type{0}){
            throw index_error("invalid index tensor subscript");

        }
    }
}

template<typename ShT, typename SizeT>
inline ShT make_index_mapping_view_shape(const ShT& pshape, const ShT& subs_shape, const SizeT& subs_number){
    using dim_type = typename ShT::difference_type;
    using shape_type = ShT;
    const dim_type pdim = detail::make_dim(pshape);
    const dim_type subs_dim = detail::make_dim(subs_shape);
    const dim_type subs_number_ = static_cast<dim_type>(subs_number);
    shape_type res(pdim - subs_number_ + subs_dim);
    std::copy(subs_shape.begin(), subs_shape.end(), res.begin());
    std::copy(pshape.begin()+subs_number_, pshape.end(), res.begin()+subs_dim);
    return res;
}

template<typename IdxT>
inline auto check_index(const IdxT& idx, const IdxT& shape_element){
    if (idx < shape_element){
        return idx;
    }else{
        throw index_error("invalid index tensor subscript");
    }
}

template<typename Order, typename ShT, typename IdxT, typename IndexMap,  typename...SubsIt>
auto fill_index_map(const ShT& pshape, const ShT& pstrides, const IdxT& subs_size, IndexMap& index_map, SubsIt...subs_traverser){
    using dim_type = typename ShT::difference_type;
    using index_type = typename ShT::value_type;

    const dim_type subs_number = sizeof...(SubsIt);
    const index_type chunk_size = mapping_view_chunk_size(pshape, subs_number);
    index_type i{0};
    if (chunk_size == index_type{1}){
        do{
            dim_type n{0};
            index_type chunk_first{0};
            ((chunk_first+=check_index(static_cast<index_type>(*subs_traverser.walker()),pshape[n])*pstrides[n],++n),...);
            index_map[i] = chunk_first;
            ++i;
        }while(((subs_traverser.template next<Order>()),...));
    }else{
        const index_type stride_step = pstrides[subs_number];
        const index_type map_step = subs_size;
        do{
            dim_type n{0};
            index_type chunk_first{0};
            ((chunk_first+=check_index(static_cast<index_type>(*subs_traverser.walker()),pshape[n])*pstrides[n],++n),...);
            if constexpr (std::is_same_v<Order, gtensor::config::c_order>){
                for(index_type chunk_last=chunk_first+chunk_size; chunk_first!=chunk_last; ++chunk_first){
                    index_map[i] = chunk_first;
                    ++i;
                }
            }else{
                for(index_type j{i}, j_last{i+chunk_size*map_step}; j!=j_last; j+=map_step,chunk_first+=stride_step){
                    index_map[j] = chunk_first;
                }
                ++i;
            }
        }while(((subs_traverser.template next<Order>()),...));
    }
}

template<typename Order, typename ShT, typename IdxT, typename IndexMap,  typename Container>
auto fill_index_map_container(const ShT& pshape, const ShT& pstrides, const IdxT& subs_size, IndexMap& index_map, Container& subs_traversers){
    using dim_type = typename ShT::difference_type;
    using index_type = typename ShT::value_type;

    const dim_type subs_number = static_cast<dim_type>(subs_traversers.size());
    const index_type chunk_size = mapping_view_chunk_size(pshape, subs_number);
    index_type i{0};

    auto pshape_it = pshape.begin();
    auto pstrides_it = pstrides.begin();
    auto traversers_it = subs_traversers.begin();
    auto traversers_last = subs_traversers.end();
    auto shape_element = *pshape_it;
    auto stride_element = *pstrides_it;
    auto& first_tr = *traversers_it;

    if (chunk_size == index_type{1}){
        do{
            index_map[i] = check_index(static_cast<index_type>(*first_tr.walker()), shape_element)*stride_element;
            ++i;
        }
        while (first_tr.template next<Order>());

        for (++traversers_it,++pshape_it,++pstrides_it; traversers_it!=traversers_last; ++traversers_it,++pshape_it,++pstrides_it){
            auto& tr = *traversers_it;
            stride_element = *pstrides_it;
            shape_element = *pshape_it;
            i = index_type{0};
            do{
                index_map[i] += check_index(static_cast<index_type>(*tr.walker()), shape_element)*stride_element;
                ++i;
            }
            while (tr.template next<Order>());
        }
    }else{
        const index_type stride_step = pstrides[subs_number];
        const index_type map_step = subs_size;
        do{
            index_type chunk_first = check_index(static_cast<index_type>(*first_tr.walker()), shape_element)*stride_element;
            if constexpr (std::is_same_v<Order, gtensor::config::c_order>){
                for(const index_type chunk_last = chunk_first+chunk_size; chunk_first!=chunk_last; ++chunk_first){
                    index_map[i] = chunk_first;
                    ++i;
                }
            }else{
                for(index_type j{i}, j_last{i+chunk_size*map_step}; j!=j_last; j+=map_step,chunk_first+=stride_step){
                    index_map[j] = chunk_first;
                }
                ++i;
            }
        }
        while (first_tr.template next<Order>());

        for (++traversers_it,++pshape_it,++pstrides_it; traversers_it!=traversers_last; ++traversers_it,++pshape_it,++pstrides_it){
            auto& tr = *traversers_it;
            stride_element = *pstrides_it;
            shape_element = *pshape_it;
            i = index_type{0};
            do{
                index_type chunk_first = check_index(static_cast<index_type>(*tr.walker()), shape_element)*stride_element;
                if constexpr (std::is_same_v<Order, gtensor::config::c_order>){
                    for(index_type i_last = i+chunk_size; i!=i_last; ++i){
                        index_map[i] += chunk_first;
                    }
                }else{
                    for(index_type j{i}, j_last{i+chunk_size*map_step}; j!=j_last; j+=map_step){
                        index_map[j] += chunk_first;
                    }
                    ++i;
                }
            }while (tr.template next<Order>());
        }
    }
}

//helpers for making bool_mapping_view
template<typename ShT>
auto check_bool_mapping_view_subs(const ShT& pshape, const ShT& subs_shape){
    using dim_type = typename ShT::difference_type;
    const dim_type pdim = detail::make_dim(pshape);
    const dim_type subs_dim = detail::make_dim(subs_shape);
    if (pdim == dim_type{0}){
        throw index_error("can't subscript 0-dim tensor");
    }
    if (subs_dim > pdim){
        throw index_error("invalid bool tensor subscript");
    }
    auto pshape_it = pshape.begin();
    for (auto subs_shape_it = subs_shape.begin(), subs_shape_last = subs_shape.end(); subs_shape_it!=subs_shape_last; ++subs_shape_it, ++pshape_it){
        if (*subs_shape_it > *pshape_it){
            throw index_error("invalid bool tensor subscript");
        }
    }
}

template<typename ShT, typename IdxT, typename DimT>
inline ShT make_bool_mapping_view_shape(const ShT& pshape, const IdxT& subs_trues_number, const DimT& subs_dim){
    using dim_type = typename ShT::difference_type;
    using index_type = typename ShT::value_type;
    using shape_type = ShT;
    const dim_type pdim = detail::make_dim(pshape);
    const dim_type subs_dim_ = static_cast<dim_type>(subs_dim);
    auto res = shape_type(pdim - subs_dim_ + dim_type{1});
    auto res_it = res.begin();
    *res_it = static_cast<index_type>(subs_trues_number);
    ++res_it;
    std::copy(pshape.begin()+subs_dim, pshape.end(), res_it);
    return res;
}

template<typename Order, typename ShT, typename DimT, typename IdxT, typename IndexContainer, typename SubsIt>
auto fill_bool_map(const ShT& pstrides, const DimT& subs_dim, const IdxT& chunk_size, IndexContainer& index_container, SubsIt subs_traverser){
    using index_type = typename ShT::value_type;
    index_type trues_number{0};
    if (chunk_size == index_type{1}){
        do{
            if(*subs_traverser.walker()){
                index_container.push_back(
                    std::inner_product(subs_traverser.index().begin(), subs_traverser.index().end(), pstrides.begin(), index_type{0})
                );
                ++trues_number;
            }
        }while(subs_traverser.template next<gtensor::config::c_order>());
    }else{
        //if subs_dim==pdim than chunk_size is 1 and if branch is executed
        const auto stride_step = pstrides[subs_dim];
        do{
            if(*subs_traverser.walker()){
                auto block_first = std::inner_product(subs_traverser.index().begin(), subs_traverser.index().end(), pstrides.begin(), index_type{0});
                if constexpr (std::is_same_v<Order, gtensor::config::c_order>){
                    for(index_type j{0}; j!=chunk_size; ++j){
                        index_container.push_back(block_first+j);
                    }
                }else{
                    for(index_type j{0}, j_last = chunk_size*stride_step; j!=j_last; j+=stride_step){
                        index_container.push_back(block_first+j);
                    }
                }
                ++trues_number;
            }
        }while(subs_traverser.template next<gtensor::config::c_order>());
    }
    return trues_number;
}

}   //end of namespace detail

class view_factory
{
    template<typename Config, typename Order> using subdim_view_descriptor = descriptor_with_offset<Config, Order>;
    template<typename Config, typename Order> using slice_view_descriptor = converting_descriptor<Config, Order>;
    template<typename Config, typename Order> using transpose_view_descriptor = converting_descriptor<Config, Order>;
    template<typename Config, typename Order> using mapping_view_descriptor = mapping_descriptor<Config, Order>;
    template<typename Config, typename Order, typename Parent> using reshape_view = tensor_implementation<reshape_view_core<Config,Order,Parent>>;
    template<typename Config, typename Order, typename Parent> using subdim_view = tensor_implementation<view_core<Config,subdim_view_descriptor<Config, Order>,Parent>>;
    template<typename Config, typename Order, typename Parent> using slice_view = tensor_implementation<view_core<Config,slice_view_descriptor<Config, Order>,Parent>>;
    template<typename Config, typename Order, typename Parent> using transpose_view = tensor_implementation<view_core<Config,transpose_view_descriptor<Config, Order>,Parent>>;
    template<typename Config, typename Order, typename Parent> using mapping_view = tensor_implementation<view_core<Config,mapping_view_descriptor<Config, Order>,Parent>>;

    //subdim view
    template<typename...Ts, typename Container>
    static auto create_subdim_view_container(const basic_tensor<Ts...>& parent, const Container& subs){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using dim_type = typename parent_type::dim_type;
        using descriptor_type = subdim_view_descriptor<config_type, order>;
        using view_type = subdim_view<config_type,order,parent_type>;
        const auto& pshape = parent.shape();
        detail::check_subdim_args(pshape,subs);
        const dim_type subs_number = subs.size();
        return std::make_shared<view_type>(
            descriptor_type{
                detail::make_subdim_view_shape(pshape,subs_number),
                detail::make_subdim_view_offset(pshape,parent.strides(),subs)
            },
            parent
        );
    }
    template<typename...Ts, typename...Subs>
    static auto create_subdim_view_variadic(const basic_tensor<Ts...>& parent, const Subs&...subs){
        using config_type = typename basic_tensor<Ts...>::config_type;
        using index_type = typename config_type::index_type;
        return create_subdim_view_container(parent, typename config_type::template container<index_type>{subs...});
    }
    //reshape view
    template<typename Order, typename...Ts, typename Container>
    static auto create_reshape_view_container(const basic_tensor<Ts...>& parent, const Container& subs){
        using parent_type = basic_tensor<Ts...>;
        using config_type = typename parent_type::config_type;
        using view_type = reshape_view<config_type,Order,parent_type>;
        const auto& psize = parent.size();
        detail::check_reshape_args(psize,subs);
        return std::make_shared<view_type>(
            detail::make_reshape_view_shape(parent.shape(),psize,subs),
            parent
        );
    }
    template<typename Order, typename...Ts, typename...Subs>
    static auto create_reshape_view_variadic(const basic_tensor<Ts...>& parent, const Subs&...subs){
        using config_type = typename basic_tensor<Ts...>::config_type;
        using index_type = typename config_type::index_type;
        return create_reshape_view_container<Order>(parent, typename config_type::template container<index_type>{subs...});
    }
    //transpose view
    template<typename...Ts, typename Container>
    static auto create_transpose_view_container(const basic_tensor<Ts...>& parent, const Container& subs){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using index_type = typename parent_type::index_type;
        using descriptor_type = transpose_view_descriptor<config_type, order>;
        using view_type = transpose_view<config_type,order,parent_type>;
        detail::check_transpose_args(parent.dim(),subs);
        return std::make_shared<view_type>(
            descriptor_type{
                detail::make_transpose_view_shape(parent.shape(),subs),
                detail::make_view_transpose_strides(parent.strides(),subs),
                index_type{0}
            },
            parent
        );
    }
    template<typename...Ts, typename...Subs>
    static auto create_transpose_view_variadic(const basic_tensor<Ts...>& parent, const Subs&...subs){
        using config_type = typename basic_tensor<Ts...>::config_type;
        using dim_type = typename config_type::dim_type;
        detail::check_transpose_args_variadic(subs...);
        return create_transpose_view_container(parent, typename config_type::template container<dim_type>{static_cast<dim_type>(subs)...});
    }
    //slice view
    template<typename...Ts, typename Container>
    static auto create_slice_view_container(const basic_tensor<Ts...>& parent, const Container& subs){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using dim_type = typename parent_type::dim_type;
        using slice_type = typename parent_type::slice_type;
        using descriptor_type = slice_view_descriptor<config_type, order>;
        using view_type = slice_view<config_type,order,parent_type>;
        static_assert(std::is_same_v<typename Container::value_type,slice_type>);
        const auto& pshape = parent.shape();
        detail::check_slice_view_args(pshape,subs);
        const auto& pstrides = parent.strides();
        dim_type res_dim = detail::make_slice_view_dim(pshape, subs);
        return std::make_shared<view_type>(
            descriptor_type{
                detail::make_slice_view_shape(pshape,res_dim,subs),
                detail::make_slice_view_cstrides(pstrides,res_dim,subs),
                detail::make_slice_view_offset(pshape,pstrides,subs)
            },
            parent
        );
    }
    template<typename...Ts>
    static auto create_slice_view_container(const basic_tensor<Ts...>& parent, std::initializer_list<std::initializer_list<typename basic_tensor<Ts...>::slice_item_type>> subs){
        using config_type = typename basic_tensor<Ts...>::config_type;
        using slice_type = typename basic_tensor<Ts...>::slice_type;
        return create_slice_view_container(parent, typename config_type::template container<slice_type>(subs.begin(),subs.end()));
    }
    template<typename...Ts,typename...Subs>
    static auto create_slice_view_variadic(const basic_tensor<Ts...>& parent, const Subs&...subs){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using slice_type = typename parent_type::slice_type;
        using slice_item_type = typename slice_type::slice_item_type;
        using reduce_tag_type = typename slice_type::reduce_tag_type;
        using index_type = typename parent_type::index_type;
        struct slice_maker{
            auto operator()(const index_type& start){return slice_type{start,reduce_tag_type{}};}
            auto operator()(std::initializer_list<slice_item_type> slice_init_list){return slice_type(slice_init_list);}
            const auto& operator()(const slice_type& slice){return slice;}
        };

        if constexpr(std::is_same_v<order,gtensor::config::c_order> && (std::is_convertible_v<Subs,index_type>&&...)){
            //can make subdim view, in c_order optimized descriptor, which uses only offset and no stride to convert index is used
            return create_subdim_view_variadic(parent, subs...);
        }else{
            slice_maker maker{};
            return create_slice_view_container(parent, typename config_type::template container<slice_type>{maker(subs)...});
        }
    }
    //index mapping view
    template<typename...Ts, typename...Subs>
    static auto create_index_mapping_view_variadic(const basic_tensor<Ts...>& parent, const Subs&...subs){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using index_map_type = typename config_type::index_map_type;
        using shape_type = typename parent_type::shape_type;
        using descriptor_type = mapping_descriptor<config_type,order>;
        using view_type = mapping_view<config_type,order,parent_type>;
        const auto& pshape = parent.shape();
        detail::check_index_mapping_view_subs_variadic(pshape, subs.shape()...);
        const auto subs_shape = detail::make_broadcast_shape<shape_type>(subs.shape()...);
        const auto subs_dim = detail::make_dim(subs_shape);
        const auto subs_size = detail::make_size(subs_shape);
        auto res_shape = detail::make_index_mapping_view_shape(pshape, subs_shape, sizeof...(Subs));
        auto res_size = detail::make_size(res_shape);
        index_map_type index_map(res_size);
        if (res_size!=0){
            detail::fill_index_map<order>(
                pshape,
                parent.strides(),
                subs_size,
                index_map,
                walker_forward_traverser<config_type, decltype(subs.create_walker())>{subs_shape, subs.create_walker(subs_dim)}...
            );
        }
        return std::make_shared<view_type>(
            descriptor_type{std::move(res_shape),std::move(index_map)},
            parent
        );
    }
    template<typename...Ts, typename Container>
    static auto create_index_mapping_view_container(const basic_tensor<Ts...>& parent, const Container& subs){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using index_map_type = typename config_type::index_map_type;
        using shape_type = typename parent_type::shape_type;
        using descriptor_type = mapping_descriptor<config_type,order>;
        using view_type = mapping_view<config_type,order,parent_type>;
        using subs_tensor_type = typename Container::value_type;
        using traverser_container_type = typename config_type::template container<walker_forward_traverser<config_type, decltype(std::declval<const subs_tensor_type&>().create_walker())>>;
        const auto& pshape = parent.shape();
        const auto shapes = detail::make_shapes_container(subs);
        detail::check_index_mapping_view_subs_container(pshape, shapes);
        const auto subs_shape = detail::make_broadcast_shape_container<shape_type>(shapes);
        const auto subs_dim = detail::make_dim(subs_shape);
        const auto subs_size = detail::make_size(subs_shape);
        const auto subs_number = subs.size();
        auto res_shape = detail::make_index_mapping_view_shape(pshape, subs_shape, subs_number);
        auto res_size = detail::make_size(res_shape);
        index_map_type index_map(res_size);
        if (res_size!=0){
            traverser_container_type subs_traversers{};
            if constexpr (detail::is_static_castable_v<typename Container::size_type, typename traverser_container_type::size_type>){
                subs_traversers.reserve(static_cast<typename traverser_container_type::size_type>(subs_number));
            }
            for (const auto& sub : subs){
                subs_traversers.emplace_back(subs_shape,sub.create_walker(subs_dim));
            }
            detail::fill_index_map_container<order>(
                pshape,
                parent.strides(),
                subs_size,
                index_map,
                subs_traversers
            );
        }
        return std::make_shared<view_type>(
            descriptor_type{std::move(res_shape),std::move(index_map)},
            parent
        );
    }
    //bool mapping view
    template<typename...Ts, typename Subs>
    static auto create_bool_mapping_view_(const basic_tensor<Ts...>& parent, const Subs& subs){
        using parent_type = basic_tensor<Ts...>;
        using order = typename parent_type::order;
        using config_type = typename parent_type::config_type;
        using index_type = typename parent_type::index_type;
        using dim_type = typename parent_type::dim_type;
        using index_map_type = typename config_type::index_map_type;
        using index_container_type = typename config_type::template container<index_type>;
        using index_container_difference_type = typename index_container_type::difference_type;
        using descriptor_type = mapping_descriptor<config_type,order>;
        using view_type = mapping_view<config_type,order,parent_type>;
        const auto& pshape = parent.shape();
        const auto& subs_shape = subs.shape();
        detail::check_bool_mapping_view_subs(pshape, subs_shape);
        if (!parent.empty() && !subs.empty()){
            index_container_type index_container{};
            index_type subs_trues_number{0};
            const dim_type subs_dim = subs.dim();
            const index_type chunk_size = detail::mapping_view_chunk_size(pshape, subs_dim);
            if constexpr (detail::is_static_castable_v<index_type,index_container_difference_type>){
                index_container.reserve(static_cast<index_container_difference_type>(parent.size()));
            }
            subs_trues_number = detail::fill_bool_map<order>(
                parent.strides(),
                subs_dim,
                chunk_size,
                index_container,
                walker_forward_traverser<config_type, decltype(subs.create_walker())>{subs_shape, subs.create_walker()}
            );
            auto res_shape = detail::make_bool_mapping_view_shape(pshape, subs_trues_number, subs.dim());
            auto res_size = detail::make_size(res_shape);
            index_map_type index_map(res_size);

            index_type i{0};
            if (chunk_size == index_type{1} || std::is_same_v<order,gtensor::config::c_order>){
                for(auto it=index_container.begin(), last=index_container.end(); it!=last; ++it,++i){
                    index_map[i] = *it;
                }
            }else{  //need reorder if parent in f_order and chunk_size > 1
                index_type j{i};
                const index_type j_delta = chunk_size*subs_trues_number;
                index_type j_last{i+j_delta};
                for(auto it=index_container.begin(), last=index_container.end(); it!=last; ++it,j+=subs_trues_number){
                    if (j == j_last){
                        j = ++i;
                        j_last = i+j_delta;
                    }
                    index_map[j] = *it;
                }
            }

            return std::make_shared<view_type>(
                descriptor_type{std::move(res_shape), std::move(index_map)},
                parent
            );
        }else{
            index_type subs_trues_number = std::count(subs.begin(),subs.end(),true);
            return std::make_shared<view_type>(
                descriptor_type{
                    detail::make_bool_mapping_view_shape(pshape, subs_trues_number, subs.dim()),
                    index_map_type(0)
                },
                parent
            );
        }
    }

public:
    //view_factory interface
    //reshape view
    template<typename Order, typename...Ts, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename basic_tensor<Ts...>::index_type>,int> = 0>
    static auto create_reshape_view(const basic_tensor<Ts...>& parent, const Container& subs){
        return create_reshape_view_container<Order>(parent, subs);
    }
    template<typename Order, typename...Ts, typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs, typename basic_tensor<Ts...>::index_type>...>,int> = 0>
    static auto create_reshape_view(const basic_tensor<Ts...>& parent, const Subs&...subs){
        return create_reshape_view_variadic<Order>(parent, subs...);
    }
    //transpose view
    template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename basic_tensor<Ts...>::dim_type>,int> = 0>
    static auto create_transpose_view(const basic_tensor<Ts...>& parent, const Container& subs){
        return create_transpose_view_container(parent, subs);
    }
    template<typename...Ts, typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs, typename basic_tensor<Ts...>::dim_type>...>,int> = 0>
    static auto create_transpose_view(const basic_tensor<Ts...>& parent, const Subs&...subs){
        return create_transpose_view_variadic(parent, subs...);
    }
    //slice view
    template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_of_type_v<Container, typename basic_tensor<Ts...>::slice_type>,int> = 0>
    static auto create_slice_view(const basic_tensor<Ts...>& parent, const Container& subs){
        return create_slice_view_container(parent, subs);
    }

    template<typename Tensor, typename...Subs> struct enable_slice_view_ : std::conjunction<
        std::disjunction<
            std::is_convertible<Subs,typename Tensor::index_type>,
            std::is_convertible<Subs,typename Tensor::slice_type>
        >...
    >{};

    template<typename...Ts, typename...Subs, std::enable_if_t<enable_slice_view_<basic_tensor<Ts...>,Subs...>::value,int> = 0>
    static auto create_slice_view(const basic_tensor<Ts...>& parent, const Subs&...subs){
        return create_slice_view_variadic(parent, subs...);
    }
    //index mapping view

    template<typename IdxT, typename...Subs> struct enable_index_mapping_view_variadic_ : std::conjunction<
        std::bool_constant<detail::is_tensor_of_type_v<Subs,IdxT>>...
    >{};

    template<typename...Ts, typename...Subs, std::enable_if_t<enable_index_mapping_view_variadic_<typename basic_tensor<Ts...>::index_type, Subs...>::value ,int> =0>
    static auto create_index_mapping_view(const basic_tensor<Ts...>& parent, const Subs&...subs){
        return create_index_mapping_view_variadic(parent, subs...);
    }
    template<typename...Ts, typename Container, std::enable_if_t<detail::is_container_of_tensor_of_type_v<Container,typename basic_tensor<Ts...>::index_type>,int> =0>
    static auto create_index_mapping_view(const basic_tensor<Ts...>& parent, const Container& subs){
        return create_index_mapping_view_container(parent, subs);
    }
    //bool mapping view
    template<typename...Ts, typename Subs>
    static auto create_bool_mapping_view(const basic_tensor<Ts...>& parent, const Subs& subs){
        return create_bool_mapping_view_(parent, subs);
    }

};  //end of class view_factory
}   //end of namespace gtensor
#endif