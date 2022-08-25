#ifndef VIEW_SLICE_DESCRIPTOR_HPP_
#define VIEW_SLICE_DESCRIPTOR_HPP_

#include <numeric>
#include "descriptor_base.hpp"
#include "libdivide_helper.hpp"

namespace gtensor{

//view slice and transpose descriptor
template<typename ValT, typename CfgT> 
class view_slice_descriptor : 
    public descriptor_base<CfgT>,
    private descriptor_common<CfgT>,
    private detail::collection_libdivide_extension<CfgT,typename CfgT::div_mode>
{
    using base_strides_libdivide = detail::collection_libdivide_extension<CfgT,typename CfgT::div_mode>;
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;    
    
    shape_type cstrides_;
    index_type offset_;

    const auto& strides_libdivide()const{return base_strides_libdivide::dividers_libdivide();}
    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides_.begin(), offset_);
    }
    template<typename C = CfgT, std::enable_if_t<detail::is_mode_div_libdivide<C>, int> =0 >
    index_type convert_helper(const index_type& idx)const{
        return convert_helper(gtensor::detail::flat_to_multi<shape_type>(strides_libdivide(), idx));
    }
    template<typename C = CfgT, std::enable_if_t<detail::is_mode_div_native<C>, int> =0 >
    index_type convert_helper(const index_type& idx)const{
        return convert_helper(gtensor::detail::flat_to_multi(strides(), idx));
    }

public:
    template<typename ShT, typename StT>
    view_slice_descriptor(ShT&& shape__, StT&& cstrides__,  const index_type& offset__):
        descriptor_common{std::forward<ShT>(shape__)},
        base_strides_libdivide{descriptor_common::strides()},
        cstrides_{std::forward<StT>(cstrides__)},
        offset_{offset__}
    {}
    index_type dim()const{return descriptor_common::dim();}
    index_type size()const{return descriptor_common::size();}
    const shape_type& shape()const{return descriptor_common::shape();}
    const shape_type& strides()const{return descriptor_common::strides();}
    std::string to_str()const{return descriptor_common::to_str();}
        
    index_type offset()const{return offset_;}
    const shape_type& cstrides()const{return cstrides_;}
    index_type convert(const shape_type& idx)const{return convert_helper(idx);}
    index_type convert(const index_type& idx)const{return convert_helper(idx);}
};


}   //end of namespace gtensor

#endif