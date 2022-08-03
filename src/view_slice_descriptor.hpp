#ifndef VIEW_SLICE_DESCRIPTOR_HPP_
#define VIEW_SLICE_DESCRIPTOR_HPP_

#include <numeric>
#include "descriptor_base.hpp"
#include "libdivide_helper.hpp"

namespace gtensor{

//view slice and transpose descriptor
template<typename ValT, template<typename> typename Cfg> 
class view_slice_descriptor : 
    public descriptor_base<ValT,Cfg>,
    private basic_descriptor<ValT,Cfg>,
    private detail::collection_libdivide_extension<ValT,Cfg,typename Cfg<ValT>::div_mode>
{
    using base_descriptor = basic_descriptor<ValT,Cfg>;
    using base_strides_libdivide = detail::collection_libdivide_extension<ValT,Cfg,typename Cfg<ValT>::div_mode>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    
    
    shape_type cstrides_;
    index_type offset_;

    const auto& strides_libdivide()const{return base_strides_libdivide::dividers_libdivide();}
    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides_.begin(), offset_);
    }
    template<typename C = config_type, std::enable_if_t<detail::is_mode_div_libdivide<C>, int> =0 >
    index_type convert_helper(const index_type& idx)const{
        return convert_helper(gtensor::detail::flat_to_multi<shape_type>(strides_libdivide(), idx));
    }
    template<typename C = config_type, std::enable_if_t<detail::is_mode_div_native<C>, int> =0 >
    index_type convert_helper(const index_type& idx)const{
        return convert_helper(gtensor::detail::flat_to_multi(strides(), idx));
    }

public:
    template<typename ShT, typename StT>
    view_slice_descriptor(ShT&& shape__, StT&& cstrides__,  const index_type& offset__):
        base_descriptor{std::forward<ShT>(shape__)},
        base_strides_libdivide{base_descriptor::strides()},
        cstrides_{std::forward<StT>(cstrides__)},
        offset_{offset__}
    {}
    index_type dim()const{return base_descriptor::dim();}
    index_type size()const{return base_descriptor::size();}
    const shape_type& shape()const{return base_descriptor::shape();}
    const shape_type& strides()const{return base_descriptor::strides();}
    std::string to_str()const{return base_descriptor::to_str();}
        
    index_type offset()const{return offset_;}
    const shape_type& cstrides()const{return cstrides_;}
    index_type convert(const shape_type& idx)const{return convert_helper(idx);}
    index_type convert(const index_type& idx)const{return convert_helper(idx);}
};


}   //end of namespace gtensor

#endif