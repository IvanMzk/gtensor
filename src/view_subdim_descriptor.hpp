#ifndef VIEW_SUBDIM_DESCRIPTOR_HPP_
#define VIEW_SUBDIM_DESCRIPTOR_HPP_

#include <numeric>
#include "descriptor_base.hpp"

namespace gtensor{

//view subdim and reshape descriptor
template<typename ValT, typename CfgT> 
class view_subdim_descriptor : 
    public descriptor_base<CfgT>,
    private descriptor_common<CfgT>
{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    
    index_type offset_;

    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides().begin(), offset_);
    }    

public:
    template<typename ShT>
    view_subdim_descriptor(ShT&& shape__, index_type offset__):
        descriptor_common{std::forward<ShT>(shape__)},
        offset_{offset__}
    {}

    index_type dim()const{return descriptor_common::dim();}
    index_type size()const{return descriptor_common::size();}
    const shape_type& shape()const{return descriptor_common::shape();}
    const shape_type& strides()const{return descriptor_common::strides();}
    std::string to_str()const{return descriptor_common::to_str();}

    index_type offset()const{return offset_;}
    const shape_type& cstrides()const{return strides();}
    index_type convert(const shape_type& idx)const{return convert_helper(idx);}
    index_type convert(const index_type& idx)const{return idx+offset_;}    
};

}   //end of namespace gtensor

#endif