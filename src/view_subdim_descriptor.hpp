#ifndef VIEW_SUBDIM_DESCRIPTOR_HPP_
#define VIEW_SUBDIM_DESCRIPTOR_HPP_

#include <numeric>
#include "descriptor_base.hpp"

namespace gtensor{

//view subdim and reshape descriptor
template<typename ValT, template<typename> typename Cfg> 
class view_subdim_descriptor : 
    public descriptor_base<ValT,Cfg>,
    private basic_descriptor<ValT,Cfg>
{
    using base_descriptor = basic_descriptor<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    
    index_type offset_;

    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides().begin(), offset_);
    }    

public:
    template<typename ShT>
    view_subdim_descriptor(ShT&& shape__, index_type offset__):
        base_descriptor{std::forward<ShT>(shape__)},
        offset_{offset__}
    {}

    index_type dim()const{return base_descriptor::dim();}
    index_type size()const{return base_descriptor::size();}
    const shape_type& shape()const{return base_descriptor::shape();}
    const shape_type& strides()const{return base_descriptor::strides();}
    std::string to_str()const{return base_descriptor::to_str();}

    index_type offset()const{return offset_;}
    const shape_type& cstrides()const{return strides();}
    index_type convert(const shape_type& idx)const{return convert_helper(idx);}
    index_type convert(const index_type& idx)const{return idx+offset_;}    
};

}   //end of namespace gtensor

#endif