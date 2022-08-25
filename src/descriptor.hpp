#ifndef DESCRIPTOR_HPP_
#define DESCRIPTOR_HPP_

#include <numeric>
#include "descriptor_base.hpp"

namespace gtensor{

template<typename CfgT>
class basic_descriptor :
    public descriptor_base<CfgT>,
    private descriptor_common<CfgT>    
{    
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;

    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides().begin(), index_type{0});
    }    

public:
    basic_descriptor() = default;       
    template<typename ShT>
    basic_descriptor(ShT&& shape__):
        descriptor_common{std::forward<ShT>(shape__)}        
    {}    
    
    index_type dim()const override{return descriptor_common::dim();}
    index_type size()const override{return descriptor_common::size();}
    const shape_type& shape()const override{return descriptor_common::shape();}
    const shape_type& strides()const override{return descriptor_common::strides();}
    std::string to_str()const override{return descriptor_common::to_str();}

    index_type offset()const override{return index_type{0};}
    const shape_type& cstrides()const override{return strides();}    
    index_type convert(const index_type& idx)const override{return idx;}
    index_type convert(const shape_type& idx)const override{return convert_helper(idx);}
};

template<typename CfgT>
class descriptor_with_lidivide :
    public basic_descriptor<CfgT>,    
    private detail::collection_libdivide_extension<CfgT,typename CfgT::div_mode>
{
    using base_strides_libdivide = detail::collection_libdivide_extension<CfgT,typename CfgT::div_mode>;    
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;    

public:
    descriptor_with_lidivide() = default;       
    template<typename ShT>
    descriptor_with_lidivide(ShT&& shape__):
        basic_descriptor{std::forward<ShT>(shape__)},
        base_strides_libdivide{basic_descriptor::strides()}
    {}    
    
    const auto& strides_libdivide()const{return base_strides_libdivide::dividers_libdivide();}
};

template<typename CfgT> 
class descriptor_with_offset : public basic_descriptor<CfgT>    
{    
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    
    index_type offset_;

    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides().begin(), offset_);
    }

public:
    template<typename ShT>
    descriptor_with_offset(ShT&& shape__, index_type offset__):
        basic_descriptor{std::forward<ShT>(shape__)},
        offset_{offset__}
    {}
    index_type offset()const override{return offset_;}    
    index_type convert(const shape_type& idx)const override{return convert_helper(idx);}
    index_type convert(const index_type& idx)const override{return idx+offset_;}    
};



}   //end of namespace gtensor

#endif