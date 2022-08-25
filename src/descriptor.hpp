#ifndef DESCRIPTOR_HPP_
#define DESCRIPTOR_HPP_

#include "descriptor_base.hpp"

namespace gtensor{

template<typename CfgT>
class descriptor :
    public descriptor_base<CfgT>,
    private basic_descriptor<CfgT>    
{    
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;

    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides().begin(), index_type{0});
    }    

public:
    descriptor() = default;       
    template<typename ShT>
    descriptor(ShT&& shape__):
        basic_descriptor{std::forward<ShT>(shape__)}        
    {}    
    
    index_type dim()const override{return basic_descriptor::dim();}
    index_type size()const override{return basic_descriptor::size();}
    const shape_type& shape()const override{return basic_descriptor::shape();}
    const shape_type& strides()const override{return basic_descriptor::strides();}
    std::string to_str()const override{return basic_descriptor::to_str();}

    index_type offset()const override{return index_type{0};}
    const shape_type& cstrides()const override{return strides();}    
    index_type convert(const index_type& idx)const override{return idx;}
    index_type convert(const shape_type& idx)const override{return convert_helper(idx);}
};

}   //end of namespace gtensor

#endif