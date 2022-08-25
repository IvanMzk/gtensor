#ifndef TENSOR_SLICE_HPP_
#define TENSOR_SLICE_HPP_

#include <numeric>
#include "descriptor_base.hpp"
#include "libdivide_helper.hpp"

namespace gtensor{

template<typename CfgT>
class stensor_descriptor :
    public descriptor_base<CfgT>,
    private basic_descriptor<CfgT>,
    private detail::collection_libdivide_extension<CfgT,typename CfgT::div_mode>
{
    using base_strides_libdivide = detail::collection_libdivide_extension<CfgT,typename CfgT::div_mode>;    
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;

    index_type convert_helper(const shape_type& idx)const{
        return std::inner_product(idx.begin(), idx.end(), cstrides().begin(), index_type{0});
    }    

public:
    stensor_descriptor() = default;       
    template<typename ShT>
    stensor_descriptor(ShT&& shape__):
        basic_descriptor{std::forward<ShT>(shape__)},
        base_strides_libdivide{basic_descriptor::strides()}
    {}    
    
    index_type dim()const{return basic_descriptor::dim();}
    index_type size()const{return basic_descriptor::size();}
    const shape_type& shape()const{return basic_descriptor::shape();}
    const shape_type& strides()const{return basic_descriptor::strides();}
    std::string to_str()const{return basic_descriptor::to_str();}

    index_type offset()const{return index_type{0};}
    const shape_type& cstrides()const{return strides();}
    const auto& strides_libdivide()const{return base_strides_libdivide::dividers_libdivide();}    
    index_type convert(const index_type& idx)const{return idx;}
    index_type convert(const shape_type& idx)const{return convert_helper(idx);}
};

}   //end of namespace gtensor
#endif