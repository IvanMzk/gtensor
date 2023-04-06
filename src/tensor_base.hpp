#ifndef TENSOR_BASE_HPP_
#define TENSOR_BASE_HPP_

#include "forward_decl.hpp"

namespace gtensor{

template<typename ValT, typename CfgT>
class tensor_base
{
    using descriptor_type = descriptor_base<CfgT>;
public:
    using value_type = ValT;
    using config_type = CfgT;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    virtual ~tensor_base(){}
    virtual index_type size()const = 0;
    virtual dim_type dim()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual const shape_type& strides()const =0;
    virtual descriptor_base<config_type>& descriptor() = 0;
    virtual const descriptor_base<config_type>& descriptor()const = 0;
};

}   //end of namespace gtensor

#endif