#ifndef TENSOR_BASE_HPP_
#define TENSOR_BASE_HPP_

#include "config.hpp"
#include "forward_decl.hpp"

namespace gtensor{

template<typename CfgT>
class tensor_base_base
{
protected:
    using config_type = CfgT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
public:
    virtual ~tensor_base_base(){}

    virtual index_type size()const = 0;
    virtual index_type dim()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual const shape_type& strides()const = 0;
    virtual const shape_type& reset_strides()const = 0;
    virtual std::string to_str()const = 0;
    virtual const descriptor_base<config_type>& descriptor()const = 0;

    virtual const converting_base<config_type>* as_converting()const{return nullptr;}
};

template<typename ValT, typename CfgT>
class tensor_base : public tensor_base_base<CfgT>
{
protected:
    using typename tensor_base_base::config_type;
    using typename tensor_base_base::index_type;
    using typename tensor_base_base::shape_type;
public:
    using value_type = ValT;
    virtual ~tensor_base(){}
};


template<typename CfgT>
class converting_base
{
    using index_type = typename CfgT::index_type;
    virtual index_type view_index_convert(const index_type&)const = 0;

public:
    virtual ~converting_base(){}
    auto convert(const index_type& idx)const{return view_index_convert(idx);}

};

}   //end of namespace gtensor

#endif