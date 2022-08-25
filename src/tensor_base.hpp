#ifndef IMPL_TENSOR_BASE_HPP_
#define IMPL_TENSOR_BASE_HPP_

#include <string>
#include "forward_decl.hpp"
#include "slice.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename ValT, typename CfgT>
class tensor_base{        
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using slices_collection_type = typename CfgT::slices_collection_type;

public:
    virtual ~tensor_base(){}    
    virtual index_type size()const = 0;
    virtual index_type dim()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual const shape_type& strides()const = 0;
    virtual std::string to_str()const = 0;
    virtual const descriptor_base<CfgT>& descriptor()const = 0;
    
    virtual value_type trivial_at(const index_type& idx)const = 0;
    virtual detail::tensor_kinds tensor_kind()const = 0;
    
    
    virtual bool is_cached()const = 0;
    virtual bool is_storage()const = 0;
    virtual bool is_trivial()const = 0;

    
    virtual const storing_base<ValT,CfgT>* as_storing()const{return nullptr;}
    virtual const evaluating_base<ValT,CfgT>* as_evaluating()const{return nullptr;}
    virtual const evaluating_trivial_base<ValT,CfgT>* as_evaluating_trivial()const{return nullptr;}    
    virtual const converting_base<ValT,CfgT>* as_converting()const{return nullptr;}
    virtual const viewing_evaluating_base<ValT,CfgT>* as_viewing_evaluating()const{return nullptr;}
        
};

template<typename ValT, typename CfgT>
class storing_base 
{
    virtual storage_walker<ValT,CfgT> create_storage_walker()const = 0;
    virtual const ValT* storage_data()const = 0;

public:
    virtual ~storing_base(){}
    auto create_walker()const{return create_storage_walker();}
    auto data()const{return storage_data();}
    
};

template<typename ValT, typename CfgT>
class evaluating_base
{    
    virtual walker<ValT,CfgT> create_evaluating_walker()const = 0;    
    virtual indexer<ValT,CfgT> create_evaluating_indexer()const = 0;
public:
    virtual ~evaluating_base(){}     
    auto create_walker()const{return create_evaluating_walker();}    
    auto create_indexer()const{return create_evaluating_indexer();}
};

template<typename ValT, typename CfgT>
class evaluating_trivial_base
{
    virtual evaluating_trivial_walker<ValT,CfgT> create_trivial_walker()const = 0;    
public:
    virtual ~evaluating_trivial_base(){}
    auto create_walker()const{return create_trivial_walker();}    
};

template<typename ValT, typename CfgT>
class converting_base
{
    using index_type = typename CfgT::index_type;
    virtual index_type view_index_convert(const index_type&)const = 0;

public:
    virtual ~converting_base(){}        
    auto convert(const index_type& idx)const{return view_index_convert(idx);}
    
};

template<typename ValT, typename CfgT>
class viewing_evaluating_base
{    
    virtual viewing_evaluating_walker<ValT,CfgT> create_view_expression_walker()const = 0;
public:
    virtual ~viewing_evaluating_base(){}    
    auto create_walker()const{return create_view_expression_walker();}
};


}   //end of namespace gtensor

#endif