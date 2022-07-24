#ifndef IMPL_TENSOR_BASE_HPP_
#define IMPL_TENSOR_BASE_HPP_

#include <string>
#include "forward_decl.hpp"
#include "slice.hpp"

namespace gtensor{

template<typename ValT, template<typename> typename Cfg>
class tensor_impl_base{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using slices_collection_type = typename config_type::slices_collection_type;

public:
    virtual ~tensor_impl_base(){}    
    virtual index_type size()const = 0;
    virtual index_type dim()const = 0;
    virtual const shape_type& shape()const = 0;
    virtual const shape_type& strides()const = 0;
    virtual std::string to_str()const = 0;    
    virtual walker<ValT,Cfg> create_walker()const = 0;
    virtual value_type trivial_at(const index_type& idx)const = 0;
    virtual detail::tensor_kinds tensor_kind()const = 0; 

    const expression_impl_base<ValT,Cfg>* as_expression()const{return dynamic_cast<const expression_impl_base<ValT,Cfg>*>(this);}
    const stensor_impl_base<ValT,Cfg>* as_storage_tensor()const{return dynamic_cast<const stensor_impl_base<ValT,Cfg>*>(this);}

};

template<typename ValT, template<typename> typename Cfg>
class expression_impl_base : public tensor_impl_base<ValT, Cfg>{
    using iterator_type = multiindex_iterator_impl<ValT,Cfg,walker<ValT,Cfg>>;
public:
    virtual ~expression_impl_base(){}    
    virtual iterator_type begin()const = 0;
    virtual iterator_type end()const = 0;
    virtual bool is_cached()const = 0;
    virtual bool is_trivial()const = 0;
};

template<typename ValT, template<typename> typename Cfg>
class stensor_impl_base : public tensor_impl_base<ValT, Cfg>{
    using config_type = Cfg<ValT>;
    using iterator_type = typename config_type::storage_type::iterator;
    using const_iterator_type = typename config_type::storage_type::const_iterator;
public:
    virtual ~stensor_impl_base(){}
    virtual const_iterator_type begin()const = 0;
    virtual const_iterator_type end()const = 0;
};



}   //end of namespace gtensor

#endif