#ifndef IMPL_TENSOR_BASE_HPP_
#define IMPL_TENSOR_BASE_HPP_

#include <string>
#include "forward_decl.hpp"
#include "slice.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


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
    
    virtual value_type trivial_at(const index_type& idx)const = 0;
    virtual detail::tensor_kinds tensor_kind()const = 0;
    virtual bool is_storage()const = 0;
    virtual bool is_trivial()const = 0;

    virtual const expression_impl_base<ValT,Cfg>* as_expression()const{return nullptr;}
    virtual const trivial_impl_base<ValT,Cfg>* as_expression_trivial()const{return nullptr;}
    virtual const storage_tensor_impl_base<ValT,Cfg>* as_storage_tensor()const{return nullptr;}
    virtual const view_impl_base<ValT,Cfg>* as_view()const{return nullptr;}
    virtual const view_index_converter<ValT,Cfg>* as_index_converter()const{return nullptr;}
};

template<typename ValT, template<typename> typename Cfg>
class expression_impl_base
{
    using iterator_type = multiindex_iterator_impl<ValT,Cfg,walker<ValT,Cfg>>;    
    virtual walker<ValT,Cfg> create_evaluating_walker()const = 0;
public:
    virtual ~expression_impl_base(){}    
    // virtual iterator_type begin()const = 0;
    // virtual iterator_type end()const = 0;
    virtual bool is_cached()const = 0;
    virtual bool is_trivial()const = 0;
    auto create_walker()const{return create_evaluating_walker();}
};

template<typename ValT, template<typename> typename Cfg>
class trivial_impl_base
{
    using iterator_type = multiindex_iterator_impl<ValT,Cfg,walker<ValT,Cfg>>;
    virtual ewalker_trivial_impl<ValT,Cfg> create_trivial_walker()const = 0;
public:
    virtual ~trivial_impl_base(){}
    auto create_walker()const{return create_trivial_walker();}
    // virtual iterator_type begin()const = 0;
    // virtual iterator_type end()const = 0;    
};

template<typename ValT, template<typename> typename Cfg>
class view_impl_base
{
    using iterator_type = multiindex_iterator_impl<ValT,Cfg,walker<ValT,Cfg>>;
    virtual vwalker_impl<ValT,Cfg> create_view_walker()const = 0;
public:
    virtual ~view_impl_base(){}
    virtual bool is_cached()const = 0;
    auto create_walker()const{return create_view_walker();}
    // virtual iterator_type begin()const = 0;
    // virtual iterator_type end()const = 0;    
};

// template<typename ValT, template<typename> typename Cfg>
// class view_expression_impl_base
// {    
//     virtual view_expression_walker_impl<ValT,Cfg> create_view_expression_walker()const = 0;
// public:
//     virtual ~view_expression_impl_base(){}
//     virtual bool is_cached()const = 0;
//     auto create_walker()const{return create_view_expression_walker();}
//     // virtual iterator_type begin()const = 0;
//     // virtual iterator_type end()const = 0;    
// };

template<typename ValT, template<typename> typename Cfg>
class storage_tensor_impl_base 
{
    using config_type = Cfg<ValT>;
    using iterator_type = typename config_type::storage_type::iterator;
    using const_iterator_type = typename config_type::storage_type::const_iterator;
    
    virtual storage_walker_impl<ValT,Cfg> create_storage_walker()const = 0;
    virtual const ValT* storage_data()const = 0;

public:
    virtual ~storage_tensor_impl_base(){}
    // virtual const_iterator_type begin()const = 0;
    // virtual const_iterator_type end()const = 0;
    auto create_walker()const{return create_storage_walker();}
    auto data()const{return storage_data();}
    
};

template<typename ValT, template<typename> typename Cfg>
class view_index_converter
{
    using index_type = typename Cfg<ValT>::index_type;
    virtual index_type view_index_convert(const index_type&)const = 0;

public:
    virtual ~view_index_converter(){}        
    auto convert(const index_type& idx)const{return view_index_convert(idx);}
    
};



}   //end of namespace gtensor

#endif