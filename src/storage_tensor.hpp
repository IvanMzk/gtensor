#ifndef IMPL_STENSOR_HPP_
#define IMPL_STENSOR_HPP_

#include "shareable_storage.hpp"
#include "tensor_base.hpp"
#include "stensor_descriptor.hpp"
#include "tensor_init_list.hpp"
#include "walker_factory.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


/*
* implementation of tensor with storage
*/
template<typename ValT, template<typename> typename Cfg>
class storage_tensor : 
    public tensor_base<ValT,Cfg>,
    public storing_base<ValT,Cfg>,
    public view_index_converter<ValT,Cfg>,
    public walker_maker<ValT, Cfg>
{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::storage_type;
    using descriptor_type = stensor_descriptor<value_type, Cfg>;
    using slices_collection_type = typename config_type::slices_collection_type;

    descriptor_type descriptor_;
    storage_type elements;    

    template<typename Nested>
    storage_tensor(std::initializer_list<Nested> init_data,int):
        descriptor_{detail::list_parse<index_type,shape_type>(init_data)},
        elements(descriptor_.size())
    {detail::fill_from_list(init_data, elements.begin());}
    
    const storing_base<ValT,Cfg>* as_storage_tensor()const override{return static_cast<const storing_base<ValT,Cfg>*>(this);}
    const view_index_converter<ValT,Cfg>* as_index_converter()const override{return static_cast<const view_index_converter<ValT,Cfg>*>(this);}

    storage_walker_inline_impl<ValT,Cfg> create_storage_walker()const override{
        return storage_walker_factory<ValT,Cfg>::create_walker(shape(),strides(),elements.data());
    }
    
    walker<ValT, Cfg> create_polymorphic_walker()const override{
        return polymorphic_walker_factory<ValT,Cfg>::create_walker(*this, shape(), strides(), elements.data());
    }
    
    bool is_storage()const override{return true;}
    bool is_trivial()const override{return true;}
    const value_type* storage_data()const override{return elements.data();}
    index_type view_index_convert(const index_type& idx)const override{return idx;}
    

protected:


    // typename storage_type::const_iterator begin()const override{return elements.begin();}
    // typename storage_type::const_iterator end()const override{return elements.end();}

public:
    const walker_maker<ValT,Cfg>* as_walker_maker()const{return static_cast<const walker_maker<ValT,Cfg>*>(this);}
    
    storage_tensor() = default;
    storage_tensor(typename detail::nested_initializer_list_type<value_type,1>::type init_data):storage_tensor(init_data,0){}
    storage_tensor(typename detail::nested_initializer_list_type<value_type,2>::type init_data):storage_tensor(init_data,0){}
    storage_tensor(typename detail::nested_initializer_list_type<value_type,3>::type init_data):storage_tensor(init_data,0){}
    storage_tensor(typename detail::nested_initializer_list_type<value_type,4>::type init_data):storage_tensor(init_data,0){}
    storage_tensor(typename detail::nested_initializer_list_type<value_type,5>::type init_data):storage_tensor(init_data,0){}

    template<typename...Dims>
    storage_tensor(const value_type& v, const Dims&...dims):
        descriptor_{shape_type{dims...}},
        elements(descriptor_.size(), v)
    {}

    const value_type* data()const{return elements.data();}

    detail::tensor_kinds tensor_kind()const override{return detail::tensor_kinds::storage_tensor;}
    const descriptor_base<ValT,Cfg>& descriptor()const override{return descriptor_;}
    index_type size()const override{return descriptor_.size();}
    index_type dim()const override{return descriptor_.dim();}
    const shape_type& shape()const override{return descriptor_.shape();}
    const shape_type& strides()const override{return descriptor_.strides();}
    value_type trivial_at(const index_type& idx)const override{return elements[idx];}
    

    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"{"<<[&ss,this](){ss<<descriptor_.to_str(); for(const auto& i:elements){ss<<i<<",";} return "}";}();
        return ss.str();
    }
};

}   //end of namespace gtensor

#endif