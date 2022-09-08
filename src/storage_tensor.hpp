#ifndef IMPL_STENSOR_HPP_
#define IMPL_STENSOR_HPP_

#include "shareable_storage.hpp"
#include "tensor_base.hpp"
#include "descriptor.hpp"
#include "tensor_init_list.hpp"
#include "engine_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


/*
* implementation of tensor with storage
*/
template<typename ValT, typename CfgT, typename EngineT = typename detail::engine_traits<storage_tensor<ValT,CfgT,void>>::type>
class storage_tensor : 
    public tensor_base<ValT,CfgT>,
    public storing_base<ValT,CfgT>,
    public converting_base<CfgT>    
{
public:
    using value_type = ValT;
    using engine_type = EngineT;
    //using engine_type = typename detail::engine_traits<storage_tensor>::type;

private:
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using storage_type = typename CfgT::template storage<value_type>;
    using descriptor_type = basic_descriptor<CfgT>;
    using slices_collection_type = typename CfgT::slices_collection_type;

    descriptor_type descriptor_;
    storage_type elements;
    engine_type engine_{this};    

    template<typename Nested>
    storage_tensor(std::initializer_list<Nested> init_data,int):
        descriptor_{detail::list_parse<index_type,shape_type>(init_data)},
        elements(descriptor_.size())
    {detail::fill_from_list(init_data, elements.begin());}
    
    const storing_base<ValT,CfgT>* as_storing()const override{return static_cast<const storing_base<ValT,CfgT>*>(this);}
    const converting_base* as_converting()const override{return static_cast<const converting_base*>(this);}

    storage_walker<ValT,CfgT> create_storage_walker()const override{
        return storage_walker_factory<ValT,CfgT>::create_walker(shape(),strides(),elements.data());
    }
    
    bool is_cached()const override{return true;}
    bool is_storage()const override{return is_cached();}
    const value_type* storage_data()const override{return elements.data();}
    index_type view_index_convert(const index_type& idx)const override{return idx;}
    

protected:


    // typename storage_type::const_iterator begin()const override{return elements.begin();}
    // typename storage_type::const_iterator end()const override{return elements.end();}

public:    
    
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

    const engine_type& engine()const override{return engine_;}
    const value_type* data()const{return elements.data();}

    detail::tensor_kinds tensor_kind()const override{return detail::tensor_kinds::storage_tensor;}
    const descriptor_base<CfgT>& descriptor()const override{return descriptor_;}
    index_type size()const override{return descriptor_.size();}
    index_type dim()const override{return descriptor_.dim();}
    const shape_type& shape()const override{return descriptor_.shape();}
    const shape_type& strides()const override{return descriptor_.strides();}    

    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"{"<<[&ss,this](){ss<<descriptor_.to_str(); for(const auto& i:elements){ss<<i<<",";} return "}";}();
        return ss.str();
    }
};

}   //end of namespace gtensor

#endif