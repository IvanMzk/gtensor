#ifndef IMPL_VIEW_HPP_
#define IMPL_VIEW_HPP_

#include "shareable_storage.hpp"
#include "impl_tensor_base.hpp"
#include "walker_factory.hpp"

namespace gtensor{

class view_impl_exception : public std::runtime_error{
    public: view_impl_exception(const char* what):runtime_error(what){}
};

/*
* ParentT is tensor_impl_base or derived
*/
template<typename ValT, template<typename> typename Cfg, typename DescT>
class view_impl : 
    public tensor_impl_base<ValT, Cfg>,
    public view_impl_base<ValT,Cfg>,
    public storage_tensor_impl_base<ValT,Cfg>
{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::storage_type;
    using slices_collection_type = typename config_type::slices_collection_type;

    DescT descriptor;
    std::shared_ptr<impl_base_type> parent;
    storage_type cache{};

    storage_walker_impl<ValT,Cfg> create_storage_walker()const override{
        if (is_cached()){
            return storage_walker_factory<ValT,Cfg>::create_walker(shape(),strides(), cache.data()+descriptor.offset());
        }else if(is_storage_parent()){
            return storage_walker_factory<ValT,Cfg>::create_walker(shape(),descriptor.cstrides(), parent->as_storage_tensor()->data()+descriptor.offset());
        }
        else{
            throw view_impl_exception("storage_walker cant be created, view not cached and parent not storage");
        }
    }
    vwalker_impl<ValT,Cfg> create_view_walker()const override{
        return vwalker_impl<ValT,Cfg>{shape(),descriptor.cstrides(),descriptor.offset()};
    }
    
    bool is_storage_parent()const{
        return 
            parent->tensor_kind() == detail::tensor_kinds::storage_tensor || 
            parent->tensor_kind() == detail::tensor_kinds::expression && parent->is_storage() ||
            parent->tensor_kind() == detail::tensor_kinds::view && parent->as_view()->is_cached();
    }
    bool is_cached()const override{return cache.size();}    
    bool is_storage()const override{return is_storage_parent() || is_cached();}
    bool is_trivial()const override{return true;}
    const value_type* storage_data()const{return cache.data();}

public:
    template<typename DtT>
    view_impl(DtT&& descriptor_, const std::shared_ptr<impl_base_type>& parent_):
        descriptor{std::forward<DtT>(descriptor_)},
        parent{parent_}
    {}
    view_impl(const view_impl& other):
        descriptor{other.descriptor},
        parent{other.parent},
        cache{other.cache}
    {}
    view_impl(view_impl&& other):
        descriptor{std::move(other.descriptor)},
        parent{std::move(other.parent)},
        cache{std::move(other.cache)},
    {}

    detail::tensor_kinds tensor_kind()const override{return detail::tensor_kinds::view;}
    index_type size()const override{return descriptor.size();}
    index_type dim()const override{return descriptor.dim();}
    const shape_type& shape()const override{return descriptor.shape();}
    const shape_type& strides()const override{return descriptor.strides();}
    value_type trivial_at(const index_type& idx)const override{return value_type(0);}
    walker<ValT,Cfg> create_walker()const{return nullptr;}

    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"{"<<[&ss,this](){ss<<descriptor.to_str(); return "}";}();
        return ss.str();
    }


};

}   //end of namespace gtensor


#endif