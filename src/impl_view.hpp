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
    public storage_tensor_impl_base<ValT,Cfg>,
    public view_index_converter<ValT,Cfg>,
    public view_expression_impl_base<ValT,Cfg>,
    public walker_maker<ValT, Cfg>
{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::storage_type;
    using slices_collection_type = typename config_type::slices_collection_type;

    DescT descriptor_;
    std::shared_ptr<impl_base_type> parent;
    const impl_base_type* view_root{parent->tensor_kind() == detail::tensor_kinds::view ? static_cast<const view_impl*>(parent.get())->get_view_root() : parent.get()};
    const view_index_converter<ValT,Cfg>* parent_converter{parent->as_index_converter()};
    storage_type cache{};

    storage_walker_impl<ValT,Cfg> create_storage_walker()const override{
        if (is_cached()){
            return storage_walker_factory<ValT,Cfg>::create_walker(shape(),strides(), cache.data()+descriptor_.offset());
        }else if(detail::is_storage(*parent)){
            return storage_walker_factory<ValT,Cfg>::create_walker(shape(),descriptor_.cstrides(), parent->as_storage_tensor()->data()+descriptor_.offset());
        }
        else{
            throw view_impl_exception("storage_walker cant be created, view not cached and parent not storage");
        }
    }
    view_expression_walker_impl<ValT,Cfg> create_view_expression_walker()const override{
        return view_expression_walker_impl<ValT,Cfg>{shape(),descriptor_.cstrides(),descriptor_.offset(), parent_converter, view_root->as_expression()->create_storage()};
    }
    walker<ValT, Cfg> create_polymorphic_walker()const override{
        return polymorphic_walker_factory<ValT,Cfg>::create_walker(*this, *parent, *view_root, cache.data());
        //return nullptr;
    }
    
    const storage_tensor_impl_base<ValT,Cfg>* as_storage_tensor()const override{return static_cast<const storage_tensor_impl_base<ValT,Cfg>*>(this);}
    const view_impl_base<ValT,Cfg>* as_view()const override{return static_cast<const view_impl_base<ValT,Cfg>*>(this);}
    const view_index_converter<ValT,Cfg>* as_index_converter()const override{return static_cast<const view_index_converter<ValT,Cfg>*>(this);}
    const view_expression_impl_base<ValT,Cfg>* as_view_expression()const{return static_cast<const view_expression_impl_base<ValT,Cfg>*>(this);}
    const walker_maker<ValT,Cfg>* as_walker_maker()const{return static_cast<const walker_maker<ValT,Cfg>*>(this);}
        
    index_type view_index_convert(const index_type& idx)const override{return parent_converter->convert(descriptor_.convert(idx));}
    bool is_cached()const override{return cache.size();}
    detail::tensor_kinds view_root_kind()const override{return view_root->tensor_kind();}
    bool is_storage()const override{return detail::is_storage(*parent) || is_cached();}
    bool is_trivial()const override{return true;}
    const value_type* storage_data()const override{return cache.data();}
    const impl_base_type* get_view_root()const{return view_root;}

public:
    template<typename DtT>
    view_impl(DtT&& descriptor__, const std::shared_ptr<impl_base_type>& parent_):
        descriptor_{std::forward<DtT>(descriptor__)},
        parent{parent_}
    {}    

    detail::tensor_kinds tensor_kind()const override{return detail::tensor_kinds::view;}
    const descriptor_base<ValT,Cfg>& descriptor()const override{return descriptor_;}
    index_type size()const override{return descriptor_.size();}
    index_type dim()const override{return descriptor_.dim();}
    const shape_type& shape()const override{return descriptor_.shape();}
    const shape_type& strides()const override{return descriptor_.strides();}
    value_type trivial_at(const index_type& idx)const override{return value_type(0);}    

    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"{"<<[&ss,this](){ss<<descriptor_.to_str(); return "}";}();
        return ss.str();
    }
};

// template<typename ValT, template<typename> typename Cfg, typename DescT>
// class view_of_expression_impl : public view_impl<ValT, Cfg, DescT>
// {
//     using base_type = view_impl<ValT, Cfg, DescT>;
// public:
//     template<typename DtT>
//     view_of_expression_impl(DtT&& descriptor_, const std::shared_ptr<impl_base_type>& parent_):
//         base_type{std::forward<DtT>(descriptor_), parent_}        
//     {}    

// };


}   //end of namespace gtensor


#endif