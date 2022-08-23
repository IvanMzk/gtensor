#ifndef VIEWING_TENSOR_HPP_
#define VIEWING_TENSOR_HPP_

#include "shareable_storage.hpp"
#include "tensor_base.hpp"
#include "walker_factory.hpp"

namespace gtensor{

class view_tensor_exception : public std::runtime_error{
    public: view_tensor_exception(const char* what):runtime_error(what){}
};


template<typename ValT, template<typename> typename Cfg, typename DescT>
class viewing_tensor : 
    public tensor_base<ValT, Cfg>,    
    public storing_base<ValT,Cfg>,
    public viewing_evaluating_base<ValT,Cfg>,
    public converting_base<ValT,Cfg>
{
    using tensor_base_type = tensor_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

    DescT descriptor_;
    std::shared_ptr<tensor_base_type> parent;
    const tensor_base_type* view_root{parent->tensor_kind() == detail::tensor_kinds::view ? static_cast<const viewing_tensor*>(parent.get())->get_view_root() : parent.get()};
    const converting_base<ValT,Cfg>* parent_converter{parent->as_converting()};
    
    //tensor_base interface

    //parent of view must be storing_tensor
    const storing_base<ValT,Cfg>* as_storing()const override{return is_storage() ? static_cast<const storing_base<ValT,Cfg>*>(this) : nullptr;}
    //root of view must be evaluating_tensor
    const viewing_evaluating_base<ValT,Cfg>* as_viewing_evaluating()const{return static_cast<const viewing_evaluating_base<ValT,Cfg>*>(this);}
    const converting_base<ValT,Cfg>* as_converting()const override{return static_cast<const converting_base<ValT,Cfg>*>(this);}
    
    bool is_cached()const override{return false;}    
    bool is_trivial()const override{return true;}
    bool is_storage()const override{return detail::is_storage(*parent);}

    //storing_base interface implementation
    const value_type* storage_data()const override{return parent->as_storing()->data();}

    storage_walker<ValT,Cfg> create_storage_walker()const override{
        return storage_walker_factory<ValT,Cfg>::create_walker(shape(),descriptor_.cstrides(), parent->as_storing()->data()+descriptor_.offset());        
    }
    
    //viewing_evaluating_base interface implementation
    viewing_evaluating_walker<ValT,Cfg> create_view_expression_walker()const override{
        return viewing_evaluating_walker<ValT,Cfg>{shape(),descriptor_.cstrides(),descriptor_.offset(), parent_converter, view_root->as_evaluating()->create_indexer()};
    }
    
    //converting_base interface implementation    
    index_type view_index_convert(const index_type& idx)const override{return parent_converter->convert(descriptor_.convert(idx));}

    const tensor_base_type* get_view_root()const{return view_root;}
public:
    template<typename DtT>
    viewing_tensor(DtT&& descriptor__, const std::shared_ptr<tensor_base_type>& parent_):
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


}   //end of namespace gtensor


#endif