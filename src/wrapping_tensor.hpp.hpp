#ifndef WRAPPING_TENSOR_HPP_
#define WRAPPING_TENSOR_HPP_

#include "tensor_base.hpp"

namespace gtensor{

template<typename ValT, template<typename> typename Cfg>
class wrapping_tensor : public tensor_base<ValT,Cfg>{
    
    using tensor_base_type = tensor_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    std::shared_ptr<tensor_base_type> impl_;

public:
    wrapping_tensor(std::shared_ptr<tensor_base_type>&& impl__):
        impl_{std::move(impl___)}
    {}    
    void reset_impl(std::shared_ptr<tensor_base_type>&& impl__){impl_.reset(impl__.get());}    
    auto impl()const{return impl_;}


    bool is_storage()const override{return impl->is_storage();}    
    bool is_cached()const override{return impl->is_cached();}
    bool is_trivial()const override{return impl->is_trivial();}
    value_type trivial_at(const index_type& idx)const override{return impl->trivial_at(idx);}

    detail::tensor_kinds tensor_kind()const override{return impl->tensor_kind();}
    const descriptor_base<ValT,Cfg>& descriptor()const override{return impl->descriptor();}
    index_type size()const override{return impl->size();}
    index_type dim()const override{return impl->dim();}
    const shape_type& shape()const override{return impl->shape();}
    const shape_type& strides()const override{return impl->strides();}
    std::string to_str()const override{return impl->to_str();}
};


}   //end of namespace gtensor


#endif