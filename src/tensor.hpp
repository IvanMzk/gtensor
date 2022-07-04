#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <memory>
#include "impl_stensor.hpp"
#include "tensor_operators.hpp"

namespace gtensor{

/*
* tensor is abstraction of stensor_impl, expression or view which are implementations
* in aplication client use tensor abstraction objects with different implementations, can combine it using operators
* client can evaluate tensor object with expression or view implementation to have tensor with stensor_impl implementation
*/
template<typename ValT, template<typename> typename Cfg>
class tensor{
    using config_type = Cfg<ValT>;
    using impl_base_type = tensor_impl_base<ValT, Cfg>;
    using stensor_impl_type = stensor_impl<ValT, Cfg>;
    friend class tensor_operators_impl;
    std::shared_ptr<impl_base_type> impl;
    
    template<typename Nested>
    tensor(std::initializer_list<Nested> init_data, int):
        impl{new stensor_impl_type(init_data)}
    {}
    
    std::shared_ptr<impl_base_type> get_impl()const{return impl;}

public:        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    
    tensor(std::shared_ptr<impl_base_type>&& impl_):
        impl{std::move(impl_)}
    {}

    tensor(typename detail::nested_initializer_list_type<value_type,1>::type init_data):tensor(init_data,0){}
    tensor(typename detail::nested_initializer_list_type<value_type,2>::type init_data):tensor(init_data,0){}
    tensor(typename detail::nested_initializer_list_type<value_type,3>::type init_data):tensor(init_data,0){}
    tensor(typename detail::nested_initializer_list_type<value_type,4>::type init_data):tensor(init_data,0){}
    tensor(typename detail::nested_initializer_list_type<value_type,5>::type init_data):tensor(init_data,0){}
    
    auto size()const{return impl->size();}
    auto dim()const{return impl->dim();}
    auto shape()const{return impl->shape();}
    auto create_walker()const{return impl->create_walker();}
    friend std::ostream& operator<<(std::ostream& os, const tensor& lhs){return os<<lhs.impl->to_str();}
};

}   //end of namespace gtensor

#endif