#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <memory>
#include "tensor_wrapper.hpp"
#include "storage_tensor.hpp"
#include "tensor_operators.hpp"
#include "slice.hpp"
#include "view_factory.hpp"

namespace gtensor{

/*
* tensor is abstraction of stensor_impl, expression or view which are implementations
* in aplication client use tensor abstraction objects with different implementations, can combine it using operators
* client can evaluate tensor object with expression or view implementation to have tensor with stensor_impl implementation
*/
template<typename ValT, template<typename> typename Cfg>
class tensor{
    using tensor_type = tensor<ValT,Cfg>;
    using config_type = Cfg<ValT>;
    using tensor_base_type = tensor_base<ValT, Cfg>;
    using storage_tensor_type = storage_tensor<ValT, Cfg>;
    using tensor_wrapper_type = tensor_wrapper<ValT, Cfg>;
    using slice_type = typename config_type::slice_type;
    using slices_init_type = typename config_type::slices_init_type;
    using slices_collection_type = typename config_type::slices_collection_type;
    

    friend std::ostream& operator<<(std::ostream& os, const tensor& lhs){return os<<lhs.impl_->to_str();}
    friend class tensor_operators_impl;
    
    std::shared_ptr<tensor_wrapper_type> impl_;

    template<typename Nested>
    tensor(std::initializer_list<Nested> init_data, int):
        impl_{std::make_shared<tensor_wrapper_type>(std::make_shared<storage_tensor_type>(init_data))}
    {}
    
protected:
    auto impl()const{return impl_;}

public:        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    
    tensor(std::shared_ptr<tensor_base_type>&& impl__):
        impl_{std::make_shared<tensor_wrapper_type>(std::move(impl__))}
    {}

    template<typename...Dims>
    tensor(const value_type& v, const Dims&...dims):
        impl_{std::make_shared<tensor_wrapper_type>(std::make_shared<storage_tensor_type>(v, dims...))}
    {}

    tensor(typename detail::nested_initializer_list_type<value_type,1>::type init_data):tensor(init_data,0){}
    tensor(typename detail::nested_initializer_list_type<value_type,2>::type init_data):tensor(init_data,0){}
    tensor(typename detail::nested_initializer_list_type<value_type,3>::type init_data):tensor(init_data,0){}
    tensor(typename detail::nested_initializer_list_type<value_type,4>::type init_data):tensor(init_data,0){}
    tensor(typename detail::nested_initializer_list_type<value_type,5>::type init_data):tensor(init_data,0){}
    
    auto size()const{return impl()->size();}
    auto dim()const{return impl()->dim();}
    auto shape()const{return impl()->shape();}
    auto to_str()const{return impl()->to_str();}
    auto as_expression()const{return expression<ValT,Cfg>{*this};}
    auto as_storage_tensor()const{return storage_tensor<ValT,Cfg>{*this};}

    tensor_type operator()(slices_init_type subs)const{
        detail::check_slices_number(subs);        
        slices_collection_type filled_subs = detail::fill_slices<slice_type>(shape(),subs);
        detail::check_slices(shape(), filled_subs);        
        return tensor_type{view_factory<ValT,Cfg>::create_view_slice(impl()->impl(), filled_subs)};
    }
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,slice_type>...>,int> = 0 >
    tensor_type operator()(const Subs&...subs)const{
        detail::check_slices_number(subs...);
        slices_collection_type filled_subs = detail::fill_slices<slice_type>(shape(),subs...);
        detail::check_slices(shape(), filled_subs);        
        return tensor_type{view_factory<ValT,Cfg>::create_view_slice(impl()->impl(), filled_subs)};
    }
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,index_type>...>,int> = 0 >
    tensor_type transpose(const Subs&...subs)const{
        detail::check_transpose_subs(dim(),subs...);        
        return tensor_type{view_factory<ValT,Cfg>::create_view_transpose(impl()->impl(), shape_type{subs...})};
    }
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,index_type>...>,int> = 0 >
    tensor_type operator()(const Subs&...subs)const{
        detail::check_subdim_subs(shape(), subs...);        
        return tensor_type{view_factory<ValT,Cfg>::create_view_subdim(impl()->impl(), shape_type{subs...})};
    }            
    tensor_type operator()()const{
        detail::check_subdim_subs(shape());        
        return tensor_type{view_factory<ValT,Cfg>::create_view_subdim(impl()->impl(), shape_type{})};
    }        
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,index_type>...>,int> = 0 >
    tensor_type reshape(const Subs&...subs)const{
        detail::check_reshape_subs(size(), subs...);        
        return tensor_type{view_factory<ValT,Cfg>::create_view_reshape(impl()->impl(), shape_type{subs...})};
    }
    
private:
    
    template<typename ValT, template<typename> typename Cfg>
    class expression : public tensor<ValT,Cfg>{        
        using base_type = tensor<ValT,Cfg>;                
    public:
        expression(const base_type& base):
            base_type{base}
        {}
        auto is_cached()const{return base_type::impl()->is_cached();}
        auto is_trivial()const{return base_type::impl()->is_trivial();}
        // auto begin()const{return impl()->begin();}
        // auto end()const{return impl()->end();}
        auto trivial_at(const index_type& idx)const{return base_type::impl()->trivial_at(idx);}
    };
    
    template<typename ValT, template<typename> typename Cfg>
    class storage_tensor : public tensor<ValT,Cfg>{        
        using base_type = tensor<ValT,Cfg>;                
    public:
        storage_tensor(const base_type& base):
            base_type{base}
        {}
        // auto begin()const{return impl()->begin();}
        // auto end()const{return impl()->end();}
    };

    
};

}   //end of namespace gtensor

#endif