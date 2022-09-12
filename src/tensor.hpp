#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "tensor_base.hpp"
#include "tensor_init_list.hpp"
#include "descriptor.hpp"
#include "engine.hpp"

namespace gtensor{

template<typename ValT, typename CfgT, typename DescT, typename EngineT>
class basic_tensor : public tensor_base<ValT, CfgT>
{
protected:
    using descriptor_type = DescT;
    using engine_type = EngineT;
    using index_type = typename tensor_base::index_type;
    using shape_type = typename tensor_base::shape_type;
    basic_tensor(engine_type&& engine__,descriptor_type&& descriptor__):
        engine_{std::move(engine__)},
        descriptor_{std::move(descriptor__)}
    {}
    const descriptor_type& descriptor()const{return descriptor_;}
    const engine_type& engine()const{return engine_;}
public:
    index_type size()const override{return descriptor_.size();}
    index_type dim()const override{return descriptor_.dim();}
    const shape_type& shape()const override{return descriptor_.shape();}
    const shape_type& strides()const override{return descriptor_.strides();}    
    std::string to_str()const override{return descriptor_.to_str();}
private:
    engine_type engine_;
    descriptor_type descriptor_;
};


//tensors with predefined constructors for descriptor and engine
template<typename ValT, typename CfgT, typename EngineT>
class storage_tensor : public basic_tensor<ValT, CfgT, basic_descriptor<CfgT>, EngineT>
{
    using engine_type = typename basic_tensor::engine_type;
    using descriptor_type = typename basic_tensor::descriptor_type;
    using index_type = typename basic_tensor::index_type;
    using shape_type = typename basic_tensor::shape_type;
    static_assert(std::is_convertible_v<engine_type*, storage_engine<ValT,CfgT>*>);

    template<typename ShT, typename Nested>
    storage_tensor(ShT&& shape, const std::initializer_list<Nested>& init_data):
        storage_tensor{detail::make_size(shape), std::forward<ShT>(shape), init_data}
    {}    
    template<typename ShT, typename InitT>
    storage_tensor(const index_type& size, ShT&& shape, const InitT& init_data):
        basic_tensor{engine_type{this, size, init_data}, descriptor_type{std::forward<ShT>(shape)}}
    {}
public:
    using value_type = ValT;

    template<typename Nested>
    storage_tensor(std::initializer_list<Nested> init_data):
        storage_tensor{detail::list_parse<index_type,shape_type>(init_data), init_data}
    {}    
    template<typename ShT, std::enable_if_t<std::is_same_v<ShT,shape_type> ,int> = 0 >
    storage_tensor(ShT&& shape, const value_type& v):
        storage_tensor{detail::make_size(shape), std::forward<ShT>(shape), v}
    {}
    template<typename ShT, std::enable_if_t<!std::is_same_v<ShT,shape_type> ,int> = 0 >
    storage_tensor(ShT&& shape, const value_type& v):
        storage_tensor{detail::make_size(shape), shape_type(shape.begin(),shape.end()), v}
    {}
};


// template<typename ValT, typename CfgT, typename EngineT>
// class evaluating_tensor : public basic_tensor<descriptor_with_libdivide<CfgT>, EngineT>
// {

// };

}   //end of namespace gtensor

#endif