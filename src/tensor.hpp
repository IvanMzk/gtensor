#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "tensor_base.hpp"
#include "tensor_init_list.hpp"
#include "descriptor.hpp"
#include "engine.hpp"
#include "broadcast.hpp"

namespace gtensor{

template<typename DescT, typename EngineT>
class basic_tensor : public tensor_base<typename EngineT::value_type, typename EngineT::config_type>
{
protected:
    using descriptor_type = DescT;
    using engine_type = EngineT;
    using typename tensor_base::value_type;
    using typename tensor_base::config_type;
    using typename tensor_base::index_type;
    using typename tensor_base::shape_type;
    basic_tensor(engine_type&& engine__,descriptor_type&& descriptor__):
        engine_{std::move(engine__)},
        descriptor_{std::move(descriptor__)}
    {}
public:
    const descriptor_type& descriptor()const{return descriptor_;}
    const engine_type& engine()const override{return engine_;}
    engine_type& engine()override{return engine_;}
    index_type size()const override{return descriptor_.size();}
    index_type dim()const override{return descriptor_.dim();}
    const shape_type& shape()const override{return descriptor_.shape();}
    const shape_type& strides()const override{return descriptor_.strides();}
    const shape_type& reset_strides()const override{return descriptor_.reset_strides();}
    std::string to_str()const override{return descriptor_.to_str();}
private:
    engine_type engine_;
    descriptor_type descriptor_;
};

template<typename EngineT>
class storage_tensor : public basic_tensor<basic_descriptor<typename EngineT::config_type>, EngineT>
{
public:
    using engine_type = typename basic_tensor::engine_type;
    using typename basic_tensor::value_type;
    using typename basic_tensor::config_type;
private:
    using typename basic_tensor::descriptor_type;
    using typename basic_tensor::index_type;
    using typename basic_tensor::shape_type;
    //static_assert(std::is_convertible_v<engine_type*, storage_engine<value_type,config_type>*>);

    template<typename ShT, typename Nested>
    storage_tensor(ShT&& shape, std::initializer_list<Nested> init_data):
        storage_tensor{detail::make_size(shape), std::forward<ShT>(shape), init_data}
    {}
    template<typename ShT, typename...InitT>
    storage_tensor(const index_type& size, ShT&& shape, InitT...init_data):
        basic_tensor{engine_type{this, size, init_data...}, descriptor_type{std::forward<ShT>(shape)}}
    {}
public:

    template<typename Nested>
    storage_tensor(std::initializer_list<Nested> init_data):
        storage_tensor{detail::list_parse<index_type,shape_type>(init_data), init_data}
    {}
    template<typename ShT>
    storage_tensor(ShT&& shape, const value_type& v):
        storage_tensor{detail::make_size(shape), std::forward<ShT>(shape), v}
    {}
    template<typename ShT, typename ItT>
    storage_tensor(ShT&& shape, ItT begin, ItT end):
        storage_tensor{detail::make_size(shape), std::forward<ShT>(shape), begin, end}
    {}
};

template<typename EngineT>
class evaluating_tensor : public basic_tensor<basic_descriptor<typename EngineT::config_type>, EngineT>
{
public:
    using engine_type = typename basic_tensor::engine_type;
    using typename basic_tensor::value_type;
    using typename basic_tensor::config_type;
private:
    using typename basic_tensor::descriptor_type;
    using typename basic_tensor::index_type;
    using typename basic_tensor::shape_type;

    template<typename F, typename...Operands>
    evaluating_tensor(shape_type&& shape, F&& f, Operands&&...operands):
        basic_tensor{engine_type{this, std::forward<F>(f),std::forward<Operands>(operands)...}, descriptor_type{std::move(shape)}}
    {}
public:
    template<typename F, typename...Operands>
    evaluating_tensor(F&& f, Operands&&...operands):
        evaluating_tensor{detail::broadcast_shape<shape_type>(operands->shape()...),std::forward<F>(f),std::forward<Operands>(operands)...}
    {
        //static_assert(std::is_convertible_v<engine_type*, evaluating_engine<value_type,config_type,std::decay_t<F>,std::decay_t<Args>::element_type...>*>);
    }
};

template<typename DescT, typename EngineT>
class viewing_tensor : public basic_tensor<DescT, EngineT>
{
public:
    using engine_type = typename basic_tensor::engine_type;
    using typename basic_tensor::value_type;
    using typename basic_tensor::config_type;
private:
    using typename basic_tensor::descriptor_type;
    using typename basic_tensor::index_type;
    using typename basic_tensor::shape_type;
public:
    template<typename U>
    viewing_tensor(descriptor_type&& descriptor, U&& parent):
        basic_tensor{engine_type{this,std::forward<U>(parent)},std::move(descriptor)}
    {
        //static_assert(std::is_convertible_v<engine_type*, viewing_engine<value_type,config_type,std::decay_t<U>::element_type>*>);
    }
};

}   //end of namespace gtensor

#endif