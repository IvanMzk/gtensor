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
    using tensor_base_type = tensor_base<typename EngineT::value_type, typename EngineT::config_type>;
protected:
    using descriptor_type = DescT;
    using engine_type = EngineT;
    basic_tensor(engine_type&& engine__,descriptor_type&& descriptor__):
        engine_{std::move(engine__)},
        descriptor_{std::move(descriptor__)}
    {}
public:
    using typename tensor_base_type::value_type;
    using typename tensor_base_type::config_type;
    using typename tensor_base_type::size_type;
    using typename tensor_base_type::index_type;
    using typename tensor_base_type::shape_type;

    basic_tensor(const basic_tensor&) = delete;
    basic_tensor& operator=(const basic_tensor&) = delete;
    basic_tensor(basic_tensor&&) = delete;
    basic_tensor& operator=(basic_tensor&&) = delete;

    const descriptor_type& descriptor()const override{return descriptor_;}
    descriptor_type& descriptor()override{return descriptor_;}
    index_type size()const override{return descriptor_.size();}
    size_type dim()const override{return descriptor_.dim();}
    const shape_type& shape()const override{return descriptor_.shape();}
    const shape_type& strides()const override{return descriptor_.strides();}

    const engine_type& engine()const{return engine_;}
    engine_type& engine(){return engine_;}

private:
    engine_type engine_;
    descriptor_type descriptor_;
};

template<typename EngineT>
class storage_tensor : public basic_tensor<basic_descriptor<typename EngineT::config_type>, EngineT>
{
    using basic_tensor_base = basic_tensor<basic_descriptor<typename EngineT::config_type>, EngineT>;
public:
    using typename basic_tensor_base::config_type;
    using typename basic_tensor_base::value_type;
    using typename basic_tensor_base::size_type;
    using typename basic_tensor_base::index_type;
    using typename basic_tensor_base::shape_type;
private:
    using typename basic_tensor_base::engine_type;
    using typename basic_tensor_base::descriptor_type;

    template<typename ShT, typename Nested>
    storage_tensor(ShT&& shape, std::initializer_list<Nested> init_data):
        storage_tensor{detail::make_size(shape), std::forward<ShT>(shape), init_data}
    {}
    template<typename ShT, typename...InitT>
    storage_tensor(const index_type& size, ShT&& shape, InitT...init_data):
        basic_tensor_base{engine_type{this, size, init_data...}, descriptor_type{std::forward<ShT>(shape)}}
    {}
public:
    template<typename Nested>
    storage_tensor(std::initializer_list<Nested> init_data):
        storage_tensor{detail::list_parse<size_type,shape_type>(init_data), init_data}
    {}
    template<typename ShT>
    storage_tensor(ShT&& shape, const value_type& v):
        storage_tensor{detail::make_size<index_type>(shape), std::forward<ShT>(shape), v}
    {}
    template<typename ShT, typename ItT>
    storage_tensor(ShT&& shape, ItT begin, ItT end):
        storage_tensor{detail::make_size<index_type>(shape), std::forward<ShT>(shape), begin, end}
    {}

    //inplace
    template<typename ShT>
    void resize(ShT&& shape){
        basic_tensor_base::descriptor() = descriptor_type{std::forward<ShT>(shape)};
        basic_tensor_base::engine().resize(basic_tensor_base::descriptor().size());
    }
};

template<typename EngineT>
class evaluating_tensor : public basic_tensor<basic_descriptor<typename EngineT::config_type>, EngineT>
{
    using basic_tensor_base = basic_tensor<basic_descriptor<typename EngineT::config_type>, EngineT>;
public:
    using typename basic_tensor_base::config_type;
    using typename basic_tensor_base::value_type;
    using typename basic_tensor_base::size_type;
    using typename basic_tensor_base::index_type;
    using typename basic_tensor_base::shape_type;
private:
    using typename basic_tensor_base::engine_type;
    using typename basic_tensor_base::descriptor_type;

    template<typename F, typename...Operands>
    evaluating_tensor(shape_type&& shape, F&& f, Operands&&...operands):
        basic_tensor_base{engine_type{this, std::forward<F>(f),std::forward<Operands>(operands)...}, descriptor_type{std::move(shape)}}
    {}
public:
    template<typename F, typename...Operands>
    evaluating_tensor(F&& f, Operands&&...operands):
        evaluating_tensor{detail::broadcast_shape<shape_type>(operands->shape()...),std::forward<F>(f),std::forward<Operands>(operands)...}
    {}
};

template<typename DescT, typename EngineT>
class viewing_tensor : public basic_tensor<DescT, EngineT>
{
    using basic_tensor_base = basic_tensor<DescT, EngineT>;
    using typename basic_tensor_base::engine_type;
    using typename basic_tensor_base::descriptor_type;
public:
    using typename basic_tensor_base::config_type;
    using typename basic_tensor_base::value_type;
    using typename basic_tensor_base::size_type;
    using typename basic_tensor_base::index_type;
    using typename basic_tensor_base::shape_type;
public:
    template<typename U>
    viewing_tensor(descriptor_type&& descriptor, U&& parent):
        basic_tensor_base{engine_type{this,std::forward<U>(parent)},std::move(descriptor)}
    {}
};

}   //end of namespace gtensor

#endif