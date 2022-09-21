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
    using typename tensor_base::index_type;
    using typename tensor_base::shape_type;
    basic_tensor(engine_type&& engine__,descriptor_type&& descriptor__):
        engine_{std::move(engine__)},
        descriptor_{std::move(descriptor__)}
    {}
public:
    const descriptor_type& descriptor()const{return descriptor_;}
    const engine_type& engine()const{return engine_;}
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

template<typename ValT, typename CfgT, typename EngineT>
class storage_tensor : public basic_tensor<ValT, CfgT, basic_descriptor<CfgT>, EngineT>
{
public:
    using value_type = ValT;
    using engine_type = typename basic_tensor::engine_type;
private:
    using typename basic_tensor::descriptor_type;
    using typename basic_tensor::index_type;
    using typename basic_tensor::shape_type;
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

template<typename ValT, typename CfgT, typename EngineT>
class evaluating_tensor : public basic_tensor<ValT, CfgT, descriptor_with_libdivide<CfgT>, EngineT>
{
public:
    using value_type = ValT;
    using engine_type = typename basic_tensor::engine_type;
private:
    using typename basic_tensor::descriptor_type;
    using typename basic_tensor::index_type;
    using typename basic_tensor::shape_type;

    template<typename F, typename...Args>
    evaluating_tensor(shape_type&& shape, F&& f, Args&&...operands):
        basic_tensor{engine_type{this, std::forward<F>(f),std::forward<Args>(operands)...}, descriptor_type{std::move(shape)}}
    {}
public:
    template<typename F, typename...Args>
    evaluating_tensor(F&& f, Args&&...operands):
        evaluating_tensor{detail::broadcast(operands->shape()...),std::forward<F>(f),std::forward<Args>(operands)...}
    {
        //static_assert(std::is_convertible_v<engine_type*, evaluating_engine<ValT,CfgT,std::decay_t<F>,std::decay_t<Args>::element_type...>*>);
    }
};

template<typename ValT, typename CfgT, typename DescT, typename EngineT>
class viewing_tensor : public basic_tensor<ValT, CfgT, DescT, EngineT>
{
public:
    using value_type = ValT;
    using engine_type = typename basic_tensor::engine_type;
private:
    using typename basic_tensor::descriptor_type;
    using typename basic_tensor::index_type;
    using typename basic_tensor::shape_type;
public:
    template<typename U>
    viewing_tensor(descriptor_type&& descriptor, U&& parent):
        basic_tensor{engine_type{this,std::forward<U>(parent)},std::move(descriptor)}
    {
        static_assert(std::is_convertible_v<engine_type*, viewing_engine<ValT,CfgT,std::decay_t<U>::element_type>*>);
    }
};





}   //end of namespace gtensor

#endif