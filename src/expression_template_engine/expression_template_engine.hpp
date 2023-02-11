#ifndef EXPRESSION_TEMPLATE_ENGINE_HPP_
#define EXPRESSION_TEMPLATE_ENGINE_HPP_

#include <memory>
#include <array>
#include "broadcast.hpp"
#include "engine.hpp"
#include "evaluating_walker.hpp"
#include "iterator.hpp"

namespace gtensor{

namespace detail{

template<typename EngineT, typename ShT> auto begin_broadcast(EngineT& engine, const ShT& shape){
    using iterator_type = broadcast_iterator<typename EngineT::config_type, decltype(engine.create_broadcast_walker())>;
    return iterator_type{engine.create_broadcast_walker(), shape, make_dividers<EngineT::config_type>(shape)};
}
template<typename EngineT, typename ShT> auto end_broadcast(EngineT& engine, const ShT& shape){
    using iterator_type = broadcast_iterator<typename EngineT::config_type, decltype(engine.create_broadcast_walker())>;
    return iterator_type{engine.create_broadcast_walker(), shape, make_dividers<EngineT::config_type>(shape), make_size(shape)};
}
template<typename EngineT> auto begin_broadcast(EngineT& engine){
    using iterator_type = broadcast_iterator<typename EngineT::config_type, decltype(engine.create_broadcast_walker())>;
    return iterator_type{engine.create_broadcast_walker(), engine.holder()->shape(), engine.holder()->descriptor().strides_div()};
}
template<typename EngineT> auto end_broadcast(EngineT& engine){
    using iterator_type = broadcast_iterator<typename EngineT::config_type, decltype(engine.create_broadcast_walker())>;
    return iterator_type{engine.create_broadcast_walker(), engine.holder()->shape(), engine.holder()->descriptor().strides_div(), engine.holder()->size()};
}
template<typename EngineT> auto begin_trivial(EngineT& engine){
    using iterator_type = trivial_broadcast_iterator<typename EngineT::config_type, decltype(engine.create_trivial_walker())>;
    return iterator_type{engine.create_trivial_walker()};
}
template<typename EngineT> auto end_trivial(EngineT& engine){
    using iterator_type = trivial_broadcast_iterator<typename EngineT::config_type, decltype(engine.create_trivial_walker())>;
    return iterator_type{engine.create_trivial_walker(), engine.holder()->size()};
}
template<typename...Ts> auto begin_broadcast(const expression_template_storage_engine<Ts...>& engine){return engine.begin();}
template<typename...Ts> auto end_broadcast(const expression_template_storage_engine<Ts...>& engine){return engine.end();}
template<typename...Ts> auto begin_trivial(const expression_template_storage_engine<Ts...>& engine){return engine.begin();}
template<typename...Ts> auto end_trivial(const expression_template_storage_engine<Ts...>& engine){return engine.end();}
template<typename...Ts> auto begin_broadcast(expression_template_storage_engine<Ts...>& engine){return engine.begin();}
template<typename...Ts> auto end_broadcast(expression_template_storage_engine<Ts...>& engine){return engine.end();}
template<typename...Ts> auto begin_trivial(expression_template_storage_engine<Ts...>& engine){return engine.begin();}
template<typename...Ts> auto end_trivial(expression_template_storage_engine<Ts...>& engine){return engine.end();}

template<typename> struct is_converting_descriptor : public std::false_type{};
template<typename...Ts> struct is_converting_descriptor<converting_descriptor<Ts...>> : public std::true_type{};
template<typename T> using is_converting_descriptor_t = typename is_converting_descriptor<T>::type;
template<typename T> constexpr bool is_converting_descriptor_v = is_converting_descriptor_t<T>();

}   //end of namespace detail

class expression_template_engine_base
{
public:

    //storage_tensor and viewing_tensor are trivial broadcast
    //evaluating_tensor is trivial broadcast when shapes of all nodes in expression tree are the same
    virtual bool is_trivial()const = 0;
};

template<typename CfgT, typename StorT>
class expression_template_storage_engine :
    public expression_template_engine_base,
    private storage_engine<CfgT,StorT>
{
    using shape_type = typename CfgT::shape_type;
public:
    using typename storage_engine::value_type;
    using typename storage_engine::config_type;
    using storage_engine::storage_engine;
    using storage_engine::holder;
    using storage_engine::begin;
    using storage_engine::end;
    using storage_engine::create_indexer;
    bool is_trivial()const override{return true;}
    //broadcasting iterators
    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}
    auto begin_broadcast(const shape_type& shape){return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape){return detail::end_broadcast(*this, shape);}

    auto create_broadcast_walker()const{return create_broadcast_walker_helper(*this);}
    auto create_broadcast_walker(){return create_broadcast_walker_helper(*this);}
    auto create_trivial_walker()const{return create_trivial_walker_helper(*this);}
    auto create_trivial_walker(){return create_trivial_walker_helper(*this);}
private:
    template<typename U> static auto create_trivial_walker_helper(U& instance){
        return instance.create_indexer();
    }
    template<typename U> static auto create_broadcast_walker_helper(U& instance){
        return broadcast_indexer_walker<CfgT, std::decay_t<decltype(instance.create_indexer())>>{
            instance.holder()->shape(),
            instance.holder()->descriptor().cstrides(),
            instance.holder()->descriptor().reset_cstrides(),
            instance.holder()->descriptor().offset(),
            instance.create_indexer()
        };
    }
};

template<typename CfgT, typename F, typename...Operands>
class expression_template_evaluating_engine :
    public expression_template_engine_base,
    private evaluating_engine<CfgT,F,Operands...>
{
    using evaluating_engine_base = evaluating_engine<CfgT,F,Operands...>;
    using evaluating_engine_base::operands_number;
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;
    using evaluating_engine_base::operands;
    using evaluating_engine_base::operand;
public:
    using typename evaluating_engine_base::value_type;
    using typename evaluating_engine_base::config_type;
    using evaluating_engine_base::evaluating_engine;
    using evaluating_engine_base::holder;
    bool is_trivial()const override{
        return is_trivial_helper(std::make_index_sequence<operands_number>{});
    }
    auto begin()const{return detail::begin_broadcast(*this);}
    auto end()const{return detail::end_broadcast(*this);}
    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}
    auto create_indexer()const{
        return basic_indexer<index_type, decltype(create_indexer_helper())>{create_indexer_helper()};
    }
    auto create_broadcast_indexer()const{
        return evaluating_indexer<value_type,CfgT,std::decay_t<decltype(create_broadcast_walker())>>{
            holder()->descriptor().strides_div(),
            create_broadcast_walker()
        };
    }
    auto create_trivial_indexer()const{
        return create_trivial_walker();
    }
    auto create_broadcast_walker()const{
        return create_broadcast_walker_helper(std::make_index_sequence<operands_number>{});
    }
    auto create_trivial_walker()const{
        return create_trivial_walker_helper(std::make_index_sequence<operands_number>{});
    }
private:
    template<std::size_t...I>
    auto is_trivial_helper(std::index_sequence<I...>)const{
        return ((holder()->size()==operand<I>().size())&&...) && (operand<I>().engine().is_trivial()&&...);
    }
    template<std::size_t...I>
    auto create_trivial_walker_helper(std::index_sequence<I...>)const{
        return evaluating_trivial_walker<CfgT,F,std::decay_t<decltype(operand<I>().engine().create_trivial_walker())>...>{operand<I>().engine().create_trivial_walker()...};
    }
    template<std::size_t...I>
    auto create_broadcast_walker_helper(std::index_sequence<I...>)const{
        return evaluating_walker<CfgT,F,std::decay_t<decltype(operand<I>().engine().create_broadcast_walker())>...>{holder()->shape(), operand<I>().engine().create_broadcast_walker()...};
    }
    auto create_indexer_helper()const{
        if (is_trivial()){
            return poly_indexer<index_type, value_type>{create_trivial_indexer()};
        }else{
            return poly_indexer<index_type, value_type>{create_broadcast_indexer()};
        }
    }
};

template<typename CfgT, typename DescT, typename ParentT>
class expression_template_viewing_engine :
    public expression_template_engine_base,
    private viewing_engine<CfgT,DescT, ParentT>
{
    using viewing_engine_base = viewing_engine<CfgT,DescT, ParentT>;
    using shape_type = typename CfgT::shape_type;
    using descriptor_type = typename viewing_engine::descriptor_type;
public:
    using typename viewing_engine_base::value_type;
    using typename viewing_engine_base::config_type;
    using viewing_engine_base::viewing_engine;
    using viewing_engine_base::create_indexer;
    using viewing_engine_base::holder;
    bool is_trivial()const override{return true;}

    auto begin()const{return begin(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto end()const{return end(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto begin(){return begin(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto end(){return end(*this, detail::is_converting_descriptor_t<descriptor_type>{});}

    //broadcasting iterators
    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}
    auto begin_broadcast(const shape_type& shape){return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape){return detail::end_broadcast(*this, shape);}

    //both broadcast and trivial walker utilize parent's indexer subscript operator to access data
    //return broadcast_indexer_walker with parent indexer (chain of indexers starts from parent's indexer)
    auto create_broadcast_walker()const{return create_broadcast_walker_helper(*this);}
    auto create_broadcast_walker(){return create_broadcast_walker_helper(*this);}
    //return indexer, chain of indexers starts from this view indexer
    auto create_trivial_walker()const{return create_trivial_walker_helper(*this);}
    auto create_trivial_walker(){return create_trivial_walker_helper(*this);}
private:
    //slice, transpose view iterator
    template<typename U>
    static auto begin(U& instance, std::true_type){return detail::begin_broadcast(instance);}
    template<typename U>
    static auto end(U& instance, std::true_type){return detail::end_broadcast(instance);}
    //reshape, subdim view iterator
    template<typename U>
    static auto begin(U& instance, std::false_type){return detail::begin_trivial(instance);}
    template<typename U>
    static auto end(U& instance, std::false_type){return detail::end_trivial(instance);}

    template<typename U>
    static auto create_broadcast_walker_helper(U& instance){
        return broadcast_indexer_walker<CfgT, std::decay_t<decltype(instance.parent()->engine().create_indexer())>>{
            instance.holder()->shape(),
            instance.holder()->descriptor().cstrides(),
            instance.holder()->descriptor().reset_cstrides(),
            instance.holder()->descriptor().offset(),
            instance.parent()->engine().create_indexer()
        };
    }
    template<typename U>
    static auto create_trivial_walker_helper(U& instance){
        return instance.create_indexer();
    }
};

}   //end of namespace gtensor

#endif