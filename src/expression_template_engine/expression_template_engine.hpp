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

/*
* is expression is trivial broadcast i.e. shapes of all nodes in expression tree is same
* flat index access without walkers is used to evaluate broadcast expression
* stensor and view are trivial
*/
template<typename IdxT,typename...Ops>
inline bool is_trivial(const IdxT& root_size, const Ops&...root_operands){
    return ((root_size==root_operands->size())&&...) && (is_trivial_operand(root_operands)&&...);
}
template<typename T>
inline bool is_trivial_operand(const T& operand){
    return operand->engine().is_trivial();
}

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
template<typename EngineT> auto begin_poly(EngineT& engine){
    using config_type = typename EngineT::config_type;
    using iterator_type = random_access_iterator<typename EngineT::value_type, typename config_type::difference_type>;
    if(engine.is_trivial()){
        return iterator_type{begin_trivial(engine)};
    }else{
        return iterator_type{begin_broadcast(engine)};
    }
}
template<typename EngineT> auto end_poly(EngineT& engine){
    using config_type = typename EngineT::config_type;
    using iterator_type = random_access_iterator<typename EngineT::value_type, typename config_type::difference_type>;
    if(engine.is_trivial()){
        return iterator_type{end_trivial(engine)};
    }else{
        return iterator_type{end_broadcast(engine)};
    }
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

template<typename ValT, typename CfgT>
class expression_template_engine_base{
public:
    virtual bool is_trivial()const = 0;
};

template<typename CfgT, typename StorT>
class expression_template_storage_engine :
    public expression_template_engine_base<typename StorT::value_type,CfgT>,
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

template<typename ValT, typename CfgT, typename F, typename...Ops>
class expression_template_nodispatching_engine :
    public expression_template_engine_base<ValT,CfgT>,
    private evaluating_engine<ValT,CfgT,F,std::integral_constant<std::size_t,sizeof...(Ops)>>
{
protected:
    using evaluating_engine::operands_number;
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;
public:
    using typename evaluating_engine::value_type;
    using typename evaluating_engine::config_type;
    using evaluating_engine::evaluating_engine;
    using evaluating_engine::holder;
    bool is_trivial()const override{
        return std::apply(
            [this](const auto&...operands){
                return gtensor::detail::is_trivial(holder()->size(),static_cast<Ops*>(operands.get())...);
            },
            operands()
        );
    }
    auto begin()const{return detail::begin_broadcast(*this);}
    auto end()const{return detail::end_broadcast(*this);}
    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}
    auto create_indexer()const{
        return basic_indexer<index_type, decltype(create_broadcast_indexer())>{create_broadcast_indexer()};
    }
    auto create_broadcast_indexer()const{
        return evaluating_indexer<ValT,CfgT,std::decay_t<decltype(create_broadcast_walker())>>{
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
    auto create_trivial_walker_helper(std::index_sequence<I...>)const{
        return create_trivial_walker_helper(static_cast<Ops*>(operands()[I].get())->engine().create_trivial_walker()...);
    }
    template<typename...Args>
    auto create_trivial_walker_helper(Args&&...walkers)const{
        return evaluating_trivial_walker<ValT,CfgT,F,std::decay_t<Args>...>{std::forward<Args>(walkers)...};
    }
    template<std::size_t...I>
    auto create_broadcast_walker_helper(std::index_sequence<I...>)const{
        return create_broadcast_walker_helper(static_cast<Ops*>(operands()[I].get())->engine().create_broadcast_walker()...);
    }
    template<typename...Args>
    auto create_broadcast_walker_helper(Args&&...walkers)const{
        return evaluating_walker<ValT,CfgT,F,std::decay_t<Args>...>{holder()->shape(), std::forward<Args>(walkers)...};
    }
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class expression_template_root_dispatching_engine :
    public expression_template_nodispatching_engine<ValT,CfgT,F,Ops...>
{
    using typename expression_template_nodispatching_engine::shape_type;
    using typename expression_template_nodispatching_engine::index_type;
public:
    using typename expression_template_nodispatching_engine::value_type;
    using typename expression_template_nodispatching_engine::config_type;
    using expression_template_nodispatching_engine::expression_template_nodispatching_engine;
    auto begin()const{return detail::begin_poly(*this);}
    auto end()const{return detail::end_poly(*this);}
    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}
    auto create_indexer()const{
        return basic_indexer<index_type, decltype(create_indexer_helper())>{create_indexer_helper()};
    }
private:
    auto create_indexer_helper()const{
        if (is_trivial()){
            return poly_indexer<index_type, value_type>{create_trivial_indexer()};
        }else{
            return poly_indexer<index_type, value_type>{create_broadcast_indexer()};
        }
    }
};

template<typename ValT, typename CfgT, typename DescT, typename ParentT>
class expression_template_viewing_engine :
    public expression_template_engine_base<ValT,CfgT>,
    private viewing_engine<ValT,CfgT,DescT, ParentT>
{
    using shape_type = typename CfgT::shape_type;
    using descriptor_type = typename viewing_engine::descriptor_type;
public:
    using typename viewing_engine::value_type;
    using typename viewing_engine::config_type;
    using viewing_engine::viewing_engine;
    using viewing_engine::create_indexer;
    using viewing_engine::holder;
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