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
    using iterator_type = broadcast_iterator<typename EngineT::config_type, decltype(engine.create_walker())>;
    return iterator_type{engine.create_walker(), shape, make_dividers<EngineT::config_type>(shape)};
}
template<typename EngineT, typename ShT> auto end_broadcast(EngineT& engine, const ShT& shape){
    using iterator_type = broadcast_iterator<typename EngineT::config_type, decltype(engine.create_walker())>;
    return iterator_type{engine.create_walker(), shape, make_dividers<EngineT::config_type>(shape), make_size(shape)};
}
template<typename EngineT> auto begin_broadcast(EngineT& engine){
    using iterator_type = broadcast_iterator<typename EngineT::config_type, decltype(engine.create_walker())>;
    return iterator_type{engine.create_walker(), engine.holder()->shape(), engine.holder()->descriptor().strides_div()};
}
template<typename EngineT> auto end_broadcast(EngineT& engine){
    using iterator_type = broadcast_iterator<typename EngineT::config_type, decltype(engine.create_walker())>;
    return iterator_type{engine.create_walker(), engine.holder()->shape(), engine.holder()->descriptor().strides_div(), engine.holder()->size()};
}
template<typename EngineT> auto rbegin_broadcast(EngineT& engine){
    using iterator_type = reverse_broadcast_iterator<typename EngineT::config_type, decltype(engine.create_walker())>;
    return iterator_type{engine.create_walker(), engine.holder()->shape(), engine.holder()->descriptor().strides_div(), engine.holder()->size()};
}
template<typename EngineT> auto rend_broadcast(EngineT& engine){
    using iterator_type = reverse_broadcast_iterator<typename EngineT::config_type, decltype(engine.create_walker())>;
    return iterator_type{engine.create_walker(), engine.holder()->shape(), engine.holder()->descriptor().strides_div()};
}
template<typename EngineT> auto begin_trivial(EngineT& engine){
    using iterator_type = trivial_broadcast_iterator<typename EngineT::config_type, decltype(engine.create_trivial_indexer())>;
    return iterator_type{engine.create_trivial_indexer()};
}
template<typename EngineT> auto end_trivial(EngineT& engine){
    using iterator_type = trivial_broadcast_iterator<typename EngineT::config_type, decltype(engine.create_trivial_indexer())>;
    return iterator_type{engine.create_trivial_indexer(), engine.holder()->size()};
}
template<typename EngineT> auto rbegin_trivial(EngineT& engine){
    using iterator_type = reverse_trivial_broadcast_iterator<typename EngineT::config_type, decltype(engine.create_trivial_indexer())>;
    return iterator_type{engine.create_trivial_indexer(), engine.holder()->size()};
}
template<typename EngineT> auto rend_trivial(EngineT& engine){
    using iterator_type = reverse_trivial_broadcast_iterator<typename EngineT::config_type, decltype(engine.create_trivial_indexer())>;
    return iterator_type{engine.create_trivial_indexer()};
}

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
    using storage_engine::rbegin;
    using storage_engine::rend;
    using storage_engine::create_indexer;
    bool is_trivial()const override{return true;}
    //broadcasting iterators
    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}
    auto begin_broadcast(const shape_type& shape){return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape){return detail::end_broadcast(*this, shape);}

    auto create_walker()const{return create_walker_helper(*this);}
    auto create_walker(){return create_walker_helper(*this);}
    auto create_trivial_indexer()const{return create_indexer();}
    auto create_trivial_indexer(){return create_indexer();}
private:
    template<typename U> static auto create_walker_helper(U& instance){
        return walker<CfgT, std::decay_t<decltype(instance.create_indexer())>>{
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
    auto rbegin()const{return detail::rbegin_broadcast(*this);}
    auto rend()const{return detail::rend_broadcast(*this);}

    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}

    auto create_indexer()const{
        if (is_trivial()){
            return poly_indexer<index_type, value_type>{create_trivial_indexer()};
        }else{
            return poly_indexer<index_type, value_type>{create_walking_indexer()};
        }
    }
    auto create_walker()const{
        return create_walker_helper(std::make_index_sequence<operands_number>{});
    }
    auto create_trivial_indexer()const{
        return create_trivial_indexer_helper(std::make_index_sequence<operands_number>{});
    }
private:
    auto create_walking_indexer()const{
        return evaluating_indexer<CfgT,decltype(create_walker())>{
            holder()->descriptor().strides_div(),
            create_walker()
        };
    }
    template<std::size_t...I>
    auto is_trivial_helper(std::index_sequence<I...>)const{
        return ((holder()->size()==operand<I>().size())&&...) && (operand<I>().engine().is_trivial()&&...);
    }
    template<std::size_t...I>
    auto create_trivial_indexer_helper(std::index_sequence<I...>)const{
        return evaluating_trivial_indexer<CfgT,F,decltype(operand<I>().engine().create_trivial_indexer())...>{operand<I>().engine().create_trivial_indexer()...};
    }
    template<std::size_t...I>
    auto create_walker_helper(std::index_sequence<I...>)const{
        return evaluating_walker<CfgT,F,decltype(operand<I>().engine().create_walker())...>{holder()->shape(), operand<I>().engine().create_walker()...};
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

    auto rbegin()const{return rbegin(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto rend()const{return rend(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto rbegin(){return rbegin(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto rend(){return rend(*this, detail::is_converting_descriptor_t<descriptor_type>{});}

    //broadcasting iterators
    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}
    auto begin_broadcast(const shape_type& shape){return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape){return detail::end_broadcast(*this, shape);}

    //both broadcast and trivial walker utilize parent's indexer subscript operator to access data
    //return walker with parent indexer (chain of indexers starts from parent's indexer)
    auto create_walker()const{return create_walker_helper(*this);}
    auto create_walker(){return create_walker_helper(*this);}
    //return indexer, chain of indexers starts from this view indexer
    auto create_trivial_indexer()const{return create_indexer();}
    auto create_trivial_indexer(){return create_indexer();}
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

    //slice, transpose view iterator
    template<typename U>
    static auto rbegin(U& instance, std::true_type){return detail::rbegin_broadcast(instance);}
    template<typename U>
    static auto rend(U& instance, std::true_type){return detail::rend_broadcast(instance);}
    //reshape, subdim view iterator
    template<typename U>
    static auto rbegin(U& instance, std::false_type){return detail::rbegin_trivial(instance);}
    template<typename U>
    static auto rend(U& instance, std::false_type){return detail::rend_trivial(instance);}

    template<typename U>
    static auto create_walker_helper(U& instance){
        return walker<CfgT, std::decay_t<decltype(instance.parent()->engine().create_indexer())>>{
            instance.holder()->shape(),
            instance.holder()->descriptor().cstrides(),
            instance.holder()->descriptor().reset_cstrides(),
            instance.holder()->descriptor().offset(),
            instance.parent()->engine().create_indexer()
        };
    }
};

}   //end of namespace gtensor

#endif