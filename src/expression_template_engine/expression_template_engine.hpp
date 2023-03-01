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
    using iterator_type = broadcast_shape_iterator<typename EngineT::config_type, decltype(engine.create_walker())>;
    return iterator_type{engine.create_walker(), shape, make_strides_div<typename EngineT::config_type>(shape)};
}
template<typename EngineT, typename ShT> auto end_broadcast(EngineT& engine, const ShT& shape){
    using iterator_type = broadcast_shape_iterator<typename EngineT::config_type, decltype(engine.create_walker())>;
    return iterator_type{engine.create_walker(), shape, make_strides_div<typename EngineT::config_type>(shape), make_size(shape)};
}
template<typename EngineT, typename ShT> auto rbegin_broadcast(EngineT& engine, const ShT& shape){
    using iterator_type = reverse_broadcast_shape_iterator<typename EngineT::config_type, decltype(engine.create_walker())>;
    return iterator_type{engine.create_walker(), shape, make_strides_div<typename EngineT::config_type>(shape), make_size(shape)};
}
template<typename EngineT, typename ShT> auto rend_broadcast(EngineT& engine, const ShT& shape){
    using iterator_type = reverse_broadcast_shape_iterator<typename EngineT::config_type, decltype(engine.create_walker())>;
    return iterator_type{engine.create_walker(), shape, make_strides_div<typename EngineT::config_type>(shape)};
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

template<typename> struct is_striding_descriptor : public std::false_type{};
template<typename...Ts> struct is_striding_descriptor<basic_descriptor<Ts...>> : public std::true_type{};
template<typename...Ts> struct is_striding_descriptor<descriptor_with_offset<Ts...>> : public std::true_type{};
template<typename...Ts> struct is_striding_descriptor<converting_descriptor<Ts...>> : public std::true_type{};
template<typename T> using is_striding_descriptor_t = typename is_striding_descriptor<T>::type;
template<typename T> constexpr bool is_striding_descriptor_v = is_striding_descriptor_t<T>();
}   //end of namespace detail

template<typename CfgT, typename StorT>
class expression_template_storage_engine : private storage_engine<CfgT,StorT>
{
    using storage_engine_base = storage_engine<CfgT,StorT>;
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;
public:
    using typename storage_engine_base::value_type;
    using typename storage_engine_base::config_type;
    using storage_engine_base::storage_engine_base;
    using storage_engine_base::holder;
    using storage_engine_base::begin;
    using storage_engine_base::end;
    using storage_engine_base::rbegin;
    using storage_engine_base::rend;
    using storage_engine_base::create_indexer;
    bool is_trivial()const{return true;}
    //broadcasting iterators
    auto begin_broadcast(const shape_type& shape){return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape){return detail::end_broadcast(*this, shape);}
    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}
    auto rbegin_broadcast(const shape_type& shape){return detail::rbegin_broadcast(*this, shape);}
    auto rend_broadcast(const shape_type& shape){return detail::rend_broadcast(*this, shape);}
    auto rbegin_broadcast(const shape_type& shape)const{return detail::rbegin_broadcast(*this, shape);}
    auto rend_broadcast(const shape_type& shape)const{return detail::rend_broadcast(*this, shape);}

    auto create_walker()const{return create_walker_helper(*this);}
    auto create_walker(){return create_walker_helper(*this);}
    auto create_trivial_indexer()const{return create_indexer();}
    auto create_trivial_indexer(){return create_indexer();}
private:
    template<typename U> static auto create_walker_helper(U& instance){
        return walker<CfgT, std::decay_t<decltype(instance.create_indexer())>>{
            instance.holder()->shape(),
            instance.holder()->descriptor().strides(),
            instance.holder()->descriptor().reset_strides(),
            index_type{0},
            instance.create_indexer()
        };
    }
};

template<typename CfgT, typename F, typename...Operands>
class expression_template_evaluating_engine : private evaluating_engine<CfgT,F,Operands...>
{
    using evaluating_engine_base = evaluating_engine<CfgT,F,Operands...>;
    using evaluating_engine_base::operands_number;
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;
    using evaluating_engine_base::operation;
public:
    using typename evaluating_engine_base::value_type;
    using typename evaluating_engine_base::config_type;
    using evaluating_engine_base::evaluating_engine;
    using evaluating_engine_base::holder;
    //true if all nodes in expression tree are the same shape, false otherwise
    bool is_trivial()const{
        return is_trivial_helper(std::make_index_sequence<operands_number>{});
    }
    auto begin(){return detail::begin_broadcast(*this);}
    auto end(){return detail::end_broadcast(*this);}
    auto begin()const{return detail::begin_broadcast(*this);}
    auto end()const{return detail::end_broadcast(*this);}
    auto rbegin(){return detail::rbegin_broadcast(*this);}
    auto rend(){return detail::rend_broadcast(*this);}
    auto rbegin()const{return detail::rbegin_broadcast(*this);}
    auto rend()const{return detail::rend_broadcast(*this);}

    auto begin_broadcast(const shape_type& shape){return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape){return detail::end_broadcast(*this, shape);}
    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}
    auto rbegin_broadcast(const shape_type& shape){return detail::rbegin_broadcast(*this, shape);}
    auto rend_broadcast(const shape_type& shape){return detail::rend_broadcast(*this, shape);}
    auto rbegin_broadcast(const shape_type& shape)const{return detail::rbegin_broadcast(*this, shape);}
    auto rend_broadcast(const shape_type& shape)const{return detail::rend_broadcast(*this, shape);}

    auto create_indexer()const{
        if (is_trivial()){
            return poly_indexer<index_type, value_type>{create_trivial_indexer()};
        }else{
            return poly_indexer<index_type, value_type>{create_walking_indexer()};
        }
    }
    auto create_walker(){return create_walker_helper(*this, std::make_index_sequence<operands_number>{});}
    auto create_walker()const{return create_walker_helper(*this, std::make_index_sequence<operands_number>{});}
    auto create_trivial_indexer()const{return create_trivial_indexer_helper(std::make_index_sequence<operands_number>{});}
private:
    template<std::size_t I> auto& operand(){return evaluating_engine_base::template operand<I>();}
    template<std::size_t I> const auto& operand()const{return evaluating_engine_base::template operand<I>();}
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
        return evaluating_trivial_indexer<CfgT,F,decltype(operand<I>().engine().create_trivial_indexer())...>{
            operation(), operand<I>().engine().create_trivial_indexer()...
        };
    }
    template<typename U, std::size_t...I>
    static auto create_walker_helper(U& instance, std::index_sequence<I...>){
        return evaluating_walker<CfgT,F,decltype(instance.template operand<I>().engine().create_walker())...>{
            instance.holder()->shape(), instance.operation(), instance.template operand<I>().engine().create_walker()...
        };
    }
};

template<typename CfgT, typename DescT, typename ParentT>
class expression_template_viewing_engine : private viewing_engine<CfgT,DescT, ParentT>
{
    using viewing_engine_base = viewing_engine<CfgT,DescT, ParentT>;
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;
    using descriptor_type = typename viewing_engine_base::descriptor_type;
public:
    using typename viewing_engine_base::value_type;
    using typename viewing_engine_base::config_type;
    using viewing_engine_base::viewing_engine;
    using viewing_engine_base::create_indexer;
    using viewing_engine_base::holder;
    bool is_trivial()const{return true;}

    auto begin()const{return begin(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto end()const{return end(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto begin(){return begin(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto end(){return end(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto rbegin()const{return rbegin(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto rend()const{return rend(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto rbegin(){return rbegin(*this, detail::is_converting_descriptor_t<descriptor_type>{});}
    auto rend(){return rend(*this, detail::is_converting_descriptor_t<descriptor_type>{});}

    //broadcasting iterators
    auto begin_broadcast(const shape_type& shape){return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape){return detail::end_broadcast(*this, shape);}
    auto begin_broadcast(const shape_type& shape)const{return detail::begin_broadcast(*this, shape);}
    auto end_broadcast(const shape_type& shape)const{return detail::end_broadcast(*this, shape);}
    auto rbegin_broadcast(const shape_type& shape){return detail::rbegin_broadcast(*this, shape);}
    auto rend_broadcast(const shape_type& shape){return detail::rend_broadcast(*this, shape);}
    auto rbegin_broadcast(const shape_type& shape)const{return detail::rbegin_broadcast(*this, shape);}
    auto rend_broadcast(const shape_type& shape)const{return detail::rend_broadcast(*this, shape);}

    //return walker with parent indexer (chain of indexers starts from parent's indexer)
    auto create_walker()const{return create_walker_helper(*this, detail::is_striding_descriptor_t<descriptor_type>{});}
    auto create_walker(){return create_walker_helper(*this, detail::is_striding_descriptor_t<descriptor_type>{});}
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
    //slice, transpose view reverse iterator
    template<typename U>
    static auto rbegin(U& instance, std::true_type){return detail::rbegin_broadcast(instance);}
    template<typename U>
    static auto rend(U& instance, std::true_type){return detail::rend_broadcast(instance);}
    //reshape, subdim view reverse iterator
    template<typename U>
    static auto rbegin(U& instance, std::false_type){return detail::rbegin_trivial(instance);}
    template<typename U>
    static auto rend(U& instance, std::false_type){return detail::rend_trivial(instance);}
    template<typename U>
    static auto create_walker_helper(U& instance, std::true_type){
        return walker<CfgT, std::decay_t<decltype(instance.parent().engine().create_indexer())>>{
            instance.holder()->shape(),
            instance.holder()->descriptor().cstrides(),
            instance.holder()->descriptor().reset_cstrides(),
            instance.holder()->descriptor().offset(),
            instance.parent().engine().create_indexer()
        };
    }
    template<typename U>
    static auto create_walker_helper(U& instance, std::false_type){
        return walker<CfgT, std::decay_t<decltype(instance.create_indexer())>>{
            instance.holder()->shape(),
            instance.holder()->descriptor().strides(),
            instance.holder()->descriptor().reset_strides(),
            index_type{0},
            instance.create_indexer()
        };
    }
};

}   //end of namespace gtensor

#endif