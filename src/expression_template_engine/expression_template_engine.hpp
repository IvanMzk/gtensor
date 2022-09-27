#ifndef EXPRESSION_TEMPLATE_ENGINE_HPP_
#define EXPRESSION_TEMPLATE_ENGINE_HPP_

#include <memory>
#include <array>
#include "engine.hpp"
#include "storage_walker.hpp"
#include "evaluating_walker.hpp"
#include "viewing_walker.hpp"
#include "trivial_walker.hpp"
#include "walker.hpp"
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

template<typename EngineT> auto begin_multiindex(const EngineT& engine){
    using iterator_type = multiindex_iterator<typename EngineT::value_type,typename EngineT::config_type,decltype(engine.create_broadcast_walker())>;
    return iterator_type{engine.create_broadcast_walker(), engine.host()->shape(), engine.host()->descriptor().as_descriptor_with_libdivide()->strides_libdivide()};
}
template<typename EngineT> auto end_multiindex(const EngineT& engine){
    using iterator_type = multiindex_iterator<typename EngineT::value_type,typename EngineT::config_type,decltype(engine.create_broadcast_walker())>;
    return iterator_type{engine.create_broadcast_walker(), engine.host()->shape(), engine.host()->descriptor().as_descriptor_with_libdivide()->strides_libdivide(), engine.host()->size()};
}
template<typename EngineT> auto begin_flatindex(const EngineT& engine){
    using iterator_type = flat_index_iterator<typename EngineT::value_type,typename EngineT::config_type,decltype(engine.create_trivial_walker())>;
    return iterator_type{engine.create_trivial_walker()};
}
template<typename EngineT> auto end_flatindex(const EngineT& engine){
    using iterator_type = flat_index_iterator<typename EngineT::value_type,typename EngineT::config_type,decltype(engine.create_trivial_walker())>;
    return iterator_type{engine.create_trivial_walker(), engine.host()->size()};
}
template<typename...Ts> auto begin_multiindex(const expression_template_storage_engine<Ts...>& engine){return engine.begin();}
template<typename...Ts> auto end_multiindex(const expression_template_storage_engine<Ts...>& engine){return engine.end();}
template<typename...Ts> auto begin_flatindex(const expression_template_storage_engine<Ts...>& engine){return engine.begin();}
template<typename...Ts> auto end_flatindex(const expression_template_storage_engine<Ts...>& engine){return engine.end();}

template<typename> constexpr bool is_converting_descriptor = true;
template<typename...Ts> constexpr bool is_converting_descriptor<converting_descriptor<Ts...>> = false;

template<typename> struct has_view_with_converting_descriptor{static constexpr bool value = false;};
template<typename V,typename C, typename P, typename...Ts> struct has_view_with_converting_descriptor<expression_template_viewing_engine<V,C,converting_descriptor<Ts...>,P>>{
    static constexpr bool value = true;
};
template<typename V,typename C, typename F, typename...Ops> struct has_view_with_converting_descriptor<expression_template_nodispatching_engine<V,C,F,Ops...>>{
    static constexpr bool value = (has_view_with_converting_descriptor<typename Ops::engine_type>::value||...);
};
template<typename V,typename C, typename F, typename...Ops> struct has_view_with_converting_descriptor<expression_template_root_dispatching_engine<V,C,F,Ops...>>{
    static constexpr bool value = (has_view_with_converting_descriptor<typename Ops::engine_type>::value||...);
};

}   //end of namespace detail

template<typename ValT, typename CfgT>
class expression_template_engine_base{
public:
    virtual bool is_trivial()const = 0;
};

template<typename ValT, typename CfgT>
class expression_template_storage_engine :
    public expression_template_engine_base<ValT,CfgT>,
    private storage_engine<ValT,CfgT>
{
public:
    using typename storage_engine::value_type;
    using typename storage_engine::config_type;
    using storage_engine::storage_engine;
    using storage_engine::host;
    using storage_engine::begin;
    using storage_engine::end;
    bool is_trivial()const override{return true;}
    auto create_broadcast_walker()const{
        return [this](const auto& it){
            return storage_walker<CfgT, std::decay_t<decltype(it)>>{host()->shape(),host()->strides(),host()->reset_strides(),it};
        }(begin());
    }
    auto create_trivial_walker()const{
        return storage_trivial_walker<ValT,CfgT>{data()};
    }
    auto create_indexer()const{return begin();}
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class expression_template_nodispatching_engine :
    public expression_template_engine_base<ValT,CfgT>,
    private evaluating_engine<ValT,CfgT,F,std::integral_constant<std::size_t,sizeof...(Ops)>>
{
    using evaluating_engine::operands_number;
public:
    using typename evaluating_engine::value_type;
    using typename evaluating_engine::config_type;
    using evaluating_engine::evaluating_engine;
    using evaluating_engine::host;
    bool is_trivial()const override{
        return std::apply(
            [this](const auto&...operands){
                return gtensor::detail::is_trivial(host()->size(),static_cast<Ops*>(operands.get())...);
            },
            operands()
        );
    }
    auto begin()const{return detail::begin_multiindex(*this);}
    auto end()const{return detail::end_multiindex(*this);}
    auto create_indexer()const{return create_broadcast_indexer();}
    auto create_broadcast_indexer()const{
        return [this](auto&& walker){
            return evaluating_indexer<ValT,CfgT,std::decay_t<decltype(walker)>>{host()->descriptor().as_descriptor_with_libdivide()->strides_libdivide(),std::forward<decltype(walker)>(walker)};
        }(create_broadcast_walker());
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
        return evaluating_walker<ValT,CfgT,F,std::decay_t<Args>...>{host()->shape(), std::forward<Args>(walkers)...};
    }
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class expression_template_root_dispatching_engine :
    public expression_template_nodispatching_engine<ValT,CfgT,F,Ops...>
{
public:
    using typename expression_template_nodispatching_engine::value_type;
    using typename expression_template_nodispatching_engine::config_type;
    using expression_template_nodispatching_engine::expression_template_nodispatching_engine;
    auto begin()const{return detail::begin_multiindex(*this);}
    auto end()const{return detail::end_multiindex(*this);}
    auto create_indexer()const{return create_broadcast_indexer();}
private:
    bool should_use_trivial()const{
        return typename detail::has_view_with_converting_descriptor<expression_template_nodispatching_engine>::value && is_trivial();
    }
};

template<typename ValT, typename CfgT, typename DescT, typename ParentT>
class expression_template_viewing_engine :
    public expression_template_engine_base<ValT,CfgT>,
    private viewing_engine<ValT,CfgT,ParentT>
{
public:
    using typename viewing_engine::value_type;
    using typename viewing_engine::config_type;
    using viewing_engine::viewing_engine;
    using viewing_engine::host;
    bool is_trivial()const override{return true;}
    //use multiindex_iterator for view with converting descriptor - slice, transpose
    template<typename D = DescT, std::enable_if_t<detail::is_converting_descriptor<D>,int> =0 > auto begin()const{return detail::begin_multiindex(*this);}
    template<typename D = DescT, std::enable_if_t<detail::is_converting_descriptor<D>,int> =0 > auto end()const{return detail::end_multiindex(*this);}
    //use flat_index_iterator for view with not converting descriptor - reshape, subdim
    template<typename D = DescT, std::enable_if_t<!detail::is_converting_descriptor<D>,int> =0 > auto begin()const{return detail::begin_flatindex(*this);}
    template<typename D = DescT, std::enable_if_t<!detail::is_converting_descriptor<D>,int> =0 > auto end()const{return detail::end_flatindex(*this);}
    auto create_indexer()const{
        return [](const auto& descriptor, auto indexer){
            return viewing_indexer<std::decay_t<decltype(descriptor)>, std::decay_t<decltype(indexer)>>{descriptor, indexer};
        }(parent()->descriptor(), parent()->engine().create_indexer());
    }
    auto create_broadcast_walker()const{
        return [this](const auto& it){
            return viewing_walker<CfgT, std::decay_t<decltype(it)>>{host()->shape(),host()->descriptor().cstrides(),host()->descriptor().reset_cstrides(),host()->descriptor().offset(),it};
        }(create_indexer());
    }
    auto create_trivial_walker()const{
        return [this](const auto& it){
            return viewing_trivial_walker<DescT, std::decay_t<decltype(it)>>{static_cast<const DescT&>(host()->descriptor()),it};
        }(create_indexer());
    }
};

}   //end of namespace gtensor

#endif