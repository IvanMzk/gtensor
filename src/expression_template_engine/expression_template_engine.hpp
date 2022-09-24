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

template<typename IdxT,template<typename...> typename TupleT, typename...Ops>
inline bool is_trivial(const IdxT& root_size, const TupleT<Ops...>& root_operands){
    return is_trivial_helper(root_size,root_operands,std::make_index_sequence<sizeof...(Ops)>{});
}
template<typename IdxT, template<typename...> typename TupleT, typename...Ops, std::size_t...I>
inline bool is_trivial_helper(const IdxT& root_size, const TupleT<Ops...>& root_operands, std::index_sequence<I...>){
    return ((root_size==std::get<I>(root_operands)->size())&&...) && (is_trivial_operand(std::get<I>(root_operands))&&...);
}
template<typename T>
inline bool is_trivial_operand(const T& operand){
    return operand->engine().is_trivial();
}

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
    bool is_trivial()const override{return true;}
    auto create_walker()const{return create_broadcast_walker();}
    auto create_broadcast_walker()const{
        return [this](const auto& it){
            return storage_walker<CfgT, std::decay_t<decltype(it)>>{host()->shape(),host()->strides(),host()->reset_strides(),it};
        }(begin());
    }
    auto create_trivial_walker()const{return storage_trivial_walker<ValT,CfgT>{data()};}
    auto create_indexer()const{return begin();}
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class expression_template_nodispatching_engine :
    public expression_template_engine_base<ValT,CfgT>,
    private evaluating_engine<ValT,CfgT,F,std::integral_constant<std::size_t,sizeof...(Ops)>>
{
public:
    using typename evaluating_engine::value_type;
    using typename evaluating_engine::config_type;
    using evaluating_engine::evaluating_engine;
    bool is_trivial()const override{
        return std::apply(
            [this](const auto&...operands){
                return gtensor::detail::is_trivial(host()->size(),static_cast<Ops*>(operands.get())...);
            },
            operands()
        );
    }
    auto create_walker()const{
        return create_broadcast_walker();
    }
    auto create_indexer()const{
        return [this](auto&& walker){
            return evaluating_indexer<ValT,CfgT,std::decay_t<decltype(walker)>>{host()->descriptor().as_descriptor_with_libdivide()->strides_libdivide(),std::forward<decltype(walker)>(walker)};
        }(create_walker());
    }
private:
    auto create_broadcast_walker()const{
        return std::apply(
            [this](const auto&...operands){
                return create_walker_helper(static_cast<Ops*>(operands.get())->engine().create_walker()...);
            },
            operands()
        );
    }
    template<typename...Wks>
    auto create_walker_helper(Wks&&...walkers)const{
        return evaluating_walker<ValT,CfgT,F,Wks...>{host()->shape(),std::forward<Wks>(walkers)...};
    }
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class expression_template_root_dispatching_engine :
    public expression_template_engine_base<ValT,CfgT>,
    private evaluating_engine<ValT,CfgT,F,std::integral_constant<std::size_t,sizeof...(Ops)>>
{
public:
    using typename evaluating_engine::value_type;
    using typename evaluating_engine::config_type;
    using evaluating_engine::evaluating_engine;
    bool is_trivial()const override{
        return std::apply(
            [this](const auto&...operands){
                return gtensor::detail::is_trivial(host()->size(),static_cast<Ops*>(operands.get())...);
            },
            operands()
        );
    }
    //make result polywalker
    template<typename U>
    auto create_walker_helper(U&& walker_)const{
        return walker<ValT,CfgT>{
            std::make_unique<walker_polymorphic<ValT,CfgT,std::decay_t<U>>>(std::forward<U>(walker_))
        };
    }
    auto create_walker()const{
        if (is_trivial()){
            return create_walker_helper(create_trivial_root_walker());
        }else{
            return create_walker_helper(create_broadcast_walker());
        }
    }
    //make broadcast walker
    template<typename...Wks>
    auto create_broadcast_walker_helper(Wks&&...walkers)const{
        return evaluating_walker<ValT,CfgT,F,Wks...>{host()->shape(),std::forward<Wks>(walkers)...};
    }
    template<std::size_t...I>
    auto create_broadcast_walker_helper(std::index_sequence<I...>)const{
        return create_broadcast_walker_helper(static_cast<Ops*>(std::get<I>(operands()).get())->engine().create_broadcast_walker()...);
    }
    auto create_broadcast_walker()const{
        return create_broadcast_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
    }
    //make trivial root walker
    template<typename...Wks>
    auto create_trivial_root_walker_helper(Wks&&...walkers)const{
        return evaluating_trivial_root_walker<ValT,CfgT,F,Wks...>{host()->shape(),host()->strides(),host()->reset_strides(),std::forward<Wks>(walkers)...};
    }
    template<std::size_t...I>
    auto create_trivial_root_walker_helper(std::index_sequence<I...>)const{
        return create_trivial_root_walker_helper(static_cast<Ops*>(std::get<I>(operands()).get())->engine().create_trivial_walker()...);
    }
    auto create_trivial_root_walker()const{
        return create_trivial_root_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
    }
    //make trivial walker
    template<typename...Wks>
    auto create_trivial_walker_helper(Wks&&...walkers)const{
        return evaluating_trivial_walker<ValT,CfgT,F,Wks...>{std::forward<Wks>(walkers)...};
    }
    template<std::size_t...I>
    auto create_trivial_walker_helper(std::index_sequence<I...>)const{
        return create_trivial_walker_helper(static_cast<Ops*>(std::get<I>(operands()).get())->engine().create_trivial_walker()...);
    }
    auto create_trivial_walker()const{
        return create_trivial_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
    }
};

template<typename ValT, typename CfgT, typename ParentT>
class expression_template_viewing_engine :
    public expression_template_engine_base<ValT,CfgT>,
    private viewing_engine<ValT,CfgT,ParentT>
{
public:
    using typename viewing_engine::value_type;
    using typename viewing_engine::config_type;
    using viewing_engine::viewing_engine;
    bool is_trivial()const override{return true;}
    auto create_indexer()const{
        return [](const auto& descriptor, auto indexer){
            return viewing_indexer<std::decay_t<decltype(descriptor)>, std::decay_t<decltype(indexer)>>{descriptor, indexer};
        }(parent()->descriptor(), parent()->engine().create_indexer());
    }
};

}   //end of namespace gtensor

#endif