#ifndef EXPRESSION_TEMPLATE_ENGINE_HPP_
#define EXPRESSION_TEMPLATE_ENGINE_HPP_

#include <memory>
#include "forward_decl.hpp"
#include "storage_walker.hpp"
#include "evaluating_walker.hpp"
#include "viewing_walker.hpp"
#include "evaluating_trivial_walker.hpp"
#include "dispatcher.hpp"

namespace gtensor{

namespace detail{

 template<typename ValT, typename CfgT>
 bool is_storage(const tensor_base<ValT,CfgT>& t){
    return t.tensor_kind() == detail::tensor_kinds::storage_tensor || 
        t.tensor_kind() == detail::tensor_kinds::expression && t.is_storage() ||
        t.tensor_kind() == detail::tensor_kinds::view && t.is_cached();
}

template<typename CfgT, std::enable_if_t<detail::is_mode_div_native<CfgT> ,int> =0 >
inline const auto& strides_div(const descriptor_with_libdivide<CfgT>& desc){
    return desc.strides();
}
template<typename CfgT, std::enable_if_t<detail::is_mode_div_libdivide<CfgT> ,int> =0 >
inline const auto& strides_div(const descriptor_with_libdivide<CfgT>& desc){
    return desc.strides_libdivide();
}


template<typename...Ts>
struct type_list{
    using type = type_list<Ts...>;
    static constexpr std::size_t size = sizeof...(Ts);     
};

template<typename, typename> struct list_concat;
template<typename...Us, typename...Vs> 
struct list_concat<type_list<Us...>,type_list<Vs...>>{
    using type = type_list<Us...,Vs...>;
};

template<template <typename...> typename, typename, typename> struct cross_product;
template<template <typename...> typename PairT, typename U, typename...Us, typename...Vs> 
struct cross_product<PairT, type_list<U, Us...>, type_list<Vs...>>{
    using cross_u_vs = type_list<PairT<U,Vs>...>;
    using cross_us_vs = typename cross_product<PairT, type_list<Us...>, type_list<Vs...>>::type;
    using type = typename list_concat<cross_u_vs, cross_us_vs>::type;
};
template<template <typename...> typename PairT, typename...Vs>
struct cross_product<PairT, type_list<>, type_list<Vs...>>{    
    using type = type_list<>;
};


}   //end of namespace detail


//expression_template_elementwise_engine class is responsible for handling arithmetic operations +,-,*,/,<,>, ...
//i.e. such operations that can be done in elemenwise fashion, evaluation is broadcasted if possible
//depending on config it also may cache operands to make broadcast evaluation more efficient
//evaluation can be done by pure elementwise calculations (trivial broadcasting) if all nodes in evaluation tree support such an evaluation
template<typename ValT, typename CfgT, typename F, typename...Ops>
class expression_template_elementwise_engine
{    
    using value_type = ValT;
    using shape_type = typename CfgT::shape_type;    
    
    const tensor_base<ValT,CfgT>* root;
    F f;
    std::tuple<std::shared_ptr<tensor_base<typename Ops::value_type, CfgT> >...> operands;

    
    auto walker_maker()const{
        return [this](const auto&...args){
            using evaluating_walker_type = evaluating_walker<ValT,CfgT,F,decltype(std::declval<decltype(args)>().create_walker())...>;
            return walker<ValT,CfgT>{std::make_unique<evaluating_walker_polymorphic<ValT,CfgT,evaluating_walker_type>>(evaluating_walker_type{root->shape(),args.create_walker()...})};
            };
    }
    
    auto indexer_maker()const{
        return [this](const auto&...args){
                using evaluating_walker_type = evaluating_walker<ValT,CfgT,F,decltype(std::declval<decltype(args)>().create_walker())...>;
                using evaluating_indexer_type = evaluating_indexer<ValT,CfgT,evaluating_walker_type>;
                return indexer<ValT,CfgT>{std::make_unique<evaluating_indexer_type>(detail::strides_div(*root->descriptor().as_descriptor_with_libdivide()) , evaluating_walker_type{root->shape(),args.create_walker()...})};
            };
    }
    
public:

    template<typename...Args>
    expression_template_elementwise_evaluation(const tensor_base<ValT,CfgT>* root_, const F& f_, const Args&...args):
        root{root_},
        f{f_},
        operands{args...}
    {}

    walker<ValT,CfgT> create_walker()const{
        return std::apply([this](const auto&...args){return detail::dispatcher<ValT,CfgT>::call(walker_maker(), *args...);}, operands);
    }        
    indexer<ValT,CfgT> create_indexer()const{
        return std::apply([this](const auto&...args){return detail::dispatcher<ValT,CfgT>::call(indexer_maker(), *args...);}, operands);
    }
};

}   //end of namespace gtensor

#endif