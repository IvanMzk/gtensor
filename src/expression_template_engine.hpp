#ifndef EXPRESSION_TEMPLATE_ENGINE_HPP_
#define EXPRESSION_TEMPLATE_ENGINE_HPP_

#include <memory>
#include "forward_decl.hpp"
#include "engine_base.hpp"
#include "storage_walker.hpp"
#include "evaluating_walker.hpp"
#include "viewing_walker.hpp"
#include "trivial_walker.hpp"
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

/*
* is expression is trivial broadcast i.e. shapes of all nodes in expression tree is same
* flat index access without walkers is used to evaluate broadcast expression
* stensor and view are trivial
*/
template<typename IdxT, typename...Ops>
inline bool is_trivial(const IdxT& root_size, const std::tuple<Ops...>& root_operands){
    return is_trivial_helper(root_size,root_operands,std::make_index_sequence<sizeof...(Ops)>{});
}
template<typename IdxT, typename...Ops, std::size_t...I>
inline bool is_trivial_helper(const IdxT& root_size, const std::tuple<Ops...>& root_operands, std::index_sequence<I...>){
    return ((root_size==std::get<I>(root_operands)->size())&&...) && (is_trivial_operand(std::get<I>(root_operands))&&...);
}
template<typename T>
inline bool is_trivial_operand(const T& operand){    
    return operand->engine().is_trivial(); 
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

template<typename DerivingT, typename CurrentT>
struct engine_type_selector{
    using type = std::conditional_t<std::is_void_v<DerivingT>,CurrentT,DerivingT>;
};

}   //end of namespace detail




template<typename DerivingT, typename ValT, typename CfgT>
class expression_template_storage_engine : 
    public expression_template_engine_base<ValT, CfgT>,
    public engine_root_accessor<storage_tensor, ValT, CfgT, typename detail::engine_type_selector<DerivingT, expression_template_storage_engine<DerivingT,ValT,CfgT>>::type>
{    
public:
    //using walker_types = detail::type_list<storage_walker<ValT,CfgT>>;
    expression_template_storage_engine() = default;    
    template<typename R>
    expression_template_storage_engine(R* root_):
        engine_root_accessor{root_}
    {}

    bool is_trivial()const override{return true;}
    auto create_walker()const{return storage_walker<ValT,CfgT>{root()->shape(),root()->strides(),root()->data()};}
};

template<typename ValT, typename CfgT, typename DescT>
class expression_template_view_engine : public expression_template_engine_base<ValT, CfgT>
{
    const viewing_tensor<ValT,CfgT,DescT>* root;
public:
    //using walker_types = detail::type_list<storage_walker<ValT,CfgT>>;
    expression_template_view_engine(const viewing_tensor<ValT,CfgT,DescT>* root_):
        root{root_}
    {}
    bool is_trivial()const override{return true;}
    //void set_root(tensor_base<ValT,CfgT>* root_)override{root = root_;}
};

//expression_template_elementwise_engine class is responsible for handling arithmetic operations +,-,*,/,<,>, ...
//i.e. such operations that can be done in elemenwise fashion, evaluation is broadcasted if possible
//depending on config it also may cache operands to make broadcast evaluation more efficient
//evaluation can be done by pure elementwise calculations (trivial broadcasting) if all nodes in evaluation tree support such an evaluation
template<typename DerivingT, typename ValT, typename CfgT, typename F, typename...Ops>
class expression_template_elementwise_engine : 
    public expression_template_engine_base<ValT, CfgT>,
    public engine_root_accessor<evaluating_tensor, ValT, CfgT, F, typename detail::engine_type_selector<DerivingT, expression_template_elementwise_engine<DerivingT,ValT,CfgT,F,Ops...>>::type, Ops...>
{    
    using shape_type = typename CfgT::shape_type;    
    
    // static constexpr std::size_t max_walker_types_size = 100;
    // static constexpr std::size_t walker_types_size = (Ops::engine_type::walker_types::size*...);
    // template<typename...Us> using evaluating_walker_alias = evaluating_walker<ValT, CfgT, F, Us...>;
    
    // template<bool> struct walker_types_traits{                
    //     using type = typename detail::list_concat< 
    //         detail::type_list<storage_walker<ValT,CfgT>>, 
    //         typename detail::cross_product<evaluating_walker_alias, typename Ops::engine_type::walker_types...>::type
    //         >::type;
    // };
    // template<> struct walker_types_traits<false>{        
    //     using type = detail::type_list<walker<ValT,CfgT>>;
    // };

    auto walker_maker()const{
        return [this](const auto&...args){
            using evaluating_walker_type = evaluating_walker<ValT,CfgT,F,decltype(std::declval<decltype(args)>().create_walker())...>;
            return walker<ValT,CfgT>{std::make_unique<evaluating_walker_polymorphic<ValT,CfgT,evaluating_walker_type>>(evaluating_walker_type{root()->shape(),args.create_walker()...})};
            };
    }
    
    auto indexer_maker()const{
        return [this](const auto&...args){
                using evaluating_walker_type = evaluating_walker<ValT,CfgT,F,decltype(std::declval<decltype(args)>().create_walker())...>;
                using evaluating_indexer_type = evaluating_indexer<ValT,CfgT,evaluating_walker_type>;
                return indexer<ValT,CfgT>{std::make_unique<evaluating_indexer_type>(detail::strides_div(*root()->descriptor().as_descriptor_with_libdivide()) , evaluating_walker_type{root()->shape(),args.create_walker()...})};
            };
    }
    
public:
    using value_type = ValT;
    //using walker_types = typename walker_types_traits<(walker_types_size<max_walker_types_size)>::type;

    bool is_trivial()const override{return detail::is_trivial(root()->size(),root()->operands());}

    walker<ValT,CfgT> create_walker()const{
        return std::apply([this](const auto&...args){return detail::dispatcher<ValT,CfgT>::call(walker_maker(), *args...);}, root()->operands());
    }        
    indexer<ValT,CfgT> create_indexer()const{
        return std::apply([this](const auto&...args){return detail::dispatcher<ValT,CfgT>::call(indexer_maker(), *args...);}, root()->operands());
    }
};

}   //end of namespace gtensor

#endif