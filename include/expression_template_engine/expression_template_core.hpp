/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef EXPRESSION_TEMPLATE_CORE_HPP_
#define EXPRESSION_TEMPLATE_CORE_HPP_

#include "descriptor.hpp"
#include "expression_template_evaluator.hpp"

namespace gtensor{

namespace detail{

template<typename F, typename It>
auto make_iterator_deref_decorator(F f, It it){
    using it_category = typename std::iterator_traits<It>::iterator_category;
    if constexpr (std::is_convertible_v<it_category,std::random_access_iterator_tag>){
        return iterator_deref_decorator<F,It,std::random_access_iterator_tag>{f,it};
    }else if constexpr (std::is_convertible_v<it_category,std::bidirectional_iterator_tag>){
        return iterator_deref_decorator<F,It,std::bidirectional_iterator_tag>{f,it};
    }else{
        static_assert(detail::always_false<It>,"at least bidirectional iterator expected");
    }
}

}   //end of namespace detail

//expression_template_core represents expression on basic_tensor operands
//F is expression operation, must provide operator()() on arguments that are operands elements
//Operands are specializations of basic_tensor
template<typename Config, typename F, typename...Operands>
class expression_template_core
{
    using common_order = detail::common_order_t<Config,typename Operands::order...>;
    using descriptor_type = basic_descriptor<Config,common_order>;
    template<typename...Ts> using tuple_type = std::tuple<Ts...>;
    using sequence_type = std::make_index_sequence<sizeof...(Operands)>;
public:
    using order = common_order;
    using value_type = std::decay_t<decltype(std::declval<F>()(std::declval<typename Operands::value_type&>()...))>;
    using config_type = config::extend_config_t<Config,value_type>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;

    template<typename F_> struct forward_args : std::bool_constant<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<F_>>,expression_template_core>>{};

    template<typename F_, typename...Operands_, std::enable_if_t<forward_args<F_>::value,int> =0>
    explicit expression_template_core(F_&& f__, Operands_&&...operands__):
        descriptor_{detail::make_broadcast_shape<shape_type>(operands__.shape()...)},
        f_{std::forward<F_>(f__)},
        operands_{std::forward<Operands_>(operands__)...}
    {}
    const descriptor_type& descriptor()const{
        return descriptor_;
    }
    auto create_walker(dim_type max_dim){
        return create_walker_helper(*this,max_dim,sequence_type{});
    }
    auto create_walker(dim_type max_dim)const{
        return create_walker_helper(*this,max_dim,sequence_type{});
    }
    auto create_walker(){
        return create_walker(descriptor_.dim());
    }
    auto create_walker()const{
        return create_walker(descriptor_.dim());
    }
    auto create_trivial_indexer(){
        return create_trivial_indexer_helper(*this,sequence_type{});
    }
    auto create_trivial_indexer()const{
        return create_trivial_indexer_helper(*this,sequence_type{});
    }
    bool is_trivial()const{
        return is_trivial_helper(sequence_type{});
    }
private:
    template<typename U, std::size_t...I>
    static auto create_walker_helper(U& instance, dim_type max_dim, std::index_sequence<I...>){
        return expression_template_walker<config_type,std::remove_cv_t<F>,decltype(std::get<I>(instance.operands_).create_walker(max_dim))...>{
            instance.f_,
            std::get<I>(instance.operands_).create_walker(max_dim)...
        };
    }
    template<typename U, std::size_t...I>
    static auto create_trivial_indexer_helper(U& instance, std::index_sequence<I...>){
        return expression_template_trivial_indexer<
            config_type,
            std::remove_cv_t<F>,
            decltype(std::get<I>(instance.operands_).traverse_order_adapter(typename Operands::order{}).create_trivial_indexer())...
        >{
            instance.f_,
            std::get<I>(instance.operands_).traverse_order_adapter(typename Operands::order{}).create_trivial_indexer()...
        };
    }

    template<std::size_t...I>
    bool is_trivial_helper(std::index_sequence<I...>)const{
        if (((descriptor_.size()==std::get<I>(operands_).size())&&...) && (std::get<I>(operands_).is_trivial()&&...)){
            if constexpr ((std::is_same_v<order,typename Operands::order>&&...)){
                return true;
            }else{
                return ((std::get<I>(operands_).dim()==1)&&...);    //order doesn't matter if 1d
            }
        }else{
            return false;
        }
    }

    descriptor_type descriptor_;
    F f_;
    tuple_type<Operands...> operands_;
};

//single operand specialization
template<typename Config, typename F, typename Operand>
class expression_template_core<Config,F,Operand>
{
    using descriptor_type = basic_descriptor<Config,typename Operand::order>;
public:
    using order = typename Operand::order;
    using value_type = std::decay_t<decltype(std::declval<F>()(std::declval<typename Operand::value_type&>()))>;
    using config_type = config::extend_config_t<Config,value_type>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;

    template<typename F_, typename Operand_>
    explicit expression_template_core(F_&& f__, Operand_&& operand__):
        descriptor_{operand__.shape()},
        f_{std::forward<F_>(f__)},
        operand_{std::forward<Operand_>(operand__)}
    {}
    const descriptor_type& descriptor()const{
        return descriptor_;
    }

    //data interface
    auto begin(){
        return begin_helper(*this);
    }
    auto end(){
        return end_helper(*this);
    }
    auto rbegin(){
        return rbegin_helper(*this);
    }
    auto rend(){
        return rend_helper(*this);
    }
    auto create_walker(dim_type max_dim){
        return create_walker_helper(*this,max_dim);
    }
    auto create_walker(){
        return create_walker(descriptor_.dim());
    }
    //trivial data interface
    auto begin_trivial(){
        return begin_trivial_helper(*this);
    }
    auto end_trivial(){
        return end_trivial_helper(*this);
    }
    auto rbegin_trivial(){
        return rbegin_trivial_helper(*this);
    }
    auto rend_trivial(){
        return rend_trivial_helper(*this);
    }
    auto create_trivial_indexer(){
        return create_trivial_indexer_helper(*this);
    }

    //const data interface
    auto begin()const{
        return begin_helper(*this);
    }
    auto end()const{
        return end_helper(*this);
    }
    auto rbegin()const{
        return rbegin_helper(*this);
    }
    auto rend()const{
        return rend_helper(*this);
    }
    auto create_walker(dim_type max_dim)const{
        return create_walker_helper(*this,max_dim);
    }
    auto create_walker()const{
        return create_walker(descriptor_.dim());
    }
    //trivial const data interface
    auto begin_trivial()const{
        return begin_trivial_helper(*this);
    }
    auto end_trivial()const{
        return end_trivial_helper(*this);
    }
    auto rbegin_trivial()const{
        return rbegin_trivial_helper(*this);
    }
    auto rend_trivial()const{
        return rend_trivial_helper(*this);
    }
    auto create_trivial_indexer()const{
        return create_trivial_indexer_helper(*this);
    }

    bool is_trivial()const{
        return operand_.is_trivial();
    }
private:
    template<typename U>
    static auto begin_helper(U& instance){
        return detail::make_iterator_deref_decorator(instance.f_,instance.operand_.traverse_order_adapter(order{}).begin());
    }
    template<typename U>
    static auto end_helper(U& instance){
        return detail::make_iterator_deref_decorator(instance.f_,instance.operand_.traverse_order_adapter(order{}).end());
    }
    template<typename U>
    static auto rbegin_helper(U& instance){
        return detail::make_iterator_deref_decorator(instance.f_,instance.operand_.traverse_order_adapter(order{}).rbegin());
    }
    template<typename U>
    static auto rend_helper(U& instance){
        return detail::make_iterator_deref_decorator(instance.f_,instance.operand_.traverse_order_adapter(order{}).rend());
    }
    template<typename U>
    static auto begin_trivial_helper(U& instance){
        return detail::make_iterator_deref_decorator(instance.f_,instance.operand_.traverse_order_adapter(order{}).begin_trivial());
    }
    template<typename U>
    static auto end_trivial_helper(U& instance){
        return detail::make_iterator_deref_decorator(instance.f_,instance.operand_.traverse_order_adapter(order{}).end_trivial());
    }
    template<typename U>
    static auto rbegin_trivial_helper(U& instance){
        return detail::make_iterator_deref_decorator(instance.f_,instance.operand_.traverse_order_adapter(order{}).rbegin_trivial());
    }
    template<typename U>
    static auto rend_trivial_helper(U& instance){
        return detail::make_iterator_deref_decorator(instance.f_,instance.operand_.traverse_order_adapter(order{}).rend_trivial());
    }
    template<typename U>
    static auto create_walker_helper(U& instance, dim_type max_dim){
        return expression_template_walker<config_type,std::remove_cv_t<F>,decltype(instance.operand_.create_walker(max_dim))>{
            instance.f_,
            instance.operand_.create_walker(max_dim)
        };
    }
    template<typename U>
    static auto create_trivial_indexer_helper(U& instance){
        return expression_template_trivial_indexer<
            config_type,
            std::remove_cv_t<F>,
            decltype(instance.operand_.traverse_order_adapter(order{}).create_trivial_indexer())
        >{
            instance.f_,
            instance.operand_.traverse_order_adapter(order{}).create_trivial_indexer()
        };
    }

    descriptor_type descriptor_;
    F f_;
    Operand operand_;
};

}   //end of namespace gtensor
#endif