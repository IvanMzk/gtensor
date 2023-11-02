/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef MODULE_SELECTOR_HPP_
#define MODULE_SELECTOR_HPP_

#include "config.hpp"

namespace gtensor{
//default module implementation forward declarations
template<typename Config, typename T, typename Layout> class tensor_factory;
class view_factory;
class expression_template_operator;
class reducer;
class manipulation;
struct tensor_operators;
struct tensor_math;
struct statistic;
struct sort_search;
struct builder;
struct random;
struct indexing;

//module selectors
//to dispatch library routine to custom module, specialization of appropriate selector should be added
//all library interface free functions pass valid config_type of its tensor argument for Config template type parameter
//in most cases it is not used, but may be used to do dispatch based on particular Config type

//storage implementation factory selector
//expected Config, element type and layout tag as template type arguments
template<typename Config, typename T, typename Layout, typename...Ts>
struct tensor_factory_selector
{
    using type = tensor_factory<Config,T,Layout>;
};
template<typename...Ts> using tensor_factory_selector_t = typename tensor_factory_selector<Ts...>::type;

//view implementation factory selector
template<typename Config, typename...Ts>
struct view_factory_selector
{
    using type = view_factory;
};
template<typename...Ts> using view_factory_selector_t = typename view_factory_selector<Ts...>::type;

//generalized operator selector
template<typename Config, typename...Ts>
struct generalized_operator_selector
{
    using type = expression_template_operator;
};
template<typename...Ts> using generalized_operator_selector_t = typename generalized_operator_selector<Ts...>::type;

//tensor operators selector
template<typename Config, typename...Ts>
struct tensor_operators_selector
{
    using type = tensor_operators;
};
template<typename...Ts> using tensor_operators_selector_t = typename tensor_operators_selector<Ts...>::type;

//tensor math selector
template<typename Config, typename...Ts>
struct tensor_math_selector
{
    using type = tensor_math;
};
template<typename...Ts> using tensor_math_selector_t = typename tensor_math_selector<Ts...>::type;

//tensor statistic selector
template<typename Config, typename...Ts>
struct statistic_selector
{
    using type = statistic;
};
template<typename...Ts> using statistic_selector_t = typename statistic_selector<Ts...>::type;

//tensor sort_search selector
template<typename Config, typename...Ts>
struct sort_search_selector
{
    using type = sort_search;
};
template<typename...Ts> using sort_search_selector_t = typename sort_search_selector<Ts...>::type;

//reducer selector
template<typename Config, typename...Ts>
struct reducer_selector
{
    using type = reducer;
};
template<typename...Ts> using reducer_selector_t = typename reducer_selector<Ts...>::type;

//manipulation selector
template<typename Config, typename...Ts>
struct manipulation_selector
{
    using type = manipulation;
};
template<typename...Ts> using manipulation_selector_t = typename manipulation_selector<Ts...>::type;

//builder selector
template<typename Config, typename...Ts>
struct builder_selector
{
    using type = builder;
};
template<typename...Ts> using builder_selector_t = typename builder_selector<Ts...>::type;

//random selector
template<typename Config, typename...Ts>
struct random_selector
{
    using type = random;
};
template<typename...Ts> using random_selector_t = typename random_selector<Ts...>::type;

//indexing selector
template<typename Config, typename...Ts>
struct indexing_selector
{
    using type = indexing;
};
template<typename...Ts> using indexing_selector_t = typename indexing_selector<Ts...>::type;

}   //end of namespace gtensor
#endif