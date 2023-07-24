#ifndef MODULE_SELECTOR_HPP_
#define MODULE_SELECTOR_HPP_

#include "config.hpp"

namespace gtensor{
//default module implementation forward declarations
template<typename Config, typename T, typename Layout> class tensor_factory;
class view_factory;
template<typename F> class expression_template_operator;
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
//should specialize inner struct selector_ to change module default implementation

//storage implementation factory selector
template<typename Config, typename...Ts>
class tensor_factory_selector
{
    using config_type = Config;
    template<typename T, typename Layout, typename...> struct selector_
    {
        using type = tensor_factory<config_type,T,Layout>;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using tensor_factory_selector_t = typename tensor_factory_selector<Ts...>::type;

//view implementation factory selector
template<typename Config, typename...Ts>
class view_factory_selector
{
    using config_type = Config;
    template<typename...> struct selector_
    {
        using type = view_factory;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using view_factory_selector_t = typename view_factory_selector<Ts...>::type;

//generalized operator selector
template<typename Config, typename...Ts>
class generalized_operator_selector
{
    using config_type = Config;
    template<typename F, typename...> struct selector_
    {
        using type = expression_template_operator<F>;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using generalized_operator_selector_t = typename generalized_operator_selector<Ts...>::type;

//tensor operators selector
template<typename Config, typename...Ts>
class tensor_operators_selector
{
    using config_type = Config;
    template<typename...> struct selector_
    {
        using type = tensor_operators;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using tensor_operators_selector_t = typename tensor_operators_selector<Ts...>::type;

//tensor math selector
template<typename Config, typename...Ts>
class tensor_math_selector
{
    using config_type = Config;
    template<typename...> struct selector_
    {
        using type = tensor_math;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using tensor_math_selector_t = typename tensor_math_selector<Ts...>::type;

//tensor statistic selector
template<typename Config, typename...Ts>
class statistic_selector
{
    using config_type = Config;
    template<typename...> struct selector_
    {
        using type = statistic;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using statistic_selector_t = typename statistic_selector<Ts...>::type;

//tensor sort_search selector
template<typename Config, typename...Ts>
class sort_search_selector
{
    using config_type = Config;
    template<typename...> struct selector_
    {
        using type = sort_search;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using sort_search_selector_t = typename sort_search_selector<Ts...>::type;

//reducer selector
template<typename Config, typename...Ts>
class reducer_selector
{
    using config_type = Config;
    template<typename...> struct selector_
    {
        using type = reducer;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using reducer_selector_t = typename reducer_selector<Ts...>::type;

//manipulation selector
template<typename Config, typename...Ts>
class manipulation_selector
{
    using config_type = Config;
    template<typename...> struct selector_
    {
        using type = manipulation;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using manipulation_selector_t = typename manipulation_selector<Ts...>::type;

//builder selector
template<typename Config, typename...Ts>
class builder_selector
{
    using config_type = Config;
    template<typename...> struct selector_
    {
        using type = builder;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using builder_selector_t = typename builder_selector<Ts...>::type;

//random selector
template<typename Config, typename...Ts>
class random_selector
{
    using config_type = Config;
    template<typename...> struct selector_
    {
        using type = random;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using random_selector_t = typename random_selector<Ts...>::type;

//indexing selector
template<typename Config, typename...Ts>
class indexing_selector
{
    using config_type = Config;
    template<typename...> struct selector_
    {
        using type = indexing;
    };
public:
    using type = typename selector_<Ts...>::type;
};
template<typename...Ts> using indexing_selector_t = typename indexing_selector<Ts...>::type;

}   //end of namespace gtensor
#endif