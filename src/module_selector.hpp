#ifndef MODULE_SELECTOR_HPP_
#define MODULE_SELECTOR_HPP_

#include "config.hpp"

namespace gtensor{
//forward declarations
template<typename Config, typename T, typename Layout> class tensor_factory;
class view_factory;
template<typename F> class expression_template_operator;
class reducer;
class combiner;
struct tensor_operators;
struct tensor_math;


//storage implementation factory selector
template<typename Config, typename...Ts>
class tensor_factory_selector
{
    using config_type = Config;
    template<typename...> struct selector_;
    template<typename T, typename Layout> struct selector_<config::engine_expression_template,T,Layout>
    {
        using type = tensor_factory<config_type,T,Layout>;
    };
public:
    using type = typename selector_<typename config_type::engine, Ts...>::type;
};
template<typename...Ts> using tensor_factory_selector_t = typename tensor_factory_selector<Ts...>::type;

//view implementation factory selector
template<typename Config, typename...Ts>
class view_factory_selector
{
    using config_type = Config;
    template<typename...> struct selector_;
    template<typename Dummy> struct selector_<config::engine_expression_template,Dummy>
    {
        using type = view_factory;
    };
public:
    using type = typename selector_<typename config_type::engine, void, Ts...>::type;
};
template<typename...Ts> using view_factory_selector_t = typename view_factory_selector<Ts...>::type;

//generalized operator selector
template<typename Config, typename...Ts>
class generalized_operator_selector
{
    using config_type = Config;
    template<typename...> struct selector_;
    template<typename F> struct selector_<config::engine_expression_template,F>
    {
        using type = expression_template_operator<F>;
    };
public:
    using type = typename selector_<typename config_type::engine, Ts...>::type;
};
template<typename...Ts> using generalized_operator_selector_t = typename generalized_operator_selector<Ts...>::type;

//tensor operators selector
template<typename Config, typename...Ts>
class tensor_operators_selector
{
    using config_type = Config;
    template<typename...> struct selector_;
    template<typename Dummy> struct selector_<config::engine_expression_template,Dummy>
    {
        using type = tensor_operators;
    };
public:
    using type = typename selector_<typename config_type::engine, void, Ts...>::type;
};
template<typename...Ts> using tensor_operators_selector_t = typename tensor_operators_selector<Ts...>::type;

//tensor math selector
template<typename Config, typename...Ts>
class tensor_math_selector
{
    using config_type = Config;
    template<typename...> struct selector_;
    template<typename Dummy> struct selector_<config::engine_expression_template,Dummy>
    {
        using type = tensor_math;
    };
public:
    using type = typename selector_<typename config_type::engine, void, Ts...>::type;
};
template<typename...Ts> using tensor_math_selector_t = typename tensor_math_selector<Ts...>::type;

//reducer selector
template<typename Config, typename...Ts>
class reducer_selector
{
    using config_type = Config;
    template<typename...> struct selector_;
    template<typename Dummy> struct selector_<config::engine_expression_template,Dummy>
    {
        using type = reducer;
    };
public:
    using type = typename selector_<typename config_type::engine, void, Ts...>::type;
};
template<typename...Ts> using reducer_selector_t = typename reducer_selector<Ts...>::type;

//combiner selector
template<typename Config, typename...Ts>
class combiner_selector
{
    using config_type = Config;
    template<typename...> struct selector_;
    template<typename Dummy> struct selector_<config::engine_expression_template,Dummy>
    {
        using type = combiner;
    };
public:
    using type = typename selector_<typename config_type::engine, void, Ts...>::type;
};
template<typename...Ts> using combiner_selector_t = typename combiner_selector<Ts...>::type;

}   //end of namespace gtensor
#endif