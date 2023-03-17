#ifndef TYPE_SELECTOR_HPP_
#define TYPE_SELECTOR_HPP_

#include "config.hpp"
#include "forward_decl.hpp"

namespace gtensor{

//engine type selector
template<typename CfgT, typename...Ts>
class storage_engine_selector
{
    using config_type = CfgT;
    template<typename...> struct storage_engine_selector_;
    template<typename StorT> struct storage_engine_selector_<config::engine_expression_template,StorT>
    {
        using type = expression_template_storage_engine<config_type, StorT>;
    };
public:
    using type = typename storage_engine_selector_<typename config_type::host_engine, Ts...>::type;
};
template<typename CfgT, typename...Ts>
struct evaluating_engine_selector
{
    using config_type = CfgT;
    template<typename...> struct evaluating_engine_selector_;
    template<typename F, typename...Operands> struct evaluating_engine_selector_<config::engine_expression_template,F,Operands...>
    {
        using type = expression_template_evaluating_engine<config_type,F,Operands...>;
    };
public:
    using type = typename evaluating_engine_selector_<typename config_type::host_engine, Ts...>::type;
};
template<typename CfgT, typename...Ts>
struct viewing_engine_selector
{
    using config_type = CfgT;
    template<typename...> struct viewing_engine_selector_;
    template<typename DescT, typename ParentT> struct viewing_engine_selector_<config::engine_expression_template,DescT,ParentT>
    {
        using type = expression_template_viewing_engine<config_type,DescT,ParentT>;
    };
public:
    using type = typename viewing_engine_selector_<typename config_type::host_engine, Ts...>::type;
};

//tensor implementation type selector
template<typename CfgT, typename ValT> class storage_tensor_implementation_selector{
    using config_type = CfgT;
    using value_type = ValT;
    template<typename...> struct storage_tensor_implementation_selector_;
    template<typename Dummy> struct storage_tensor_implementation_selector_<config::engine_expression_template,Dummy>
    {
        using type = storage_tensor<typename storage_engine_selector<config_type,typename config_type::template storage<value_type>>::type>;
    };
public:
    using type = typename storage_tensor_implementation_selector_<typename config_type::host_engine, void>::type;
};

//tensor type selector
template<typename CfgT, typename ValT> class storage_tensor_selector{
    using config_type = CfgT;
    using value_type = ValT;
public:
    using type = tensor<value_type, config_type, typename storage_tensor_implementation_selector<config_type, value_type>::type>;
};

}   //end of namespace gtensor

#endif