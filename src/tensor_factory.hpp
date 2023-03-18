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
    template<typename ValT> struct storage_engine_selector_<config::engine_expression_template,ValT>
    {
        using type = expression_template_storage_engine<config_type, typename config_type::template storage<ValT>>;
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
template<typename CfgT, typename...Ts> class storage_tensor_implementation_selector
{
    using config_type = CfgT;
    template<typename...> struct storage_tensor_implementation_selector_;
    template<typename Dummy, typename ValT> struct storage_tensor_implementation_selector_<config::engine_expression_template,Dummy,ValT>
    {
        using type = storage_tensor<typename storage_engine_selector<config_type,ValT>::type>;
        using base_type = type;
    };
public:
    using type = typename storage_tensor_implementation_selector_<typename config_type::host_engine, void, Ts...>::type;
    using base_type = typename storage_tensor_implementation_selector_<typename config_type::host_engine, void, Ts...>::base_type;
};
template<typename CfgT, typename...Ts> class evaluating_tensor_implementation_selector
{
    using config_type = CfgT;
    template<typename...> struct evaluating_tensor_implementation_selector_;
    template<typename F, typename...Operands> struct evaluating_tensor_implementation_selector_<config::engine_expression_template,F,Operands...>
    {
        using type = evaluating_tensor<typename evaluating_engine_selector<config_type,F,Operands...>::type>;
        using base_type = type;
    };
public:
    using type = typename evaluating_tensor_implementation_selector_<typename config_type::host_engine, Ts...>::type;
    using base_type = typename evaluating_tensor_implementation_selector_<typename config_type::host_engine, Ts...>::base_type;
};
template<typename CfgT, typename...Ts> class viewing_tensor_implementation_selector
{
    using config_type = CfgT;
    template<typename...> struct viewing_tensor_implementation_selector_;
    template<typename Descriptor, typename Parent> struct viewing_tensor_implementation_selector_<config::engine_expression_template,Descriptor,Parent>
    {
        using type = viewing_tensor<Descriptor, typename viewing_engine_selector<config_type,Descriptor,Parent>::type>;
        using base_type = type;
    };
public:
    using type = typename viewing_tensor_implementation_selector_<typename config_type::host_engine, Ts...>::type;
    using base_type = typename viewing_tensor_implementation_selector_<typename config_type::host_engine, Ts...>::base_type;
};

//tensor factory
template<typename CfgT, typename Selector> class tensor_factory
{
    using impl_base_type = typename Selector::base_type;
    using impl_type = typename Selector::type;
public:
    template<typename...Args>
    static auto make(Args&&...args){
        return tensor<typename impl_type::value_type, CfgT, impl_base_type>::template make_tensor<impl_type>(std::forward<Args>(args)...);
    }
};
template<typename CfgT, typename...Ts> class storage_tensor_factory : public tensor_factory<CfgT, storage_tensor_implementation_selector<CfgT, Ts...>>{};
template<typename CfgT, typename...Ts> class evaluating_tensor_factory : public tensor_factory<CfgT, evaluating_tensor_implementation_selector<CfgT, Ts...>>{};
template<typename CfgT, typename...Ts> class viewing_tensor_factory : public tensor_factory<CfgT, viewing_tensor_implementation_selector<CfgT, Ts...>>{};

}   //end of namespace gtensor

#endif