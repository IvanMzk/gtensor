#ifndef ENGINE_BASE_HPP_
#define ENGINE_BASE_HPP_

#include "config.hpp"

namespace gtensor{

template<typename ValT, typename CfgT> class storage_engine;
template<typename ValT, typename CfgT, typename F, typename...Ops> class evaluating_engine;
template<typename ValT, typename CfgT, typename ParentT> class viewing_engine;

// template<typename ValT, typename CfgT> class expression_template_storage_engine;
// template<typename ValT, typename CfgT, typename DescT> class expression_template_view_engine;
// template<typename ValT, typename CfgT, typename F, typename...Ops> class expression_template_engine;
// template<typename ValT, typename CfgT> class expression_template_engine_base;

namespace detail{

template<typename...> struct storage_engine_traits;
template<typename ValT, typename CfgT> struct storage_engine_traits<config::engine_expression_template,ValT,CfgT>{
    using type = storage_engine<ValT,CfgT>;
};

template<typename...> struct evaluating_engine_traits;
template<typename ValT, typename CfgT,  typename F, typename...Ops> struct evaluating_engine_traits<config::engine_expression_template, ValT,CfgT,F,Ops...>{
    using type = evaluating_engine<ValT,CfgT,F,Ops...>;
};

template<typename...> struct viewing_engine_traits;
template<typename ValT, typename CfgT,  typename ParentT> struct viewing_engine_traits<config::engine_expression_template, ValT,CfgT,ParentT>{
    using type = viewing_engine<ValT,CfgT,ParentT>;
};

}   //end of namespace detail

template<typename ValT, typename CfgT>
class expression_template_engine_base{
public:
    virtual bool is_trivial()const = 0;
};

}   //end of namespace gtensor

#endif