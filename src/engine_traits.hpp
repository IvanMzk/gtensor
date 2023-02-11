#ifndef ENGINE_TRAITS_HPP_
#define ENGINE_TRAITS_HPP_

#include "config.hpp"

namespace gtensor{

template<typename ValT, typename CfgT> class storage_engine;
template<typename CfgT, typename F, typename...Operands> class evaluating_engine;
template<typename CfgT, typename DescT, typename ParentT> class viewing_engine;

class expression_template_engine_base;
template<typename CfgT, typename DescT, typename ParentT> class expression_template_viewing_engine;
template<typename ValT, typename CfgT> class expression_template_storage_engine;
template<typename CfgT, typename F, typename...Operands> class expression_template_evaluating_engine;

// template<typename ValT, typename CfgT> class expression_template_engine_base;
// template<typename ValT, typename CfgT, typename DescT> class expression_template_view_engine;

namespace detail{



template<typename...> struct engine_base_traits;
template<typename ValT, typename CfgT> struct engine_base_traits<config::engine_expression_template,ValT,CfgT>{
    using type = expression_template_engine_base;
};

template<typename...> struct storage_engine_traits;
template<typename CfgT, typename StorT> struct storage_engine_traits<config::engine_expression_template,CfgT,StorT>{
    using type = expression_template_storage_engine<CfgT,StorT>;
};

template<typename...> struct evaluating_engine_traits;
template<typename CfgT,  typename F, typename...Operands> struct evaluating_engine_traits<config::engine_expression_template,CfgT,F,Operands...>{
    using type = expression_template_evaluating_engine<CfgT,F,Operands...>;
};

template<typename...> struct viewing_engine_traits;
template<typename CfgT, typename DescT, typename ParentT> struct viewing_engine_traits<config::engine_expression_template,CfgT,DescT,ParentT>{
    using type = expression_template_viewing_engine<CfgT,DescT,ParentT>;
};

}   //end of namespace detail

}   //end of namespace gtensor

#endif