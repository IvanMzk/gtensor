#ifndef ENGINE_TRAITS_HPP_
#define ENGINE_TRAITS_HPP_

#include "config.hpp"

namespace gtensor{

template<typename ValT, typename CfgT> class storage_engine;
template<typename ValT, typename CfgT, typename F, typename OperandsNumber> class evaluating_engine;
template<typename ValT, typename CfgT, typename ParentT> class viewing_engine;

template<typename ValT, typename CfgT> class expression_template_engine_base;
template<typename ValT, typename CfgT, typename DescT, typename ParentT> class expression_template_viewing_engine;
template<typename ValT, typename CfgT> class expression_template_storage_engine;
template<typename ValT, typename CfgT, typename F, typename...Ops> class expression_template_nodispatching_engine;
template<typename ValT, typename CfgT, typename F, typename...Ops> class expression_template_root_dispatching_engine;

// template<typename ValT, typename CfgT> class expression_template_engine_base;
// template<typename ValT, typename CfgT, typename DescT> class expression_template_view_engine;

namespace detail{



template<typename...> struct engine_base_traits;
template<typename ValT, typename CfgT> struct engine_base_traits<config::engine_expression_template,ValT,CfgT>{
    using type = expression_template_engine_base<ValT,CfgT>;
};

template<typename...> struct storage_engine_traits;
template<typename ValT, typename CfgT> struct storage_engine_traits<config::engine_expression_template,ValT,CfgT>{
    using type = expression_template_storage_engine<ValT,CfgT>;
};

template<typename...> struct evaluating_engine_traits;
template<typename ValT, typename CfgT,  typename F, typename...Ops> struct evaluating_engine_traits<config::engine_expression_template, ValT,CfgT,F,Ops...>{
    //using type = expression_template_nodispatching_engine<ValT,CfgT,F,Ops...>;
    using type = expression_template_root_dispatching_engine<ValT,CfgT,F,Ops...>;
};

template<typename...> struct viewing_engine_traits;
template<typename ValT, typename CfgT, typename DescT, typename ParentT> struct viewing_engine_traits<config::engine_expression_template, ValT,CfgT,DescT,ParentT>{
    using type = expression_template_viewing_engine<ValT,CfgT,DescT,ParentT>;
};

}   //end of namespace detail

}   //end of namespace gtensor

#endif