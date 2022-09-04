#ifndef ENGINE_BASE_HPP_
#define ENGINE_BASE_HPP_

#include "config.hpp"

namespace gtensor{


template<typename> struct engine_traits;

template<typename ValT, typename CfgT> 
struct engine_traits<tensor_base<ValT,CfgT>>{using type = expression_template_engine_base<ValT,CfgT>;};

template<typename ValT, typename CfgT> 
struct engine_traits<storage_tensor<ValT,CfgT>>{using type = expression_template_storage_engine<ValT,CfgT>;};

template<typename ValT, typename CfgT, typename DescT> 
struct engine_traits<viewing_tensor<ValT,CfgT, DescT>>{using type = expression_template_view_engine<ValT,CfgT>;};

template<typename ValT, typename CfgT, typename F, typename...Ops> 
struct engine_traits<evaluating_tensor<ValT, CfgT, F, Ops...>>{using type = expression_template_elementwise_engine<ValT,CfgT,F,Ops...>;};
// template<typename ValT, typename CfgT, typename F, typename...Ops> 
// struct engine_traits<evaluating_tensor<ValT, CfgT, F, Ops...>>{using type = expression_template_elementwise_engine<ValT,CfgT,F,Ops...>;};


template<typename ValT, typename CfgT>
class expression_template_engine_base{
public:
    virtual bool is_trivial()const = 0;
    virtual void set_root(const tensor_base<ValT,CfgT>*)const = 0;
};

}   //end of namespace gtensor

#endif