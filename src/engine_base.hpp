#ifndef ENGINE_BASE_HPP_
#define ENGINE_BASE_HPP_

#include "config.hpp"

namespace gtensor{

namespace detail{

template<typename> struct engine_traits;

template<typename ValT, typename CfgT> 
struct engine_traits<tensor_base<ValT,CfgT>>{using type = expression_template_engine_base<ValT,CfgT>;};

template<typename ValT, typename CfgT> 
struct engine_traits<storage_tensor<ValT,CfgT>>{using type = expression_template_storage_engine<ValT,CfgT>;};

template<typename ValT, typename CfgT, typename DescT> 
struct engine_traits<viewing_tensor<ValT,CfgT, DescT>>{using type = expression_template_view_engine<ValT,CfgT, DescT>;};

template<typename ValT, typename CfgT, typename F, typename...Ops> 
struct engine_traits<evaluating_tensor<ValT, CfgT, F, Ops...>>{using type = expression_template_elementwise_engine<ValT,CfgT,F,Ops...>;};

}   //end of namespace detail


// template<typename EngineT, typename ValT, typename CfgT, typename F, typename...Ops>
// class evaluating_engine_root_accessor
// {        
//     using root_type = evaluating_tensor<ValT, CfgT, F, EngineT, Ops...>;    
//     root_type* root_{nullptr};
//     friend root_type;
//     void set_root(tensor_base<ValT,CfgT>* root__){root_ = static_cast<root_type*>(root__);}
// protected:
//     auto root()const{return root_;}
// };

template<template<typename...> typename RootT, typename...Args>
class engine_root_accessor
{        
    using root_type = RootT<Args...>;    
    root_type* root_{nullptr};
    friend root_type;
    void set_root(root_type* root__){root_ = root__;}
protected:
    auto root()const{return root_;}    
    engine_root_accessor() = default;
    engine_root_accessor(root_type* root__):
        root_{root__}
    {}
};


template<typename ValT, typename CfgT>
class expression_template_engine_base{
public:
    virtual bool is_trivial()const = 0;
};

}   //end of namespace gtensor

#endif