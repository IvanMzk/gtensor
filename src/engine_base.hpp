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


/**
 * 
 * engines are about meaning not form, as opposed to descriptors that are about form
 * 
 * **/


template<typename ValT, typename CfgT>
class expression_template_engine_base{
public:
    virtual bool is_trivial()const = 0;
};

template<typename ValT, typename CfgT>
class engine_host_accessor
{        
protected:
    using host_type = tensor_base<ValT,CfgT>;
    host_type* host_{nullptr};    
    engine_host_accessor() = default;
    engine_host_accessor(host_type* host__):
        host_{host__}
    {}
    void set_host(host_type* host__){host_ = host__;}
    auto host()const{return host_;}    
};


template<typename ValT, typename CfgT>
class storage_engine : 
    protected engine_host_accessor<ValT, CfgT>
{
    using engine_host_accessor::host_type;
    using storage_type = typename CfgT::template storage<ValT>;    
    storage_type elements_;
public:    
    explicit storage_engine(storage_type&& elements):        
        elements_{std::move(elements)}
    {}
    storage_engine(host_type* host, storage_type&& elements):
        engine_host_accessor{host},
        elements_{std::move(elements)}
    {}
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class evaluating_engine : 
    protected engine_host_accessor<ValT, CfgT>
{
    using engine_host_accessor::host_type;
    std::tuple<std::shared_ptr<Ops>...> operands_;
public:
    template<typename...Args, std::enable_if_t<sizeof...(Args)==sizeof...(Ops),int> = 0 >
    explicit evaluating_engine(Args&&...args):
        operands_{std::forward<Args>(args)...}
    {}
    template<typename...Args>
    explicit evaluating_engine(host_type* host, Args&&...args):
        operands_{std::forward<Args>(args)...},
        engine_host_accessor{host}
    {}
};






}   //end of namespace gtensor

#endif