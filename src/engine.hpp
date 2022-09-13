#ifndef ENGINE_HPP_
#define ENGINE_HPP_

#include "config.hpp"
#include "tensor_init_list.hpp"

namespace gtensor{

/**
 * 
 * engines are about meaning not form, as opposed to descriptors that are about form
 * 
 * **/

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
    using index_type = typename CfgT::index_type;
    using value_type = ValT;
    using storage_type = typename CfgT::template storage<value_type>;

    storage_type elements_;
public:
    template<typename Nested>
    storage_engine(host_type* host, const index_type& size, std::initializer_list<Nested> init_data):
        engine_host_accessor{host},
        elements_(size)
    {detail::fill_from_list(init_data, elements_.begin());}

    storage_engine(host_type* host, const index_type& size, const value_type& v):
        engine_host_accessor{host},
        elements_(size, v)
    {}
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class evaluating_engine : 
    protected engine_host_accessor<ValT, CfgT>
{
    using engine_host_accessor::host_type;
    F f_;
    std::tuple<std::shared_ptr<Ops>...> operands_;
public:
    // template<typename...Args, std::enable_if_t<sizeof...(Args)==sizeof...(Ops),int> = 0 >
    // explicit evaluating_engine(Args&&...args):
    //     operands_{std::forward<Args>(args)...}
    // {}
    template<typename...Ts>
    evaluating_engine(host_type* host, F&& f, Ts&&...operands):
        engine_host_accessor{host},
        f_{std::move(f)},
        operands_{std::forward<Ts>(operands)...}
    {}
};



}   //end of namespace gtensor

#endif