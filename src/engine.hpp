#ifndef ENGINE_HPP_
#define ENGINE_HPP_

#include <vector>
#include <array>
#include "tensor_base.hpp"
#include "tensor_init_list.hpp"
#include "shareable_storage.hpp"
#include "indexer.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail

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

/**
 *
 * engines are about meaning not form, as opposed to descriptors that are about form
 *
 * **/

template<typename CfgT, typename StorT>
class storage_engine :
    protected engine_host_accessor<typename StorT::value_type, CfgT>
{
public:
    using value_type = typename StorT::value_type;
    using config_type = CfgT;
protected:
    using storage_type = StorT;
    using index_type = typename config_type::index_type;
    using typename engine_host_accessor::host_type;
    auto create_indexer()const{return create_indexer_helper(*this);}
    auto create_indexer(){return create_indexer_helper(*this);}
    auto begin()const{return elements_.begin();}
    auto end()const{return elements_.end();}
    auto begin(){return elements_.begin();}
    auto end(){return elements_.end();}
public:
    template<typename Nested>
    storage_engine(host_type* host, const index_type& size, std::initializer_list<Nested> init_data):
        engine_host_accessor{host},
        elements_(size)
    {detail::fill_from_list(init_data, elements_.begin());}
    storage_engine(host_type* host, const index_type& size, const value_type& init_data):
        engine_host_accessor{host},
        elements_(size, init_data)
    {}
    template<typename ItT>
    storage_engine(host_type* host, ItT begin, ItT end):
        engine_host_accessor{host},
        elements_(begin, end)
    {}
private:
    template<typename U> static auto create_indexer_helper(U& instance){
        return basic_indexer<index_type, decltype(instance.begin())>{instance.begin()};
    }
    storage_type elements_;
};

template<typename ValT, typename CfgT, typename F, typename OperandsNumber>
class evaluating_engine :
    protected engine_host_accessor<ValT, CfgT>
{
public:
    using value_type = ValT;
    using config_type = CfgT;
protected:
    constexpr static std::size_t operands_number  = OperandsNumber::value;
    using operand_base_type = tensor_base_base<config_type>;
    using typename engine_host_accessor::host_type;
    const auto& operands()const{return operands_;}
public:
    template<typename...Ts>
    evaluating_engine(host_type* host, F&& f, Ts&&...operands):
        engine_host_accessor{host},
        f_{std::move(f)},
        operands_{std::forward<Ts>(operands)...}
    {}
private:
    F f_;
    std::array<std::shared_ptr<operand_base_type>,operands_number> operands_;
};

template<typename ValT, typename CfgT, typename DescT, typename ParentT>
class viewing_engine :
    protected engine_host_accessor<ValT, CfgT>
{
public:
    using value_type = ValT;
    using config_type = CfgT;
protected:
    using typename engine_host_accessor::host_type;
    using descriptor_type = DescT;
    using parent_type = ParentT;
    const parent_type* parent()const{return parent_.get();}
    parent_type* parent(){return parent_.get();}
    auto create_indexer()const{return create_indexer_helper(*this);}
    auto create_indexer(){return create_indexer_helper(*this);}
public:
    template<typename U>
    viewing_engine(host_type* host, U&& parent):
        engine_host_accessor{host},
        parent_{std::forward<U>(parent)}
    {}
private:
    template<typename U>
    static auto create_indexer_helper(U& instance){
        return basic_indexer<typename config_type::index_type, decltype(instance.parent()->engine().create_indexer()), descriptor_type>{
            instance.parent()->engine().create_indexer(),
            static_cast<const descriptor_type&>(instance.host()->descriptor())
        };
    }
    std::shared_ptr<parent_type> parent_;
};

}   //end of namespace gtensor

#endif