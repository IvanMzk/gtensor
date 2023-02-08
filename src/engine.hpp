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
    template<typename, typename = void> constexpr bool is_iterator = false;
    template<typename T> constexpr bool is_iterator<T,std::void_t<typename std::iterator_traits<T>::iterator_category>> = true;
}   //end of namespace detail

template<typename ValT, typename CfgT>
class engine_holder_accessor
{
public:
    using holder_type = tensor_base<ValT,CfgT>;
    engine_holder_accessor() = default;
    engine_holder_accessor(holder_type* holder__):
        holder_{holder__}
    {}
    void set_holder(holder_type* holder__){holder_ = holder__;}
    auto holder()const{return holder_;}
private:
    holder_type* holder_{nullptr};
};

/**
 *
 * engines are about meaning not form, as opposed to descriptors that are about form
 *
 * **/

template<typename CfgT, typename StorT>
class storage_engine
{
public:
    using value_type = typename StorT::value_type;
    using config_type = CfgT;
protected:
    using storage_type = StorT;
    using index_type = typename config_type::index_type;
    using holder_accessor_type = engine_holder_accessor<value_type, config_type>;
    using holder_type = typename holder_accessor_type::holder_type;
    auto holder()const{return holder_accessor.holder();}
    auto create_indexer()const{return create_indexer_helper(*this);}
    auto create_indexer(){return create_indexer_helper(*this);}
    auto begin()const{return elements_.begin();}
    auto end()const{return elements_.end();}
    auto begin(){return elements_.begin();}
    auto end(){return elements_.end();}
public:
    template<typename Nested>
    storage_engine(holder_type* holder, const index_type& size, std::initializer_list<Nested> init_data):
        holder_accessor{holder},
        elements_(size)
    {detail::fill_from_list(init_data, elements_.begin());}
    storage_engine(holder_type* holder, const index_type& size, const value_type& init_data):
        holder_accessor{holder},
        elements_(size, init_data)
    {}
    template<typename ItT, std::enable_if_t<detail::is_iterator<ItT> ,int> =0 >
    storage_engine(holder_type* holder, const index_type& size, ItT begin, ItT end):
        holder_accessor{holder},
        elements_(size)
    {
        auto n = std::distance(begin,end);
        auto n_to_copy = size > n ? n : size;
        std::copy_n(begin,n_to_copy,elements_.begin());
    }
private:
    template<typename U>
    static auto create_indexer_helper(U& instance){
        return basic_indexer<index_type, decltype(instance.begin())>{instance.begin()};
    }
    holder_accessor_type holder_accessor;
    storage_type elements_;
};

template<typename ValT, typename CfgT, typename F, typename OperandsNumber>
class evaluating_engine
{
public:
    using value_type = ValT;
    using config_type = CfgT;
protected:
    constexpr static std::size_t operands_number  = OperandsNumber::value;
    using operand_base_type = tensor_base_base<config_type>;
    using holder_accessor_type = engine_holder_accessor<value_type, config_type>;
    using holder_type = typename holder_accessor_type::holder_type;
    auto holder()const{return holder_accessor.holder();}
    const auto& operands()const{return operands_;}
public:
    template<typename...Ts>
    evaluating_engine(holder_type* holder, F&& f, Ts&&...operands):
        holder_accessor{holder},
        f_{std::move(f)},
        operands_{std::forward<Ts>(operands)...}
    {}
private:
    holder_accessor_type holder_accessor;
    F f_;
    std::array<std::shared_ptr<operand_base_type>,operands_number> operands_;
};

template<typename ValT, typename CfgT, typename DescT, typename ParentT>
class viewing_engine
{
public:
    using value_type = ValT;
    using config_type = CfgT;
protected:
    using holder_accessor_type = engine_holder_accessor<value_type, config_type>;
    using holder_type = typename holder_accessor_type::holder_type;
    using descriptor_type = DescT;
    using parent_type = ParentT;
    auto holder()const{return holder_accessor.holder();}
    const parent_type* parent()const{return parent_.get();}
    parent_type* parent(){return parent_.get();}
    auto create_indexer()const{return create_indexer_helper(*this);}
    auto create_indexer(){return create_indexer_helper(*this);}
public:
    template<typename U>
    viewing_engine(holder_type* holder, U&& parent):
        holder_accessor{holder},
        parent_{std::forward<U>(parent)}
    {}
private:
    template<typename U>
    static auto create_indexer_helper(U& instance){
        return basic_indexer<typename config_type::index_type, decltype(instance.parent()->engine().create_indexer()), descriptor_type>{
            instance.parent()->engine().create_indexer(),
            static_cast<const descriptor_type&>(instance.holder()->descriptor())
        };
    }
    holder_accessor_type holder_accessor;
    std::shared_ptr<parent_type> parent_;
};

}   //end of namespace gtensor

#endif