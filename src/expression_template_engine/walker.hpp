#ifndef WALKER_HPP_
#define WALKER_HPP_

#include "walker_base.hpp"

namespace gtensor{

template<typename ValT, typename CfgT, typename ImplT>
class walker_polymorphic : public walker_base<ValT, CfgT>
{
    using impl_type = ImplT;
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    impl_type impl_;

    std::unique_ptr<walker_base> clone()const override{return std::make_unique<walker_polymorphic>(*this);}

public:
    walker_polymorphic(impl_type&& impl__):
        impl_{std::move(impl__)}
    {}

    void walk(const index_type& direction, const index_type& steps)override{return impl_.walk(direction,steps);}
    void step(const index_type& direction)override{return impl_.step(direction);}
    void step_back(const index_type& direction)override{return impl_.step_back(direction);}
    void reset(const index_type& direction)override{return impl_.reset(direction);}
    void reset()override{return impl_.reset();}
    value_type operator*() const override{return impl_.operator*();}
};

template<typename ValT, typename CfgT, typename ImplT>
class trivial_walker_polymorphic : public indexer_base<ValT, CfgT>
{
    using typename indexer_base::index_type;
    using typename indexer_base::value_type;
    using impl_type = ImplT;

    impl_type impl_;

    std::unique_ptr<indexer_base> clone()const override{return std::make_unique<trivial_walker_polymorphic>(*this);}
public:
    trivial_walker_polymorphic(impl_type&& impl__):
        impl_{std::move(impl__)}
    {}
    value_type operator[](const index_type& idx)const override{return impl_.operator[](idx);}
};

template<typename ValT, typename CfgT>
class walker{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using impl_base_type = walker_base<ValT, CfgT>;
    std::unique_ptr<impl_base_type> impl;
public:
    walker(std::unique_ptr<impl_base_type>&& impl_):
        impl{std::move(impl_)}
    {}
    walker(const walker& other):
        impl{other.impl->clone()}
    {}
    walker(walker&& other) = default;

    walker& walk(const index_type& direction, const index_type& steps){
        impl->walk(direction,steps);
        return *this;
    }
    walker& step(const index_type& direction){
        impl->step(direction);
        return *this;
    }
    walker& step_back(const index_type& direction){
        impl->step_back(direction);
        return *this;
    }
    walker& reset(const index_type& direction){
        impl->reset(direction);
        return *this;
    }
    walker& reset(){
        impl->reset();
        return *this;
    }
    value_type operator*() const{return impl->operator*();}
    auto& as_trivial()const{return static_cast<const walker_trivial_base<ValT,CfgT>&>(*impl.get());}
};

template<typename ValT, typename CfgT>
class indexer{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using impl_base_type = indexer_base<ValT, CfgT>;

    std::unique_ptr<impl_base_type> impl;
public:
    indexer(std::unique_ptr<impl_base_type>&& impl_):
        impl{std::move(impl_)}
    {}
    indexer(const indexer& other):
        impl{other.impl->clone()}
    {}
    indexer(indexer&& other) = default;

    value_type operator[](const index_type& idx)const{return impl->operator[](idx);}
};



}   //end of namespace gtensor

#endif