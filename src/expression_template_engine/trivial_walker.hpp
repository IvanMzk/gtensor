#ifndef TRIVIAL_WALKER_HPP_
#define TRIVIAL_WALKER_HPP_

#include "walker_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail



template<typename ValT, typename CfgT>
class storage_trivial_walker
{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    const value_type* offset_;

public:
    storage_trivial_walker(const value_type* offset__):
        offset_{offset__}
    {}
    value_type operator[](const index_type& idx)const{return *(offset_+idx);}
};

template<typename ValT, typename CfgT, typename F, typename...Wks>
class evaluating_trivial_walker
{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;

    //std::tuple<Wks...> walkers;
    std::pair<Wks...> walkers;
    F f{};

    // template<typename...U>
    // auto& as_trivial(const walker<U...>& w)const{return w.as_trivial();}
    // template<typename...U>
    // auto& as_trivial(const storage_walker<U...>& w)const{return w;}
    // template<typename...U>
    // auto& as_trivial(const evaluating_trivial_walker<U...>& w)const{return w;}
    // template<typename...U>
    // auto& as_trivial(const evaluating_trivial_root_walker<U...>& w)const{return w;}
    template<typename U>
    auto& as_trivial(const U& w)const{return w;}

public:
    evaluating_trivial_walker(Wks&&...walkers_):
        walkers{std::move(walkers_)...}
    {}
    value_type operator[](const index_type& idx)const {
        return std::apply([&](const auto&...args){return f(as_trivial(args)[idx]...);}, walkers);
    }
};

template<typename ValT, typename CfgT, typename F, typename...Wks>
class evaluating_trivial_root_walker :
    private basic_walker<CfgT, typename CfgT::index_type>,
    private evaluating_trivial_walker<ValT,CfgT,F,Wks...>
{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

public:
    evaluating_trivial_root_walker(const shape_type& shape_, const shape_type& strides_, const shape_type& reset_strides_,  Wks&&...walkers_):
        basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, reset_strides_, index_type{0}},
        evaluating_trivial_walker{std::move(walkers_)...}
    {}

    void walk(const index_type& direction, const index_type& steps){basic_walker::walk(direction,steps);}
    void step(const index_type& direction){basic_walker::step(direction);}
    void step_back(const index_type& direction){basic_walker::step_back(direction);}
    void reset(const index_type& direction){basic_walker::reset(direction);}
    void reset(){basic_walker::reset();}
    value_type operator[](const index_type& idx)const {return evaluating_trivial_walker::operator[](idx);}
    value_type operator*() const {return operator[](cursor());}
};

template<typename CfgT, typename IndexerT, typename DescT>
class viewing_trivial_walker
{
    using typename basic_walker::index_type;
    using typename basic_walker::shape_type;
    using indexer_type = IndexerT;

    mutable indexer_type indexer;
public:
    viewing_trivial_walker(const shape_type& shape_,  const shape_type& strides_, const shape_type& reset_strides_, const index_type& offset_, const indexer_type& indexer_):
        basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, reset_strides_, offset_},
        indexer{indexer_}
    {}

    void walk(const index_type& direction, const index_type& steps){basic_walker::walk(direction,steps);}
    void step(const index_type& direction){basic_walker::step(direction);}
    void step_back(const index_type& direction){basic_walker::step_back(direction);}
    void reset(const index_type& direction){basic_walker::reset(direction);}
    void reset(){basic_walker::reset();}
    auto operator*()const{return indexer[cursor()];}
};

}   //end of namespace gtensor

#endif