#ifndef TRIVIAL_WALKER_HPP_
#define TRIVIAL_WALKER_HPP_

#include "walker_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename CfgT, typename IndexerT>
class indexer_trivial_walker
{
    using indexer_type = IndexerT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using result_type = decltype(std::declval<indexer_type>()[std::declval<index_type>()]);

    indexer_type indexer_;

public:
    indexer_trivial_walker(const indexer_type& indexer__):
        indexer_{indexer__}
    {}
    result_type operator[](const index_type& idx)const{return indexer_[idx];}
};

template<typename ValT, typename CfgT, typename F, typename...Wks>
class evaluating_trivial_walker
{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;

    //std::tuple<Wks...> walkers;
    std::pair<Wks...> walkers;
    F f{};
public:
    evaluating_trivial_walker(Wks&&...walkers_):
        walkers{std::move(walkers_)...}
    {}
    value_type operator[](const index_type& idx)const {
        return std::apply([&](const auto&...args){return f(args[idx]...);}, walkers);
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

// template<typename DescT, typename IndexerT>
// class viewing_trivial_walker
// {
//     using descriptor_type = DescT;
//     using indexer_type = IndexerT;
//     using index_type = typename descriptor_type::index_type;
//     using result_type = typename indexer_type::result_type;

//     const descriptor_type* converter;
//     indexer_type parent_indexer;
// public:
//     viewing_trivial_walker(const descriptor_type& converter_, const indexer_type& parent_indexer_):
//         converter{&converter_},
//         parent_indexer{parent_indexer_}
//     {}
//     result_type operator[](const index_type& idx)const{return parent_indexer[converter->convert(idx)];}
// };

}   //end of namespace gtensor

#endif