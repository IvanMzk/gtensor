#ifndef VIEWING_WALKER_HPP_
#define VIEWING_WALKER_HPP_

#include "walker_base.hpp"
#include "tensor_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename DescT, typename PrevT>
class viewing_indexer
{
    using index_type = typename DescT::index_type;
    const DescT* descriptor;
    PrevT parent_indexer;
public:
    template<typename P>
    viewing_indexer(const DescT& descriptor_, P&& parent_indexer_):
        descriptor{&descriptor_},
        parent_indexer{std::forward<P>(parent_indexer_)}
    {}
    auto operator[](const index_type& idx){return parent_indexer[descriptor->convert(idx)];}
};

template<typename CfgT, typename IndexerT>
class viewing_walker : private basic_walker<CfgT, typename CfgT::index_type>
{
    using typename basic_walker::index_type;
    using typename basic_walker::shape_type;
    using indexer_type = IndexerT;

    mutable indexer_type indexer;
public:
    viewing_walker(const shape_type& shape_,  const shape_type& strides_, const shape_type& reset_strides_, const index_type& offset_, const indexer_type& indexer_):
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

// template<typename ValT, typename CfgT>
// class viewing_evaluating_walker :
//     public walker_base<ValT, CfgT>,
//     private basic_walker<CfgT, typename CfgT::index_type>
// {
//     using value_type = ValT;
//     using index_type = typename CfgT::index_type;
//     using shape_type = typename CfgT::shape_type;

//     mutable indexer<ValT,CfgT> estorage;
//     const converting_base<CfgT>* converter;
//     std::unique_ptr<walker_base<ValT,CfgT>> clone()const override{return std::make_unique<viewing_evaluating_walker>(*this);}

// public:
//     viewing_evaluating_walker(const shape_type& shape_,  const shape_type& strides_, const index_type& offset_, const converting_base<CfgT>* converter_, indexer<ValT,CfgT> estorage_):
//         basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, offset_},
//         converter{converter_},
//         estorage{std::move(estorage_)}
//     {}

//     void walk(const index_type& direction, const index_type& steps)override{basic_walker::walk(direction,steps);}
//     void step(const index_type& direction)override{basic_walker::step(direction);}
//     void step_back(const index_type& direction)override{basic_walker::step_back(direction);}
//     void reset(const index_type& direction)override{basic_walker::reset(direction);}
//     void reset()override{basic_walker::reset();}
//     value_type operator*()const override{
//         return estorage[converter->convert(basic_walker::cursor())];
//     }
// };

}   //end of namespace gtensor


#endif