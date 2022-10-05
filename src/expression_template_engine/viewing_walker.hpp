#ifndef VIEWING_WALKER_HPP_
#define VIEWING_WALKER_HPP_

#include "walker_base.hpp"
#include "tensor_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


template<typename...> class viewing_indexer;
template<typename IdxT, typename IndexerT>
class viewing_indexer<IdxT, IndexerT>
{
public:
    using index_type = IdxT;
    using indexer_type = IndexerT;
    using result_type = decltype(std::declval<indexer_type>()[std::declval<index_type>()]);
    template<typename U>
    explicit viewing_indexer(U&& indexer_):
        indexer{std::forward<U>(indexer_)}
    {}
    result_type operator[](const index_type& idx)const{return indexer[idx];}
private:
    indexer_type indexer;
};
template<typename IdxT ,typename IndexerT, typename DescT>
class viewing_indexer<IdxT, IndexerT, DescT> : public viewing_indexer<IdxT, IndexerT>
{
    using base_indexer_type = viewing_indexer<IdxT, IndexerT>;
    using descriptor_type = DescT;
public:
    using typename base_indexer_type::index_type;
    using typename base_indexer_type::indexer_type;
    using typename base_indexer_type::result_type;
    template<typename U>
    viewing_indexer(U&& indexer_, const descriptor_type& converter_):
        base_indexer_type{std::forward<U>(indexer_)},
        converter{&converter_}
    {}
    result_type operator[](const index_type& idx)const{return base_indexer_type::operator[](converter->convert(idx));}
private:
    const descriptor_type* converter;
};

template<typename CfgT, typename IndexerT>
class viewing_walker : private basic_walker<CfgT, typename CfgT::index_type>
{
    using typename basic_walker::index_type;
    using typename basic_walker::shape_type;
    using indexer_type = IndexerT;
    using result_type = typename indexer_type::result_type;

    indexer_type indexer;
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
    result_type operator*()const{return indexer[cursor()];}
};

}   //end of namespace gtensor


#endif