#ifndef VIEWING_WALKER_HPP_
#define VIEWING_WALKER_HPP_

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

}   //end of namespace gtensor


#endif