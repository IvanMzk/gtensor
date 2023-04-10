#ifndef DATA_ACCESSOR_HPP_
#define DATA_ACCESSOR_HPP_

#include <type_traits>

namespace gtensor{

//basic_indexer provides interface for random access of underlaying storage using subscription operator
//for storage engine it may wrapps its iterator, random access iterator required
//for viewing engine it is wrapper around its parent's indexer and use descriptor for index mapping
//subscription operator result can be used for assigning, if underlaying indexer's subscription result allows it
template<typename...> class basic_indexer;
template<typename Indexer>
class basic_indexer<Indexer>{
    using indexer_type = Indexer;
    indexer_type indexer_;
public:
    template<typename Indexer_>
    basic_indexer(Indexer_&& indexer__):
        indexer_{std::forward<Indexer_>(indexer__)}
    {}
    template<typename U>
    decltype(std::declval<indexer_type>()[std::declval<U>()]) operator[](const U& i){
        return indexer_[i];
    }
};

template<typename Indexer>
class basic_indexer<Indexer&>{
    using indexer_type = Indexer;
    indexer_type* indexer_;
public:
    basic_indexer(indexer_type& indexer__):
        indexer_{&indexer__}
    {}
    template<typename U>
    decltype(std::declval<indexer_type>()[std::declval<U>()]) operator[](const U& i){
        return (*indexer_)[i];
    }
};

template<typename Indexer, typename Descriptor>
class basic_indexer<Indexer, Descriptor> : public basic_indexer<Indexer>
{
    using basic_indexer_base = basic_indexer<Indexer>;
    using descriptor_type = Descriptor;
public:
    using typename basic_indexer_base::indexer_type;
    template<typename Indexer_>
    basic_indexer(Indexer_&& indexer__, const descriptor_type& converter__):
        basic_indexer_base{std::forward<Indexer_>(indexer__)},
        converter_{&converter__}
    {}
    template<typename U>
    decltype(std::declval<basic_indexer_base>()[std::declval<descriptor_type>().convert(std::declval<U>())]) operator[](const U& i){
        return basic_indexer_base::operator[](converter_->convert(i));
    }
private:
    const descriptor_type* converter_;
};



}   //end of namespace gtensor

#endif