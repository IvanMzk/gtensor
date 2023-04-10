#ifndef DATA_ACCESSOR_HPP_
#define DATA_ACCESSOR_HPP_

#include <type_traits>

namespace gtensor{

//basic indexer is data accessor that uses flat index to address data
//indexers can be chained to make data view
template<typename...> class basic_indexer;
//Should be the first indexer in chain
//Parent is data storage that provide subscript operator
template<typename Parent>
class basic_indexer<Parent&>
{
    using parent_type = Parent;
    parent_type* parent_;
public:
    template<typename Parent_, std::enable_if_t<std::is_lvalue_reference_v<Parent_> && !std::is_convertible_v<std::decay_t<Parent_>, basic_indexer>,int> =0>
    explicit basic_indexer(Parent_&& parent__):
        parent_{&parent__}
    {}
    template<typename U>
    decltype(std::declval<parent_type>()[std::declval<U>()]) operator[](const U& i)const{
        return (*parent_)[i];
    }
};
//shouldn't be the first indexer in the chain
//Indexer is type of previous indexer in the chain
template<typename Indexer>
class basic_indexer<Indexer>
{
    static_assert(!std::is_reference_v<Indexer>);
    using indexer_type = Indexer;
    indexer_type indexer_;
public:
    template<typename Indexer_, std::enable_if_t<!std::is_convertible_v<std::decay_t<Indexer_>, basic_indexer>,int> =0>
    explicit basic_indexer(Indexer_&& indexer__):
        indexer_{std::forward<Indexer_>(indexer__)}
    {}
    template<typename U>
    decltype(std::declval<indexer_type>()[std::declval<U>()]) operator[](const U& i)const{
        return indexer_[i];
    }
};
//map data elements using converter
//Indexer is type of previous indexer in the chain or data storage
//Converter is flat index mapper, must provide operator()() that take index as parameter and return mapped index
template<typename Indexer, typename Converter>
class basic_indexer<Indexer, Converter> : public basic_indexer<Indexer>
{
    using basic_indexer_base = basic_indexer<Indexer>;
    using Converter_type = Converter;
public:
    template<typename Indexer_>
    basic_indexer(Indexer_&& indexer__, const Converter_type& converter__):
        basic_indexer_base{std::forward<Indexer_>(indexer__)},
        converter_{&converter__}
    {}
    template<typename U>
    decltype(std::declval<basic_indexer_base>()[std::declval<Converter_type>().operator()(std::declval<U>())]) operator[](const U& i)const{
        return basic_indexer_base::operator[](converter_->operator()(i));
    }
private:
    const Converter_type* converter_;
};



}   //end of namespace gtensor

#endif