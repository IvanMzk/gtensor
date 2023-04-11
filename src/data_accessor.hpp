#ifndef DATA_ACCESSOR_HPP_
#define DATA_ACCESSOR_HPP_

#include <type_traits>

namespace gtensor{

namespace detail{

template<typename SizeT, typename IdxT>
constexpr inline bool can_walk(const SizeT& direction, const SizeT& dim, const IdxT& direction_dim){
    return direction < dim && direction_dim != IdxT(1);
}

template<typename CfgT>
class shape_inverter
{
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using dim_type = typename CfgT::dim_type;

    const index_type* shape_last;

public:
    shape_inverter(const shape_type& shape_):
        shape_last{shape_.data()+shape_.size()-1}
    {}

    //direction must be in range [0,dim-1]
    //0 direction corresponding to last shape element - direction with minimal stride
    //1 direction corresponding to shape element befor last
    //...
    //dim-1 direction correcponding to 0 shape element - direction with max stride
    index_type element(const dim_type& direction)const{return *(shape_last-direction);}
};

}   //end of namespace detail


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

//walker is indexer adapter that allows address data elements using multidimensional index
//Cursor is responsible for storing flat position, it may have semantic of integral type or random access iterator
template<typename Config, typename Cursor>
class walker_common
{
public:
    using cursor_type = Cursor;
    using index_type = typename Config::index_type;
    using dim_type = typename Config::dim_type;
    using shape_type = typename Config::shape_type;

    walker_common(const shape_type& adapted_strides__, const shape_type& reset_strides__, const cursor_type& offset__, const dim_type& max_dim__):
        adapted_strides_{&adapted_strides__},
        reset_strides_{&reset_strides__},
        offset_{offset__},
        cursor_{offset__},
        dim_offset_{max_dim__ - adapted_strides__.size()}
    {}
    //direction argument must be in range [0,max_dim_-1]
    void walk(const dim_type& direction, const index_type& steps){
        if (direction >= dim_offset_){
            cursor_+=steps*(*adapted_strides_)[direction - dim_offset_];
        }
    }
    void step(const dim_type& direction){
        if (direction >= dim_offset_){
            cursor_+=(*adapted_strides_)[direction - dim_offset_];
        }
    }
    void step_back(const dim_type& direction){
        if (direction >= dim_offset_){
            cursor_-=(*adapted_strides_)[direction - dim_offset_];
        }
    }
    void reset(const dim_type& direction){
        if (direction >= dim_offset_){
            cursor_+=(*reset_strides_)[direction - dim_offset_];
        }
    }
    void reset_back(const dim_type& direction){
        if (direction >= dim_offset_){
            cursor_-=(*reset_strides_)[direction - dim_offset_];
        }
    }
    void reset_back(){cursor_ = offset_;}
    cursor_type cursor()const{return cursor_;}
    cursor_type offset()const{return offset_;}
private:
    const shape_type* adapted_strides_;
    const shape_type* reset_strides_;
    cursor_type offset_;
    cursor_type cursor_;
    dim_type dim_offset_;
};

//adapter of Indexer
template<typename Config, typename Indexer>
class walker
{
    using config_type = Config;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using indexer_type = Indexer;

    walker_common<config_type, index_type> index_walker;
    indexer_type indexer;
public:
    walker(const shape_type& adapted_strides_, const shape_type& reset_strides_, const index_type& offset_, const indexer_type& indexer_, const dim_type& max_dim_):
        index_walker{adapted_strides_, reset_strides_, offset_, max_dim_},
        indexer{indexer_}
    {}
    void walk(const dim_type& direction, const index_type& steps){index_walker.walk(direction,steps);}
    void step(const dim_type& direction){index_walker.step(direction);}
    void step_back(const dim_type& direction){index_walker.step_back(direction);}
    void reset(const dim_type& direction){index_walker.reset(direction);}
    void reset_back(const dim_type& direction){index_walker.reset_back(direction);}
    void reset_back(){index_walker.reset_back();}
    decltype(indexer[index_walker.cursor()]) operator*()const{return indexer[index_walker.cursor()];}
};




}   //end of namespace gtensor

#endif