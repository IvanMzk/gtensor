#ifndef DATA_ACCESSOR_HPP_
#define DATA_ACCESSOR_HPP_

#include <type_traits>
#include "descriptor.hpp"

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

//Indexer is adaptee
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

//walker_traverser implement algorithms to iterate walker using given shape
//traverse shape may be not native walker shape but shapes must be broadcastable
template<typename Config, typename Walker>
class walker_forward_traverser
{
protected:
    using config_type = Config;
    using walker_type = Walker;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;

    const shape_type* shape_;
    const dim_type dim_;
    walker_type walker_;
    shape_type index_;

public:
    template<typename Walker_>
    walker_forward_traverser(const shape_type& shape__, Walker_&& walker__):
        shape_{&shape__},
        dim_(shape__.size()),
        walker_{std::forward<Walker_>(walker__)},
        index_(dim_, index_type{0})
    {}
    const auto& index()const{return index_;}
    const auto& walker()const{return walker_;}
    auto& walker(){return walker_;}
    bool next(){
        auto direction = dim_;
        auto index_it = index_.end();
        while(direction!=dim_type{0}){
            if (*--index_it == (*shape_)[--direction]-index_type{1}){   //direction at their max
                *index_it = index_type{0};
                walker_.reset_back(direction);
            }else{  //can next on direction
                ++(*index_it);
                walker_.step(direction);
                return true;
            }
        }
        return false;
    }
};

template<typename Config, typename Walker>
class walker_bidirectional_traverser : public walker_forward_traverser<Config, Walker>
{
protected:
    using walker_forward_traverser_base = walker_forward_traverser<Config, Walker>;
    using typename walker_forward_traverser_base::config_type;
    using typename walker_forward_traverser_base::walker_type;
    using typename walker_forward_traverser_base::shape_type;
    using typename walker_forward_traverser_base::index_type;
    using typename walker_forward_traverser_base::dim_type;
    using walker_forward_traverser_base::walker_;
    using walker_forward_traverser_base::dim_;
    using walker_forward_traverser_base::index_;
    using walker_forward_traverser_base::shape_;

    index_type overflow_{0};
public:
    using walker_forward_traverser_base::walker_forward_traverser_base;

    bool next(){
        if (walker_forward_traverser_base::next()){
            return true;
        }else{
            if (overflow_ == index_type{-1}){
                ++overflow_;
                return true;
            }else{
                ++overflow_;
                return false;
            }
        }
    }
    bool prev(){
        dim_type direction{dim_}; //start from direction with min stride
        auto index_it = index_.end();
        while(direction!=dim_type{0}){
            --index_it;
            --direction;
            if (*index_it == index_type{0}){   //direction at their min
                *index_it = (*shape_)[direction]-index_type{1};
                walker_.reset(direction);
            }else{  //can prev on direction
                --(*index_it);
                walker_.step_back(direction);
                return true;
            }
        }
        if (overflow_ == index_type{1}){
            --overflow_;
            return true;
        }else{
            --overflow_;
            return false;
        }
    }
};

template<typename CfgT, typename Walker>
class walker_random_access_traverser : public walker_bidirectional_traverser<CfgT, Walker>
{
    using walker_bidirectional_traverser_base = walker_bidirectional_traverser<CfgT, Walker>;
    using typename walker_bidirectional_traverser_base::config_type;
    using typename walker_bidirectional_traverser_base::walker_type;
    using typename walker_bidirectional_traverser_base::shape_type;
    using typename walker_bidirectional_traverser_base::index_type;
    using typename walker_bidirectional_traverser_base::dim_type;
    using strides_div_type = typename detail::strides_div_traits<CfgT>::type;
    using walker_bidirectional_traverser_base::walker_;
    using walker_bidirectional_traverser_base::dim_;
    using walker_bidirectional_traverser_base::index_;
    using walker_bidirectional_traverser_base::overflow_;
    const strides_div_type* strides_;
public:
    template<typename Walker_>
    walker_random_access_traverser(const shape_type& shape__, const strides_div_type& strides__ ,Walker_&& walker__):
        walker_bidirectional_traverser_base(shape__,walker__),
        strides_{&strides__}
    {}
    //in must be in range [0,size-1], where size = make_size(shape__)
    void move(index_type n){
        walker_.reset_back();
        overflow_ = index_type{0};
        auto index_it = index_.begin();
        dim_type direction{0};
        for(auto strides_it = strides_->begin(); strides_it!=strides_->end(); ++strides_it,++index_it,++direction){
            auto steps = detail::divide(n,*strides_it);
            if (steps!=index_type{0}){
                walker_.walk(direction,steps);
            }
            *index_it = steps;
        }
    }
};

}   //end of namespace gtensor

#endif