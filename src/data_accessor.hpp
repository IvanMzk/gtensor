#ifndef DATA_ACCESSOR_HPP_
#define DATA_ACCESSOR_HPP_

#include <type_traits>
#include "descriptor.hpp"

namespace gtensor{

namespace detail{

template<typename Config, typename ShT, typename Predicate, typename Order>
auto make_strides_div_predicate(const ShT& shape, const Predicate& predicate, Order order){
    using dim_type = typename ShT::difference_type;
    const dim_type dim = detail::make_dim(shape);
    ShT tmp{};
    tmp.reserve(dim);
    for (dim_type i{0}; i!=dim; ++i){
        if (predicate(i)){
            tmp.push_back(shape[i]);
        }
    }
    return detail::make_strides_div<Config>(tmp, order);
}

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
    template<typename Parent_> struct enable_parent_ : std::conjunction<
        std::is_lvalue_reference<Parent_>,
        std::negation<std::is_same<std::remove_cv_t<std::remove_reference_t<Parent_>>, basic_indexer>>
    >{};

    template<typename Parent_, std::enable_if_t<enable_parent_<Parent_>::value,int> =0>
    explicit basic_indexer(Parent_&& parent__):
        parent_{&parent__}
    {}
    template<typename U>
    decltype(auto) operator[](const U& i)const{
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
    template<typename Indexer_, std::enable_if_t<!std::is_same_v<std::remove_cv_t<std::remove_reference_t<Indexer_>>, basic_indexer>,int> =0>
    explicit basic_indexer(Indexer_&& indexer__):
        indexer_{std::forward<Indexer_>(indexer__)}
    {}
    template<typename U>
    decltype(auto) operator[](const U& i)const{
        return indexer_[i];
    }
};
//map data elements using converter
//Indexer is type of previous indexer in the chain or data storage
//Converter is flat index mapper, must provide operator()() that take index as parameter and return mapped index
template<typename Indexer, typename Converter>
class basic_indexer<Indexer, Converter&>
{
    using indexer_type = basic_indexer<Indexer>;
    using Converter_type = Converter;
public:
    template<typename Indexer_, typename Converter_, std::enable_if_t<std::is_lvalue_reference_v<Converter_> ,int> = 0>
    basic_indexer(Indexer_&& indexer__,  Converter_&& converter__):
        indexer_{std::forward<Indexer_>(indexer__)},
        converter_{&converter__}
    {}
    template<typename U>
    decltype(auto) operator[](const U& i)const{
        return indexer_[converter_->operator()(i)];
    }
private:
    indexer_type indexer_;
    const Converter_type* converter_;
};
template<typename Indexer, typename Converter>
class basic_indexer<Indexer, Converter>
{
    static_assert(!std::is_reference_v<Converter>);
    using indexer_type = basic_indexer<Indexer>;
    using Converter_type = Converter;
public:
    template<typename Indexer_, typename Converter_>
    basic_indexer(Indexer_&& indexer__,  Converter_&& converter__):
        indexer_{std::forward<Indexer_>(indexer__)},
        converter_{std::forward<Converter_>(converter__)}
    {}
    template<typename U>
    decltype(auto) operator[](const U& i)const{
        return indexer_[converter_(i)];
    }
private:
    indexer_type indexer_;
    Converter_type converter_;
};

//walker-indexer adaptes walker to indexer interface
//Order can be c_order or f_order, specifies how to treat index parameter of subscript operator
//strides must be in this order
template<typename Walker, typename Order>
class walker_indexer
{
    static_assert(!std::is_reference_v<Walker>);
    using walker_type = Walker;
    using config_type = typename walker_type::config_type;
    using dim_type = typename walker_type::dim_type;
    using index_type = typename walker_type::index_type;
    using strides_div_type = detail::strides_div_t<config_type>;

    const strides_div_type* strides_;
    mutable walker_type walker_;
public:
    template<typename Walker_>
    walker_indexer(const strides_div_type& strides__, Walker_&& walker__):
        strides_{&strides__},
        walker_{std::forward<Walker_>(walker__)}
    {}
    decltype(auto) operator[](index_type i)const{
        walker_.reset_back();
        subscript_helper(i, Order{});
        return *walker_;
    }
private:
    decltype(auto) subscript_helper(index_type i, gtensor::config::c_order)const{
        dim_type direction{0};
        for(auto strides_it = strides_->begin(), srtides_end=strides_->end(); strides_it!=srtides_end; ++strides_it, ++direction){
            auto steps = detail::divide(i,*strides_it);
            if (steps!=index_type{0}){
                walker_.walk(direction,steps);
            }
        }
    }
    decltype(auto) subscript_helper(index_type i, gtensor::config::f_order)const{
        auto direction = static_cast<dim_type>(strides_->size());
        for(auto strides_it = strides_->end(), strides_first=strides_->begin(); strides_it!=strides_first;){
            --strides_it;
            --direction;
            auto steps = detail::divide(i,*strides_it);
            if (steps!=index_type{0}){
                walker_.walk(direction,steps);
            }
        }
    }
};
//iterator-indexer adaptes iterator to indexer interface
template<typename Iterator>
class iterator_indexer
{
    static_assert(!std::is_reference_v<Iterator>);
    using iterator_type = Iterator;
    using difference_type = typename std::iterator_traits<iterator_type>::difference_type;
    iterator_type iterator_;
public:
    //iterator__ argument should point to the first element
    template<typename Iterator_, std::enable_if_t<!std::is_convertible_v<std::decay_t<Iterator_>, iterator_indexer>,int> =0>
    explicit iterator_indexer(Iterator_&& iterator__):
        iterator_{std::forward<Iterator_>(iterator__)}
    {}
    template<typename U>
    decltype(std::declval<iterator_type>().operator*()) operator[](const U& i)const{
        static_assert(std::is_convertible_v<U,difference_type>);
        iterator_type tmp = iterator_;
        detail::advance(tmp, i);
        return *tmp;
    }
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
        dim_offset_{max_dim__ - detail::make_dim(adapted_strides__)}
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
    using indexer_type = Indexer;
public:
    using config_type = Config;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;

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
    decltype(std::declval<indexer_type&>()[std::declval<index_type&>()]) operator*()const{return indexer[index_walker.cursor()];}
private:
    walker_common<config_type, index_type> index_walker;
    indexer_type indexer;
};

//default traverse predicate
struct TraverseAllPredicate{};

//walker_traverser implement algorithms to iterate walker using given shape
//traverse shape may be not native walker shape but shapes must be broadcastable
//Predicate specify what directions should be traversed
template<typename Config, typename Walker, typename Predicate = TraverseAllPredicate>
class walker_forward_traverser
{
public:
    using config_type = Config;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
protected:
    using walker_type = Walker;
    using predicate_type = Predicate;

    const shape_type* shape_;
    dim_type dim_;
    walker_type walker_;
    shape_type index_;
    predicate_type predicate_;
public:
    template<typename Walker_,typename Predicate_>
    walker_forward_traverser(const shape_type& shape__, Walker_&& walker__, Predicate_&& predicate__):
        shape_{&shape__},
        dim_(detail::make_dim(shape__)),
        walker_{std::forward<Walker_>(walker__)},
        index_(dim_, index_type{0}),
        predicate_{std::forward<Predicate_>(predicate__)}
    {}
    template<typename P = Predicate, typename Walker_, std::enable_if_t<std::is_same_v<P,TraverseAllPredicate>,int> = 0>
    walker_forward_traverser(const shape_type& shape__, Walker_&& walker__):
        walker_forward_traverser(shape__, std::forward<Walker_>(walker__), TraverseAllPredicate{})
    {}
    const auto& index()const{return index_;}
    const auto& walker()const{return walker_;}
    auto& walker(){return walker_;}
    template<typename Order>
    bool next(){
        ASSERT_ORDER(Order);
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            return next_c();
        }else{
            return next_f();
        }
    }
private:
    bool next_c(){
        auto index_it = index_.end();
        for (dim_type direction{dim_}; direction!=dim_type{0};){
            --index_it;
            --direction;
            if (next_on_direction(direction, *index_it)){
                return true;
            }
        }
        return false;
    }
    bool next_f(){
        auto index_it = index_.begin();
        for (dim_type direction{0}; direction!=dim_; ++direction,++index_it){
            if (next_on_direction(direction, *index_it)){
                return true;
            }
        }
        return false;
    }
    bool next_on_direction(dim_type direction, index_type& index){
        if constexpr (!std::is_same_v<predicate_type,TraverseAllPredicate>){
            if (predicate_(direction) == false){    //traverse directions with true predicate only
                return false;
            }
        }
        if (index == (*shape_)[direction]-index_type{1}){   //direction at their max
            index = index_type{0};
            walker_.reset_back(direction);
        }else{  //can next on direction
            ++index;
            walker_.step(direction);
            return true;
        }
        return false;
    }
};

template<typename Config, typename Walker, typename Predicate = TraverseAllPredicate>
class walker_bidirectional_traverser : public walker_forward_traverser<Config, Walker, Predicate>
{
    using walker_forward_traverser_base = walker_forward_traverser<Config, Walker, Predicate>;
protected:
    using typename walker_forward_traverser_base::predicate_type;
    using walker_forward_traverser_base::walker_;
    using walker_forward_traverser_base::dim_;
    using walker_forward_traverser_base::index_;
    using walker_forward_traverser_base::shape_;
    using walker_forward_traverser_base::predicate_;
public:
    using typename walker_forward_traverser_base::config_type;
    using typename walker_forward_traverser_base::dim_type;
    using typename walker_forward_traverser_base::index_type;
    using typename walker_forward_traverser_base::shape_type;
    using walker_forward_traverser_base::walker_forward_traverser_base;

    template<typename Order>
    bool prev(){
        ASSERT_ORDER(Order);
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            return prev_c();
        }else{
            return prev_f();
        }
    }
    void to_last(){
        dim_type direction{dim_};
        if constexpr (std::is_same_v<predicate_type,TraverseAllPredicate>){
            walker_.reset_back();
            while(direction!=dim_type{0}){
                --direction;
                walker_.reset(direction);
                index_[direction] = (*shape_)[direction]-index_type{1};
            }
        }else{
            while(direction!=dim_type{0}){
                --direction;
                if (predicate_(direction)){
                    auto dec_direction_size = (*shape_)[direction]-index_type{1};
                    walker_.walk(direction, dec_direction_size-index_[direction]);
                    index_[direction] = dec_direction_size;
                }
            }
        }
    }
private:
    bool prev_c(){
        auto index_it = index_.end();
        for (dim_type direction{dim_}; direction!=dim_type{0};){
            --index_it;
            --direction;
            if (prev_on_direction(direction, *index_it)){
                return true;
            }
        }
        return false;
    }
    bool prev_f(){
        auto index_it = index_.begin();
        for (dim_type direction{0}; direction!=dim_; ++direction,++index_it){
            if (prev_on_direction(direction, *index_it)){
                return true;
            }
        }
        return false;
    }
    bool prev_on_direction(dim_type direction, index_type& index){
        if constexpr (!std::is_same_v<predicate_type,TraverseAllPredicate>){
            if (predicate_(direction) == false){    //traverse directions with true predicate only
                return false;
            }
        }
        if (index == index_type{0}){   //direction at their min
            index = (*shape_)[direction]-index_type{1};
            walker_.reset(direction);
        }else{  //can prev on direction
            --index;
            walker_.step_back(direction);
            return true;
        }
        return false;
    }
};

template<typename Config, typename Walker, typename Order, typename Predicate>
class walker_random_access_traverser_common : public walker_bidirectional_traverser<Config, Walker, Predicate>
{
    ASSERT_ORDER(Order);
    using walker_bidirectional_traverser_base = walker_bidirectional_traverser<Config, Walker, Predicate>;
    using typename walker_bidirectional_traverser_base::predicate_type;
    using walker_bidirectional_traverser_base::walker_;
    using walker_bidirectional_traverser_base::index_;
    using walker_bidirectional_traverser_base::dim_;
    using walker_bidirectional_traverser_base::predicate_;
protected:
    using typename walker_bidirectional_traverser_base::config_type;
    using typename walker_bidirectional_traverser_base::dim_type;
    using typename walker_bidirectional_traverser_base::index_type;
    using typename walker_bidirectional_traverser_base::shape_type;
    using strides_div_type = detail::strides_div_t<Config>;
    void move_(index_type n){
        if constexpr (std::is_same_v<Order,gtensor::config::c_order>){
            move_c(n);
        }else{
            move_f(n);
        }
    }
public:
    template<typename Walker_, typename Predicate_>
    walker_random_access_traverser_common(const shape_type& shape__, const strides_div_type& strides__, Walker_&& walker__, Predicate_&& predicate__):
        walker_bidirectional_traverser_base(shape__, std::forward<Walker_>(walker__), std::forward<Predicate_>(predicate__)),
        strides_{&strides__}
    {}
    bool next(){return walker_bidirectional_traverser_base::template next<Order>();}
    bool prev(){return walker_bidirectional_traverser_base::template prev<Order>();}

private:
    void move_c(index_type n){
        auto index_it = index_.begin();
        auto strides_it = strides_->begin();
        for (dim_type direction{0}, last{dim_}; direction!=last; ++direction,++index_it){
            if constexpr (!std::is_same_v<predicate_type,TraverseAllPredicate>){
                if (predicate_(direction) == false){    //traverse directions with true predicate only
                    continue;
                }
            }
            auto steps = detail::divide(n,*strides_it);
            if (steps!=index_type{0}){
                walker_.walk(direction,steps);
            }
            *index_it = steps;
            ++strides_it;

        }
    }
    void move_f(index_type n){
        auto index_it = index_.end();
        auto strides_it = strides_->end();
        for (dim_type direction{dim_}, first{0}; direction!=first;){
            --direction;
            --index_it;
            if constexpr (!std::is_same_v<predicate_type,TraverseAllPredicate>){
                if (predicate_(direction) == false){    //traverse directions with true predicate only
                    continue;
                }
            }
            --strides_it;
            auto steps = detail::divide(n,*strides_it);
            if (steps!=index_type{0}){
                walker_.walk(direction,steps);
            }
            *index_it = steps;
        }
    }

    const strides_div_type* strides_;
};

template<typename Config, typename Walker, typename Order, typename Predicate = TraverseAllPredicate>
class walker_random_access_traverser : public walker_random_access_traverser_common<Config,Walker,Order,Predicate>
{
    ASSERT_ORDER(Order);
    using walker_random_access_traverser_common_base = walker_random_access_traverser_common<Config,Walker,Order,Predicate>;
    using typename walker_random_access_traverser_common_base::strides_div_type;
    Walker offset_;
public:
    using typename walker_random_access_traverser_common_base::config_type;
    using typename walker_random_access_traverser_common_base::dim_type;
    using typename walker_random_access_traverser_common_base::index_type;
    using typename walker_random_access_traverser_common_base::shape_type;
    using walker_random_access_traverser_common_base::walker;
    template<typename Walker_, typename Predicate_>
    walker_random_access_traverser(const shape_type& shape__, const strides_div_type& strides__, Walker_&& walker__, Predicate_&& predicate__):
        walker_random_access_traverser_common_base{shape__, strides__, std::forward<Walker_>(walker__), std::forward<Predicate_>(predicate__)},
        offset_{walker()}
    {}

    //n must be in range [0,size-1], where size = make_size(shape__)
    void move(index_type n){
        walker() = offset_;
        walker_random_access_traverser_common_base::move_(n);
    }
};

template<typename Config, typename Walker, typename Order>
class walker_random_access_traverser<Config,Walker,Order,TraverseAllPredicate> : public walker_random_access_traverser_common<Config,Walker,Order,TraverseAllPredicate>
{
    ASSERT_ORDER(Order);
    using walker_random_access_traverser_common_base = walker_random_access_traverser_common<Config,Walker,Order,TraverseAllPredicate>;
    using strides_div_type = typename walker_random_access_traverser_common_base::strides_div_type;
public:
    using typename walker_random_access_traverser_common_base::config_type;
    using typename walker_random_access_traverser_common_base::dim_type;
    using typename walker_random_access_traverser_common_base::index_type;
    using typename walker_random_access_traverser_common_base::shape_type;
    using walker_random_access_traverser_common_base::walker_random_access_traverser_common_base;

    template<typename Walker_>
    walker_random_access_traverser(const shape_type& shape__, const strides_div_type& strides__, Walker_&& walker__):
        walker_random_access_traverser_common_base{shape__,strides__,std::forward<Walker_>(walker__),TraverseAllPredicate{}}
    {}

    //n must be in range [0,size-1], where size = make_size(shape__)
    void move(index_type n){
        walker_random_access_traverser_common_base::walker().reset_back();
        walker_random_access_traverser_common_base::move_(n);
    }
};

}   //end of namespace gtensor
#endif