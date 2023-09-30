/*
* GTensor - matrix computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef ITERATOR_HPP_
#define ITERATOR_HPP_

#include <iterator>
#include "config.hpp"
#include "libdivide_helper.hpp"
#include "data_accessor.hpp"

namespace gtensor{

#define GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(ITERATOR)\
template<typename...Ts> inline ITERATOR<Ts...>& operator-=(ITERATOR<Ts...>& lhs, typename ITERATOR<Ts...>::difference_type n){return lhs+=-n;}
#define GTENSOR_ITERATOR_OPERATOR_PLUS(ITERATOR)\
template<typename...Ts> inline ITERATOR<Ts...> operator+(const ITERATOR<Ts...>& lhs, typename ITERATOR<Ts...>::difference_type n){auto tmp = lhs; tmp+=n; return tmp;}
#define GTENSOR_ITERATOR_OPERATOR_MINUS(ITERATOR)\
template<typename...Ts> inline ITERATOR<Ts...> operator-(const ITERATOR<Ts...>& lhs, typename ITERATOR<Ts...>::difference_type n){auto tmp = lhs; tmp+=-n; return tmp;}
#define GTENSOR_ITERATOR_OPERATOR_PREFIX_INC(ITERATOR)\
template<typename...Ts> inline ITERATOR<Ts...>& operator++(ITERATOR<Ts...>& lhs){return lhs+=typename ITERATOR<Ts...>::difference_type{1};}
#define GTENSOR_ITERATOR_OPERATOR_PREFIX_DEC(ITERATOR)\
template<typename...Ts> inline ITERATOR<Ts...>& operator--(ITERATOR<Ts...>& lhs){return lhs+=typename ITERATOR<Ts...>::difference_type{-1};}
#define GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(ITERATOR)\
template<typename...Ts> inline ITERATOR<Ts...> operator++(ITERATOR<Ts...>& lhs, int){auto tmp = lhs; ++lhs; return tmp;}
#define GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(ITERATOR)\
template<typename...Ts> inline ITERATOR<Ts...> operator--(ITERATOR<Ts...>& lhs, int){auto tmp = lhs; --lhs; return tmp;}
#define GTENSOR_ITERATOR_OPERATOR_EQUAL(ITERATOR)\
template<typename...Ts> inline bool operator==(const ITERATOR<Ts...>& lhs, const ITERATOR<Ts...>& rhs){return (lhs - rhs) == typename ITERATOR<Ts...>::difference_type(0);}
#define GTENSOR_ITERATOR_OPERATOR_NOT_EQUAL(ITERATOR)\
template<typename...Ts> inline bool operator!=(const ITERATOR<Ts...>& lhs, const ITERATOR<Ts...>& rhs){return !(lhs==rhs);}
#define GTENSOR_ITERATOR_OPERATOR_GREATER(ITERATOR)\
template<typename...Ts> inline bool operator>(const ITERATOR<Ts...>& lhs, const ITERATOR<Ts...>& rhs){return (lhs - rhs) > typename ITERATOR<Ts...>::difference_type(0);}
#define GTENSOR_ITERATOR_OPERATOR_LESS(ITERATOR)\
template<typename...Ts> inline bool operator<(const ITERATOR<Ts...>& lhs, const ITERATOR<Ts...>& rhs){return (rhs - lhs) > typename ITERATOR<Ts...>::difference_type(0);}
#define GTENSOR_ITERATOR_OPERATOR_GREATER_EQUAL(ITERATOR)\
template<typename...Ts> inline bool operator>=(const ITERATOR<Ts...>& lhs, const ITERATOR<Ts...>& rhs){return !(lhs < rhs);}
#define GTENSOR_ITERATOR_OPERATOR_LESS_EQUAL(ITERATOR)\
template<typename...Ts> inline bool operator<=(const ITERATOR<Ts...>& lhs, const ITERATOR<Ts...>& rhs){return !(lhs > rhs);}

namespace detail{
    template<typename Config>
    class broadcast_iterator_extension
    {
        using shape_type = typename Config::shape_type;
        using strides_div_type = detail::strides_div_t<Config>;
        shape_type shape_;
        strides_div_type strides_;
    public:
        template<typename ShT, typename StT>
        broadcast_iterator_extension(ShT&& shape__, StT&& strides__):
            shape_{std::forward<ShT>(shape__)},
            strides_{std::forward<StT>(strides__)}
        {}
        const auto& shape()const{return shape_;}
        const auto& strides()const{return strides_;}
    };

}   //end of namespace detail

//random access iterator, use indexer data accessor
template<typename Config, typename Indexer>
class indexer_iterator
{
protected:
    using indexer_type = Indexer;
    using index_type = typename Config::index_type;
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = index_type;
    using reference = decltype(std::declval<indexer_type>()[std::declval<index_type>()]);
    using value_type = typename indexer_type::value_type;
    using pointer = std::add_pointer_t<reference>;

    //assuming usual stoarge subscript operator semantic i.e. subscript index in range [0,size()-1]:
    //begin should be constructed with zero flat_index_ argument, end with size() flat_index_argument
    template<typename Indexer_>
    indexer_iterator(Indexer_&& indexer_, const difference_type& flat_index_):
        indexer{std::forward<Indexer_>(indexer_)},
        flat_index{flat_index_}
    {}
    indexer_iterator& operator++(){
        ++flat_index;
        return *this;
    }
    indexer_iterator& operator--(){
        --flat_index;
        return *this;
    }
    indexer_iterator& operator+=(difference_type n){
        flat_index+=n;
        return *this;
    }
    reference operator[](difference_type n)const{
        return *(*this+n);
    }
    reference operator*()const{
        return indexer[flat_index];
    }
    difference_type operator-(const indexer_iterator& rhs)const{
        return flat_index - rhs.flat_index;
    }
    bool operator==(const indexer_iterator& rhs)const{
        return flat_index == rhs.flat_index;
    }
    bool operator!=(const indexer_iterator& rhs)const{
        return flat_index != rhs.flat_index;
    }
    bool operator>(const indexer_iterator& rhs)const{
        return flat_index > rhs.flat_index;
    }
    bool operator>=(const indexer_iterator& rhs)const{
        return flat_index >= rhs.flat_index;
    }
    bool operator<(const indexer_iterator& rhs)const{
        return flat_index < rhs.flat_index;
    }
    bool operator<=(const indexer_iterator& rhs)const{
        return flat_index <= rhs.flat_index;
    }
private:
    indexer_type indexer;
    difference_type flat_index;
};

GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_PLUS(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_MINUS(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(indexer_iterator);

//random access iterator, use walker data accessor
template<typename Config, typename Traverser>
class walker_iterator
{
protected:
    using config_type = Config;
    using traverser_type = Traverser;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using strides_div_type = detail::strides_div_t<config_type>;
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = typename config_type::index_type;
    using reference = decltype(*std::declval<traverser_type>());
    using value_type = typename traverser_type::value_type;
    using pointer = std::add_pointer_t<reference>;

    //begin should be constructed with zero flat_index_ argument, end with size() flat_index_argument
    //args should be axis_min,axis_max for range traverser and empty otherwise
    template<typename Walker_, typename...Args>
    walker_iterator(Walker_&& walker_, const shape_type& shape_, const strides_div_type& strides_, const difference_type& flat_index_, const Args&...args):
        traverser{shape_, strides_, std::forward<Walker_>(walker_),args...},
        flat_index{flat_index_}
    {}
    walker_iterator& operator++(){
        traverser.next();
        ++flat_index;
        return *this;
    }
    walker_iterator& operator--(){
        traverser.prev();
        --flat_index;
        return *this;
    }
    walker_iterator& operator+=(difference_type n){
        flat_index+=n;
        traverser.to(flat_index);
        return *this;
    }
    reference operator[](difference_type n)const{
        return *(*this+n);
    }
    reference operator*()const{
        return *traverser;
    }
    difference_type operator-(const walker_iterator& rhs)const{
        return flat_index - rhs.flat_index;
    }
    bool operator==(const walker_iterator& rhs)const{
        return flat_index == rhs.flat_index;
    }
    bool operator!=(const walker_iterator& rhs)const{
        return flat_index != rhs.flat_index;
    }
    bool operator>(const walker_iterator& rhs)const{
        return flat_index > rhs.flat_index;
    }
    bool operator>=(const walker_iterator& rhs)const{
        return flat_index >= rhs.flat_index;
    }
    bool operator<(const walker_iterator& rhs)const{
        return flat_index < rhs.flat_index;
    }
    bool operator<=(const walker_iterator& rhs)const{
        return flat_index <= rhs.flat_index;
    }

private:
    traverser_type traverser;
    difference_type flat_index;
};

GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_PLUS(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_MINUS(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(walker_iterator);

//random access iterator, use walker data accessor
//iterate along specified axis
template<typename Config, typename Walker>
class axis_iterator
{
protected:
    using walker_type = Walker;
    using index_type = typename Config::index_type;
    using dim_type = typename Config::dim_type;
public:
    using iterator_category = std::random_access_iterator_tag;
    using difference_type = index_type;
    using reference = decltype(*std::declval<walker_type>());
    using value_type = typename walker_type::value_type;
    using pointer = std::add_pointer_t<reference>;

    //begin should be constructed with zero flat_index_ argument, end with size() flat_index_argument
    template<typename Walker_>
    axis_iterator(Walker_&& walker_, const dim_type& axis_, const difference_type& flat_index_):
        walker{std::forward<Walker_>(walker_)},
        axis{axis_},
        flat_index{flat_index_}
    {
        if (flat_index_ > 0){
            walker.walk(axis_, flat_index_ - difference_type{1});
            walker.step(axis_);
        }
    }
    axis_iterator& operator++(){
        walker.step(axis);
        ++flat_index;
        return *this;
    }
    axis_iterator& operator--(){
        walker.step_back(axis);
        --flat_index;
        return *this;
    }
    axis_iterator& operator+=(difference_type n){
        walker.walk(axis, n);
        flat_index+=n;
        return *this;
    }
    reference operator[](difference_type n)const{
        return *(*this+n);
    }
    reference operator*()const{
        return *walker;
    }
    difference_type operator-(const axis_iterator& rhs)const{
        return flat_index - rhs.flat_index;
    }
    bool operator==(const axis_iterator& rhs)const{
        return flat_index == rhs.flat_index;
    }
    bool operator!=(const axis_iterator& rhs)const{
        return flat_index != rhs.flat_index;
    }
    bool operator>(const axis_iterator& rhs)const{
        return flat_index > rhs.flat_index;
    }
    bool operator>=(const axis_iterator& rhs)const{
        return flat_index >= rhs.flat_index;
    }
    bool operator<(const axis_iterator& rhs)const{
        return flat_index < rhs.flat_index;
    }
    bool operator<=(const axis_iterator& rhs)const{
        return flat_index <= rhs.flat_index;
    }
private:
    walker_type walker;
    dim_type axis;
    difference_type flat_index;
};

GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_PLUS(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_MINUS(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(axis_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(axis_iterator);

//random access broadcast iterator
template<typename Config, typename Traverser>
class broadcast_iterator:
    private detail::broadcast_iterator_extension<Config>,
    private walker_iterator<Config,Traverser>
{
    using extension_base = detail::broadcast_iterator_extension<Config>;
    using walker_iterator_base = walker_iterator<Config,Traverser>;
protected:
    using typename walker_iterator_base::shape_type;
    using typename walker_iterator_base::index_type;
    using typename walker_iterator_base::strides_div_type;
public:
    using typename walker_iterator_base::iterator_category;
    using typename walker_iterator_base::value_type;
    using typename walker_iterator_base::difference_type;
    using typename walker_iterator_base::pointer;
    using typename walker_iterator_base::reference;
    using walker_iterator_base::operator*;
    using walker_iterator_base::operator[];

    template<typename Walker_, typename ShT, typename StT>
    broadcast_iterator(Walker_&& walker__, ShT&& shape__, StT&& strides__, const difference_type& flat_index_):
        extension_base{std::forward<ShT>(shape__), std::forward<StT>(strides__)},
        walker_iterator_base{std::forward<Walker_>(walker__), extension_base::shape(), extension_base::strides(), flat_index_}
    {}

    broadcast_iterator& operator++(){
        ++static_cast<walker_iterator_base&>(*this);
        return *this;
    }
    broadcast_iterator& operator--(){
        --static_cast<walker_iterator_base&>(*this);
        return *this;
    }
    broadcast_iterator& operator+=(difference_type n){
        static_cast<walker_iterator_base&>(*this)+=n;
        return *this;
    }
    difference_type operator-(const broadcast_iterator& rhs)const{
        return static_cast<const walker_iterator_base&>(*this) - static_cast<const walker_iterator_base&>(rhs);
    }
    bool operator==(const broadcast_iterator& rhs)const{
        return static_cast<const walker_iterator_base&>(*this) == static_cast<const walker_iterator_base&>(rhs);
    }
    bool operator!=(const broadcast_iterator& rhs)const{
        return static_cast<const walker_iterator_base&>(*this) != static_cast<const walker_iterator_base&>(rhs);
    }
    bool operator>(const broadcast_iterator& rhs)const{
        return static_cast<const walker_iterator_base&>(*this) > static_cast<const walker_iterator_base&>(rhs);
    }
    bool operator>=(const broadcast_iterator& rhs)const{
        return static_cast<const walker_iterator_base&>(*this) >= static_cast<const walker_iterator_base&>(rhs);
    }
    bool operator<(const broadcast_iterator& rhs)const{
        return static_cast<const walker_iterator_base&>(*this) < static_cast<const walker_iterator_base&>(rhs);
    }
    bool operator<=(const broadcast_iterator& rhs)const{
        return static_cast<const walker_iterator_base&>(*this) <= static_cast<const walker_iterator_base&>(rhs);
    }
};

GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(broadcast_iterator);
GTENSOR_ITERATOR_OPERATOR_PLUS(broadcast_iterator);
GTENSOR_ITERATOR_OPERATOR_MINUS(broadcast_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(broadcast_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(broadcast_iterator);

template<typename Iterator>
class reverse_iterator_generic : private Iterator
{
    using iterator_base = Iterator;
public:
    using iterator_category = std::random_access_iterator_tag;
    using typename iterator_base::value_type;
    using typename iterator_base::difference_type;
    using typename iterator_base::pointer;
    using typename iterator_base::reference;
    using iterator_base::operator*;

    template<typename...> struct forward_args : std::true_type{};
    template<typename U> struct forward_args<U> : std::bool_constant<!std::is_same_v<U,reverse_iterator_generic>>{};

    template<typename...Args, std::enable_if_t<forward_args<std::remove_cv_t<std::remove_reference_t<Args>>...>::value ,int> =0>
    explicit reverse_iterator_generic(Args&&...args):
        iterator_base{std::forward<Args>(args)...}
    {
        ++(*this);
    }

    reverse_iterator_generic& operator++(){
        --static_cast<iterator_base&>(*this);
        return *this;
    }
    reverse_iterator_generic& operator--(){
        ++static_cast<iterator_base&>(*this);
        return *this;
    }
    reverse_iterator_generic& operator+=(difference_type n){
        static_cast<iterator_base&>(*this)+=-n;
        return *this;
    }
    reference operator[](difference_type n)const{
        return *(*this+n);
    }
    difference_type operator-(const reverse_iterator_generic& rhs)const{
        return static_cast<const iterator_base&>(rhs) - static_cast<const iterator_base&>(*this);
    }
    bool operator==(const reverse_iterator_generic& rhs)const{
        return static_cast<const iterator_base&>(*this) == static_cast<const iterator_base&>(rhs);
    }
    bool operator!=(const reverse_iterator_generic& rhs)const{
        return static_cast<const iterator_base&>(*this) != static_cast<const iterator_base&>(rhs);
    }
    bool operator>(const reverse_iterator_generic& rhs)const{
        return static_cast<const iterator_base&>(rhs) > static_cast<const iterator_base&>(*this);
    }
    bool operator>=(const reverse_iterator_generic& rhs)const{
        return static_cast<const iterator_base&>(rhs) >= static_cast<const iterator_base&>(*this);
    }
    bool operator<(const reverse_iterator_generic& rhs)const{
        return static_cast<const iterator_base&>(rhs) < static_cast<const iterator_base&>(*this);
    }
    bool operator<=(const reverse_iterator_generic& rhs)const{
        return static_cast<const iterator_base&>(rhs) <= static_cast<const iterator_base&>(*this);
    }
};

GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_PLUS(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_MINUS(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(reverse_iterator_generic);

template<typename Config, typename Indexer> using reverse_indexer_iterator = reverse_iterator_generic<indexer_iterator<Config,Indexer>>;
template<typename Config, typename Traverser> using reverse_walker_iterator = reverse_iterator_generic<walker_iterator<Config,Traverser>>;
template<typename Config, typename Walker> using reverse_axis_iterator = reverse_iterator_generic<axis_iterator<Config,Walker>>;
template<typename Config, typename Traverser> using reverse_broadcast_iterator = reverse_iterator_generic<broadcast_iterator<Config,Traverser>>;

#undef GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS
#undef GTENSOR_ITERATOR_OPERATOR_PLUS
#undef GTENSOR_ITERATOR_OPERATOR_MINUS
#undef GTENSOR_ITERATOR_OPERATOR_PREFIX_INC
#undef GTENSOR_ITERATOR_OPERATOR_PREFIX_DEC
#undef GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC
#undef GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC
#undef GTENSOR_ITERATOR_OPERATOR_EQUAL
#undef GTENSOR_ITERATOR_OPERATOR_NOT_EQUAL
#undef GTENSOR_ITERATOR_OPERATOR_GREATER
#undef GTENSOR_ITERATOR_OPERATOR_LESS
#undef GTENSOR_ITERATOR_OPERATOR_GREATER_EQUAL
#undef GTENSOR_ITERATOR_OPERATOR_LESS_EQUAL

}   //end of namespace gtensor
#endif