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
template<typename...Ts> inline ITERATOR<Ts...> operator+(const ITERATOR<Ts...>& lhs, typename ITERATOR<Ts...>::difference_type n){auto tmp = lhs; return tmp+=n;}
#define GTENSOR_ITERATOR_OPERATOR_MINUS(ITERATOR)\
template<typename...Ts> inline ITERATOR<Ts...> operator-(const ITERATOR<Ts...>& lhs, typename ITERATOR<Ts...>::difference_type n){auto tmp = lhs; return tmp+=-n;}
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
    template<typename ValT> struct iterator_internals_selector{
        using value_type = ValT;
        using pointer = value_type*;
        using reference = value_type&;
        using const_reference = const value_type&;
    };
    template<> struct iterator_internals_selector<void>{
        using value_type = void;
        using pointer = void;
        using reference = void;
        using const_reference = void;
    };

    template<typename CfgT>
    class broadcast_shape_iterator_extension
    {
        using shape_type = typename CfgT::shape_type;
        using strides_div_type = typename detail::strides_div_traits<CfgT>::type;
    protected:
        shape_type shape_;
        strides_div_type strides_;
    public:
        template<typename ShT, typename StT>
        broadcast_shape_iterator_extension(ShT&& shape__, StT&& strides__):
            shape_{std::forward<ShT>(shape__)},
            strides_{std::forward<StT>(strides__)}
        {}
    };

}   //end of namespace detail

//random access iterator, use indexer data accessor
template<typename Config, typename Indexer>
class indexer_iterator
{
protected:
    using indexer_type = Indexer;
    using index_type = typename Config::index_type;
    using result_type = decltype(std::declval<indexer_type>()[std::declval<index_type>()]);
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::decay_t<result_type>;
    using difference_type = index_type;
    using pointer = typename detail::iterator_internals_selector<value_type>::pointer;
    using reference = typename detail::iterator_internals_selector<value_type>::reference;
    using const_reference = typename detail::iterator_internals_selector<value_type>::const_reference;

    //assuming ussual stoarge subscript operator semantic i.e. subscript index in range [0,size()-1]:
    //begin should be constructed with zero flat_index_ argument, end with size() flat_index_argument
    template<typename Indexer_, std::enable_if_t<!std::is_convertible_v<Indexer_,indexer_iterator> ,int> =0>
    indexer_iterator(Indexer_&& indexer_, const difference_type& flat_index_):
        indexer{std::forward<Indexer_>(indexer_)},
        flat_index{flat_index_}
    {}
    auto& operator+=(difference_type n){return advance(n);}
    result_type operator[](difference_type n)const{return *(*this+n);}
    result_type operator*() const{return indexer[flat_index];}
    inline difference_type friend operator-(const indexer_iterator& lhs, const indexer_iterator& rhs){return lhs.flat_index - rhs.flat_index;}
private:
    auto& advance(difference_type n){
        flat_index+=n;
        return *this;
    }
    indexer_type indexer;
    difference_type flat_index;
};

GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_PLUS(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_MINUS(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_PREFIX_INC(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_PREFIX_DEC(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_EQUAL(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_NOT_EQUAL(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_GREATER(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_LESS(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_GREATER_EQUAL(indexer_iterator);
GTENSOR_ITERATOR_OPERATOR_LESS_EQUAL(indexer_iterator);

//random access iterator, use walker data accessor
template<typename Config, typename Walker>
class walker_iterator
{
protected:
    using walker_type = Walker;
    using config_type = Config;
    using result_type = decltype(*std::declval<walker_type>());
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using strides_div_type = typename detail::strides_div_traits<config_type>::type;
    using traverser_type = walker_random_access_traverser<config_type, walker_type>;
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::decay_t<result_type>;
    using difference_type = typename config_type::index_type;
    using pointer = typename detail::iterator_internals_selector<value_type>::pointer;
    using reference = typename detail::iterator_internals_selector<value_type>::reference;
    using const_reference = typename detail::iterator_internals_selector<value_type>::const_reference;

    //begin should be constructed with zero flat_index_ argument, end with size() flat_index_argument
    template<typename Walker_, std::enable_if_t<!std::is_convertible_v<Walker_,walker_iterator> ,int> =0>
    walker_iterator(Walker_&& walker_, const shape_type& shape_, const strides_div_type& strides_, const difference_type& flat_index_):
        traverser{shape_, strides_, std::forward<Walker_>(walker_)},
        flat_index{flat_index_}
    {
        if (flat_index_ > difference_type{0}){
            traverser.move(flat_index_-difference_type(1));
            traverser.next();
        }
    }
    auto& operator+=(difference_type n){
        advance(n);
        return *this;
    }
    result_type operator[](difference_type n)const{return *(*this+n);}
    result_type operator*() const{return *traverser.walker();}
    inline difference_type friend operator-(const walker_iterator& lhs, const walker_iterator& rhs){return lhs.flat_index - rhs.flat_index;}
private:
    void advance(difference_type n){
        flat_index+=n;
        traverser.move(flat_index);
    }
    traverser_type traverser;
    difference_type flat_index;
};

GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_PLUS(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_MINUS(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_PREFIX_INC(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_PREFIX_DEC(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_EQUAL(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_NOT_EQUAL(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_GREATER(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_LESS(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_GREATER_EQUAL(walker_iterator);
GTENSOR_ITERATOR_OPERATOR_LESS_EQUAL(walker_iterator);

// template<typename CfgT, typename WalkerT>
// class broadcast_shape_iterator:
//     private detail::broadcast_shape_iterator_extension<CfgT>,
//     public broadcast_iterator<CfgT,WalkerT>
// {
//     using extension_base = detail::broadcast_shape_iterator_extension<CfgT>;
//     using broadcast_iterator_base = broadcast_iterator<CfgT,WalkerT>;
// protected:
//     using typename broadcast_iterator_base::walker_type;
//     using typename broadcast_iterator_base::result_type;
//     using typename broadcast_iterator_base::shape_type;
//     using typename broadcast_iterator_base::index_type;
//     using typename broadcast_iterator_base::strides_div_type;
// public:
//     using typename broadcast_iterator_base::iterator_category;
//     using typename broadcast_iterator_base::value_type;
//     using typename broadcast_iterator_base::difference_type;
//     using typename broadcast_iterator_base::pointer;
//     using typename broadcast_iterator_base::reference;
//     using typename broadcast_iterator_base::const_reference;

//     template<typename WalkerT_, typename ShT, typename StT>
//     broadcast_shape_iterator(WalkerT_&& walker__, ShT&& shape__, StT&& strides__):
//         extension_base{std::forward<ShT>(shape__), std::forward<StT>(strides__)},
//         broadcast_iterator_base{std::forward<WalkerT_>(walker__), extension_base::shape_, extension_base::strides_}
//     {}
//     template<typename WalkerT_, typename ShT, typename StT, typename IdxT>
//     broadcast_shape_iterator(WalkerT_&& walker__, ShT&& shape__, StT&& strides__, const IdxT& size__):
//         extension_base{std::forward<ShT>(shape__), std::forward<StT>(strides__)},
//         broadcast_iterator_base{std::forward<WalkerT_>(walker__), extension_base::shape_, extension_base::strides_, size__}
//     {}
// };

template<typename Iterator>
class reverse_iterator_generic : private Iterator
{
    using iterator_base = Iterator;
protected:
    using typename iterator_base::result_type;
public:
    using iterator_category = std::random_access_iterator_tag;
    using typename iterator_base::value_type;
    using typename iterator_base::difference_type;
    using typename iterator_base::pointer;
    using typename iterator_base::reference;
    using typename iterator_base::const_reference;
    using iterator_base::operator*;

    explicit reverse_iterator_generic(Iterator it):
        iterator_base{std::move(it)}
    {
        ++(*this);
    }

    template<typename...> struct forward_args : std::true_type{};
    template<typename U> struct forward_args<U> : std::bool_constant<!std::is_same_v<U,Iterator>&&!std::is_same_v<U,reverse_iterator_generic>>{};

    template<typename...Args, std::enable_if_t<forward_args<Args...>::value ,int> =0>
    explicit reverse_iterator_generic(Args&&...args):
        iterator_base{std::forward<Args>(args)...}
    {
        ++(*this);
    }
    auto& operator+=(difference_type n){
        iterator_base::operator+=(-n);
        return *this;
    }
    result_type operator[](difference_type n)const{return *(*this+n);}
    inline difference_type friend operator-(const reverse_iterator_generic& lhs, const reverse_iterator_generic& rhs){
        return static_cast<const iterator_base&>(rhs) - static_cast<const iterator_base&>(lhs);
    }
};

GTENSOR_ITERATOR_OPERATOR_ASSIGN_MINUS(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_PLUS(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_MINUS(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_PREFIX_INC(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_PREFIX_DEC(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_INC(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_POSTFIX_DEC(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_EQUAL(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_NOT_EQUAL(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_GREATER(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_LESS(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_GREATER_EQUAL(reverse_iterator_generic);
GTENSOR_ITERATOR_OPERATOR_LESS_EQUAL(reverse_iterator_generic);

template<typename Config, typename Indexer> using reverse_indexer_iterator = reverse_iterator_generic<indexer_iterator<Config,Indexer>>;
template<typename Config, typename Walker> using reverse_walker_iterator = reverse_iterator_generic<walker_iterator<Config,Walker>>;

// template<typename CfgT, typename WalkerT> using reverse_broadcast_shape_iterator = reverse_broadcast_iterator_generic<broadcast_shape_iterator<CfgT,WalkerT>>;

}   //end of namespace gtensor



#endif