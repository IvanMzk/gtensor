#ifndef ITERATOR_HPP_
#define ITERATOR_HPP_

#include <iterator>
#include "config.hpp"
#include "libdivide_helper.hpp"
#include "broadcast.hpp"

namespace gtensor{

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

/*
* random access iterator, use indexer to access data
* Indexer is indexer type
*/
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
    auto& operator++(){
        ++flat_index;
        return *this;
    };
    auto& operator--(){
        --flat_index;
        return *this;
    };
    auto& operator+=(difference_type n){return advance(n);}
    auto& operator-=(difference_type n){return advance(-n);}
    indexer_iterator operator+(difference_type n) const{
        auto it = *this;
        it+=n;
        return it;
    }
    indexer_iterator operator-(difference_type n) const{
        auto it = *this;
        it-=n;
        return it;
    }
    bool operator==(const indexer_iterator& it)const{return flat_index == it.flat_index;}
    bool operator!=(const indexer_iterator& it)const{return flat_index != it.flat_index;}
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

template<typename Config, typename Indexer>
inline bool operator>(const indexer_iterator<Config,Indexer>& lhs, const indexer_iterator<Config,Indexer>& rhs){
    return (lhs - rhs) > typename indexer_iterator<Config,Indexer>::difference_type(0);
}
template<typename Config, typename Indexer>
inline bool operator<(const indexer_iterator<Config,Indexer>& lhs, const indexer_iterator<Config,Indexer>& rhs){
    return (rhs - lhs) > typename indexer_iterator<Config,Indexer>::difference_type(0);
}
template<typename Config, typename Indexer>
inline bool operator>=(const indexer_iterator<Config,Indexer>& lhs, const indexer_iterator<Config,Indexer>& rhs){return !(lhs < rhs);}
template<typename Config, typename Indexer>
inline bool operator<=(const indexer_iterator<Config,Indexer>& lhs, const indexer_iterator<Config,Indexer>& rhs){return !(lhs > rhs);}


template<typename Config, typename Indexer>
class reverse_indexer_iterator : private indexer_iterator<Config,Indexer>
{
    using indexer_iterator_base = indexer_iterator<Config,Indexer>;
protected:
    using typename indexer_iterator_base::indexer_type;
    using typename indexer_iterator_base::result_type;
    using typename indexer_iterator_base::index_type;
public:
    using iterator_category = std::random_access_iterator_tag;
    using typename indexer_iterator_base::value_type;
    using typename indexer_iterator_base::difference_type;
    using typename indexer_iterator_base::pointer;
    using typename indexer_iterator_base::reference;
    using typename indexer_iterator_base::const_reference;
    using indexer_iterator_base::operator*;

    //rbegin should be constructed with size() flat_index_ argument, rend with zero flat_index_ argument
    template<typename Indexer_, std::enable_if_t<!std::is_convertible_v<Indexer_,indexer_iterator> ,int> =0>
    reverse_indexer_iterator(Indexer_&& indexer_, const difference_type& flat_index_):
        indexer_iterator_base{std::forward<Indexer_>(indexer_), flat_index_}
    {
        indexer_iterator_base::operator--();
    }
    auto& operator++(){
        indexer_iterator_base::operator--();
        return *this;
    }
    auto& operator--(){
        indexer_iterator_base::operator++();
        return *this;
    }
    auto& operator+=(difference_type n){
        indexer_iterator_base::operator-=(n);
        return *this;
    }
    auto& operator-=(difference_type n){
        indexer_iterator_base::operator+=(n);
        return *this;
    }
    reverse_indexer_iterator operator+(difference_type n)const{
        auto tmp = *this;
        tmp+=n;
        return tmp;
    }
    reverse_indexer_iterator operator-(difference_type n)const{
        auto tmp = *this;
        tmp-=n;
        return tmp;
    }
    result_type operator[](difference_type n)const{return *(*this+n);}
    bool operator==(const reverse_indexer_iterator& other)const{return static_cast<const indexer_iterator_base&>(*this) == static_cast<const indexer_iterator_base&>(other);}
    bool operator!=(const reverse_indexer_iterator& other)const{return !(*this == other);}
    inline difference_type friend operator-(const reverse_indexer_iterator& lhs, const reverse_indexer_iterator& rhs){
        return static_cast<const indexer_iterator_base&>(rhs) - static_cast<const indexer_iterator_base&>(lhs);
    }
};

template<typename Config, typename Indexer>
inline bool operator>(const reverse_indexer_iterator<Config,Indexer>& lhs, const reverse_indexer_iterator<Config,Indexer>& rhs){
    return (lhs - rhs) > typename reverse_indexer_iterator<Config,Indexer>::difference_type(0);
}
template<typename Config, typename Indexer>
inline bool operator<(const reverse_indexer_iterator<Config,Indexer>& lhs, const reverse_indexer_iterator<Config,Indexer>& rhs){
    return (rhs - lhs) > typename reverse_indexer_iterator<Config,Indexer>::difference_type(0);
}
template<typename Config, typename Indexer>
inline bool operator>=(const reverse_indexer_iterator<Config,Indexer>& lhs, const reverse_indexer_iterator<Config,Indexer>& rhs){return !(lhs < rhs);}
template<typename Config, typename Indexer>
inline bool operator<=(const reverse_indexer_iterator<Config,Indexer>& lhs, const reverse_indexer_iterator<Config,Indexer>& rhs){return !(lhs > rhs);}


/*
* random access broadcast iterator, use walker to access data
* WalkerT broadcast walker type, must satisfy broadcast walker interface
*/
template<typename CfgT, typename WalkerT>
class broadcast_iterator{
protected:
    using walker_type = WalkerT;
    using config_type = CfgT;
    using result_type = decltype(*std::declval<walker_type>());
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using strides_div_type = typename detail::strides_div_traits<config_type>::type;
    using adapter_type = walker_iterator_adapter<config_type, walker_type>;
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = std::decay_t<result_type>;
    using difference_type = typename config_type::index_type;
    using pointer = typename detail::iterator_internals_selector<value_type>::pointer;
    using reference = typename detail::iterator_internals_selector<value_type>::reference;
    using const_reference = typename detail::iterator_internals_selector<value_type>::const_reference;

    //begin constructor
    template<typename W, std::enable_if_t<std::is_same_v<W,walker_type> ,int> =0 >
    broadcast_iterator(W&& walker_, const shape_type& shape_, const strides_div_type& strides_):
        adapted_walker{shape_, strides_, std::forward<W>(walker_)},
        flat_index{0}
    {}
    //end constructor
    template<typename W, std::enable_if_t<std::is_same_v<W,walker_type> ,int> =0 >
    broadcast_iterator(W&& walker_, const shape_type& shape_, const strides_div_type& strides_, const difference_type& size_):
        adapted_walker{shape_, strides_, std::forward<W>(walker_)},
        flat_index{size_}
    {
        adapted_walker.move(size_-difference_type(1));
        adapted_walker.next();
    }
    auto& operator++(){
        adapted_walker.next();
        ++flat_index;
        return *this;
    }
    auto& operator--(){
        adapted_walker.prev();
        --flat_index;
        return *this;
    }
    auto& operator+=(difference_type n){
        advance(n);
        return *this;
    }
    auto& operator-=(difference_type n){
        advance(-n);
        return *this;
    }
    broadcast_iterator operator+(difference_type n)const{
        auto it = *this;
        it+=n;
        return it;
    }
    broadcast_iterator operator-(difference_type n)const{
        auto it = *this;
        it-=n;
        return it;
    }
    result_type operator[](difference_type n)const{return *(*this+n);}
    inline difference_type friend operator-(const broadcast_iterator& lhs, const broadcast_iterator& rhs){return lhs.flat_index - rhs.flat_index;}
    bool operator==(const broadcast_iterator& other)const{return flat_index == other.flat_index;}
    bool operator!=(const broadcast_iterator& other)const{return flat_index != other.flat_index;}
    result_type operator*() const{return *adapted_walker.walker();}
private:
    void advance(difference_type n){
        flat_index+=n;
        adapted_walker.move(flat_index);
    }

    adapter_type adapted_walker;
    difference_type flat_index;
};

template<typename CfgT, typename WalkerT>
inline bool operator>(const broadcast_iterator<CfgT,WalkerT>& lhs, const broadcast_iterator<CfgT,WalkerT>& rhs){return (lhs - rhs) > typename broadcast_iterator<CfgT,WalkerT>::difference_type(0);}
template<typename CfgT, typename WalkerT>
inline bool operator<(const broadcast_iterator<CfgT,WalkerT>& lhs, const broadcast_iterator<CfgT,WalkerT>& rhs){return (rhs - lhs) > typename broadcast_iterator<CfgT,WalkerT>::difference_type(0);}
template<typename CfgT, typename WalkerT>
inline bool operator>=(const broadcast_iterator<CfgT,WalkerT>& lhs, const broadcast_iterator<CfgT,WalkerT>& rhs){return !(lhs < rhs);}
template<typename CfgT, typename WalkerT>
inline bool operator<=(const broadcast_iterator<CfgT,WalkerT>& lhs, const broadcast_iterator<CfgT,WalkerT>& rhs){return !(lhs > rhs);}

template<typename CfgT, typename WalkerT>
class broadcast_shape_iterator:
    private detail::broadcast_shape_iterator_extension<CfgT>,
    public broadcast_iterator<CfgT,WalkerT>
{
    using extension_base = detail::broadcast_shape_iterator_extension<CfgT>;
    using broadcast_iterator_base = broadcast_iterator<CfgT,WalkerT>;
protected:
    using typename broadcast_iterator_base::walker_type;
    using typename broadcast_iterator_base::result_type;
    using typename broadcast_iterator_base::shape_type;
    using typename broadcast_iterator_base::index_type;
    using typename broadcast_iterator_base::strides_div_type;
public:
    using typename broadcast_iterator_base::iterator_category;
    using typename broadcast_iterator_base::value_type;
    using typename broadcast_iterator_base::difference_type;
    using typename broadcast_iterator_base::pointer;
    using typename broadcast_iterator_base::reference;
    using typename broadcast_iterator_base::const_reference;

    template<typename WalkerT_, typename ShT, typename StT>
    broadcast_shape_iterator(WalkerT_&& walker__, ShT&& shape__, StT&& strides__):
        extension_base{std::forward<ShT>(shape__), std::forward<StT>(strides__)},
        broadcast_iterator_base{std::forward<WalkerT_>(walker__), extension_base::shape_, extension_base::strides_}
    {}
    template<typename WalkerT_, typename ShT, typename StT, typename IdxT>
    broadcast_shape_iterator(WalkerT_&& walker__, ShT&& shape__, StT&& strides__, const IdxT& size__):
        extension_base{std::forward<ShT>(shape__), std::forward<StT>(strides__)},
        broadcast_iterator_base{std::forward<WalkerT_>(walker__), extension_base::shape_, extension_base::strides_, size__}
    {}
};

template<typename IterT>
class reverse_broadcast_iterator_generic : private IterT
{
    using broadcast_iterator_base = IterT;
protected:
    using typename broadcast_iterator_base::walker_type;
    using typename broadcast_iterator_base::result_type;
    using typename broadcast_iterator_base::shape_type;
    using typename broadcast_iterator_base::index_type;
    using typename broadcast_iterator_base::strides_div_type;
public:
    using iterator_category = std::random_access_iterator_tag;
    using typename broadcast_iterator_base::value_type;
    using typename broadcast_iterator_base::difference_type;
    using typename broadcast_iterator_base::pointer;
    using typename broadcast_iterator_base::reference;
    using typename broadcast_iterator_base::const_reference;
    using broadcast_iterator_base::operator*;

    //rbegin constructor
    template<typename W, std::enable_if_t<std::is_same_v<W,walker_type> ,int> =0 >
    reverse_broadcast_iterator_generic(W&& walker_, const shape_type& shape_, const strides_div_type& strides_, const difference_type& size_):
        broadcast_iterator_base{std::forward<W>(walker_), shape_, strides_, size_}
    {
        broadcast_iterator_base::operator--();
    }
    //rend constructor
    template<typename W, std::enable_if_t<std::is_same_v<W,walker_type> ,int> =0 >
    reverse_broadcast_iterator_generic(W&& walker_, const shape_type& shape_, const strides_div_type& strides_):
        broadcast_iterator_base{std::forward<W>(walker_), shape_, strides_}
    {
        broadcast_iterator_base::operator--();
    }

    auto& operator++(){
        broadcast_iterator_base::operator--();
        return *this;
    }
    auto& operator--(){
        broadcast_iterator_base::operator++();
        return *this;
    }
    auto& operator+=(difference_type n){
        broadcast_iterator_base::operator-=(n);
        return *this;
    }
    auto& operator-=(difference_type n){
        broadcast_iterator_base::operator+=(n);
        return *this;
    }
    reverse_broadcast_iterator_generic operator+(difference_type n)const{
        auto tmp = *this;
        tmp+=n;
        return tmp;
    }
    reverse_broadcast_iterator_generic operator-(difference_type n)const{
        auto tmp = *this;
        tmp-=n;
        return tmp;
    }
    result_type operator[](difference_type n)const{return *(*this+n);}
    bool operator==(const reverse_broadcast_iterator_generic& other)const{return static_cast<const broadcast_iterator_base&>(*this) == static_cast<const broadcast_iterator_base&>(other);}
    bool operator!=(const reverse_broadcast_iterator_generic& other)const{return !(*this == other);}
    inline difference_type friend operator-(const reverse_broadcast_iterator_generic& lhs, const reverse_broadcast_iterator_generic& rhs){
        return static_cast<const broadcast_iterator_base&>(rhs) - static_cast<const broadcast_iterator_base&>(lhs);
    }
};

template<typename IterT>
inline bool operator>(const reverse_broadcast_iterator_generic<IterT>& lhs, const reverse_broadcast_iterator_generic<IterT>& rhs){return (lhs - rhs) > typename reverse_broadcast_iterator_generic<IterT>::difference_type(0);}
template<typename IterT>
inline bool operator<(const reverse_broadcast_iterator_generic<IterT>& lhs, const reverse_broadcast_iterator_generic<IterT>& rhs){return (rhs - lhs) > typename reverse_broadcast_iterator_generic<IterT>::difference_type(0);}
template<typename IterT>
inline bool operator>=(const reverse_broadcast_iterator_generic<IterT>& lhs, const reverse_broadcast_iterator_generic<IterT>& rhs){return !(lhs < rhs);}
template<typename IterT>
inline bool operator<=(const reverse_broadcast_iterator_generic<IterT>& lhs, const reverse_broadcast_iterator_generic<IterT>& rhs){return !(lhs > rhs);}

template<typename CfgT, typename WalkerT> using reverse_broadcast_iterator = reverse_broadcast_iterator_generic<broadcast_iterator<CfgT,WalkerT>>;
template<typename CfgT, typename WalkerT> using reverse_broadcast_shape_iterator = reverse_broadcast_iterator_generic<broadcast_shape_iterator<CfgT,WalkerT>>;

}   //end of namespace gtensor



#endif