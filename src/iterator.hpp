#ifndef ITERATOR_HPP_
#define ITERATOR_HPP_

#include <iterator>
#include "config.hpp"
#include "libdivide_helper.hpp"
#include "walker_base.hpp"

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
}   //end of namespace detail

/*
* random access trivial-broadcast iterator
* WalkerT is trivial-broadcast walker type, must satisfy trivial-broadcast walker interface
*/
template<typename CfgT, typename WalkerT>
class trivial_broadcast_iterator
{
    using walker_type = WalkerT;
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;
    using result_type = decltype(std::declval<walker_type>()[std::declval<index_type>()]);
public:
    using iterator_category = std::random_access_iterator_tag;
    //using value_type = ValT;
    using value_type = std::decay_t<result_type>;
    using difference_type = typename CfgT::difference_type;
    using pointer = typename detail::iterator_internals_selector<value_type>::pointer;
    using reference = typename detail::iterator_internals_selector<value_type>::reference;
    using const_reference = typename detail::iterator_internals_selector<value_type>::const_reference;
    //begin constructor
    template<typename W, std::enable_if_t<std::is_same_v<W,walker_type> ,int> =0 >
    explicit trivial_broadcast_iterator(W&& walker_):
        walker{std::forward<W>(walker_)},
        flat_index{0}
    {}
    //end constructor
    template<typename W, std::enable_if_t<std::is_same_v<W,walker_type> ,int> =0 >
    trivial_broadcast_iterator(W&& walker_, const difference_type& flat_index_):
        walker{std::forward<W>(walker_)},
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
    auto operator+(difference_type n) const{
        auto it = *this;
        return it.advance(n);
    }
    auto operator-(difference_type n) const{
        auto it = *this;
        return it.advance(-n);
    }
    bool operator==(const trivial_broadcast_iterator& it)const{return flat_index == it.flat_index;}
    bool operator!=(const trivial_broadcast_iterator& it)const{return flat_index != it.flat_index;}
    result_type operator[](difference_type n)const{return *(*this+n);}
    result_type operator*() const{return walker[static_cast<index_type>(flat_index)];}
    inline difference_type friend operator-(const trivial_broadcast_iterator& lhs, const trivial_broadcast_iterator& rhs){return lhs.flat_index - rhs.flat_index;}
private:
    auto& advance(difference_type n){
        flat_index+=n;
        return *this;
    }

    walker_type walker;
    difference_type flat_index;
};

template<typename CfgT, typename WalkerT>
inline bool operator>(const trivial_broadcast_iterator<CfgT,WalkerT>& lhs, const trivial_broadcast_iterator<CfgT,WalkerT>& rhs){return (lhs - rhs) > typename trivial_broadcast_iterator<CfgT,WalkerT>::difference_type(0);}
template<typename CfgT, typename WalkerT>
inline bool operator<(const trivial_broadcast_iterator<CfgT,WalkerT>& lhs, const trivial_broadcast_iterator<CfgT,WalkerT>& rhs){return (rhs - lhs) > typename trivial_broadcast_iterator<CfgT,WalkerT>::difference_type(0);}
template<typename CfgT, typename WalkerT>
inline bool operator>=(const trivial_broadcast_iterator<CfgT,WalkerT>& lhs, const trivial_broadcast_iterator<CfgT,WalkerT>& rhs){return !(lhs < rhs);}
template<typename CfgT, typename WalkerT>
inline bool operator<=(const trivial_broadcast_iterator<CfgT,WalkerT>& lhs, const trivial_broadcast_iterator<CfgT,WalkerT>& rhs){return !(lhs > rhs);}

/*
* bidirectional broadcast iterator
* WalkerT broadcast walker type, must satisfy broadcast walker interface
*/
template<typename CfgT, typename WalkerT>
class broadcast_bidirectional_iterator{
protected:
    using walker_type = WalkerT;
    using result_type = decltype(*std::declval<walker_type>());
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;
public:
    using iterator_category = std::bidirectional_iterator_tag;
    //using value_type = ValT;
    using value_type = std::decay_t<result_type>;
    using difference_type = typename CfgT::difference_type;
    using pointer = typename detail::iterator_internals_selector<value_type>::pointer;
    using reference = typename detail::iterator_internals_selector<value_type>::reference;
    using const_reference = typename detail::iterator_internals_selector<value_type>::const_reference;

    //begin constructor
    template<typename W, std::enable_if_t<std::is_same_v<W,walker_type> ,int> =0 >
    broadcast_bidirectional_iterator(W&& walker_, const shape_type& shape_):
        walker{std::forward<W>(walker_)},
        dim_dec{static_cast<index_type>(shape_.size()-1)},
        shape{shape_},
        flat_index{0}
    {}
    //end constructor
    template<typename W, std::enable_if_t<std::is_same_v<W,walker_type> ,int> =0 >
    broadcast_bidirectional_iterator(W&& walker_, const shape_type& shape_, const difference_type& size_):
        walker{std::forward<W>(walker_)},
        dim_dec{static_cast<index_type>(shape_.size()-1)},
        shape{shape_},
        flat_index{size_}
    {
        ++multi_index.front();
    }
    auto& operator++();
    auto& operator--();
    bool operator==(const broadcast_bidirectional_iterator& other)const{return flat_index == other.flat_index;}
    bool operator!=(const broadcast_bidirectional_iterator& other)const{return flat_index != other.flat_index;}
    result_type operator*() const{return *walker;}
protected:
    walker_type walker;
    index_type dim_dec;
    detail::shape_inverter<index_type,shape_type> shape;
    difference_type flat_index;
    shape_type multi_index = shape_type(dim_dec+2,index_type(1));
};

template<typename CfgT, typename WalkerT>
auto& broadcast_bidirectional_iterator<CfgT,WalkerT>::operator++(){
    index_type d{0};
    auto idx_first = multi_index.begin();
    auto idx_it = std::prev(multi_index.end());
    while(idx_it!=idx_first){
        if (*idx_it == shape.element(d)){
            walker.reset(d);
            *idx_it = index_type(1);
            ++d;
            --idx_it;
        }
        else{
            walker.step(d);
            ++(*idx_it);
            ++flat_index;
            return *this;
        }
    }
    ++flat_index;
    ++(*idx_it);
    //in this place iterator is at the end, next increment leads to UB
    return *this;
}
template<typename CfgT, typename WalkerT>
auto& broadcast_bidirectional_iterator<CfgT,WalkerT>::operator--(){
    index_type d{0};
    auto idx_first = multi_index.begin();
    auto idx_it = std::prev(multi_index.end());
    while(idx_it!=idx_first){
        if (*idx_it==index_type(1)){
            walker.reset_back(d);
            *idx_it = shape.element(d);
            ++d;
            --idx_it;
        }
        else{
            walker.step_back(d);
            --(*idx_it);
            --flat_index;
            return *this;
        }
    }
    --flat_index;
    --(*idx_it);
    //in this place iterator is at the beginning, next decrement leads to UB
    return *this;
}

/*
* random access broadcast iterator
*/
template<typename CfgT, typename WalkerT>
class broadcast_iterator : public broadcast_bidirectional_iterator<CfgT,WalkerT>
{
protected:
    using typename broadcast_bidirectional_iterator::walker_type;
    using typename broadcast_bidirectional_iterator::result_type;
    using typename broadcast_bidirectional_iterator::shape_type;
    using typename broadcast_bidirectional_iterator::index_type;
    using strides_div_type = typename detail::strides_div_traits<CfgT>::type;
public:
    using iterator_category = std::random_access_iterator_tag;
    using typename broadcast_bidirectional_iterator::value_type;
    using typename broadcast_bidirectional_iterator::difference_type;
    using typename broadcast_bidirectional_iterator::pointer;
    using typename broadcast_bidirectional_iterator::reference;
    using typename broadcast_bidirectional_iterator::const_reference;

    //begin constructor
    template<typename W, std::enable_if_t<std::is_same_v<W,walker_type> ,int> =0 >
    broadcast_iterator(W&& walker_, const shape_type& shape_, const strides_div_type& strides_):
        broadcast_bidirectional_iterator{std::forward<W>(walker_), shape_},
        strides{&strides_}
    {}
    //end constructor
    template<typename W, std::enable_if_t<std::is_same_v<W,walker_type> ,int> =0 >
    broadcast_iterator(W&& walker_, const shape_type& shape_, const strides_div_type& strides_, const difference_type& size_):
        broadcast_bidirectional_iterator{std::forward<W>(walker_), shape_, size_},
        strides{&strides_}
    {}
    auto& operator+=(difference_type n){return advance(n);}
    auto& operator-=(difference_type n){return advance(-n);}
    auto operator+(difference_type n) const{
        auto it = *this;
        return it.advance(n);
    }
    auto operator-(difference_type n) const{
        auto it = *this;
        return it.advance(-n);
    }
    result_type operator[](difference_type n)const{return *(*this+n);}
    inline difference_type friend operator-(const broadcast_iterator& lhs, const broadcast_iterator& rhs){return lhs.flat_index - rhs.flat_index;}
private:
    auto& advance(difference_type);
    const strides_div_type* strides;
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
auto& broadcast_iterator<CfgT,WalkerT>::advance(difference_type n){
    index_type idx{flat_index + n};
    flat_index = idx;
    walker.reset();
    auto strides_it{(*strides).begin()};
    auto strides_end{(*strides).end()};
    auto multi_it{multi_index.begin()};
    ++multi_it;
    for(index_type d{dim_dec};strides_it!=strides_end; ++strides_it,++multi_it,--d){
        auto q = detail::divide(idx,*strides_it);
        if (q!=0){
            walker.walk(d,q);
            *multi_it = q+1;
        }
    }
    return *this;
}

template<typename ValT, typename DiffT>
class random_access_iterator
{
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = ValT;
    using difference_type = DiffT;
    using pointer = value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;

    template<typename U, std::enable_if_t<!std::is_convertible_v<std::decay_t<U>, random_access_iterator>,int> =0>
    explicit random_access_iterator(U&& iter__):
        impl_{std::make_unique<engine<std::decay_t<U>>>(std::forward<U>(iter__))}
    {}

    random_access_iterator(const random_access_iterator& other):
        impl_{other.impl_->clone()}
    {}
    random_access_iterator& operator=(const random_access_iterator& other){
        impl_->assign(*other.impl_.get());
        return *this;
    }
    random_access_iterator(random_access_iterator&&) = default;
    random_access_iterator& operator=(random_access_iterator&&) = default;

    auto& operator++(){
        impl_->advance(difference_type{1});
        return *this;
    };
    auto& operator--(){
        impl_.advance(difference_type{-1});
        return *this;
    };
    auto& operator+=(difference_type n){
        impl_.advance(n);
        return *this;
    }
    auto& operator-=(difference_type n){
        return this->operator+=(-n);
    }
    auto operator+(difference_type n) const{
        random_access_iterator it{*this};
        it.impl_.advance(n);
        return it;
    }
    auto operator-(difference_type n) const{
        return this->operator+(-n);
    }
    bool operator==(const random_access_iterator& other)const{return impl_->equal(*other.impl_.get());}
    bool operator!=(const random_access_iterator& other)const{return !(*this == other);}
    difference_type operator-(const random_access_iterator& other){return impl_->diff(*other.impl_.get());}
    value_type operator*() const{return impl_->deref();}
    value_type operator[](difference_type n)const{return *(*this+n);}

private:
    class engine_base
    {
        virtual void advance(difference_type) = 0;
        virtual value_type deref()const = 0;
        virtual difference_type diff(const engine_base& other)const = 0;
        virtual bool equal(const engine_base& other)const = 0;
        virtual void assign(const engine_base& other)const = 0;
        virtual std::unique_ptr<engine_base> clone()const = 0;
    };

    template<typename Iter>
    class engine : public engine_base
    {
        Iter iter_;

        std::unique_ptr<engine_base> clone()const override{return std::make_unique<engine>{*this};}
        void advance(difference_type n) override{std::advance(iter_, n);}
        difference_type diff(const engine_base& other) const override{ return iter_ - static_cast<const engine&>(other).iter_;}
        bool equal(const engine_base& other) const override{return iter_ == static_cast<const engine&>(other).iter_;}
        void assign(const engine_base& other) const override{iter_ = static_cast<const engine&>(other).iter_}
        value_type deref()const override{return *iter_;}

        engine(const engine&) = default;
        engine(engine&&) = default;
        engine& operator=(const engine&) = default;
        engine& operator=(engine&&) = default;
    public:
        template<typename U>
        explicit engine(U&& iter__):
            iter_{std::forward<U>(iter__)}
        {}
    };

    std::unique_ptr<engine_base> impl_;
};

template<typename ValT, typename DiffT>
inline bool operator>(const random_access_iterator<ValT,DiffT>& lhs, const random_access_iterator<ValT,DiffT>& rhs){return (lhs - rhs) > typename random_access_iterator<ValT,DiffT>::difference_type(0);}
template<typename ValT, typename DiffT>
inline bool operator<(const random_access_iterator<ValT,DiffT>& lhs, const random_access_iterator<ValT,DiffT>& rhs){return (rhs - lhs) > typename random_access_iterator<ValT,DiffT>::difference_type(0);}
template<typename ValT, typename DiffT>
inline bool operator>=(const random_access_iterator<ValT,DiffT>& lhs, const random_access_iterator<ValT,DiffT>& rhs){return !(lhs < rhs);}
template<typename ValT, typename DiffT>
inline bool operator<=(const random_access_iterator<ValT,DiffT>& lhs, const random_access_iterator<ValT,DiffT>& rhs){return !(lhs > rhs);}

}   //end of namespace gtensor



#endif