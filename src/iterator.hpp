#ifndef ITERATOR_HPP_
#define ITERATOR_HPP_

#include <iterator>
#include "config.hpp"
#include "libdivide_helper.hpp"
#include "walker_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail

/*
* multiindex_iterator
*/
template<typename ValT, template<typename> typename Cfg, typename Wkr>
class multiindex_iterator{
    using walker_type = Wkr;
    using config_type = Cfg<ValT>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using strides_type = typename detail::libdiv_strides_traits<config_type>::type;

public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = ValT;
    using difference_type = typename config_type::difference_type;
    using pointer = value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;

    walker_type walker;
    index_type dim_dec;
    detail::shape_inverter<ValT,Cfg> shape;
    const strides_type* strides;
    difference_type flat_index;
    
    shape_type multi_index = shape_type(dim_dec+2,index_type(1));

    auto& advance(difference_type);

public:    

    //begin constructor
    template<typename W>
    multiindex_iterator(W&& walker_, const shape_type& shape_, const strides_type& strides_):
        walker{std::forward<W>(walker_)},
        dim_dec{static_cast<index_type>(shape_.size()-1)},
        shape{shape_},
        strides{&strides_},
        flat_index{0}
    {}
    //end constructor
    template<typename W>
    multiindex_iterator(W&& walker_, const shape_type& shape_, const strides_type& strides_, const difference_type& size_):
        walker{std::forward<W>(walker_)},
        dim_dec{static_cast<index_type>(shape_.size()-1)},
        shape{shape_},
        strides{&strides_},
        flat_index{size_}
    {
        ++multi_index.front();
    }
    
    bool operator==(const multiindex_iterator& it)const{return flat_index == it.flat_index;}    
    bool operator!=(const multiindex_iterator& it)const{return flat_index != it.flat_index;}

    auto& operator++();
    auto& operator--();
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
    value_type operator[](difference_type n)const{return *(*this+n);}
    value_type operator*() const{return *walker;}

    inline difference_type friend operator-(const multiindex_iterator& lhs, const multiindex_iterator& rhs){return lhs.flat_index - rhs.flat_index;}
    inline bool friend operator>(const multiindex_iterator& lhs, const multiindex_iterator& rhs){return (lhs - rhs) > difference_type(0);}
    inline bool friend operator<(const multiindex_iterator& lhs, const multiindex_iterator& rhs){return (rhs - lhs) > difference_type(0);}
    inline bool friend operator>=(const multiindex_iterator& lhs, const multiindex_iterator& rhs){return !(lhs < rhs);}
    inline bool friend operator<=(const multiindex_iterator& lhs, const multiindex_iterator& rhs){return !(lhs > rhs);}
};

template<typename ValT, template<typename> typename Cfg, typename Wkr>
auto& multiindex_iterator<ValT,Cfg,Wkr>::operator++(){
    index_type d{0};
    auto idx_first = multi_index.begin();
    auto idx_it = std::prev(multi_index.end());
    while(idx_it!=idx_first){
        if (*idx_it == shape.element(d)){
            walker.reset(d);
            *idx_it = index_type(1);
            ++d;
            --idx_it;
            if (idx_it == idx_first){
                ++flat_index;
                ++(*idx_it);
            }            
        }
        else{
            walker.step(d);
            ++(*idx_it);
            ++flat_index;
            break;
        }
    }           
    return *this;
}

template<typename ValT, template<typename> typename Cfg, typename Wkr>
auto& multiindex_iterator<ValT,Cfg,Wkr>::operator--(){
    index_type d{0};
    auto idx_first = multi_index.begin();
    auto idx_it = std::prev(multi_index.end());
    while(idx_it!=idx_first){
        if (*idx_it==index_type(1)){
            walker.walk(d,shape.element(d)-1);
            *idx_it = shape.element(d);
            ++d;
            --idx_it;
            if (idx_it == idx_first){
                --flat_index;
                --(*idx_it);
            }                           
        }
        else{            
            walker.step_back(d);
            --(*idx_it);
            --flat_index;            
            break;
        }
    }        
    return *this;
}

template<typename ValT, template<typename> typename Cfg, typename Wkr>
auto& multiindex_iterator<ValT,Cfg,Wkr>::advance(difference_type n){
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


}   //end of namespace gtensor



#endif