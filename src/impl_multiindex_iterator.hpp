#ifndef IMPL_MULTIINDEX_ITERATOR_HPP_
#define IMPL_MULTIINDEX_ITERATOR_HPP_

#include <iterator>

namespace gtensor{

/*
* multiindex_iterator
*/
template<typename ValT, template<typename> typename Cfg, typename Wkr>
class multiindex_iterator_impl{
    using walker_type = Wkr;
    using config_type = Cfg<ValT>;
    using difference_type = typename config_type::difference_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    

    walker_type walker;
    const shape_type* shape;
    const shape_type* strides;
    difference_type flat_index;
    index_type dim_dec{static_cast<index_type>(shape->size()-1)};
    shape_type multi_index = shape_type(dim_dec+2,index_type(1));

    auto& advance(difference_type);

public:
    using value_type = ValT;
    using difference_type = typename config_type::difference_type;
    using iterator_category = std::random_access_iterator_tag;

    //begin constructor
    template<typename W>
    multiindex_iterator_impl(W&& walker_, const shape_type& shape_, const shape_type& strides_):
        walker{std::forward<W>(walker_)},
        shape{&shape_},
        strides{&strides_},
        flat_index{0}
    {}
    //end constructor
    template<typename W>
    multiindex_iterator_impl(W&& walker_, const shape_type& shape_, const shape_type& strides_, const index_type& size_):
        walker{std::forward<W>(walker_)},
        shape{&shape_},
        strides{&strides_},
        flat_index{size_}
    {
        ++multi_index.front();
    }
    
    bool operator==(const multiindex_iterator_impl& it)const{return flat_index == it.flat_index;}    
    bool operator!=(const multiindex_iterator_impl& it)const{return flat_index != it.flat_index;}

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

    inline difference_type friend operator-(const multiindex_iterator_impl& lhs, const multiindex_iterator_impl& rhs){return lhs.flat_index - rhs.flat_index;}
    inline bool friend operator>(const multiindex_iterator_impl& lhs, const multiindex_iterator_impl& rhs){return (lhs - rhs) > difference_type(0);}
    inline bool friend operator<(const multiindex_iterator_impl& lhs, const multiindex_iterator_impl& rhs){return (rhs - lhs) > difference_type(0);}
    inline bool friend operator>=(const multiindex_iterator_impl& lhs, const multiindex_iterator_impl& rhs){return !(lhs < rhs);}
    inline bool friend operator<=(const multiindex_iterator_impl& lhs, const multiindex_iterator_impl& rhs){return !(lhs > rhs);}
};

template<typename ValT, template<typename> typename Cfg, typename Wkr>
auto& multiindex_iterator_impl<ValT,Cfg,Wkr>::operator++(){
    index_type d{dim_dec};
    auto idx_first = multi_index.begin();
    auto idx_it = std::prev(multi_index.end());
    while(idx_it!=idx_first){
        if (*idx_it == (*shape)[d]){
            walker.reset(d);
            *idx_it = index_type(1);
            --idx_it;
            --d;
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
auto& multiindex_iterator_impl<ValT,Cfg,Wkr>::operator--(){
    index_type d{dim_dec};
    auto idx_first = multi_index.begin();
    auto idx_it = std::prev(multi_index.end());
    while(idx_it!=idx_first){
        if (*idx_it==index_type(1)){
            walker.walk(d,(*shape)[d]-1);
            *idx_it = (*shape)[d];
            --idx_it;
            --d;
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
auto& multiindex_iterator_impl<ValT,Cfg,Wkr>::advance(difference_type n){
    index_type idx{flat_index + n};
    flat_index = idx;
    walker.reset();
    auto sit_begin{(*strides).begin()};
    auto sit_end{(*strides).end()};
    for(index_type d{0};sit_begin!=sit_end; ++sit_begin,++d){
        auto q = idx / *sit_begin;
        idx %= *sit_begin;
        if (q!=0){
            walker.walk(d,q);
            multi_index[d] = q+1;
        }
    }    
    return *this;
}




}   //end of namespace gtensor



#endif