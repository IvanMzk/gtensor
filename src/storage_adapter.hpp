#ifndef STORAGE_ADAPTER_HPP__
#define STORAGE_ADAPTER_HPP__

#include <vector>
#include "integral_type.hpp"

namespace gtensor{

template<typename Impl>
class storage_adapter
{
    using impl_type = Impl;
    using inner_index_type = typename impl_type::difference_type;
    using index_type = integral<inner_index_type>;
    impl_type impl_;
public:
    using value_type = typename impl_type::value_type;
    using difference_type = index_type;
    using size_type = index_type;

    storage_adapter(const index_type& n):
        impl_(n.value())
    {}
    storage_adapter(const index_type& n, const value_type& v):
        impl_(n.value(), v)
    {}
    storage_adapter(std::initializer_list<value_type> init_list):
        impl_(init_list)
    {}

    decltype(std::declval<const impl_type&>()[std::declval<inner_index_type>()]) operator[](index_type i)const{return impl_[static_cast<inner_index_type>(i)];}
    decltype(std::declval<impl_type&>()[std::declval<inner_index_type>()]) operator[](index_type i){return impl_[static_cast<inner_index_type>(i)];}

    auto resize(const index_type& n){return impl_.resize(n.value());}
    auto shrink_to_fit(){impl_.shrink_to_fit();};
};

template<typename ValT> using storage_vector = storage_adapter<std::vector<ValT>>;

}   //end of namespace gtensor

#endif