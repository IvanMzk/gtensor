#ifndef INDEXABLE_ADAPTER_HPP_
#define INDEXABLE_ADAPTER_HPP_

#include "integral_type.hpp"

namespace gtensor{

template<typename T, typename IdxT>
class indexable_common
{
protected:
    using wrapped_type = T;
    using index_type = IdxT;
    using result_type = decltype(std::declval<wrapped_type>()[std::declval<index_type>()]);
    wrapped_type impl_;
public:
    template<typename U>
    indexable_common(U&& impl__):
        impl_{std::forward<U>(impl__)}
    {}
};
template<typename T, typename IdxT>
class indexable : private indexable_common<T,IdxT>
{
    using indexable_common_base = indexable_common<T,IdxT>;
    using typename indexable_common_base::wrapped_type;
    using typename indexable_common_base::index_type;
    using typename indexable_common_base::result_type;
public:
    using indexable_common_base::indexable_common_base;
    result_type operator[](const index_type& idx)const{return this->impl_[idx];}
};
template<typename T, typename InnerIdx>
class indexable<T,integral<InnerIdx>> : private indexable_common<T,InnerIdx>
{
    using indexable_common_base = indexable_common<T,InnerIdx>;
    using typename indexable_common_base::wrapped_type;
    using index_type = integral<InnerIdx>;
    using typename indexable_common_base::result_type;
public:
    using indexable_common_base::indexable_common_base;
    result_type operator[](const index_type& idx)const{return this->impl_[idx.value()];}
};

template<typename T, typename IdxT>
class indexable_ref_common
{
protected:
    using wrapped_type = T;
    using index_type = IdxT;
    using result_type = decltype(std::declval<wrapped_type>()[std::declval<index_type>()]);
    wrapped_type* impl_;
public:
    indexable_ref_common(wrapped_type& impl__):
        impl_{&impl__}
    {}
};
template<typename T, typename IdxT>
class indexable_ref : private indexable_ref_common<T, IdxT>
{
    using indexable_ref_common_base = indexable_ref_common<T, IdxT>;
    using typename indexable_ref_common_base::wrapped_type;
    using typename indexable_ref_common_base::index_type;
    using typename indexable_ref_common_base::result_type;
public:
    using indexable_ref_common_base::indexable_ref_common_base;
    result_type operator[](const index_type& idx)const{return (*this->impl_)[idx];}
};
template<typename T, typename InnerIdx>
class indexable_ref<T, integral<InnerIdx>> : private indexable_ref_common<T, InnerIdx>
{
    using indexable_ref_common_base = indexable_ref_common<T, InnerIdx>;
    using typename indexable_ref_common_base::wrapped_type;
    using index_type = integral<InnerIdx>;
    using typename indexable_ref_common_base::result_type;
public:
    using indexable_ref_common_base::indexable_ref_common_base;
    result_type operator[](const index_type& idx)const{return (*this->impl_)[idx.value()];}
};


template<typename IdxT, typename T>
auto make_indexable(T&& t){
    return indexable<std::remove_reference_t<T>,IdxT>{std::forward<T>(t)};
}
template<typename IdxT, typename T>
auto make_indexable_ref(T&& t){
    return indexable_ref<std::remove_reference_t<T>,IdxT>{std::forward<T>(t)};
}


}   //end of namespace gtensor

#endif