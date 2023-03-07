#ifndef INDEX_TYPE_ADAPTER_HPP_
#define INDEX_TYPE_ADAPTER_HPP_

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


template<typename StorT, typename IdxT> class storage_adapter;
template<typename StorT, typename InnerIdx>
class storage_adapter<StorT, integral<InnerIdx>>
{
    using storage_type = StorT;
    using index_type = integral<InnerIdx>;
    using subscription_result_type = decltype(std::declval<storage_type>()[std::declval<InnerIdx>()]);
    using const_subscription_result_type = decltype(std::declval<const storage_type>()[std::declval<InnerIdx>()]);

    StorT impl_;
public:
    using value_type = typename storage_type::value_type;

    storage_adapter(const index_type& n):
        impl_(n.value())
    {}
    storage_adapter(const index_type& n, const value_type& v):
        impl_(n.value(), v)
    {}
    storage_adapter(std::initializer_list<value_type> init_list):
        impl_(init_list)
    {}
    auto resize(const index_type& n){return impl_.resize(n.value());}
    auto begin(){return impl_.begin();}
    auto end(){return impl_.end();}
    auto begin()const{return impl_.begin();}
    auto end()const{return impl_.end();}
    auto rbegin(){return impl_.rbegin();}
    auto rend(){return impl_.rend();}
    auto rbegin()const{return impl_.rbegin();}
    auto rend()const{return impl_.rend();}
    subscription_result_type operator[](const index_type& i)const{return impl_[i.value()];}
    const_subscription_result_type operator[](const index_type& i){return impl_[i.value()];}
};

// template<typename StorT, typename IdxT> class storage_adapter;
// template<typename StorT, typename IdxT>
// class storage_adapter<StorT, integral<IdxT>>
// {
//     using storage_type = StorT;
//     using index_type = integral<IdxT>;
//     using value_type = typename storage_type::value_type;
//     StorT impl_;
// public:
//     storage_adapter(const index_type& n):
//         impl_(n.value())
//     {}
//     storage_adapter(const index_type& n, const value_type& v):
//         impl_(n.value(), v)
//     {}
//     auto resize(const index_type& n){return impl_.resize(n.value());}
//     auto data()const{return impl_.data();}
//     auto data(){return impl_.data();}

//     auto begin(){return impl_.begin();}
//     auto end(){return impl_.end();}
//     auto begin()const{return impl_.begin();}
//     auto end()const{return impl_.end();}
//     auto rbegin(){return impl_.rbegin();}
//     auto rend(){return impl_.rend();}
//     auto rbegin()const{return impl_.rbegin();}
//     auto rend()const{return impl_.rend();}
// };


}   //end of namespace gtensor

#endif