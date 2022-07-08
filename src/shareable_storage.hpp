#ifndef STORAGE_HPP_
#define STORAGE_HPP_
#include <memory>
#include <initializer_list>

namespace gtensor{
namespace detail{

template<typename T> struct is_not_bool_vector{
    template<typename> struct selector_:std::true_type{};
    template<typename A> struct selector_<std::vector<bool,A>>:std::false_type{};
    constexpr static bool value = selector_<T>::value;
};

template<typename ImplT>
class shareable_storage{
public:
    using this_type = shareable_storage<ImplT>;
    using storage_impl_type = ImplT;
    using value_type = typename storage_impl_type::value_type;
    using iterator = typename storage_impl_type::iterator;
    using size_type = typename storage_impl_type::size_type;
    using const_iterator = typename storage_impl_type::const_iterator;
    using reverse_iterator = typename storage_impl_type::reverse_iterator;
    using const_reverse_iterator = typename storage_impl_type::const_reverse_iterator;
    static_assert(detail::is_not_bool_vector<storage_impl_type>::value);

    /*copy and move have reference semantic, call copy() to make deep copy*/
    shareable_storage(const shareable_storage& other) = default;
    shareable_storage& operator=(const shareable_storage& other) = default;
    shareable_storage(shareable_storage&& other) = default;
    shareable_storage& operator=(shareable_storage&& other) = default;
    /*construct using storage interface*/
    shareable_storage():
        impl{std::make_shared<storage_impl_type>()}
    {}
    explicit shareable_storage(const size_type& n):
        impl{std::make_shared<storage_impl_type>(n)}
    {}
    shareable_storage(const size_type& n, const value_type& v):
        impl{std::make_shared<storage_impl_type>(n,v)}
    {}
    template<typename It>
    shareable_storage(It first, It last):
        impl{std::make_shared<storage_impl_type>(first,last)}
    {}
    shareable_storage(std::initializer_list<value_type> list):
        impl{std::make_shared<storage_impl_type>(list)}
    {}


    /*construct from storage_impl_type*/    
    explicit shareable_storage(const storage_impl_type& impl_):
        impl{std::make_shared<storage_impl_type>(impl_)}
    {}
    explicit shareable_storage(storage_impl_type&& impl_):
        impl{std::make_shared<storage_impl_type>(std::move(impl_))}
    {}
    
    /*makes deep copy*/
    shareable_storage copy()const{return shareable_storage{*this, 0};}    
    auto use_count()const{return impl.use_count();}

    iterator begin(){return impl->begin();}
    iterator end(){return impl->end();}
    const_iterator begin()const{return impl->begin();}
    const_iterator end()const{return impl->end();}
    reverse_iterator rbegin(){return impl->rbegin();}
    reverse_iterator rend(){return impl->rend();}
    const_reverse_iterator rbegin()const{return impl->rbegin();}
    const_reverse_iterator rend()const{return impl->rend();}
    
    value_type* data(){return impl->data();}
    const value_type* data()const{return impl->data();}
    value_type& back(){return impl->back();}
    const value_type& back()const{return impl->back();}
    size_type size()const{return impl ? impl->size() : 0;}
    bool empty()const{return impl ? impl->empty() : true;}
    void reserve(size_type n)const{impl->reserve(n);}
    
    /*
    * vector of bools
    * The lifetime of a temporary bound to the returned value in a function return statement (9.6.3) is not
    * extended; the temporary is destroyed at the end of the full-expression in the return statement.
    */
    value_type& operator[](size_type i){return impl->operator[](i);}
    const value_type& operator[](size_type i)const{return impl->operator[](i);}
    
    void push_back(const value_type& v){impl->push_back(v);}
    void push_back(value_type&& v){impl->push_back(std::move(v));}
    void swap(const this_type& other){impl->swap(*other.impl);}

    friend bool operator==(const this_type& lhs, const this_type& rhs){return lhs.impl == rhs.impl || *lhs.impl == *rhs.impl;}
    friend bool operator!=(const this_type& lhs, const this_type& rhs){return !(lhs==rhs);}
    friend std::ostream& operator<<(std::ostream& os, const this_type& lhs){return os<<"{("<<lhs.size()<<")"<<[&](){for(const auto& i:lhs) os<<i<<","; return "";}()<<"}";}

private:
    /*deep copy constructor*/
    shareable_storage(const shareable_storage& other, int):
        impl{std::make_shared<storage_impl_type>(*other.impl)}
    {}
    std::shared_ptr<storage_impl_type> impl;
};


}   //end of namespace detail
}   //end of namespace gtensor


#endif