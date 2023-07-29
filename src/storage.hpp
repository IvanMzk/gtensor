#ifndef STORAGE_HPP_
#define STORAGE_HPP_

#include <type_traits>
#include <memory>
#include <iterator>

namespace gtensor{

namespace detail{

template<typename It, typename Alloc>
void destroy(It first, It last, Alloc& alloc){
    using value_type = typename std::iterator_traits<It>::value_type;
    if constexpr (!std::is_trivially_copyable_v<value_type>){
        for (;first!=last; ++first){
            std::allocator_traits<Alloc>::destroy(alloc,std::addressof(*first));
        }
    }
}

template<typename It, typename T, typename Alloc>
void uninitialized_fill(It first, It last, const T& v, Alloc& alloc){
    auto it=first;
    try
    {
        for (;it!=last; ++it){
            std::allocator_traits<Alloc>::construct(alloc,std::addressof(*it),v);
        }
    }
    catch (...)
    {
        destroy(first,it,alloc);
        throw;
    }
}

template<typename It, typename DstIt, typename Alloc>
auto uninitialized_copy(It first, It last, DstIt dfirst, Alloc& alloc){
    auto it=dfirst;
    try
    {
        for (;first!=last; ++first,(void) ++it){
            std::allocator_traits<Alloc>::construct(alloc,std::addressof(*it),*first);
        }
        return it;
    }
    catch (...)
    {
        destroy(dfirst,it,alloc);
        throw;
    }
}

template<typename Alloc>
struct row_buffer
{
    using allocator_type = Alloc;
    using pointer = typename std::allocator_traits<Alloc>::pointer;
    using difference_type = typename std::allocator_traits<Alloc>::difference_type;

    allocator_type& allocator_;
    difference_type size_;
    pointer ptr_;
    ~row_buffer(){
        if (ptr_){
            allocator_.deallocate(ptr_,size_);
        }
    }

    row_buffer() = default;
    row_buffer(const row_buffer&) = delete;
    row_buffer(row_buffer&&) = delete;
    row_buffer(allocator_type& allocator__, difference_type size__, pointer ptr__):
        allocator_{allocator__},
        size_{size__},
        ptr_{ptr__}
    {}

    allocator_type& get_allocator(){
        return allocator_;
    }

    pointer get()const{
        return ptr_;
    }


    pointer release(){
        auto res = ptr_;
        ptr_ = nullptr;
        return res;
    }
};

}

template<typename T, typename Alloc = std::allocator<T>>
class minimal_storage
{
public:
    using allocator_type = Alloc;
    using value_type = T;
    using pointer = typename std::allocator_traits<Alloc>::pointer;
    using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
    using reference = T&;
    using const_reference = const T&;
    using difference_type = typename std::allocator_traits<Alloc>::difference_type;
    using size_type = typename std::allocator_traits<Alloc>::size_type;

    virtual ~minimal_storage()
    {
        free();
    }

    minimal_storage(const minimal_storage&) = delete;
    minimal_storage& operator=(const minimal_storage&) = delete;
    minimal_storage& operator=(minimal_storage&&) = delete;

    minimal_storage(minimal_storage&& other):
        allocator_{std::move(other.allocator_)},
        size_{other.size_},
        begin_{other.begin_}
    {
        other.size_ = 0;
        other.begin_ = nullptr;
    }

    //construct storage of n elements, no initialization is performed for trivially copyable value_type
    explicit minimal_storage(const difference_type& n, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(n,value_type{},false);
    }

    pointer data(){return begin_;}
    const_pointer data()const{return begin_;}
    reference operator[](const difference_type& i){return *(begin_+i);}
    const_reference operator[](const difference_type& i)const{return *(begin_+i);}

private:

    void init(const difference_type& n, const value_type& v, bool init_trivial){
        auto new_buffer = allocate_buffer(n);
        if (!std::is_trivially_copyable_v<value_type> || init_trivial){
            detail::uninitialized_fill(new_buffer.get(),new_buffer.get()+n,v,new_buffer.get_allocator());
        }
        size_=n;
        begin_=new_buffer.release();
    }

    auto allocate_buffer(const difference_type& n, allocator_type& alloc){
        return detail::row_buffer<allocator_type>{alloc,n,alloc.allocate(n)};
    }
    auto allocate_buffer(const difference_type& n){
        return allocate_buffer(n,allocator_);
    }

    //destroy and deallocate
    void free(allocator_type& alloc){
        if (begin_){
            detail::destroy(begin_,begin_+size_,alloc);
            alloc.deallocate(begin_,size_);
            size_ = 0;
            begin_ = nullptr;
        }
    }
    void free(){
        free(allocator_);
    }

    allocator_type allocator_;
    difference_type size_{0};
    pointer begin_{nullptr};
};

template<typename T, typename Alloc = std::allocator<T>>
class basic_storage
{
public:
    using allocator_type = Alloc;
    using value_type = T;
    using pointer = typename std::allocator_traits<Alloc>::pointer;
    using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using difference_type = typename std::allocator_traits<Alloc>::difference_type;
    using size_type = typename std::allocator_traits<Alloc>::size_type;

    virtual ~basic_storage()
    {
        free();
    }
    //default constructor, no allocation take place
    basic_storage(const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {}
    //reallocate if not equal sizes or not equal allocators
    basic_storage& operator=(const basic_storage& other){
        if (this != &other){
            copy_assign(other, typename std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment{});
        }
        return *this;
    }
    //use copy assignment if other's allocator disallow to propagate and allocators not equal, otherwise steal from other and put other in default state
    basic_storage& operator=(basic_storage&& other){
        if (this != &other){
            move_assign(std::move(other),  typename std::allocator_traits<allocator_type>::propagate_on_container_move_assignment{});
        }
        return *this;
    }
    basic_storage(const basic_storage& other):
        allocator_{std::allocator_traits<allocator_type>::select_on_container_copy_construction(other.get_allocator())}
    {
        init(other.begin(),other.end());
    }
    //no reallocation
    basic_storage(basic_storage&& other):
        allocator_{std::move(other.allocator_)},
        size_{other.size_},
        begin_{other.begin_}
    {
        other.size_ = 0;
        other.begin_ = nullptr;
    }
    //construct storage of n elements, no initialization is performed for trivially copyable value_type
    explicit basic_storage(const difference_type& n, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(n,value_type{},false);
    }
    //construct storage of n elements initialized to v
    basic_storage(const difference_type& n, const value_type& v, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(n,v,true);
    }

    template<typename, typename = void> struct is_input_iterator : std::false_type{};
    template<typename U> struct is_input_iterator<U,std::void_t<typename std::iterator_traits<U>::iterator_category>> : std::is_convertible<typename std::iterator_traits<U>::iterator_category*,std::input_iterator_tag*>{};

    //construct storage from iterators range
    template<typename It, std::enable_if_t<is_input_iterator<It>::value,int> =0 >
    basic_storage(It first, It last, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(first,last);
    }
    //construct basic_storage from init list
    basic_storage(std::initializer_list<value_type> init_list, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(init_list.begin(),init_list.end());
    }

    void swap(basic_storage& other){
        if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_swap::value){
            std::swap(allocator_,other.allocator_);
        }
        std::swap(size_,other.size_);
        std::swap(begin_,other.begin_);
    }
    difference_type size()const{return size_;}
    bool empty()const{return begin()==end();}
    pointer data(){return begin_;}
    const_pointer data()const{return  begin_;}
    iterator begin(){return begin_;}
    iterator end(){return  begin_+size_;}
    reverse_iterator rbegin(){return std::make_reverse_iterator(end());}
    reverse_iterator rend(){return  std::make_reverse_iterator(begin());}
    const_iterator begin()const{return begin_;}
    const_iterator end()const{return  begin_+size_;}
    const_reverse_iterator rbegin()const{return std::make_reverse_iterator(end());}
    const_reverse_iterator rend()const{return  std::make_reverse_iterator(begin());}
    reference operator[](const difference_type& i){return *(begin_+i);}
    const_reference operator[](const difference_type& i)const{return *(begin_+i);}
    allocator_type get_allocator()const{return allocator_;}

private:
    //no copy assign other's allocator
    void copy_assign(const basic_storage& other, std::false_type){
        const auto other_size = other.size();
        //reallocate even if size()>other.size()
        if (size()!=other_size){
            auto new_buffer = allocate_buffer(other_size);
            detail::uninitialized_copy(other.begin(),other.end(),new_buffer.get(),new_buffer.get_allocator());
            free();
            size_ = other_size;
            begin_ = new_buffer.release();
        }else{
            std::copy(other.begin(),other.end(),begin_);
        }
    }

    //copy assign other's allocator
    void copy_assign(const basic_storage& other, std::true_type){
        if (std::allocator_traits<allocator_type>::is_always_equal::value || allocator_ ==  other.allocator_){
            copy_assign(other, std::false_type{});
        }else{
            auto other_size = other.size();
            auto other_allocator = other.get_allocator();
            auto new_buffer = allocate_buffer(other_size,other_allocator);
            detail::uninitialized_copy(other.begin(),other.end(),new_buffer.get(),new_buffer.get_allocator());
            auto old_alloc = std::move(allocator_);
            allocator_ = std::move(other_allocator);
            free(old_alloc);
            size_ = other_size;
            begin_ = new_buffer.release();
        }
    }

    //no move assign other's allocator, if allocators not equal copy are made
    void move_assign(basic_storage&& other, std::false_type){
        if (std::allocator_traits<allocator_type>::is_always_equal::value || allocator_ ==  other.allocator_){
            free();
            size_ = other.size_;
            begin_ = other.begin_;
            other.size_ = 0;
            other.begin_ = nullptr;
        }else{
            copy_assign(other, std::false_type{});
        }
    }

    //move assign other's allocator
    void move_assign(basic_storage&& other, std::true_type){
        auto old_alloc = std::move(allocator_);
        allocator_ = std::move(other.allocator_);
        free(old_alloc);
        size_ = other.size_;
        begin_ = other.begin_;
        other.size_ = 0;
        other.begin_ = nullptr;
    }

    void init(const difference_type& n, const value_type& v, bool init_trivial){
        auto new_buffer = allocate_buffer(n);
        if (!std::is_trivially_copyable_v<value_type> || init_trivial){
            detail::uninitialized_fill(new_buffer.get(),new_buffer.get()+n,v,new_buffer.get_allocator());
        }
        size_=n;
        begin_=new_buffer.release();
    }
    template<typename It>
    void init(It first, It last){
        const auto n = static_cast<const difference_type&>(std::distance(first,last));
        auto new_buffer = allocate_buffer(n);
        detail::uninitialized_copy(first,last,new_buffer.get(),new_buffer.get_allocator());
        size_ = n;
        begin_=new_buffer.release();
    }

    auto allocate_buffer(const difference_type& n, allocator_type& alloc){
        return detail::row_buffer<allocator_type>{alloc,n,alloc.allocate(n)};
    }
    auto allocate_buffer(const difference_type& n){
        return allocate_buffer(n,allocator_);
    }

    //destroy and deallocate
    void free(allocator_type& alloc){
        if (begin_){
            detail::destroy(begin(),end(),alloc);
            alloc.deallocate(begin_,size_);
            size_ = 0;
            begin_ = nullptr;
        }
    }
    void free(){
        free(allocator_);
    }

    allocator_type allocator_;
    difference_type size_{0};
    pointer begin_{nullptr};
};


template<typename T, typename Alloc>
void swap(basic_storage<T,Alloc>& lhs, basic_storage<T,Alloc>& rhs){
    lhs.swap(rhs);
}

template<typename T, typename Alloc>
bool operator==(const basic_storage<T,Alloc>& lhs, const basic_storage<T,Alloc>& rhs){
    if (&lhs == &rhs){
        return true;
    }else if (lhs.size()==rhs.size()){
        return std::equal(lhs.begin(),lhs.end(),rhs.begin());
    }else{
        return false;
    }
}

template<typename T, typename Alloc>
bool operator!=(const basic_storage<T,Alloc>& lhs, const basic_storage<T,Alloc>& rhs){
    return !(lhs==rhs);
}

}   //end of namespace gtensor
#endif