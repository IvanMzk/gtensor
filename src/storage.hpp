#ifndef STORAGE_HPP_
#define STORAGE_HPP_

#include <type_traits>
#include <memory>
#include <iterator>

namespace gtensor{

namespace detail{

template<typename T, typename... Args>
T* construct_at(T* p, Args&&... args){
    return ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
}

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

template<typename It, typename DstIt, typename Alloc>
auto uninitialized_move(It first, It last, DstIt dfirst, Alloc& alloc){
    auto it=dfirst;
    try
    {
        for (;first!=last; ++first,(void) ++it){
            std::allocator_traits<Alloc>::construct(alloc,std::addressof(*it),std::move(*first));
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
    using size_type = typename std::allocator_traits<Alloc>::size_type;

    allocator_type& allocator_;
    size_type size_;
    pointer ptr_;
    ~row_buffer(){
        if (ptr_){
            allocator_.deallocate(ptr_,size_);
        }
    }

    row_buffer() = default;
    row_buffer(const row_buffer&) = delete;
    row_buffer(row_buffer&&) = delete;
    row_buffer(allocator_type& allocator__, size_type size__, pointer ptr__):
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
        begin_{other.begin_},
        end_{other.end_}
    {
        other.begin_ = nullptr;
        other.end_ = nullptr;
    }

    //construct storage of n elements, no initialization is performed for trivially copyable value_type
    explicit minimal_storage(const size_type& n, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(n,value_type{},false);
    }

    pointer data(){return begin_;}
    const_pointer data()const{return begin_;}
    reference operator[](const size_type& i){return *(begin_+i);}
    const_reference operator[](const size_type& i)const{return *(begin_+i);}

private:

    void init(const size_type& n, const value_type& v, bool init_trivial){
        auto new_buffer = allocate_buffer(n);
        if (!std::is_trivially_copyable_v<value_type> || init_trivial){
            detail::uninitialized_fill(new_buffer.get(),new_buffer.get()+n,v,new_buffer.get_allocator());
        }
        begin_=new_buffer.release();
        end_=begin_+n;
    }

    auto allocate_buffer(const size_type& n, allocator_type& alloc){
        return detail::row_buffer<allocator_type>{alloc,n,alloc.allocate(n)};
    }
    auto allocate_buffer(const size_type& n){
        return allocate_buffer(n,allocator_);
    }

    //destroy and deallocate
    void free(allocator_type& alloc){
        if (begin_){
            detail::destroy(begin_,end_,alloc);
            alloc.deallocate(begin_,end_-begin_);
            begin_ = nullptr;
            end_ = nullptr;
        }
    }
    void free(){
        free(allocator_);
    }

    allocator_type allocator_;
    pointer begin_{nullptr};
    pointer end_{nullptr};
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
        begin_{other.begin_},
        end_{other.end_}
    {
        other.begin_ = nullptr;
        other.end_ = nullptr;
    }
    //construct storage of n elements, no initialization is performed for trivially copyable value_type
    explicit basic_storage(const size_type& n, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(n,value_type{},false);
    }
    //construct storage of n elements initialized to v
    basic_storage(const size_type& n, const value_type& v, const allocator_type& alloc = allocator_type()):
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
        std::swap(begin_,other.begin_);
        std::swap(end_,other.end_);
    }
    size_type size()const{return end_-begin_;}
    bool empty()const{return begin()==end();}
    pointer data(){return begin_;}
    const_pointer data()const{return begin_;}
    iterator begin(){return begin_;}
    iterator end(){return  end_;}
    reverse_iterator rbegin(){return std::make_reverse_iterator(end());}
    reverse_iterator rend(){return  std::make_reverse_iterator(begin());}
    const_iterator begin()const{return begin_;}
    const_iterator end()const{return end_;}
    const_reverse_iterator rbegin()const{return std::make_reverse_iterator(end());}
    const_reverse_iterator rend()const{return  std::make_reverse_iterator(begin());}
    reference operator[](const size_type& i){return *(begin_+i);}
    const_reference operator[](const size_type& i)const{return *(begin_+i);}
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
            begin_ = new_buffer.release();
            end_ = begin_+other_size;
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
            begin_ = new_buffer.release();
            end_ = begin_+other_size;
        }
    }

    //no move assign other's allocator, if allocators not equal copy are made
    void move_assign(basic_storage&& other, std::false_type){
        if (std::allocator_traits<allocator_type>::is_always_equal::value || allocator_ ==  other.allocator_){
            free();
            begin_ = other.begin_;
            end_ = other.end_;
            other.begin_ = nullptr;
            other.end_ = nullptr;
        }else{
            copy_assign(other, std::false_type{});
        }
    }

    //move assign other's allocator
    void move_assign(basic_storage&& other, std::true_type){
        auto old_alloc = std::move(allocator_);
        allocator_ = std::move(other.allocator_);
        free(old_alloc);
        begin_ = other.begin_;
        end_ = other.end_;
        other.begin_ = nullptr;
        other.end_ = nullptr;
    }

    void init(const size_type& n, const value_type& v, bool init_trivial){
        auto new_buffer = allocate_buffer(n);
        if (!std::is_trivially_copyable_v<value_type> || init_trivial){
            detail::uninitialized_fill(new_buffer.get(),new_buffer.get()+n,v,new_buffer.get_allocator());
        }
        begin_=new_buffer.release();
        end_=begin_+n;
    }
    template<typename It>
    void init(It first, It last){
        const auto n = static_cast<const size_type&>(std::distance(first,last));
        auto new_buffer = allocate_buffer(n);
        detail::uninitialized_copy(first,last,new_buffer.get(),new_buffer.get_allocator());
        begin_=new_buffer.release();
        end_=begin_+n;
    }

    auto allocate_buffer(const size_type& n, allocator_type& alloc){
        return detail::row_buffer<allocator_type>{alloc,n,alloc.allocate(n)};
    }
    auto allocate_buffer(const size_type& n){
        return allocate_buffer(n,allocator_);
    }

    //destroy and deallocate
    void free(allocator_type& alloc){
        if (begin_){
            detail::destroy(begin(),end(),alloc);
            alloc.deallocate(begin_,size());
            begin_ = nullptr;
            end_ = nullptr;
        }
    }
    void free(){
        free(allocator_);
    }

    allocator_type allocator_;
    pointer begin_{nullptr};
    pointer end_{nullptr};
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


template<typename T, std::size_t Size=8, typename Alloc = std::allocator<T>>
class stack_prealloc_vector
{
    static_assert(Size!=0,"stack preallocated size should be > 0");
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

    virtual ~stack_prealloc_vector()
    {
        free();
    }
    //default constructor, no allocation take place
    stack_prealloc_vector(const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {}
    //reallocate if not equal sizes or not equal allocators
    stack_prealloc_vector& operator=(const stack_prealloc_vector& other){
        if (this != &other){
            if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment::value){
                copy_assign(other.begin(),other.end(),other.get_allocator());
            }else{
                copy_assign(other.begin(),other.end());
            }
        }
        return *this;
    }
    //use copy assignment if other's allocator disallow to propagate and allocators not equal, otherwise steal from other and put other in default state
    stack_prealloc_vector& operator=(stack_prealloc_vector&& other){
        if (this != &other){
            move_assign(std::move(other),  typename std::allocator_traits<allocator_type>::propagate_on_container_move_assignment{});
        }
        return *this;
    }
    stack_prealloc_vector(const stack_prealloc_vector& other):
        allocator_{std::allocator_traits<allocator_type>::select_on_container_copy_construction(other.get_allocator())}
    {
        init(other.begin(),other.end());
    }
    //move construct elements from other or steal
    stack_prealloc_vector(stack_prealloc_vector&& other):
        allocator_{std::move(other.allocator_)}
    {
        init(std::move(other));
    }
    //construct n elements, no initialization is performed for trivially copyable value_type
    explicit stack_prealloc_vector(const size_type& n, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(n,value_type{},false);
    }
    //construct n elements initialized to v
    stack_prealloc_vector(const size_type& n, const value_type& v, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(n,v,true);
    }

    template<typename, typename = void> struct is_input_iterator : std::false_type{};
    template<typename U> struct is_input_iterator<U,std::void_t<typename std::iterator_traits<U>::iterator_category>> : std::is_convertible<typename std::iterator_traits<U>::iterator_category*,std::input_iterator_tag*>{};

    //copy construct from iterators range
    template<typename It, std::enable_if_t<is_input_iterator<It>::value,int> =0 >
    stack_prealloc_vector(It first, It last, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(first,last);
    }
    //copy construct from init list
    stack_prealloc_vector(std::initializer_list<value_type> init_list, const allocator_type& alloc = allocator_type()):
        allocator_{alloc}
    {
        init(init_list.begin(),init_list.end());
    }

    void swap(stack_prealloc_vector& other){
        using std::swap;
        // if constexpr (std::allocator_traits<allocator_type>::propagate_on_container_swap::value){
        //     swap(allocator_,other.allocator_);
        // }
        if (std::allocator_traits<allocator_type>::is_always_equal::value || allocator_ ==  other.allocator_){
            if (!is_on_stack() && !other.is_on_stack()){    //UB if allocators not swapped and not equal
                swap(begin_,other.begin_);
                swap(end_,other.end_);
                swap(capacity_end_,other.capacity_end_);
            }else{  //UB if allocators swapped and not equal
                const auto size_ = size();
                const auto other_size = other.size();
                if (size_==other_size){
                    std::swap_ranges(begin(),end(),other.begin());
                }else if (capacity() < other_size){
                    swap_realloc_(*this,other);
                }else if (other.capacity() < size_){
                    swap_realloc_(other,*this);
                }else if (size_<other_size){
                    swap_copy_(*this,other);
                }else{
                    swap_copy_(other,*this);
                }
            }
        }else{
            //throw or can use move or copy assign if propagate_on_container_swap same as propagate_on_container_move_assignment or propagate_on_container_copy_assignment,
            //which may degrade performance
            throw std::runtime_error("swap on stack_prealloc_vector requires equal allocators");
        }
    }

    void reserve(const size_type& cap){
        if (cap>capacity()){
            reallocate(begin_,end_,size(),cap);
        }
    }

    template<typename...Args>
    reference emplace_back(Args&&...args){
        if (end_==capacity_end_){//size()==capacity()
            const auto size_ = size();
            const auto cap = size_==0 ? 1 : 2*size_;
            reallocate(begin_,end_,size_,cap);
        }
        emplace_at_end(std::forward<Args>(args)...);
        return *end_++;
    }

    void push_back(const value_type& v){
        emplace_back(v);
    }

    void push_back(value_type&& v){
        emplace_back(std::move(v));
    }

    size_type size()const{return end_-begin_;}
    size_type capacity()const{return capacity_end_-begin_;}
    bool empty()const{return begin()==end();}
    pointer data(){return begin_;}
    const_pointer data()const{return begin_;}
    iterator begin(){return begin_;}
    iterator end(){return  end_;}
    reverse_iterator rbegin(){return std::make_reverse_iterator(end());}
    reverse_iterator rend(){return  std::make_reverse_iterator(begin());}
    const_iterator begin()const{return begin_;}
    const_iterator end()const{return end_;}
    const_reverse_iterator rbegin()const{return std::make_reverse_iterator(end());}
    const_reverse_iterator rend()const{return  std::make_reverse_iterator(begin());}
    reference operator[](const size_type& i){return *(begin_+i);}
    const_reference operator[](const size_type& i)const{return *(begin_+i);}
    allocator_type get_allocator()const{return allocator_;}

private:

    const_pointer buffer_array_begin()const{
        return std::launder(reinterpret_cast<const value_type*>(buffer_));
    }
    pointer buffer_array_begin(){
        return std::launder(reinterpret_cast<value_type*>(buffer_));
    }
    pointer buffer_array_end(){
        return std::launder(reinterpret_cast<value_type*>(buffer_+Size*sizeof(value_type)));
    }

    bool is_on_stack()const{
        return begin_== buffer_array_begin();
    }

    template<typename...Args>
    void emplace_at_end(Args&&...args){
        if (is_on_stack()){
            detail::construct_at(end_,std::forward<Args>(args)...);
        }else{
            std::allocator_traits<allocator_type>::construct(allocator_,end_,std::forward<Args>(args)...);
        }
    }

    template<typename It, typename DstIt>
    void construct(It first, It last, DstIt dfirst, stack_prealloc_vector& dv){
        if (dv.is_on_stack()){
            std::uninitialized_copy(first,last,dfirst);
        }else{
            detail::uninitialized_copy(first,last,dfirst,dv.allocator_);
        }
    }
    template<typename It, typename DstIt>
    void construct(It first, It last, DstIt dfirst){
        construct(first,last,dfirst,*this);
    }

    template<typename It>
    void construct_fill(It first, It last, const value_type& v){
        if (is_on_stack()){
            std::uninitialized_fill(first,last,v);
        }else{
            detail::uninitialized_fill(first,last,v,allocator_);
        }
    }

    template<typename It>
    void destroy(It first, It last, stack_prealloc_vector& v){
        if (v.is_on_stack()){
            std::destroy(first,last);
        }else{
            detail::destroy(first,last,v.allocator_);
        }
    }
    template<typename It>
    void destroy(It first, It last){
        destroy(first,last,*this);
    }

    //allocate new storage and copy construct elements from range first,last to it, free old storage, set new allocator if any
    //new capacity must be >= last-first, n must be equal to last-first
    template<typename NewAlloc=std::true_type, typename It, typename A>
    void reallocate(It first, It last, const size_type& n, const size_type& cap, A&& alloc){
        auto new_buffer = allocate_buffer(cap,alloc);
        detail::uninitialized_copy(first,last,new_buffer.get(),new_buffer.get_allocator());
        if constexpr (NewAlloc::value){
            auto old_alloc = std::move(allocator_);
            allocator_ = std::move(alloc);
            free(old_alloc);
        }else{
            free();
        }
        begin_ = new_buffer.release();
        end_ = begin_+n;
        capacity_end_ = begin_+cap;
    }
    template<typename It>
    void reallocate(It first, It last, const size_type& n, const size_type& cap){
        reallocate<std::false_type>(first,last,n,cap,allocator_);
    }


    void init(){
        begin_= buffer_array_begin();
        end_= buffer_array_begin();
        capacity_end_= buffer_array_end();
    }

    void steal(stack_prealloc_vector&& other){
        begin_ = other.begin_;
        end_ = other.end_;
        capacity_end_ = other.capacity_end_;
        other.init();
    }

    void init(const size_type& n, const value_type& v, bool init_trivial){
        init_trivial = init_trivial || !std::is_trivially_copyable_v<value_type>;
        if (Size<n){//exceeds stack preallocated size
            auto new_buffer = allocate_buffer(n);
            if (init_trivial){
                detail::uninitialized_fill(new_buffer.get(),new_buffer.get()+n,v,new_buffer.get_allocator());
            }
            begin_=new_buffer.release();
            capacity_end_=begin_+n;
        }else{
            if (init_trivial){
                std::uninitialized_fill(begin_,begin_+n,v);
            }
        }
        end_=begin_+n;
    }

    template<typename It>
    void init(It first, It last){
        const auto n = static_cast<const size_type&>(std::distance(first,last));
        if (Size<n){//exceeds stack preallocated size
            auto new_buffer = allocate_buffer(n);
            detail::uninitialized_copy(first,last,new_buffer.get(),new_buffer.get_allocator());
            begin_=new_buffer.release();
            capacity_end_=begin_+n;
        }else{  //construct in stack
            std::uninitialized_copy(first,last,begin_);
        }
        end_=begin_+n;
    }

    void init(stack_prealloc_vector&& other){
        if (other.is_on_stack()){   //move elements
            init(std::make_move_iterator(other.begin()),std::make_move_iterator(other.end()));
            other.free();
            other.init();
        }else{  //steal
            steal(std::move(other));
        }
    }

    //swap in case not enough capacity
    //allocators must be equal
    //smaller.capacity()<bigger.size() must hold
    void swap_realloc_(stack_prealloc_vector& smaller, stack_prealloc_vector& bigger){
        auto new_buffer = allocate_buffer(bigger.size(),smaller.allocator_);
        detail::uninitialized_copy(bigger.begin(),bigger.end(),new_buffer.get(),new_buffer.get_allocator());
        std::copy(smaller.begin(),smaller.end(),bigger.begin());
        const auto smaller_size = smaller.size();
        destroy(bigger.begin()+smaller_size,bigger.end(),bigger);
        smaller.free();
        smaller.begin_ = new_buffer.release();
        smaller.end_ = smaller.begin_+bigger.size();
        smaller.capacity_end_ = smaller.end_;
        bigger.end_ = bigger.begin_+smaller_size;
    }

    //swap in case enough capacity
    //allocators must be equal
    //smaller.size()<bigger.size() must hold
    void swap_copy_(stack_prealloc_vector& smaller, stack_prealloc_vector& bigger){
        std::swap_ranges(smaller.begin(),smaller.end(),bigger.begin());
        const auto smaller_size = smaller.size();
        construct(bigger.begin()+smaller_size,bigger.end(),smaller.end(),smaller);
        destroy(bigger.begin()+smaller_size,bigger.end(),bigger);
        smaller.end_=smaller.begin_+bigger.size();
        bigger.end_=bigger.begin_+smaller_size;
    }

    void assign_(const size_type& n, const value_type& v, bool init_trivial){
        init_trivial = init_trivial || !std::is_trivially_copyable_v<value_type>;
        if (capacity()<n){  //reallocate
            auto new_buffer = allocate_buffer(n);
            if (init_trivial){
                detail::uninitialized_fill(new_buffer.get(),new_buffer.get()+n,v,new_buffer.get_allocator());
            }
            free();
            begin_=new_buffer.release();
            capacity_end_=begin_+n;
        }else{
            if (init_trivial){
                const auto size_ = size();
                if (size_<n){
                    std::fill(begin_,begin_+size_,v);
                    construct_fill(begin_+size_,begin_+n,v);
                }else{
                    std::fill(begin_,begin_+n,v);
                    destroy(begin_+n,end_);
                }
            }
        }
        end_=begin_+n;
    }

    template<typename It>
    void assign_(It first, It last, const size_type& n){
        const auto size_ = size();
        if (size_<n){
            std::copy(first,first+size_,begin_);
            construct(first+size_,last,end_);
        }else{
            std::copy(first,last,begin_);
            destroy(begin_+n,end_);
        }
        end_ = begin_+n;
    }

    //no copy assign other's allocator
    template<typename It>
    void copy_assign(It other_first, It other_last){
        const auto other_size = other_last-other_first;
        if (capacity()<other_size){
            reallocate(other_first,other_last,other_size,other_size);
        }else{
            assign_(other_first,other_last,other_size);
        }
    }

    //copy assign other's allocator
    template<typename It>
    void copy_assign(It other_first, It other_last, allocator_type other_alloc){
        if (std::allocator_traits<allocator_type>::is_always_equal::value || allocator_ ==  other_alloc){
            copy_assign(other_first,other_last);
        }else{
            const auto other_size = other_last-other_first;
            if(is_on_stack() && capacity()>=other_size){    //stay on stack
                assign_(other_first,other_last,other_size);
                allocator_ = std::move(other_alloc);
            }else{  //reallocate, new allocator
                reallocate(other_first,other_last,other_size,other_size,other_alloc);
            }
        }
    }

    //no move assign other's allocator, if allocators not equal copy are made
    void move_assign(stack_prealloc_vector&& other, std::false_type){
        if (std::allocator_traits<allocator_type>::is_always_equal::value || allocator_ ==  other.allocator_){
            if (other.is_on_stack()){
                copy_assign(std::make_move_iterator(other.begin()),std::make_move_iterator(other.end()));
            }else{  //free and steal
                free();
                steal(std::move(other));
                return;
            }
        }else{
            copy_assign(std::make_move_iterator(other.begin()),std::make_move_iterator(other.end()));
        }
        other.free();
        other.init();
    }

    //move assign other's allocator
    void move_assign(stack_prealloc_vector&& other, std::true_type){
        if (std::allocator_traits<allocator_type>::is_always_equal::value || allocator_ ==  other.allocator_){
            move_assign(std::move(other),std::false_type{});
        }else{
            if (other.is_on_stack()){   //cant steal, need move by elements
                const auto other_size = other.size();
                if (is_on_stack() && capacity()>=other_size){
                    assign_(std::make_move_iterator(other.begin()),std::make_move_iterator(other.end()),other_size);
                    allocator_ = other.get_allocator();
                }else{
                    reallocate(std::make_move_iterator(other.begin()),std::make_move_iterator(other.end()),other_size,other_size,other.get_allocator());
                }
                other.free();
                other.init();
            }else{  //move alloc, free and steal
                auto old_alloc = std::move(allocator_);
                allocator_ = other.get_allocator();
                free(old_alloc);
                steal(std::move(other));
            }
        }
    }

    template<typename A>
    auto allocate_buffer(const size_type& n, A&& alloc){
        using A_ = std::remove_cv_t<std::remove_reference_t<A>>;
        return detail::row_buffer<A_>{alloc,n,alloc.allocate(n)};
    }
    auto allocate_buffer(const size_type& n){
        return allocate_buffer(n,allocator_);
    }

    //destroy and deallocate
    void free(allocator_type& alloc){
        destroy(begin_,end_);
        if (!is_on_stack()){
            alloc.deallocate(begin_,size());
        }
    }
    void free(){
        free(allocator_);
    }

    std::byte buffer_[Size*sizeof(value_type)];
    allocator_type allocator_;
    pointer begin_{buffer_array_begin()};
    pointer end_{buffer_array_begin()};
    pointer capacity_end_{buffer_array_end()};
};

template<typename T, std::size_t Size, typename Alloc>
void swap(stack_prealloc_vector<T,Size,Alloc>& lhs, stack_prealloc_vector<T,Size,Alloc>& rhs){
    lhs.swap(rhs);
}

template<typename T, std::size_t Size, typename Alloc>
bool operator==(const stack_prealloc_vector<T,Size,Alloc>& lhs, const stack_prealloc_vector<T,Size,Alloc>& rhs){
    if (&lhs == &rhs){
        return true;
    }else if (lhs.size()==rhs.size()){
        return std::equal(lhs.begin(),lhs.end(),rhs.begin());
    }else{
        return false;
    }
}

template<typename T, std::size_t Size, typename Alloc>
bool operator!=(const stack_prealloc_vector<T,Size,Alloc>& lhs, const stack_prealloc_vector<T,Size,Alloc>& rhs){
    return !(lhs==rhs);
}


}   //end of namespace gtensor
#endif