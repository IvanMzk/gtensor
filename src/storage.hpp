#ifndef STORAGE_HPP_
#define STORAGE_HPP_

#include <type_traits>
#include <memory>
#include <iterator>

namespace gtensor{

template<typename T, typename Alloc = std::allocator<T>>
class storage
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

    virtual ~storage(){deallocate();}
    //default constructor, no allocation take place
    storage(const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{0},
        begin_{}
    {}
    //reallocate if not equal sizes or not equal allocators
    storage& operator=(const storage& other){
        if (this != &other){
            copy_assign(other, typename std::allocator_traits<allocator_type>::propagate_on_container_copy_assignment{});
        }
        return *this;
    }
    //use copy assignment if other's allocator disallow to propagate and allocators not equal, otherwise steal from other and put other in default state
    storage& operator=(storage&& other){
        if (this != &other){
            move_assign(std::move(other),  typename std::allocator_traits<allocator_type>::propagate_on_container_move_assignment{});
        }
        return *this;
    }
    storage(const storage& other):
        allocator_{std::allocator_traits<allocator_type>::select_on_container_copy_construction(other.get_allocator())},
        size_{other.size_},
        begin_{allocate(size_)}
    {
        init(other.begin(),other.end());
    }
    //no reallocation
    storage(storage&& other):
        allocator_{std::move(other.allocator_)},
        size_{other.size_},
        begin_{other.begin_}
    {
        other.size_ = 0;
        other.begin_ = nullptr;
    }
    //construct storage of n elements, no initialization is performed for trivially copyable value_type
    explicit storage(const difference_type& n, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{n},
        begin_{allocate(n)}
    {
        init(value_type{}, typename std::is_trivially_copyable<value_type>::type{});
    }
    //construct storage with n elements initialized to v
    storage(const difference_type& n, const value_type& v, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{n},
        begin_{allocate(n)}
    {
        init(v, std::false_type{});
    }
    //construct storage from iterators range
    template<typename It, std::enable_if_t<detail::is_iterator<It>,int> =0 >
    storage(It first, It last, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{static_cast<const difference_type&>std::distance(first,last)},
        begin_{allocate(size_)}
    {
        init(first,last);
    }
    //construct storage from init list
    storage(std::initializer_list<value_type> init, const allocator_type& alloc = allocator_type()):
        allocator_{alloc},
        size_{static_cast<const difference_type&>(init.size())},
        begin_{allocate(size_)}
    {
        init(init.begin(),init.end());
    }

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
    difference_type size()const{return size_;}
    bool empty()const{return begin()==end();}
    allocator_type get_allocator()const{return allocator_;}

private:
    //no copy assign other's allocator
    void copy_assign(const storage& other, std::false_type){
        auto other_size = other.size();
        if (size()!=other_size){
            auto new_buffer = allocate(other_size);
            deallocate();
            size_ = other_size;
            begin_ = new_buffer;
        }
        std::copy(other.begin(),other.end(),begin_);
    }

    //copy assign other's allocator
    void copy_assign(const storage& other, std::true_type){
        if (allocator_ ==  other.allocator_ || typename std::allocator_traits<allocator_type>::is_always_equal()){
            copy_assign(other, std::false_type{});
        }else{
            auto other_size = other.size();
            auto other_allocator = other.get_allocator();
            auto new_buffer = other_allocator.allocate(other_size);
            deallocate();
            size_ = other_size;
            begin_ = new_buffer;
            allocator_ = other_allocator;
        }
        std::copy(other.begin(),other.end(),begin_);
    }

    //no move assign other's allocator, if allocators not equal copy are made
    void move_assign(storage&& other, std::false_type){
        if (allocator_ ==  other.allocator_ || typename std::allocator_traits<allocator_type>::is_always_equal()){
            deallocate();
            size_ = other.size_;
            begin_ = other.begin_;
            other.size_ = 0;
            other.begin_ = nullptr;
        }else{
            copy_assign(other, std::false_type{});
        }
    }

    //move assign other's allocator
    void move_assign(storage&& other, std::true_type){
        deallocate();
        allocator_ = std::move(other.allocator_);
        size_ = other.size_;
        begin_ = other.begin_;
        other.size_ = 0;
        other.begin_ = nullptr;
    }

    //leave uninitialized
    void init(const value_type& v, std::true_type){
    }
    //initialize to v
    void init(const value_type& v, std::false_type){
        std::uninitialized_fill(begin(),end(),v);
    }
    template<typename It>
    void init(It first, It last){
        std::uninitialized_copy(first,last,begin());
    }

    pointer allocate(const difference_type& n){
        return allocator_.allocate(n);
    }
    void deallocate(){
        if (begin_){
            allocator_.deallocate(begin_,size_);
            size_ = 0;
            begin_ = nullptr;
        }
    }

    allocator_type allocator_;
    difference_type size_;
    pointer begin_;
};

}   //end of namespace gtensor
#endif