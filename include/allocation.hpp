#ifndef ALLOCATION_HPP_
#define ALLOCATION_HPP_

#include <new>

namespace allocation{

#ifdef __cpp_aligned_new
#define HAS_ALIGNED_NEW_ 1
#else
#define HAS_ALIGNED_NEW_ 0
#endif

template<typename T, std::size_t Alignment>
class aligned_allocator
{
    static_assert(HAS_ALIGNED_NEW_);
    static_assert(!std::is_const_v<T>);
    static_assert(Alignment != 0);
    static_assert((Alignment&(Alignment-1))  == 0);
    static_assert(Alignment >= alignof(T));
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;
    using propagate_on_container_swap = std::false_type;
    using propagate_on_container_copy_assignment = std::false_type;

    aligned_allocator() noexcept
    {}
    aligned_allocator(const aligned_allocator&) noexcept
    {}
    template<typename U>
    aligned_allocator(const aligned_allocator<U>&) noexcept
    {}

    T* allocate(const std::size_t n){
        if constexpr (sizeof(T)>1){
            if (n > std::numeric_limits<std::size_t>::max()/sizeof(T)){
                throw std::bad_array_new_length{};
            }
        }
        if (n==0){
            return nullptr;
        }
        return static_cast<T*>(::operator new(n*sizeof(T),std::align_val_t{Alignment}));
    }
    T* allocate(const std::size_t n, const void*){
        return allocate(n);
    }

    void deallocate(T* const p, std::size_t n){
        ::operator delete(p,n*sizeof(T),std::align_val_t{Alignment});
    }

    template<typename U, typename...Args>
    void construct(U* const p, Args&&...args){
        ::new (const_cast<void*>(static_cast<const volatile void*>(p))) U(std::forward<Args>(args)...);
    }

    template<typename U>
    void destroy(U* const p){
        p->~U();
    }

};

template<typename U, typename V>
bool operator==(const aligned_allocator<U>&, const aligned_allocator<V>&){
    return true;
}
template<typename U, typename V>
bool operator!=(const aligned_allocator<U>&, const aligned_allocator<V>&){
    return false;
}

}   //end of namespace allocation
#endif