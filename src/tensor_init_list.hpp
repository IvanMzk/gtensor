#ifndef TENSOR_INIT_LIST_HPP_
#define TENSOR_INIT_LIST_HPP_

#include <initializer_list>
#include <iostream>
#include <utility>
#include <vector>
#include <numeric>
#include <functional>

namespace gtensor{

class tensor_init_list_exception : public std::runtime_error{
    public: tensor_init_list_exception(const char* what):runtime_error(what){}
};

namespace detail{

/*
* type of nested init list
* T type of list elements
* N list depth
*/
template<typename T, std::size_t N>
struct nested_initializer_list_type{
    using type = typename std::initializer_list<typename nested_initializer_list_type<T,N-1>::type>;
};
template<typename T>
struct nested_initializer_list_type<T, 0>{
    using type = T;
};

// template<typename T> using nested_init_list1_t = std::initializer_list<T>;
// template<typename T> using nested_init_list2_t = std::initializer_list<nested_init_list1_t<T>>;
// template<typename T> using nested_init_list3_t = std::initializer_list<nested_init_list2_t<T>>;
// template<typename T> using nested_init_list4_t = std::initializer_list<nested_init_list3_t<T>>;
// template<typename T> using nested_init_list5_t = std::initializer_list<nested_init_list4_t<T>>;

template<typename T> using nested_init_list1_type = std::initializer_list<T>;
template<typename T> using nested_init_list2_type = std::initializer_list<std::initializer_list<T>>;
template<typename T> using nested_init_list3_type = std::initializer_list<std::initializer_list<std::initializer_list<T>>>;
template<typename T> using nested_init_list4_type = std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>>;
template<typename T> using nested_init_list5_type = std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>>>;


/*
* compile time nested init list depth
* T list type
*/
template<typename T, std::size_t Depth=0>
struct nested_initialiser_list_depth{
    static constexpr std::size_t value = Depth;
};
template<typename T, std::size_t Depth>
struct nested_initialiser_list_depth<std::initializer_list<T>, Depth>{
    static constexpr std::size_t value = nested_initialiser_list_depth<T, Depth+1>::value;
};

template<typename T, std::size_t Depth=0>
struct nested_initialiser_list_value_type{
    using type = T;
};
template<typename T, std::size_t Depth>
struct nested_initialiser_list_value_type<std::initializer_list<T>, Depth>{
    using type = typename nested_initialiser_list_value_type<T, Depth+1>::type;
};



/*
* nested list elements number
*/
template<typename T, typename U>
inline auto list_size_(T , U*){
    return true;
}
template<typename T, typename U>
inline auto list_size_(std::initializer_list<T> list, U* size){
    for (auto p=list.begin();p!=list.end(); ++p){
        if (list_size_(*p, size)){
            *size += list.size();
            break;
        }
    }
    return false;
}
template<typename T>
inline auto list_size(std::initializer_list<T> list){
    std::size_t size = 0;
    list_size_(list, &size);
    return size;
}

/*
* parse nested list and return its shape
* exception if list has invalid structure
*/
template<typename SizeT, std::size_t, std::size_t, typename T, typename ShT>
inline void list_parse_(const T&, ShT*){}
template<typename SizeT, std::size_t Dims_number, std::size_t Dim = 0, typename T, typename ShT>
inline void list_parse_(std::initializer_list<T> list, ShT* shape_){
    using size_type = SizeT;
    using index_type = typename ShT::value_type;
    size_type shape_size = shape_->size();
    if (shape_size == size_type(Dims_number)){
        if ( (*shape_)[size_type(Dim)] != index_type(list.size())){
            throw tensor_init_list_exception("list bad shape - different list size for dim");
        }
    }
    else{
        if (list.size() == 0){
            throw tensor_init_list_exception("list bad shape - zero list size");
        }
        shape_->push_back(index_type(list.size()));
    }
    for (auto p=list.begin();p!=list.end(); ++p){
        list_parse_<SizeT, Dims_number, Dim+1>(*p, shape_);
    }
}
template<typename SizeT = std::size_t, typename ShT = std::vector<std::size_t>, typename T>
inline auto list_parse(std::initializer_list<T> list){
    using size_type = SizeT;
    constexpr std::size_t dims_number{nested_initialiser_list_depth<decltype(list)>::value};
    ShT shape;
    shape.reserve(size_type(dims_number));
    list_parse_<size_type, dims_number>(list,&shape);
    return shape;
}

/*
* copy init list to Dst
*/
template<typename Dst_It, typename T>
void fill_from_list_(const T& v, Dst_It& dst_it, std::size_t& size_){
    *dst_it = v;
    ++dst_it;
    ++size_;
}
template<typename Dst_It, typename T>
void fill_from_list_(std::initializer_list<T> list, Dst_It& dst_it, std::size_t& size_){
    for (auto p=list.begin();p!=list.end(); ++p){
        fill_from_list_(*p, dst_it, size_);
    }
}
template<typename Dst_It, typename T>
std::size_t fill_from_list(std::initializer_list<T> list, Dst_It dst_it){
    std::size_t size_{0};
    fill_from_list_(list, dst_it, size_);
    return size_;
}
}   //end of namespace detail

}   //end of namespace gtensor

#endif