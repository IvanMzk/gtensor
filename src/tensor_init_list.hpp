#ifndef TENSOR_INIT_LIST_HPP_
#define TENSOR_INIT_LIST_HPP_

#include <initializer_list>
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
template<std::size_t, std::size_t, typename T, typename S>
inline void list_parse_(const T&, S*){}
template<std::size_t Dims_number, std::size_t Dim = 0, typename T, typename S>
inline void list_parse_(std::initializer_list<T> list, S* shape_){
    if (shape_->size() == Dims_number){
        if (static_cast<std::size_t>((*shape_)[Dim]) != list.size()){
            throw tensor_init_list_exception("list bad shape - different list size for dim");
        }
    }
    else{
        if (list.size() == 0){
            throw tensor_init_list_exception("list bad shape - zero list size");
        }
        shape_->push_back(list.size());
    }
    for (auto p=list.begin();p!=list.end(); ++p){
        list_parse_<Dims_number, Dim+1>(*p, shape_);
    }
}
template<typename IdxT = std::size_t, typename S = std::vector<IdxT>, typename T>
inline auto list_parse(std::initializer_list<T> list){
    constexpr std::size_t dims_number{nested_initialiser_list_depth<decltype(list)>::value};
    S shape;
    shape.reserve(dims_number);
    list_parse_<dims_number>(list,&shape);
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