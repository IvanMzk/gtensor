#ifndef LIBDIVIDE_HELPER_HPP_
#define LIBDIVIDE_HELPER_HPP_

#include <type_traits>
#include "libdivide.h"

namespace gtensor{
namespace detail{

template<typename> inline constexpr bool is_libdivide_div = false;
template<typename T, libdivide::Branching Algo> inline constexpr bool is_libdivide_div<libdivide::divider<T,Algo>> = true;

template<template<typename> typename D, template<typename...> typename U, typename T>
auto make_libdiv_strides(const U<T>& strides){
    using div_type = D<T>;
    using strides_type = std::vector<div_type>;
    strides_type res{};
    res.reserve(strides.size());
    for(const auto& i:strides){
        res.push_back(div_type(i));
    }
    return res;
}

template<template<typename...> typename U, typename T>
auto make_brfl_strides(const U<T>& strides){
    return make_libdiv_strides<libdivide::divider>(strides);
}
template<template<typename...> typename U, typename T>
auto make_brfr_strides(const U<T>& strides){
    return make_libdiv_strides<libdivide::branchfree_divider>(strides);
}


template<template<typename...> typename U, typename T, std::enable_if_t<!is_libdivide_div<T> ,int> =0 >
auto flat_to_multi(const U<T>& strides, typename const U<T>::value_type& idx){
    using shape_type = U<T>;
    using index_type = typename U<T>::value_type;
    shape_type res(strides.size(), index_type(0));    
    index_type idx_{idx};
    auto st_it = strides.end();
    auto res_it = res.end();
    while(idx_ != 0){
        auto q = idx_ / *--st_it;
        idx_ %= *st_it;
        *--res_it = q;
    }    
    return res;
}

template<typename ShT, template<typename...> typename U, typename T, std::enable_if_t<is_libdivide_div<T> ,int> =0 >
auto flat_to_multi(const U<T>& strides, typename const T::value_type& idx){
    //using shape_type = U<T>;
    using shape_type = ShT;
    using index_type = typename T::value_type;
    shape_type res(strides.size(), index_type(0));    
    index_type idx_{idx};
    auto st_it = strides.end();
    auto res_it = res.end();
    while(idx_ != 0){
        auto q = idx_ / *--st_it;
        idx_ -= (*st_it).divisor()*q;
        *--res_it = q;
    }    
    return res;
}



// template<typename Wkr, template<typename...> typename U, typename T, std::enable_if_t<!is_libdivide_div<T> ,int> =0 >
// void walker_advance(Wkr& walker, const U<T>& strides, typename const U<T>::value_type& distance){    
//     using index_type = typename U<T>::value_type;    
//     index_type idx_{idx};
//     auto st_it = strides.end();    
//     while(idx_ != 0){
//         auto q = idx_ / *--st_it;
//         idx_ %= *st_it;
//         *--res_it = q;
//     }    
//     return res;
// }


}   //end of namespace detail
}   //end of namespace gtensor


#endif