#ifndef LIBDIVIDE_HELPER_HPP_
#define LIBDIVIDE_HELPER_HPP_

#include <type_traits>
#include "libdivide.h"
#include "config.hpp"

namespace gtensor{
namespace detail{

template<typename> inline constexpr bool is_libdivide_div = false;
template<typename T, libdivide::Branching Algo> inline constexpr bool is_libdivide_div<libdivide::divider<T,Algo>> = true;
template<typename T> using libdivide_divider = libdivide::divider<T>;
template<typename T> using libdivide_vector = std::vector<libdivide_divider<T>>;

template<typename CfgT>
struct libdiv_strides_traits{
    template<typename> struct selector{using type = typename CfgT::shape_type;};
    template<> struct selector<config::mode_div_native>{using type = typename CfgT::shape_type;};
    template<> struct selector<config::mode_div_libdivide>{using type = libdivide_vector<typename CfgT::index_type>;};
    using type = typename selector<typename CfgT::div_mode>::type;
};

template<template<typename> typename D, template<typename...> typename U, typename T>
auto make_libdiv_vector_helper(const U<T>& src){
    using div_type = D<T>;
    std::vector<div_type> res{};
    res.reserve(src.size());
    for(const auto& i:src){
        res.push_back(div_type(i));
    }
    return res;
}

template<typename CfgT, template<typename...> typename U, typename T, std::enable_if_t<is_mode_div_libdivide<CfgT>, int> =0 >
auto make_libdive_vector(const U<T>& src){
    return make_libdiv_vector_helper<libdivide::divider>(src);
}
template<typename CfgT, template<typename...> typename U, typename T, std::enable_if_t<is_mode_div_native<CfgT>, int> =0 >
auto make_libdive_vector(const U<T>& src){
    using div_type = libdivide::divider<T>;
    return std::vector<div_type>{};
}


// template<template<typename...> typename U, typename T>
// auto make_brfr_strides(const U<T>& strides){
//     return make_libdiv_strides<libdivide::branchfree_divider>(strides);
// }

/*
* convert flat index to multi index given strides
* result multiindex is of same type as strides container
*/
template<template<typename...> typename U, typename T, std::enable_if_t<!is_libdivide_div<T> ,int> =0 >
auto flat_to_multi(const U<T>& strides, typename const U<T>::value_type& idx){
    using shape_type = U<T>;
    using index_type = typename U<T>::value_type;
    shape_type res(strides.size(), index_type(0));    
    index_type idx_{idx};
    auto st_it = strides.begin();
    auto res_it = res.begin();
    while(idx_ != 0){
        auto q = idx_ / *st_it;
        idx_ %= *st_it;
        *res_it = q;
        ++st_it,++res_it;
    }        
    return res;
}

/*
* convert flat index to multi index given strides as container of libdivide objects
* need explicity specialize return type container for multiindex
*/
template<typename ShT, template<typename...> typename U, typename T, std::enable_if_t<is_libdivide_div<T> ,int> =0 >
auto flat_to_multi(const U<T>& strides, typename const T::value_type& idx){    
    using shape_type = ShT;
    using index_type = typename T::value_type;
    shape_type res(strides.size(), index_type(0));    
    index_type idx_{idx};
    auto st_it = strides.begin();
    auto res_it = res.begin();
    while(idx_ != 0){
        auto q = idx_ / *st_it;
        idx_ -= (*st_it).divisor()*q;
        *res_it = q;
        ++st_it,++res_it;
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