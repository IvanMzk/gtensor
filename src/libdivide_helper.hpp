#ifndef LIBDIVIDE_HELPER_HPP_
#define LIBDIVIDE_HELPER_HPP_

#include <type_traits>
#include "libdivide.h"
#include "config.hpp"

namespace gtensor{
namespace detail{

template<typename T> using libdivide_divider = libdivide::divider<T>;
template<typename T> using libdivide_vector = std::vector<libdivide_divider<T>>;

template<template<typename> typename D, template<typename...> typename U, typename T>
auto make_libdivide_vector_helper(const U<T>& src){
    using div_type = D<T>;
    std::vector<div_type> res{};
    res.reserve(src.size());
    for(const auto& i:src){
        res.push_back(div_type(i));
    }
    return res;
}

template<template<typename...> typename U, typename T>
auto make_libdivide_vector(const U<T>& src){
    return make_libdivide_vector_helper<libdivide::divider>(src);
}

template<typename CfgT, template<typename...> typename U, typename T, std::enable_if_t<is_mode_div_libdivide<CfgT>, int> =0 >
auto make_dividers(const U<T>& src){
    return make_libdivide_vector(src);
}
template<typename CfgT, template<typename...> typename U, typename T, std::enable_if_t<is_mode_div_native<CfgT>, int> =0 >
auto make_dividers(const U<T>& src){
    return src;
}

/*
* return quotient
* reminder write back to dividend
*/
template<typename T>
inline auto divide(T& dividend, const T& divider){
    auto q = dividend/divider;
    dividend -=q*divider;
    return q;
}
template<typename T>
inline auto divide(T& dividend, const libdivide_divider<T>& divider){
    auto q = dividend/divider;
    dividend -= q*divider.divisor();
    return q;
}

/*
* convert flat index to multi index given strides
* ShT result multiindex type, must specialize explicit
*/
template<typename ShT, typename StT, typename IdxT>
auto flat_to_multi(const StT& strides, const IdxT& idx){
    using shape_type = ShT;
    using index_type = IdxT;
    shape_type res(strides.size(), index_type(0));
    index_type idx_{idx};
    auto st_it = strides.begin();
    auto res_it = res.begin();
    while(idx_ != 0){
        *res_it = divide(idx_,*st_it);
        ++st_it,++res_it;
    }
    return res;
}

template<typename StT, typename CStT, typename IdxT>
auto flat_to_flat(const StT& strides, const CStT& cstrides, const IdxT& offset, const IdxT& idx){
    using index_type = IdxT;
    index_type res{offset};
    index_type idx_{idx};
    auto st_it = strides.begin();
    auto cst_it = cstrides.begin();
    while(idx_ != 0){
        res += *cst_it*divide(idx_,*st_it);
        ++st_it;
        ++cst_it;
    }
    return res;
}

template<typename CfgT, typename Mode> class collection_libdivide_extension;

template<typename CfgT>
class collection_libdivide_extension<CfgT,gtensor::config::mode_div_libdivide>
{
    using shape_type = typename CfgT::shape_type;
    using index_type = typename CfgT::index_type;
    detail::libdivide_vector<index_type> dividers_libdivide_;
protected:
    collection_libdivide_extension() = default;
    collection_libdivide_extension(const shape_type& dividers):
        dividers_libdivide_{detail::make_libdivide_vector(dividers)}
    {}
    const auto&  dividers_libdivide()const{return dividers_libdivide_;}
};

template<typename CfgT>
class collection_libdivide_extension<CfgT,gtensor::config::mode_div_native>
{
    using shape_type = typename CfgT::shape_type;
protected:
    collection_libdivide_extension() = default;
    collection_libdivide_extension(const shape_type&)
    {}
};

}   //end of namespace detail
}   //end of namespace gtensor


#endif