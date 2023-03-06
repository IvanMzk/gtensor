#ifndef LIBDIVIDE_HELPER_HPP_
#define LIBDIVIDE_HELPER_HPP_

#include <type_traits>
#include "libdivide.h"
#include "integral_type.hpp"
#include "config.hpp"

namespace gtensor{
namespace detail{

template<typename T>
class libdivide_divider : public libdivide::divider<T>
{
    using divider_base = libdivide::divider<T>;
    T divider_;
public:
    libdivide_divider() = default;
    explicit libdivide_divider(const T& divider__):
        divider_base{divider__},
        divider_{divider__}
    {}
    auto divider()const{return divider_;}
};

template<typename T>
class libdivide_divider<integral<T>> : public libdivide::divider<T>
{
    using divider_base = libdivide::divider<T>;
    integral<T> divider_;
public:
    libdivide_divider() = default;
    explicit libdivide_divider(const integral<T>& divider__):
        divider_base{divider__.value()},
        divider_{divider__}
    {}
    auto divider()const{return divider_;}
};

template<typename T>
auto operator/(const integral<T>& n, const libdivide_divider<integral<T>>& divider){
    return integral<T>(n.value()/divider);
}

template<typename T> using libdivide_vector = std::vector<libdivide_divider<T>>;

template<typename ShT>
inline auto make_libdivide_vector(const ShT& src){
    using value_type = typename ShT::value_type;
    libdivide_vector<value_type> res{};
    res.reserve(src.size());
    for(const auto& i:src){
        res.push_back(libdivide_divider<value_type>(i));
    }
    return res;
}

template<typename ShT>
inline auto make_dividers(const ShT& src, gtensor::config::mode_div_libdivide){
    return make_libdivide_vector(src);
}
template<typename ShT>
inline auto make_dividers(const ShT& src, gtensor::config::mode_div_native){
    return src;
}
template<typename CfgT, typename ShT>
inline auto make_dividers(const ShT& src){
    return make_dividers(src, typename CfgT::div_mode{});
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
    dividend -= q*divider.divider();
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
    while(idx_ != index_type(0)){
        *res_it = divide(idx_,*st_it);
        ++st_it,++res_it;
    }
    return res;
}

//converts flat index to flat index given strides and converting strides
template<typename StT, typename CStT, typename IdxT>
auto flat_to_flat(const StT& strides, const CStT& cstrides, const IdxT& offset, const IdxT& idx){
    using index_type = IdxT;
    index_type res{offset};
    index_type idx_{idx};
    auto st_it = strides.begin();
    auto cst_it = cstrides.begin();
    while(idx_ != index_type(0)){
        res += *cst_it*divide(idx_,*st_it);
        ++st_it;
        ++cst_it;
    }
    return res;
}

}   //end of namespace detail
}   //end of namespace gtensor


#endif