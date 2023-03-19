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

template<typename CfgT>
struct libdivide_container_selector
{
    template<typename T> using container = typename CfgT::template container<libdivide_divider<T>>;
};

template<typename CfgT, typename ShT>
inline auto make_libdivide_container(const ShT& src){
    using value_type = typename ShT::value_type;
    typename libdivide_container_selector<CfgT>::template container<value_type> res{};
    res.reserve(src.size());
    for(const auto& i:src){
        res.push_back(libdivide_divider<value_type>(i));
    }
    return res;
}

template<typename CfgT, typename ShT>
inline auto make_dividers(const ShT& src, gtensor::config::mode_div_libdivide){
    return make_libdivide_container<CfgT>(src);
}
template<typename CfgT, typename ShT>
inline auto make_dividers(const ShT& src, gtensor::config::mode_div_native){
    return src;
}
template<typename CfgT, typename ShT>
inline auto make_dividers(const ShT& src){
    return make_dividers<CfgT>(src, typename CfgT::div_mode{});
}

//returns quotient, reminder write back to dividend
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

}   //end of namespace detail
}   //end of namespace gtensor


#endif