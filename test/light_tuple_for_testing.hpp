#ifndef LIGHT_TUPLE_FOR_TESTING_HPP
#define LIGHT_TUPLE_FOR_TESTING_HPP

#include <type_traits>

#define LIGHT_TUPLE_GET(I)\
template<> struct getter<I>{\
    template<typename T> static auto& get(T& t){return t.t##I;}\
    template<typename T> static auto&& get(T&& t){return std::forward<T>(t).t##I;}\
};

namespace ltp{

template<typename...> struct ltuple;
template<> struct ltuple<>{};
template<typename T0>
struct ltuple<T0>
{
    T0 t0;
};
template<typename T0,typename T1>
struct ltuple<T0,T1>
{
    T0 t0;
    T1 t1;
};
template<typename T0,typename T1,typename T2>
struct ltuple<T0,T1,T2>
{
    T0 t0;
    T1 t1;
    T2 t2;
};

template<std::size_t I, typename Tuple> auto& get(Tuple&);
template<std::size_t I, typename Tuple> auto&& get(Tuple&&);

namespace ltuple_details{

template<typename...Ts,typename...Us, std::size_t...I>
inline bool equals(const ltuple<Ts...>& lhs, const ltuple<Us...>& rhs, std::index_sequence<I...>){
    return (...&&(get<I>(lhs)==get<I>(rhs)));
}

template<std::size_t> struct getter;
LIGHT_TUPLE_GET(0);
LIGHT_TUPLE_GET(1);
LIGHT_TUPLE_GET(2);

};

template<std::size_t I, typename Tuple> auto& get(Tuple& t){return ltuple_details::getter<I>::get(t);};
template<std::size_t I, typename Tuple> auto&& get(Tuple&& t){return ltuple_details::getter<I>::get(std::forward<Tuple>(t));};

template<typename...Ts,typename...Us>
inline bool operator==(const ltuple<Ts...>& lhs, const ltuple<Us...>& rhs){
    static_assert(sizeof...(Ts) == sizeof...(Us));
    return ltuple_details::equals(lhs, rhs, std::make_index_sequence<sizeof...(Ts)>{});
}
template<typename...Ts,typename...Us>
inline bool operator!=(const ltuple<Ts...>& lhs, const ltuple<Us...>& rhs){return !(lhs==rhs);}

template<typename...Args>
auto create_ltuple(Args&&...args){
    return ltuple<std::decay_t<Args>...>{std::forward<Args>(args)...};
}


}   //end of namespace ltp

#endif