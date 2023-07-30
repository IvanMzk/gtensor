#ifndef BENCHMARK_HELPERS_HPP_
#define BENCHMARK_HELPERS_HPP_

#include <type_traits>
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"

namespace benchmark_helpers{

template<std::size_t Depth, typename F=std::plus<void>, typename T, typename...Ts, std::enable_if_t< Depth==1,int> = 0 >
auto make_asymmetric_tree(const T& t, const Ts&...ts){
    return F{}(t,ts...);
}
template<std::size_t Depth, typename F=std::plus<void>, typename T, typename...Ts, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_asymmetric_tree(const T& t, const Ts&...ts){
    return make_asymmetric_tree<Depth-1,F>(F{}(t,ts...),ts...);
}

template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_symmetric_tree(const T1& t1, const T2& t2){
    return make_symmetric_tree<Depth-1>(t2+t1,t2+t1);
}
template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< Depth==1,int> = 0 >
auto make_symmetric_tree(const T1& t1, const T2& t2){
    return t2+t1;
}

template<typename F, typename...Args>
auto benchmark(std::string label, const F& f, Args&&...args){
    BENCHMARK_ADVANCED(label.c_str())(Catch::Benchmark::Chronometer meter) {
        meter.measure([&] { return f(std::forward<Args>(args)...); });
    };
    return 0;
}

}   //end of namespace benchmark_helpers


#endif