#ifndef BENCHMARK_HELPERS_HPP_
#define BENCHMARK_HELPERS_HPP_

#include <type_traits>
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"

namespace benchmark_helpers{

template<std::size_t Depth>
struct asymmetric_tree_maker{
    static constexpr std::size_t depth = Depth;
    static constexpr char* name = "asymmetric_tree_maker";
    template<std::size_t Dep = Depth, typename T1, typename T2, std::enable_if_t< (Dep>1) ,int> = 0 >
    auto operator()(const T1& t1, const T2& t2){
        return operator()<Dep-1>(t1,t2+t1);
    }
    template<std::size_t Dep = Depth, typename T1, typename T2, std::enable_if_t< Dep==1,int> = 0 >
    auto operator()(const T1& t1, const T2& t2){
        return t2+t1;
    }
};
template<std::size_t Depth>
struct asymmetric_tree_trivial_subtree_maker{
    static constexpr std::size_t depth = Depth;
    static constexpr char* name = "asymmetric_tree_trivial_subtree_maker";
    template<typename T1, typename T2>
    auto operator()(const T1& t1, const T2& t2){
        return helper(t1,t2,t1);
    }
    template<std::size_t Dep = Depth, typename T1, typename T2, typename T3, std::enable_if_t< (Dep>Depth/2) ,int> = 0 >
    auto helper(const T1& t1, const T2& t2, const T3& sum){
        return helper<Dep-1>(t1,t2,sum+t1);
    }
    template<std::size_t Dep = Depth, typename T1, typename T2, typename T3, std::enable_if_t< (Dep<=Depth/2) && (Dep>1) ,int> = 0 >
    auto helper(const T1& t1, const T2& t2, const T3& sum){
        return helper<Dep-1>(t1,t2,sum+t2);
    }
    template<std::size_t Dep = Depth, typename T1, typename T2, typename T3, std::enable_if_t< Dep==1,int> = 0 >
    auto helper(const T1& t1, const T2& t2, const T3& sum){
        return sum+t2;
    }
};
template<std::size_t Depth>
struct symmetric_tree_maker{
    static constexpr std::size_t depth = Depth;
    static constexpr char* name = "symmetric_tree_maker";
    template<std::size_t Dep = Depth, typename T1, typename T2, std::enable_if_t< (Dep>1) ,int> = 0 >
    auto operator()(const T1& t1, const T2& t2){
        return operator()<Dep-1>(t2+t1,t2+t1);
    }
    template<std::size_t Dep = Depth, typename T1, typename T2, std::enable_if_t< Dep==1,int> = 0 >
    auto operator()(const T1& t1, const T2& t2){
        return t2+t1;
    }
};


template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_asymmetric_tree(const T1& t1, const T2& t2){
    return make_asymmetric_tree<Depth-1>(t1,t2+t1);
}
template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< Depth==1,int> = 0 >
auto make_asymmetric_tree(const T1& t1, const T2& t2){
    return t2+t1;
}

template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_symmetric_tree(const T1& t1, const T2& t2){
    return make_symmetric_tree<Depth-1>(t2+t1,t2+t1);
}
template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< Depth==1,int> = 0 >
auto make_symmetric_tree(const T1& t1, const T2& t2){
    return t2+t1;
}

auto iterate_deref = [](auto& it_begin, auto& it_end){
    std::size_t c{};
    while (it_begin!=it_end){
        if (*it_begin > 2){
            ++c;
        }
        ++it_begin;
    }
    return c;
};
auto making_iter_iterate_deref = [](const auto& t){
    auto it = t.begin();
    auto end = t.end();
    typename std::iterator_traits<decltype(it)>::value_type tmp{};
    while (it!=end){
        tmp = *it;
        ++it;
    }
    return tmp;
};
auto making_riter_iterate_deref = [](const auto& t){
    auto it = t.rbegin();
    auto end = t.rend();
    typename std::iterator_traits<decltype(it)>::value_type tmp{};
    while (it!=end){
        tmp = *it;
        ++it;
    }
    return tmp;
};

template<typename F, typename...Args>
auto benchmark(std::string label, const F& f, Args&&...args){
    BENCHMARK_ADVANCED(label.c_str())(Catch::Benchmark::Chronometer meter) {
        meter.measure([&] { return f(std::forward<Args>(args)...); });
    };
    return 0;
}

template<typename F, typename Arg>
auto benchmark_with_making_iter(const Arg& arg, std::string label, const F& f = making_iter_iterate_deref){
    BENCHMARK_ADVANCED(label.c_str())(Catch::Benchmark::Chronometer meter) {
        meter.measure([&] { return f(arg); });
    };
    return 0;
}
template<typename Arg>
auto benchmark_without_making_iter(std::string label, const Arg& arg){
    BENCHMARK_ADVANCED(label.c_str())(Catch::Benchmark::Chronometer meter) {
        auto f = iterate_deref;
        auto v = make_iterators(meter.runs(),arg);
        meter.measure([&f,&v,&arg](int i) { return f(v[i].first, v[i].second); });
    };
    return 0;
}

}   //end of namespace benchmark_helpers


#endif