#ifndef BENCHMARK_HELPERS_HPP_
#define BENCHMARK_HELPERS_HPP_

#include <type_traits>
#include <string>
#include <cmath>
#include <chrono>
#include "config.hpp"
#include "descriptor.hpp"

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

class cpu_timer
{
    using clock_type = std::chrono::steady_clock;
    using time_point = typename clock_type::time_point;
    time_point point_;
public:
    cpu_timer():
        point_{clock_type::now()}
    {}
    friend auto operator-(const cpu_timer& end, const cpu_timer& start){
        return std::chrono::duration<double,std::milli>(end.point_-start.point_).count();
    }
};

class cpu_interval
{
    cpu_timer start_{};
    cpu_timer stop_{};
public:
    cpu_interval() = default;
    void start(){
        start_=cpu_timer{};
    }
    void stop(){
        stop_=cpu_timer{};
    }
    auto interval()const{
        return stop_-start_;
    }
    operator double()const{
        return interval();
    }
};

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
template<typename T> void fake_use(const T& t){asm volatile("":"+g"(const_cast<T&>(t)));}
#elif defined(_MSC_VER)
#pragma optimize("", off)
template<typename T> void fake_use(T&& t){t=t;}
#pragma optimize("", on)
#endif

template<typename F, typename...Args>
auto timing(F&& f, Args&&...args){
    cpu_interval dt{};
    dt.start(),fake_use(f(std::forward<Args>(args)...)),dt.stop();
    return dt;
}

template<typename Axes>
auto axes_to_str(const Axes& axes){
    if constexpr (gtensor::detail::is_container_v<Axes>){
        return gtensor::detail::shape_to_str(axes);
    }else{
        return std::to_string(axes);
    }
}

inline auto order_to_str(gtensor::config::c_order){return std::string{"c_order"};}
inline auto order_to_str(gtensor::config::f_order){return std::string{"f_order"};}

template<typename Container>
auto mean(const Container& intervals){
    using value_type = typename Container::value_type;
    return std::accumulate(intervals.begin(),intervals.end(),value_type{0})/intervals.size();
}
template<typename Container>
auto stdev(const Container& intervals){
    using value_type = typename Container::value_type;
    auto m = mean(intervals);
    auto v = std::accumulate(intervals.begin(),intervals.end(),value_type{0},[m](const auto& init, const auto& e){auto d=e-m; return init+d*d;})/intervals.size();
    return std::sqrt(v);
}

template<typename Container>
auto statistic(const Container& intervals){
    std::stringstream ss{};
    ss<<"min "<<*std::min_element(intervals.begin(),intervals.end())<<" ";
    ss<<"max "<<*std::max_element(intervals.begin(),intervals.end())<<" ";
    ss<<"mean "<<mean(intervals)<<" ";
    ss<<"stdev "<<stdev(intervals);
    return ss.str();
}


inline const std::vector<std::vector<int>> shapes{
    std::vector<int>{100000000,3,1,2},
    std::vector<int>{10000000,3,1,20},
    std::vector<int>{1000000,3,10,20},
    std::vector<int>{100000,3,100,20},
    std::vector<int>{10000,3,100,200},
    std::vector<int>{1000,3,1000,200},
    std::vector<int>{100,3,1000,2000},
    std::vector<int>{50,6,1000,2000}
};

inline const std::vector<std::vector<int>> shapes_2d{
    std::vector<int>{100000000,6},
    std::vector<int>{10000000,60},
    std::vector<int>{1000000,600},
    std::vector<int>{100000,6000},
    std::vector<int>{10000,60000},
    std::vector<int>{1000,600000},
    std::vector<int>{100,6000000},
    std::vector<int>{10,60000000},
    std::vector<int>{5,120000000},
};

inline const std::vector<std::vector<int>> small_shapes{
    std::vector<int>{1000000,3,1,2},
    std::vector<int>{100000,3,1,20},
    std::vector<int>{10000,3,10,20},
    std::vector<int>{1000,3,100,20},
    std::vector<int>{100,3,100,200},
    std::vector<int>{50,30,20,200},
    std::vector<int>{20,10,30,1000}
};

inline const std::vector<std::vector<int>> small_shapes_1{
    std::vector<int>{100000,3,1,2},
    std::vector<int>{10000,3,1,20},
    std::vector<int>{1000,3,10,20},
    std::vector<int>{100,3,100,20},
    std::vector<int>{10,3,100,200},
    std::vector<int>{10,30,10,200},
    std::vector<int>{100,30,10,20}
};

inline const std::vector<std::vector<int>> small_shapes_2{
    std::vector<int>{20,20,20,20},
    std::vector<int>{20,20,20,20},
    std::vector<int>{20,20,20,20},
    std::vector<int>{20,20,20,20},
    std::vector<int>{20,20,20,20},
    std::vector<int>{20,20,20,20},
    std::vector<int>{20,20,20,20},
    std::vector<int>{20,20,20,20},
    std::vector<int>{20,20,20,20}
};

inline const auto axes = std::vector<std::vector<int>>{
    std::vector<int>{0},
    std::vector<int>{1},
    std::vector<int>{2},
    std::vector<int>{3},
    std::vector<int>{0,1},
    std::vector<int>{0,2},
    std::vector<int>{0,3},
    std::vector<int>{1,2},
    std::vector<int>{1,3},
    std::vector<int>{2,3},
    std::vector<int>{0,1,2},
    std::vector<int>{1,2,3},
    std::vector<int>{0,1,2,3}
};

inline const auto axes_container = std::vector<std::vector<int>>{
    std::vector<int>{0,1},
    std::vector<int>{0,2},
    std::vector<int>{0,3},
    std::vector<int>{1,2},
    std::vector<int>{1,3},
    std::vector<int>{2,3},
    std::vector<int>{0,1,2},
    std::vector<int>{1,2,3}
};

inline const auto axes_scalar = std::vector<int>{0,1,2,3};
inline const auto axes_scalar_2d = std::vector<int>{0,1};

}   //end of namespace benchmark_helpers


#endif