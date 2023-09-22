#include <vector>
#include <execution>
#include "multithreading.hpp"
#include "../benchmark_helpers.hpp"

namespace benchmark_multithreading_{

using benchmark_helpers::timing;
using benchmark_helpers::statistic;
using multithreading::get_pool;
using multithreading::exec_policy_traits;

struct bench_reduce_helper{
    using value_type = double;

    template<typename Policy, typename Sizes, typename TimingF>
    auto operator()(std::string mes, Policy policy, const std::size_t n_iters, const Sizes& sizes, TimingF timing_f){
        std::cout<<std::endl<<"par_tasks "<<exec_policy_traits<Policy>::par_tasks::value<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto it=sizes.begin(),last=sizes.end(); it!=last; ++it){
            const auto size = *it;
            std::vector<double> intervals{};
            for (auto n=n_iters; n!=0; --n){
                std::vector<value_type> vec(size,2);
                double dt = timing(timing_f,vec,policy);
                intervals.push_back(dt);
                total_intervals.push_back(dt);
            }
            std::cout<<std::endl<<"input size "<<size<<" "<<statistic(intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    }
};

template<typename Sizes, typename TimingF>
auto bench_reduce(std::string mes, const std::size_t n_iters, const Sizes& sizes, TimingF timing_f){
    using multithreading::exec_pol;
    bench_reduce_helper{}(mes,exec_pol<1>{},n_iters,sizes,timing_f);
    bench_reduce_helper{}(mes,exec_pol<2>{},n_iters,sizes,timing_f);
    bench_reduce_helper{}(mes,exec_pol<4>{},n_iters,sizes,timing_f);
    bench_reduce_helper{}(mes,exec_pol<8>{},n_iters,sizes,timing_f);
    bench_reduce_helper{}(mes,exec_pol<0>{},n_iters,sizes,timing_f);
}

struct bench_transform_helper{
    using value_type = double;

    template<typename Policy, typename Sizes, typename TimingF>
    auto operator()(std::string mes, Policy policy, const std::size_t n_iters, const Sizes& sizes, TimingF timing_f){
        std::cout<<std::endl<<"par_tasks "<<exec_policy_traits<Policy>::par_tasks::value<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto it=sizes.begin(),last=sizes.end(); it!=last; ++it){
            const auto size = *it;
            std::vector<double> intervals{};
            for (auto n=n_iters; n!=0; --n){
                std::vector<value_type> vec1(size,2);
                std::vector<value_type> vec2(size,3);
                std::vector<value_type> dvec(size);
                double dt = timing(timing_f,vec1,vec2,dvec,policy);
                intervals.push_back(dt);
                total_intervals.push_back(dt);
            }
            std::cout<<std::endl<<"input size "<<size<<" "<<statistic(intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    }
};

template<typename Sizes, typename TimingF>
auto bench_transform(std::string mes, const std::size_t n_iters, const Sizes& sizes, TimingF timing_f){
    using multithreading::exec_pol;
    bench_transform_helper{}(mes,exec_pol<1>{},n_iters,sizes,timing_f);
    bench_transform_helper{}(mes,exec_pol<2>{},n_iters,sizes,timing_f);
    bench_transform_helper{}(mes,exec_pol<4>{},n_iters,sizes,timing_f);
    bench_transform_helper{}(mes,exec_pol<8>{},n_iters,sizes,timing_f);
    bench_transform_helper{}(mes,exec_pol<0>{},n_iters,sizes,timing_f);
}

struct bench_copy_helper{
    using value_type = double;

    template<typename Policy, typename Sizes, typename TimingF>
    auto operator()(std::string mes, Policy policy, const std::size_t n_iters, const Sizes& sizes, TimingF timing_f){
        std::cout<<std::endl<<"par_tasks "<<exec_policy_traits<Policy>::par_tasks::value<<" "<<mes;
        std::vector<double> total_intervals{};
        for (auto it=sizes.begin(),last=sizes.end(); it!=last; ++it){
            const auto size = *it;
            std::vector<double> intervals{};
            for (auto n=n_iters; n!=0; --n){
                std::vector<value_type> vec(size,22);
                std::vector<value_type> dvec(size);
                double dt = timing(timing_f,vec,dvec,policy);
                intervals.push_back(dt);
                total_intervals.push_back(dt);
            }
            std::cout<<std::endl<<"input size "<<size<<" "<<statistic(intervals);
        }
        std::cout<<std::endl<<"TOTAL "<<statistic(total_intervals);
    }
};

template<typename Sizes, typename TimingF>
auto bench_copy(std::string mes, const std::size_t n_iters, const Sizes& sizes, TimingF timing_f){
    using multithreading::exec_pol;
    bench_copy_helper{}(mes,exec_pol<1>{},n_iters,sizes,timing_f);
    bench_copy_helper{}(mes,exec_pol<2>{},n_iters,sizes,timing_f);
    bench_copy_helper{}(mes,exec_pol<4>{},n_iters,sizes,timing_f);
    bench_copy_helper{}(mes,exec_pol<8>{},n_iters,sizes,timing_f);
    bench_copy_helper{}(mes,exec_pol<0>{},n_iters,sizes,timing_f);
}

const std::vector<std::size_t> sizes{10,100,1000,10000,100000,1000000,10000000};

}   //end of namespace benchmark_multithreading_

TEST_CASE("benchmark_multithreading_reduce","benchmark_multithreading")
{

    using benchmark_multithreading_::bench_reduce;
    using benchmark_multithreading_::bench_transform;
    using benchmark_multithreading_::bench_copy;

    auto reduce_sum = [](const auto& v, auto pol){
        using value_type = typename std::iterator_traits<decltype(v.begin())>::value_type;
        value_type initial{0};
        auto r = multithreading::reduce(pol,v.begin(),v.end(),initial,std::plus<void>{});
        return r;
    };

    auto reduce_prod = [](const auto& v, auto pol){
        using value_type = typename std::iterator_traits<decltype(v.begin())>::value_type;
        value_type initial{0};
        auto r = multithreading::reduce(pol,v.begin(),v.end(),initial,std::multiplies<void>{});
        return r;
    };

    auto std_reduce_sum = [](auto& v, auto pol){
        using value_type = typename std::iterator_traits<decltype(v.begin())>::value_type;
        value_type initial{0};
        return std::reduce(v.begin(),v.end(),initial,std::plus<void>{});
    };

    auto std_accumulate_sum = [](auto& v, auto pol){
        using value_type = typename std::iterator_traits<decltype(v.begin())>::value_type;
        value_type initial{0};
        return std::accumulate(v.begin(),v.end(),initial,std::plus<void>{});
    };

    auto transform_sum = [](auto& v1, const auto& v2, auto& dv, auto pol){
        multithreading::transform(pol,v1.begin(),v1.end(),v2.begin(),dv.begin(),std::plus<void>{});
        return *dv.begin();
    };


    auto transform_sum_dst_first = [](auto& v1, const auto& v2, auto&, auto pol){
        multithreading::transform(pol,v1.begin(),v1.end(),v2.begin(),v1.begin(),std::plus<void>{});
        return *v1.begin();
    };

    auto transform_sum_overload = [](auto& v1, const auto& v2, auto&, auto pol){
        multithreading::transform(pol,v1.begin(),v1.end(),v2.begin(),std::plus<void>{});
        return *v1.begin();
    };

    auto std_transform_sum_dst_first = [](auto& v1, const auto& v2, auto&, auto pol){
        return std::transform(v1.begin(),v1.end(),v2.begin(),v1.begin(),std::plus<void>{});
    };

    auto std_transform_sum = [](auto& v1, const auto& v2, auto& dv, auto pol){
        return std::transform(v1.begin(),v1.end(),v2.begin(),dv.begin(),std::plus<void>{});
    };

    auto copy = [](const auto& v, auto& dv, auto pol){
        multithreading::copy(pol,v.begin(),v.end(),dv.begin());
        return *dv.begin();
    };

    const auto n_iters = 100;
    const auto sizes = benchmark_multithreading_::sizes;

    // const auto n_iters = 1;
    // const auto sizes = std::vector<std::size_t>{600000000};

    //bench_reduce("reduce sum",n_iters,sizes,reduce_sum);
    //bench_reduce("reduce prod",n_iters,sizes,reduce_prod);
    //bench_reduce("std_reduce sum",n_iters,sizes,std_reduce_sum);
    //bench_reduce("std_accumulate sum",n_iters,sizes,std_accumulate_sum);

    //bench_transform("transform sum",n_iters,sizes,transform_sum);
    //bench_transform("transform sum dst first",n_iters,sizes,transform_sum_dst_first);
    //bench_transform("transform sum overload",n_iters,sizes,transform_sum_overload);
    //bench_transform("std_transform sum",n_iters,sizes,std_transform_sum);
    //bench_transform("std_transform sum dst first",n_iters,sizes,std_transform_sum_dst_first);

    bench_copy("copy",n_iters,sizes,copy);

}

