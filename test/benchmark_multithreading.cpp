#include <vector>
#include "multithreading.hpp"
#include "benchmark_helpers.hpp"

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


const std::vector<std::size_t> sizes{10,100,1000,10000,100000,1000000,10000000};

}   //end of namespace benchmark_multithreading_

TEST_CASE("benchmark_multithreading_reduce","benchmark_multithreading")
{

    using benchmark_multithreading_::bench_reduce;

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

    const auto n_iters = 100;
    const auto sizes = benchmark_multithreading_::sizes;

    //bench_reduce("reduce sum",n_iters,sizes,reduce_sum);
    bench_reduce("reduce prod",n_iters,sizes,reduce_prod);


    //const std::size_t size = 100'000'007;
    // const std::size_t size = 123'456'789;
    // const double value = 0.1;
    // const std::vector<double> vec1(size,value);
    // auto dt1 = benchmark_helpers::timing(reduce_sum,vec1,multithreading::exec_pol<1>{});
    // std::cout<<std::endl<<dt1;

    // //const std::vector<double> vec2(size,value);
    // auto dt2 = benchmark_helpers::timing(reduce_sum,vec1,multithreading::exec_pol<0>{});
    // std::cout<<std::endl<<dt2;
}

