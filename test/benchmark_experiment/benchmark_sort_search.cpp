#include <tuple>
#include <vector>
#include <numeric>
#include <execution>
#include <algorithm>
#include "tensor.hpp"
#include "reduce.hpp"
#include "sort_search.hpp"
#include "../benchmark_helpers.hpp"
#include "../helpers_for_testing.hpp"

namespace benchmark_sort_search{

using gtensor::basic_tensor;

struct sort0_functor
{
    //Comparator can be binary predicate functor or no_value
    template<typename It, typename DstIt, typename Comparator>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, const Comparator& comparator){
        std::copy(first,last,dfirst);
        if constexpr (std::is_same_v<Comparator,gtensor::detail::no_value>){
            std::sort(dfirst,dlast);
        }else{
            std::sort(dfirst,dlast,comparator);
        }
    }
};

struct sort1_functor
{
    //Comparator can be binary predicate functor or no_value
    template<typename It, typename DstIt, typename Comparator, typename Config>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, const Comparator& comparator, Config){
        using value_type = typename std::iterator_traits<It>::value_type;
        using container_type = typename Config::template container<value_type>;
        container_type elements(first,last);
        if constexpr (std::is_same_v<Comparator,gtensor::detail::no_value>){
            std::sort(elements.begin(),elements.end());
        }else{
            std::sort(elements.begin(),elements.end(),comparator);
        }
        std::copy(elements.begin(),elements.end(),dfirst);
    }
};

template<typename Config, typename T>
struct sort2_functor
{
    using container_type = typename Config::template container<T>;
    std::shared_ptr<container_type> elements{};
    //Comparator can be binary predicate functor or no_value
    template<typename It, typename DstIt, typename Comparator>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, const Comparator& comparator){
        if (!static_cast<bool>(elements)){
            const auto n = last-first;
            elements = std::make_shared<container_type>(n);
        }
        std::copy(first,last,elements->begin());
        if constexpr (std::is_same_v<Comparator,gtensor::detail::no_value>){
            std::sort(elements->begin(),elements->end());
        }else{
            std::sort(elements->begin(),elements->end(),comparator);
        }
        std::copy(elements->begin(),elements->end(),dfirst);
    }
};
// template<typename Config, typename T>
// struct sort2_functor
// {
//     typename Config::template container<T> elements{};
//     //Comparator can be binary predicate functor or no_value
//     template<typename It, typename DstIt, typename Comparator>
//     void operator()(It first, It last, DstIt dfirst, DstIt dlast, const Comparator& comparator){
//         elements.assign(first,last);
//         if constexpr (std::is_same_v<Comparator,gtensor::detail::no_value>){
//             std::sort(elements.begin(),elements.end());
//         }else{
//             std::sort(elements.begin(),elements.end(),comparator);
//         }
//         std::copy(elements.begin(),elements.end(),dfirst);
//     }
// };


template<typename...Ts, typename DimT, typename Comparator = gtensor::detail::no_value>
auto sort0(const basic_tensor<Ts...>& t, const DimT& axis, const Comparator& comparator=gtensor::detail::no_value{}){
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return gtensor::slide(t,axis,sort0_functor{}, window_size, window_step, comparator);
}

template<typename...Ts, typename DimT, typename Comparator = gtensor::detail::no_value>
auto sort1(const basic_tensor<Ts...>& t, const DimT& axis, const Comparator& comparator=gtensor::detail::no_value{}){
    using config_type = typename basic_tensor<Ts...>::config_type;
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return gtensor::slide(t,axis,sort1_functor{}, window_size, window_step, comparator, config_type{});
}

template<typename...Ts, typename DimT, typename Comparator = gtensor::detail::no_value>
auto sort2(const basic_tensor<Ts...>& t, const DimT& axis, const Comparator& comparator=gtensor::detail::no_value{}){
    using config_type = typename basic_tensor<Ts...>::config_type;
    using value_type = typename basic_tensor<Ts...>::value_type;
    using index_type = typename basic_tensor<Ts...>::index_type;
    const index_type window_size = 1;
    const index_type window_step = 1;
    return gtensor::slide(t,axis,sort2_functor<config_type,value_type>{}, window_size, window_step, comparator);
}

auto generate_uniform = [](auto first, auto last, const auto& min, const auto& max){
    using value_type = typename std::iterator_traits<decltype(first)>::value_type;
    std::mt19937_64 gen{std::random_device{}()};
    if constexpr(std::is_integral_v<value_type>){
        std::uniform_int_distribution<value_type> distr{static_cast<value_type>(min),static_cast<value_type>(max)};
        for (;first!=last; ++first){
            *first = distr(gen);
        }
    }else if constexpr(std::is_floating_point_v<value_type>){
        std::uniform_real_distribution<value_type> distr{static_cast<value_type>(min),static_cast<value_type>(max)};
        for (;first!=last; ++first){
            *first = distr(gen);
        }
    }else{
        //exception
    }
};

}   //end of namespace benchmark_sort_search

TEST_CASE("test_sort","[benchmark_algorithm]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::detail::no_value;
    using benchmark_sort_search::sort0;
    using benchmark_sort_search::sort1;
    using benchmark_sort_search::sort2;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2comparator,3expected
    auto test_data = std::make_tuple(
        //no comparator
        std::make_tuple(tensor_type{},0,no_value{},tensor_type{}),
        std::make_tuple(tensor_type{1},0,no_value{},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,no_value{},tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,no_value{},tensor_type{0,1,1,2,2,3,5,6}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,no_value{},tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,4,4},{8,7,6,6,5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,no_value{},tensor_type{{-1,1,2,3,6},{0,1,2,5,8},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}}),
        //comparator
        std::make_tuple(tensor_type{},0,std::less<void>{},tensor_type{}),
        std::make_tuple(tensor_type{1},0,std::less<void>{},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,std::less<void>{},tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,std::greater<void>{},tensor_type{6,5,3,2,2,1,1,0}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,std::less<void>{},tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,4,4},{8,7,6,6,5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,std::greater<void>{},tensor_type{{6,3,2,1,-1},{8,5,2,1,0},{7,4,2,0,-1},{4,4,4,2,1},{6,4,3,3,1}})
    );

    SECTION("test_sort0")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto comparator = std::get<2>(t);
            auto expected = std::get<3>(t);

            auto result = sort0(ten,axis,comparator);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_sort1")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto comparator = std::get<2>(t);
            auto expected = std::get<3>(t);

            auto result = sort1(ten,axis,comparator);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_sort2")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto comparator = std::get<2>(t);
            auto expected = std::get<3>(t);

            auto result = sort2(ten,axis,comparator);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("benchmark_sort_search_sort","[benchmark_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,gtensor::config::c_order>;
    using benchmark_helpers::benchmark;
    using benchmark_sort_search::sort0;
    using benchmark_sort_search::sort1;
    using benchmark_sort_search::sort2;
    using benchmark_sort_search::generate_uniform;

    auto bench_sort = [](const auto& t, const auto& axis){
        return gtensor::sort(t,axis);
    };
    auto bench_sort0 = [](const auto& t, const auto& axis){
        return sort0(t,axis);
    };
    auto bench_sort0_comparator = [](const auto& t, const auto& axis, const auto& comparator){
        return sort0(t,axis,comparator);
    };
    auto bench_sort1 = [](const auto& t, const auto& axis){
        return sort1(t,axis);
    };
    auto bench_sort2 = [](const auto& t, const auto& axis){
        return sort2(t,axis);
    };

    // tensor_type t_1d_1000({1000},0);
    // generate_uniform(t_1d_1000.begin(), t_1d_1000.end(), 0,1);
    // benchmark("sort0_1d_1000", bench_sort0, t_1d_1000, 0);
    // benchmark("sort1_1d_1000", bench_sort1, t_1d_1000, 0);


    // tensor_type t_1d_1000000({1000000},0);
    // generate_uniform(t_1d_1000000.begin(), t_1d_1000000.end(), 0,1);
    // benchmark("sort0_1d_1000000", bench_sort0, t_1d_1000000, 0);
    // benchmark("sort1_1d_1000000", bench_sort1, t_1d_1000000, 0);

    tensor_type t_2d_1000_1000({1000,1000},0);
    generate_uniform(t_2d_1000_1000.begin(), t_2d_1000_1000.end(), 0,1);
    benchmark("sort_2d_1000_1000", bench_sort, t_2d_1000_1000, 1);
    benchmark("sort0_2d_1000_1000", bench_sort0, t_2d_1000_1000, 1);
    benchmark("sort0_comparator_2d_1000_1000", bench_sort0_comparator, t_2d_1000_1000, 1, std::less<void>{});
    benchmark("sort1_2d_1000_1000", bench_sort1, t_2d_1000_1000, 1);
    benchmark("sort2_2d_1000_1000", bench_sort2, t_2d_1000_1000, 1);

    // tensor_type t_1d_10000000({10000000},0);
    // generate_uniform(t_1d_10000000.begin(), t_1d_10000000.end(), 0,1);
    // benchmark("sort0_1d_10000000", bench_sort0, t_1d_10000000, 0);
    // benchmark("sort1_1d_10000000", bench_sort1, t_1d_10000000, 0);
}

// TEST_CASE("benchmark_algorithm_min_element_reduce_polcy","[benchmark_algorithm]")
// {
//     using value_type = double;
//     using benchmark_helpers::benchmark;
//     using benchmark_algorithm::min_element;
//     using benchmark_algorithm::min_element_value_assign;
//     using benchmark_algorithm::min_element_accumulate;
//     using benchmark_algorithm::min_element_reduce;
//     using benchmark_algorithm::min_element_reduce_policy;

//     std::vector<value_type> elements_100(100,1);
//     std::vector<value_type> elements_10000(10000,2);
//     std::vector<value_type> elements_1000000(1000000,3);
//     std::vector<value_type> elements_100000000(100000000,4);


//     // benchmark("min_element_reduce_100", min_element_reduce, elements_100.begin(), elements_100.end());
//     // benchmark("min_element_reduce_policy_seq_100", min_element_reduce_policy, std::execution::seq, elements_100.begin(), elements_100.end());
//     // benchmark("min_element_reduce_policy_par_100", min_element_reduce_policy, std::execution::par, elements_100.begin(), elements_100.end());
//     // benchmark("min_element_reduce_policy_par_unseq_100", min_element_reduce_policy, std::execution::par_unseq, elements_100.begin(), elements_100.end());


//     // benchmark("min_element_reduce_10000", min_element_reduce, elements_10000.begin(), elements_10000.end());
//     // benchmark("min_element_reduce_policy_seq_10000", min_element_reduce_policy, std::execution::seq, elements_10000.begin(), elements_10000.end());
//     // benchmark("min_element_reduce_policy_par_10000", min_element_reduce_policy, std::execution::par, elements_10000.begin(), elements_10000.end());
//     // benchmark("min_element_reduce_policy_par_unseq_10000", min_element_reduce_policy, std::execution::par_unseq, elements_10000.begin(), elements_10000.end());


//     // benchmark("min_element_reduce_1000000", min_element_reduce, elements_1000000.begin(), elements_1000000.end());
//     // benchmark("min_element_reduce_policy_seq_1000000", min_element_reduce_policy, std::execution::seq, elements_1000000.begin(), elements_1000000.end());
//     // benchmark("min_element_reduce_policy_par_1000000", min_element_reduce_policy, std::execution::par, elements_1000000.begin(), elements_1000000.end());
//     // benchmark("min_element_reduce_policy_parunseq_1000000", min_element_reduce_policy, std::execution::par_unseq, elements_1000000.begin(), elements_1000000.end());


//     // benchmark("min_element_100000000", min_element, elements_100000000.begin(), elements_100000000.end());
//     // benchmark("min_element_value_assign_100000000", min_element_value_assign, elements_100000000.begin(), elements_100000000.end());
//     // benchmark("min_element_accumulate_100000000", min_element_accumulate, elements_100000000.begin(), elements_100000000.end());
//     // benchmark("min_element_reduce_100000000", min_element_reduce, elements_100000000.begin(), elements_100000000.end());
//     // benchmark("min_element_reduce_policy_seq_100000000", min_element_reduce_policy, std::execution::seq, elements_100000000.begin(), elements_100000000.end());
//     // benchmark("min_element_reduce_policy_par_100000000", min_element_reduce_policy, std::execution::par, elements_100000000.begin(), elements_100000000.end());
//     // benchmark("min_element_reduce_policy_par_unseq_100000000", min_element_reduce_policy, std::execution::par_unseq, elements_100000000.begin(), elements_100000000.end());
// }

