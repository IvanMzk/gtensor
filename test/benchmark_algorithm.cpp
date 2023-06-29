#include <tuple>
#include <vector>
#include <numeric>
#include <execution>
#include <algorithm>
#include "benchmark_helpers.hpp"
#include "helpers_for_testing.hpp"

namespace benchmark_algorithm{

auto min_element = [](auto first, auto last){
    return *std::min_element(first,last);
};

auto min_element_value_assign = [](auto first, auto last){
    auto res = *first;
    for(++first;first!=last;++first){
        const auto& e = *first;
        if (e<res){
            res = e;
        }
    }
    return res;
};

auto min_element_accumulate = [](auto first, auto last){
    auto init = *first;
    return std::accumulate(
        ++first,
        last,
        init,
        [](const auto& res, const auto& e){
            if (e<res){
                return e;
            }else{
                return res;
            }
        }
    );
};

auto min_element_reduce = [](auto first, auto last){
    auto init = *first;
    return std::reduce(
        ++first,
        last,
        init,
        [](const auto& res, const auto& e){
            if (e<res){
                return e;
            }else{
                return res;
            }
        }
    );
};

auto min_element_reduce_policy = [](auto policy, auto first, auto last){
    auto init = *first;
    return std::reduce(
        policy,
        ++first,
        last,
        init,
        [](const auto& res, const auto& e){
            if (e<res){
                return e;
            }else{
                return res;
            }
        }
    );
};

};

TEST_CASE("test_algorithm","[benchmark_algorithm]")
{
    using helpers_for_testing::apply_by_element;
    using value_type = double;

    //0elements,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::vector<value_type>{3}, 3.0),
        std::make_tuple(std::vector<value_type>{3,2,0,1,-1,4,8,1}, -1.0),
        std::make_tuple(std::vector<value_type>{-2,3,2,0,1,-1,4,8,1}, -2.0),
        std::make_tuple(std::vector<value_type>{-2,3,2,0,1,-1,4,8,1,-8}, -8.0),
        std::make_tuple(std::vector<value_type>{3,3,3,3,3,3,3}, 3.0)
    );
    SECTION("test_min_element")
    {
        auto test = [](const auto& t){
            auto elements = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = benchmark_algorithm::min_element(elements.begin(),elements.end());
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_min_element_accumulate")
    {
        auto test = [](const auto& t){
            auto elements = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = benchmark_algorithm::min_element_accumulate(elements.begin(),elements.end());
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_min_element_value_assign")
    {
        auto test = [](const auto& t){
            auto elements = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = benchmark_algorithm::min_element_value_assign(elements.begin(),elements.end());
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_min_element_reduce")
    {
        auto test = [](const auto& t){
            auto elements = std::get<0>(t);
            auto expected = std::get<1>(t);
            auto result = benchmark_algorithm::min_element_reduce(elements.begin(),elements.end());
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("benchmark_algorithm","[benchmark_algorithm]")
{
    using value_type = double;
    using benchmark_helpers::benchmark;
    using benchmark_algorithm::min_element;
    using benchmark_algorithm::min_element_value_assign;
    using benchmark_algorithm::min_element_accumulate;
    using benchmark_algorithm::min_element_reduce;

    std::vector<value_type> elements_100(100,1);
    std::vector<value_type> elements_10000(10000,2);
    std::vector<value_type> elements_1000000(1000000,3);
    std::vector<value_type> elements_100000000(100000000,4);

    benchmark("min_element_100", min_element, elements_100.begin(), elements_100.end());
    benchmark("min_element_value_assign_100", min_element_value_assign, elements_100.begin(), elements_100.end());
    benchmark("min_element_accumulate_100", min_element_accumulate, elements_100.begin(), elements_100.end());
    benchmark("min_element_reduce_100", min_element_reduce, elements_100.begin(), elements_100.end());

    benchmark("min_element_10000", min_element, elements_10000.begin(), elements_10000.end());
    benchmark("min_element_value_assign_10000", min_element_value_assign, elements_10000.begin(), elements_10000.end());
    benchmark("min_element_accumulate_10000", min_element_accumulate, elements_10000.begin(), elements_10000.end());
    benchmark("min_element_reduce_10000", min_element_reduce, elements_10000.begin(), elements_10000.end());

    benchmark("min_element_1000000", min_element, elements_1000000.begin(), elements_1000000.end());
    benchmark("min_element_value_assign_1000000", min_element_value_assign, elements_1000000.begin(), elements_1000000.end());
    benchmark("min_element_accumulate_1000000", min_element_accumulate, elements_1000000.begin(), elements_1000000.end());
    benchmark("min_element_reduce_1000000", min_element_reduce, elements_1000000.begin(), elements_1000000.end());

    benchmark("min_element_100000000", min_element, elements_100000000.begin(), elements_100000000.end());
    benchmark("min_element_value_assign_100000000", min_element_value_assign, elements_100000000.begin(), elements_100000000.end());
    benchmark("min_element_accumulate_100000000", min_element_accumulate, elements_100000000.begin(), elements_100000000.end());
    benchmark("min_element_reduce_100000000", min_element_reduce, elements_100000000.begin(), elements_100000000.end());
}

TEST_CASE("benchmark_algorithm_min_element_reduce_polcy","[benchmark_algorithm]")
{
    using value_type = double;
    using benchmark_helpers::benchmark;
    using benchmark_algorithm::min_element;
    using benchmark_algorithm::min_element_value_assign;
    using benchmark_algorithm::min_element_accumulate;
    using benchmark_algorithm::min_element_reduce;
    using benchmark_algorithm::min_element_reduce_policy;

    std::vector<value_type> elements_100(100,1);
    std::vector<value_type> elements_10000(10000,2);
    std::vector<value_type> elements_1000000(1000000,3);
    std::vector<value_type> elements_100000000(100000000,4);


    // benchmark("min_element_reduce_100", min_element_reduce, elements_100.begin(), elements_100.end());
    // benchmark("min_element_reduce_policy_seq_100", min_element_reduce_policy, std::execution::seq, elements_100.begin(), elements_100.end());
    // benchmark("min_element_reduce_policy_par_100", min_element_reduce_policy, std::execution::par, elements_100.begin(), elements_100.end());
    // benchmark("min_element_reduce_policy_par_unseq_100", min_element_reduce_policy, std::execution::par_unseq, elements_100.begin(), elements_100.end());


    // benchmark("min_element_reduce_10000", min_element_reduce, elements_10000.begin(), elements_10000.end());
    // benchmark("min_element_reduce_policy_seq_10000", min_element_reduce_policy, std::execution::seq, elements_10000.begin(), elements_10000.end());
    // benchmark("min_element_reduce_policy_par_10000", min_element_reduce_policy, std::execution::par, elements_10000.begin(), elements_10000.end());
    // benchmark("min_element_reduce_policy_par_unseq_10000", min_element_reduce_policy, std::execution::par_unseq, elements_10000.begin(), elements_10000.end());


    // benchmark("min_element_reduce_1000000", min_element_reduce, elements_1000000.begin(), elements_1000000.end());
    // benchmark("min_element_reduce_policy_seq_1000000", min_element_reduce_policy, std::execution::seq, elements_1000000.begin(), elements_1000000.end());
    // benchmark("min_element_reduce_policy_par_1000000", min_element_reduce_policy, std::execution::par, elements_1000000.begin(), elements_1000000.end());
    // benchmark("min_element_reduce_policy_parunseq_1000000", min_element_reduce_policy, std::execution::par_unseq, elements_1000000.begin(), elements_1000000.end());


    // benchmark("min_element_100000000", min_element, elements_100000000.begin(), elements_100000000.end());
    // benchmark("min_element_value_assign_100000000", min_element_value_assign, elements_100000000.begin(), elements_100000000.end());
    // benchmark("min_element_accumulate_100000000", min_element_accumulate, elements_100000000.begin(), elements_100000000.end());
    // benchmark("min_element_reduce_100000000", min_element_reduce, elements_100000000.begin(), elements_100000000.end());
    // benchmark("min_element_reduce_policy_seq_100000000", min_element_reduce_policy, std::execution::seq, elements_100000000.begin(), elements_100000000.end());
    // benchmark("min_element_reduce_policy_par_100000000", min_element_reduce_policy, std::execution::par, elements_100000000.begin(), elements_100000000.end());
    // benchmark("min_element_reduce_policy_par_unseq_100000000", min_element_reduce_policy, std::execution::par_unseq, elements_100000000.begin(), elements_100000000.end());
}