#include <tuple>
#include <vector>
#include <numeric>
#include <execution>
#include <algorithm>
#include "tensor.hpp"
#include "reduce.hpp"
#include "builder.hpp"
#include "random.hpp"
#include "../benchmark_helpers.hpp"
#include "../helpers_for_testing.hpp"

TEST_CASE("test_permuted","[benchmark_random]")
{
    using gtensor::arange;
    using gtensor::default_rng;

    auto rng = default_rng();
    auto t = arange<int>(1000);
    auto permuted = rng.permuted(t,0);
    //auto permuted_1 = rng.permuted_1(t,0);
    auto permuted_2 = rng.permuted_2(t,0);


    REQUIRE(std::is_permutation(t.begin(),t.end(),permuted.begin()));
    //REQUIRE(std::is_permutation(t.begin(),t.end(),permuted_1.begin()));
    REQUIRE(std::is_permutation(t.begin(),t.end(),permuted_2.begin()));
}

TEST_CASE("benchmark_random_permuted","[benchmark_random]")
{
    using gtensor::arange;
    using gtensor::default_rng;
    using benchmark_helpers::benchmark;

    auto rng = default_rng();

    auto bench_permuted = [&rng](const auto& t, const auto& axis){
        return rng.permuted(t,axis);
    };
    auto bench_permuted_1 = [&rng](const auto& t, const auto& axis){
        return rng.permuted_1(t,axis);
    };
    auto bench_permuted_2 = [&rng](const auto& t, const auto& axis){
        return rng.permuted_2(t,axis);
    };

    auto t_1000 = arange<int>(1000000);
    benchmark("permuted_1000", bench_permuted, t_1000, 0);
    //benchmark("permuted_1_1000", bench_permuted_1, t_1000, 0);
    benchmark("permuted_2_1000", bench_permuted_2, t_1000, 0);

    auto t_1000_1000 = arange<int>(1000000).reshape(10,100,1000);
    benchmark("permuted_1000_1000", bench_permuted, t_1000_1000, 2);
    //benchmark("permuted_1_1000_1000", bench_permuted_1, t_1000_1000, 2);
    benchmark("permuted_2_1000_1000", bench_permuted_2, t_1000_1000, 2);



    // auto t_1000000 = arange<int>(1000000);
    // benchmark("permuted_1000000", bench_permuted, t_1000000, 0);
    // benchmark("permuted_1_1000000", bench_permuted_1, t_1000000, 0);
}

TEST_CASE("benchmark_random_shuffle_1d","[benchmark_random]")
{
    using gtensor::arange;
    using gtensor::default_rng;
    using benchmark_helpers::benchmark;

    auto rng = default_rng();

    auto bench_shuffle = [&rng](auto& t, const auto& axis){
        return rng.shuffle(t,axis);
    };
    auto bench_shuffle_1 = [&rng](auto& t, const auto& axis){
        return rng.shuffle_1(t,axis);
    };

    auto t_1000 = arange<int>(1000000).reshape(1000,1000).transpose().reshape(-1);
    benchmark("shuffle_1000", bench_shuffle, t_1000, 0);
    benchmark("shuffle_1_1000", bench_shuffle_1, t_1000, 0);
}

