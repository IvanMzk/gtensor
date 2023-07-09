#include <tuple>
#include <vector>
#include <iomanip>
#include "catch.hpp"
#include "random.hpp"
#include "helpers_for_testing.hpp"

TEMPLATE_TEST_CASE("test_random_integers","[test_random]",
    int,
    std::int64_t
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using bit_generator_type = std::mt19937_64;
    using gtensor::make_rng;
    using helpers_for_testing::apply_by_element;

    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template integers<value_type>(std::declval<int>(),std::declval<int>(),std::declval<int>(),std::declval<bool>()))::value_type,
            value_type
        >
    );
    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template integers<value_type>(std::declval<int>(),std::declval<int>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,
            value_type
        >
    );

    //0seeds,1low,2high,3size,4end_point,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(1,2,3),0,5,50,false,tensor_type{2,3,2,2,1,3,0,3,0,0,2,4,0,0,1,3,2,0,1,2,2,4,4,3,2,3,4,0,2,4,2,2,3,4,4,3,4,2,1,1,2,3,2,1,1,3,3,3,0,4}),
        std::make_tuple(std::make_tuple(3,2,1),0,5,50,false,tensor_type{3,0,1,1,4,1,2,2,2,0,4,2,0,4,3,0,4,4,1,4,0,1,1,0,1,1,3,2,0,4,4,4,2,2,0,2,2,2,3,2,4,1,2,4,1,4,0,0,3,2}),
        std::make_tuple(std::make_tuple(1,2,3),-3,3,50,false,tensor_type{-1,2,0,-1,-1,-2,0,-3,0,-3,-3,-1,1,-3,2,-3,-2,0,-1,-3,2,2,-2,-1,-1,2,2,1,1,0,2,-1,0,1,-3,-1,1,-1,-1,0,1,1,0,1,-1,2,-2,-2,2,-1}),
        std::make_tuple(std::make_tuple(1,2,3),-3,3,std::vector<int>{5,7},true,tensor_type{{-1,3,3,2,0,-1,-1},{-2,0,-3,0,3,-3,-3},{-1,1,-3,3,2,-3,-2},{0,-1,3,-3,2,2,-2},{-1,-1,2,2,1,1,0}})
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto low = std::get<1>(t);
        auto high = std::get<2>(t);
        auto size = std::get<3>(t);
        auto end_point = std::get<4>(t);
        auto expected = std::get<5>(t);

        auto rng_maker = [](const auto&...seeds_){
            return make_rng<bit_generator_type,config_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.template integers<value_type>(low,high,size,end_point);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_random_random","[test_random]",
    float,
    double
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using bit_generator_type = std::mt19937_64;
    using gtensor::tensor_close;
    using gtensor::make_rng;
    using helpers_for_testing::apply_by_element;

    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template random<value_type>(std::declval<int>(),std::declval<int>(),std::declval<int>()))::value_type,
            value_type
        >
    );
    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template random<value_type>(std::declval<int>(),std::declval<int>(),std::declval<std::vector<int>>()))::value_type,
            value_type
        >
    );

    //0seeds,1low,2high,3size,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(1,2,3),0,1,20,tensor_type{0.0993,0.238,0.124,0.245,0.168,0.0132,0.0589,0.0691,0.681,0.999,0.344,0.35,0.345,0.302,0.373,0.736,0.276,0.515,0.01,0.712}),
        std::make_tuple(std::make_tuple(3,2,1),0,1,20,tensor_type{0.447,0.949,0.144,0.546,0.0925,0.222,0.586,0.0271,0.372,0.653,0.0854,0.454,0.419,0.496,0.841,0.943,0.85,0.172,0.322,0.178}),
        std::make_tuple(std::make_tuple(1,2,3),-0.5,0.5,20,tensor_type{-0.401,-0.262,-0.376,-0.255,-0.332,-0.487,-0.441,-0.431,0.181,0.499,-0.156,-0.15,-0.155,-0.198,-0.127,0.236,-0.224,0.0148,-0.49,0.212}),
        std::make_tuple(std::make_tuple(1,2,3),0,1,std::vector<int>{4,5},tensor_type{{0.0993,0.238,0.124,0.245,0.168},{0.0132,0.0589,0.0691,0.681,0.999},{0.344,0.35,0.345,0.302,0.373},{0.736,0.276,0.515,0.01,0.712}})
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto low = std::get<1>(t);
        auto high = std::get<2>(t);
        auto size = std::get<3>(t);
        auto expected = std::get<4>(t);

        auto rng_maker = [](const auto&...seeds_){
            return make_rng<bit_generator_type,config_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.template random<value_type>(low,high,size);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_random_binomial","[test_random]",
    int,
    std::int64_t
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using bit_generator_type = std::mt19937_64;
    using gtensor::make_rng;
    using helpers_for_testing::apply_by_element;

    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template binomial<value_type>(std::declval<int>(),std::declval<double>(),std::declval<int>()))::value_type,
            value_type
        >
    );
    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template binomial<value_type>(std::declval<int>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
            value_type
        >
    );

    //0seeds,1n,2p,3size,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(1,2,3),10,0,50,tensor_type{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}),
        std::make_tuple(std::make_tuple(3,2,1),0,1,50,tensor_type{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}),
        //successes number in sequence of 10 trials, given probability of success of single trial
        std::make_tuple(std::make_tuple(1,2,3),10,0.1,50,tensor_type{1,1,1,1,1,1,1,1,0,5,1,1,1,1,1,2,1,0,1,0,1,1,2,0,0,0,3,2,2,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1,0,0,2,0,0}),
        std::make_tuple(std::make_tuple(1,2,3),10,0.5,50,tensor_type{5,5,5,5,5,5,5,5,3,10,4,4,4,4,4,3,4,6,5,3,4,4,7,6,6,3,8,3,7,6,6,3,6,5,4,3,4,6,5,6,5,5,4,5,5,4,4,7,3,6}),
        std::make_tuple(std::make_tuple(3,2,1),10,0.1,50,tensor_type{0,3,1,0,1,1,0,1,1,0,1,0,0,0,2,3,2,1,1,1,0,2,3,1,1,2,1,1,1,1,2,0,0,1,2,3,1,0,3,2,1,1,3,1,0,1,0,1,0,0}),
        std::make_tuple(std::make_tuple(3,2,1),10,0.5,50,tensor_type{4,8,5,6,5,5,6,5,4,6,5,6,4,6,7,8,7,5,4,5,3,2,8,5,5,2,5,4,4,5,2,6,3,5,2,8,5,4,8,7,4,5,8,5,6,5,4,4,4,6}),
        std::make_tuple(std::make_tuple(1,2,3),10,0.3,std::vector<int>{7,7},tensor_type{{3,3,3,3,3,3,3},{3,4,8,2,2,2,2},{2,1,2,4,3,1,2},{2,1,2,4,4,6,1},{5,4,4,1,4,3,2},{1,2,2,3,2,3,3},{2,3,3,2,2,5,4}})
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto n = std::get<1>(t);
        auto p = std::get<2>(t);
        auto size = std::get<3>(t);
        auto expected = std::get<4>(t);

        auto rng_maker = [](const auto&...seeds_){
            return make_rng<bit_generator_type,config_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.template binomial<value_type>(n,p,size);
        REQUIRE(result==expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_random_negative_binomial","[test_random]",
    int,
    std::int64_t
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using bit_generator_type = std::mt19937_64;
    using gtensor::make_rng;
    using helpers_for_testing::apply_by_element;

    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template negative_binomial<value_type>(std::declval<int>(),std::declval<double>(),std::declval<int>()))::value_type,
            value_type
        >
    );
    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template negative_binomial<value_type>(std::declval<int>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
            value_type
        >
    );

    //0seeds,1k,2p,3size,5expected
    auto test_data = std::make_tuple(
        //geometric - failures before single success
        std::make_tuple(std::make_tuple(1,2,3),1,0.1,50,tensor_type{0,4,0,0,10,19,2,2,5,7,13,0,5,0,1,27,16,4,1,10,1,5,1,3,1,5,10,10,4,4,18,4,1,6,13,2,0,6,11,1,6,9,1,16,6,22,47,9,5,2}),
        std::make_tuple(std::make_tuple(1,2,3),1,0.3,50,tensor_type{0,0,0,0,0,0,0,0,8,1,14,4,1,0,1,0,5,6,0,9,3,0,1,2,0,0,1,4,1,16,3,4,6,4,0,3,1,2,4,0,1,1,3,1,1,0,1,0,1,5}),
        std::make_tuple(std::make_tuple(1,2,3),1,0.7,50,tensor_type{0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,3,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0}),
        std::make_tuple(std::make_tuple(3,2,1),1,0.1,50,tensor_type{4,2,2,12,0,11,6,1,6,3,5,15,15,12,15,13,3,15,7,2,2,1,16,3,0,15,3,19,6,1,5,3,22,11,14,13,18,30,20,5,20,13,6,4,15,12,31,7,19,4}),
        std::make_tuple(std::make_tuple(3,2,1),1,0.3,50,tensor_type{2,1,0,1,2,6,1,3,0,1,0,0,0,3,2,4,1,1,1,4,3,2,2,3,2,8,5,4,2,1,4,3,3,5,0,1,6,0,0,11,0,2,4,6,1,1,4,2,1,0}),
        std::make_tuple(std::make_tuple(3,2,1),1,0.7,50,tensor_type{0,1,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,2,0,1,0,0,0,1,0,0,2,0,2,0,1,0,0,0,0,0,0,0,1,0,2,0,0,0,0,0,0,3,0,0}),
        std::make_tuple(std::make_tuple(1,2,3),1,0.5,std::vector<int>{7,7},tensor_type{{0,0,0,0,0,0,0},{0,2,0,0,0,0,1},{1,1,0,1,9,0,1},{0,0,0,0,0,0,0},{0,0,0,3,1,2,0},{0,0,9,1,1,0,0},{0,0,1,0,0,1,1}}),
        //failures before two successes
        std::make_tuple(std::make_tuple(1,2,3),2,0.1,50,tensor_type{4,0,29,4,12,13,5,28,20,11,6,4,6,20,8,22,7,15,6,12,15,17,28,56,7,10,30,34,11,28,11,14,10,6,14,6,11,9,9,2,4,14,1,15,11,15,3,11,4,5}),
        std::make_tuple(std::make_tuple(1,2,3),2,0.3,50,tensor_type{0,0,0,0,9,18,1,1,11,9,3,3,0,5,17,7,10,3,3,4,2,4,1,1,6,4,2,0,8,5,3,0,10,1,4,6,12,2,11,5,2,6,6,3,3,6,12,4,3,3}),
        std::make_tuple(std::make_tuple(1,2,3),2,0.7,50,tensor_type{0,0,0,0,1,0,0,1,1,1,0,3,1,1,0,0,0,0,0,1,1,1,0,1,0,2,0,2,0,0,1,0,0,1,0,2,0,1,1,1,2,1,2,1,1,1,2,2,1,1}),
        //failures before three successes
        std::make_tuple(std::make_tuple(1,2,3),3,0.1,50,tensor_type{6,27,30,23,16,1,39,49,11,52,24,60,27,8,12,15,10,15,23,16,60,17,12,11,6,30,53,45,44,15,19,33,44,18,7,27,45,7,16,21,18,30,15,8,26,29,17,39,27,28}),
        std::make_tuple(std::make_tuple(1,2,3),3,0.3,50,tensor_type{0,0,8,19,2,11,12,3,1,21,13,7,7,2,5,1,6,6,6,7,3,10,5,6,14,11,5,8,6,6,11,11,6,3,29,6,8,4,0,7,2,14,4,1,3,7,8,2,3,8}),
        std::make_tuple(std::make_tuple(1,2,3),3,0.7,50,tensor_type{0,0,0,1,1,1,1,3,1,1,0,0,1,1,1,1,2,0,2,0,1,0,1,2,0,2,3,1,2,2,3,2,2,2,0,0,2,0,0,3,1,2,0,0,0,4,2,0,2,2})
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto k = std::get<1>(t);
        auto p = std::get<2>(t);
        auto size = std::get<3>(t);
        auto expected = std::get<4>(t);

        auto rng_maker = [](const auto&...seeds_){
            return make_rng<bit_generator_type,config_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.template negative_binomial<value_type>(k,p,size);
        REQUIRE(result==expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_random_poisson","[test_random]",
    int,
    std::int64_t
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using bit_generator_type = std::mt19937_64;
    using gtensor::make_rng;
    using helpers_for_testing::apply_by_element;

    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template poisson<value_type>(std::declval<double>(),std::declval<int>()))::value_type,
            value_type
        >
    );
    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template poisson<value_type>(std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
            value_type
        >
    );

    //0seeds,1mean,2size,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(1,2,3),1,50,tensor_type{0,0,0,0,0,0,0,0,2,0,0,0,1,0,1,1,1,1,4,2,0,1,1,0,1,0,1,0,1,2,0,1,2,1,0,2,3,1,0,3,0,1,1,0,0,0,1,1,2,1}),
        std::make_tuple(std::make_tuple(1,2,3),1.5,50,tensor_type{0,1,1,0,0,0,3,1,2,1,2,3,4,2,2,1,1,0,1,0,1,3,2,2,1,4,3,0,3,0,1,2,0,0,1,3,0,1,3,1,2,2,4,2,0,4,3,2,3,0}),
        std::make_tuple(std::make_tuple(1,2,3),2.0,50,tensor_type{0,1,1,0,0,0,3,1,2,1,2,5,4,1,3,1,0,1,0,3,1,4,1,0,6,2,3,0,1,2,0,0,1,4,1,4,2,2,5,3,4,4,2,3,3,0,1,2,4,1}),
        std::make_tuple(std::make_tuple(3,2,1),2.0,50,tensor_type{2,1,1,0,2,2,3,1,3,0,1,1,2,1,2,3,0,1,1,1,2,1,3,2,2,3,2,3,4,3,2,3,2,1,3,1,2,3,3,2,2,3,1,1,2,2,2,2,2,0}),
        std::make_tuple(std::make_tuple(1,2,3),2.0,std::vector<int>{7,7},tensor_type{{0,1,1,0,0,0,3},{1,2,1,2,5,4,1},{3,1,0,1,0,3,1},{4,1,0,6,2,3,0},{1,2,0,0,1,4,1},{4,2,2,5,3,4,4},{2,3,3,0,1,2,4}})
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto mean = std::get<1>(t);
        auto size = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto rng_maker = [](const auto&...seeds_){
            return make_rng<bit_generator_type,config_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.template poisson<value_type>(mean,size);
        REQUIRE(result==expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_random_exponential","[test_random]",
    float,
    double
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using bit_generator_type = std::mt19937_64;
    using gtensor::tensor_close;
    using gtensor::make_rng;
    using helpers_for_testing::apply_by_element;

    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template exponential<value_type>(std::declval<double>(),std::declval<int>()))::value_type,
            value_type
        >
    );
    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template exponential<value_type>(std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
            value_type
        >
    );

    //0seeds,1lambda,2size,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(1,2,3),0.1,20,tensor_type{1.05,2.72,1.32,2.8,1.84,0.133,0.607,0.716,11.4,74.1,4.22,4.3,4.23,3.6,4.67,13.3,3.23,7.23,0.101,12.5}),
        std::make_tuple(std::make_tuple(1,2,3),0.5,20,tensor_type{0.209,0.545,0.264,0.561,0.369,0.0266,0.121,0.143,2.29,14.8,0.843,0.86,0.846,0.719,0.934,2.67,0.646,1.45,0.0201,2.49}),
        std::make_tuple(std::make_tuple(1,2,3),1.0,20,tensor_type{0.105,0.272,0.132,0.28,0.184,0.0133,0.0607,0.0716,1.14,7.41,0.422,0.43,0.423,0.36,0.467,1.33,0.323,0.723,0.0101,1.25}),
        std::make_tuple(std::make_tuple(3,2,1),1.0,std::vector<int>{4,5},tensor_type{{0.592,2.97,0.156,0.789,0.0971},{0.252,0.881,0.0274,0.466,1.06},{0.0893,0.605,0.543,0.685,1.84},{2.87,1.9,0.188,0.389,0.196}})
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto lambda = std::get<1>(t);
        auto size = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto rng_maker = [](const auto&...seeds_){
            return make_rng<bit_generator_type,config_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.template exponential<value_type>(lambda,size);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_random_gamma","[test_random]",
    float,
    double
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using bit_generator_type = std::mt19937_64;
    using gtensor::tensor_close;
    using gtensor::make_rng;
    using helpers_for_testing::apply_by_element;

    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template gamma<value_type>(std::declval<double>(),std::declval<double>(),std::declval<int>()))::value_type,
            value_type
        >
    );
    REQUIRE(
        std::is_same_v<
            typename decltype(make_rng<bit_generator_type,config_type>().template gamma<value_type>(std::declval<double>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
            value_type
        >
    );

    //0seeds,1shape,2scale,3size,4expected
    auto test_data = std::make_tuple(
        //shape 1
        //rate 0.1
        std::make_tuple(std::make_tuple(1,2,3),1,10,20,tensor_type{1.05,2.72,1.32,2.8,1.84,0.133,0.607,0.716,11.4,74.1,4.22,4.3,4.23,3.6,4.67,13.3,3.23,7.23,0.101,12.5}),
        std::make_tuple(std::make_tuple(3,2,1),1,10,20,tensor_type{5.92,29.7,1.56,7.89,0.971,2.52,8.81,0.274,4.66,10.6,0.893,6.05,5.43,6.85,18.4,28.7,19.0,1.88,3.89,1.96}),
        //rate 0.5
        std::make_tuple(std::make_tuple(1,2,3),1,2,20,tensor_type{0.209,0.545,0.264,0.561,0.369,0.0266,0.121,0.143,2.29,14.8,0.843,0.86,0.846,0.719,0.934,2.67,0.646,1.45,0.0201,2.49}),
        std::make_tuple(std::make_tuple(3,2,1),1,2,20,tensor_type{1.18,5.94,0.312,1.58,0.194,0.503,1.76,0.0549,0.932,2.12,0.179,1.21,1.09,1.37,3.67,5.73,3.8,0.377,0.777,0.392}),
        //shape 2
        //rate 0.1
        std::make_tuple(std::make_tuple(1,2,3),2,10,20,tensor_type{2.47,2.52,3.99,1.91,28.9,15.5,29.5,13.3,12.1,4.77,7.74,6.96,5.21,17.8,5.82,38.9,5.59,14.2,19.8,34.0}),
        //rate 0.5
        std::make_tuple(std::make_tuple(1,2,3),2,2,20,tensor_type{0.495,0.504,0.797,0.382,5.79,3.11,5.89,2.66,2.42,0.955,1.55,1.39,1.04,3.57,1.16,7.79,1.12,2.84,3.96,6.8}),
        //rate 1
        std::make_tuple(std::make_tuple(1,2,3),2,1,std::vector<int>{4,5},tensor_type{{0.247,0.252,0.399,0.191,2.89},{1.55,2.95,1.33,1.21,0.477},{0.774,0.696,0.521,1.78,0.582},{3.89,0.559,1.42,1.98,3.4}})
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto shape = std::get<1>(t);
        auto scale = std::get<2>(t);
        auto size = std::get<3>(t);
        auto expected = std::get<4>(t);

        auto rng_maker = [](const auto&...seeds_){
            return make_rng<bit_generator_type,config_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.template gamma<value_type>(shape,scale,size);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

