#include <tuple>
#include <vector>
#include <iomanip>
#include "catch.hpp"
#include "random.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"

// TEMPLATE_TEST_CASE("test_random_integers","[test_random]",
//     int,
//     std::int64_t
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template integers<value_type>(std::declval<int>(),std::declval<int>(),std::declval<int>(),std::declval<bool>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template integers<value_type>(std::declval<int>(),std::declval<int>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1low,2high,3size,4end_point,5expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),0,5,50,false,tensor_type{2,3,2,2,1,3,0,3,0,0,2,4,0,0,1,3,2,0,1,2,2,4,4,3,2,3,4,0,2,4,2,2,3,4,4,3,4,2,1,1,2,3,2,1,1,3,3,3,0,4}),
//         std::make_tuple(std::make_tuple(3,2,1),0,5,50,false,tensor_type{3,0,1,1,4,1,2,2,2,0,4,2,0,4,3,0,4,4,1,4,0,1,1,0,1,1,3,2,0,4,4,4,2,2,0,2,2,2,3,2,4,1,2,4,1,4,0,0,3,2}),
//         std::make_tuple(std::make_tuple(1,2,3),-3,3,50,false,tensor_type{-1,2,0,-1,-1,-2,0,-3,0,-3,-3,-1,1,-3,2,-3,-2,0,-1,-3,2,2,-2,-1,-1,2,2,1,1,0,2,-1,0,1,-3,-1,1,-1,-1,0,1,1,0,1,-1,2,-2,-2,2,-1}),
//         std::make_tuple(std::make_tuple(1,2,3),-3,3,std::vector<int>{5,7},true,tensor_type{{-1,3,3,2,0,-1,-1},{-2,0,-3,0,3,-3,-3},{-1,1,-3,3,2,-3,-2},{0,-1,3,-3,2,2,-2},{-1,-1,2,2,1,1,0}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto low = std::get<1>(t);
//         auto high = std::get<2>(t);
//         auto size = std::get<3>(t);
//         auto end_point = std::get<4>(t);
//         auto expected = std::get<5>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template integers<value_type>(low,high,size,end_point);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_uniform","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template uniform<value_type>(std::declval<double>(),std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template uniform<value_type>(std::declval<double>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1low,2high,3size,5expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),0,1,20,tensor_type{0.0993,0.238,0.124,0.245,0.168,0.0132,0.0589,0.0691,0.681,0.999,0.344,0.35,0.345,0.302,0.373,0.736,0.276,0.515,0.01,0.712}),
//         std::make_tuple(std::make_tuple(3,2,1),0,1,20,tensor_type{0.447,0.949,0.144,0.546,0.0925,0.222,0.586,0.0271,0.372,0.653,0.0854,0.454,0.419,0.496,0.841,0.943,0.85,0.172,0.322,0.178}),
//         std::make_tuple(std::make_tuple(1,2,3),-0.5,0.5,20,tensor_type{-0.401,-0.262,-0.376,-0.255,-0.332,-0.487,-0.441,-0.431,0.181,0.499,-0.156,-0.15,-0.155,-0.198,-0.127,0.236,-0.224,0.0148,-0.49,0.212}),
//         std::make_tuple(std::make_tuple(1,2,3),0,1,std::vector<int>{4,5},tensor_type{{0.0993,0.238,0.124,0.245,0.168},{0.0132,0.0589,0.0691,0.681,0.999},{0.344,0.35,0.345,0.302,0.373},{0.736,0.276,0.515,0.01,0.712}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto low = std::get<1>(t);
//         auto high = std::get<2>(t);
//         auto size = std::get<3>(t);
//         auto expected = std::get<4>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template uniform<value_type>(low,high,size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_random","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template random<value_type>(std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template random<value_type>(std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1size,2expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),20,tensor_type{0.0993,0.238,0.124,0.245,0.168,0.0132,0.0589,0.0691,0.681,0.999,0.344,0.35,0.345,0.302,0.373,0.736,0.276,0.515,0.01,0.712}),
//         std::make_tuple(std::make_tuple(3,2,1),20,tensor_type{0.447,0.949,0.144,0.546,0.0925,0.222,0.586,0.0271,0.372,0.653,0.0854,0.454,0.419,0.496,0.841,0.943,0.85,0.172,0.322,0.178}),
//         std::make_tuple(std::make_tuple(1,2,3),std::vector<int>{4,5},tensor_type{{0.0993,0.238,0.124,0.245,0.168},{0.0132,0.0589,0.0691,0.681,0.999},{0.344,0.35,0.345,0.302,0.373},{0.736,0.276,0.515,0.01,0.712}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto size = std::get<1>(t);
//         auto expected = std::get<2>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template random<value_type>(size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_binomial","[test_random]",
//     int,
//     std::int64_t
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template binomial<value_type>(std::declval<int>(),std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template binomial<value_type>(std::declval<int>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1n,2p,3size,5expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),10,0,50,tensor_type{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}),
//         std::make_tuple(std::make_tuple(3,2,1),0,1,50,tensor_type{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}),
//         //successes number in sequence of 10 trials, given probability of success of single trial
//         std::make_tuple(std::make_tuple(1,2,3),10,0.1,50,tensor_type{1,1,1,1,1,1,1,1,0,5,1,1,1,1,1,2,1,0,1,0,1,1,2,0,0,0,3,2,2,0,0,0,0,1,0,0,0,0,1,0,1,1,0,1,1,0,0,2,0,0}),
//         std::make_tuple(std::make_tuple(1,2,3),10,0.5,50,tensor_type{5,5,5,5,5,5,5,5,3,10,4,4,4,4,4,3,4,6,5,3,4,4,7,6,6,3,8,3,7,6,6,3,6,5,4,3,4,6,5,6,5,5,4,5,5,4,4,7,3,6}),
//         std::make_tuple(std::make_tuple(3,2,1),10,0.1,50,tensor_type{0,3,1,0,1,1,0,1,1,0,1,0,0,0,2,3,2,1,1,1,0,2,3,1,1,2,1,1,1,1,2,0,0,1,2,3,1,0,3,2,1,1,3,1,0,1,0,1,0,0}),
//         std::make_tuple(std::make_tuple(3,2,1),10,0.5,50,tensor_type{4,8,5,6,5,5,6,5,4,6,5,6,4,6,7,8,7,5,4,5,3,2,8,5,5,2,5,4,4,5,2,6,3,5,2,8,5,4,8,7,4,5,8,5,6,5,4,4,4,6}),
//         std::make_tuple(std::make_tuple(1,2,3),10,0.3,std::vector<int>{7,7},tensor_type{{3,3,3,3,3,3,3},{3,4,8,2,2,2,2},{2,1,2,4,3,1,2},{2,1,2,4,4,6,1},{5,4,4,1,4,3,2},{1,2,2,3,2,3,3},{2,3,3,2,2,5,4}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto n = std::get<1>(t);
//         auto p = std::get<2>(t);
//         auto size = std::get<3>(t);
//         auto expected = std::get<4>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template binomial<value_type>(n,p,size);
//         REQUIRE(result==expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_negative_binomial","[test_random]",
//     int,
//     std::int64_t
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template negative_binomial<value_type>(std::declval<int>(),std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template negative_binomial<value_type>(std::declval<int>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1k,2p,3size,5expected
//     auto test_data = std::make_tuple(
//         //geometric - failures before single success
//         std::make_tuple(std::make_tuple(1,2,3),1,0.1,50,tensor_type{0,4,0,0,10,19,2,2,5,7,13,0,5,0,1,27,16,4,1,10,1,5,1,3,1,5,10,10,4,4,18,4,1,6,13,2,0,6,11,1,6,9,1,16,6,22,47,9,5,2}),
//         std::make_tuple(std::make_tuple(1,2,3),1,0.3,50,tensor_type{0,0,0,0,0,0,0,0,8,1,14,4,1,0,1,0,5,6,0,9,3,0,1,2,0,0,1,4,1,16,3,4,6,4,0,3,1,2,4,0,1,1,3,1,1,0,1,0,1,5}),
//         std::make_tuple(std::make_tuple(1,2,3),1,0.7,50,tensor_type{0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,0,1,0,0,3,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0}),
//         std::make_tuple(std::make_tuple(3,2,1),1,0.1,50,tensor_type{4,2,2,12,0,11,6,1,6,3,5,15,15,12,15,13,3,15,7,2,2,1,16,3,0,15,3,19,6,1,5,3,22,11,14,13,18,30,20,5,20,13,6,4,15,12,31,7,19,4}),
//         std::make_tuple(std::make_tuple(3,2,1),1,0.3,50,tensor_type{2,1,0,1,2,6,1,3,0,1,0,0,0,3,2,4,1,1,1,4,3,2,2,3,2,8,5,4,2,1,4,3,3,5,0,1,6,0,0,11,0,2,4,6,1,1,4,2,1,0}),
//         std::make_tuple(std::make_tuple(3,2,1),1,0.7,50,tensor_type{0,1,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,2,0,1,0,0,0,1,0,0,2,0,2,0,1,0,0,0,0,0,0,0,1,0,2,0,0,0,0,0,0,3,0,0}),
//         std::make_tuple(std::make_tuple(1,2,3),1,0.5,std::vector<int>{7,7},tensor_type{{0,0,0,0,0,0,0},{0,2,0,0,0,0,1},{1,1,0,1,9,0,1},{0,0,0,0,0,0,0},{0,0,0,3,1,2,0},{0,0,9,1,1,0,0},{0,0,1,0,0,1,1}}),
//         //failures before two successes
//         std::make_tuple(std::make_tuple(1,2,3),2,0.1,50,tensor_type{4,0,29,4,12,13,5,28,20,11,6,4,6,20,8,22,7,15,6,12,15,17,28,56,7,10,30,34,11,28,11,14,10,6,14,6,11,9,9,2,4,14,1,15,11,15,3,11,4,5}),
//         std::make_tuple(std::make_tuple(1,2,3),2,0.3,50,tensor_type{0,0,0,0,9,18,1,1,11,9,3,3,0,5,17,7,10,3,3,4,2,4,1,1,6,4,2,0,8,5,3,0,10,1,4,6,12,2,11,5,2,6,6,3,3,6,12,4,3,3}),
//         std::make_tuple(std::make_tuple(1,2,3),2,0.7,50,tensor_type{0,0,0,0,1,0,0,1,1,1,0,3,1,1,0,0,0,0,0,1,1,1,0,1,0,2,0,2,0,0,1,0,0,1,0,2,0,1,1,1,2,1,2,1,1,1,2,2,1,1}),
//         //failures before three successes
//         std::make_tuple(std::make_tuple(1,2,3),3,0.1,50,tensor_type{6,27,30,23,16,1,39,49,11,52,24,60,27,8,12,15,10,15,23,16,60,17,12,11,6,30,53,45,44,15,19,33,44,18,7,27,45,7,16,21,18,30,15,8,26,29,17,39,27,28}),
//         std::make_tuple(std::make_tuple(1,2,3),3,0.3,50,tensor_type{0,0,8,19,2,11,12,3,1,21,13,7,7,2,5,1,6,6,6,7,3,10,5,6,14,11,5,8,6,6,11,11,6,3,29,6,8,4,0,7,2,14,4,1,3,7,8,2,3,8}),
//         std::make_tuple(std::make_tuple(1,2,3),3,0.7,50,tensor_type{0,0,0,1,1,1,1,3,1,1,0,0,1,1,1,1,2,0,2,0,1,0,1,2,0,2,3,1,2,2,3,2,2,2,0,0,2,0,0,3,1,2,0,0,0,4,2,0,2,2})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto k = std::get<1>(t);
//         auto p = std::get<2>(t);
//         auto size = std::get<3>(t);
//         auto expected = std::get<4>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template negative_binomial<value_type>(k,p,size);
//         REQUIRE(result==expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_poisson","[test_random]",
//     int,
//     std::int64_t
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template poisson<value_type>(std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template poisson<value_type>(std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1mean,2size,3expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),1,50,tensor_type{0,0,0,0,0,0,0,0,2,0,0,0,1,0,1,1,1,1,4,2,0,1,1,0,1,0,1,0,1,2,0,1,2,1,0,2,3,1,0,3,0,1,1,0,0,0,1,1,2,1}),
//         std::make_tuple(std::make_tuple(1,2,3),1.5,50,tensor_type{0,1,1,0,0,0,3,1,2,1,2,3,4,2,2,1,1,0,1,0,1,3,2,2,1,4,3,0,3,0,1,2,0,0,1,3,0,1,3,1,2,2,4,2,0,4,3,2,3,0}),
//         std::make_tuple(std::make_tuple(1,2,3),2.0,50,tensor_type{0,1,1,0,0,0,3,1,2,1,2,5,4,1,3,1,0,1,0,3,1,4,1,0,6,2,3,0,1,2,0,0,1,4,1,4,2,2,5,3,4,4,2,3,3,0,1,2,4,1}),
//         std::make_tuple(std::make_tuple(3,2,1),2.0,50,tensor_type{2,1,1,0,2,2,3,1,3,0,1,1,2,1,2,3,0,1,1,1,2,1,3,2,2,3,2,3,4,3,2,3,2,1,3,1,2,3,3,2,2,3,1,1,2,2,2,2,2,0}),
//         std::make_tuple(std::make_tuple(1,2,3),2.0,std::vector<int>{7,7},tensor_type{{0,1,1,0,0,0,3},{1,2,1,2,5,4,1},{3,1,0,1,0,3,1},{4,1,0,6,2,3,0},{1,2,0,0,1,4,1},{4,2,2,5,3,4,4},{2,3,3,0,1,2,4}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto mean = std::get<1>(t);
//         auto size = std::get<2>(t);
//         auto expected = std::get<3>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template poisson<value_type>(mean,size);
//         REQUIRE(result==expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_exponential","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template exponential<value_type>(std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template exponential<value_type>(std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1lambda,2size,3expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),0.1,20,tensor_type{1.05,2.72,1.32,2.8,1.84,0.133,0.607,0.716,11.4,74.1,4.22,4.3,4.23,3.6,4.67,13.3,3.23,7.23,0.101,12.5}),
//         std::make_tuple(std::make_tuple(1,2,3),0.5,20,tensor_type{0.209,0.545,0.264,0.561,0.369,0.0266,0.121,0.143,2.29,14.8,0.843,0.86,0.846,0.719,0.934,2.67,0.646,1.45,0.0201,2.49}),
//         std::make_tuple(std::make_tuple(1,2,3),1.0,20,tensor_type{0.105,0.272,0.132,0.28,0.184,0.0133,0.0607,0.0716,1.14,7.41,0.422,0.43,0.423,0.36,0.467,1.33,0.323,0.723,0.0101,1.25}),
//         std::make_tuple(std::make_tuple(3,2,1),1.0,std::vector<int>{4,5},tensor_type{{0.592,2.97,0.156,0.789,0.0971},{0.252,0.881,0.0274,0.466,1.06},{0.0893,0.605,0.543,0.685,1.84},{2.87,1.9,0.188,0.389,0.196}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto lambda = std::get<1>(t);
//         auto size = std::get<2>(t);
//         auto expected = std::get<3>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template exponential<value_type>(lambda,size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_gamma","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template gamma<value_type>(std::declval<double>(),std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template gamma<value_type>(std::declval<double>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1shape,2scale,3size,4expected
//     auto test_data = std::make_tuple(
//         //shape 1
//         //scale 10 (rate 0.1)
//         std::make_tuple(std::make_tuple(1,2,3),1,10,20,tensor_type{1.05,2.72,1.32,2.8,1.84,0.133,0.607,0.716,11.4,74.1,4.22,4.3,4.23,3.6,4.67,13.3,3.23,7.23,0.101,12.5}),
//         std::make_tuple(std::make_tuple(3,2,1),1,10,20,tensor_type{5.92,29.7,1.56,7.89,0.971,2.52,8.81,0.274,4.66,10.6,0.893,6.05,5.43,6.85,18.4,28.7,19.0,1.88,3.89,1.96}),
//         //scale 2 (rate 0.5)
//         std::make_tuple(std::make_tuple(1,2,3),1,2,20,tensor_type{0.209,0.545,0.264,0.561,0.369,0.0266,0.121,0.143,2.29,14.8,0.843,0.86,0.846,0.719,0.934,2.67,0.646,1.45,0.0201,2.49}),
//         std::make_tuple(std::make_tuple(3,2,1),1,2,20,tensor_type{1.18,5.94,0.312,1.58,0.194,0.503,1.76,0.0549,0.932,2.12,0.179,1.21,1.09,1.37,3.67,5.73,3.8,0.377,0.777,0.392}),
//         //shape 2
//         //scale 10 (rate 0.1)
//         std::make_tuple(std::make_tuple(1,2,3),2,10,20,tensor_type{2.47,2.52,3.99,1.91,28.9,15.5,29.5,13.3,12.1,4.77,7.74,6.96,5.21,17.8,5.82,38.9,5.59,14.2,19.8,34.0}),
//         //scale 2 (rate 0.5)
//         std::make_tuple(std::make_tuple(1,2,3),2,2,20,tensor_type{0.495,0.504,0.797,0.382,5.79,3.11,5.89,2.66,2.42,0.955,1.55,1.39,1.04,3.57,1.16,7.79,1.12,2.84,3.96,6.8}),
//         //scale 1 (rate 1)
//         std::make_tuple(std::make_tuple(1,2,3),2,1,std::vector<int>{4,5},tensor_type{{0.247,0.252,0.399,0.191,2.89},{1.55,2.95,1.33,1.21,0.477},{0.774,0.696,0.521,1.78,0.582},{3.89,0.559,1.42,1.98,3.4}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto shape = std::get<1>(t);
//         auto scale = std::get<2>(t);
//         auto size = std::get<3>(t);
//         auto expected = std::get<4>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template gamma<value_type>(shape,scale,size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_weibull","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template weibull<value_type>(std::declval<double>(),std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template weibull<value_type>(std::declval<double>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1shape,2scale,3size,4expected
//     auto test_data = std::make_tuple(
//         //scale 1
//         //shape 0.5
//         std::make_tuple(std::make_tuple(1,2,3),0.5,1,20,tensor_type{0.0109,0.0742,0.0174,0.0786,0.034,0.000177,0.00369,0.00512,1.31,54.9,0.178,0.185,0.179,0.129,0.218,1.78,0.104,0.523,0.000101,1.55}),
//         //shape 1
//         std::make_tuple(std::make_tuple(1,2,3),1,1,20,tensor_type{0.105,0.272,0.132,0.28,0.184,0.0133,0.0607,0.0716,1.14,7.41,0.422,0.43,0.423,0.36,0.467,1.33,0.323,0.723,0.0101,1.25}),
//         //shape 2
//         std::make_tuple(std::make_tuple(1,2,3),2,1,std::vector<int>{4,5},tensor_type{{0.323,0.522,0.363,0.53,0.43},{0.115,0.246,0.268,1.07,2.72},{0.649,0.656,0.65,0.6,0.683},{1.15,0.568,0.85,0.1,1.12}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto shape = std::get<1>(t);
//         auto scale = std::get<2>(t);
//         auto size = std::get<3>(t);
//         auto expected = std::get<4>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template weibull<value_type>(shape,scale,size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_normal","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template normal<value_type>(std::declval<double>(),std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template normal<value_type>(std::declval<double>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1mean,2stdev,3size,4expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),0,1,20,tensor_type{-0.351,-0.229,-0.509,-0.345,-1.32,-1.27,-1.02,-1.3,-0.747,1.39,-1.79,0.118,-1.5,-1.12,1.32,-0.16,1.03,1.54,1.3,0.136}),
//         std::make_tuple(std::make_tuple(3,2,1),0,1,20,tensor_type{-0.0748,0.632,-1.14,0.147,-0.196,-0.133,0.0708,-0.391,-1.23,1.47,-0.846,-0.0939,-2.7,-0.144,0.296,-0.277,-0.536,-0.97,0.216,0.523}),
//         std::make_tuple(std::make_tuple(1,2,3),0,3,20,tensor_type{-1.05,-0.687,-1.53,-1.04,-3.95,-3.81,-3.07,-3.91,-2.24,4.17,-5.36,0.354,-4.5,-3.37,3.97,-0.481,3.09,4.63,3.91,0.407}),
//         std::make_tuple(std::make_tuple(1,2,3),5.1,1.5,std::vector<int>{4,5},tensor_type{{4.57,4.76,4.34,4.58,3.13},{3.2,3.57,3.14,3.98,7.19},{2.42,5.28,2.85,3.42,7.08},{4.86,6.65,7.41,7.06,5.3}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto mean = std::get<1>(t);
//         auto stdev = std::get<2>(t);
//         auto size = std::get<3>(t);
//         auto expected = std::get<4>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template normal<value_type>(mean,stdev,size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_lognormal","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template lognormal<value_type>(std::declval<double>(),std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template lognormal<value_type>(std::declval<double>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1mean,2stdev,3size,4expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),0,1,20,tensor_type{0.704,0.795,0.601,0.708,0.268,0.281,0.359,0.271,0.474,4.02,0.168,1.13,0.223,0.326,3.75,0.852,2.8,4.68,3.68,1.15}),
//         std::make_tuple(std::make_tuple(3,2,1),0,1,20,tensor_type{0.928,1.88,0.319,1.16,0.822,0.875,1.07,0.676,0.293,4.37,0.429,0.91,0.0675,0.866,1.34,0.758,0.585,0.379,1.24,1.69}),
//         std::make_tuple(std::make_tuple(1,2,3),0,3,20,tensor_type{0.349,0.503,0.217,0.355,0.0193,0.0222,0.0464,0.02,0.106,64.8,0.0047,1.42,0.0111,0.0345,52.8,0.618,22.0,102.0,50.0,1.5}),
//         std::make_tuple(std::make_tuple(1,2,3),1,1,std::vector<int>{4,5},tensor_type{{1.91,2.16,1.63,1.92,0.729},{0.764,0.977,0.737,1.29,10.9},{0.455,3.06,0.606,0.885,10.2},{2.32,7.62,12.7,10.0,3.11}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto mean = std::get<1>(t);
//         auto stdev = std::get<2>(t);
//         auto size = std::get<3>(t);
//         auto expected = std::get<4>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template lognormal<value_type>(mean,stdev,size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_chisquare","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template chisquare<value_type>(std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template chisquare<value_type>(std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1df,2size,3expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),1,20,tensor_type{0.0197,0.0306,0.00694,1.05,0.237,0.238,0.278,0.152,0.000201,0.223,2.03,0.804,6.72,2.08,0.667,0.601,0.302,0.407,0.00598,0.053}),
//         std::make_tuple(std::make_tuple(3,2,1),1,20,tensor_type{0.399,0.0417,0.0171,0.277,0.0146,0.351,2.3,2.43,0.207,1.03,0.00556,0.0268,0.148,3.27,1.13,4.01,0.00809,5.21,0.225,0.314}),
//         std::make_tuple(std::make_tuple(1,2,3),2,20,tensor_type{0.209,0.545,0.264,0.561,0.369,0.0266,0.121,0.143,2.29,14.8,0.843,0.86,0.846,0.719,0.934,2.67,0.646,1.45,0.0201,2.49}),
//         std::make_tuple(std::make_tuple(1,2,3),3,std::vector<int>{4,5},tensor_type{{4.2,1.94,4.29,1.56,1.36},{0.618,0.486,2.32,0.294,5.89},{0.255,1.71,2.66,5.06,1.2},{0.699,0.135,2.02,0.873,5.6}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto df = std::get<1>(t);
//         auto size = std::get<2>(t);
//         auto expected = std::get<3>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template chisquare<value_type>(df,size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_cauchy","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template cauchy<value_type>(std::declval<double>(),std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template cauchy<value_type>(std::declval<double>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1location,2scale,3size,4expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),0,1,20,tensor_type{0.322,0.93,0.409,0.966,0.585,0.0416,0.187,0.221,-1.56,-0.00191,1.87,1.96,1.89,1.4,2.37,-1.09,1.18,-21.5,0.0315,-1.27}),
//         std::make_tuple(std::make_tuple(3,2,1),0,1,20,tensor_type{5.94,-0.163,0.487,-6.92,0.299,0.84,-3.63,0.0852,2.36,-1.92,0.275,6.87,3.85,73.6,-0.547,-0.181,-0.509,0.599,1.6,0.626}),
//         std::make_tuple(std::make_tuple(1,2,3),0,2.5,20,tensor_type{0.806,2.33,1.02,2.42,1.46,0.104,0.468,0.551,-3.9,-0.00477,4.68,4.89,4.71,3.49,5.93,-2.72,2.95,-53.8,0.0787,-3.18}),
//         std::make_tuple(std::make_tuple(1,2,3),2.5,1.5,std::vector<int>{4,5},tensor_type{{2.98,3.9,3.11,3.95,3.38},{2.56,2.78,2.83,0.159,2.5},{5.31,5.43,5.33,4.59,6.06},{0.865,4.27,-29.8,2.55,0.594}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto location = std::get<1>(t);
//         auto scale = std::get<2>(t);
//         auto size = std::get<3>(t);
//         auto expected = std::get<4>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template cauchy<value_type>(location,scale,size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_f","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template f<value_type>(std::declval<double>(),std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template f<value_type>(std::declval<double>(),std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1dfnum,2dfden,3size,4expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),1,2,20,tensor_type{0.149,8.98,0.00607,0.679,0.861,2.56,2.11,0.727,2.46,4.62,0.504,0.0134,0.00169,2.71,0.00215,1.25,14.3,1.55,0.57,0.813}),
//         std::make_tuple(std::make_tuple(3,2,1),1,2,20,tensor_type{2.56,3.61,3.11,0.602,1.21,0.301,0.354,0.00392,0.0846,0.0747,0.449,0.195,0.0615,0.471,0.557,0.503,0.000117,6.7,1.98,0.426}),
//         std::make_tuple(std::make_tuple(1,2,3),2,1.5,20,tensor_type{0.513,0.436,3.39,0.965,0.253,0.245,0.856,0.721,0.628,0.703,1.05,1.42,0.042,1.62,0.00926,0.262,0.293,0.98,2.23,9.82}),
//         std::make_tuple(std::make_tuple(1,2,3),2,2,std::vector<int>{4,5},tensor_type{{0.384,0.471,13.8,0.848,0.154},{0.98,1.18,0.35,0.447,0.00809},{0.862,2.76,0.853,2.37,2.27},{0.652,2.82,0.391,0.911,0.0836}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto dfnum = std::get<1>(t);
//         auto dfden = std::get<2>(t);
//         auto size = std::get<3>(t);
//         auto expected = std::get<4>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template f<value_type>(dfnum,dfden,size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_random_t","[test_random]",
//     float,
//     double
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using bit_generator_type = std::mt19937_64;
//     using gtensor::tensor_close;
//     using gtensor::rng;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template t<value_type>(std::declval<double>(),std::declval<int>()))::value_type,
//             value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(rng<bit_generator_type,config_type>().template t<value_type>(std::declval<double>(),std::declval<std::vector<int>>()))::value_type,
//             value_type
//         >
//     );

//     //0seeds,1df,3size,4expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(std::make_tuple(1,2,3),1,20,tensor_type{-2.01,-2.75,-2.7,-2.41,-126.0,0.25,1.47,-0.0619,1.6,0.175,-1.22,19.5,-12.1,-2.62,61.1,0.748,1.32,-9.58,0.727,1.04}),
//         std::make_tuple(std::make_tuple(3,2,1),1,20,tensor_type{-0.367,4.83,0.134,-3.24,-1.78,-0.0925,-0.529,-13.0,-1.44,-0.187,0.218,-8.74,-0.551,-1.21,-41.6,0.716,-0.517,-0.368,-1.01,-0.127}),
//         std::make_tuple(std::make_tuple(1,2,3),2,20,tensor_type{-0.966,-0.432,-2.02,-2.12,-1.31,1.63,-1.15,-1.43,0.562,1.3,1.42,0.119,0.405,-1.42,-11.9,-1.24,-2.45,-1.92,-0.363,1.36}),
//         std::make_tuple(std::make_tuple(1,2,3),2.5,std::vector<int>{4,5},tensor_type{{-0.303,-0.313,2.06,0.237,-3.09},{11.9,-0.46,-0.0303,-1.07,-1.8},{1.95,0.999,-0.779,-3.55,0.973},{0.576,1.03,-0.466,0.483,-0.0886}})
//     );
//     auto test = [](const auto& t){
//         auto seeds = std::get<0>(t);
//         auto df = std::get<1>(t);
//         auto size = std::get<2>(t);
//         auto expected = std::get<3>(t);

//         auto rng_maker = [](const auto&...seeds_){
//             return rng<bit_generator_type,config_type>(seeds_...);
//         };
//         auto rng = std::apply(rng_maker,seeds);
//         auto result = rng.template t<value_type>(df,size);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_random_rng_default_rng","[test_random]")
// {
//     using gtensor::rng;
//     using gtensor::default_rng;

//     //reproducible results using the same seeds
//     REQUIRE(rng<std::mt19937_64>(1,2,3).integers(0,5,50) == rng<std::mt19937_64>(1,2,3).integers(0,5,50));
//     REQUIRE(rng<std::mt19937_64>(3,2,1).normal(0,1,50) == rng<std::mt19937_64>(3,2,1).normal(0,1,50));
//     //another engine
//     REQUIRE(rng<std::mt19937>(4,5,6).integers(0,5,50) == rng<std::mt19937>(4,5,6).integers(0,5,50));
//     REQUIRE(rng<std::mt19937>(6,5,4).normal(0,1,50) == rng<std::mt19937>(6,5,4).normal(0,1,50));

//     REQUIRE(rng<std::minstd_rand>(1,2,3).integers(0,5,50) == rng<std::minstd_rand>(1,2,3).integers(0,5,50));
//     REQUIRE(rng<std::minstd_rand>(1,2,3).normal(0,1,50) == rng<std::minstd_rand>(1,2,3).normal(0,1,50));

//     //default engine
//     REQUIRE(default_rng(1,2,3).integers(0,5,50) == default_rng(1,2,3).integers(0,5,50));
//     REQUIRE(default_rng(3,2,1).normal(0,1,50) == default_rng(3,2,1).normal(0,1,50));

//     //random seeds
//     REQUIRE(rng<std::mt19937_64>().integers(0,5,50) != rng<std::mt19937_64>().integers(0,5,50));
//     REQUIRE(rng<std::mt19937_64>().normal(0,1,50) != rng<std::mt19937_64>().normal(0,1,50));

//     REQUIRE(default_rng().integers(0,5,50) != default_rng().integers(0,5,50));
//     REQUIRE(default_rng().normal(0,1,50) != default_rng().normal(0,1,50));
// }

TEST_CASE("test_random_shuffle","[test_random]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bit_generator_type = std::mt19937_64;
    using gtensor::rng;
    using helpers_for_testing::apply_by_element;

    //0seeds,1tensor,2axis,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{},0,tensor_type{}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{},0,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{}.reshape(0,2,3),0,tensor_type{}.reshape(0,2,3)),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{}.reshape(0,2,3),0,tensor_type{}.reshape(0,2,3)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{3},0,tensor_type{3}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{3},0,tensor_type{3}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{{3}}},0,tensor_type{{{3}}}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{{{3}}},0,tensor_type{{{3}}}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},0,tensor_type{10,14,5,11,6,8,9,1,13,2,12,15,4,7,3}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},0,tensor_type{11,1,4,8,3,14,13,5,2,10,7,9,12,15,6}),
        std::make_tuple(std::make_tuple(4,5,6),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},0,tensor_type{8,1,12,15,9,10,7,3,13,14,4,6,2,11,5}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},0,tensor_type{{1,2,3,4},{5,6,7,8},{13,14,15,16},{17,18,19,20},{9,10,11,12}}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},0,tensor_type{{17,18,19,20},{5,6,7,8},{9,10,11,12},{1,2,3,4},{13,14,15,16}}),
        std::make_tuple(std::make_tuple(1,3,2),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},0,tensor_type{{17,18,19,20},{1,2,3,4},{13,14,15,16},{5,6,7,8},{9,10,11,12}}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},1,tensor_type{{2,1,4,3},{6,5,8,7},{10,9,12,11},{14,13,16,15},{18,17,20,19}}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},1,tensor_type{{1,3,4,2},{5,7,8,6},{9,11,12,10},{13,15,16,14},{17,19,20,18}}),
        std::make_tuple(std::make_tuple(1,3,2),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},1,tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}}),
        std::make_tuple(
            std::make_tuple(1,2,3),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            0,
            tensor_type{{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}}
        ),
        std::make_tuple(
            std::make_tuple(2,3,4),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            0,
            tensor_type{{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}},{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}}
        ),
        std::make_tuple(
            std::make_tuple(1,2,3),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            1,
            tensor_type{{{4,5,6},{1,2,3},{10,11,12},{7,8,9}},{{16,17,18},{13,14,15},{22,23,24},{19,20,21}},{{28,29,30},{25,26,27},{34,35,36},{31,32,33}}}
        ),
        std::make_tuple(
            std::make_tuple(2,3,4),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            1,
            tensor_type{{{7,8,9},{4,5,6},{10,11,12},{1,2,3}},{{19,20,21},{16,17,18},{22,23,24},{13,14,15}},{{31,32,33},{28,29,30},{34,35,36},{25,26,27}}}
        ),
        std::make_tuple(
            std::make_tuple(9,8,7),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            2,
            tensor_type{{{2,3,1},{5,6,4},{8,9,7},{11,12,10}},{{14,15,13},{17,18,16},{20,21,19},{23,24,22}},{{26,27,25},{29,30,28},{32,33,31},{35,36,34}}}
        ),
        std::make_tuple(
            std::make_tuple(7,8,9),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            2,
            tensor_type{{{3,2,1},{6,5,4},{9,8,7},{12,11,10}},{{15,14,13},{18,17,16},{21,20,19},{24,23,22}},{{27,26,25},{30,29,28},{33,32,31},{36,35,34}}}
        )
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto ten = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto rng_maker = [](const auto&...seeds_){
            return rng<bit_generator_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        rng.shuffle(ten,axis);
        REQUIRE(ten == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_random_shuffle_exception","[test_random]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::default_rng;
    using gtensor::random_exception;

    tensor_type t0(1);
    REQUIRE_THROWS_AS(default_rng().shuffle(t0,0),random_exception);
    tensor_type t1{1,2,3,4,5};
    REQUIRE_THROWS_AS(default_rng().shuffle(t1,1),random_exception);
    tensor_type t2{{1,2,3},{4,5,6}};
    REQUIRE_THROWS_AS(default_rng().shuffle(t2,2),random_exception);
}

TEST_CASE("test_random_permutation","[test_random]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bit_generator_type = std::mt19937_64;
    using gtensor::rng;
    using helpers_for_testing::apply_by_element;

    //0seeds,1tensor_or_int,2axis,3expected
    auto test_data = std::make_tuple(
        //integral
        std::make_tuple(std::make_tuple(1,2,3),0,0,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),1,0,tensor_type{0}),
        std::make_tuple(std::make_tuple(1,2,3),10,0,tensor_type{4,8,0,1,7,9,3,5,6,2}),
        std::make_tuple(std::make_tuple(2,3,4),15,0,tensor_type{9,3,10,2,1,4,0,12,6,7,11,13,5,14,8}),
        std::make_tuple(std::make_tuple(2,3,4),15,1,tensor_type{9,3,10,2,1,4,0,12,6,7,11,13,5,14,8}),
        std::make_tuple(std::make_tuple(2,3,4),15,5,tensor_type{9,3,10,2,1,4,0,12,6,7,11,13,5,14,8}),
        //tensor
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{},0,tensor_type{}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{},0,tensor_type{}),
        //std::make_tuple(std::make_tuple(1,2,3),tensor_type(2),0,tensor_type(2)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{3},0,tensor_type{3}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{3},0,tensor_type{3}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},0,tensor_type{10,14,5,11,6,8,9,1,13,2,12,15,4,7,3}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},0,tensor_type{11,1,4,8,3,14,13,5,2,10,7,9,12,15,6}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},0,tensor_type{{1,2,3,4},{5,6,7,8},{13,14,15,16},{17,18,19,20},{9,10,11,12}}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},0,tensor_type{{17,18,19,20},{5,6,7,8},{9,10,11,12},{1,2,3,4},{13,14,15,16}}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},1,tensor_type{{2,1,4,3},{6,5,8,7},{10,9,12,11},{14,13,16,15},{18,17,20,19}}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},1,tensor_type{{1,3,4,2},{5,7,8,6},{9,11,12,10},{13,15,16,14},{17,19,20,18}}),
        std::make_tuple(
            std::make_tuple(1,2,3),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            0,
            tensor_type{{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}}
        ),
        std::make_tuple(
            std::make_tuple(1,2,3),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            1,
            tensor_type{{{4,5,6},{1,2,3},{10,11,12},{7,8,9}},{{16,17,18},{13,14,15},{22,23,24},{19,20,21}},{{28,29,30},{25,26,27},{34,35,36},{31,32,33}}}
        ),
        std::make_tuple(
            std::make_tuple(9,8,7),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            2,
            tensor_type{{{2,3,1},{5,6,4},{8,9,7},{11,12,10}},{{14,15,13},{17,18,16},{20,21,19},{23,24,22}},{{26,27,25},{29,30,28},{32,33,31},{35,36,34}}}
        )
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto ten_or_int = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto rng_maker = [](const auto&...seeds_){
            return rng<bit_generator_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.permutation(ten_or_int,axis);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_random_permutation_exception","[test_random]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::default_rng;
    using gtensor::random_exception;

    REQUIRE_THROWS_AS(default_rng().permutation(tensor_type(1),0),random_exception);
    REQUIRE_THROWS_AS(default_rng().permutation(tensor_type{1,2,3,4,5},1),random_exception);
    REQUIRE_THROWS_AS(default_rng().permutation(tensor_type{{1,2,3},{4,5,6}},2),random_exception);
}

TEST_CASE("test_random_permuted","[test_random]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bit_generator_type = std::mt19937_64;
    using gtensor::detail::no_value;
    using gtensor::rng;
    using helpers_for_testing::apply_by_element;

    //0seeds,1tensor,2axis,3expected
    auto test_data = std::make_tuple(
        //axis no_value
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{},no_value{},tensor_type{}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{},no_value{},tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{}.reshape(0,2,3),no_value{},tensor_type{}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{}.reshape(0,2,3),no_value{},tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{3},no_value{},tensor_type{3}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{3},no_value{},tensor_type{3}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{{3}}},no_value{},tensor_type{3}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{{{3}}},no_value{},tensor_type{3}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},no_value{},tensor_type{10,14,5,11,6,8,9,1,13,2,12,15,4,7,3}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},no_value{},tensor_type{11,1,4,8,3,14,13,5,2,10,7,9,12,15,6}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},no_value{},tensor_type{10,14,5,11,6,8,9,1,13,2,12,15,4,7,3}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},no_value{},tensor_type{11,1,4,8,3,14,13,5,2,10,7,9,12,15,6}),
        //axis integral
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{},0,tensor_type{}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{},0,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{}.reshape(0,2,3),0,tensor_type{}.reshape(0,2,3)),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{}.reshape(0,2,3),0,tensor_type{}.reshape(0,2,3)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{3},0,tensor_type{3}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{3},0,tensor_type{3}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{{3}}},0,tensor_type{{{3}}}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{{{3}}},0,tensor_type{{{3}}}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},0,tensor_type{10,14,5,11,6,8,9,1,13,2,12,15,4,7,3}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},0,tensor_type{11,1,4,8,3,14,13,5,2,10,7,9,12,15,6}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},0,tensor_type{{1,2,7,8},{5,6,19,16},{13,18,11,20},{17,10,3,12},{9,14,15,4}}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},0,tensor_type{{17,14,7,8},{5,10,3,16},{9,18,15,4},{1,2,19,12},{13,6,11,20}}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},1,tensor_type{{2,1,4,3},{8,5,7,6},{11,12,9,10},{14,16,13,15},{18,20,17,19}}),
        std::make_tuple(std::make_tuple(3,2,1),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},1,tensor_type{{1,3,4,2},{8,6,7,5},{11,12,9,10},{16,13,15,14},{18,20,17,19}}),
        std::make_tuple(
            std::make_tuple(1,2,3),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            0,
            tensor_type{{{13,26,3},{4,17,30},{19,20,9},{10,23,24}},{{1,14,15},{16,5,18},{7,32,33},{34,35,12}},{{25,2,27},{28,29,6},{31,8,21},{22,11,36}}}
        ),
        std::make_tuple(
            std::make_tuple(1,2,3),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            1,
            tensor_type{{{4,5,3},{1,11,9},{10,2,6},{7,8,12}},{{22,17,18},{13,23,15},{19,14,24},{16,20,21}},{{31,35,27},{34,26,33},{25,29,36},{28,32,30}}}
        ),
        std::make_tuple(
            std::make_tuple(9,8,7),
            tensor_type{{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}},{{25,26,27},{28,29,30},{31,32,33},{34,35,36}}},
            2,
            tensor_type{{{2,3,1},{5,6,4},{7,8,9},{11,10,12}},{{13,14,15},{16,17,18},{21,20,19},{23,24,22}},{{27,26,25},{29,28,30},{32,33,31},{34,36,35}}}
        )
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto ten = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto rng_maker = [](const auto&...seeds_){
            return rng<bit_generator_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.permuted(ten,axis);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_random_permuted_exception","[test_random]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::default_rng;
    using gtensor::reduce_exception;

    REQUIRE_THROWS_AS(default_rng().permuted(tensor_type(1),0),reduce_exception);
    REQUIRE_THROWS_AS(default_rng().permuted(tensor_type{1,2,3,4,5},1),reduce_exception);
    REQUIRE_THROWS_AS(default_rng().permuted(tensor_type{{1,2,3},{4,5,6}},2),reduce_exception);
}

TEST_CASE("test_random_choice","[test_random]")
{
    using value_type = double;
    using gtensor::tensor;
    //random generator uses tensor's layout order to traverse tensor when generate random values
    //here test cases generates in f_order
    using tensor_type = gtensor::tensor<value_type,gtensor::config::f_order>;
    using bit_generator_type = std::mt19937_64;
    using gtensor::detail::no_value;
    using gtensor::rng;
    using helpers_for_testing::apply_by_element;

    //0seeds,1tensor,2size,3replace,4p,5axis,6shuffle,7expected
    auto test_data = std::make_tuple(
        //integral argument
        std::make_tuple(std::make_tuple(1,2,3),0,0,true,no_value{},0,false,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),20,0,true,no_value{},0,false,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),20,20,true,no_value{},0,false,tensor_type{18,6,13,19,18,18,1,3,3,16,0,2,4,16,14,5,0,7,15,17}),
        std::make_tuple(std::make_tuple(1,2,3),20,10,true,no_value{},0,false,tensor_type{18,6,13,19,18,18,1,3,3,16}),
        std::make_tuple(std::make_tuple(1,2,3),20,20,false,no_value{},0,true,tensor_type{1,0,6,18,11,3,19,4,12,5,2,10,9,16,13,8,14,17,15,7}),
        std::make_tuple(std::make_tuple(1,2,3),20,10,false,no_value{},0,true,tensor_type{18,0,6,13,17,1,16,2,14,3}),
        //replace true
        //no probabilities
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{},0,true,no_value{},0,true,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{},std::vector<int>{0,3,4},true,no_value{},0,true,tensor_type{}.reshape(0,3,4)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{}.reshape(2,3,0),std::vector<int>{0,3,4},true,no_value{},1,true,tensor_type{}.reshape(2,0,3,4,0)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5},0,true,no_value{},0,true,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5},std::vector<int>{0,3,4},true,no_value{},0,true,tensor_type{}.reshape(0,3,4)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3},{4,5,6}},std::vector<int>{0,3,4},true,no_value{},0,true,tensor_type{}.reshape(0,3,4,3)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{3},20,true,no_value{},0,true,tensor_type{3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3}),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{3},20,true,no_value{},0,true,tensor_type{3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{2},std::vector<int>{2,2,5},true,no_value{},0,true,tensor_type{{{2,2,2,2,2},{2,2,2,2,2}},{{2,2,2,2,2},{2,2,2,2,2}}}),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{2},std::vector<int>{2,2,5},true,no_value{},0,true,tensor_type{{{2,2,2,2,2},{2,2,2,2,2}},{{2,2,2,2,2},{2,2,2,2,2}}}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},30,true,no_value{},0,true,tensor_type{3,7,4,3,3,2,4,9,4,7,1,1,3,5,1,6,1,8,2,3,9,10,3,6,6,5,4,3,7,7}),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10},30,true,no_value{},0,true,tensor_type{9,9,6,9,9,8,7,6,9,10,1,9,6,4,7,10,5,2,7,5,4,10,3,1,4,5,3,10,6,5}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},std::vector<int>{3,2,5},true,no_value{},0,true,tensor_type{{{3,4,3,2,6},{3,7,6,10,3}},{{7,9,5,3,5},{3,1,1,3,7}},{{4,4,1,9,4},{2,1,8,6,7}}}),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10},std::vector<int>{3,2,5},true,no_value{},0,true,tensor_type{{{9,7,6,7,4},{9,10,10,10,10}},{{9,6,4,5,5},{9,1,5,3,6}},{{6,9,7,4,3},{8,9,2,1,5}}}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,no_value{},0,true,tensor_type{}.reshape(0,2,3)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},4,true,no_value{},0,true,tensor_type{{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}},{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},5,true,no_value{},1,true,tensor_type{{{1,2,3},{1,2,3},{1,2,3},{4,5,6},{4,5,6}},{{7,8,9},{7,8,9},{7,8,9},{10,11,12},{10,11,12}}}),
        std::make_tuple(std::make_tuple(1,2,3,4,5),tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},5,true,no_value{},2,true,tensor_type{{{2,1,3,3,3},{5,4,6,6,6}},{{8,7,9,9,9},{11,10,12,12,12}}}),
        std::make_tuple(std::make_tuple(1,2,3,4,5),tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{3,2},true,no_value{},1,true,tensor_type{{{{4,5,6},{1,2,3}},{{1,2,3},{4,5,6}},{{4,5,6},{1,2,3}}},{{{10,11,12},{7,8,9}},{{7,8,9},{10,11,12}},{{10,11,12},{7,8,9}}}}),
        //probabilities
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10},30,true,tensor<double>{0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},0,true,tensor_type{9,4,3,6,4,6,1,7,7,8,8,8,5,6,4,4,10,7,8,2,3,7,5,5,8,2,10,2,2,7}),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10},30,true,tensor<double>{1,1,1,1,1,1,1,1,1,1},0,true,tensor_type{9,4,3,6,4,6,1,7,7,8,8,8,5,6,4,4,10,7,8,2,3,7,5,5,8,2,10,2,2,7}),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10},30,true,tensor<double>{10,1,1,1,1,1,1,1,1,10},0,true,tensor_type{10,1,1,7,1,8,1,10,9,10,10,10,3,8,1,1,10,9,10,1,1,10,3,3,10,1,10,1,1,10}),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5},30,true,std::vector<double>{0.0,0.5,0.5,0.0,0.0},0,true,tensor_type{3,2,2,3,2,3,2,3,3,3,3,3,2,3,2,2,3,3,3,2,2,3,2,2,3,2,3,2,2,3}),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5},30,true,std::vector<double>{0.3,0.05,0.3,0.05,0.3},0,true,tensor_type{5,3,1,3,2,3,1,3,3,5,5,5,3,3,2,2,5,3,5,1,1,4,3,3,5,1,5,1,1,4}),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::vector<int>{2,10},true,std::vector<double>{0.1,0.8,0.1},0,true,
            tensor_type{{{4,5,6},{4,5,6},{4,5,6},{1,2,3},{4,5,6},{4,5,6},{4,5,6},{4,5,6},{7,8,9},{4,5,6}},{{4,5,6},{4,5,6},{4,5,6},{4,5,6},{4,5,6},{4,5,6},{4,5,6},{4,5,6},{4,5,6},{4,5,6}}}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::vector<int>{2,10},true,std::vector<double>{0.1,0.8,0.1},1,true,
            tensor_type{{{2,2,2,1,2,2,2,2,3,2},{2,2,2,2,2,2,2,2,2,2}},{{5,5,5,4,5,5,5,5,6,5},{5,5,5,5,5,5,5,5,5,5}},{{8,8,8,7,8,8,8,8,9,8},{8,8,8,8,8,8,8,8,8,8}}}
        ),
        //replace false, no probability
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{},0,false,no_value{},0,false,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{},std::vector<int>{0,3,4},false,no_value{},0,false,tensor_type{}.reshape(0,3,4)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{}.reshape(2,3,0),std::vector<int>{0,3,4},false,no_value{},1,false,tensor_type{}.reshape(2,0,3,4,0)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5},0,false,no_value{},0,false,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5},std::vector<int>{0,3,4},false,no_value{},0,false,tensor_type{}.reshape(0,3,4)),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3},{4,5,6}},std::vector<int>{0,3,4},false,no_value{},0,false,tensor_type{}.reshape(0,3,4,3)),

        //no shuffle
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},10,false,no_value{},0,false,tensor_type{1,2,3,4,5,6,7,8,9,10}),
        std::make_tuple(std::make_tuple(1,2,3,4,5),tensor_type{1,2,3,4,5,6,7,8,9,10},10,false,no_value{},0,false,tensor_type{1,2,3,4,5,6,7,8,9,10}),
        std::make_tuple(std::make_tuple(3,4,5,6,7),tensor_type{1,2,3,4,5,6,7},7,false,no_value{},0,false,tensor_type{1,2,3,4,5,6,7}),
        //shuffle
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},10,false,no_value{},0,true,tensor_type{10,2,6,9,5,3,8,1,7,4}),
        std::make_tuple(std::make_tuple(1,2,3,4,5),tensor_type{1,2,3,4,5,6,7,8,9,10},10,false,no_value{},0,true,tensor_type{5,2,10,6,4,3,9,7,8,1}),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7},7,false,no_value{},0,true,tensor_type{4,7,3,2,5,1,6}),

        //no shuffle
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},9,false,no_value{},0,false,tensor_type{1,3,4,5,6,7,2,9,10}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},8,false,no_value{},0,false,tensor_type{3,4,5,6,7,2,9,10}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},5,false,no_value{},0,false,tensor_type{3,7,8,4,10}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},3,false,no_value{},0,false,tensor_type{3,7,4}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},5,false,no_value{},0,false,
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},3,false,no_value{},0,false,
            tensor_type{{1,2,3,4},{13,14,15,16},{17,18,19,20}}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},std::vector<int>{2,2},false,no_value{},0,false,
            tensor_type{{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{17,18,19,20}}}
        ),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},4,false,no_value{},1,false,
            tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},3,false,no_value{},1,false,
            tensor_type{{1,3,2},{5,7,6},{9,11,10},{13,15,14},{17,19,18}}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},std::vector<int>{2,2},false,no_value{},1,false,
            tensor_type{{{1,3},{2,4}},{{5,7},{6,8}},{{9,11},{10,12}},{{13,15},{14,16}},{{17,19},{18,20}}}
        ),
        //shuffle
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},9,false,no_value{},0,true,tensor_type{3,6,10,7,4,9,1,2,5}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},8,false,no_value{},0,true,tensor_type{4,10,7,5,2,3,9,6}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},5,false,no_value{},0,true,tensor_type{10,4,3,7,8}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10},3,false,no_value{},0,true,tensor_type{7,3,4}),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},5,false,no_value{},0,true,
            tensor_type{{1,2,3,4},{13,14,15,16},{5,6,7,8},{17,18,19,20},{9,10,11,12}}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},3,false,no_value{},0,true,
            tensor_type{{17,18,19,20},{13,14,15,16},{1,2,3,4}}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},std::vector<int>{2,2},false,no_value{},0,true,
            tensor_type{{{17,18,19,20},{5,6,7,8}},{{9,10,11,12},{1,2,3,4}}}
        ),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},4,false,no_value{},1,true,
            tensor_type{{4,1,3,2},{8,5,7,6},{12,9,11,10},{16,13,15,14},{20,17,19,18}}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},3,false,no_value{},1,true,
            tensor_type{{3,2,1},{7,6,5},{11,10,9},{15,14,13},{19,18,17}}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16},{17,18,19,20}},std::vector<int>{2,2},false,no_value{},1,true,
            tensor_type{{{3,4},{2,1}},{{7,8},{6,5}},{{11,12},{10,9}},{{15,16},{14,13}},{{19,20},{18,17}}}
        ),
        //replace false, probability
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},20,false,
            std::vector<double>{0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},0,false,
            tensor_type{17,8,5,12,7,1,13,16,9,19,4,6,14,15,3,20,2,11,10,18}
        ),
        //shuffle doesnt matter if select without replacement with probabilities
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},20,false,
            std::vector<double>{0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1},0,true,
            tensor_type{17,8,5,12,7,1,13,16,9,19,4,6,14,15,3,20,2,11,10,18}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},20,false,
            std::vector<double>{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false,
            tensor_type{17,8,5,12,7,1,13,16,9,19,4,6,14,15,3,20,2,11,10,18}
        ),
        //seed matters
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},20,false,
            std::vector<double>{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false,
            tensor_type{2,5,3,4,1,14,20,7,8,15,6,11,17,10,13,16,12,9,18,19}
        ),
        //small probabilities
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},20,false,
            std::vector<double>{1E-20,1,1,1,1,1,1,1,1E-10,1,1,1,1,1,1,1,1,1,1,1E-30},0,false,
            tensor_type{3,6,4,2,14,19,7,8,15,11,16,10,13,17,12,5,18,9,1,20}
        ),
        //zero probabilities
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},14,false,
            std::vector<double>{0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0},0,false,
            tensor_type{4,6,5,3,14,18,7,8,15,12,16,11,13,17}
        ),
        //not all elements
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},10,false,
            std::vector<double>{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false,
            tensor_type{2,5,3,4,1,14,20,7,8,15}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},10,false,
            std::vector<double>{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false,
            tensor_type{17,8,5,12,7,1,13,16,9,19}
        ),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},5,false,
            std::vector<double>{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false,
            tensor_type{2,5,3,4,1}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},5,false,
            std::vector<double>{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false,
            tensor_type{17,8,5,12,7}
        ),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},1,false,
            std::vector<double>{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false,
            tensor_type{2}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},1,false,
            std::vector<double>{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false,
            tensor_type{17}
        ),
        //not all elements, not equal probabilities
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},10,false,
            std::vector<double>{10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1},0,false,
            tensor_type{2,5,3,1,13,20,7,15,11,9}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},10,false,
            std::vector<double>{10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1},0,false,
            tensor_type{17,7,5,11,1,13,15,9,19,4}
        ),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},5,false,
            std::vector<double>{1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10},0,false,
            tensor_type{2,6,4,14,20}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},5,false,
            std::vector<double>{1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10,1,10},0,false,
            tensor_type{18,8,6,12,2}
        ),
        std::make_tuple(std::make_tuple(1,2,3),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},1,false,
            std::vector<double>{1,1,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false,
            tensor_type{3}
        ),
        std::make_tuple(std::make_tuple(2,3,4),tensor_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20},1,false,
            std::vector<double>{1,1,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},0,false,
            tensor_type{15}
        )

    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto ten = std::get<1>(t);
        auto size = std::get<2>(t);
        auto replace = std::get<3>(t);
        auto p = std::get<4>(t);
        auto axis = std::get<5>(t);
        auto shuffle = std::get<6>(t);
        auto expected = std::get<7>(t);

        auto rng_maker = [](const auto&...seeds_){
            return rng<bit_generator_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.choice(ten,size,replace,p,axis,shuffle);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_random_choice_exception","[test_random]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::detail::no_value;
    using gtensor::default_rng;
    using gtensor::random_exception;

    //axis<t.dim()
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},10,true,no_value{},1),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},10,true,no_value{},2),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},10,true,no_value{},3),random_exception);
    //if is_p: p.size() must equal axis_size,
    REQUIRE_THROWS_AS(default_rng().choice(5,10,true,tensor_type{},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(5,10,true,tensor_type{1,1,1,1,1,1},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(5,10,true,tensor_type{1,1,1},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},10,true,tensor_type{},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},10,true,tensor_type{1,1,1},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},10,true,std::vector<double>{1,1,1,1,1,1},0),random_exception);
    //if is_p: p must not sum to zero
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},10,true,std::vector<double>{0,0,0,0,0},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},4,false,std::vector<double>{0,0,0,0,0},0),random_exception);
    //if is_p: p elements must not be negative
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},10,true,std::vector<double>{1,2,-1,3,5},0),random_exception);
    //if t.size()==0 and size of size > 0 (a cannot be empty unless no samples are taken)
    REQUIRE_THROWS_AS(default_rng().choice(0,1,true,no_value{},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{},1,true,no_value{},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{}.reshape(2,3,0),std::vector<int>(3,3),true,no_value{},0),random_exception);
    //if !replace size of size must be <= axis_size,
    REQUIRE_THROWS_AS(default_rng().choice(5,6,false,std::vector<double>{1,1,1,1,1},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},6,false,std::vector<double>{1,1,1,1,1},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},4,false,no_value{},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},3,false,no_value{},1),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}}},3,false,no_value{},2),random_exception);
    //if !replace and there are zero probabilities that is cant select size elements (due to zero probs)
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},5,false,std::vector<double>{1,1,0,1,1},0),random_exception);
    REQUIRE_THROWS_AS(default_rng().choice(tensor_type{1,2,3,4,5},4,false,std::vector<double>{1,1,0,1,0},0),random_exception);
}

