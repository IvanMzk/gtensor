#include <tuple>
#include <vector>
#include "catch.hpp"
#include "tensor_math.hpp"
#include "statistic.hpp"
#include "sort_search.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

//math
TEST_CASE("test_tensor_math_member_function_default_policy","[test_tensor]")
{
    using value_type = int;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;

    //all
    REQUIRE(tensor_type(2).all() == tensor<bool>(true));
    REQUIRE(tensor_type(0).all(true) == tensor<bool>(false));
    REQUIRE(tensor_type{1,2,3,4,5}.all() == tensor<bool>(true));
    REQUIRE(tensor_type{1,0,3,0,5}.all() == tensor<bool>(false));
    REQUIRE(tensor_type{{{3,1,2,3},{1,2,3,2}},{{1,2,3,4},{1,2,3,4}}}.all() == tensor<bool>(true));
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(true) == tensor<bool>{{{false}}});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(2) == tensor<bool>{{false,false},{true,true}});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(1,true) == tensor<bool>{{{false,true,true,false}},{{true,true,true,true}}});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all({0,1}) == tensor<bool>{false,true,true,false});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all({1,-1},true) == tensor<bool>{{{false}},{{true}}});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(std::vector<int>{0,1}) == tensor<bool>{false,true,true,false});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(std::vector<int>{1,-1},true) == tensor<bool>{{{false}},{{true}}});

    //any
    REQUIRE(tensor_type(2).any() == tensor<bool>(true));
    REQUIRE(tensor_type(0).any(true) == tensor<bool>(false));
    REQUIRE(tensor_type{1,2,3,4,5}.any() == tensor<bool>(true));
    REQUIRE(tensor_type{0,0,0,0,0}.any() == tensor<bool>(false));
    REQUIRE(tensor_type{{{3,1,2,3},{1,2,3,2}},{{1,2,3,4},{1,2,3,4}}}.any() == tensor<bool>(true));
    REQUIRE(tensor_type{{{0,0,0,0},{0,0,0,0}},{{0,0,0,0},{0,0,0,0}}}.any(true) == tensor<bool>{{{false}}});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(2) == tensor<bool>{{true,false},{true,true}});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(1,true) == tensor<bool>{{{false,true,true,false}},{{true,true,true,false}}});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any({0,1}) == tensor<bool>{true,true,true,false});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any({1,-1},true) == tensor<bool>{{{true}},{{true}}});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(std::vector<int>{0,1}) == tensor<bool>{true,true,true,false});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(std::vector<int>{1,-1},true) == tensor<bool>{{{true}},{{true}}});

    //max
    REQUIRE(tensor_type(2).max() == tensor_type(2));
    REQUIRE(tensor_type(2).max(true) == tensor_type(2));
    REQUIRE(tensor_type(2).max(true,10) == tensor_type(10));
    REQUIRE(tensor_type(2).max(true,-10) == tensor_type(2));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max() == tensor_type(4));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(true) == tensor_type{{{4}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(true,10) == tensor_type{{{10}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(true,-10) == tensor_type{{{4}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(2) == tensor_type{{3,4},{3,3}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(0,true) == tensor_type{{{-2,3,3,3},{4,3,3,2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(0,true,1) == tensor_type{{{1,3,3,3},{4,3,3,2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max({1,-1}) == tensor_type{4,3});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max({0,1},true) == tensor_type{{{4,3,3,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max({0,1},true,0) == tensor_type{{{4,3,3,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(std::vector<int>{0,1},true) == tensor_type{{{4,3,3,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(std::vector<int>{0,1},true,0) == tensor_type{{{4,3,3,3}}});

    //min
    REQUIRE(tensor_type(2).min() == tensor_type(2));
    REQUIRE(tensor_type(2).min(true) == tensor_type(2));
    REQUIRE(tensor_type(2).min(true,10) == tensor_type(2));
    REQUIRE(tensor_type(2).min(true,-10) == tensor_type(-10));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min() == tensor_type(-3));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(true) == tensor_type{{{-3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(true,10) == tensor_type{{{-3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(true,-10) == tensor_type{{{-10}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(2) == tensor_type{{-3,0},{-2,-2}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(0,true) == tensor_type{{{-3,3,1,-2},{0,3,-2,0}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(0,true,1) == tensor_type{{{-3,1,1,-2},{0,1,-2,0}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min({1,-1}) == tensor_type{-3,-2});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min({0,1},true) == tensor_type{{{-3,3,-2,-2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min({0,1},true,0) == tensor_type{{{-3,0,-2,-2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(std::vector<int>{0,1},true) == tensor_type{{{-3,3,-2,-2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(std::vector<int>{0,1},true,0) == tensor_type{{{-3,0,-2,-2}}});

    //sum
    REQUIRE(tensor_type(2).sum() == tensor_type(2));
    REQUIRE(tensor_type(2).sum(true) == tensor_type(2));
    REQUIRE(tensor_type(2).sum(true,10) == tensor_type(12));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum() == tensor_type(19));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(true) == tensor_type{{{19}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(true,10) == tensor_type{{{29}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(2) == tensor_type{{1,10},{5,3}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(0,true) == tensor_type{{{-5,6,4,1},{4,6,1,2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(0,true,1) == tensor_type{{{-4,7,5,2},{5,7,2,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum({1,-1}) == tensor_type{11,8});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum({0,1},true) == tensor_type{{{-1,12,5,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum({0,1},true,0) == tensor_type{{{-1,12,5,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(std::vector<int>{0,1},true) == tensor_type{{{-1,12,5,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(std::vector<int>{0,1},true,0) == tensor_type{{{-1,12,5,3}}});

    //prod
    REQUIRE(tensor_type(2).prod() == tensor_type(2));
    REQUIRE(tensor_type(2).prod(true) == tensor_type(2));
    REQUIRE(tensor_type(2).prod(true,10) == tensor_type(20));
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod() == tensor_type(7776));
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(true) == tensor_type{{{7776}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(true,10) == tensor_type{{{77760}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(2) == tensor_type{{12,12},{3,18}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(0,true) == tensor_type{{{2,6,1,3},{6,3,3,4}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(0,true,1) == tensor_type{{{2,6,1,3},{6,3,3,4}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod({1,-1}) == tensor_type{144,54});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod({0,1},true) == tensor_type{{{12,18,3,12}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod({0,1},true,0) == tensor_type{{{0,0,0,0}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(std::vector<int>{0,1},true) == tensor_type{{{12,18,3,12}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(std::vector<int>{0,1},true,0) == tensor_type{{{0,0,0,0}}});

    //cumsum
    REQUIRE(tensor_type(2).cumsum() == tensor_type{2});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum() == tensor_type{-3,0,3,1,5,8,11,11,9,12,13,16,16,19,17,19});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum(0) == tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-5,6,4,1},{4,6,1,2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum(1) == tensor_type{{{-3,3,3,-2},{1,6,6,-2}},{{-2,3,1,3},{-2,6,-1,5}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum(2) == tensor_type{{{-3,0,3,1},{4,7,10,10}},{{-2,1,2,5},{0,3,1,3}}});

    //cumprod
    REQUIRE(tensor_type(2).cumprod() == tensor_type{2});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod() == tensor_type{-3,-9,-27,54,216,648,1944,0,0,0,0,0,0,0,0,0});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod(0) == tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{6,9,3,-6},{0,9,-6,0}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod(1) == tensor_type{{{-3,3,3,-2},{-12,9,9,0}},{{-2,3,1,3},{0,9,-2,6}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod(2) == tensor_type{{{-3,-9,-27,54},{4,12,36,0}},{{-2,-6,-6,-18},{0,0,0,0}}});
}

TEST_CASE("test_tensor_math_member_function_policy","[test_tensor]")
{
    using policy = multithreading::exec_pol<4>;
    using value_type = int;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;

    //all
    REQUIRE(tensor_type(2).all(policy{}) == tensor<bool>(true));
    REQUIRE(tensor_type(0).all(policy{},true) == tensor<bool>(false));
    REQUIRE(tensor_type{1,2,3,4,5}.all(policy{}) == tensor<bool>(true));
    REQUIRE(tensor_type{1,0,3,0,5}.all(policy{}) == tensor<bool>(false));
    REQUIRE(tensor_type{{{3,1,2,3},{1,2,3,2}},{{1,2,3,4},{1,2,3,4}}}.all(policy{}) == tensor<bool>(true));
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(policy{},true) == tensor<bool>{{{false}}});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(policy{},2) == tensor<bool>{{false,false},{true,true}});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(policy{},1,true) == tensor<bool>{{{false,true,true,false}},{{true,true,true,true}}});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(policy{},{0,1}) == tensor<bool>{false,true,true,false});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(policy{},{1,-1},true) == tensor<bool>{{{false}},{{true}}});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(policy{},std::vector<int>{0,1}) == tensor<bool>{false,true,true,false});
    REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(policy{},std::vector<int>{1,-1},true) == tensor<bool>{{{false}},{{true}}});

    //any
    REQUIRE(tensor_type(2).any(policy{}) == tensor<bool>(true));
    REQUIRE(tensor_type(0).any(policy{},true) == tensor<bool>(false));
    REQUIRE(tensor_type{1,2,3,4,5}.any(policy{}) == tensor<bool>(true));
    REQUIRE(tensor_type{0,0,0,0,0}.any(policy{}) == tensor<bool>(false));
    REQUIRE(tensor_type{{{3,1,2,3},{1,2,3,2}},{{1,2,3,4},{1,2,3,4}}}.any(policy{}) == tensor<bool>(true));
    REQUIRE(tensor_type{{{0,0,0,0},{0,0,0,0}},{{0,0,0,0},{0,0,0,0}}}.any(policy{},true) == tensor<bool>{{{false}}});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(policy{},2) == tensor<bool>{{true,false},{true,true}});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(policy{},1,true) == tensor<bool>{{{false,true,true,false}},{{true,true,true,false}}});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(policy{},{0,1}) == tensor<bool>{true,true,true,false});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(policy{},{1,-1},true) == tensor<bool>{{{true}},{{true}}});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(policy{},std::vector<int>{0,1}) == tensor<bool>{true,true,true,false});
    REQUIRE(tensor_type{{{0,1,2,0},{0,0,0,0}},{{1,2,3,0},{1,2,3,0}}}.any(policy{},std::vector<int>{1,-1},true) == tensor<bool>{{{true}},{{true}}});

    //max
    REQUIRE(tensor_type(2).max(policy{}) == tensor_type(2));
    REQUIRE(tensor_type(2).max(policy{},true) == tensor_type(2));
    REQUIRE(tensor_type(2).max(policy{},true,10) == tensor_type(10));
    REQUIRE(tensor_type(2).max(policy{},true,-10) == tensor_type(2));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{}) == tensor_type(4));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},true) == tensor_type{{{4}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},true,10) == tensor_type{{{10}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},true,-10) == tensor_type{{{4}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},2) == tensor_type{{3,4},{3,3}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},0,true) == tensor_type{{{-2,3,3,3},{4,3,3,2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},0,true,1) == tensor_type{{{1,3,3,3},{4,3,3,2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},{1,-1}) == tensor_type{4,3});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},{0,1},true) == tensor_type{{{4,3,3,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},{0,1},true,0) == tensor_type{{{4,3,3,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},std::vector<int>{0,1},true) == tensor_type{{{4,3,3,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.max(policy{},std::vector<int>{0,1},true,0) == tensor_type{{{4,3,3,3}}});

    //min
    REQUIRE(tensor_type(2).min() == tensor_type(2));
    REQUIRE(tensor_type(2).min(true) == tensor_type(2));
    REQUIRE(tensor_type(2).min(true,10) == tensor_type(2));
    REQUIRE(tensor_type(2).min(true,-10) == tensor_type(-10));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min() == tensor_type(-3));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(true) == tensor_type{{{-3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(true,10) == tensor_type{{{-3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(true,-10) == tensor_type{{{-10}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(2) == tensor_type{{-3,0},{-2,-2}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(0,true) == tensor_type{{{-3,3,1,-2},{0,3,-2,0}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(0,true,1) == tensor_type{{{-3,1,1,-2},{0,1,-2,0}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min({1,-1}) == tensor_type{-3,-2});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min({0,1},true) == tensor_type{{{-3,3,-2,-2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min({0,1},true,0) == tensor_type{{{-3,0,-2,-2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(std::vector<int>{0,1},true) == tensor_type{{{-3,3,-2,-2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.min(std::vector<int>{0,1},true,0) == tensor_type{{{-3,0,-2,-2}}});

    //sum
    REQUIRE(tensor_type(2).sum(policy{}) == tensor_type(2));
    REQUIRE(tensor_type(2).sum(policy{},true) == tensor_type(2));
    REQUIRE(tensor_type(2).sum(policy{},true,10) == tensor_type(12));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{}) == tensor_type(19));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{},true) == tensor_type{{{19}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{},true,10) == tensor_type{{{29}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{},2) == tensor_type{{1,10},{5,3}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{},0,true) == tensor_type{{{-5,6,4,1},{4,6,1,2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{},0,true,1) == tensor_type{{{-4,7,5,2},{5,7,2,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{},{1,-1}) == tensor_type{11,8});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{},{0,1},true) == tensor_type{{{-1,12,5,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{},{0,1},true,0) == tensor_type{{{-1,12,5,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{},std::vector<int>{0,1},true) == tensor_type{{{-1,12,5,3}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.sum(policy{},std::vector<int>{0,1},true,0) == tensor_type{{{-1,12,5,3}}});

    //prod
    REQUIRE(tensor_type(2).prod(policy{}) == tensor_type(2));
    REQUIRE(tensor_type(2).prod(policy{},true) == tensor_type(2));
    REQUIRE(tensor_type(2).prod(policy{},true,10) == tensor_type(20));
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{}) == tensor_type(7776));
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{},true) == tensor_type{{{7776}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{},true,10) == tensor_type{{{77760}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{},2) == tensor_type{{12,12},{3,18}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{},0,true) == tensor_type{{{2,6,1,3},{6,3,3,4}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{},0,true,1) == tensor_type{{{2,6,1,3},{6,3,3,4}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{},{1,-1}) == tensor_type{144,54});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{},{0,1},true) == tensor_type{{{12,18,3,12}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{},{0,1},true,0) == tensor_type{{{0,0,0,0}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{},std::vector<int>{0,1},true) == tensor_type{{{12,18,3,12}}});
    REQUIRE(tensor_type{{{2,2,1,3},{2,3,1,2}},{{1,3,1,1},{3,1,3,2}}}.prod(policy{},std::vector<int>{0,1},true,0) == tensor_type{{{0,0,0,0}}});

    //cumsum
    REQUIRE(tensor_type(2).cumsum(policy{}) == tensor_type{2});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum(policy{}) == tensor_type{-3,0,3,1,5,8,11,11,9,12,13,16,16,19,17,19});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum(policy{},0) == tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-5,6,4,1},{4,6,1,2}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum(policy{},1) == tensor_type{{{-3,3,3,-2},{1,6,6,-2}},{{-2,3,1,3},{-2,6,-1,5}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumsum(policy{},2) == tensor_type{{{-3,0,3,1},{4,7,10,10}},{{-2,1,2,5},{0,3,1,3}}});

    //cumprod
    REQUIRE(tensor_type(2).cumprod(policy{}) == tensor_type{2});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod(policy{}) == tensor_type{-3,-9,-27,54,216,648,1944,0,0,0,0,0,0,0,0,0});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod(policy{},0) == tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{6,9,3,-6},{0,9,-6,0}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod(policy{},1) == tensor_type{{{-3,3,3,-2},{-12,9,9,0}},{{-2,3,1,3},{0,9,-2,6}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.cumprod(policy{},2) == tensor_type{{{-3,-9,-27,54},{4,12,36,0}},{{-2,-6,-6,-18},{0,0,0,0}}});
}

//statistic
TEST_CASE("test_tensor_statistic_member_function_default_policy","[test_tensor]")
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::tensor_close;
    using tensor_type = gtensor::tensor<value_type>;

    //ptp
    REQUIRE(tensor_type(2).ptp() == tensor_type(0));
    REQUIRE(tensor_type(0).ptp(true) == tensor_type(0));
    REQUIRE(tensor_type{1,2,3,4,5}.ptp(true) == tensor_type{4});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp() == tensor_type(7));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(true) == tensor_type{{{7}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(2) == tensor_type{{6,4},{5,5}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(1,true) == tensor_type{{{7,0,0,2}},{{2,0,3,1}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp({0,1}) == tensor_type{7,0,5,5});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp({1,-1},true) == tensor_type{{{7}},{{5}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(std::vector<int>{0,1}) == tensor_type{7,0,5,5});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(std::vector<int>{1,-1},true) == tensor_type{{{7}},{{5}}});

    //mean
    REQUIRE(tensor_type(2).mean() == tensor_type(2));
    REQUIRE(tensor_type(0).mean(true) == tensor_type(0));
    REQUIRE(tensor_close(tensor_type{1,2,3,4,5}.mean(true), tensor_type{3},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean() ,tensor_type(1.187),1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(true) ,tensor_type{{{1.187}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(2) ,tensor_type{{0.25,2.5},{1.25,0.75}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(1,true) ,tensor_type{{{0.5,3.0,3.0,-1.0}},{{-1.0,3.0,-0.5,2.5}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean({0,1}) ,tensor_type{-0.25,3.0,1.25,0.75},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean({1,-1},true) ,tensor_type{{{1.375}},{{1.0}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(std::vector<int>{0,1}) ,tensor_type{-0.25,3.0,1.25,0.75},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(std::vector<int>{1,-1},true) ,tensor_type{{{1.375}},{{1.0}}},1E-2,1E-2));

    //median
    REQUIRE(tensor_type(2).median() == tensor_type(2));
    REQUIRE(tensor_type(0).median(true) == tensor_type(0));
    REQUIRE(tensor_type{1,2,3,4,5}.median(true) == tensor_type{3});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median() == tensor_type(2.5));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(true) == tensor_type{{{2.5}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(2) == tensor_type{{0.5,3.0},{2.0,1.0}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(1,true) == tensor_type{{{0.5,3.0,3.0,-1.0}},{{-1.0,3.0,-0.5,2.5}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median({0,1}) == tensor_type{-1,3,2,1});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median({1,-1},true) == tensor_type{{{3.0}},{{1.5}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(std::vector<int>{0,1}) == tensor_type{-1,3,2,1});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(std::vector<int>{1,-1},true) == tensor_type{{{3.0}},{{1.5}}});

    //var
    REQUIRE(tensor_type(2).var() == tensor_type(0));
    REQUIRE(tensor_type(0).var(true) == tensor_type(0));
    REQUIRE(tensor_close(tensor_type{1,2,3,4,5}.var(true), tensor_type{2},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var() ,tensor_type(5.152),1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(true) ,tensor_type{{{5.152}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(2) ,tensor_type{{7.6875,2.25},{4.1875,3.6875}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(1,true) ,tensor_type{{{12.25,0.0,0.0,1.0}},{{1.0,0.0,2.25,0.25}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var({0,1}) ,tensor_type{7.1875,0.0,4.1875,3.6875},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var({1,-1},true) ,tensor_type{{{6.234}},{{4.0}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(std::vector<int>{0,1}) ,tensor_type{7.1875,0.0,4.1875,3.6875},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(std::vector<int>{1,-1},true) ,tensor_type{{{6.234}},{{4.0}}},1E-2,1E-2));

    //stdev
    REQUIRE(tensor_type(2).stdev() == tensor_type(0));
    REQUIRE(tensor_type(0).stdev(true) == tensor_type(0));
    REQUIRE(tensor_close(tensor_type{1,2,3,4,5}.stdev(true), tensor_type{1.414},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev() ,tensor_type(2.279),1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(true) ,tensor_type{{{2.270}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(2) ,tensor_type{{2.772,1.5},{2.046,1.920}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(1,true) ,tensor_type{{{3.5,0.0,0.0,1.0}},{{1.0,0.0,1.5,0.5}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev({0,1}) ,tensor_type{2.680,0.0,2.046,1.920},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev({1,-1},true) ,tensor_type{{{2.496}},{{2.0}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(std::vector<int>{0,1}) ,tensor_type{2.680,0.0,2.046,1.920},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(std::vector<int>{1,-1},true) ,tensor_type{{{2.496}},{{2.0}}},1E-2,1E-2));
}

TEST_CASE("test_tensor_statistic_member_function_policy","[test_tensor]")
{
    using policy = multithreading::exec_pol<4>;
    using value_type = double;
    using gtensor::tensor;
    using gtensor::tensor_close;
    using tensor_type = gtensor::tensor<value_type>;

    //ptp
    REQUIRE(tensor_type(2).ptp(policy{}) == tensor_type(0));
    REQUIRE(tensor_type(0).ptp(policy{},true) == tensor_type(0));
    REQUIRE(tensor_type{1,2,3,4,5}.ptp(policy{},true) == tensor_type{4});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(policy{}) == tensor_type(7));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(policy{},true) == tensor_type{{{7}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(policy{},2) == tensor_type{{6,4},{5,5}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(policy{},1,true) == tensor_type{{{7,0,0,2}},{{2,0,3,1}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(policy{},{0,1}) == tensor_type{7,0,5,5});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(policy{},{1,-1},true) == tensor_type{{{7}},{{5}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(policy{},std::vector<int>{0,1}) == tensor_type{7,0,5,5});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.ptp(policy{},std::vector<int>{1,-1},true) == tensor_type{{{7}},{{5}}});

    //mean
    REQUIRE(tensor_type(2).mean(policy{}) == tensor_type(2));
    REQUIRE(tensor_type(0).mean(policy{},true) == tensor_type(0));
    REQUIRE(tensor_close(tensor_type{1,2,3,4,5}.mean(policy{},true), tensor_type{3},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(policy{}) ,tensor_type(1.187),1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(policy{},true) ,tensor_type{{{1.187}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(policy{},2) ,tensor_type{{0.25,2.5},{1.25,0.75}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(policy{},1,true) ,tensor_type{{{0.5,3.0,3.0,-1.0}},{{-1.0,3.0,-0.5,2.5}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(policy{},{0,1}) ,tensor_type{-0.25,3.0,1.25,0.75},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(policy{},{1,-1},true) ,tensor_type{{{1.375}},{{1.0}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(policy{},std::vector<int>{0,1}) ,tensor_type{-0.25,3.0,1.25,0.75},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.mean(policy{},std::vector<int>{1,-1},true) ,tensor_type{{{1.375}},{{1.0}}},1E-2,1E-2));

    //median
    REQUIRE(tensor_type(2).median(policy{}) == tensor_type(2));
    REQUIRE(tensor_type(0).median(policy{},true) == tensor_type(0));
    REQUIRE(tensor_type{1,2,3,4,5}.median(policy{},true) == tensor_type{3});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(policy{}) == tensor_type(2.5));
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(policy{},true) == tensor_type{{{2.5}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(policy{},2) == tensor_type{{0.5,3.0},{2.0,1.0}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(policy{},1,true) == tensor_type{{{0.5,3.0,3.0,-1.0}},{{-1.0,3.0,-0.5,2.5}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(policy{},{0,1}) == tensor_type{-1,3,2,1});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(policy{},{1,-1},true) == tensor_type{{{3.0}},{{1.5}}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(policy{},std::vector<int>{0,1}) == tensor_type{-1,3,2,1});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.median(policy{},std::vector<int>{1,-1},true) == tensor_type{{{3.0}},{{1.5}}});

    //var
    REQUIRE(tensor_type(2).var(policy{}) == tensor_type(0));
    REQUIRE(tensor_type(0).var(policy{},true) == tensor_type(0));
    REQUIRE(tensor_close(tensor_type{1,2,3,4,5}.var(policy{},true), tensor_type{2},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(policy{}) ,tensor_type(5.152),1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(policy{},true) ,tensor_type{{{5.152}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(policy{},2) ,tensor_type{{7.6875,2.25},{4.1875,3.6875}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(policy{},1,true) ,tensor_type{{{12.25,0.0,0.0,1.0}},{{1.0,0.0,2.25,0.25}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(policy{},{0,1}) ,tensor_type{7.1875,0.0,4.1875,3.6875},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(policy{},{1,-1},true) ,tensor_type{{{6.234}},{{4.0}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(policy{},std::vector<int>{0,1}) ,tensor_type{7.1875,0.0,4.1875,3.6875},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.var(policy{},std::vector<int>{1,-1},true) ,tensor_type{{{6.234}},{{4.0}}},1E-2,1E-2));

    //stdev
    REQUIRE(tensor_type(2).stdev(policy{}) == tensor_type(0));
    REQUIRE(tensor_type(0).stdev(policy{},true) == tensor_type(0));
    REQUIRE(tensor_close(tensor_type{1,2,3,4,5}.stdev(policy{},true), tensor_type{1.414},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(policy{}) ,tensor_type(2.279),1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(policy{},true) ,tensor_type{{{2.270}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(policy{},2) ,tensor_type{{2.772,1.5},{2.046,1.920}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(policy{},1,true) ,tensor_type{{{3.5,0.0,0.0,1.0}},{{1.0,0.0,1.5,0.5}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(policy{},{0,1}) ,tensor_type{2.680,0.0,2.046,1.920},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(policy{},{1,-1},true) ,tensor_type{{{2.496}},{{2.0}}},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(policy{},std::vector<int>{0,1}) ,tensor_type{2.680,0.0,2.046,1.920},1E-2,1E-2));
    REQUIRE(tensor_close(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.stdev(policy{},std::vector<int>{1,-1},true) ,tensor_type{{{2.496}},{{2.0}}},1E-2,1E-2));
}

//sort_search
TEST_CASE("test_tensor_sort_search_member_function_sort_default_policy","[test_tensor]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;

    //sort
    tensor_type t{4,6,-6,-6,-5,6,-1,-8,-5,-9,-5,-1,5,-8,0,-6,-7,6,-7,-8,-5,7,7,-8,3,0,-2,-10,-7,-7,9,-9,-3,-4,-7,-9,-3,8,-4,-5,-1,-2,-1,-6,-2,8,-6,2,-7,9};
    tensor_type x{{{7,-3,2,-2,4},{4,6,6,4,5},{7,6,4,4,1},{5,-6,-10,-3,-9}},{{-8,0,8,-3,-3},{-7,-4,-9,-2,7},{3,0,5,-10,-7},{7,9,-3,-3,9}},{{4,9,-5,-9,-5},{-1,7,-7,3,-9},{-4,7,6,6,2},{9,6,-4,-8,-3}}};
    SECTION("sort")
    {
        t.sort();
        REQUIRE(t == tensor_type{-10,-9,-9,-9,-8,-8,-8,-8,-7,-7,-7,-7,-7,-7,-6,-6,-6,-6,-6,-5,-5,-5,-5,-5,-4,-4,-3,-3,-2,-2,-2,-1,-1,-1,-1,0,0,2,3,4,5,6,6,6,7,7,8,8,9,9});
        x.sort();
        REQUIRE(x == tensor_type{{{-3,-2,2,4,7},{4,4,5,6,6},{1,4,4,6,7},{-10,-9,-6,-3,5}},{{-8,-3,-3,0,8},{-9,-7,-4,-2,7},{-10,-7,0,3,5},{-3,-3,7,9,9}},{{-9,-5,-5,4,9},{-9,-7,-1,3,7},{-4,2,6,6,7},{-8,-4,-3,6,9}}});
    }
    SECTION("sort_axis")
    {
        t.sort(0);
        REQUIRE(t == tensor_type{-10,-9,-9,-9,-8,-8,-8,-8,-7,-7,-7,-7,-7,-7,-6,-6,-6,-6,-6,-5,-5,-5,-5,-5,-4,-4,-3,-3,-2,-2,-2,-1,-1,-1,-1,0,0,2,3,4,5,6,6,6,7,7,8,8,9,9});
        SECTION("sort_0")
        {
            x.sort(0);
            REQUIRE(x == tensor_type{{{-8,-3,-5,-9,-5},{-7,-4,-9,-2,-9},{-4,0,4,-10,-7},{5,-6,-10,-8,-9}},{{4,0,2,-3,-3},{-1,6,-7,3,5},{3,6,5,4,1},{7,6,-4,-3,-3}},{{7,9,8,-2,4},{4,7,6,4,7},{7,7,6,6,2},{9,9,-3,-3,9}}});
        }
        SECTION("sort_1")
        {
            x.sort(1);
            REQUIRE(x == tensor_type{{{4,-6,-10,-3,-9},{5,-3,2,-2,1},{7,6,4,4,4},{7,6,6,4,5}},{{-8,-4,-9,-10,-7},{-7,0,-3,-3,-3},{3,0,5,-3,7},{7,9,8,-2,9}},{{-4,6,-7,-9,-9},{-1,7,-5,-8,-5},{4,7,-4,3,-3},{9,9,6,6,2}}});
        }
    }
    SECTION("sort_axis_comparator")
    {
        t.sort(0,std::greater<void>{});
        REQUIRE(t == tensor_type{9,9,8,8,7,7,6,6,6,5,4,3,2,0,0,-1,-1,-1,-1,-2,-2,-2,-3,-3,-4,-4,-5,-5,-5,-5,-5,-6,-6,-6,-6,-6,-7,-7,-7,-7,-7,-7,-8,-8,-8,-8,-9,-9,-9,-10});
        x.sort(1,std::greater<void>{});
        REQUIRE(x == tensor_type{{{7,6,6,4,5},{7,6,4,4,4},{5,-3,2,-2,1},{4,-6,-10,-3,-9}},{{7,9,8,-2,9},{3,0,5,-3,7},{-7,0,-3,-3,-3},{-8,-4,-9,-10,-7}},{{9,9,6,6,2},{4,7,-4,3,-3},{-1,7,-5,-8,-5},{-4,6,-7,-9,-9}}});
    }
}

TEST_CASE("test_tensor_sort_search_member_function_sort_policy","[test_tensor]")
{
    using policy = multithreading::exec_pol<1>;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;

    //sort
    tensor_type t{4,6,-6,-6,-5,6,-1,-8,-5,-9,-5,-1,5,-8,0,-6,-7,6,-7,-8,-5,7,7,-8,3,0,-2,-10,-7,-7,9,-9,-3,-4,-7,-9,-3,8,-4,-5,-1,-2,-1,-6,-2,8,-6,2,-7,9};
    tensor_type x{{{7,-3,2,-2,4},{4,6,6,4,5},{7,6,4,4,1},{5,-6,-10,-3,-9}},{{-8,0,8,-3,-3},{-7,-4,-9,-2,7},{3,0,5,-10,-7},{7,9,-3,-3,9}},{{4,9,-5,-9,-5},{-1,7,-7,3,-9},{-4,7,6,6,2},{9,6,-4,-8,-3}}};
    SECTION("sort")
    {
        t.sort(policy{});
        REQUIRE(t == tensor_type{-10,-9,-9,-9,-8,-8,-8,-8,-7,-7,-7,-7,-7,-7,-6,-6,-6,-6,-6,-5,-5,-5,-5,-5,-4,-4,-3,-3,-2,-2,-2,-1,-1,-1,-1,0,0,2,3,4,5,6,6,6,7,7,8,8,9,9});
        x.sort(policy{});
        REQUIRE(x == tensor_type{{{-3,-2,2,4,7},{4,4,5,6,6},{1,4,4,6,7},{-10,-9,-6,-3,5}},{{-8,-3,-3,0,8},{-9,-7,-4,-2,7},{-10,-7,0,3,5},{-3,-3,7,9,9}},{{-9,-5,-5,4,9},{-9,-7,-1,3,7},{-4,2,6,6,7},{-8,-4,-3,6,9}}});
    }
    SECTION("sort_axis")
    {
        t.sort(policy{},0);
        REQUIRE(t == tensor_type{-10,-9,-9,-9,-8,-8,-8,-8,-7,-7,-7,-7,-7,-7,-6,-6,-6,-6,-6,-5,-5,-5,-5,-5,-4,-4,-3,-3,-2,-2,-2,-1,-1,-1,-1,0,0,2,3,4,5,6,6,6,7,7,8,8,9,9});
        SECTION("sort_0")
        {
            x.sort(policy{},0);
            REQUIRE(x == tensor_type{{{-8,-3,-5,-9,-5},{-7,-4,-9,-2,-9},{-4,0,4,-10,-7},{5,-6,-10,-8,-9}},{{4,0,2,-3,-3},{-1,6,-7,3,5},{3,6,5,4,1},{7,6,-4,-3,-3}},{{7,9,8,-2,4},{4,7,6,4,7},{7,7,6,6,2},{9,9,-3,-3,9}}});
        }
        SECTION("sort_1")
        {
            x.sort(policy{},1);
            REQUIRE(x == tensor_type{{{4,-6,-10,-3,-9},{5,-3,2,-2,1},{7,6,4,4,4},{7,6,6,4,5}},{{-8,-4,-9,-10,-7},{-7,0,-3,-3,-3},{3,0,5,-3,7},{7,9,8,-2,9}},{{-4,6,-7,-9,-9},{-1,7,-5,-8,-5},{4,7,-4,3,-3},{9,9,6,6,2}}});
        }
    }
    SECTION("sort_axis_comparator")
    {
        t.sort(policy{},0,std::greater<void>{});
        REQUIRE(t == tensor_type{9,9,8,8,7,7,6,6,6,5,4,3,2,0,0,-1,-1,-1,-1,-2,-2,-2,-3,-3,-4,-4,-5,-5,-5,-5,-5,-6,-6,-6,-6,-6,-7,-7,-7,-7,-7,-7,-8,-8,-8,-8,-9,-9,-9,-10});
        x.sort(policy{},1,std::greater<void>{});
        REQUIRE(x == tensor_type{{{7,6,6,4,5},{7,6,4,4,4},{5,-3,2,-2,1},{4,-6,-10,-3,-9}},{{7,9,8,-2,9},{3,0,5,-3,7},{-7,0,-3,-3,-3},{-8,-4,-9,-10,-7}},{{9,9,6,6,2},{4,7,-4,3,-3},{-1,7,-5,-8,-5},{-4,6,-7,-9,-9}}});
    }
}

TEST_CASE("test_tensor_sort_search_member_function_default_policy","[test_tensor]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::take_along_axis;

    const tensor_type t{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}};
    //argsort
    REQUIRE(take_along_axis(t,t.argsort(),-1) == tensor_type{{{-3,-2,3,3},{0,3,3,4}},{{-2,1,3,3},{-2,0,2,3}}});
    REQUIRE(take_along_axis(t,t.argsort(0),0) == tensor_type{{{-3,3,1,-2},{0,3,-2,0}},{{-2,3,3,3},{4,3,3,2}}});
    REQUIRE(take_along_axis(t,t.argsort(1,std::greater<void>{}),1) == tensor_type{{{4,3,3,0},{-3,3,3,-2}},{{0,3,1,3},{-2,3,-2,2}}});
    //argmax
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax() == tensor<int>(20));
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax(0) == tensor<int>{{2,0,2,1,2},{2,1,1,1,1}});
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax(1,true) == tensor<int>{{{1,0,1,1,0}},{{1,0,1,0,0}},{{0,0,0,1,0}}});
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax(2) == tensor<int>{{1,0},{3,2},{0,0}});
    REQUIRE(take_along_axis(t,t.argmax(0,true),0) == tensor_type{{{-2,3,3,3},{4,3,3,2}}});
    REQUIRE(take_along_axis(t,t.argmax(1,true),1) == tensor_type{{{4,3,3,0}},{{0,3,1,3}}});
    REQUIRE(take_along_axis(t,t.argmax(2,true),2) == tensor_type{{{3},{4}},{{3},{3}}});
    //argmin
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin() == tensor<int>(29));
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin(0) == tensor<int>{{0,1,0,0,0},{0,0,2,0,2}});
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin(1,true) == tensor<int>{{{0,1,0,0,1}},{{0,1,0,1,1}},{{1,1,1,0,1}}});
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin(2) == tensor<int>{{2,1},{0,4},{3,4}});
    REQUIRE(take_along_axis(t,t.argmin(0,true),0) == tensor_type{{{-3,3,1,-2},{0,3,-2,0}}});
    REQUIRE(take_along_axis(t,t.argmin(1,true),1) == tensor_type{{{-3,3,3,-2}},{{-2,3,-2,2}}});
    REQUIRE(take_along_axis(t,t.argmin(2,true),2) == tensor_type{{{-3},{0}},{{-2},{-2}}});
    //nonzero
    REQUIRE(t(t.nonzero()) == tensor_type{-3,3,3,-2,4,3,3,-2,3,1,3,3,-2,2});
}

TEST_CASE("test_tensor_sort_search_member_function_policy","[test_tensor]")
{
    using policy = multithreading::exec_pol<4>;
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::take_along_axis;

    const tensor_type t{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}};
    //argsort
    REQUIRE(take_along_axis(t,t.argsort(policy{}),-1) == tensor_type{{{-3,-2,3,3},{0,3,3,4}},{{-2,1,3,3},{-2,0,2,3}}});
    REQUIRE(take_along_axis(t,t.argsort(policy{},0),0) == tensor_type{{{-3,3,1,-2},{0,3,-2,0}},{{-2,3,3,3},{4,3,3,2}}});
    REQUIRE(take_along_axis(t,t.argsort(policy{},1,std::greater<void>{}),1) == tensor_type{{{4,3,3,0},{-3,3,3,-2}},{{0,3,1,3},{-2,3,-2,2}}});
    //argmax
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax(policy{}) == tensor<int>(20));
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax(policy{},0) == tensor<int>{{2,0,2,1,2},{2,1,1,1,1}});
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax(policy{},1,true) == tensor<int>{{{1,0,1,1,0}},{{1,0,1,0,0}},{{0,0,0,1,0}}});
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmax(policy{},2) == tensor<int>{{1,0},{3,2},{0,0}});
    REQUIRE(take_along_axis(t,t.argmax(policy{},0,true),0) == tensor_type{{{-2,3,3,3},{4,3,3,2}}});
    REQUIRE(take_along_axis(t,t.argmax(policy{},1,true),1) == tensor_type{{{4,3,3,0}},{{0,3,1,3}}});
    REQUIRE(take_along_axis(t,t.argmax(policy{},2,true),2) == tensor_type{{{3},{4}},{{3},{3}}});
    //argmin
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin(policy{}) == tensor<int>(29));
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin(policy{},0) == tensor<int>{{0,1,0,0,0},{0,0,2,0,2}});
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin(policy{},1,true) == tensor<int>{{{0,1,0,0,1}},{{0,1,0,1,1}},{{1,1,1,0,1}}});
    REQUIRE(tensor_type{{{10,24,2,3,7},{19,1,18,9,4}},{{11,17,14,27,13},{23,16,25,20,12}},{{29,21,22,5,28},{26,8,6,15,0}}}.argmin(policy{},2) == tensor<int>{{2,1},{0,4},{3,4}});
    REQUIRE(take_along_axis(t,t.argmin(policy{},0,true),0) == tensor_type{{{-3,3,1,-2},{0,3,-2,0}}});
    REQUIRE(take_along_axis(t,t.argmin(policy{},1,true),1) == tensor_type{{{-3,3,3,-2}},{{-2,3,-2,2}}});
    REQUIRE(take_along_axis(t,t.argmin(policy{},2,true),2) == tensor_type{{{-3},{0}},{{-2},{-2}}});
    //nonzero
    REQUIRE(t(t.nonzero()) == tensor_type{-3,3,3,-2,4,3,3,-2,3,1,3,3,-2,2});
}

//indexing
TEST_CASE("test_tensor_indexing_member_function","[test_tensor]")
{
    using value_type = double;
    using gtensor::tensor;
    using gtensor::tensor_close;
    using tensor_type = gtensor::tensor<value_type>;

    //take
    REQUIRE(tensor_type(2).take(tensor<int>{}) == tensor_type{});
    REQUIRE(tensor_type(3).take(tensor<int>{},0) == tensor_type{});
    REQUIRE(tensor_type(2).take(tensor<int>{0}) == tensor_type{2});
    REQUIRE(tensor_type(2).take(tensor<int>{{0,0},{0,0}}) == tensor_type{{2,2},{2,2}});
    REQUIRE(tensor_type{1,2,3,4,5}.take(tensor<int>{}) == tensor_type{});
    REQUIRE(tensor_type{1,2,3,4,5}.take(tensor<int>{1,0,2,0,3,3,1}) == tensor_type{2,1,3,1,4,4,2});
    REQUIRE(tensor_type{1,2,3,4,5}.take(tensor<int>{{1,0,2},{0,3,3}},0) == tensor_type{{2,1,3},{1,4,4}});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{}) == tensor_type{});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{1,2,2,0,3}) == tensor_type{3,3,3,-3,-2});
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{1,0,0,1},0) ==
        tensor_type{{{-2,3,1,3},{0,3,-2,2}},{{-3,3,3,-2},{4,3,3,0}},{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}
    );
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{1,0,1,0},1) ==
        tensor_type{{{4,3,3,0},{-3,3,3,-2},{4,3,3,0},{-3,3,3,-2}},{{0,3,-2,2},{-2,3,1,3},{0,3,-2,2},{-2,3,1,3}}}
    );
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{1,0,3,2,2},2) ==
        tensor_type{{{3,-3,-2,3,3},{3,4,0,3,3}},{{3,-2,3,1,1},{3,0,2,-2,-2}}}
    );
    REQUIRE(tensor_type{{{-3,3,3,-2},{4,3,3,0}},{{-2,3,1,3},{0,3,-2,2}}}.take(tensor<int>{{1,0,1},{0,1,0}},1) ==
        tensor_type{{{{4,3,3,0},{-3,3,3,-2},{4,3,3,0}},{{-3,3,3,-2},{4,3,3,0},{-3,3,3,-2}}},{{{0,3,-2,2},{-2,3,1,3},{0,3,-2,2}},{{-2,3,1,3},{0,3,-2,2},{-2,3,1,3}}}}
    );
}

