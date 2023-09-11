#include <vector>
#include <numeric>
#include <functional>
#include "catch.hpp"
#include "multithreading.hpp"
#include "helpers_for_testing.hpp"

TEMPLATE_TEST_CASE("test_multithreading_reduce","[test_multithreading]",
    (multithreading::exec_pol<1>),
    (multithreading::exec_pol<2>),
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<8>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = std::size_t;
    using helpers_for_testing::apply_by_element;

    const std::size_t n = 123456789;
    std::vector<value_type> test_vec(n);
    std::iota(test_vec.begin(),test_vec.end(),1);
    const value_type test_vec_sum = (n*(n+1))/value_type{2};

    auto min = [](const auto& l, const auto& r){
        return l<r ? l : r;
    };
    auto max = [](const auto& l, const auto& r){
        return l<r ? r : l;
    };

    //0vec,1reduce_f,2initial,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::cref(test_vec),std::plus<void>{},value_type{0},test_vec_sum+value_type{0}),
        std::make_tuple(std::cref(test_vec),std::plus<void>{},value_type{123},test_vec_sum+value_type{123}),
        std::make_tuple(std::cref(test_vec),min,value_type{123},value_type{1}),
        std::make_tuple(std::cref(test_vec),min,value_type{0},value_type{0}),
        std::make_tuple(std::cref(test_vec),max,value_type{123},value_type{123456789}),
        std::make_tuple(std::cref(test_vec),max,value_type{1234567891},value_type{1234567891}),
        std::make_tuple(std::vector<value_type>{},std::plus<void>{},value_type{2},value_type{2}),
        std::make_tuple(std::vector<value_type>{3},std::plus<void>{},value_type{2},value_type{5}),
        std::make_tuple(std::vector<value_type>{1,1,3,2,4},std::plus<void>{},value_type{4},value_type{15})
    );

    auto test = [](const auto& t){
        auto& vec = std::get<0>(t);
        auto reduce_f = std::get<1>(t);
        auto initial = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = multithreading::reduce(policy{},vec.begin(),vec.end(),initial,reduce_f);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_multithreading_transform","[test_multithreading]",
    (multithreading::exec_pol<1>),
    (multithreading::exec_pol<2>),
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<8>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = std::size_t;
    using helpers_for_testing::apply_by_element;

    const std::size_t n = 123456789;
    std::vector<value_type> test_vec1(n);
    std::iota(test_vec1.begin(),test_vec1.end(),1);
    std::vector<value_type> test_vec2(n);
    std::iota(test_vec2.rbegin(),test_vec2.rend(),1);
    std::vector<value_type> expected_vec(n,n+1);


    auto min = [](const auto& l, const auto& r){
        return l<r ? l : r;
    };
    auto max = [](const auto& l, const auto& r){
        return l<r ? r : l;
    };

    //0vec1,1vec2,2reduce_f,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::cref(test_vec1),std::cref(test_vec2),std::plus<void>{},std::cref(expected_vec)),
        std::make_tuple(std::vector<value_type>{},std::vector<value_type>{},std::plus<void>{},std::vector<value_type>{}),
        std::make_tuple(std::vector<value_type>{1},std::vector<value_type>{2},std::plus<void>{},std::vector<value_type>{3}),
        std::make_tuple(std::vector<value_type>{1,2,3,4,5},std::vector<value_type>{6,7,8,9,10},std::plus<void>{},std::vector<value_type>{7,9,11,13,15}),
        std::make_tuple(std::vector<value_type>{1,6,3,8,2},std::vector<value_type>{2,7,0,9,1},min,std::vector<value_type>{1,6,0,8,1}),
        std::make_tuple(std::vector<value_type>{1,6,3,8,2},std::vector<value_type>{2,7,0,9,1},max,std::vector<value_type>{2,7,3,9,2})
    );

    auto test = [](const auto& t){
        auto& vec1 = std::get<0>(t);
        auto& vec2 = std::get<1>(t);
        auto reduce_f = std::get<2>(t);
        auto& expected = std::get<3>(t);

        std::vector<value_type> result(vec1.size());
        multithreading::transform(policy{},vec1.begin(),vec1.end(),vec2.begin(),result.begin(),reduce_f);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

