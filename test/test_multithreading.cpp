/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <vector>
#include <numeric>
#include <functional>
#include "catch.hpp"
#include "multithreading.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_multithreading_par_task_size","[test_multithreading]")
{
    using size_type = int;
    using multithreading::par_task_size;
    using helpers_for_testing::apply_by_element;

    //0max_par_tasks_number,1min_tasks_per_par_task,2tasks_number,3expected_size,4expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(1,1,0,0,std::vector<size_type>{}),
        std::make_tuple(1,2,0,0,std::vector<size_type>{}),
        std::make_tuple(1,1,1,1,std::vector<size_type>{1}),
        std::make_tuple(1,2,1,0,std::vector<size_type>{}),
        std::make_tuple(1,1,5,1,std::vector<size_type>{5}),
        std::make_tuple(1,2,5,1,std::vector<size_type>{5}),
        std::make_tuple(1,3,5,1,std::vector<size_type>{5}),
        std::make_tuple(1,5,5,1,std::vector<size_type>{5}),
        std::make_tuple(1,6,5,0,std::vector<size_type>{}),
        std::make_tuple(4,1,0,0,std::vector<size_type>{}),
        std::make_tuple(4,1,1,1,std::vector<size_type>{1}),
        std::make_tuple(4,1,3,3,std::vector<size_type>{1,1,1}),
        std::make_tuple(4,1,4,4,std::vector<size_type>{1,1,1,1}),
        std::make_tuple(4,1,5,4,std::vector<size_type>{2,1,1,1}),
        std::make_tuple(4,1,6,4,std::vector<size_type>{2,2,1,1}),
        std::make_tuple(4,1,7,4,std::vector<size_type>{2,2,2,1}),
        std::make_tuple(4,1,8,4,std::vector<size_type>{2,2,2,2}),
        std::make_tuple(8,1,6,6,std::vector<size_type>{1,1,1,1,1,1}),
        std::make_tuple(8,1,60,8,std::vector<size_type>{8,8,8,8,7,7,7,7}),
        std::make_tuple(8,1,111,8,std::vector<size_type>{14,14,14,14,14,14,14,13}),
        std::make_tuple(8,3,111,8,std::vector<size_type>{14,14,14,14,14,14,14,13}),
        std::make_tuple(8,20,111,5,std::vector<size_type>{23,22,22,22,22})
    );

    auto test = [](const auto& t){
        auto max_par_tasks_number = std::get<0>(t);
        auto min_tasks_per_par_task = std::get<1>(t);
        auto tasks_number = std::get<2>(t);
        auto expected_size = std::get<3>(t);
        auto expected_elements = std::get<4>(t);

        par_task_size<size_type> par_sizes{tasks_number,max_par_tasks_number,min_tasks_per_par_task};

        std::vector<int> result_elements{};
        for(auto i=0; i!=par_sizes.size(); ++i){
            result_elements.push_back(par_sizes[i]);
        }
        auto result_size = par_sizes.size();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_elements == expected_elements);
    };
    apply_by_element(test,test_data);
}

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

TEMPLATE_TEST_CASE("test_multithreading_transform_first1","[test_multithreading]",
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
        std::make_tuple(test_vec1,std::cref(test_vec2),std::plus<void>{},std::cref(expected_vec)),
        std::make_tuple(std::vector<value_type>{},std::vector<value_type>{},std::plus<void>{},std::vector<value_type>{}),
        std::make_tuple(std::vector<value_type>{1},std::vector<value_type>{2},std::plus<void>{},std::vector<value_type>{3}),
        std::make_tuple(std::vector<value_type>{1,2,3,4,5},std::vector<value_type>{6,7,8,9,10},std::plus<void>{},std::vector<value_type>{7,9,11,13,15}),
        std::make_tuple(std::vector<value_type>{1,6,3,8,2},std::vector<value_type>{2,7,0,9,1},min,std::vector<value_type>{1,6,0,8,1}),
        std::make_tuple(std::vector<value_type>{1,6,3,8,2},std::vector<value_type>{2,7,0,9,1},max,std::vector<value_type>{2,7,3,9,2})
    );

    auto test = [](const auto& t){
        auto vec1 = std::get<0>(t);
        auto& vec2 = std::get<1>(t);
        auto reduce_f = std::get<2>(t);
        auto& expected = std::get<3>(t);

        multithreading::transform(policy{},vec1.begin(),vec1.end(),vec2.begin(),vec1.begin(),reduce_f);
        REQUIRE(vec1 == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_multithreading_transform_two_range_overload","[test_multithreading]",
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
        std::make_tuple(test_vec1,std::cref(test_vec2),std::plus<void>{},std::cref(expected_vec)),
        std::make_tuple(std::vector<value_type>{},std::vector<value_type>{},std::plus<void>{},std::vector<value_type>{}),
        std::make_tuple(std::vector<value_type>{1},std::vector<value_type>{2},std::plus<void>{},std::vector<value_type>{3}),
        std::make_tuple(std::vector<value_type>{1,2,3,4,5},std::vector<value_type>{6,7,8,9,10},std::plus<void>{},std::vector<value_type>{7,9,11,13,15}),
        std::make_tuple(std::vector<value_type>{1,6,3,8,2},std::vector<value_type>{2,7,0,9,1},min,std::vector<value_type>{1,6,0,8,1}),
        std::make_tuple(std::vector<value_type>{1,6,3,8,2},std::vector<value_type>{2,7,0,9,1},max,std::vector<value_type>{2,7,3,9,2})
    );

    auto test = [](const auto& t){
        auto vec1 = std::get<0>(t);
        auto& vec2 = std::get<1>(t);
        auto reduce_f = std::get<2>(t);
        auto& expected = std::get<3>(t);

        multithreading::transform(policy{},vec1.begin(),vec1.end(),vec2.begin(),reduce_f);
        REQUIRE(vec1 == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_multithreading_copy","[test_multithreading]",
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

    //0vec
    auto test_data = std::make_tuple(
        std::vector<value_type>{},
        std::vector<value_type>{3},
        std::vector<value_type>{1,2,3,4,5,4,3,2,1},
        std::cref(test_vec)
    );

    auto test = [](const auto& v){
        const auto& vec = v;
        const auto& expected = vec;
        std::vector<value_type> dst(vec.size());
        auto dst_last_expected = dst.end();
        auto dst_last_result = multithreading::copy(policy{},vec.begin(),vec.end(),dst.begin());
        REQUIRE(dst == expected);
        REQUIRE(dst_last_result == dst_last_expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_multithreading_inner_product","[test_multithreading]",
    (multithreading::exec_pol<1>),
    (multithreading::exec_pol<2>),
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<8>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using helpers_for_testing::apply_by_element;
    using helpers_for_testing::generate_lehmer;

    const auto n = 1234567;
    std::vector<value_type> test_vec1(n,0);
    std::vector<value_type> test_vec2(n,0);
    generate_lehmer(test_vec1.begin(),test_vec1.end(),[](auto e){return e%3;},123);
    generate_lehmer(test_vec2.begin(),test_vec2.end(),[](auto e){return e%3;},456);

    //0vec1,1vec2,2initial,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::vector<value_type>{2},std::vector<value_type>{3},value_type{0},value_type{6}),
        std::make_tuple(std::vector<value_type>{2},std::vector<value_type>{3},value_type{1},value_type{7}),
        std::make_tuple(std::vector<value_type>{2},std::vector<value_type>{3,4,5,6},value_type{2},value_type{8}),
        std::make_tuple(std::vector<value_type>{2,5},std::vector<value_type>{-2,3},value_type{0},value_type{11}),
        std::make_tuple(std::vector<value_type>{1,2,3,4,5},std::vector<value_type>{2,3,4,5,6},value_type{0},value_type{70}),
        std::make_tuple(std::vector<value_type>{1,2,3,4,5},std::vector<value_type>{2,3,4,5,6},value_type{-1},value_type{69}),
        std::make_tuple(std::cref(test_vec1),std::cref(test_vec2),value_type{0},value_type{1236077}),
        std::make_tuple(std::cref(test_vec2),std::cref(test_vec1),value_type{3},value_type{1236080})
    );

    auto test = [](const auto& t){
        auto& vec1 = std::get<0>(t);
        auto& vec2 = std::get<1>(t);
        auto initial = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = multithreading::inner_product(policy{},vec1.begin(),vec1.end(),vec2.begin(),initial);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}
