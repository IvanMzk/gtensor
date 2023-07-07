#include <vector>
#include "catch.hpp"
#include "builder.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEMPLATE_TEST_CASE("test_builder_empty","[test_builder]",
    //0value_type,1layout
    (std::tuple<double, gtensor::config::c_order>),
    (std::tuple<int, gtensor::config::f_order>)
)
{
    using value_type = std::tuple_element_t<0,TestType>;
    using layout = std::tuple_element_t<1,TestType>;
    using config_type = gtensor::config::default_config;
    using extended_config_type = gtensor::config::extend_config_t<config_type,value_type>;
    using gtensor::tensor;
    using tensor_type = tensor<value_type,layout,extended_config_type>;
    using shape_type = typename tensor_type::shape_type;
    using gtensor::empty;

    REQUIRE(std::is_same_v<decltype(empty<value_type,layout,config_type>(std::declval<std::vector<int>>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(empty<value_type,layout,config_type>(std::declval<std::initializer_list<int>>())),tensor_type>);

    REQUIRE(empty<value_type,layout,config_type>(std::vector<int>{}).shape() == shape_type{});
    REQUIRE(empty<value_type,layout,config_type>(std::vector<int>{0,2,3}).shape() == shape_type{0,2,3});
    REQUIRE(empty<value_type,layout,config_type>(std::vector<int>{1,2,3,4}).shape() == shape_type{1,2,3,4});
    REQUIRE(empty<value_type,layout,config_type>({0}).shape() == shape_type{0});
    REQUIRE(empty<value_type,layout,config_type>({10}).shape() == shape_type{10});
    REQUIRE(empty<value_type,layout,config_type>({4,2,3,1}).shape() == shape_type{4,2,3,1});
}

TEMPLATE_TEST_CASE("test_builder_full_zeros_ones","[test_builder]",
    //0value_type,1layout
    (std::tuple<double, gtensor::config::c_order>),
    (std::tuple<int, gtensor::config::f_order>)
)
{
    using value_type = std::tuple_element_t<0,TestType>;
    using layout = std::tuple_element_t<1,TestType>;
    using config_type = gtensor::config::default_config;
    using extended_config_type = gtensor::config::extend_config_t<config_type,value_type>;
    using gtensor::tensor;
    using tensor_type = tensor<value_type,layout,extended_config_type>;
    using gtensor::full;
    using gtensor::zeros;
    using gtensor::ones;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(full<value_type,layout,config_type>(std::declval<std::vector<int>>(),std::declval<value_type>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(full<value_type,layout,config_type>(std::declval<std::initializer_list<int>>(),std::declval<value_type>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(zeros<value_type,layout,config_type>(std::declval<std::vector<int>>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(zeros<value_type,layout,config_type>(std::declval<std::initializer_list<int>>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(ones<value_type,layout,config_type>(std::declval<std::vector<int>>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(ones<value_type,layout,config_type>(std::declval<std::initializer_list<int>>())),tensor_type>);
    //result,1expected
    auto test_data = std::make_tuple(
        //full
        std::make_tuple(full<value_type,layout,config_type>({0},1),tensor_type{}),
        std::make_tuple(full<value_type,layout,config_type>({5},2),tensor_type{2,2,2,2,2}),
        std::make_tuple(full<value_type,layout,config_type>({2,3},3),tensor_type{{3,3,3},{3,3,3}}),
        std::make_tuple(full<value_type,layout,config_type>(std::vector<int>{4,2},4),tensor_type{{4,4},{4,4},{4,4},{4,4}}),
        //zeros
        std::make_tuple(zeros<value_type,layout,config_type>({0}),tensor_type{}),
        std::make_tuple(zeros<value_type,layout,config_type>({2,3}),tensor_type{{0,0,0},{0,0,0}}),
        std::make_tuple(zeros<value_type,layout,config_type>(std::vector<int>{4,2}),tensor_type{{0,0},{0,0},{0,0},{0,0}}),
        //ones
        std::make_tuple(ones<value_type,layout,config_type>({0}),tensor_type{}),
        std::make_tuple(ones<value_type,layout,config_type>({2,3}),tensor_type{{1,1,1},{1,1,1}}),
        std::make_tuple(ones<value_type,layout,config_type>(std::vector<int>{4,2}),tensor_type{{1,1},{1,1},{1,1},{1,1}})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_builder_identity","[test_builder]",
    //0value_type,1layout,2traverse order
    (std::tuple<double, gtensor::config::c_order, gtensor::config::c_order>),
    (std::tuple<int, gtensor::config::f_order, gtensor::config::f_order>),
    (std::tuple<double, gtensor::config::c_order, gtensor::config::f_order>),
    (std::tuple<int, gtensor::config::f_order, gtensor::config::c_order>)
)
{
    using value_type = std::tuple_element_t<0,TestType>;
    using layout = std::tuple_element_t<1,TestType>;
    using traverse_order = std::tuple_element_t<2,TestType>;
    using config_type = test_config::config_order_selector_t<traverse_order>;
    using extended_config_type = gtensor::config::extend_config_t<config_type,value_type>;
    using gtensor::tensor;
    using tensor_type = tensor<value_type,layout,extended_config_type>;
    using gtensor::identity;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(identity<value_type,layout,config_type>(std::declval<int>())),tensor_type>);

    //0n,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(0,tensor_type{}.reshape(0,0)),
        std::make_tuple(1,tensor_type{{1}}),
        std::make_tuple(2,tensor_type{{1,0},{0,1}}),
        std::make_tuple(3,tensor_type{{1,0,0},{0,1,0},{0,0,1}}),
        std::make_tuple(4,tensor_type{{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}),
        std::make_tuple(5,tensor_type{{1,0,0,0,0},{0,1,0,0,0},{0,0,1,0,0},{0,0,0,1,0},{0,0,0,0,1}})
    );
    auto test = [](const auto& t){
        auto n = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto result = identity<value_type,layout,config_type>(n);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_builder_eye","[test_builder]",
    //0value_type,1layout,2traverse order
    (std::tuple<double, gtensor::config::c_order, gtensor::config::c_order>),
    (std::tuple<int, gtensor::config::f_order, gtensor::config::f_order>),
    (std::tuple<double, gtensor::config::c_order, gtensor::config::f_order>),
    (std::tuple<int, gtensor::config::f_order, gtensor::config::c_order>)
)
{
    using value_type = std::tuple_element_t<0,TestType>;
    using layout = std::tuple_element_t<1,TestType>;
    using traverse_order = std::tuple_element_t<2,TestType>;
    using config_type = test_config::config_order_selector_t<traverse_order>;
    using extended_config_type = gtensor::config::extend_config_t<config_type,value_type>;
    using gtensor::tensor;
    using tensor_type = tensor<value_type,layout,extended_config_type>;
    using gtensor::eye;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(eye<value_type,layout,config_type>(std::declval<int>(),std::declval<int>(),std::declval<int>())),tensor_type>);

    //0n,1m,2k,3expected
    auto test_data = std::make_tuple(
        //n=m, k=0
        std::make_tuple(0,0,0,tensor_type{}.reshape(0,0)),
        std::make_tuple(1,1,0,tensor_type{{1}}),
        std::make_tuple(2,2,0,tensor_type{{1,0},{0,1}}),
        std::make_tuple(3,3,0,tensor_type{{1,0,0},{0,1,0},{0,0,1}}),
        std::make_tuple(5,5,0,tensor_type{{1,0,0,0,0},{0,1,0,0,0},{0,0,1,0,0},{0,0,0,1,0},{0,0,0,0,1}}),
        //n=m,k!=0
        std::make_tuple(1,1,1,tensor_type{{0}}),
        std::make_tuple(1,1,-1,tensor_type{{0}}),
        std::make_tuple(2,2,1,tensor_type{{0,1},{0,0}}),
        std::make_tuple(2,2,2,tensor_type{{0,0},{0,0}}),
        std::make_tuple(2,2,-1,tensor_type{{0,0},{1,0}}),
        std::make_tuple(2,2,-2,tensor_type{{0,0},{0,0}}),
        std::make_tuple(3,3,1,tensor_type{{0,1,0},{0,0,1},{0,0,0}}),
        std::make_tuple(3,3,2,tensor_type{{0,0,1},{0,0,0},{0,0,0}}),
        std::make_tuple(3,3,3,tensor_type{{0,0,0},{0,0,0},{0,0,0}}),
        std::make_tuple(3,3,-1,tensor_type{{0,0,0},{1,0,0},{0,1,0}}),
        std::make_tuple(3,3,-2,tensor_type{{0,0,0},{0,0,0},{1,0,0}}),
        std::make_tuple(3,3,-3,tensor_type{{0,0,0},{0,0,0},{0,0,0}}),
        std::make_tuple(5,5,1,tensor_type{{0,1,0,0,0},{0,0,1,0,0},{0,0,0,1,0},{0,0,0,0,1},{0,0,0,0,0}}),
        std::make_tuple(5,5,2,tensor_type{{0,0,1,0,0},{0,0,0,1,0},{0,0,0,0,1},{0,0,0,0,0},{0,0,0,0,0}}),
        std::make_tuple(5,5,4,tensor_type{{0,0,0,0,1},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}),
        std::make_tuple(5,5,6,tensor_type{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}),
        std::make_tuple(5,5,-2,tensor_type{{0,0,0,0,0},{0,0,0,0,0},{1,0,0,0,0},{0,1,0,0,0},{0,0,1,0,0}}),
        std::make_tuple(5,5,-3,tensor_type{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{1,0,0,0,0},{0,1,0,0,0}}),
        std::make_tuple(5,5,-4,tensor_type{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{1,0,0,0,0}}),
        std::make_tuple(5,5,-6,tensor_type{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}),
        //n!=m,k=0
        std::make_tuple(3,5,0,tensor_type{{1,0,0,0,0},{0,1,0,0,0},{0,0,1,0,0}}),
        std::make_tuple(5,3,0,tensor_type{{1,0,0},{0,1,0},{0,0,1},{0,0,0},{0,0,0}}),
        //n!=m,n<m,k!=0
        std::make_tuple(3,5,1,tensor_type{{0,1,0,0,0},{0,0,1,0,0},{0,0,0,1,0}}),
        std::make_tuple(3,5,2,tensor_type{{0,0,1,0,0},{0,0,0,1,0},{0,0,0,0,1}}),
        std::make_tuple(3,5,3,tensor_type{{0,0,0,1,0},{0,0,0,0,1},{0,0,0,0,0}}),
        std::make_tuple(3,5,4,tensor_type{{0,0,0,0,1},{0,0,0,0,0},{0,0,0,0,0}}),
        std::make_tuple(3,5,5,tensor_type{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}),
        std::make_tuple(3,5,6,tensor_type{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}),
        std::make_tuple(3,5,-1,tensor_type{{0,0,0,0,0},{1,0,0,0,0},{0,1,0,0,0}}),
        std::make_tuple(3,5,-2,tensor_type{{0,0,0,0,0},{0,0,0,0,0},{1,0,0,0,0}}),
        std::make_tuple(3,5,-3,tensor_type{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}),
        std::make_tuple(3,5,-4,tensor_type{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}),
        //n!=m,n>m,k!=0
        std::make_tuple(5,3,1,tensor_type{{0,1,0},{0,0,1},{0,0,0},{0,0,0},{0,0,0}}),
        std::make_tuple(5,3,2,tensor_type{{0,0,1},{0,0,0},{0,0,0},{0,0,0},{0,0,0}}),
        std::make_tuple(5,3,3,tensor_type{{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0}}),
        std::make_tuple(5,3,4,tensor_type{{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0}}),
        std::make_tuple(5,3,-1,tensor_type{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,0,0}}),
        std::make_tuple(5,3,-2,tensor_type{{0,0,0},{0,0,0},{1,0,0},{0,1,0},{0,0,1}}),
        std::make_tuple(5,3,-3,tensor_type{{0,0,0},{0,0,0},{0,0,0},{1,0,0},{0,1,0}}),
        std::make_tuple(5,3,-4,tensor_type{{0,0,0},{0,0,0},{0,0,0},{0,0,0},{1,0,0}}),
        std::make_tuple(5,3,-5,tensor_type{{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0}}),
        std::make_tuple(5,3,-6,tensor_type{{0,0,0},{0,0,0},{0,0,0},{0,0,0},{0,0,0}})
    );
    auto test = [](const auto& t){
        auto n = std::get<0>(t);
        auto m = std::get<1>(t);
        auto k = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = eye<value_type,layout,config_type>(n,m,k);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_builder_empty_like","[test_builder]",
    //0value_type,1layout,2traverse order
    (std::tuple<double, gtensor::config::c_order, gtensor::config::c_order>),
    (std::tuple<int, gtensor::config::f_order, gtensor::config::f_order>),
    (std::tuple<double, gtensor::config::c_order, gtensor::config::f_order>),
    (std::tuple<int, gtensor::config::f_order, gtensor::config::c_order>)
)
{
    using value_type = std::tuple_element_t<0,TestType>;
    using layout = std::tuple_element_t<1,TestType>;
    using traverse_order = std::tuple_element_t<2,TestType>;
    using config_type = test_config::config_order_selector_t<traverse_order>;
    using extended_config_type = gtensor::config::extend_config_t<config_type,value_type>;
    using gtensor::tensor;
    using tensor_type = tensor<value_type,layout,extended_config_type>;
    using shape_type = typename tensor_type::shape_type;
    using gtensor::empty_like;

    REQUIRE(std::is_same_v<decltype(empty_like(std::declval<tensor_type>())),tensor_type>);

    REQUIRE(empty_like(tensor_type(2)).shape() == shape_type{});
    REQUIRE(empty_like(tensor_type{}).shape() == shape_type{0});
    REQUIRE(empty_like(tensor_type{1,2,3,4}).shape() == shape_type{4});
    REQUIRE(empty_like(tensor_type{{1,2},{3,4},{5,6}}).shape() == shape_type{3,2});
    REQUIRE(empty_like(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}).shape() == shape_type{2,2,2});
}

TEMPLATE_TEST_CASE("test_builder_full_zeros_ones_like","[test_builder]",
    //0value_type,1layout,2traverse order
    (std::tuple<double, gtensor::config::c_order, gtensor::config::c_order>),
    (std::tuple<int, gtensor::config::f_order, gtensor::config::f_order>),
    (std::tuple<double, gtensor::config::c_order, gtensor::config::f_order>),
    (std::tuple<int, gtensor::config::f_order, gtensor::config::c_order>)
)
{
    using value_type = std::tuple_element_t<0,TestType>;
    using layout = std::tuple_element_t<1,TestType>;
    using traverse_order = std::tuple_element_t<2,TestType>;
    using config_type = test_config::config_order_selector_t<traverse_order>;
    using extended_config_type = gtensor::config::extend_config_t<config_type,value_type>;
    using gtensor::tensor;
    using tensor_type = tensor<value_type,layout,extended_config_type>;
    using gtensor::full_like;
    using gtensor::zeros_like;
    using gtensor::ones_like;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(full_like(std::declval<tensor_type>(),std::declval<value_type>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(zeros_like(std::declval<tensor_type>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(ones_like(std::declval<tensor_type>())),tensor_type>);
    //result,1expected
    auto test_data = std::make_tuple(
        //full
        std::make_tuple(full_like(tensor_type(2),1),tensor_type(1)),
        std::make_tuple(full_like(tensor_type{},1),tensor_type{}),
        std::make_tuple(full_like(tensor_type{1,2,3,4,5},2),tensor_type{2,2,2,2,2}),
        std::make_tuple(full_like(tensor_type{{1,2,3},{4,5,6}},3),tensor_type{{3,3,3},{3,3,3}}),
        //zeros
        std::make_tuple(zeros_like(tensor_type(2)),tensor_type(0)),
        std::make_tuple(zeros_like(tensor_type{}),tensor_type{}),
        std::make_tuple(zeros_like(tensor_type{1,2,3,4,5}),tensor_type{0,0,0,0,0}),
        std::make_tuple(zeros_like(tensor_type{{1,2,3},{4,5,6}}),tensor_type{{0,0,0},{0,0,0}}),
        //ones
        std::make_tuple(ones_like(tensor_type(2)),tensor_type(1)),
        std::make_tuple(ones_like(tensor_type{}),tensor_type{}),
        std::make_tuple(ones_like(tensor_type{1,2,3,4,5}),tensor_type{1,1,1,1,1}),
        std::make_tuple(ones_like(tensor_type{{1,2,3},{4,5,6}}),tensor_type{{1,1,1},{1,1,1}})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_builder_arange","[test_builder]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using gtensor::tensor_close;
    using gtensor::arange;
    using helpers_for_testing::apply_by_element;

    //0start,1stop,2step,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(0,0,1,tensor_type{}),
        std::make_tuple(0,10,1,tensor_type{0,1,2,3,4,5,6,7,8,9}),
        std::make_tuple(0,10,2,tensor_type{0,2,4,6,8}),
        std::make_tuple(0,10,3,tensor_type{0,3,6,9}),
        std::make_tuple(2,10,2,tensor_type{2,4,6,8}),
        std::make_tuple(2,10,3,tensor_type{2,5,8}),
        std::make_tuple(2,11,2,tensor_type{2,4,6,8,10}),
        std::make_tuple(3,10,1,tensor_type{3,4,5,6,7,8,9}),
        std::make_tuple(3,10,2,tensor_type{3,5,7,9}),
        std::make_tuple(3,10,3,tensor_type{3,6,9}),
        std::make_tuple(3,11,2,tensor_type{3,5,7,9}),
        std::make_tuple(1.0,5.0,1.3,tensor_type{1.0,2.3,3.6,4.9}),
        std::make_tuple(0.0,1.0,0.08,tensor_type{0.0,0.08,0.16,0.24,0.32,0.4,0.48,0.56,0.64,0.72,0.8,0.88,0.96})
    );
    auto test = [](const auto& t){
        auto start = std::get<0>(t);
        auto stop = std::get<1>(t);
        auto step = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = arange<value_type>(start,stop,step);
        REQUIRE(tensor_close(result,expected));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_builder_arange_overload","[test_builder]")
{
    using gtensor::tensor;
    using gtensor::arange;
    using helpers_for_testing::apply_by_element;

    REQUIRE(arange<int>(5) == tensor<int>{0,1,2,3,4});
    REQUIRE(arange<double>(10) == tensor<double>{0,1,2,3,4,5,6,7,8,9});

    REQUIRE(arange<int>(2.2,5.0,0.5) == tensor<int>{2,2,2,2,2,2});
    REQUIRE(arange<double>(2.2,5.0,0.5) == tensor<double>{2.2,2.7,3.2,3.7,4.2,4.7});
}

TEST_CASE("test_builder_linspace","[test_builder]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using gtensor::tensor_close;
    using gtensor::linspace;
    using helpers_for_testing::apply_by_element;

    //0start,1stop,2num,3axis,4expected
    auto test_data = std::make_tuple(
        //numeric interval
        std::make_tuple(0,0,0,0,tensor_type{}),
        std::make_tuple(0,1,5,0,tensor_type{0.0,0.25,0.5,0.75,1.0}),
        std::make_tuple(3,8,11,0,tensor_type{3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0}),
        std::make_tuple(2,4,10,0,tensor_type{2.0,2.222,2.444,2.667,2.889,3.111,3.333,3.556,3.778,4.0}),
        std::make_tuple(1.3,3.7,10,0,tensor_type{1.3,1.567,1.833,2.1,2.367,2.633,2.9,3.167,3.433,3.7}),
        //tensor interval
        std::make_tuple(tensor_type{},tensor_type{},5,-1,tensor_type{}.reshape(0,5)),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(2,0),5,-1,tensor_type{}.reshape(2,0,5)),
        std::make_tuple(tensor_type{{{0}},{{0}},{{0}}},tensor_type{}.reshape(2,0),5,1,tensor_type{}.reshape(3,5,2,0)),
        std::make_tuple(0,tensor_type{1,2,3},5,-1,tensor_type{{0.0,0.25,0.5,0.75,1.0},{0.0,0.5,1.0,1.5,2.0},{0.0,0.75,1.5,2.25,3.0}}),
        std::make_tuple(tensor_type{1,2,3},4,6,0,tensor_type{{1.0,2.0,3.0},{1.6,2.4,3.2},{2.2,2.8,3.4},{2.8,3.2,3.6},{3.4,3.6,3.8},{4.0,4.0,4.0}}),
        std::make_tuple(
            tensor_type{1.1,2.2,3.3},
            tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},
            5,
            -1,
            tensor_type{{{1.1,1.925,2.75,3.575,4.4},{2.2,3.025,3.85,4.675,5.5},{3.3,4.125,4.95,5.775,6.6}},{{1.1,2.75,4.4,6.05,7.7},{2.2,3.85,5.5,7.15,8.8},{3.3,4.95,6.6,8.25,9.9}}}
        )
    );
    auto test = [](const auto& t){
        auto start = std::get<0>(t);
        auto stop = std::get<1>(t);
        auto num = std::get<2>(t);
        auto axis = std::get<3>(t);
        auto expected = std::get<4>(t);
        auto result = linspace<value_type>(start,stop,num,axis);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

