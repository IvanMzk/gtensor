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

    REQUIRE(arange<int>(5) == tensor<int>{0,1,2,3,4});
    REQUIRE(arange<double>(10) == tensor<double>{0,1,2,3,4,5,6,7,8,9});

    REQUIRE(arange<int>(2.2,5.0,0.5) == tensor<int>{2,2,2,2,2,2});
    REQUIRE(arange<double>(2.2,5.0,0.5) == tensor<double>{2.2,2.7,3.2,3.7,4.2,4.7});
}

TEMPLATE_TEST_CASE("test_builder_linspace","[test_builder]",
    int,
    double
)
{
    using requested_value_type = TestType;
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using gtensor::tensor_close;
    using gtensor::linspace;
    using helpers_for_testing::apply_by_element;

    //0start,1stop,2num,3end_point,4axis,5expected
    auto test_data = std::make_tuple(
        //end_point true
        //numeric interval
        std::make_tuple(0,0,0,true,0,tensor_type{}),
        std::make_tuple(0,1,5,true,0,tensor_type{0.0,0.25,0.5,0.75,1.0}),
        std::make_tuple(3,8,11,true,0,tensor_type{3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0}),
        std::make_tuple(2,4,10,true,0,tensor_type{2.0,2.222,2.444,2.667,2.889,3.111,3.333,3.556,3.778,4.0}),
        std::make_tuple(1.3,3.7,10,true,0,tensor_type{1.3,1.567,1.833,2.1,2.367,2.633,2.9,3.167,3.433,3.7}),
        //tensor interval
        std::make_tuple(tensor_type{},tensor_type{},5,true,-1,tensor_type{}.reshape(0,5)),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(2,0),5,true,-1,tensor_type{}.reshape(2,0,5)),
        std::make_tuple(tensor_type{{{0}},{{0}},{{0}}},tensor_type{}.reshape(2,0),5,true,1,tensor_type{}.reshape(3,5,2,0)),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},0,true,-1,tensor_type{}.reshape(2,3,0)),
        std::make_tuple(0,tensor_type{1,2,3},5,true,-1,tensor_type{{0.0,0.25,0.5,0.75,1.0},{0.0,0.5,1.0,1.5,2.0},{0.0,0.75,1.5,2.25,3.0}}),
        std::make_tuple(tensor_type{1,2,3},4,6,true,0,tensor_type{{1.0,2.0,3.0},{1.6,2.4,3.2},{2.2,2.8,3.4},{2.8,3.2,3.6},{3.4,3.6,3.8},{4.0,4.0,4.0}}),
        std::make_tuple(
            tensor_type{1.1,2.2,3.3},
            tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},
            5,
            true,
            -1,
            tensor_type{{{1.1,1.925,2.75,3.575,4.4},{2.2,3.025,3.85,4.675,5.5},{3.3,4.125,4.95,5.775,6.6}},{{1.1,2.75,4.4,6.05,7.7},{2.2,3.85,5.5,7.15,8.8},{3.3,4.95,6.6,8.25,9.9}}}
        ),
        //end_point false
        std::make_tuple(1.3,3.7,10,false,0,tensor_type{1.3,1.54,1.78,2.02,2.26,2.5,2.74,2.98,3.22,3.46}),
        std::make_tuple(
            tensor_type{1.1,2.2,3.3},
            tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},
            5,
            false,
            -1,
            tensor_type{{{1.1,1.76,2.42,3.08,3.74},{2.2,2.86,3.52,4.18,4.84},{3.3,3.96,4.62,5.28,5.94}},{{1.1,2.42,3.74,5.06,6.38},{2.2,3.52,4.84,6.16,7.48},{3.3,4.62,5.94,7.26,8.58}}}
        ),
        //reverse
        std::make_tuple(3.7,1.3,10,true,0,tensor_type{3.7,3.433,3.167,2.9,2.633,2.367,2.1,1.833,1.567,1.3}),
        std::make_tuple(3.7,1.3,10,false,0,tensor_type{3.7,3.46,3.22,2.98,2.74,2.5,2.26,2.02,1.78,1.54}),
        std::make_tuple(
            tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},
            tensor_type{1.1,2.2,3.3},
            5,
            true,
            -1,
            tensor_type{{{4.4,3.575,2.75,1.925,1.1},{5.5,4.675,3.85,3.025,2.2},{6.6,5.775,4.95,4.125,3.3}},{{7.7,6.05,4.4,2.75,1.1},{8.8,7.15,5.5,3.85,2.2},{9.9,8.25,6.6,4.95,3.3}}}
        )
    );
    auto test = [](const auto& t){
        auto start = std::get<0>(t);
        auto stop = std::get<1>(t);
        auto num = std::get<2>(t);
        auto end_point = std::get<3>(t);
        auto axis = std::get<4>(t);
        auto expected = std::get<5>(t);
        auto result = linspace<requested_value_type>(start,stop,num,end_point,axis);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_builder_logspace","[test_builder]",
    int,
    double
)
{
    using requested_value_type = TestType;
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using gtensor::tensor_close;
    using gtensor::logspace;
    using helpers_for_testing::apply_by_element;

    //0start,1stop,2num,3end_point,4base,5axis,6expected
    auto test_data = std::make_tuple(
        //end_point true
        //numeric interval
        std::make_tuple(0,0,0,true,10,0,tensor_type{}),
        std::make_tuple(0,1,5,true,10,0,tensor_type{1.0,1.778,3.162,5.623,10.0}),
        std::make_tuple(3,8,11,true,2,0,tensor_type{8.0,11.314,16.0,22.627,32.0,45.255,64.0,90.51,128.0,181.019,256.0}),
        //tensor interval
        std::make_tuple(tensor_type{},tensor_type{},5,true,10,-1,tensor_type{}.reshape(0,5)),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(2,0),5,true,10,-1,tensor_type{}.reshape(2,0,5)),
        std::make_tuple(tensor_type{{{0}},{{0}},{{0}}},tensor_type{}.reshape(2,0),5,true,10,1,tensor_type{}.reshape(3,5,2,0)),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},0,true,10,-1,tensor_type{}.reshape(2,3,0)),
        std::make_tuple(0,tensor_type{1,2,3},5,true,10,-1,tensor_type{{1.0,1.778,3.162,5.623,10.0},{1.0,3.162,10.,31.623,100.0},{1.0,5.623,31.623,177.828,1000.0}}),
        std::make_tuple(
            tensor_type{1.1,2.2,3.3},
            tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},
            5,
            true,
            2,
            -1,
            tensor_type{{{2.144,3.797,6.727,11.917,21.112},{4.595,8.14,14.42,25.546,45.255},{9.849,17.448,30.91,54.758,97.006}},{{2.144,6.727,21.112,66.257,207.937},{4.595,14.42,45.255,142.025,445.722},{9.849,30.91,97.006,304.437,955.426}}}
        ),
        //end_point false
        std::make_tuple(0,1,5,false,10,0,tensor_type{1.0,1.585,2.512,3.981,6.31}),
        std::make_tuple(
            tensor_type{1.1,2.2,3.3},
            tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},
            5,
            false,
            2,
            -1,
            tensor_type{{{2.144,3.387,5.352,8.456,13.361},{4.595,7.26,11.472,18.126,28.641},{9.849,15.562,24.59,38.854,61.393}},{{2.144,5.352,13.361,33.359,83.286},{4.595,11.472,28.641,71.506,178.527},{9.849,24.59,61.393,153.277,382.681}}}
        ),
        //reverse
        std::make_tuple(3.7,1.3,10,true,10,0,tensor_type{5011.872,2712.273,1467.799,794.328,429.866,232.631,125.893,68.129,36.869,19.953}),
        std::make_tuple(3.7,1.3,10,false,10,0,tensor_type{5011.872,2884.032,1659.587,954.993,549.541,316.228,181.97,104.713,60.256,34.674})
    );
    auto test = [](const auto& t){
        auto start = std::get<0>(t);
        auto stop = std::get<1>(t);
        auto num = std::get<2>(t);
        auto end_point = std::get<3>(t);
        auto base = std::get<4>(t);
        auto axis = std::get<5>(t);
        auto expected = std::get<6>(t);
        auto result = logspace<requested_value_type>(start,stop,num,end_point,base,axis);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_builder_geomspace","[test_builder]",
    int,
    double
)
{
    using requested_value_type = TestType;
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using gtensor::tensor_close;
    using gtensor::geomspace;
    using helpers_for_testing::apply_by_element;

    //0start,1stop,2num,3end_point,4axis,5expected
    auto test_data = std::make_tuple(
        //end_point true
        //numeric interval
        std::make_tuple(1,1,5,true,0,tensor_type{1,1,1,1,1}),
        std::make_tuple(1,2,5,true,0,tensor_type{1.0,1.189,1.414,1.682,2.0}),
        std::make_tuple(1.3,3.7,10,true,0,tensor_type{1.3,1.46,1.64,1.842,2.069,2.324,2.611,2.933,3.294,3.7}),
        //tensor interval
        std::make_tuple(tensor_type{},tensor_type{},5,true,-1,tensor_type{}.reshape(0,5)),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(2,0),5,true,-1,tensor_type{}.reshape(2,0,5)),
        std::make_tuple(tensor_type{{{0}},{{0}},{{0}}},tensor_type{}.reshape(2,0),5,true,1,tensor_type{}.reshape(3,5,2,0)),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},0,true,-1,tensor_type{}.reshape(2,3,0)),
        std::make_tuple(1,tensor_type{1,2,3},5,true,-1,tensor_type{{1.0,1.0,1.0,1.0,1.0},{1.0,1.189,1.414,1.682,2.0},{1.0,1.316,1.732,2.28,3.0}}),
        std::make_tuple(
            tensor_type{1.1,2.2,3.3},
            tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},
            5,
            true,
            -1,
            tensor_type{{{1.1,1.556,2.2,3.111,4.4},{2.2,2.766,3.479,4.374,5.5},{3.3,3.924,4.667,5.55,6.6}},{{1.1,1.789,2.91,4.734,7.7},{2.2,3.111,4.4,6.223,8.8},{3.3,4.343,5.716,7.522,9.9}}}
        ),
        //end_point false
        std::make_tuple(1.3,3.7,10,false,0,tensor_type{1.3,1.443,1.602,1.779,1.975,2.193,2.435,2.703,3.002,3.333}),
        std::make_tuple(
            tensor_type{1.1,2.2,3.3},
            tensor_type{{4.4,5.5,6.6},{7.7,8.8,9.9}},
            5,
            false,
            0,
            tensor_type{{{1.1,2.2,3.3},{1.1,2.2,3.3}},{{1.451,2.642,3.791},{1.623,2.903,4.111}},{{1.915,3.174,4.354},{2.396,3.83,5.121}},{{2.527,3.812,5.002},{3.536,5.054,6.38}},{{3.335,4.579,5.746},{5.218,6.669,7.947}}}
        ),
        //reverse
        std::make_tuple(3.7,1.3,10,true,0,tensor_type{3.7,3.294,2.933,2.611,2.324,2.069,1.842,1.64,1.46,1.3}),
        std::make_tuple(3.7,1.3,10,false,0,tensor_type{3.7,3.333,3.002,2.703,2.435,2.193,1.975,1.779,1.602,1.443})
    );
    auto test = [](const auto& t){
        auto start = std::get<0>(t);
        auto stop = std::get<1>(t);
        auto num = std::get<2>(t);
        auto end_point = std::get<3>(t);
        auto axis = std::get<4>(t);
        auto expected = std::get<5>(t);
        auto result = geomspace<requested_value_type>(start,stop,num,end_point,axis);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_builder_space_default_args","[test_builder]",
    int,
    double
)
{
    using requested_value_type = TestType;
    using tensor_type = gtensor::tensor<double>;
    using gtensor::tensor_close;
    using gtensor::linspace;
    using gtensor::logspace;
    using gtensor::geomspace;

    REQUIRE(
        tensor_close(
            linspace<requested_value_type>(0,1),
            tensor_type{
                0.0,0.02,0.041,0.061,0.082,0.102,0.122,0.143,0.163,0.184,0.204,0.224,0.24,0.265,0.286,0.306,0.327,0.347,0.367,0.388,0.408,0.429,0.449,0.469,0.49,
                0.51,0.531,0.551,0.571,0.592,0.612,0.633,0.653,0.673,0.694,0.714,0.735,0.755,0.776,0.796,0.816,0.837,0.857,0.878,0.898,0.918,0.939,0.959,0.98,1.0
            },
            1E-2,
            1E-2
        )
    );
    REQUIRE(
        tensor_close(
            logspace<requested_value_type>(0,1),
            tensor_type{
                1.0,1.048,1.099,1.151,1.207,1.265,1.326,1.389,1.456,1.526,1.6,1.677,1.758,1.842,1.931,2.024,2.121,2.223,2.33,2.442,2.56,2.683,2.812,2.947,3.089,
                3.237,3.393,3.556,3.728,3.907,4.095,4.292,4.498,4.715,4.942,5.179,5.429,5.69,5.964,6.251,6.551,6.866,7.197,7.543,7.906,8.286,8.685,9.103,9.541,10.0
            },
            1E-2,
            1E-2
        )
    );
    REQUIRE(
        tensor_close(
            geomspace<requested_value_type>(1,2),
            tensor_type{
                1.0,1.014,1.029,1.043,1.058,1.073,1.089,1.104,1.12,1.136,1.152,1.168,1.185,1.202,1.219,1.236,1.254,1.272,1.29,1.308,1.327,1.346,1.365,1.385,1.404,
                1.424,1.445,1.465,1.486,1.507,1.529,1.55,1.573,1.595,1.618,1.641,1.664,1.688,1.712,1.736,1.761,1.786,1.811,1.837,1.863,1.89,1.917,1.944,1.972,2.0
            },
            1E-2,
            1E-2
        )
    );
}

TEST_CASE("test_builder_space_exception","[test_builder]")
{
    using value_type = double;
    using gtensor::builder_exception;
    using gtensor::linspace;
    using gtensor::logspace;
    using gtensor::geomspace;

    REQUIRE_THROWS_AS(linspace<value_type>(0,1,-10), builder_exception);
    REQUIRE_THROWS_AS(logspace<value_type>(0,1,-10), builder_exception);
    REQUIRE_THROWS_AS(geomspace<value_type>(1,2,-10), builder_exception);
    REQUIRE_THROWS_AS(geomspace<value_type>(0,1,10), builder_exception);
    REQUIRE_THROWS_AS(geomspace<value_type>(1,0,10), builder_exception);
}

TEST_CASE("test_builder_diag","[test_builder]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::diag;
    using helpers_for_testing::apply_by_element;

    //0tensor,1k,2expected
    auto test_data = std::make_tuple(
        //1d tensor is kth diagonal - result is 2d tensor with kth diagonal
        std::make_tuple(tensor_type{},0,tensor_type{}.reshape(0,0)),
        std::make_tuple(tensor_type{},1,tensor_type{{0}}),
        std::make_tuple(tensor_type{},2,tensor_type{{0,0},{0,0}}),
        std::make_tuple(tensor_type{},3,tensor_type{{0,0,0},{0,0,0},{0,0,0}}),
        std::make_tuple(tensor_type{},-1,tensor_type{{0}}),
        std::make_tuple(tensor_type{},-2,tensor_type{{0,0},{0,0}}),
        std::make_tuple(tensor_type{},-3,tensor_type{{0,0,0},{0,0,0},{0,0,0}}),
        std::make_tuple(tensor_type{1},0,tensor_type{{1}}),
        std::make_tuple(tensor_type{1},1,tensor_type{{0,1},{0,0}}),
        std::make_tuple(tensor_type{1},2,tensor_type{{0,0,1},{0,0,0},{0,0,0}}),
        std::make_tuple(tensor_type{1},3,tensor_type{{0,0,0,1},{0,0,0,0},{0,0,0,0},{0,0,0,0}}),
        std::make_tuple(tensor_type{1},-1,tensor_type{{0,0},{1,0}}),
        std::make_tuple(tensor_type{1},-2,tensor_type{{0,0,0},{0,0,0},{1,0,0}}),
        std::make_tuple(tensor_type{1},-3,tensor_type{{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,0,0,0}}),
        std::make_tuple(tensor_type{1,2,3,4},0,tensor_type{{1,0,0,0},{0,2,0,0},{0,0,3,0},{0,0,0,4}}),
        std::make_tuple(tensor_type{1,2,3,4},1,tensor_type{{0,1,0,0,0},{0,0,2,0,0},{0,0,0,3,0},{0,0,0,0,4},{0,0,0,0,0}}),
        std::make_tuple(tensor_type{1,2,3,4},2,tensor_type{{0,0,1,0,0,0},{0,0,0,2,0,0},{0,0,0,0,3,0},{0,0,0,0,0,4},{0,0,0,0,0,0},{0,0,0,0,0,0}}),
        std::make_tuple(tensor_type{1,2,3,4},-1,tensor_type{{0,0,0,0,0},{1,0,0,0,0},{0,2,0,0,0},{0,0,3,0,0},{0,0,0,4,0}}),
        std::make_tuple(tensor_type{1,2,3,4},-2,tensor_type{{0,0,0,0,0,0},{0,0,0,0,0,0},{1,0,0,0,0,0},{0,2,0,0,0,0},{0,0,3,0,0,0},{0,0,0,4,0,0}}),
        //2d tensor - result is 1d tensor that is kth diagonal
        std::make_tuple(tensor_type{}.reshape(0,0),0,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,0),1,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,0),2,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,0),-1,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,0),-2,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(3,0),0,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,3),0,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(3,0),1,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,3),1,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(3,0),-1,tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,3),-1,tensor_type{}),
        std::make_tuple(tensor_type{{1}},0,tensor_type{1}),
        std::make_tuple(tensor_type{{1}},1,tensor_type{}),
        std::make_tuple(tensor_type{{1}},-1,tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3}},0,tensor_type{1}),
        std::make_tuple(tensor_type{{1,2,3}},1,tensor_type{2}),
        std::make_tuple(tensor_type{{1,2,3}},2,tensor_type{3}),
        std::make_tuple(tensor_type{{1,2,3}},3,tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3}},-1,tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},0,tensor_type{1,6,11,16}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},1,tensor_type{2,7,12}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},2,tensor_type{3,8}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},3,tensor_type{4}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},4,tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},-1,tensor_type{5,10,15}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},-2,tensor_type{9,14}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},-3,tensor_type{13}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}},-4,tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},0,tensor_type{1,7,13}),
        std::make_tuple(tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},1,tensor_type{2,8,14}),
        std::make_tuple(tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},2,tensor_type{3,9,15}),
        std::make_tuple(tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},3,tensor_type{4,10}),
        std::make_tuple(tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},4,tensor_type{5}),
        std::make_tuple(tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},5,tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},-1,tensor_type{6,12}),
        std::make_tuple(tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},-2,tensor_type{11}),
        std::make_tuple(tensor_type{{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}},-3,tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}},0,tensor_type{1,5,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}},1,tensor_type{2,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}},2,tensor_type{3}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}},3,tensor_type{}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}},-1,tensor_type{4,8,12}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}},-2,tensor_type{7,11,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}},-3,tensor_type{10,14}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}},-4,tensor_type{13}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12},{13,14,15}},-5,tensor_type{})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto k = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = diag(ten,k);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_builder_diag_exception","[test_builder]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::builder_exception;
    using gtensor::diag;

    REQUIRE_THROWS_AS(diag(tensor_type(1)), builder_exception);
    REQUIRE_THROWS_AS(diag(tensor_type{{{1}}}), builder_exception);
    REQUIRE_THROWS_AS(diag(tensor_type{}.reshape(0,2,2)), builder_exception);
}