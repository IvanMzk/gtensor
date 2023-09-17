#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "tensor_math.hpp"
#include "tensor.hpp"

//all
TEST_CASE("test_math_all","test_math")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::all;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(all(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,bool>);
    REQUIRE(std::is_same_v<typename decltype(all(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,bool>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,bool_tensor_type{true,true}),
        std::make_tuple(tensor_type{5},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{0},0,false,bool_tensor_type(false)),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{5,0,1,-1,4,4},0,false,bool_tensor_type(false)),
        std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},0,false,bool_tensor_type{{true,false,false},{true,false,true}}),
        std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},1,false,bool_tensor_type{{true,false,false},{true,false,true}}),
        std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},2,false,bool_tensor_type{{false,false},{false,true}}),
        std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{0,1},false,bool_tensor_type{true,false,false}),
        std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{2,1},false,bool_tensor_type{false,false}),
        std::make_tuple(tensor_type{},std::vector<int>{},false,bool_tensor_type{}),
        std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{},false,bool_tensor_type{{{true,true,false},{true,false,true}},{{true,false,true},{true,true,true}}}),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,bool_tensor_type{true}),
        std::make_tuple(tensor_type{5,0,1,-1,4,4},0,true,bool_tensor_type{false}),
        std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},1,true,bool_tensor_type{{{true,false,false}},{{true,false,true}}}),
        std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{2,1},true,bool_tensor_type{{{false}},{{false}}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,bool_tensor_type{}),
        std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{},true,bool_tensor_type{{{true,true,false},{true,false,true}},{{true,false,true},{true,true,true}}})
    );

    auto test_all = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = all(policy...,ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_all();
    }
    SECTION("exec_pol<4>")
    {
        test_all(multithreading::exec_pol<4>{});
    }
    SECTION("exec_pol<0>")
    {
        test_all(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_math_all_overloads","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::all;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(all(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,bool>);

    REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},{1},false) == bool_tensor_type{{true,false,false},{true,false,true}});
    REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},{2,1},false) == bool_tensor_type{false,false});
    REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},{0,1},true) == bool_tensor_type{{{true,false,false}}});
    //all axes
    REQUIRE(all(tensor_type{{{1,5,6},{2,6,-1}},{{7,2,9},{1,11,3}}}) == bool_tensor_type(true));
    REQUIRE(all(tensor_type{{{1,5,6},{2,6,-1}},{{7,2,9},{1,11,3}}},false) == bool_tensor_type(true));
    REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}}) == bool_tensor_type(false));
    REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},false) == bool_tensor_type(false));
    REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},true) == bool_tensor_type{{{false}}});
    REQUIRE(all(tensor_type{{{1,5,6},{2,6,-1}},{{7,2,9},{1,11,3}}},true) == bool_tensor_type{{{true}}});
}

TEMPLATE_TEST_CASE("test_math_all_overloads_policy","test_math",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::all;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(all(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,bool>);

    REQUIRE(all(policy{},tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},{1},false) == bool_tensor_type{{true,false,false},{true,false,true}});
    REQUIRE(all(policy{},tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},{2,1},false) == bool_tensor_type{false,false});
    REQUIRE(all(policy{},tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},{0,1},true) == bool_tensor_type{{{true,false,false}}});
    //all axes
    REQUIRE(all(policy{},tensor_type{{{1,5,6},{2,6,-1}},{{7,2,9},{1,11,3}}}) == bool_tensor_type(true));
    REQUIRE(all(policy{},tensor_type{{{1,5,6},{2,6,-1}},{{7,2,9},{1,11,3}}},false) == bool_tensor_type(true));
    REQUIRE(all(policy{},tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}}) == bool_tensor_type(false));
    REQUIRE(all(policy{},tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},false) == bool_tensor_type(false));
    REQUIRE(all(policy{},tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},true) == bool_tensor_type{{{false}}});
    REQUIRE(all(policy{},tensor_type{{{1,5,6},{2,6,-1}},{{7,2,9},{1,11,3}}},true) == bool_tensor_type{{{true}}});
}

TEST_CASE("test_math_all_nan_values","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::all;
    using helpers_for_testing::apply_by_element;

    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{nan,nan,nan,nan,nan,nan,nan,nan,nan},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{nan,0.0,nan,nan,nan,0.0,nan,nan,nan},0,false,bool_tensor_type(false)),
        std::make_tuple(tensor_type{nan,pos_inf,nan,neg_inf},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{nan,pos_inf,0.0,nan,neg_inf},std::vector<int>{},false,bool_tensor_type{true,true,false,true,true})
    );
    auto test_all = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = all(policy...,ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_all();
    }
    SECTION("exec_pol<4>")
    {
        test_all(multithreading::exec_pol<4>{});
    }
    SECTION("exec_pol<0>")
    {
        test_all(multithreading::exec_pol<0>{});
    }
}

//any
TEST_CASE("test_math_any","test_math")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::any;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(any(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,bool>);
    REQUIRE(std::is_same_v<typename decltype(any(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,bool>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,bool_tensor_type(false)),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,bool_tensor_type(false)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,bool_tensor_type{false,false}),
        std::make_tuple(tensor_type{5},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{0},0,false,bool_tensor_type(false)),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{5,0,1,-1,0,4},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{0,0,0},0,false,bool_tensor_type(false)),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},0,false,bool_tensor_type{{true,false,true},{true,false,true}}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},1,false,bool_tensor_type{{true,false,true},{false,false,true}}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},2,false,bool_tensor_type{{true,true},{true,true}}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{0,1},false,bool_tensor_type{true,false,true}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{2,1},false,bool_tensor_type{true,true}),
        std::make_tuple(tensor_type{},std::vector<int>{},false,bool_tensor_type{}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{},false,bool_tensor_type{{{true,false,false},{true,false,true}},{{false,false,true},{false,false,true}}}),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,bool_tensor_type{false}),
        std::make_tuple(tensor_type{},std::vector<int>{0},true,bool_tensor_type{false}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,bool_tensor_type{{{false},{false}}}),
        std::make_tuple(tensor_type{5},0,true,bool_tensor_type{true}),
        std::make_tuple(tensor_type{0},0,true,bool_tensor_type{false}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,true,bool_tensor_type{true}),
        std::make_tuple(tensor_type{5,0,1,-1,0,4},0,true,bool_tensor_type{true}),
        std::make_tuple(tensor_type{0,0,0},0,true,bool_tensor_type{false}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},0,true,bool_tensor_type{{{true,false,true},{true,false,true}}}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},1,true,bool_tensor_type{{{true,false,true}},{{false,false,true}}}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},2,true,bool_tensor_type{{{true},{true}},{{true},{true}}}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{0,1},true,bool_tensor_type{{{true,false,true}}}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{2,1},true,bool_tensor_type{{{true}},{{true}}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,bool_tensor_type{}),
        std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{},true,bool_tensor_type{{{true,false,false},{true,false,true}},{{false,false,true},{false,false,true}}})
    );
    auto test_any = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = any(policy...,ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_any();
    }
    SECTION("exec_pol<4>")
    {
        test_any(multithreading::exec_pol<4>{});
    }
    SECTION("exec_pol<0>")
    {
        test_any(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_math_any_overloads","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::any;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(any(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,bool>);

    REQUIRE(any(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},{1},true) == bool_tensor_type{{{true,false,true}},{{false,false,true}}});
    REQUIRE(any(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},{1}) == bool_tensor_type{{true,false,true},{false,false,true}});
    REQUIRE(any(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},{2,1}) == bool_tensor_type{true,true});
    REQUIRE(any(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}}) == bool_tensor_type(true));
}

TEMPLATE_TEST_CASE("test_math_any_overloads_policy","test_math",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::any;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(any(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,bool>);

    REQUIRE(any(policy{},tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},{1},true) == bool_tensor_type{{{true,false,true}},{{false,false,true}}});
    REQUIRE(any(policy{},tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},{1}) == bool_tensor_type{{true,false,true},{false,false,true}});
    REQUIRE(any(policy{},tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},{2,1}) == bool_tensor_type{true,true});
    REQUIRE(any(policy{},tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}}) == bool_tensor_type(true));
}

TEST_CASE("test_math_any_nan_values","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::any;
    using helpers_for_testing::apply_by_element;

    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{nan,nan,nan,nan,nan,nan,nan,nan,nan},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{0.0,0.0,nan,0.0,0.0,0.0,0.0,nan,0.0,0.0},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{0.0,0.0,pos_inf},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{0.0,0.0,neg_inf},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{nan,pos_inf,nan,neg_inf},0,false,bool_tensor_type(true)),
        std::make_tuple(tensor_type{nan,pos_inf,0.0,nan,neg_inf},std::vector<int>{},false,bool_tensor_type{true,true,false,true,true})
    );
    auto test_any = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);
            auto result = any(policy...,ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    SECTION("default_policy")
    {
        test_any();
    }
    SECTION("exec_pol<4>")
    {
        test_any(multithreading::exec_pol<4>{});
    }
    SECTION("exec_pol<0>")
    {
        test_any(multithreading::exec_pol<0>{});
    }
}

