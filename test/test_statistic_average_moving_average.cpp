#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "statistic.hpp"
#include "tensor.hpp"

//average
TEMPLATE_TEST_CASE("test_statistic_average","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using gtensor::average;
    using helpers_for_testing::apply_by_element;

    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<
        typename decltype(average(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<std::vector<value_type>>(),std::declval<bool>()))::value_type,
        result_value_type>
    );
    REQUIRE(std::is_same_v<
        typename decltype(average(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<std::vector<value_type>>(),std::declval<bool>()))::value_type,
        result_value_type>
    );

    //0tensor,1axes,2keep_dims,3weights,4expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{6},dim_type{0},false,std::vector<value_type>{2},result_tensor_type(6.0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6},dim_type{0},false,std::vector<value_type>{6,5,4,3,2,1},result_tensor_type(2.666)),
        std::make_tuple(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},dim_type{0},false,tensor_type{1,2,2,1},result_tensor_type{1.833,2.5,3.333,1.166,1.5,2.166}),
        std::make_tuple(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},dim_type{1},false,tensor_type{1,2,2,2,2,1},result_tensor_type{1.4,1.7,1.8,4.2}),
        std::make_tuple(
            tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},
            std::vector<dim_type>{1,0},
            false,
            tensor_type{1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1},
            result_tensor_type(2.15)
        ),
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::vector<int>{},false,std::vector<value_type>{3},result_tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{},false,std::vector<value_type>{3},result_tensor_type{{1,2,3},{4,5,6}}),
        //keep_dim true
        std::make_tuple(tensor_type{6},dim_type{0},true,std::vector<value_type>{2},result_tensor_type{6.0}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},dim_type{0},true,std::vector<value_type>{6,5,4,3,2,1},result_tensor_type{2.666}),
        std::make_tuple(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},dim_type{0},true,tensor_type{1,2,2,1},result_tensor_type{{1.833,2.5,3.333,1.166,1.5,2.166}}),
        std::make_tuple(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},dim_type{1},true,tensor_type{1,2,2,2,2,1},result_tensor_type{{1.4},{1.7},{1.8},{4.2}}),
        std::make_tuple(
            tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},
            std::vector<dim_type>{1,0},
            true,
            tensor_type{1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1},
            result_tensor_type{{2.15}}
        ),
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::vector<int>{},true,std::vector<value_type>{3},result_tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::vector<int>{},true,std::vector<value_type>{3},result_tensor_type{{1,2,3},{4,5,6}})
    );
    auto test_average = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto weights = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = average(policy...,ten,axes,weights,keep_dims);
            REQUIRE(tensor_close(result,expected,1E-2,1E-2));
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_average_default_policy")
    {
        test_average();
    }
    SECTION("test_average_exec_pol<4>")
    {
        test_average(multithreading::exec_pol<4>{});
    }
    SECTION("test_average_exec_pol<0>")
    {
        test_average(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_statistic_average_overload_default_policy","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using gtensor::average;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<
        typename decltype(average(std::declval<tensor_type>(),std::declval<std::initializer_list<dim_type>>(),std::declval<std::vector<value_type>>(),std::declval<bool>()))::value_type,
        result_value_type>
    );
    REQUIRE(
        tensor_close(
            average(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},{0},tensor_type{1,2,2,1}),
            result_tensor_type{1.833,2.5,3.333,1.166,1.5,2.166},
            1E-2,
            1E-2
        )
    );
    REQUIRE(
        tensor_close(
            average(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},{0,1},tensor_type{1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1}),
            result_tensor_type(2.15),
            1E-2,
            1E-2
        )
    );
    //all axes
    REQUIRE(
        tensor_close(
            average(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},tensor_type{1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1}),
            result_tensor_type(2.15),
            1E-2,
            1E-2
        )
    );
}

TEMPLATE_TEST_CASE("test_statistic_average_overload_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using gtensor::average;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<
        typename decltype(average(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<dim_type>>(),std::declval<std::vector<value_type>>(),std::declval<bool>()))::value_type,
        result_value_type>
    );
    REQUIRE(
        tensor_close(
            average(policy{},tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},{0},tensor_type{1,2,2,1}),
            result_tensor_type{1.833,2.5,3.333,1.166,1.5,2.166},
            1E-2,
            1E-2
        )
    );
    REQUIRE(
        tensor_close(
            average(policy{},tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},{0,1},tensor_type{1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1}),
            result_tensor_type(2.15),
            1E-2,
            1E-2
        )
    );
    //all axes
    REQUIRE(
        tensor_close(
            average(policy{},tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},tensor_type{1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1}),
            result_tensor_type(2.15),
            1E-2,
            1E-2
        )
    );
}

TEMPLATE_TEST_CASE("test_statistic_average_exception","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::average;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axes,2keep_dims,3weights
    auto test_data = std::make_tuple(
        //zero size weights
        std::make_tuple(tensor_type{},0,false,tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,tensor_type{1,1,0,-1,-1}),
        //weights size not match size along axes
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,tensor_type{1,1,2}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},0,false,tensor_type{1,1,2,2}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},0,false,tensor_type{1,2}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},1,false,tensor_type{1,2,2}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},1,false,tensor_type{1,2,2,1,1}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},std::vector<int>{1,0},false,tensor_type{1,2,2,1,1})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto weights = std::get<3>(t);
        REQUIRE_THROWS_AS(average(ten,axes,weights,keep_dims), value_error);
    };
    apply_by_element(test,test_data);
}

//moving average
TEMPLATE_TEST_CASE("test_statistic_moving_average","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using gtensor::tensor_close;
    using gtensor::moving_average;
    using helpers_for_testing::apply_by_element;

    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    REQUIRE(std::is_same_v<
        typename decltype(moving_average(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<std::vector<value_type>>(),std::declval<index_type>()))::value_type,
        result_value_type>
    );

    //0tensor,1axis,2weights,3step,4expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}.reshape(0,4,5),1,tensor_type{1,1},1,result_tensor_type{}.reshape(0,4,5)),
        std::make_tuple(tensor_type{}.reshape(0,4,5),2,tensor_type{1,1,1},1,result_tensor_type{}.reshape(0,4,5)),
        std::make_tuple(tensor_type{5},0,tensor_type{2},1,result_tensor_type{5}),
        std::make_tuple(tensor_type{5},0,tensor_type{2},2,result_tensor_type{5}),
        std::make_tuple(tensor_type{5,6},0,tensor_type{2},1,result_tensor_type{5,6}),
        std::make_tuple(tensor_type{5,6},0,tensor_type{2,3},1,result_tensor_type{5.6}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,tensor_type{1,2,2,1},1,result_tensor_type{2.5,3.5,4.5,5.5,6.5,7.5,8.5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,tensor_type{1,1,1,2},1,result_tensor_type{2.8,3.8,4.8,5.8,6.8,7.8,8.8}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,tensor_type{1,2,2,1},2,result_tensor_type{2.5,4.5,6.5,8.5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,tensor_type{2,3,3,2},5,result_tensor_type{2.5,7.5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,tensor_type{1,2,3},3,result_tensor_type{2.333,5.333,8.333}),
        std::make_tuple(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},0,tensor_type{1,2,3},1,result_tensor_type{{0.833,1.333,0.666,3.0,5.166},{2.666,1.666,0.166,4.0,3.333}}),
        std::make_tuple(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},1,tensor_type{1,2,3},1,result_tensor_type{{0.833,-0.333,1.666},{3.333,3.0,3.0},{-0.666,2.0,4.833},{1.5,2.333,1.833}})
    );
    auto test_moving_average = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto weights = std::get<2>(t);
            auto step = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = moving_average(policy...,ten,axis,weights,step);
            REQUIRE(tensor_close(result,expected,1E-2,1E-2));
        };
        apply_by_element(test,test_data);
    };
    SECTION("test_moving_average_default_policy")
    {
        test_moving_average();
    }
    SECTION("test_moving_average_exec_pol<4>")
    {
        test_moving_average(multithreading::exec_pol<4>{});
    }
    SECTION("test_moving_average_exec_pol<0>")
    {
        test_moving_average(multithreading::exec_pol<0>{});
    }
}

TEMPLATE_TEST_CASE("test_statistic_moving_average_exception","test_statistic",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::moving_average;
    using gtensor::value_error;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2weights,3step
    auto test_data = std::make_tuple(
        //zero size weights
        std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{},1),
        //weights size greater than axis size
        std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,1,2,2,3,3},1),
        //zero step
        std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,1,2},0)
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto weights = std::get<2>(t);
        auto step = std::get<3>(t);
        REQUIRE_THROWS_AS(moving_average(ten,axis,weights,step), value_error);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_statistic_moving_average_overload_default_policy","test_statistic")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    using gtensor::moving_average;
    using helpers_for_testing::apply_by_element;

    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    //like over flatten, step=1
    REQUIRE(tensor_close(
        moving_average(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},tensor_type{1,2,3},1),
        result_tensor_type{0.833,-0.333,1.667,1.667,2.0,3.333,3.0,3.0,1.333,1.0,-0.667,2.0,4.833,5.667,3.833,1.5,2.333,1.833},1E-2,1E-2)
    );
    //like over flatten, step=3
    REQUIRE(tensor_close(
        moving_average(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},tensor_type{1,2,3},3),
        result_tensor_type{0.833,1.667,3.0,1.0,4.833,1.5},1E-2,1E-2)
    );
}

TEMPLATE_TEST_CASE("test_statistic_moving_average_overload_policy","test_statistic",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    using gtensor::moving_average;
    using helpers_for_testing::apply_by_element;

    using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
    using result_tensor_type = gtensor::tensor<result_value_type>;

    //like over flatten, step=1
    REQUIRE(tensor_close(
        moving_average(policy{},tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},tensor_type{1,2,3},1),
        result_tensor_type{0.833,-0.333,1.667,1.667,2.0,3.333,3.0,3.0,1.333,1.0,-0.667,2.0,4.833,5.667,3.833,1.5,2.333,1.833},1E-2,1E-2)
    );
    //like over flatten, step=3
    REQUIRE(tensor_close(
        moving_average(policy{},tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},tensor_type{1,2,3},3),
        result_tensor_type{0.833,1.667,3.0,1.0,4.833,1.5},1E-2,1E-2)
    );
}

