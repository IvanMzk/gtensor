#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"
#include "sort_search.hpp"

//sort
TEST_CASE("test_sort_search_sort","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::detail::no_value;
    using gtensor::sort;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2comparator,3expected
    auto test_data = std::make_tuple(
        //no comparator
        std::make_tuple(tensor_type{},0,no_value{},tensor_type{}),
        std::make_tuple(tensor_type{1},0,no_value{},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,no_value{},tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,no_value{},tensor_type{0,1,1,2,2,3,5,6}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,no_value{},tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,4,4},{8,7,6,6,5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,no_value{},tensor_type{{-1,1,2,3,6},{0,1,2,5,8},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}}),
        //comparator
        std::make_tuple(tensor_type{},0,std::less<void>{},tensor_type{}),
        std::make_tuple(tensor_type{1},0,std::less<void>{},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,std::less<void>{},tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,std::greater<void>{},tensor_type{6,5,3,2,2,1,1,0}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,std::less<void>{},tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,4,4},{8,7,6,6,5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,std::greater<void>{},tensor_type{{6,3,2,1,-1},{8,5,2,1,0},{7,4,2,0,-1},{4,4,4,2,1},{6,4,3,3,1}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto comparator = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = sort(ten,axis,comparator);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//argsort
TEST_CASE("test_sort_search_argsort","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::detail::no_value;
    using gtensor::argsort;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(argsort(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<no_value>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argsort(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<std::less<void>>())),result_tensor_type>);

    //0tensor,1axis,2comparator,3expected
    auto test_data = std::make_tuple(
        //no comparator
        std::make_tuple(tensor_type{},0,no_value{},result_tensor_type{}),
        std::make_tuple(tensor_type{1},0,no_value{},result_tensor_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,no_value{},result_tensor_type{0,1,2,3,4,5}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,no_value{},result_tensor_type{6,1,5,0,4,3,7,2}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,no_value{},result_tensor_type{{2,0,0,1,2},{0,4,2,3,0},{4,1,1,2,4},{3,3,3,4,3},{1,2,4,0,1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,no_value{},result_tensor_type{{2,1,0,4,3},{3,2,1,4,0},{0,2,4,3,1},{3,2,0,1,4},{1,0,4,3,2}}),
        // //comparator
        std::make_tuple(tensor_type{},0,std::less<void>{},result_tensor_type{}),
        std::make_tuple(tensor_type{1},0,std::less<void>{},result_tensor_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,std::less<void>{},result_tensor_type{0,1,2,3,4,5}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,std::greater<void>{},result_tensor_type{2,7,3,0,4,1,5,6}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,std::less<void>{},result_tensor_type{{2,0,0,1,2},{0,4,2,3,0},{4,1,1,2,4},{3,3,3,4,3},{1,2,4,0,1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,std::greater<void>{},result_tensor_type{{3,4,0,1,2},{0,4,1,2,3},{1,3,4,2,0},{0,1,4,2,3},{2,3,0,4,1}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto comparator = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = argsort(ten,axis,comparator);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//argmin,nanargmin
TEST_CASE("test_sort_search_argmin_nanargmin","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::detail::no_value;
    using gtensor::argmin;
    using gtensor::nanargmin;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(argmin(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argmin(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(nanargmin(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(nanargmin(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<bool>())),result_tensor_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,false,result_tensor_type(6)),
        std::make_tuple(tensor_type{2,1,6,3,2,1,3,5},0,false,result_tensor_type(1)),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,false,result_tensor_type{2,0,0,1,2}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,false,result_tensor_type{2,3,0,3,1}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,true,result_tensor_type{{2,0,0,1,2}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,true,result_tensor_type{{2},{3},{0},{3},{1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{0,1},false,result_tensor_type(2))
    );
    SECTION("test_argmin")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);

            auto result = argmin(ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_nanargmin")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);

            auto result = nanargmin(ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("test_math_argmin_nanargmin_nan_values","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmin;
    using gtensor::nanargmin;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //argmin
        std::make_tuple(argmin(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(1)),
        std::make_tuple(argmin(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(argmin(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(1)),
        std::make_tuple(argmin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), result_tensor_type(0)),
        std::make_tuple(argmin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), result_tensor_type{0,0,0,1,0,1}),
        std::make_tuple(argmin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), result_tensor_type{0,2}),
        std::make_tuple(argmin(tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), result_tensor_type{1,0,0,0}),
        std::make_tuple(argmin(tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), result_tensor_type{3,1}),
        std::make_tuple(argmin(tensor_type{{nan,nan,nan},{nan,nan,nan}}), result_tensor_type(0)),
        //nanargmin
        std::make_tuple(nanargmin(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(1)),
        std::make_tuple(nanargmin(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(nanargmin(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(3)),
        std::make_tuple(nanargmin(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), result_tensor_type(6)),
        std::make_tuple(nanargmin(tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},0), result_tensor_type{2,1,0}),
        std::make_tuple(nanargmin(tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},1), result_tensor_type{2,1,0})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_math_argmin_nanargmin_exception","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::argmin;
    using gtensor::nanargmin;
    using gtensor::reduce_exception;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    //empty
    REQUIRE_THROWS_AS(argmin(tensor_type{}), reduce_exception);
    REQUIRE_THROWS_AS(argmin(tensor_type{}.reshape(0,2,3),0), reduce_exception);
    REQUIRE_THROWS_AS(nanargmin(tensor_type{}), reduce_exception);
    REQUIRE_THROWS_AS(nanargmin(tensor_type{}.reshape(0,2,3),0), reduce_exception);
    //all nan
    REQUIRE_THROWS_AS(nanargmin(tensor_type{{nan,nan,nan},{nan,nan,nan}},0), reduce_exception);
    REQUIRE_THROWS_AS(nanargmin(tensor_type{{nan,nan,nan},{nan,nan,nan}},1), reduce_exception);
}

//argmax,nanargmax
TEST_CASE("test_sort_search_argmax_nanargmax","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::detail::no_value;
    using gtensor::argmax;
    using gtensor::nanargmax;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(argmax(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argmax(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(nanargmax(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<bool>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(nanargmax(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<bool>())),result_tensor_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,false,result_tensor_type(5)),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,false,result_tensor_type(2)),
        std::make_tuple(tensor_type{6,1,6,3,2,1,6,5},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,false,result_tensor_type{1,2,4,0,1}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,false,result_tensor_type{3,0,1,0,2}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,true,result_tensor_type{{1,2,4,0,1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,true,result_tensor_type{{3},{0},{1},{0},{2}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{0,1},false,result_tensor_type(5))

    );
    SECTION("test_argmax")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);

            auto result = argmax(ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_nanargmax")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto expected = std::get<3>(t);

            auto result = nanargmax(ten,axes,keep_dims);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("test_math_argmax_nanargmax_nan_values","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmax;
    using gtensor::nanargmax;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //argmax
        std::make_tuple(argmax(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(argmax(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(4)),
        std::make_tuple(argmax(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(1)),
        std::make_tuple(argmax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), result_tensor_type(0)),
        std::make_tuple(argmax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), result_tensor_type{0,0,0,0,1,0}),
        std::make_tuple(argmax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), result_tensor_type{0,2}),
        std::make_tuple(argmax(tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), result_tensor_type{1,1,1,0}),
        std::make_tuple(argmax(tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), result_tensor_type{3,2}),
        std::make_tuple(argmax(tensor_type{{nan,nan,nan},{nan,nan,nan}}), result_tensor_type(0)),
        //nanargmax
        std::make_tuple(nanargmax(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), result_tensor_type(3)),
        std::make_tuple(nanargmax(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), result_tensor_type(4)),
        std::make_tuple(nanargmax(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), result_tensor_type(5)),
        std::make_tuple(nanargmax(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), result_tensor_type(7)),
        std::make_tuple(nanargmax(tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},0), result_tensor_type{2,2,0}),
        std::make_tuple(nanargmax(tensor_type{{nan,nan,1.0},{nan,1.1,nan},{0.1,2.0,nan}},1), result_tensor_type{2,1,1})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_math_argmax_nanargmax_exception","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::argmax;
    using gtensor::nanargmax;
    using gtensor::reduce_exception;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    //empty
    REQUIRE_THROWS_AS(argmax(tensor_type{}), reduce_exception);
    REQUIRE_THROWS_AS(argmax(tensor_type{}.reshape(0,2,3),0), reduce_exception);
    REQUIRE_THROWS_AS(nanargmax(tensor_type{}), reduce_exception);
    REQUIRE_THROWS_AS(nanargmax(tensor_type{}.reshape(0,2,3),0), reduce_exception);
    //all nan
    REQUIRE_THROWS_AS(nanargmax(tensor_type{{nan,nan,nan},{nan,nan,nan}},0), reduce_exception);
    REQUIRE_THROWS_AS(nanargmax(tensor_type{{nan,nan,nan},{nan,nan,nan}},1), reduce_exception);
}


