#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"
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

TEST_CASE("test_sort_search_sort_overload","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::sort;

    //default comparator = std::less<void>
    REQUIRE(sort(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0) == tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,4,4},{8,7,6,6,5}});
    //default comparator, default axis = -1
    REQUIRE(sort(tensor_type{2,1,6,3,2,1,0,5}) == tensor_type{0,1,1,2,2,3,5,6});
    REQUIRE(sort(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == tensor_type{{-1,1,2,3,6},{0,1,2,5,8},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}});
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

TEST_CASE("test_sort_search_argsort_overload","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argsort;

    //default comparator
    REQUIRE(argsort(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0) == result_tensor_type{{2,0,0,1,2},{0,4,2,3,0},{4,1,1,2,4},{3,3,3,4,3},{1,2,4,0,1}});
    //default comparator, default axis
    REQUIRE(argsort(tensor_type{2,1,6,3,2,1,0,5}) == result_tensor_type{6,1,5,0,4,3,7,2});
    REQUIRE(argsort(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type{{2,1,0,4,3},{3,2,1,4,0},{0,2,4,3,1},{3,2,0,1,4},{1,0,4,3,2}});
}

//partition
TEST_CASE("test_sort_search_partition_nth_scalar","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = tensor_type::index_type;
    using slice_type = tensor_type::slice_type;
    using slices_container_type = std::vector<slice_type>;
    using slices_container_difference_type = typename slices_container_type::difference_type;
    using gtensor::detail::no_value;
    using gtensor::partition;
    using helpers_for_testing::apply_by_element;

    //0tensor,1nth,2axis,3comparator,4expected
    auto test_data = std::make_tuple(
        //no comparator
        std::make_tuple(tensor_type{},0,0,no_value{},tensor_type{}),
        std::make_tuple(tensor_type{1},0,0,no_value{},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,0,no_value{},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},4,0,no_value{},tensor_type{5}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,0,no_value{},tensor_type{0}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},2,0,no_value{},tensor_type{1}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},4,0,no_value{},tensor_type{2}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},7,0,no_value{},tensor_type{6}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,0,no_value{},tensor_type{{-1,1,-1,0,2}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},2,0,no_value{},tensor_type{{3,2,1,4,3}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},3,0,no_value{},tensor_type{{4,4,2,4,4}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},4,0,no_value{},tensor_type{{8,7,6,6,5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,1,no_value{},tensor_type{{-1},{0},{-1},{1},{1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},2,1,no_value{},tensor_type{{2},{2},{2},{4},{3}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},3,1,no_value{},tensor_type{{3},{5},{4},{4},{4}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},4,1,no_value{},tensor_type{{6},{8},{7},{4},{6}}),
        //comparator
        std::make_tuple(tensor_type{},0,0,std::greater<void>{},tensor_type{}),
        std::make_tuple(tensor_type{1},0,0,std::greater<void>{},tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,0,std::greater<void>{},tensor_type{6}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},4,0,std::greater<void>{},tensor_type{2}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,0,std::greater<void>{},tensor_type{6}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},2,0,std::greater<void>{},tensor_type{3}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},4,0,std::greater<void>{},tensor_type{2}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},7,0,std::greater<void>{},tensor_type{0})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto nth = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto comparator = std::get<3>(t);
        auto expected = std::get<4>(t);

        auto result = partition(ten,nth,axis,comparator);
        slices_container_type slices(static_cast<slices_container_difference_type>(ten.dim()));
        const auto nth_ = static_cast<index_type>(nth);
        slices[static_cast<slices_container_difference_type>(axis)] = slice_type{nth_,nth_+1};
        auto result_nth_elements = result(slices);
        REQUIRE(result_nth_elements == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_sort_search_partition_nth_container","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = tensor_type::index_type;
    using index_tensor_type = gtensor::tensor<index_type>;
    using gtensor::detail::no_value;
    using gtensor::partition;

    //no comparator
    REQUIRE(partition(tensor_type{},std::vector<int>{0,2,4},0,no_value{}) == tensor_type{});
    REQUIRE(partition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{4},0,no_value{})(index_tensor_type{4}) == tensor_type{2});
    REQUIRE(partition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{8,2,5},0,no_value{})(index_tensor_type{8,2,5}) == tensor_type{5,1,2});
    REQUIRE(partition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{0,7,13},0,no_value{})(index_tensor_type{0,7,13}) == tensor_type{0,3,7});
    REQUIRE(
        partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},0,no_value{})(index_tensor_type{1,3}) ==
        tensor_type{{2,1,0,1,3},{4,4,2,4,4}}
    );
    REQUIRE(
        partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},1,no_value{})(index_tensor_type{{0},{1},{2},{3},{4}},index_tensor_type{1,3}) ==
        tensor_type{{1,3},{1,5},{0,4},{2,4},{3,4}}
    );
    REQUIRE(
        partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{4,0,2},0,no_value{})(index_tensor_type{4,0,2}) ==
        tensor_type{{8,7,6,6,5},{-1,1,-1,0,2},{3,2,1,4,3}}
    );
    REQUIRE(
        partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{4,0,2},1,no_value{})(index_tensor_type{{0},{1},{2},{3},{4}},index_tensor_type{4,0,2}) ==
        tensor_type{{6,-1,2},{8,0,2},{7,-1,2},{4,1,4},{6,1,3}}
    );
    //comparator
    REQUIRE(partition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{8,2,5},0,std::greater<void>{})(index_tensor_type{8,2,5}) == tensor_type{2,7,5});
    REQUIRE(
        partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},1,std::greater<void>{})(index_tensor_type{{0},{1},{2},{3},{4}},index_tensor_type{1,3}) ==
        tensor_type{{3,1},{5,1},{4,0},{4,2},{4,3}}
    );
}

TEST_CASE("test_sort_search_partition_overload","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = tensor_type::index_type;
    using index_tensor_type = gtensor::tensor<index_type>;
    using slice_type = tensor_type::slice_type;
    using gtensor::partition;

    //default comparator
    REQUIRE(partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},3,0)(3) == tensor_type{4,4,2,4,4});
    REQUIRE(
        partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},0)(index_tensor_type{1,3}) ==
        tensor_type{{2,1,0,1,3},{4,4,2,4,4}}
    );
    //default comparator, axis
    REQUIRE(partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},3)(slice_type{},3) == tensor_type{3,5,4,4,4});
    REQUIRE(
        partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3})(index_tensor_type{{0},{1},{2},{3},{4}},index_tensor_type{1,3}) ==
        tensor_type{{1,3},{1,5},{0,4},{2,4},{3,4}}
    );
}

TEST_CASE("test_sort_search_partition_exception","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::detail::no_value;
    using gtensor::reduce_exception;
    using gtensor::partition;

    //nth out of bound
    REQUIRE_THROWS_AS(partition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},14,0,no_value{}), reduce_exception);
    REQUIRE_THROWS_AS(partition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{2,3,14},0,no_value{}), reduce_exception);
    REQUIRE_THROWS_AS(partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,5},0,std::greater<void>{}), reduce_exception);
    REQUIRE_THROWS_AS(partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{6,2},1,std::greater<void>{}), reduce_exception);
    //nth empty container
    REQUIRE_THROWS_AS(partition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{},0,no_value{}), reduce_exception);
}

//argpartition
TEST_CASE("test_sort_search_argpartition_nth_scalar","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = tensor_type::dim_type;
    using index_type = tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using slice_type = tensor_type::slice_type;
    using slices_container_type = std::vector<slice_type>;
    using slices_container_difference_type = typename slices_container_type::difference_type;
    using gtensor::detail::no_value;
    using gtensor::argpartition;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(argpartition(std::declval<tensor_type>(),std::declval<index_type>(),std::declval<dim_type>(),std::declval<no_value>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argpartition(std::declval<tensor_type>(),std::declval<index_type>(),std::declval<dim_type>(),std::declval<std::less<void>>())),result_tensor_type>);

    //0tensor,1nth,2axis,3comparator,4expected
    auto test_data = std::make_tuple(
        //no comparator
        std::make_tuple(tensor_type{},0,0,no_value{},result_tensor_type{}),
        std::make_tuple(tensor_type{1},0,0,no_value{},result_tensor_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,0,no_value{},result_tensor_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},4,0,no_value{},result_tensor_type{4}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,0,no_value{},result_tensor_type{6}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},2,0,no_value{},result_tensor_type{5}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},4,0,no_value{},result_tensor_type{0}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},7,0,no_value{},result_tensor_type{2}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,0,no_value{},result_tensor_type{{2,0,0,1,2}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},2,0,no_value{},result_tensor_type{{4,1,1,2,4}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},3,0,no_value{},result_tensor_type{{3,3,3,4,3}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},4,0,no_value{},result_tensor_type{{1,2,4,0,1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,1,no_value{},result_tensor_type{{2},{3},{0},{3},{1}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},2,1,no_value{},result_tensor_type{{0},{1},{4},{1},{4}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},3,1,no_value{},result_tensor_type{{4},{4},{3},{0},{3}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},4,1,no_value{},result_tensor_type{{3},{0},{1},{4},{2}}),
        //comparator
        std::make_tuple(tensor_type{},0,0,std::greater<void>{},result_tensor_type{}),
        std::make_tuple(tensor_type{1},0,0,std::greater<void>{},result_tensor_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,0,std::greater<void>{},result_tensor_type{5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},4,0,std::greater<void>{},result_tensor_type{1}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,0,std::greater<void>{},result_tensor_type{2}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},2,0,std::greater<void>{},result_tensor_type{3}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},4,0,std::greater<void>{},result_tensor_type{4}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},7,0,std::greater<void>{},result_tensor_type{6})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto nth = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto comparator = std::get<3>(t);
        auto expected = std::get<4>(t);

        auto result = argpartition(ten,nth,axis,comparator);
        slices_container_type slices(static_cast<slices_container_difference_type>(ten.dim()));
        const auto nth_ = static_cast<index_type>(nth);
        slices[static_cast<slices_container_difference_type>(axis)] = slice_type{nth_,nth_+1};
        auto result_nth_elements = result(slices);
        REQUIRE(result_nth_elements == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_sort_search_argpartition_nth_container","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = tensor_type::dim_type;
    using index_type = tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using index_tensor_type = gtensor::tensor<index_type>;
    using gtensor::detail::no_value;
    using gtensor::argpartition;

    REQUIRE(std::is_same_v<decltype(argpartition(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<dim_type>(),std::declval<no_value>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argpartition(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<dim_type>(),std::declval<std::less<void>>())),result_tensor_type>);

    //no comparator
    REQUIRE(argpartition(tensor_type{},std::vector<int>{0,2,4},0,no_value{}) == result_tensor_type{});
    REQUIRE(argpartition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{4},0,no_value{})(index_tensor_type{4}) == result_tensor_type{9});
    REQUIRE(argpartition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{8,2,5},0,no_value{})(index_tensor_type{8,2,5}) == result_tensor_type{5,4,9});
    REQUIRE(argpartition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{0,7,13},0,no_value{})(index_tensor_type{0,7,13}) == result_tensor_type{13,8,0});
    REQUIRE(
        argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},0,no_value{})(index_tensor_type{1,3}) ==
        result_tensor_type{{0,4,2,3,0},{3,3,3,4,3}}
    );
    REQUIRE(
        argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},1,no_value{})(index_tensor_type{{0},{1},{2},{3},{4}},index_tensor_type{1,3}) ==
        result_tensor_type{{1,4},{2,4},{2,3},{2,0},{0,3}}
    );
    REQUIRE(
        argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{4,0,2},0,no_value{})(index_tensor_type{4,0,2}) ==
        result_tensor_type{{1,2,4,0,1},{2,0,0,1,2},{4,1,1,2,4}}
    );
    REQUIRE(
        argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{4,0,2},1,no_value{})(index_tensor_type{{0},{1},{2},{3},{4}},index_tensor_type{4,0,2}) ==
        result_tensor_type{{3,2,0},{0,3,1},{1,0,4},{4,3,1},{2,1,4}}
    );
    //comparator
    REQUIRE(argpartition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{8,2,5},0,std::greater<void>{})(index_tensor_type{8,2,5}) == result_tensor_type{7,2,5});
    REQUIRE(
        argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},1,std::greater<void>{})(index_tensor_type{{0},{1},{2},{3},{4}},index_tensor_type{1,3}) ==
        result_tensor_type{{4,1},{4,2},{3,2},{1,2},{3,4}}
    );
}

TEST_CASE("test_sort_search_argpartition_overload","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using index_tensor_type = gtensor::tensor<index_type>;
    using slice_type = tensor_type::slice_type;
    using gtensor::argpartition;

    //default comparator
    REQUIRE(argpartition(tensor_type{2,1,6,3,2,1,0,5},2,0)(2) == result_tensor_type(5));
    REQUIRE(argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},3,1)(slice_type{},3) == result_tensor_type{4,4,3,0,3});
    REQUIRE(
        argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},0)(index_tensor_type{1,3}) ==
        result_tensor_type{{0,4,2,3,0},{3,3,3,4,3}}
    );
    //default comparator,axis
    REQUIRE(argpartition(tensor_type{2,1,6,3,2,1,0,5},4)(4) == result_tensor_type(0));
    REQUIRE(argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},3)(slice_type{},3) == result_tensor_type{4,4,3,0,3});
    REQUIRE(
        argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3})(index_tensor_type{{0},{1},{2},{3},{4}},index_tensor_type{1,3}) ==
        result_tensor_type{{1,4},{2,4},{2,3},{2,0},{0,3}}
    );
}

TEST_CASE("test_sort_search_argpartition_exception","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::detail::no_value;
    using gtensor::reduce_exception;
    using gtensor::argpartition;

    //nth out of bound
    REQUIRE_THROWS_AS(argpartition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},14,0,no_value{}), reduce_exception);
    REQUIRE_THROWS_AS(argpartition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{2,3,14},0,no_value{}), reduce_exception);
    REQUIRE_THROWS_AS(argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,5},0,std::greater<void>{}), reduce_exception);
    REQUIRE_THROWS_AS(argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{6,2},1,std::greater<void>{}), reduce_exception);
    //nth empty container
    REQUIRE_THROWS_AS(argpartition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{},0,no_value{}), reduce_exception);
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

TEST_CASE("test_sort_search_argmin_nanargmin_overload","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmin;
    using gtensor::nanargmin;

    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();

    //default axes and keep_dims
    REQUIRE(argmin(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-4,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type(10));
    REQUIRE(nanargmin(tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{nan,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}}) == result_tensor_type(2));
    //default axes
    REQUIRE(argmin(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-4,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},true) == result_tensor_type{{10}});
    REQUIRE(nanargmin(tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{nan,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}},true) == result_tensor_type{{2}});
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

TEST_CASE("test_sort_search_argmax_nanargmax_overload","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::argmax;
    using gtensor::nanargmax;

    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();

    //default axes and keep_dims
    REQUIRE(argmax(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type(5));
    REQUIRE(nanargmax(tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{-1.0,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}}) == result_tensor_type(11));
    //default axes
    REQUIRE(argmax(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},true) == result_tensor_type{{5}});
    REQUIRE(nanargmax(tensor_type{{2.0,1.0,-1.0,6.0,3.0},{nan,2.0,1.0,0.0,5.0},{-1.0,7.0,0.0,4.0,2.0},{4.0,4.0,2.0,1.0,4.0},{3.0,1.0,6.0,4.0,3.0}},true) == result_tensor_type{{11}});
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

//count_nonzero
TEST_CASE("test_sort_search_count_nonzero","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::count_nonzero;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(count_nonzero(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<bool>()))::value_type,index_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{1},0,false,result_tensor_type(1)),
        std::make_tuple(tensor_type{0},0,false,result_tensor_type(0)),
        std::make_tuple(tensor_type{1,3,0,0,1,4,6,-2,1,0},0,false,result_tensor_type(7)),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,false,result_tensor_type{5,5,4,4,5}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,false,result_tensor_type{5,4,4,5,5}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{0,1},false,result_tensor_type(23)),
        std::make_tuple(tensor_type{1,3,0,0,1,4,6,-2,1,0},0,true,result_tensor_type{7}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,true,result_tensor_type{{5,5,4,4,5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},1,true,result_tensor_type{{5},{4},{4},{5},{5}}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{0,1},true,result_tensor_type{{23}})

    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = count_nonzero(ten,axes,keep_dims);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_sort_search_count_nonzero_overload","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::count_nonzero;

    REQUIRE(count_nonzero(tensor_type{1,3,0,0,1,4,6,-2,1,0}) == result_tensor_type(7));
    REQUIRE(count_nonzero(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}}) == result_tensor_type(23));
    REQUIRE(count_nonzero(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},{1,0}) == result_tensor_type(23));
    REQUIRE(count_nonzero(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},true) == result_tensor_type{{23}});
    REQUIRE(count_nonzero(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},{0},true) == result_tensor_type{{5,5,4,4,5}});
}

//nonzero
TEST_CASE("test_sort_search_nonzero","[test_sort_search]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using order = tensor_type::order;
    using config_type = tensor_type::config_type;
    using index_type = tensor_type::index_type;
    using result_config_type = gtensor::config::extend_config_t<config_type,index_type>;
    using result_tensor_type = gtensor::tensor<index_type,order,result_config_type>;
    using result_container_type = config_type::template container<result_tensor_type>;
    using result_tensor_c_order_type = gtensor::tensor<index_type,gtensor::config::c_order,result_config_type>;
    using result_container_c_order_type = config_type::template container<result_tensor_c_order_type>;

    using gtensor::nonzero;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(nonzero(std::declval<tensor_type>())),result_container_type>);

    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},result_container_type{result_tensor_type{}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),result_container_c_order_type{result_tensor_c_order_type{},result_tensor_c_order_type{},result_tensor_c_order_type{}}),
        std::make_tuple(tensor_type{1},result_container_type{result_tensor_type{0}}),
        std::make_tuple(tensor_type{0},result_container_type{result_tensor_type{}}),
        std::make_tuple(tensor_type{0,0,0,0},result_container_type{result_tensor_type{}}),
        std::make_tuple(tensor_type{1,3,0,0,1,4,6,-2,1,0},result_container_type{result_tensor_type{0,1,4,5,6,7,8}}),
        std::make_tuple(
            tensor_type{{2,1,-1,6,0},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{0,0,6,0,3}},
            result_container_type{result_tensor_type{0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,3,4,4},result_tensor_type{0,1,2,3,0,1,2,4,0,1,3,4,0,1,2,3,4,2,4}}
        ),
        std::make_tuple(
            tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}},
            result_container_type{result_tensor_type{0,0,0,0,0,1,1,1,1,2,2,2},result_tensor_type{0,0,1,1,1,0,0,0,1,0,1,1},result_tensor_type{0,1,0,1,2,0,1,2,2,1,0,2}}
        ),
        std::make_tuple(
            tensor_type{{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}}},
            result_container_type{result_tensor_type{},result_tensor_type{},result_tensor_type{}}
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);

        auto result = nonzero(ten);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_sort_search_nonzero_index_map_view","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::nonzero;
    using helpers_for_testing::apply_by_element;

    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),tensor_type{}),
        std::make_tuple(tensor_type{1},tensor_type{1}),
        std::make_tuple(tensor_type{0},tensor_type{}),
        std::make_tuple(tensor_type{0,0,0,0},tensor_type{}),
        std::make_tuple(tensor_type{1,3,0,0,1,4,6,-2,1,0},tensor_type{1,3,1,4,6,-2,1}),
        std::make_tuple(tensor_type{{2,1,-1,6,0},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{0,0,6,0,3}},tensor_type{2,1,-1,6,8,2,1,5,-1,7,4,2,4,4,2,1,4,6,3}),
        std::make_tuple(tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}},tensor_type{1,3,2,2,2,1,5,2,1,1,1,1})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);

        auto result = ten(nonzero(ten));
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//argwhere
TEMPLATE_TEST_CASE("test_sort_search_argwhere","[test_sort_search]",
    //0tensor layout, 1traverse order
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using value_type = int;
    using layout = typename std::tuple_element_t<0,TestType>;
    using traverse_order = typename std::tuple_element_t<1,TestType>;
    using config_type = gtensor::config::extend_config_t<test_config::config_order_selector_t<traverse_order>,value_type>;
    using tensor_type = gtensor::tensor<value_type,layout,config_type>;
    using index_type = typename tensor_type::index_type;
    using result_config_type = gtensor::config::extend_config_t<config_type,index_type>;
    using result_tensor_type = gtensor::tensor<index_type,layout,result_config_type>;

    using gtensor::argwhere;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(argwhere(std::declval<tensor_type>())),result_tensor_type>);

    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},result_tensor_type{}.reshape(0,1)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),result_tensor_type{}.reshape(0,3)),
        std::make_tuple(tensor_type{1},result_tensor_type{{0}}),
        std::make_tuple(tensor_type{0},result_tensor_type{}.reshape(0,1)),
        std::make_tuple(tensor_type{0,0,0,0},result_tensor_type{}.reshape(0,1)),
        std::make_tuple(tensor_type{1,3,0,0,1,4,6,-2,1,0},result_tensor_type{{0},{1},{4},{5},{6},{7},{8}}),
        std::make_tuple(
            tensor_type{{2,1,-1,6,0},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{0,0,6,0,3}},
            result_tensor_type{{0,0},{0,1},{0,2},{0,3},{1,0},{1,1},{1,2},{1,4},{2,0},{2,1},{2,3},{2,4},{3,0},{3,1},{3,2},{3,3},{3,4},{4,2},{4,4}}
        ),
        std::make_tuple(
            tensor_type{{{1,3,0},{2,2,2}},{{1,5,2},{0,0,1}},{{0,1,0},{1,0,1}}},
            result_tensor_type{{0,0,0},{0,0,1},{0,1,0},{0,1,1},{0,1,2},{1,0,0},{1,0,1},{1,0,2},{1,1,2},{2,0,1},{2,1,0},{2,1,2}}
        ),
        std::make_tuple(
            tensor_type{{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}},{{0,0,0},{0,0,0}}},
            result_tensor_type{}.reshape(0,3)
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);

        auto result = argwhere(ten);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}