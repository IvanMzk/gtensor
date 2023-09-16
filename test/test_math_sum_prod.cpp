#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "tensor_math.hpp"
#include "tensor.hpp"


//sum,nansum
TEMPLATE_TEST_CASE("test_math_sum_nansum","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::sum;
    using gtensor::nansum;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(sum(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(sum(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(sum(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nansum(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nansum(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nansum(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);

    //0tensor,1axes,2keep_dims,3initial,4expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,value_type{0},tensor_type(value_type{0})),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,value_type{0},tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,value_type{0},tensor_type{value_type{0},value_type{0}}),
        std::make_tuple(tensor_type{5},0,false,value_type{0},tensor_type(5)),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,value_type{0},tensor_type(15)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,value_type{0},tensor_type(15)),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,value_type{0},tensor_type{{8,10,12},{14,16,18}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,value_type{0},tensor_type{{5,7,9},{17,19,21}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,value_type{0},tensor_type{{6,15},{24,33}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,value_type{0},tensor_type{22,26,30}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,value_type{0},tensor_type{21,57}),
        std::make_tuple(tensor_type{},std::vector<int>{},false,value_type{0},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,value_type{0},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,value_type{0},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,value_type{0},tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,true,value_type{0},tensor_type{15}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,value_type{0},tensor_type{{{5,7,9}},{{17,19,21}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,value_type{0},tensor_type{{{21}},{{57}}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,value_type{0},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,value_type{0},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,value_type{0},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}),
        //non zero initial
        std::make_tuple(tensor_type{},0,false,value_type{-1},tensor_type(-1)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),0,false,value_type{-1},tensor_type{{-1,-1,-1},{-1,-1,-1}}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,value_type{-1},tensor_type(14)),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,value_type{1},tensor_type{23,27,31})
    );
    auto test_sum = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = sum(policy...,ten,axes,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_nansum = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = nansum(policy...,ten,axes,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    //default policy
    SECTION("test_sum_default_policy")
    {
        test_sum();
    }
    SECTION("test_nansum_default_policy")
    {
        test_nansum();
    }
    //reduce_auto<4>
    SECTION("test_sum_reduce_auto<4>")
    {
        test_sum(gtensor::reduce_auto<4>{});
    }
    SECTION("test_nansum_reduce_auto<4>")
    {
        test_nansum(gtensor::reduce_auto<4>{});
    }
    //reduce_rng<1>
    SECTION("test_sum_reduce_rng<1>")
    {
        test_sum(gtensor::reduce_rng<1>{});
    }
    SECTION("test_nansum_reduce_rng<1>")
    {
        test_nansum(gtensor::reduce_rng<1>{});
    }
    //reduce_rng<4>
    SECTION("test_sum_reduce_rng<4>")
    {
        test_sum(gtensor::reduce_rng<4>{});
    }
    SECTION("test_nansum_reduce_rng<4>")
    {
        test_nansum(gtensor::reduce_rng<4>{});
    }
    //reduce_bin<1>
    SECTION("test_sum_reduce_bin<1>")
    {
        test_sum(gtensor::reduce_bin<1>{});
    }
    SECTION("test_nansum_reduce_bin<1>")
    {
        test_nansum(gtensor::reduce_bin<1>{});
    }
    //reduce_bin<4>
    SECTION("test_sum_reduce_bin<4>")
    {
        test_sum(gtensor::reduce_bin<4>{});
    }
    SECTION("test_nansum_reduce_bin<4>")
    {
        test_nansum(gtensor::reduce_bin<4>{});
    }
}

TEMPLATE_TEST_CASE("test_math_sum_nansum_overloads_default_policy","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::sum;
    using gtensor::nansum;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(sum(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(sum(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nansum(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nansum(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
    //sum
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{8,10,12},{14,16,18}});
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,1) == tensor_type{23,27,31});
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{22,26,30});
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{22,26,30});
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{21}},{{57}}});
    //all axes
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,1) == tensor_type(79));
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(78));
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(78));
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{78}}});
    //nansum
    REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{8,10,12},{14,16,18}});
    REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,1) == tensor_type{23,27,31});
    REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{22,26,30});
    REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{22,26,30});
    REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{21}},{{57}}});
    //all axes
    REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,1) == tensor_type(79));
    REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(78));
    REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(78));
    REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{78}}});
}

TEMPLATE_TEST_CASE("test_math_sum_nansum_overloads_policy","test_math",
    (gtensor::reduce_auto<4>),
    (gtensor::reduce_rng<1>),
    (gtensor::reduce_rng<4>),
    (gtensor::reduce_bin<1>),
    (gtensor::reduce_bin<4>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::sum;
    using gtensor::nansum;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(sum(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(sum(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nansum(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nansum(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
    //sum
    REQUIRE(sum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{8,10,12},{14,16,18}});
    REQUIRE(sum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,1) == tensor_type{23,27,31});
    REQUIRE(sum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{22,26,30});
    REQUIRE(sum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{22,26,30});
    REQUIRE(sum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{21}},{{57}}});
    //all axes
    REQUIRE(sum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,1) == tensor_type(79));
    REQUIRE(sum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(78));
    REQUIRE(sum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(78));
    REQUIRE(sum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{78}}});
    //nansum
    REQUIRE(nansum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{8,10,12},{14,16,18}});
    REQUIRE(nansum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,1) == tensor_type{23,27,31});
    REQUIRE(nansum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{22,26,30});
    REQUIRE(nansum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{22,26,30});
    REQUIRE(nansum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{21}},{{57}}});
    //all axes
    REQUIRE(nansum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,1) == tensor_type(79));
    REQUIRE(nansum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(78));
    REQUIRE(nansum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(78));
    REQUIRE(nansum(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{78}}});
}

TEST_CASE("test_math_sum_nansum_nan_values_default_policy","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::sum;
    using gtensor::nansum;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //sum
        std::make_tuple(sum(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(sum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(sum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(sum(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(sum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(sum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(sum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(sum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
        std::make_tuple(sum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,10.0}),
        std::make_tuple(sum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
        std::make_tuple(sum(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}),
        //nansum
        std::make_tuple(nansum(tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(nansum(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(nansum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf,0.0}), tensor_type(nan)),
        std::make_tuple(nansum(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nansum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(0.0)),
        std::make_tuple(nansum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{0.0,0.0,0.0}),
        std::make_tuple(nansum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{0.0,0.0,0.0}),
        std::make_tuple(nansum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(16.5)),
        std::make_tuple(nansum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{1.0,5.5,0.0,10.0}),
        std::make_tuple(nansum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{1.0,3.5,5.5,6.5}),
        std::make_tuple(nansum(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_math_sum_nansum_nan_values_policy","test_math",
    (gtensor::reduce_auto<4>),
    (gtensor::reduce_rng<1>),
    (gtensor::reduce_rng<4>),
    (gtensor::reduce_bin<1>),
    (gtensor::reduce_bin<4>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::sum;
    using gtensor::nansum;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //sum
        std::make_tuple(sum(policy{},tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(sum(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(sum(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(sum(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(sum(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(sum(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(sum(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(sum(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
        std::make_tuple(sum(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,10.0}),
        std::make_tuple(sum(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
        std::make_tuple(sum(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}),
        //nansum
        std::make_tuple(nansum(policy{},tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(nansum(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(nansum(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nansum(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nansum(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(0.0)),
        std::make_tuple(nansum(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{0.0,0.0,0.0}),
        std::make_tuple(nansum(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{0.0,0.0,0.0}),
        std::make_tuple(nansum(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(16.5)),
        std::make_tuple(nansum(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{1.0,5.5,0.0,10.0}),
        std::make_tuple(nansum(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{1.0,3.5,5.5,6.5}),
        std::make_tuple(nansum(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

//prod,nanprod
TEMPLATE_TEST_CASE("test_math_prod_nanprod","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::prod;
    using gtensor::nanprod;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(prod(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(prod(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(prod(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanprod(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanprod(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanprod(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);

    //0tensor,1axes,2keep_dims,3initial,4expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,value_type{1},tensor_type(value_type{1})),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,value_type{1},tensor_type(value_type{1})),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,value_type{1},tensor_type{value_type{1},value_type{1}}),
        std::make_tuple(tensor_type{5},0,false,value_type{1},tensor_type(5)),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,value_type{1},tensor_type(120)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,value_type{1},tensor_type(120)),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,value_type{1},tensor_type{{7,16,27},{40,55,72}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,value_type{1},tensor_type{{4,10,18},{70,88,108}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,value_type{1},tensor_type{{6,120},{504,1320}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,value_type{1},tensor_type{280,880,1944}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,value_type{1},tensor_type{720,665280}),
        std::make_tuple(tensor_type{},std::vector<int>{},false,value_type{1},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,value_type{1},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,value_type{1},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,value_type{1},tensor_type{value_type{1}}),
        std::make_tuple(tensor_type{},std::vector<int>{0},true,value_type{1},tensor_type{value_type{1}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,value_type{1},tensor_type{{{value_type{1}},{value_type{1}}}}),
        std::make_tuple(tensor_type{5},0,true,value_type{1},tensor_type{5}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,true,value_type{1},tensor_type{120}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,value_type{1},tensor_type{120}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,value_type{1},tensor_type{{{7,16,27},{40,55,72}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,value_type{1},tensor_type{{{4,10,18}},{{70,88,108}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,value_type{1},tensor_type{{{6},{120}},{{504},{1320}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,value_type{1},tensor_type{{{280,880,1944}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,value_type{1},tensor_type{{{720}},{{665280}}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,value_type{1},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,value_type{1},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,value_type{1},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}),
        //not one initial
        std::make_tuple(tensor_type{},0,false,value_type{-2},tensor_type(-2)),
        std::make_tuple(tensor_type{}.reshape(0,2,3),0,false,value_type{-2},tensor_type{{-2,-2,-2},{-2,-2,-2}}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,value_type{-2},tensor_type(-240)),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,value_type{-2},tensor_type{-560,-1760,-3888})
    );
    auto test_prod = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = prod(policy...,ten,axes,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    auto test_nanprod = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto initial = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = nanprod(policy...,ten,axes,keep_dims,initial);
            REQUIRE(result == expected);
        };
        apply_by_element(test,test_data);
    };
    //default policy
    SECTION("test_prod_default_policy")
    {
        test_prod();
    }
    SECTION("test_nanprod_default_policy")
    {
        test_nanprod();
    }
    //reduce_auto<4>
    SECTION("test_prod_reduce_auto<4>")
    {
        test_prod(gtensor::reduce_auto<4>{});
    }
    SECTION("test_nanprod_reduce_auto<4>")
    {
        test_nanprod(gtensor::reduce_auto<4>{});
    }
    //reduce_rng<1>
    SECTION("test_prod_reduce_rng<1>")
    {
        test_prod(gtensor::reduce_rng<1>{});
    }
    SECTION("test_nanprod_reduce_rng<1>")
    {
        test_nanprod(gtensor::reduce_rng<1>{});
    }
    //reduce_rng<4>
    SECTION("test_prod_reduce_rng<4>")
    {
        test_prod(gtensor::reduce_rng<4>{});
    }
    SECTION("test_nanprod_reduce_rng<4>")
    {
        test_nanprod(gtensor::reduce_rng<4>{});
    }
    //reduce_bin<1>
    SECTION("test_prod_reduce_bin<1>")
    {
        test_prod(gtensor::reduce_bin<1>{});
    }
    SECTION("test_nanprod_reduce_bin<1>")
    {
        test_nanprod(gtensor::reduce_bin<1>{});
    }
    //reduce_bin<4>
    SECTION("test_prod_reduce_bin<4>")
    {
        test_prod(gtensor::reduce_bin<4>{});
    }
    SECTION("test_nanprod_reduce_bin<4>")
    {
        test_nanprod(gtensor::reduce_bin<4>{});
    }
}

TEMPLATE_TEST_CASE("test_math_prod_nanprod_overloads_default_policy","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::prod;
    using gtensor::nanprod;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(prod(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(prod(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanprod(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanprod(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
    //prod
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{7,16,27},{40,55,72}});
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,-2) == tensor_type{-560,-1760,-3888});
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{280,880,1944});
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{280,880,1944});
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{720}},{{665280}}});
    //all axes
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,-2) == tensor_type(-958003200));
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(479001600));
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(479001600));
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{479001600}}});
    //nanprod
    REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{7,16,27},{40,55,72}});
    REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,-2) == tensor_type{-560,-1760,-3888});
    REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{280,880,1944});
    REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{280,880,1944});
    REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{720}},{{665280}}});
    //all axes
    REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,-2) == tensor_type(-958003200));
    REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(479001600));
    REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(479001600));
    REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{479001600}}});
}

TEMPLATE_TEST_CASE("test_math_prod_nanprod_overloads_policy","test_math",
    (gtensor::reduce_auto<4>),
    (gtensor::reduce_rng<1>),
    (gtensor::reduce_rng<4>),
    (gtensor::reduce_bin<1>),
    (gtensor::reduce_bin<4>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::prod;
    using gtensor::nanprod;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<typename decltype(prod(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(prod(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanprod(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
    REQUIRE(std::is_same_v<typename decltype(nanprod(policy{},std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
    //prod
    REQUIRE(prod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{7,16,27},{40,55,72}});
    REQUIRE(prod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,-2) == tensor_type{-560,-1760,-3888});
    REQUIRE(prod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{280,880,1944});
    REQUIRE(prod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{280,880,1944});
    REQUIRE(prod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{720}},{{665280}}});
    //all axes
    REQUIRE(prod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,-2) == tensor_type(-958003200));
    REQUIRE(prod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(479001600));
    REQUIRE(prod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(479001600));
    REQUIRE(prod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{479001600}}});
    //nanprod
    REQUIRE(nanprod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{7,16,27},{40,55,72}});
    REQUIRE(nanprod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,-2) == tensor_type{-560,-1760,-3888});
    REQUIRE(nanprod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{280,880,1944});
    REQUIRE(nanprod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{280,880,1944});
    REQUIRE(nanprod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{720}},{{665280}}});
    //all axes
    REQUIRE(nanprod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,-2) == tensor_type(-958003200));
    REQUIRE(nanprod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(479001600));
    REQUIRE(nanprod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(479001600));
    REQUIRE(nanprod(policy{},tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{479001600}}});
}

TEST_CASE("test_math_prod_nanprod_nan_values_default_policy","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::prod;
    using gtensor::nanprod;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //prod
        std::make_tuple(prod(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(prod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(prod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
        std::make_tuple(prod(tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(prod(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(prod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(prod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(prod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(prod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
        std::make_tuple(prod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,24.0}),
        std::make_tuple(prod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
        std::make_tuple(prod(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}),
        //nanprod
        std::make_tuple(nanprod(tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(nanprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(nanprod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
        std::make_tuple(nanprod(tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
        std::make_tuple(nanprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(1.0)),
        std::make_tuple(nanprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{1.0,1.0,1.0}),
        std::make_tuple(nanprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{1.0,1.0,1.0}),
        std::make_tuple(nanprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(36.0)),
        std::make_tuple(nanprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{0.25,6.0,1.0,24.0}),
        std::make_tuple(nanprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{1.0,3.0,3.0,4.0}),
        std::make_tuple(nanprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,1.0,2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_math_prod_nanprod_nan_values_policy","test_math",
    (gtensor::reduce_auto<4>),
    (gtensor::reduce_rng<1>),
    (gtensor::reduce_rng<4>),
    (gtensor::reduce_bin<1>),
    (gtensor::reduce_bin<4>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::prod;
    using gtensor::nanprod;
    using gtensor::tensor_equal;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
    //0result,1expected
    auto test_data = std::make_tuple(
        //prod
        std::make_tuple(prod(policy{},tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(prod(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(prod(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
        std::make_tuple(prod(policy{},tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(prod(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(prod(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
        std::make_tuple(prod(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
        std::make_tuple(prod(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
        std::make_tuple(prod(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
        std::make_tuple(prod(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,24.0}),
        std::make_tuple(prod(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
        std::make_tuple(prod(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}),
        //nanprod
        std::make_tuple(nanprod(policy{},tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
        std::make_tuple(nanprod(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
        std::make_tuple(nanprod(policy{},tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
        std::make_tuple(nanprod(policy{},tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
        std::make_tuple(nanprod(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
        std::make_tuple(nanprod(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(1.0)),
        std::make_tuple(nanprod(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{1.0,1.0,1.0}),
        std::make_tuple(nanprod(policy{},tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{1.0,1.0,1.0}),
        std::make_tuple(nanprod(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(36.0)),
        std::make_tuple(nanprod(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{0.25,6.0,1.0,24.0}),
        std::make_tuple(nanprod(policy{},tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{1.0,3.0,3.0,4.0}),
        std::make_tuple(nanprod(policy{},tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf},std::vector<int>{}), tensor_type{1.0,1.0,2.0,neg_inf,3.0,pos_inf})
    );
    auto test = [](const auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        REQUIRE(tensor_equal(result,expected,true));
    };
    apply_by_element(test,test_data);
}

