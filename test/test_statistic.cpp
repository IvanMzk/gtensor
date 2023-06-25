#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"
#include "statistic.hpp"

// //ptp
// TEMPLATE_TEST_CASE("test_statistic_ptp","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::ptp;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1axes,2keep_dims,3expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{5},0,false,tensor_type(0)),
//         std::make_tuple(tensor_type{5,6},0,false,tensor_type(1)),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,false,tensor_type(4)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,tensor_type(4)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,tensor_type(4)),
//         std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},0,false,tensor_type{{2,5,1},{2,2,9}}),
//         std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},1,false,tensor_type{{1,1,8},{1,4,2}}),
//         std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},2,false,tensor_type{{6,4},{4,7}}),
//         std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},std::vector<int>{0,2},false,tensor_type{6,9}),
//         std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},std::vector<int>{},false,tensor_type(9)),
//         //keep_dim true
//         std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},0,true,tensor_type{{{2,5,1},{2,2,9}}}),
//         std::make_tuple(tensor_type{{{5,2,8},{4,1,0}},{{3,7,7},{2,3,9}}},std::vector<int>{},true,tensor_type{{{9}}})
//     );

//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto keep_dims = std::get<2>(t);
//         auto expected = std::get<3>(t);
//         auto result = ptp(ten,axes,keep_dims);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_statistic_ptp_exception","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::reduce_exception;
//     using gtensor::ptp;
//     //zero size axis
//     REQUIRE_THROWS_AS(ptp(tensor_type{},0), reduce_exception);
// }

// //mean,nanmean
// TEMPLATE_TEST_CASE("test_statistic_mean_nanmean_normal_values","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::mean;
//     using gtensor::nanmean;
//     using helpers_for_testing::apply_by_element;

//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     static constexpr result_value_type nan = gtensor::math::numeric_traits<result_value_type>::nan();
//     using result_tensor_type = gtensor::tensor<result_value_type>;

//     REQUIRE(std::is_same_v<typename decltype(mean(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(mean(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmean(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmean(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);

//     //0tensor,1axes,2keep_dims,3expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{},std::vector<int>{0},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{},std::vector<int>{},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,result_tensor_type{nan,nan}),
//         std::make_tuple(tensor_type{5},0,false,result_tensor_type(5)),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,false,result_tensor_type(3)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,result_tensor_type(3)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,result_tensor_type(3)),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,result_tensor_type{{4,5,6},{7,8,9}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,result_tensor_type{{2.5,3.5,4.5},{8.5,9.5,10.5}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,result_tensor_type{{2,5},{8,11}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,result_tensor_type{5.5,6.5,7.5}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,result_tensor_type{3.5,9.5}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,result_tensor_type(6.5)),
//         //keep_dim true
//         std::make_tuple(tensor_type{},0,true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{},std::vector<int>{0},true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{},std::vector<int>{},true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,result_tensor_type{{{nan},{nan}}}),
//         std::make_tuple(tensor_type{5},0,true,result_tensor_type{5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,true,result_tensor_type{3}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,result_tensor_type{3}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,result_tensor_type{3}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,result_tensor_type{{{4,5,6},{7,8,9}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,result_tensor_type{{{2.5,3.5,4.5}},{{8.5,9.5,10.5}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,result_tensor_type{{{2},{5}},{{8},{11}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,result_tensor_type{{{5.5,6.5,7.5}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,result_tensor_type{{{3.5}},{{9.5}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,result_tensor_type{{{6.5}}})
//     );
//     SECTION("test_mean")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             auto result = mean(ten,axes,keep_dims);
//             REQUIRE(gtensor::tensor_equal(result,expected,true));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nanmean")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             auto result = nanmean(ten,axes,keep_dims);
//             REQUIRE(gtensor::tensor_equal(result,expected,true));
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_statistic_mean_nanmean_initializer_list_axes_all_axes","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::mean;
//     using gtensor::nanmean;
//     using helpers_for_testing::apply_by_element;
//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     using result_tensor_type = gtensor::tensor<result_value_type>;

//     REQUIRE(std::is_same_v<typename decltype(mean(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmean(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     //mean
//     REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == result_tensor_type{{4,5,6},{7,8,9}});
//     REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == result_tensor_type{5.5,6.5,7.5});
//     REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == result_tensor_type{{{3.5}},{{9.5}}});
//     //all axes
//     REQUIRE(mean(tensor_type{{{5}}}) == result_tensor_type(5));
//     REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == result_tensor_type(6.5));
//     REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == result_tensor_type(6.5));
//     REQUIRE(mean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == result_tensor_type{{{6.5}}});

//     //nanmean
//     REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == result_tensor_type{{4,5,6},{7,8,9}});
//     REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == result_tensor_type{5.5,6.5,7.5});
//     REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == result_tensor_type{{{3.5}},{{9.5}}});
//     //all axes
//     REQUIRE(nanmean(tensor_type{{{5}}}) == result_tensor_type(5));
//     REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == result_tensor_type(6.5));
//     REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == result_tensor_type(6.5));
//     REQUIRE(nanmean(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == result_tensor_type{{{6.5}}});
// }

// TEST_CASE("test_statistic_mean_nanmean_nan_values","test_statistic")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::mean;
//     using gtensor::nanmean;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //mean
//         std::make_tuple(mean(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
//         std::make_tuple(mean(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
//         std::make_tuple(mean(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(mean(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(mean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(mean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
//         std::make_tuple(mean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
//         std::make_tuple(mean(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
//         std::make_tuple(mean(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,2.5}),
//         std::make_tuple(mean(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
//         //nanmean
//         std::make_tuple(nanmean(tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
//         std::make_tuple(nanmean(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
//         std::make_tuple(nanmean(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nanmean(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nanmean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(nanmean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
//         std::make_tuple(nanmean(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
//         std::make_tuple(nanmean(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(2.0)),
//         std::make_tuple(nanmean(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.5,nan,4.5}},0), tensor_type{0.5,2.0,nan,3.0}),
//         std::make_tuple(nanmean(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,5.0},{0.5,2.0,nan,2.0}},1), tensor_type{2.5,1.75,2.5,1.5})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_equal(result,expected,true));
//     };
//     apply_by_element(test,test_data);
// }

// //var,nanvar
// TEMPLATE_TEST_CASE("test_statistic_var_nanvar_normal_values","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::var;
//     using gtensor::nanvar;
//     using gtensor::tensor_close;
//     using helpers_for_testing::apply_by_element;

//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     static constexpr result_value_type nan = gtensor::math::numeric_traits<result_value_type>::nan();
//     using result_tensor_type = gtensor::tensor<result_value_type>;
//     REQUIRE(std::is_same_v<typename decltype(var(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(var(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanvar(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanvar(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);

//     //0tensor,1axes,2keep_dims,3expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{},std::vector<int>{0},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{},std::vector<int>{},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,result_tensor_type{nan,nan}),
//         std::make_tuple(tensor_type{5},0,false,result_tensor_type(0)),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,false,result_tensor_type(2)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,result_tensor_type(2)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,result_tensor_type(2)),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,result_tensor_type{{9,9,9},{9,9,9}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,result_tensor_type{{2.25,2.25,2.25},{2.25,2.25,2.25}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,result_tensor_type{{0.666,0.666},{0.666,0.666}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,result_tensor_type{11.25,11.25,11.25}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,result_tensor_type{2.916,2.916}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,result_tensor_type(11.916)),
//         // //keep_dim true
//         std::make_tuple(tensor_type{},0,true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{},std::vector<int>{0},true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{},std::vector<int>{},true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,result_tensor_type{{{nan},{nan}}}),
//         std::make_tuple(tensor_type{5},0,true,result_tensor_type{0}),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,true,result_tensor_type{2}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,result_tensor_type{2}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,result_tensor_type{2}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,result_tensor_type{{{9,9,9},{9,9,9}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,result_tensor_type{{{2.25,2.25,2.25}},{{2.25,2.25,2.25}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,result_tensor_type{{{0.666},{0.666}},{{0.666},{0.666}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,result_tensor_type{{{11.25,11.25,11.25}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,result_tensor_type{{{2.916}},{{2.916}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,result_tensor_type{{{11.916}}})
//     );
//     SECTION("test_var")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             auto result = var(ten,axes,keep_dims);
//             REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nanvar")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             auto result = nanvar(ten,axes,keep_dims);
//             REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_statistic_var_nanvar_initializer_list_axes_all_axes","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::var;
//     using gtensor::nanvar;
//     using gtensor::tensor_close;
//     using helpers_for_testing::apply_by_element;
//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     using result_tensor_type = gtensor::tensor<result_value_type>;

//     REQUIRE(std::is_same_v<typename decltype(var(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanvar(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     //var
//     REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{9,9,9},{9,9,9}}, 1E-2, 1E-2));
//     REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{11.25,11.25,11.25}, 1E-2, 1E-2));
//     REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{2.916}},{{2.916}}}, 1E-2, 1E-2));
//     //all axes
//     REQUIRE(tensor_close(var(tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
//     REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(11.916), 1E-2, 1E-2));
//     REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(11.916), 1E-2, 1E-2));
//     REQUIRE(tensor_close(var(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{11.916}}}, 1E-2, 1E-2));

//     //nanvar
//     REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{9,9,9},{9,9,9}}, 1E-2, 1E-2));
//     REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{11.25,11.25,11.25}, 1E-2, 1E-2));
//     REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{2.916}},{{2.916}}}, 1E-2, 1E-2));
//     //all axes
//     REQUIRE(tensor_close(nanvar(tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
//     REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(11.916), 1E-2, 1E-2));
//     REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(11.916), 1E-2, 1E-2));
//     REQUIRE(tensor_close(nanvar(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{11.916}}}, 1E-2, 1E-2));
// }

// TEST_CASE("test_statistic_var_nanvar_nan_values","test_statistic")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::var;
//     using gtensor::nanvar;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //var
//         std::make_tuple(var(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(nan)),
//         std::make_tuple(var(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(var(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
//         std::make_tuple(var(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(var(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(var(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(var(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
//         std::make_tuple(var(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
//         std::make_tuple(var(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
//         std::make_tuple(var(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,1.25}),
//         std::make_tuple(var(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
//         //nanvar
//         std::make_tuple(nanvar(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(1.64)),
//         std::make_tuple(nanvar(tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nanvar(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
//         std::make_tuple(nanvar(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nanvar(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nanvar(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(nanvar(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
//         std::make_tuple(nanvar(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
//         std::make_tuple(nanvar(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(1.111)),
//         std::make_tuple(nanvar(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.5,nan,4.5}},0), tensor_type{0.0,0.166,nan,0.875}),
//         std::make_tuple(nanvar(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,5.0},{0.5,2.0,nan,2.0}},1), tensor_type{0.0,0.0625,3.5,0.5})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
//     };
//     apply_by_element(test,test_data);
// }

// //std,nanstd
// TEMPLATE_TEST_CASE("test_statistic_std_nanstd_normal_values","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::std;
//     using gtensor::nanstd;
//     using gtensor::tensor_close;
//     using helpers_for_testing::apply_by_element;

//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     static constexpr result_value_type nan = gtensor::math::numeric_traits<result_value_type>::nan();
//     using result_tensor_type = gtensor::tensor<result_value_type>;
//     REQUIRE(std::is_same_v<typename decltype(std(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(std(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanstd(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanstd(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);

//     //0tensor,1axes,2keep_dims,3expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{},std::vector<int>{0},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{},std::vector<int>{},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,result_tensor_type{nan,nan}),
//         std::make_tuple(tensor_type{5},0,false,result_tensor_type(0)),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,false,result_tensor_type(1.414)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,result_tensor_type(1.414)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,result_tensor_type(1.414)),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,result_tensor_type{{3,3,3},{3,3,3}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,result_tensor_type{{1.5,1.5,1.5},{1.5,1.5,1.5}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,result_tensor_type{{0.816,0.816},{0.816,0.816}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,result_tensor_type{3.354,3.354,3.354}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,result_tensor_type{1.707,1.707}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,result_tensor_type(3.452)),
//         // //keep_dim true
//         std::make_tuple(tensor_type{},0,true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{},std::vector<int>{0},true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{},std::vector<int>{},true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,result_tensor_type{{{nan},{nan}}}),
//         std::make_tuple(tensor_type{5},0,true,result_tensor_type{0}),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,true,result_tensor_type{1.414}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,result_tensor_type{1.414}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,result_tensor_type{1.414}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,result_tensor_type{{{3,3,3},{3,3,3}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,result_tensor_type{{{1.5,1.5,1.5}},{{1.5,1.5,1.5}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,result_tensor_type{{{0.816},{0.816}},{{0.816},{0.816}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,result_tensor_type{{{3.354,3.354,3.354}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,result_tensor_type{{{1.707}},{{1.707}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,result_tensor_type{{{3.452}}})
//     );
//     SECTION("test_std")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             auto result = std(ten,axes,keep_dims);
//             REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nanstd")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             auto result = nanstd(ten,axes,keep_dims);
//             REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_statistic_std_nanstd_initializer_list_axes_all_axes","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::std;
//     using gtensor::nanstd;
//     using gtensor::tensor_close;
//     using helpers_for_testing::apply_by_element;
//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     using result_tensor_type = gtensor::tensor<result_value_type>;

//     REQUIRE(std::is_same_v<typename decltype(std(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanstd(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     //std
//     REQUIRE(tensor_close(std(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{3,3,3},{3,3,3}}, 1E-2, 1E-2));
//     REQUIRE(tensor_close(std(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{3.354,3.354,3.354}, 1E-2, 1E-2));
//     REQUIRE(tensor_close(std(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{1.707}},{{1.707}}}, 1E-2, 1E-2));
//     //all axes
//     REQUIRE(tensor_close(std(tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
//     REQUIRE(tensor_close(std(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(3.452), 1E-2, 1E-2));
//     REQUIRE(tensor_close(std(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(3.452), 1E-2, 1E-2));
//     REQUIRE(tensor_close(std(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{3.452}}}, 1E-2, 1E-2));

//     //nanstd
//     REQUIRE(tensor_close(nanstd(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false), result_tensor_type{{3,3,3},{3,3,3}}, 1E-2, 1E-2));
//     REQUIRE(tensor_close(nanstd(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false), result_tensor_type{3.354,3.354,3.354}, 1E-2, 1E-2));
//     REQUIRE(tensor_close(nanstd(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true), result_tensor_type{{{1.707}},{{1.707}}}, 1E-2, 1E-2));
//     //all axes
//     REQUIRE(tensor_close(nanstd(tensor_type{{{5}}}), result_tensor_type(0), 1E-2, 1E-2));
//     REQUIRE(tensor_close(nanstd(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false), result_tensor_type(3.452), 1E-2, 1E-2));
//     REQUIRE(tensor_close(nanstd(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), result_tensor_type(3.452), 1E-2, 1E-2));
//     REQUIRE(tensor_close(nanstd(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true), result_tensor_type{{{3.452}}}, 1E-2, 1E-2));
// }

// TEST_CASE("test_statistic_std_nanstd_nan_values","test_statistic")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::std;
//     using gtensor::nanstd;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //std
//         std::make_tuple(std(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(nan)),
//         std::make_tuple(std(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(std(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
//         std::make_tuple(std(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(std(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(std(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(std(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
//         std::make_tuple(std(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
//         std::make_tuple(std(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
//         std::make_tuple(std(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,1.118}),
//         std::make_tuple(std(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
//         //nanstd
//         std::make_tuple(nanstd(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(1.28)),
//         std::make_tuple(nanstd(tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nanstd(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(nan)),
//         std::make_tuple(nanstd(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nanstd(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nanstd(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(nanstd(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
//         std::make_tuple(nanstd(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
//         std::make_tuple(nanstd(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(1.054)),
//         std::make_tuple(nanstd(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.5,nan,4.5}},0), tensor_type{0.0,0.408,nan,0.935}),
//         std::make_tuple(nanstd(tensor_type{{nan,nan,nan,2.5},{nan,1.5,nan,2.0},{0.5,2.0,nan,5.0},{0.5,2.0,nan,2.0}},1), tensor_type{0.0,0.25,1.87,0.707})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2,true));
//     };
//     apply_by_element(test,test_data);
// }

// //median,nanmedian
// TEMPLATE_TEST_CASE("test_statistic_median_nanmedian_normal_values","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::median;
//     using gtensor::nanmedian;
//     using gtensor::tensor_close;
//     using helpers_for_testing::apply_by_element;

//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     static constexpr result_value_type nan = gtensor::math::numeric_traits<result_value_type>::nan();
//     using result_tensor_type = gtensor::tensor<result_value_type>;
//     REQUIRE(std::is_same_v<typename decltype(median(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(median(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmedian(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmedian(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,result_value_type>);

//     //0tensor,1axes,2keep_dims,3expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{},std::vector<int>{0},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{},std::vector<int>{},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,result_tensor_type{nan,nan}),
//         std::make_tuple(tensor_type{5},0,false,result_tensor_type(5)),
//         std::make_tuple(tensor_type{5,6},0,false,result_tensor_type(5.5)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,false,result_tensor_type(2)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,false,result_tensor_type(2.5)),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0,false,result_tensor_type{1,2,1,5,3,1}),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},1,false,result_tensor_type{1.0,2.0,2.0,1.5,4.0}),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},std::vector<int>{1,0},false,result_tensor_type(2.0)),
//         //keep_dim true
//         std::make_tuple(tensor_type{},0,true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{},std::vector<int>{0},true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{},std::vector<int>{},true,result_tensor_type{nan}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,result_tensor_type{{{nan},{nan}}}),
//         std::make_tuple(tensor_type{5},0,true,result_tensor_type{5}),
//         std::make_tuple(tensor_type{5,6},0,true,result_tensor_type{5.5}),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,true,result_tensor_type{2}),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,true,result_tensor_type{2.5}),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0,true,result_tensor_type{{1,2,1,5,3,1}}),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},1,true,result_tensor_type{{1.0},{2.0},{2.0},{1.5},{4.0}}),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},std::vector<int>{1,0},true,result_tensor_type{{2.0}})
//     );
//     SECTION("test_median")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             auto result = median(ten,axes,keep_dims);
//             REQUIRE(tensor_equal(result,expected,true));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nanmedian")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             auto result = nanmedian(ten,axes,keep_dims);
//             REQUIRE(tensor_equal(result,expected,true));
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_statistic_median_nanmedian_initializer_list_axes_all_axes","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::median;
//     using gtensor::nanmedian;
//     using gtensor::tensor_close;
//     using helpers_for_testing::apply_by_element;
//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     using result_tensor_type = gtensor::tensor<result_value_type>;

//     REQUIRE(std::is_same_v<typename decltype(median(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmedian(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,result_value_type>);
//     //median
//     REQUIRE(median(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},false) == result_tensor_type{1,2,1,5,3,1});
//     REQUIRE(median(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1}) == result_tensor_type{1.0,2.0,2.0,1.5,4.0});
//     REQUIRE(median(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0}) == result_tensor_type(2.0));
//     REQUIRE(median(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}}) == result_tensor_type(2.0));
//     //nanmedian
//     REQUIRE(nanmedian(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},false) == result_tensor_type{1,2,1,5,3,1});
//     REQUIRE(nanmedian(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1}) == result_tensor_type{1.0,2.0,2.0,1.5,4.0});
//     REQUIRE(nanmedian(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0}) == result_tensor_type(2.0));
//     REQUIRE(nanmedian(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}}) == result_tensor_type(2.0));
// }

// TEST_CASE("test_statistic_median_nanmedian_nan_values","test_statistic")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::median;
//     //using gtensor::nanmedian;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //median
//         std::make_tuple(median(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(nan)),
//         std::make_tuple(median(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(2.5)),
//         std::make_tuple(median(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(1.5)),
//         std::make_tuple(median(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(1.5)),
//         std::make_tuple(median(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(median(tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(median(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}), tensor_type(nan)),
//         std::make_tuple(median(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0), tensor_type{nan,nan,nan,3.0}),
//         std::make_tuple(median(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1), tensor_type{nan,nan,nan,nan,1.5}),
//         //nanmedian
//         std::make_tuple(nanmedian(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}), tensor_type(2.0)),
//         std::make_tuple(nanmedian(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(2.5)),
//         std::make_tuple(nanmedian(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(1.5)),
//         std::make_tuple(nanmedian(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(1.5)),
//         std::make_tuple(nanmedian(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(2.0)),
//         std::make_tuple(nanmedian(tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(nanmedian(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}), tensor_type(2.0)),
//         std::make_tuple(nanmedian(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0), tensor_type{1.5,1.0,0.0,3.0}),
//         std::make_tuple(nanmedian(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1), tensor_type{1.5,-3.0,3.0,8.0,1.5})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_equal(result,expected,true));
//     };
//     apply_by_element(test,test_data);
// }

// //quantile,nanquantile
// TEMPLATE_TEST_CASE("test_statistic_quantile_nanquantile_normal_values","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::quantile;
//     using gtensor::nanquantile;
//     using gtensor::tensor_close;
//     using helpers_for_testing::apply_by_element;

//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     static constexpr result_value_type nan = gtensor::math::numeric_traits<result_value_type>::nan();
//     using result_tensor_type = gtensor::tensor<result_value_type>;
//     REQUIRE(std::is_same_v<typename decltype(quantile(std::declval<tensor_type>(),std::declval<int>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(quantile(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanquantile(std::declval<tensor_type>(),std::declval<int>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,result_value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanquantile(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,result_value_type>);

//     //0tensor,1axes,2quantile,3keep_dims,4expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,result_value_type{0.5},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{},std::vector<int>{0},result_value_type{0.5},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{},std::vector<int>{},result_value_type{0.5},false,result_tensor_type(nan)),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},result_value_type{0.5},false,result_tensor_type{nan,nan}),
//         std::make_tuple(tensor_type{5},0,result_value_type{0.5},false,result_tensor_type(5)),
//         std::make_tuple(tensor_type{5},0,result_value_type{0.0},false,result_tensor_type(5)),
//         std::make_tuple(tensor_type{5},0,result_value_type{0.2},false,result_tensor_type(5)),
//         std::make_tuple(tensor_type{5},0,result_value_type{0.8},false,result_tensor_type(5)),
//         std::make_tuple(tensor_type{5},0,result_value_type{1.0},false,result_tensor_type(5)),
//         std::make_tuple(tensor_type{5,6},0,result_value_type{0.5},false,result_tensor_type(5.5)),
//         std::make_tuple(tensor_type{6,5},0,result_value_type{0.0},false,result_tensor_type(5.0)),
//         std::make_tuple(tensor_type{6,5},0,result_value_type{0.2},false,result_tensor_type(5.2)),
//         std::make_tuple(tensor_type{5,6},0,result_value_type{0.8},false,result_tensor_type(5.8)),
//         std::make_tuple(tensor_type{5,6},0,result_value_type{1.0},false,result_tensor_type(6.0)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,result_value_type{0.5},false,result_tensor_type(2.0)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,result_value_type{0.0},false,result_tensor_type(-1.0)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,result_value_type{0.2},false,result_tensor_type(1.0)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,result_value_type{0.8},false,result_tensor_type(4.0)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4},0,result_value_type{1.0},false,result_tensor_type(6.0)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,result_value_type{0.5},false,result_tensor_type(2.5)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,result_value_type{0.0},false,result_tensor_type(-1.0)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,result_value_type{0.2},false,result_tensor_type(1.0)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,result_value_type{0.8},false,result_tensor_type(4.8)),
//         std::make_tuple(tensor_type{1,3,3,5,2,6,0,2,-1,1,4,5},0,result_value_type{1.0},false,result_tensor_type(6.0)),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0,result_value_type{0.3},false,result_tensor_type{1.0,1.2,0.2,4.2,1.4,-0.6}),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},1,result_value_type{0.3},false,result_tensor_type{0.5,0.5,1.5,0.5,2.0}),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},std::vector<int>{1,0},result_value_type{0.3},false,result_tensor_type(1.0)),
//         // //keep_dim true
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0,result_value_type{0.3},true,result_tensor_type{{1.0,1.2,0.2,4.2,1.4,-0.6}}),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},1,result_value_type{0.3},true,result_tensor_type{{0.5},{0.5},{1.5},{0.5},{2.0}}),
//         std::make_tuple(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},std::vector<int>{1,0},result_value_type{0.3},true,result_tensor_type{{1.0}})

//     );
//     SECTION("test_quantile")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto q = std::get<2>(t);
//             auto keep_dims = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto result = quantile(ten,axes,q,keep_dims);
//             REQUIRE(tensor_close(result,expected,true));
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nanquantile")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto q = std::get<2>(t);
//             auto keep_dims = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto result = nanquantile(ten,axes,q,keep_dims);
//             REQUIRE(tensor_close(result,expected,true));
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_statistic_quantile_nanquantile_initializer_list_axes_all_axes","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::quantile;
//     using gtensor::nanquantile;
//     using gtensor::tensor_close;
//     using helpers_for_testing::apply_by_element;
//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     using result_tensor_type = gtensor::tensor<result_value_type>;

//     REQUIRE(
//         std::is_same_v<
//             typename decltype(quantile(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,
//             result_value_type
//         >
//     );
//     REQUIRE(
//         std::is_same_v<
//             typename decltype(nanquantile(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<result_value_type>(),std::declval<bool>()))::value_type,
//             result_value_type
//         >
//     );
//     //quantile
//     REQUIRE(tensor_close(quantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},0.3,false), result_tensor_type{1.0,1.2,0.2,4.2,1.4,-0.6}));
//     REQUIRE(tensor_close(quantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1},0.3), result_tensor_type{0.5,0.5,1.5,0.5,2.0}));
//     REQUIRE(tensor_close(quantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0},0.3), result_tensor_type(1.0)));
//     REQUIRE(tensor_close(quantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0.3), result_tensor_type(1.0)));
//     //nanquantile
//     REQUIRE(tensor_close(nanquantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{0},0.3,false), result_tensor_type{1.0,1.2,0.2,4.2,1.4,-0.6}));
//     REQUIRE(tensor_close(nanquantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1},0.3), result_tensor_type{0.5,0.5,1.5,0.5,2.0}));
//     REQUIRE(tensor_close(nanquantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},{1,0},0.3), result_tensor_type(1.0)));
//     REQUIRE(tensor_close(nanquantile(tensor_type{{1,1,0,2,5,-1},{-2,3,1,6,0,4},{3,2,2,7,1,-2},{1,0,-2,4,3,2},{5,3,1,5,7,1}},0.3), result_tensor_type(1.0)));
// }

// TEMPLATE_TEST_CASE("test_statistic_quantile_nanquantile_exception","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::reduce_exception;
//     using gtensor::quantile;
//     using gtensor::nanquantile;

//     REQUIRE_THROWS_AS(quantile(tensor_type{1,2,3,4,5},1.1), reduce_exception);
//     REQUIRE_THROWS_AS(nanquantile(tensor_type{1,2,3,4,5},1.1), reduce_exception);
// }

// TEST_CASE("test_statistic_quantile_nanquantile_nan_values","test_statistic")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::quantile;
//     using gtensor::nanquantile;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //quantile
//         std::make_tuple(quantile(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}, value_type{0.5}), tensor_type(nan)),
//         std::make_tuple(quantile(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}, value_type{0.5}), tensor_type(2.5)),
//         std::make_tuple(quantile(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}, value_type{0.5}), tensor_type(1.5)),
//         std::make_tuple(quantile(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(1.5)),
//         std::make_tuple(quantile(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(nan)),
//         std::make_tuple(quantile(tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}, value_type{0.5}), tensor_type(nan)),
//         std::make_tuple(quantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}, value_type{0.5}), tensor_type(nan)),
//         std::make_tuple(quantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0, value_type{0.5}), tensor_type{nan,nan,nan,3.0}),
//         std::make_tuple(quantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1, value_type{0.5}), tensor_type{nan,nan,nan,nan,1.5}),
//         //nanquantile
//         std::make_tuple(nanquantile(tensor_type{1.0,0.5,nan,4.0,3.0,2.0}, value_type{0.5}), tensor_type(2.0)),
//         std::make_tuple(nanquantile(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}, value_type{0.5}), tensor_type(2.5)),
//         std::make_tuple(nanquantile(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}, value_type{0.5}), tensor_type(1.5)),
//         std::make_tuple(nanquantile(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(1.5)),
//         std::make_tuple(nanquantile(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}, value_type{0.5}), tensor_type(2.0)),
//         std::make_tuple(nanquantile(tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan},{nan,nan,nan,nan}}, value_type{0.5}), tensor_type(nan)),
//         std::make_tuple(nanquantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}}, value_type{0.5}), tensor_type(2.0)),
//         std::make_tuple(nanquantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},0, value_type{0.5}), tensor_type{1.5,1.0,0.0,3.0}),
//         std::make_tuple(nanquantile(tensor_type{{1.0,nan,nan,2.0},{nan,nan,nan,-3.0},{nan,nan,nan,3.0},{nan,nan,nan,8.0},{2.0,1.0,0.0,4.0}},1, value_type{0.5}), tensor_type{1.5,-3.0,3.0,8.0,1.5})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_equal(result,expected,true));
//     };
//     apply_by_element(test,test_data);
// }

// //average
// TEMPLATE_TEST_CASE("test_statistic_average","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using gtensor::average;
//     using helpers_for_testing::apply_by_element;

//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     using result_tensor_type = gtensor::tensor<result_value_type>;

//     REQUIRE(std::is_same_v<
//         typename decltype(average(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<std::vector<value_type>>(),std::declval<bool>()))::value_type,
//         result_value_type>
//     );
//     REQUIRE(std::is_same_v<
//         typename decltype(average(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<std::vector<value_type>>(),std::declval<bool>()))::value_type,
//         result_value_type>
//     );

//     //0tensor,1axes,2keep_dims,3weights,4expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{6},dim_type{0},false,std::vector<value_type>{2},result_tensor_type(6.0)),
//         std::make_tuple(tensor_type{1,2,3,4,5,6},dim_type{0},false,std::vector<value_type>{6,5,4,3,2,1},result_tensor_type(2.666)),
//         std::make_tuple(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},dim_type{0},false,tensor_type{1,2,2,1},result_tensor_type{1.833,2.5,3.333,1.166,1.5,2.166}),
//         std::make_tuple(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},dim_type{1},false,tensor_type{1,2,2,2,2,1},result_tensor_type{1.4,1.7,1.8,4.2}),
//         std::make_tuple(
//             tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},
//             std::vector<dim_type>{1,0},
//             false,
//             tensor_type{1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1},
//             result_tensor_type(2.15)
//         ),
//         //keep_dim true
//         std::make_tuple(tensor_type{6},dim_type{0},true,std::vector<value_type>{2},result_tensor_type{6.0}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6},dim_type{0},true,std::vector<value_type>{6,5,4,3,2,1},result_tensor_type{2.666}),
//         std::make_tuple(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},dim_type{0},true,tensor_type{1,2,2,1},result_tensor_type{{1.833,2.5,3.333,1.166,1.5,2.166}}),
//         std::make_tuple(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},dim_type{1},true,tensor_type{1,2,2,2,2,1},result_tensor_type{{1.4},{1.7},{1.8},{4.2}}),
//         std::make_tuple(
//             tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},
//             std::vector<dim_type>{1,0},
//             true,
//             tensor_type{1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1},
//             result_tensor_type{{2.15}}
//         )
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto keep_dims = std::get<2>(t);
//         auto weights = std::get<3>(t);
//         auto expected = std::get<4>(t);
//         auto result = average(ten,axes,weights,keep_dims);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_statistic_average_initializer_list_axes_all_axes","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using gtensor::average;
//     using gtensor::tensor_close;
//     using helpers_for_testing::apply_by_element;
//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     using result_tensor_type = gtensor::tensor<result_value_type>;

//     REQUIRE(std::is_same_v<
//         typename decltype(average(std::declval<tensor_type>(),std::declval<std::initializer_list<dim_type>>(),std::declval<std::vector<value_type>>(),std::declval<bool>()))::value_type,
//         result_value_type>
//     );
//     REQUIRE(
//         tensor_close(
//             average(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},{0},tensor_type{1,2,2,1}),
//             result_tensor_type{1.833,2.5,3.333,1.166,1.5,2.166},
//             1E-2,
//             1E-2
//         )
//     );
//     REQUIRE(
//         tensor_close(
//             average(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},{0,1},tensor_type{1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1}),
//             result_tensor_type(2.15),
//             1E-2,
//             1E-2
//         )
//     );
//     //all axes
//     REQUIRE(
//         tensor_close(
//             average(tensor_type{{1,2,2,0,1,3},{-1,3,2,2,0,4},{4,2,3,1,1,0},{4,3,8,1,6,2}},tensor_type{1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1}),
//             result_tensor_type(2.15),
//             1E-2,
//             1E-2
//         )
//     );
// }

// TEMPLATE_TEST_CASE("test_statistic_average_exception","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::average;
//     using gtensor::reduce_exception;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1axes,2keep_dims,3weights
//     auto test_data = std::make_tuple(
//         //zero size weights
//         std::make_tuple(tensor_type{},0,false,tensor_type{}),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,false,tensor_type{1,1,0,-1,-1}),
//         //weights size not match size along axes
//         std::make_tuple(tensor_type{1,2,3,4,5},0,false,tensor_type{1,1,2}),
//         std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},0,false,tensor_type{1,1,2,2}),
//         std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},0,false,tensor_type{1,2}),
//         std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},1,false,tensor_type{1,2,2}),
//         std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},1,false,tensor_type{1,2,2,1,1}),
//         std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12}},std::vector<int>{1,0},false,tensor_type{1,2,2,1,1})
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto keep_dims = std::get<2>(t);
//         auto weights = std::get<3>(t);
//         REQUIRE_THROWS_AS(average(ten,axes,weights,keep_dims), reduce_exception);
//     };
//     apply_by_element(test,test_data);
// }

// //moving average
// TEMPLATE_TEST_CASE("test_statistic_moving_average","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using index_type = typename tensor_type::index_type;
//     using gtensor::tensor_close;
//     using gtensor::moving_average;
//     using helpers_for_testing::apply_by_element;

//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     using result_tensor_type = gtensor::tensor<result_value_type>;

//     REQUIRE(std::is_same_v<
//         typename decltype(moving_average(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<std::vector<value_type>>(),std::declval<index_type>()))::value_type,
//         result_value_type>
//     );

//     //0tensor,1axis,2weights,3step,4expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{}.reshape(0,4,5),1,tensor_type{1,1},1,result_tensor_type{}.reshape(0,3,5)),
//         std::make_tuple(tensor_type{}.reshape(0,4,5),2,tensor_type{1,1,1},1,result_tensor_type{}.reshape(0,4,3)),
//         std::make_tuple(tensor_type{5},0,tensor_type{2},1,result_tensor_type{5}),
//         std::make_tuple(tensor_type{5},0,tensor_type{2},2,result_tensor_type{5}),
//         std::make_tuple(tensor_type{5,6},0,tensor_type{2},1,result_tensor_type{5,6}),
//         std::make_tuple(tensor_type{5,6},0,tensor_type{2,3},1,result_tensor_type{5.6}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,tensor_type{1,2,2,1},1,result_tensor_type{2.5,3.5,4.5,5.5,6.5,7.5,8.5}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,tensor_type{1,1,1,2},1,result_tensor_type{2.8,3.8,4.8,5.8,6.8,7.8,8.8}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,tensor_type{1,2,2,1},2,result_tensor_type{2.5,4.5,6.5,8.5}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,tensor_type{2,3,3,2},5,result_tensor_type{2.5,7.5}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,tensor_type{1,2,3},3,result_tensor_type{2.333,5.333,8.333}),
//         std::make_tuple(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},0,tensor_type{1,2,3},1,result_tensor_type{{0.833,1.333,0.666,3.0,5.166},{2.666,1.666,0.166,4.0,3.333}}),
//         std::make_tuple(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},1,tensor_type{1,2,3},1,result_tensor_type{{0.833,-0.333,1.666},{3.333,3.0,3.0},{-0.666,2.0,4.833},{1.5,2.333,1.833}})
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axis = std::get<1>(t);
//         auto weights = std::get<2>(t);
//         auto step = std::get<3>(t);
//         auto expected = std::get<4>(t);
//         auto result = moving_average(ten,axis,weights,step);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_statistic_moving_average_exception","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::moving_average;
//     using gtensor::reduce_exception;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1axis,2weights,3step
//     auto test_data = std::make_tuple(
//         //zero window size
//         std::make_tuple(tensor_type{},0,tensor_type{},1),
//         std::make_tuple(tensor_type{}.reshape(0,4,5),0,tensor_type{},1),
//         std::make_tuple(tensor_type{1,2,3},0,tensor_type{},1),
//         //zero size weights
//         std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,-1,0},1),
//         //weights size greater than axis size
//         std::make_tuple(tensor_type{},0,tensor_type{1},1),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,1,2,2,3,3},1),
//         //zero step
//         std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,1,2},0)
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axis = std::get<1>(t);
//         auto weights = std::get<2>(t);
//         auto step = std::get<3>(t);
//         REQUIRE_THROWS_AS(moving_average(ten,axis,weights,step), reduce_exception);
//     };
//     apply_by_element(test,test_data);
// }

// //moving mean
// TEMPLATE_TEST_CASE("test_statistic_moving_mean","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using dim_type = typename tensor_type::dim_type;
//     using index_type = typename tensor_type::index_type;
//     using gtensor::tensor_close;
//     using gtensor::moving_mean;
//     using helpers_for_testing::apply_by_element;

//     using result_value_type = typename gtensor::math::numeric_traits<value_type>::floating_point_type;
//     using result_tensor_type = gtensor::tensor<result_value_type>;

//     REQUIRE(std::is_same_v<
//         typename decltype(moving_mean(std::declval<tensor_type>(),std::declval<dim_type>(),std::declval<index_type>(),std::declval<index_type>()))::value_type,
//         result_value_type>
//     );

//     //0tensor,1axis,2window_size,3step,4expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{}.reshape(0,4,5),1,2,1,result_tensor_type{}.reshape(0,3,5)),
//         std::make_tuple(tensor_type{}.reshape(0,4,5),2,3,1,result_tensor_type{}.reshape(0,4,3)),
//         std::make_tuple(tensor_type{5},0,1,1,result_tensor_type{5}),
//         std::make_tuple(tensor_type{5},0,1,2,result_tensor_type{5}),
//         std::make_tuple(tensor_type{5,6},0,1,1,result_tensor_type{5,6}),
//         std::make_tuple(tensor_type{5,6},0,2,1,result_tensor_type{5.5}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,4,1,result_tensor_type{2.5,3.5,4.5,5.5,6.5,7.5,8.5}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,4,2,result_tensor_type{2.5,4.5,6.5,8.5}),
//         std::make_tuple(tensor_type{1,2,3,4,5,5,4,3,2,1},0,4,3,result_tensor_type{2.5,4.5,2.5}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},0,4,4,result_tensor_type{2.5,6.5}),
//         std::make_tuple(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},0,3,1,result_tensor_type{{1.333,1.333,1.0,2.0,4.666},{2.0,1.666,1.0,3.666,3.666}}),
//         std::make_tuple(tensor_type{{3,1,0,-1,4},{1,2,5,2,3},{0,1,-2,5,7},{5,2,0,4,1}},1,3,1,result_tensor_type{{1.333,0.0,1.0},{2.666,3.0,3.333},{-0.333,1.333,3.333},{2.333,2.0,1.666}})
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axis = std::get<1>(t);
//         auto window_size = std::get<2>(t);
//         auto step = std::get<3>(t);
//         auto expected = std::get<4>(t);
//         auto result = moving_mean(ten,axis,window_size,step);
//         REQUIRE(tensor_close(result,expected,1E-2,1E-2));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_statistic_moving_mean_exception","test_statistic",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::moving_mean;
//     using gtensor::reduce_exception;
//     using helpers_for_testing::apply_by_element;

//     //0tensor,1axis,2window_size,3step
//     auto test_data = std::make_tuple(
//         //zero window size
//         std::make_tuple(tensor_type{},0,0,1),
//         std::make_tuple(tensor_type{}.reshape(0,4,5),0,0,1),
//         std::make_tuple(tensor_type{1,2,3},0,0,1),
//         //window_size size greater than axis size
//         std::make_tuple(tensor_type{},0,1,1),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,6,1),
//         //zero step
//         std::make_tuple(tensor_type{1,2,3,4,5},0,3,0)
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axis = std::get<1>(t);
//         auto window_size = std::get<2>(t);
//         auto step = std::get<3>(t);
//         REQUIRE_THROWS_AS(moving_mean(ten,axis,window_size,step), reduce_exception);
//     };
//     apply_by_element(test,test_data);
// }

//histogram
TEMPLATE_TEST_CASE("test_statistic_histogram","test_statistic",
    double
    //int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using result_value_type = gtensor::math::make_floating_point_t<value_type>;
    using result_tensor_type = gtensor::tensor<result_value_type>;
    using gtensor::detail::no_value;
    using gtensor::histogram_algorithm;
    using gtensor::histogram;
    using helpers_for_testing::apply_by_element;

    //0tensor,1bins,2range,3density,4expected
    auto test_data = std::make_tuple(
        //empty source
        std::make_tuple(tensor_type{},1,no_value{},false,result_tensor_type{0}),
        std::make_tuple(tensor_type{},5,no_value{},false,result_tensor_type{0,0,0,0,0}),
        //bins integral, no range
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},1,no_value{},false,result_tensor_type{13}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},2,no_value{},false,result_tensor_type{6,7}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},5,no_value{},false,result_tensor_type{2,4,2,2,3}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,no_value{},false,result_tensor_type{2,3,1,2,1,1,3}),
        std::make_tuple(
            tensor_type{0.93,0.69,0.774,0.977,0.437,0.606,0.485,0.394,0.466,0.888,0.47,0.941,0.396,0.886,0.857,0.565,0.368,0.52,0.671,0.764,0.006,0.096,0.921,0.778,0.366,0.477,0.666,0.381,0.375,0.65},
            6,
            no_value{},
            false,
            result_tensor_type{2,0,11,4,6,7}
        ),
        //bins integral, range
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},5,std::make_pair(2,6),false,result_tensor_type{1,2,1,1,2}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(2,6),false,result_tensor_type{1,2,0,1,0,1,2}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(-4,6),false,result_tensor_type{0,0,2,3,3,1,3}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(3,11),false,result_tensor_type{3,1,2,1,0,0,0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(-5.3,13.3),false,result_tensor_type{0,2,4,4,3,0,0}),
        std::make_tuple(
            tensor_type{0.93,0.69,0.774,0.977,0.437,0.606,0.485,0.394,0.466,0.888,0.47,0.941,0.396,0.886,0.857,0.565,0.368,0.52,0.671,0.764,0.006,0.096,0.921,0.778,0.366,0.477,0.666,0.381,0.375,0.65},
            6,
            std::make_pair(0.25,0.75),
            false,
            result_tensor_type{0,6,5,2,3,2}
        ),
        //source out of range
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},5,std::make_pair(-32,-10),false,result_tensor_type{0,0,0,0,0}),
        std::make_tuple(tensor_type{3,1,4,2,6,5,6,1,0,-1,3,7,1},7,std::make_pair(10,32),false,result_tensor_type{0,0,0,0,0,0,0})
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto bins = std::get<1>(t);
        auto range = std::get<2>(t);
        auto density = std::get<3>(t);
        auto expected = std::get<4>(t);

        auto result = histogram(ten,bins,range,density);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}