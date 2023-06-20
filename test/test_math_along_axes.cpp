#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"

// //test math functions along axes
// //all
// TEST_CASE("test_math_all","test_math")
// {
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using bool_tensor_type = gtensor::tensor<bool>;
//     using gtensor::all;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(all(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,bool>);
//     REQUIRE(std::is_same_v<typename decltype(all(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,bool>);

//     //0tensor,1axes,2keep_dims,3expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{},std::vector<int>{0},false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{},std::vector<int>{},false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,bool_tensor_type{true,true}),
//         std::make_tuple(tensor_type{5},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{0},0,false,bool_tensor_type(false)),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{5,0,1,-1,4,4},0,false,bool_tensor_type(false)),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},0,false,bool_tensor_type{{true,false,false},{true,false,true}}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},1,false,bool_tensor_type{{true,false,false},{true,false,true}}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},2,false,bool_tensor_type{{false,false},{false,true}}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{0,1},false,bool_tensor_type{true,false,false}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{2,1},false,bool_tensor_type{false,false}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{},false,bool_tensor_type(false)),
//         //keep_dim true
//         std::make_tuple(tensor_type{},0,true,bool_tensor_type{true}),
//         std::make_tuple(tensor_type{},std::vector<int>{0},true,bool_tensor_type{true}),
//         std::make_tuple(tensor_type{},std::vector<int>{},true,bool_tensor_type{true}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,bool_tensor_type{{{true},{true}}}),
//         std::make_tuple(tensor_type{5},0,true,bool_tensor_type{true}),
//         std::make_tuple(tensor_type{0},0,true,bool_tensor_type{false}),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},0,true,bool_tensor_type{true}),
//         std::make_tuple(tensor_type{5,0,1,-1,4,4},0,true,bool_tensor_type{false}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},0,true,bool_tensor_type{{{true,false,false},{true,false,true}}}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},1,true,bool_tensor_type{{{true,false,false}},{{true,false,true}}}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},2,true,bool_tensor_type{{{false},{false}},{{false},{true}}}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{0,1},true,bool_tensor_type{{{true,false,false}}}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{2,1},true,bool_tensor_type{{{false}},{{false}}}),
//         std::make_tuple(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},std::vector<int>{},true,bool_tensor_type{{{false}}})
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto keep_dims = std::get<2>(t);
//         auto expected = std::get<3>(t);
//         auto result = all(ten,axes,keep_dims);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_math_all_initializer_list_axes_all_axes","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using bool_tensor_type = gtensor::tensor<bool>;
//     using gtensor::all;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(all(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,bool>);

//     REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},{1},false) == bool_tensor_type{{true,false,false},{true,false,true}});
//     REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},{2,1},false) == bool_tensor_type{false,false});
//     REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},{0,1},true) == bool_tensor_type{{{true,false,false}}});
//     //all axes
//     REQUIRE(all(tensor_type{{{1,5,6},{2,6,-1}},{{7,2,9},{1,11,3}}}) == tensor_type(true));
//     REQUIRE(all(tensor_type{{{1,5,6},{2,6,-1}},{{7,2,9},{1,11,3}}},false) == tensor_type(true));
//     REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}}) == tensor_type(false));
//     REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},false) == tensor_type(false));
//     REQUIRE(all(tensor_type{{{1,5,0},{2,0,-1}},{{7,0,9},{1,11,3}}},true) == tensor_type{{{false}}});
//     REQUIRE(all(tensor_type{{{1,5,6},{2,6,-1}},{{7,2,9},{1,11,3}}},true) == tensor_type{{{true}}});
// }

// TEST_CASE("test_math_all_nan_values","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using bool_tensor_type = gtensor::tensor<bool>;
//     using gtensor::all;
//     using helpers_for_testing::apply_by_element;

//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();

//     //0tensor,1axes,2keep_dims,3expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{nan,nan,nan},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{nan,0.0,nan},0,false,bool_tensor_type(false)),
//         std::make_tuple(tensor_type{nan,pos_inf,nan,neg_inf},0,false,bool_tensor_type(true))
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto keep_dims = std::get<2>(t);
//         auto expected = std::get<3>(t);
//         auto result = all(ten,axes,keep_dims);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// //any
// TEST_CASE("test_math_any","test_math")
// {
//     using value_type = int;
//     using tensor_type = gtensor::tensor<value_type>;
//     using bool_tensor_type = gtensor::tensor<bool>;
//     using gtensor::any;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(any(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,bool>);
//     REQUIRE(std::is_same_v<typename decltype(any(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,bool>);

//     //0tensor,1axes,2keep_dims,3expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,false,bool_tensor_type(false)),
//         std::make_tuple(tensor_type{},std::vector<int>{0},false,bool_tensor_type(false)),
//         std::make_tuple(tensor_type{},std::vector<int>{},false,bool_tensor_type(false)),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,bool_tensor_type{false,false}),
//         std::make_tuple(tensor_type{5},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{0},0,false,bool_tensor_type(false)),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{5,0,1,-1,0,4},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{0,0,0},0,false,bool_tensor_type(false)),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},0,false,bool_tensor_type{{true,false,true},{true,false,true}}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},1,false,bool_tensor_type{{true,false,true},{false,false,true}}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},2,false,bool_tensor_type{{true,true},{true,true}}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{0,1},false,bool_tensor_type{true,false,true}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{2,1},false,bool_tensor_type{true,true}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{},false,bool_tensor_type(true)),
//         //keep_dim true
//         std::make_tuple(tensor_type{},0,true,bool_tensor_type{false}),
//         std::make_tuple(tensor_type{},std::vector<int>{0},true,bool_tensor_type{false}),
//         std::make_tuple(tensor_type{},std::vector<int>{},true,bool_tensor_type{false}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,bool_tensor_type{{{false},{false}}}),
//         std::make_tuple(tensor_type{5},0,true,bool_tensor_type{true}),
//         std::make_tuple(tensor_type{0},0,true,bool_tensor_type{false}),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},0,true,bool_tensor_type{true}),
//         std::make_tuple(tensor_type{5,0,1,-1,0,4},0,true,bool_tensor_type{true}),
//         std::make_tuple(tensor_type{0,0,0},0,true,bool_tensor_type{false}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},0,true,bool_tensor_type{{{true,false,true},{true,false,true}}}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},1,true,bool_tensor_type{{{true,false,true}},{{false,false,true}}}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},2,true,bool_tensor_type{{{true},{true}},{{true},{true}}}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{0,1},true,bool_tensor_type{{{true,false,true}}}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{2,1},true,bool_tensor_type{{{true}},{{true}}}),
//         std::make_tuple(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},std::vector<int>{},true,bool_tensor_type{{{true}}})
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto keep_dims = std::get<2>(t);
//         auto expected = std::get<3>(t);
//         auto result = any(ten,axes,keep_dims);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_math_any_initializer_list_axes_any_axes","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using bool_tensor_type = gtensor::tensor<bool>;
//     using gtensor::any;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(any(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,bool>);

//     REQUIRE(any(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},{1},true) == bool_tensor_type{{{true,false,true}},{{false,false,true}}});
//     REQUIRE(any(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},{1}) == bool_tensor_type{{true,false,true},{false,false,true}});
//     REQUIRE(any(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}},{2,1}) == bool_tensor_type{true,true});
//     REQUIRE(any(tensor_type{{{1,0,0},{2,0,-1}},{{0,0,9},{0,0,3}}}) == tensor_type(true));
// }

// TEST_CASE("test_math_any_nan_values","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using bool_tensor_type = gtensor::tensor<bool>;
//     using gtensor::any;
//     using helpers_for_testing::apply_by_element;

//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();

//     //0tensor,1axes,2keep_dims,3expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{nan,nan,nan},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{0.0,0.0,nan},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{0.0,0.0,pos_inf},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{0.0,0.0,neg_inf},0,false,bool_tensor_type(true)),
//         std::make_tuple(tensor_type{nan,pos_inf,nan,neg_inf},0,false,bool_tensor_type(true))
//     );
//     auto test = [](const auto& t){
//         auto ten = std::get<0>(t);
//         auto axes = std::get<1>(t);
//         auto keep_dims = std::get<2>(t);
//         auto expected = std::get<3>(t);
//         auto result = any(ten,axes,keep_dims);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }


// //amin,nanmin
// TEMPLATE_TEST_CASE("test_math_amin_nanmin","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::amin;
//     using gtensor::nanmin;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type, value_type>);
//     REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type, value_type>);
//     REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type, value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type, value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type, value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type, value_type>);

//     //0tensor,1axes,2keep_dims,3initial,4expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,false,value_type{100},tensor_type(value_type{100})),
//         std::make_tuple(tensor_type{},std::vector<int>{0},false,value_type{100},tensor_type(value_type{100})),
//         std::make_tuple(tensor_type{},std::vector<int>{},false,value_type{100},tensor_type(value_type{100})),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,value_type{100},tensor_type{value_type{100},value_type{100}}),
//         std::make_tuple(tensor_type{5},0,false,value_type{100},tensor_type(5)),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,value_type{100},tensor_type(-1)),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},false,value_type{100},tensor_type(-1)),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},false,value_type{100},tensor_type(-1)),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},0,false,value_type{100},tensor_type{{1,4,3},{1,0,-1}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,false,value_type{100},tensor_type{{1,0,-1},{1,4,2}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,false,value_type{100},tensor_type{{1,-1},{4,1}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{0,1},false,value_type{100},tensor_type{1,0,-1}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},false,value_type{100},tensor_type{-1,1}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},false,value_type{100},tensor_type(-1)),
//         //keep_dim true
//         std::make_tuple(tensor_type{},0,true,value_type{100},tensor_type{value_type{100}}),
//         std::make_tuple(tensor_type{},std::vector<int>{0},true,value_type{100},tensor_type{value_type{100}}),
//         std::make_tuple(tensor_type{},std::vector<int>{},true,value_type{100},tensor_type{value_type{100}}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,value_type{100},tensor_type{{{value_type{100}},{value_type{100}}}}),
//         std::make_tuple(tensor_type{5},0,true,value_type{100},tensor_type{5}),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},0,true,value_type{100},tensor_type{-1}),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},true,value_type{100},tensor_type{-1}),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},true,value_type{100},tensor_type{-1}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},0,true,value_type{100},tensor_type{{{1,4,3},{1,0,-1}}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,true,value_type{100},tensor_type{{{1,0,-1}},{{1,4,2}}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,true,value_type{100},tensor_type{{{1},{-1}},{{4},{1}}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{0,1},true,value_type{100},tensor_type{{{1,0,-1}}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},true,value_type{100},tensor_type{{{-1}},{{1}}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},true,value_type{100},tensor_type{{{-1}}}),
//         //initial is min
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,value_type{-2},tensor_type(-2)),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,false,value_type{1},tensor_type{{1,-1},{1,1}})
//     );
//     SECTION("test_amin")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto initial = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto result = amin(ten,axes,keep_dims,initial);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nanmin")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto initial = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto result = nanmin(ten,axes,keep_dims,initial);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_math_amin_nanmin_initializer_list_axes_all_axes","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::amin;
//     using gtensor::nanmin;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(amin(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmin(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);

//     //amin
//     REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{1},false) == tensor_type{{1,0,-1},{1,4,2}});
//     REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1},false) == tensor_type{-1,1});
//     REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1}) == tensor_type{-1,1});
//     REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true) == tensor_type{{{1,0,-1}}});
//     REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true,0) == tensor_type{{{0,0,-1}}});
//     //all axes
//     REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false) == tensor_type(-1));
//     REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,-2) == tensor_type(-2));
//     REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,0) == tensor_type(-1));
//     REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}}) == tensor_type(-1));
//     REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true) == tensor_type{{{-1}}});

//     //nanmin
//     REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{1},false) == tensor_type{{1,0,-1},{1,4,2}});
//     REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1},false) == tensor_type{-1,1});
//     REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1}) == tensor_type{-1,1});
//     REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true) == tensor_type{{{1,0,-1}}});
//     REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true,0) == tensor_type{{{0,0,-1}}});
//     //all axes
//     REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false) == tensor_type(-1));
//     REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,-2) == tensor_type(-2));
//     REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}}) == tensor_type(-1));
//     REQUIRE(nanmin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true) == tensor_type{{{-1}}});
// }

// TEMPLATE_TEST_CASE("test_math_amin_nanmin_exception","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::reduce_exception;
//     using gtensor::amin;
//     using gtensor::nanmin;

//     //amin
//     REQUIRE_THROWS_AS(amin(tensor_type{}),reduce_exception);
//     REQUIRE_THROWS_AS(amin(tensor_type{}.reshape(0,2,3),{0,1}),reduce_exception);
//     REQUIRE_NOTHROW(amin(tensor_type{}.reshape(0,2,3),{1,2}));
//     //nanmin
//     REQUIRE_THROWS_AS(nanmin(tensor_type{}),reduce_exception);
//     REQUIRE_THROWS_AS(nanmin(tensor_type{}.reshape(0,2,3),{0,1}),reduce_exception);
//     REQUIRE_NOTHROW(nanmin(tensor_type{}.reshape(0,2,3),{1,2}));
// }

// TEST_CASE("test_math_amin_nanmin_nan_values","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::amin;
//     using gtensor::nanmin;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //amin
//         std::make_tuple(amin(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(0.5)),
//         std::make_tuple(amin(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(neg_inf)),
//         std::make_tuple(amin(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(amin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), tensor_type(nan)),
//         std::make_tuple(amin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), tensor_type{nan,nan,nan,-1.0,0.0,1.0}),
//         std::make_tuple(amin(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), tensor_type{nan,nan}),
//         std::make_tuple(amin(tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), tensor_type{nan,-1.0,3.0,nan}),
//         std::make_tuple(amin(tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), tensor_type{nan,0.1}),
//         //nanmin
//         std::make_tuple(nanmin(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(0.5)),
//         std::make_tuple(nanmin(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(neg_inf)),
//         std::make_tuple(nanmin(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
//         std::make_tuple(nanmin(tensor_type{{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(nanmin(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), tensor_type(0.1)),
//         std::make_tuple(nanmin(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},0), tensor_type{0.1,1.1,nan}),
//         std::make_tuple(nanmin(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},1), tensor_type{nan,1.1,0.1})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_equal(result,expected,true));
//     };
//     apply_by_element(test,test_data);
// }

// //amax,nanmax
// TEMPLATE_TEST_CASE("test_math_amax_nanmax","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::amax;
//     using gtensor::nanmax;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(amax(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(amax(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(amax(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmax(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmax(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmax(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);

//     //0tensor,1axes,2keep_dims,3initial,4expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,false,value_type{-100},tensor_type(value_type{-100})),
//         std::make_tuple(tensor_type{},std::vector<int>{0},false,value_type{-100},tensor_type(value_type{-100})),
//         std::make_tuple(tensor_type{},std::vector<int>{},false,value_type{-100},tensor_type(value_type{-100})),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,value_type{-100},tensor_type{value_type{-100},value_type{-100}}),
//         std::make_tuple(tensor_type{5},0,false,value_type{-100},tensor_type(5)),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,value_type{-100},tensor_type(5)),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},false,value_type{-100},tensor_type(5)),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},false,value_type{-100},tensor_type(5)),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},0,false,value_type{-100},tensor_type{{7,5,9},{2,11,2}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,false,value_type{-100},tensor_type{{2,5,3},{7,11,9}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,false,value_type{-100},tensor_type{{5,2},{9,11}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{0,1},false,value_type{-100},tensor_type{7,11,9}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},false,value_type{-100},tensor_type{5,11}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},false,value_type{-100},tensor_type(11)),
//         //keep_dim true
//         std::make_tuple(tensor_type{},0,true,value_type{-100},tensor_type{value_type{-100}}),
//         std::make_tuple(tensor_type{},std::vector<int>{0},true,value_type{-100},tensor_type{value_type{-100}}),
//         std::make_tuple(tensor_type{},std::vector<int>{},true,value_type{-100},tensor_type{value_type{-100}}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,value_type{-100},tensor_type{{{value_type{-100}},{value_type{-100}}}}),
//         std::make_tuple(tensor_type{5},0,true,value_type{-100},tensor_type{5}),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},0,true,value_type{-100},tensor_type{5}),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},true,value_type{-100},tensor_type{5}),
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},true,value_type{-100},tensor_type{5}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},0,true,value_type{-100},tensor_type{{{7,5,9},{2,11,2}}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,true,value_type{-100},tensor_type{{{2,5,3}},{{7,11,9}}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,true,value_type{-100},tensor_type{{{5},{2}},{{9},{11}}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{0,1},true,value_type{-100},tensor_type{{{7,11,9}}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},true,value_type{-100},tensor_type{{{5}},{{11}}}),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},true,value_type{-100},tensor_type{{{11}}}),
//         //initial is max
//         std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,value_type{6},tensor_type(6)),
//         std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,false,value_type{3},tensor_type{{5,3},{9,11}})
//     );
//     SECTION("test_amax")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto initial = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto result = amax(ten,axes,keep_dims,initial);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nanmax")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto initial = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto result = nanmax(ten,axes,keep_dims,initial);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_math_amax_nanmax_initializer_list_axes_all_axes","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::amax;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(amax(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(amax(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmax(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanmax(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);

//     //amax
//     REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{1},false) == tensor_type{{2,5,3},{7,11,9}});
//     REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1},false) == tensor_type{5,11});
//     REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1}) == tensor_type{5,11});
//     REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true) == tensor_type{{{7,11,9}}});
//     REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true,8) == tensor_type{{{8,11,9}}});
//     //all axes
//     REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false) == tensor_type(11));
//     REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}}) == tensor_type(11));
//     REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,12) == tensor_type(12));
//     REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,0) == tensor_type(11));
//     REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true) == tensor_type{{{11}}});

//     //nanmax
//     REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{1},false) == tensor_type{{2,5,3},{7,11,9}});
//     REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1},false) == tensor_type{5,11});
//     REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1}) == tensor_type{5,11});
//     REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true) == tensor_type{{{7,11,9}}});
//     REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true,8) == tensor_type{{{8,11,9}}});
//     //all axes
//     REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false) == tensor_type(11));
//     REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}}) == tensor_type(11));
//     REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,12) == tensor_type(12));
//     REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false,0) == tensor_type(11));
//     REQUIRE(nanmax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true) == tensor_type{{{11}}});
// }

// TEMPLATE_TEST_CASE("test_math_amax_nanmax_exception","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::reduce_exception;
//     using gtensor::amax;
//     using gtensor::nanmax;

//     //amax
//     REQUIRE_THROWS_AS(amax(tensor_type{}),reduce_exception);
//     REQUIRE_THROWS_AS(amax(tensor_type{}.reshape(0,2,3),{0,1}),reduce_exception);
//     REQUIRE_NOTHROW(amax(tensor_type{}.reshape(0,2,3),{1,2}));
//     //nanmax
//     REQUIRE_THROWS_AS(nanmax(tensor_type{}),reduce_exception);
//     REQUIRE_THROWS_AS(nanmax(tensor_type{}.reshape(0,2,3),{0,1}),reduce_exception);
//     REQUIRE_NOTHROW(nanmax(tensor_type{}.reshape(0,2,3),{1,2}));
// }

// TEST_CASE("test_math_amax_nanmax_nan_values","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::amax;
//     using gtensor::nanmax;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //amax
//         std::make_tuple(amax(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(pos_inf)),
//         std::make_tuple(amax(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(3.0)),
//         std::make_tuple(amax(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(amax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}}), tensor_type(nan)),
//         std::make_tuple(amax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},0), tensor_type{nan,nan,nan,4.0,1.0,3.0}),
//         std::make_tuple(amax(tensor_type{{nan,nan,nan,4.0,0.0,3.0},{2.0,0.0,nan,-1.0,1.0,1.0}},1), tensor_type{nan,nan}),
//         std::make_tuple(amax(tensor_type{{4.0,-1.0,3.0,nan},{nan,0.1,5.0,1.0}},0), tensor_type{nan,0.1,5.0,nan}),
//         std::make_tuple(amax(tensor_type{{4.0,-1.0,3.0,nan},{2.0,0.1,5.0,1.0}},1), tensor_type{nan,5.0}),
//         //nanmax
//         std::make_tuple(nanmax(tensor_type{1.0,0.5,2.0,pos_inf,3.0}), tensor_type(pos_inf)),
//         std::make_tuple(nanmax(tensor_type{1.0,0.5,2.0,neg_inf,3.0}), tensor_type(3.0)),
//         std::make_tuple(nanmax(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(pos_inf)),
//         std::make_tuple(nanmax(tensor_type{{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(nanmax(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}}), tensor_type(2.0)),
//         std::make_tuple(nanmax(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},0), tensor_type{0.1,2.0,nan}),
//         std::make_tuple(nanmax(tensor_type{{nan,nan,nan},{nan,1.1,nan},{0.1,2.0,nan}},1), tensor_type{nan,1.1,2.0})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_equal(result,expected,true));
//     };
//     apply_by_element(test,test_data);
// }

// //sum,nansum
// TEMPLATE_TEST_CASE("test_math_sum_nansum","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::sum;
//     using gtensor::nansum;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(sum(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(sum(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(sum(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nansum(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nansum(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nansum(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);

//     //0tensor,1axes,2keep_dims,3initial,4expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,false,value_type{0},tensor_type(value_type{0})),
//         std::make_tuple(tensor_type{},std::vector<int>{0},false,value_type{0},tensor_type(value_type{0})),
//         std::make_tuple(tensor_type{},std::vector<int>{},false,value_type{0},tensor_type(value_type{0})),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,value_type{0},tensor_type{value_type{0},value_type{0}}),
//         std::make_tuple(tensor_type{5},0,false,value_type{0},tensor_type(5)),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,false,value_type{0},tensor_type(15)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,value_type{0},tensor_type(15)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,value_type{0},tensor_type(15)),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,value_type{0},tensor_type{{8,10,12},{14,16,18}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,value_type{0},tensor_type{{5,7,9},{17,19,21}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,value_type{0},tensor_type{{6,15},{24,33}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,value_type{0},tensor_type{22,26,30}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,value_type{0},tensor_type{21,57}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,value_type{0},tensor_type(78)),
//         //keep_dim true
//         std::make_tuple(tensor_type{},0,true,value_type{0},tensor_type{value_type{0}}),
//         std::make_tuple(tensor_type{},std::vector<int>{0},true,value_type{0},tensor_type{value_type{0}}),
//         std::make_tuple(tensor_type{},std::vector<int>{},true,value_type{0},tensor_type{value_type{0}}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,value_type{0},tensor_type{{{value_type{0}},{value_type{0}}}}),
//         std::make_tuple(tensor_type{5},0,true,value_type{0},tensor_type{5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,true,value_type{0},tensor_type{15}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,value_type{0},tensor_type{15}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,value_type{0},tensor_type{15}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,value_type{0},tensor_type{{{8,10,12},{14,16,18}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,value_type{0},tensor_type{{{5,7,9}},{{17,19,21}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,value_type{0},tensor_type{{{6},{15}},{{24},{33}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,value_type{0},tensor_type{{{22,26,30}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,value_type{0},tensor_type{{{21}},{{57}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,value_type{0},tensor_type{{{78}}}),
//         //non zero initial
//         std::make_tuple(tensor_type{},0,false,value_type{-1},tensor_type(-1)),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),0,false,value_type{-1},tensor_type{{-1,-1,-1},{-1,-1,-1}}),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,false,value_type{-1},tensor_type(14)),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,value_type{1},tensor_type{23,27,31})
//     );
//     SECTION("test_sum")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto initial = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto result = sum(ten,axes,keep_dims,initial);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nansum")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto initial = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto result = nansum(ten,axes,keep_dims,initial);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_math_sum_nansum_initializer_list_axes_all_axes","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::sum;
//     using gtensor::nansum;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(sum(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(sum(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nansum(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nansum(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
//     //sum
//     REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{8,10,12},{14,16,18}});
//     REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,1) == tensor_type{23,27,31});
//     REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{22,26,30});
//     REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{22,26,30});
//     REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{21}},{{57}}});
//     //all axes
//     REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,1) == tensor_type(79));
//     REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(78));
//     REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(78));
//     REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{78}}});
//     //nansum
//     REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{8,10,12},{14,16,18}});
//     REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,1) == tensor_type{23,27,31});
//     REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{22,26,30});
//     REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{22,26,30});
//     REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{21}},{{57}}});
//     //all axes
//     REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,1) == tensor_type(79));
//     REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(78));
//     REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(78));
//     REQUIRE(nansum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{78}}});
// }

// TEST_CASE("test_math_sum_nansum_nan_values","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::sum;
//     using gtensor::nansum;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //sum
//         std::make_tuple(sum(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
//         std::make_tuple(sum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
//         std::make_tuple(sum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(sum(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(sum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(sum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
//         std::make_tuple(sum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
//         std::make_tuple(sum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
//         std::make_tuple(sum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,10.0}),
//         std::make_tuple(sum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
//         //nansum
//         std::make_tuple(nansum(tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
//         std::make_tuple(nansum(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
//         std::make_tuple(nansum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nansum(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nansum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(0.0)),
//         std::make_tuple(nansum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{0.0,0.0,0.0}),
//         std::make_tuple(nansum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{0.0,0.0,0.0}),
//         std::make_tuple(nansum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(16.5)),
//         std::make_tuple(nansum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{1.0,5.5,0.0,10.0}),
//         std::make_tuple(nansum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{1.0,3.5,5.5,6.5})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_equal(result,expected,true));
//     };
//     apply_by_element(test,test_data);
// }

// //prod,nanprod
// TEMPLATE_TEST_CASE("test_math_prod_nanprod","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::prod;
//     using gtensor::nanprod;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(prod(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(prod(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(prod(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanprod(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanprod(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanprod(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>()))::value_type,value_type>);

//     //0tensor,1axes,2keep_dims,3initial,4expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,false,value_type{1},tensor_type(value_type{1})),
//         std::make_tuple(tensor_type{},std::vector<int>{0},false,value_type{1},tensor_type(value_type{1})),
//         std::make_tuple(tensor_type{},std::vector<int>{},false,value_type{1},tensor_type(value_type{1})),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,value_type{1},tensor_type{value_type{1},value_type{1}}),
//         std::make_tuple(tensor_type{5},0,false,value_type{1},tensor_type(5)),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,false,value_type{1},tensor_type(120)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,value_type{1},tensor_type(120)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,value_type{1},tensor_type(120)),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,value_type{1},tensor_type{{7,16,27},{40,55,72}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,value_type{1},tensor_type{{4,10,18},{70,88,108}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,value_type{1},tensor_type{{6,120},{504,1320}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,value_type{1},tensor_type{280,880,1944}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,value_type{1},tensor_type{720,665280}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,value_type{1},tensor_type(479001600)),
//         //keep_dim true
//         std::make_tuple(tensor_type{},0,true,value_type{1},tensor_type{value_type{1}}),
//         std::make_tuple(tensor_type{},std::vector<int>{0},true,value_type{1},tensor_type{value_type{1}}),
//         std::make_tuple(tensor_type{},std::vector<int>{},true,value_type{1},tensor_type{value_type{1}}),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,value_type{1},tensor_type{{{value_type{1}},{value_type{1}}}}),
//         std::make_tuple(tensor_type{5},0,true,value_type{1},tensor_type{5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,true,value_type{1},tensor_type{120}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,value_type{1},tensor_type{120}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,value_type{1},tensor_type{120}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,value_type{1},tensor_type{{{7,16,27},{40,55,72}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,value_type{1},tensor_type{{{4,10,18}},{{70,88,108}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,value_type{1},tensor_type{{{6},{120}},{{504},{1320}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,value_type{1},tensor_type{{{280,880,1944}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,value_type{1},tensor_type{{{720}},{{665280}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,value_type{1},tensor_type{{{479001600}}}),
//         //not one initial
//         std::make_tuple(tensor_type{},0,false,value_type{-2},tensor_type(-2)),
//         std::make_tuple(tensor_type{}.reshape(0,2,3),0,false,value_type{-2},tensor_type{{-2,-2,-2},{-2,-2,-2}}),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,false,value_type{-2},tensor_type(-240)),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,value_type{-2},tensor_type{-560,-1760,-3888})
//     );
//     SECTION("test_prod")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto initial = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto result = prod(ten,axes,keep_dims,initial);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nanprod")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto keep_dims = std::get<2>(t);
//             auto initial = std::get<3>(t);
//             auto expected = std::get<4>(t);
//             auto result = nanprod(ten,axes,keep_dims,initial);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_math_prod_nanprod_initializer_list_axes_all_axes","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::prod;
//     using gtensor::nanprod;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(prod(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(prod(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanprod(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>(),std::declval<value_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nanprod(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>()))::value_type,value_type>);
//     //prod
//     REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{7,16,27},{40,55,72}});
//     REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,-2) == tensor_type{-560,-1760,-3888});
//     REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{280,880,1944});
//     REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{280,880,1944});
//     REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{720}},{{665280}}});
//     //all axes
//     REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,-2) == tensor_type(-958003200));
//     REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(479001600));
//     REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(479001600));
//     REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{479001600}}});
//     //nanprod
//     REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{7,16,27},{40,55,72}});
//     REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false,-2) == tensor_type{-560,-1760,-3888});
//     REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{280,880,1944});
//     REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1}) == tensor_type{280,880,1944});
//     REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{720}},{{665280}}});
//     //all axes
//     REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false,-2) == tensor_type(-958003200));
//     REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(479001600));
//     REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}) == tensor_type(479001600));
//     REQUIRE(nanprod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{479001600}}});
// }

// TEST_CASE("test_math_prod_nanprod_nan_values","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::prod;
//     using gtensor::nanprod;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //prod
//         std::make_tuple(prod(tensor_type{1.0,0.5,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
//         std::make_tuple(prod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
//         std::make_tuple(prod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
//         std::make_tuple(prod(tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(prod(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(prod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(nan)),
//         std::make_tuple(prod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{nan,nan,nan}),
//         std::make_tuple(prod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{nan,nan,nan}),
//         std::make_tuple(prod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(nan)),
//         std::make_tuple(prod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{nan,nan,nan,24.0}),
//         std::make_tuple(prod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{nan,nan,nan,nan}),
//         //nanprod
//         std::make_tuple(nanprod(tensor_type{1.0,nan,2.0,4.0,3.0,pos_inf}), tensor_type(pos_inf)),
//         std::make_tuple(nanprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type(neg_inf)),
//         std::make_tuple(nanprod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
//         std::make_tuple(nanprod(tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf}), tensor_type(nan)),
//         std::make_tuple(nanprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type(neg_inf)),
//         std::make_tuple(nanprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type(1.0)),
//         std::make_tuple(nanprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{1.0,1.0,1.0}),
//         std::make_tuple(nanprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{1.0,1.0,1.0}),
//         std::make_tuple(nanprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}}), tensor_type(36.0)),
//         std::make_tuple(nanprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},0), tensor_type{0.25,6.0,1.0,24.0}),
//         std::make_tuple(nanprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,2.0,nan,4.0}},1), tensor_type{1.0,3.0,3.0,4.0})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_equal(result,expected,true));
//     };
//     apply_by_element(test,test_data);
// }

// //cumsum, nancumsum
// TEMPLATE_TEST_CASE("test_math_cumsum_nancumsum","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::cumsum;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(cumsum(std::declval<tensor_type>(),std::declval<int>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nancumsum(std::declval<tensor_type>(),std::declval<int>()))::value_type,value_type>);

//     //0tensor,1axes,2expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,tensor_type{}),
//         std::make_tuple(tensor_type{5},0,tensor_type{5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,3,6,10,15}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,tensor_type{{{1,2,3},{4,5,6}},{{8,10,12},{14,16,18}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,tensor_type{{{1,2,3},{5,7,9}},{{7,8,9},{17,19,21}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,tensor_type{{{1,3,6},{4,9,15}},{{7,15,24},{10,21,33}}})
//     );
//     SECTION("test_cumsum")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto result = cumsum(ten,axes);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nancumsum")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto result = nancumsum(ten,axes);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_math_cumsum_nancumsum_all_axes","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::cumsum;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(cumsum(std::declval<tensor_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nancumsum(std::declval<tensor_type>()))::value_type,value_type>);

//     //0tensor,1expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},tensor_type{}),
//         std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{1,3,6,10,15}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},tensor_type{1,3,6,10,15,21,28,36,45,55,66,78})
//     );
//     SECTION("test_cumsum")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto expected = std::get<1>(t);
//             auto result = cumsum(ten);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nancumsum")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto expected = std::get<1>(t);
//             auto result = nancumsum(ten);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEST_CASE("test_math_cumsum_nancumsum_nan_values","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::cumsum;
//     using gtensor::nancumsum;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //cumsum
//         std::make_tuple(cumsum(tensor_type{1.0,0.5,2.0,pos_inf,3.0,4.0}), tensor_type{1.0,1.5,3.5,pos_inf,pos_inf,pos_inf}),
//         std::make_tuple(cumsum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type{1.0,1.5,3.5,neg_inf,neg_inf,neg_inf}),
//         std::make_tuple(cumsum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,1.5,3.5,neg_inf,neg_inf,nan}),
//         std::make_tuple(cumsum(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,nan,nan,nan,nan,nan}),
//         std::make_tuple(cumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type{nan,nan,nan,nan,nan,nan,nan,nan,nan}),
//         std::make_tuple(cumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
//         std::make_tuple(cumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
//         std::make_tuple(cumsum(tensor_type{{0.5,1.0,0.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{0.5,1.5,1.5,0.0,3.0,nan,nan,nan,nan}),
//         std::make_tuple(cumsum(tensor_type{{nan,1.0,0.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{nan,nan,nan,nan,nan,nan,nan,nan,nan}),
//         std::make_tuple(
//             cumsum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
//             tensor_type{{nan,nan,nan,1.0},{nan,nan,nan,3.0},{nan,nan,nan,6.0},{nan,nan,nan,10.0}}
//         ),
//         std::make_tuple(
//             cumsum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
//             tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{0.5,2.5,nan,nan},{0.5,3.5,nan,nan}}
//         ),
//         //nancumsum
//         std::make_tuple(nancumsum(tensor_type{1.0,0.5,2.0,pos_inf,3.0,4.0}), tensor_type{1.0,1.5,3.5,pos_inf,pos_inf,pos_inf}),
//         std::make_tuple(nancumsum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type{1.0,1.5,3.5,neg_inf,neg_inf,neg_inf}),
//         std::make_tuple(nancumsum(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,1.5,3.5,neg_inf,neg_inf,nan}),
//         std::make_tuple(nancumsum(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,1.0,3.0,neg_inf,neg_inf,nan}),
//         std::make_tuple(nancumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type{0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}),
//         std::make_tuple(nancumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}}),
//         std::make_tuple(nancumsum(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}}),
//         std::make_tuple(nancumsum(tensor_type{{0.5,1.0,0.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{0.5,1.5,1.5,0.0,3.0,3.0,3.5,5.5,5.5}),
//         std::make_tuple(nancumsum(tensor_type{{nan,1.0,0.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{0.0,1.0,1.0,-0.5,2.5,2.5,3.0,5.0,5.0}),
//         std::make_tuple(
//             nancumsum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
//             tensor_type{{0.0,0.0,0.0,1.0},{0.0,1.5,0.0,3.0},{0.5,3.5,0.0,6.0},{1.0,6.5,0.0,10.0}}
//         ),
//         std::make_tuple(
//             nancumsum(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
//             tensor_type{{0.0,0.0,0.0,1.0},{0.0,1.5,1.5,3.50},{0.5,2.5,2.5,5.5},{0.5,3.5,3.5,7.5}}
//         )
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_equal(result,expected,true));
//     };
//     apply_by_element(test,test_data);
// }

// //cumprod,nancumprod
// TEMPLATE_TEST_CASE("test_math_cumprod_nancumprod","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::cumprod;
//     using gtensor::nancumprod;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(cumprod(std::declval<tensor_type>(),std::declval<int>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nancumprod(std::declval<tensor_type>(),std::declval<int>()))::value_type,value_type>);

//     //0tensor,1axes,2expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},0,tensor_type{}),
//         std::make_tuple(tensor_type{5},0,tensor_type{5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,2,6,24,120}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,tensor_type{{{1,2,3},{4,5,6}},{{7,16,27},{40,55,72}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,tensor_type{{{1,2,3},{4,10,18}},{{7,8,9},{70,88,108}}}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,tensor_type{{{1,2,6},{4,20,120}},{{7,56,504},{10,110,1320}}})
//     );
//     SECTION("test_cumprod")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto result = cumprod(ten,axes);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nancumprod")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto axes = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto result = nancumprod(ten,axes);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_math_cumprod_nancumprod_all_axes","test_math",
//     double,
//     int
// )
// {
//     using value_type = TestType;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::cumprod;
//     using gtensor::nancumprod;
//     using helpers_for_testing::apply_by_element;

//     REQUIRE(std::is_same_v<typename decltype(cumprod(std::declval<tensor_type>()))::value_type,value_type>);
//     REQUIRE(std::is_same_v<typename decltype(nancumprod(std::declval<tensor_type>()))::value_type,value_type>);

//     //0tensor,1expected
//     auto test_data = std::make_tuple(
//         //keep_dim false
//         std::make_tuple(tensor_type{},tensor_type{}),
//         std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{1,2,6,24,120}),
//         std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{1,2,3},{0,4,5}}},tensor_type{1,2,6,24,120,720,720,1440,4320,0,0,0})
//     );
//     SECTION("test_cumprod")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto expected = std::get<1>(t);
//             auto result = cumprod(ten);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_nancumprod")
//     {
//         auto test = [](const auto& t){
//             auto ten = std::get<0>(t);
//             auto expected = std::get<1>(t);
//             auto result = nancumprod(ten);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEST_CASE("test_math_cumprod_nancumprod_nan_values","test_math")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using gtensor::cumprod;
//     using gtensor::nancumprod;
//     using gtensor::tensor_equal;
//     using helpers_for_testing::apply_by_element;
//     static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
//     static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
//     static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         //cumprod
//         std::make_tuple(cumprod(tensor_type{1.0,0.5,2.0,pos_inf,4.0,3.0}), tensor_type{1.0,0.5,1.0,pos_inf,pos_inf,pos_inf}),
//         std::make_tuple(cumprod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,4.0}), tensor_type{1.0,0.5,1.0,neg_inf,neg_inf,neg_inf}),
//         std::make_tuple(cumprod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,0.5,1.0,neg_inf,neg_inf,neg_inf}),
//         std::make_tuple(cumprod(tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,0.0,0.0,nan,nan,nan}),
//         std::make_tuple(cumprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,nan,nan,nan,nan,nan}),
//         std::make_tuple(cumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type{nan,nan,nan,nan,nan,nan,nan,nan,nan}),
//         std::make_tuple(cumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
//         std::make_tuple(cumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}),
//         std::make_tuple(cumprod(tensor_type{{0.5,1.0,2.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{0.5,0.5,1.0,-1.5,-4.5,nan,nan,nan,nan}),
//         std::make_tuple(cumprod(tensor_type{{nan,1.0,2.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{nan,nan,nan,nan,nan,nan,nan,nan,nan}),
//         std::make_tuple(
//             cumprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
//             tensor_type{{nan,nan,nan,1.0},{nan,nan,nan,2.0},{nan,nan,nan,6.0},{nan,nan,nan,24.0}}
//         ),
//         std::make_tuple(
//             cumprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
//             tensor_type{{nan,nan,nan,nan},{nan,nan,nan,nan},{0.5,1.0,nan,nan},{0.5,1.5,nan,nan}}
//         ),
//         //nancumprod
//         std::make_tuple(nancumprod(tensor_type{1.0,nan,2.0,pos_inf,4.0,3.0}), tensor_type{1.0,1.0,2.0,pos_inf,pos_inf,pos_inf}),
//         std::make_tuple(nancumprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,4.0}), tensor_type{1.0,1.0,2.0,neg_inf,neg_inf,neg_inf}),
//         std::make_tuple(nancumprod(tensor_type{1.0,0.5,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,0.5,1.0,neg_inf,neg_inf,neg_inf}),
//         std::make_tuple(nancumprod(tensor_type{1.0,0.0,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,0.0,0.0,nan,nan,nan}),
//         std::make_tuple(nancumprod(tensor_type{1.0,nan,2.0,neg_inf,3.0,pos_inf}), tensor_type{1.0,1.0,2.0,neg_inf,neg_inf,neg_inf}),
//         std::make_tuple(nancumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}}), tensor_type{1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0}),
//         std::make_tuple(nancumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},0), tensor_type{{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0}}),
//         std::make_tuple(nancumprod(tensor_type{{nan,nan,nan},{nan,nan,nan},{nan,nan,nan}},1), tensor_type{{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0}}),
//         std::make_tuple(nancumprod(tensor_type{{0.5,1.0,2.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{0.5,0.5,1.0,-1.5,-4.5,-4.5,-2.25,-4.5,-4.5}),
//         std::make_tuple(nancumprod(tensor_type{{nan,1.0,2.0},{-1.5,3.0,nan},{0.5,2.0,nan}}), tensor_type{1.0,1.0,2.0,-3.0,-9.0,-9.0,-4.5,-9.0,-9.0}),
//         std::make_tuple(
//             nancumprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},0),
//             tensor_type{{1.0,1.0,1.0,1.0},{1.0,1.5,1.0,2.0},{0.5,3.0,1.0,6.0},{0.25,9.0,1.0,24.0}}
//         ),
//         std::make_tuple(
//             nancumprod(tensor_type{{nan,nan,nan,1.0},{nan,1.5,nan,2.0},{0.5,2.0,nan,3.0},{0.5,3.0,nan,4.0}},1),
//             tensor_type{{1.0,1.0,1.0,1.0},{1.0,1.5,1.5,3.0},{0.5,1.0,1.0,3.0},{0.5,1.5,1.5,6.0}}
//         )
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(tensor_equal(result,expected,true));
//     };
//     apply_by_element(test,test_data);
// }

// //mean,nanmean
// TEMPLATE_TEST_CASE("test_math_mean_nanmean_floating_point_values","test_math",
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

// TEMPLATE_TEST_CASE("test_math_mean_nanmean_initializer_list_axes_all_axes","test_math",
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

// TEST_CASE("test_math_mean_nanmean_nan_values","test_math")
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
// TEMPLATE_TEST_CASE("test_math_var_nanvar_floating_point_values","test_math",
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

// TEMPLATE_TEST_CASE("test_math_var_nanvar_initializer_list_axes_all_axes","test_math",
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

// TEST_CASE("test_math_var_nanvar_nan_values","test_math")
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
// TEMPLATE_TEST_CASE("test_math_std_nanstd_floating_point_values","test_math",
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

// TEMPLATE_TEST_CASE("test_math_std_nanstd_initializer_list_axes_all_axes","test_math",
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

// TEST_CASE("test_math_std_nanstd_nan_values","test_math")
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
// TEMPLATE_TEST_CASE("test_math_median_nanmedian_floating_point_values","test_math",
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

// TEMPLATE_TEST_CASE("test_math_median_nanmedian_initializer_list_axes_all_axes","test_math",
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

// TEST_CASE("test_math_median_nanmedian_nan_values","test_math")
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

//diff
TEST_CASE("test_math_diff","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::diff;
    using helpers_for_testing::apply_by_element;

    //0tensor,1n,2axis,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8},0,0,tensor_type{1,3,2,5,7,4,6,7,8}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8},1,0,tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8},2,0,tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8},3,0,tensor_type{7,-5,-4,10,-6,1}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},0,0,tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},1,0,tensor_type{{2,4,2,-2,-2},{-6,0,0,1,0},{6,-3,-1,0,2},{-2,1,2,-1,-4}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},2,0,tensor_type{{-8,-4,-2,3,2},{12,-3,-1,-1,2},{-8,4,3,-1,-6}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},3,0,tensor_type{{20,1,1,-4,0},{-20,7,4,0,-8}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},1,1,tensor_type{{-3,1,2,1},{-1,-1,-2,1},{5,-1,-1,0},{-4,1,0,2},{-1,2,-3,-1}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},2,1,tensor_type{{4,1,-1},{0,-1,3},{-6,0,1},{5,-1,2},{3,-5,2}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},3,1,tensor_type{{-3,-2},{-1,4},{6,1},{-6,3},{-8,7}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto n = std::get<1>(t);
        auto axis = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = diff(ten,n,axis);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_math_diff2","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::diff2;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8},0,tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},0,tensor_type{{-8,-4,-2,3,2},{12,-3,-1,-1,2},{-8,4,3,-1,-6}}),
        std::make_tuple(tensor_type{{2,-1,0,2,3},{4,3,2,0,1},{-2,3,2,1,1},{4,0,1,1,3},{2,1,3,0,-1}},1,tensor_type{{4,1,-1},{0,-1,3},{-6,0,1},{5,-1,2},{3,-5,2}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = diff2(ten,axis);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//average
TEMPLATE_TEST_CASE("test_math_average","test_math",
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
        typename decltype(average(std::declval<tensor_type>(),std::declval<std::vector<dim_type>>(),std::declval<std::vector<value_type>>(),std::declval<bool>()))::value_type,
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
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto weights = std::get<3>(t);
        auto expected = std::get<4>(t);
        auto result = average(ten,axes,weights,keep_dims);
        REQUIRE(tensor_close(result,expected,1E-2,1E-2));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_math_average_initializer_list_axes_all_axes","test_math",
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

TEMPLATE_TEST_CASE("test_math_average_exception","test_math",
    double,
    int
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::average;
    using gtensor::reduce_exception;
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
        REQUIRE_THROWS_AS(average(ten,axes,weights,keep_dims), reduce_exception);
    };
    apply_by_element(test,test_data);
}