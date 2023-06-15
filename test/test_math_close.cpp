#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "tensor.hpp"

//test tensor fuzzy equality
TEST_CASE("test_tensor_close","test_math")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type inf = std::numeric_limits<value_type>::infinity();
    //0ten_0,1ten_1,2relative_tolerance,3absolute_tolerance,4equal_nan,5expected
    auto test_data = std::make_tuple(
        //nan_equal false
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(inf),tensor_type(inf),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(-inf),tensor_type(-inf),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{},tensor_type{},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2+1E-11,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{{1.1},{2.2},{3.3},{inf}},tensor_type{{1.1},{2.2},{3.3},{inf}},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-9),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(inf),tensor_type(-inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(inf),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(-inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(nan),tensor_type(nan),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(nan),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(nan),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(0,1),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2-1E-9,3.3},1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{1.1,2.2,3.3}},1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,nan,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,false,false),
        //nan_equal true
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(inf),tensor_type(inf),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(-inf),tensor_type(-inf),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{},tensor_type{},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{1.1,2.2+1E-11,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{{1.1},{2.2},{3.3},{inf}},tensor_type{{1.1},{2.2},{3.3},{inf}},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-9),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(inf),tensor_type(-inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(-inf),tensor_type(inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(inf),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(-inf),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(-inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(nan),tensor_type(nan),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(nan),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(nan),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(0,1),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2-1E-9,3.3},1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{1.1,2.2,3.3}},1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,nan,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,true,true),
        //vary tolerance
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-12,1E-12,false,false),
        //near zero: absolute tolerance plays
        std::make_tuple(tensor_type(1E-15),tensor_type(1E-15+1E-30),1E-16,1E-10,false,true),
        std::make_tuple(tensor_type(1E-15),tensor_type(1E-15+1E-30),1E-16,1E-32,false,false),
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-14,1E-10,false,true),
        //big: relative tolerance plays
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-15,1E-10,false,true),
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-16,1E-10,false,false),
        //not floating point type
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,true),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,6},1E-10,1E-10,false,false),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,true),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,6},1E-10,1E-10,true,false),
        //mixed types
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<double>{1.0,2.0,3.0,4.0,5.0},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,false),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<double>{1.0,2.0,3.0,4.0,5.0},1E-10,1E-10,true,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,false),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,false),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,false),
        std::make_tuple(tensor<double>{1.0,2.0,3.0,4.0,5.0},tensor<float>{1.0,2.0+1E-11,3.0,4.0,5.0},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0,5.0},tensor<float>{1.0,nan,3.0,4.0,5.0},1E-10,1E-10,false,false),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0,5.0},tensor<float>{1.0,nan,3.0,4.0,5.0},1E-10,1E-10,true,true)
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto relative_tolerance = std::get<2>(t);
        auto absolute_tolerance = std::get<3>(t);
        auto equal_nan = std::get<4>(t);
        auto expected = std::get<5>(t);

        auto result = tensor_close(ten_0,ten_1,relative_tolerance,absolute_tolerance,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_close_default_precision","test_math")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type e = std::numeric_limits<value_type>::epsilon();
    //0ten_0,1ten_1,2equal_nan,3expected
    auto test_data = std::make_tuple(
        //nan_equal false
        std::make_tuple(tensor_type(0.0),tensor_type(0.0),false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),false,true),
        std::make_tuple(tensor_type(1.0+e),tensor_type(1.0),false,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1},false,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1+1E-10},false,false),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,nan},tensor_type{-1.0-e,0.0,0.0,1.0+e,nan},false,false),
        std::make_tuple(tensor_type{{-1.0,0.0,0.0+e},{1.0+e,2.1,3.3}},tensor_type{{-1.0-e,0.0},{0.0,1.0+e},{2.1,3.3}},false,false),
        //nan_equal true
        std::make_tuple(tensor_type(0.0),tensor_type(0.0),true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),true,true),
        std::make_tuple(tensor_type(1.0+e),tensor_type(1.0),true,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1},true,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1+1E-10},true,false),
        std::make_tuple(tensor_type{nan,-1.0,0.0,0.0+e,1.0+e,nan},tensor_type{nan,-1.0-e,0.0,0.0,1.0+e,nan},true,true),
        std::make_tuple(tensor_type{{nan,-1.0,0.0},{0.0+e,1.0+e,nan}},tensor_type{{nan,-1.0-e},{0.0,0.0},{1.0+e,nan}},true,false)
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto equal_nan = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = tensor_close(ten_0,ten_1,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_allclose","test_math")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::allclose;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type inf = std::numeric_limits<value_type>::infinity();
    //0ten_0,1ten_1,2relative_tolerance,3absolute_tolerance,4equal_nan,5expected
    auto test_data = std::make_tuple(
        //nan_equal false
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(inf),tensor_type(inf),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(-inf),tensor_type(-inf),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{},tensor_type{},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2+1E-11,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{{1.1},{2.2},{3.3},{inf}},tensor_type{{1.1},{2.2},{3.3},{inf}},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-9),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(inf),tensor_type(-inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(inf),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(-inf),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(-inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(inf),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(nan),tensor_type(nan),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(nan),tensor_type(1.0),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type(1.0),tensor_type(nan),1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2-1E-9,3.3},1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,false,false),
        std::make_tuple(tensor_type{1.1,nan,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,false,false),
        //nan_equal true
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(inf),tensor_type(inf),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(-inf),tensor_type(-inf),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{},tensor_type{},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{1.1,2.2+1E-11,3.3},tensor_type{1.1,2.2,3.3},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{{1.1},{2.2},{3.3},{inf}},tensor_type{{1.1},{2.2},{3.3},{inf}},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-9),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(inf),tensor_type(-inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(-inf),tensor_type(inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(inf),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(-inf),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(-inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(inf),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(nan),tensor_type(nan),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type(nan),tensor_type(1.0),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type(1.0),tensor_type(nan),1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,2.2-1E-9,3.3},1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,true,false),
        std::make_tuple(tensor_type{1.1,nan,3.3},tensor_type{1.1,nan,3.3},1E-10,1E-10,true,true),
        //vary tolerance
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0+1E-11),1E-12,1E-12,false,false),
        //near zero: absolute tolerance plays
        std::make_tuple(tensor_type(1E-15),tensor_type(1E-15+1E-30),1E-16,1E-10,false,true),
        std::make_tuple(tensor_type(1E-15),tensor_type(1E-15+1E-30),1E-16,1E-32,false,false),
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-14,1E-10,false,true),
        //big: relative tolerance plays
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-15,1E-10,false,true),
        std::make_tuple(tensor_type(1E15),tensor_type(1E15+1),1E-16,1E-10,false,false),
        //not floating point type
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,true),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,6},1E-10,1E-10,false,false),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,true),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<int>{1,2,3,4,6},1E-10,1E-10,true,false),
        //mixed types
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<double>{1.0,2.0,3.0,4.0,5.0},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,false),
        std::make_tuple(tensor<int>{1,2,3,4,5},tensor<double>{1.0,2.0,3.0,4.0,5.0},1E-10,1E-10,true,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,true),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,false),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,false,false),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0,5.0},tensor<int>{1,2,3,4,5},1E-10,1E-10,true,false),
        std::make_tuple(tensor<double>{1.0,2.0,3.0,4.0,5.0},tensor<float>{1.0,2.0+1E-11,3.0,4.0,5.0},1E-10,1E-10,false,true),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0,5.0},tensor<float>{1.0,nan,3.0,4.0,5.0},1E-10,1E-10,false,false),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0,5.0},tensor<float>{1.0,nan,3.0,4.0,5.0},1E-10,1E-10,true,true),
        //broadcast shape
        std::make_tuple(tensor_type{},tensor_type{}.reshape(0,1),1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{},tensor_type{}.reshape(0,1),1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{1.1,2.2+1E-11,3.3}},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{1.1,2.2,3.3}},1E-10,1E-10,true,true),
        std::make_tuple(tensor_type{1.1,2.2,3.3},tensor_type{{1.1,2.2-1E-11,3.3},{1.1,2.2+1E-11,3.3}},1E-10,1E-10,false,true),
        std::make_tuple(tensor_type{1.1,2.2,nan},tensor_type{{1.1,2.2-1E-11,nan},{1.1,2.2+1E-11,nan}},1E-10,1E-10,true,true)
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto relative_tolerance = std::get<2>(t);
        auto absolute_tolerance = std::get<3>(t);
        auto equal_nan = std::get<4>(t);
        auto expected = std::get<5>(t);

        auto result = allclose(ten_0,ten_1,relative_tolerance,absolute_tolerance,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_allclose_default_precision","test_math")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::allclose;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type e = std::numeric_limits<value_type>::epsilon();
    //0ten_0,1ten_1,2equal_nan,3expected
    auto test_data = std::make_tuple(
        //nan_equal false
        std::make_tuple(tensor_type(0.0),tensor_type(0.0),false,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),false,true),
        std::make_tuple(tensor_type(1.0+e),tensor_type(1.0),false,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1},false,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1+1E-10},false,false),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,nan},tensor_type{-1.0-e,0.0,0.0,1.0+e,nan},false,false),
        std::make_tuple(tensor_type{{-1.0,0.0,0.0+e,1.0+e,2.1,3.3},{-1.0,0.0,0.0+e,1.0+e,2.1,3.3}},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1,3.3},false,true),
        //nan_equal true
        std::make_tuple(tensor_type(0.0),tensor_type(0.0),true,true),
        std::make_tuple(tensor_type(1.0),tensor_type(1.0),true,true),
        std::make_tuple(tensor_type(1.0+e),tensor_type(1.0),true,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1},true,true),
        std::make_tuple(tensor_type{-1.0,0.0,0.0+e,1.0+e,2.1},tensor_type{-1.0-e,0.0,0.0,1.0+e,2.1+1E-10},true,false),
        std::make_tuple(tensor_type{nan,-1.0,0.0,0.0+e,1.0+e,nan},tensor_type{nan,-1.0-e,0.0,0.0,1.0+e,nan},true,true),
        std::make_tuple(tensor_type{{0.0+e,1.0+e,nan},{0.0+e,1.0+e,nan}},tensor_type{{0.0+e,1.0+e,nan}},true,true)
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto equal_nan = std::get<2>(t);
        auto expected = std::get<3>(t);

        auto result = allclose(ten_0,ten_1,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}
