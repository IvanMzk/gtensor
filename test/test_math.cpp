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

//test tensor fuzzy elementwise equality
TEST_CASE("test_is_close","test_math")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    using gtensor::is_close;
    using helpers_for_testing::apply_by_element;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type inf = std::numeric_limits<value_type>::infinity();
    //0ten_0,1ten_1,2relative_tolerance,3absolute_tolerance,4equal_nan,5expected
    auto test_data = std::make_tuple(
        //nan_equal false
        //near zero
        std::make_tuple(tensor_type(0.0), tensor_type(0.0), 1E-10, 1E-10, std::false_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-11), 1E-10, 1E-10, std::false_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-21), 1E-10, 1E-10, std::false_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-9), 1E-10, 1E-10, std::false_type{}, bool_tensor_type(false)),
        std::make_tuple(tensor_type(0.0), tensor_type{0.0+1E-11,0.0-1E-11,0.0+1E-8}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,false}),
        //near one
        std::make_tuple(tensor_type(1.1), tensor_type{1.1+1E-11,1.1-1E-11,1.1+1E-8}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,false}),
        //near big
        std::make_tuple(tensor_type(4E15), tensor_type{4E15+1.0,4E15-1.0,4E15+10.0}, 1E-15, 1E-16, std::false_type{}, bool_tensor_type{true,true,false}),
        //nans
        std::make_tuple(tensor_type{inf,2.2,nan,3.3}, tensor_type{inf,2.2,nan,3.3}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,false,true}),
        std::make_tuple(tensor_type{inf,2.2,nan,3.3}, tensor_type{inf,2.2,1.1,3.3}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,false,true}),
        std::make_tuple(tensor_type{inf,2.2,1.1,3.3}, tensor_type{inf,2.2,nan,3.3}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,false,true}),
        //nan_equal true
        //near zero
        std::make_tuple(tensor_type(0.0), tensor_type(0.0), 1E-10, 1E-10, std::true_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-11), 1E-10, 1E-10, std::true_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-21), 1E-10, 1E-10, std::true_type{}, bool_tensor_type(true)),
        std::make_tuple(tensor_type(0.0), tensor_type(0.0+1E-9), 1E-10, 1E-10, std::true_type{}, bool_tensor_type(false)),
        std::make_tuple(tensor_type(0.0), tensor_type{0.0+1E-11,0.0-1E-11,0.0+1E-8}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,false}),
        //near one
        std::make_tuple(tensor_type(1.1), tensor_type{1.1+1E-11,1.1-1E-11,1.1+1E-8}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,false}),
        //near big
        std::make_tuple(tensor_type(4E15), tensor_type{4E15+1.0,4E15-1.0,4E15+10.0}, 1E-15, 1E-16, std::true_type{}, bool_tensor_type{true,true,false}),
        //nans
        std::make_tuple(tensor_type{inf,2.2,nan,3.3}, tensor_type{inf,2.2,nan,3.3}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor_type{inf,2.2,nan,3.3}, tensor_type{inf,2.2,1.1,3.3}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,false,true}),
        std::make_tuple(tensor_type{inf,2.2,1.1,3.3}, tensor_type{inf,2.2,nan,3.3}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,false,true}),
        //not floating point type
        std::make_tuple(tensor<int>{1,2,3,4}, tensor<int>{1,2,3,4}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<int>{1,2,3,4}, tensor<int>{1,2,3,3}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,true,false}),
        std::make_tuple(tensor<int>{1,2,3,4}, tensor<int>{1,2,3,4}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<int>{1,2,3,4}, tensor<int>{1,2,3,3}, 1E-10, 1E-10, std::true_type{}, bool_tensor_type{true,true,true,false}),
        //mixed types
        std::make_tuple(tensor<int>{1,2,3,4}, tensor<double>{1.0,2.0,3.0,4.0}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0}, tensor<int>{1,2,3,4}, 1E-10, 1E-10, std::false_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0},tensor<int>{1,2,3,4},1E-10,1E-10, std::false_type{}, bool_tensor_type{true,false,true,true}),
        std::make_tuple(tensor<int>{1,2,3,4},tensor<double>{1.0,2.0,3.0,4.0},1E-10,1E-10,std::true_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<double>{1.0,2.0+1E-11,3.0,4.0},tensor<int>{1,2,3,4},1E-10,1E-10,std::true_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<double>{1.0,2.0+1E-8,3.0,4.0},tensor<int>{1,2,3,4},1E-10,1E-10,std::true_type{}, bool_tensor_type{true,false,true,true}),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0},tensor<int>{1,2,3,4},1E-10,1E-10, std::false_type{}, bool_tensor_type{true,true,false,true}),
        std::make_tuple(tensor<double>{1.0,2.0,nan,4.0},tensor<int>{1,2,3,4},1E-10,1E-10,std::true_type{}, bool_tensor_type{true,true,false,true}),
        std::make_tuple(tensor<double>{1.0,2.0,3.0,4.0},tensor<float>{1.0,2.0+1E-11,3.0,4.0},1E-10,1E-10, std::false_type{}, bool_tensor_type{true,true,true,true}),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0},tensor<float>{1.0,nan,3.0,4.0},1E-10,1E-10, std::false_type{}, bool_tensor_type{true,false,true,true}),
        std::make_tuple(tensor<double>{1.0,nan,3.0,4.0},tensor<float>{1.0,nan,3.0,4.0},1E-10,1E-10,std::true_type{}, bool_tensor_type{true,true,true,true})
    );
    auto test = [](const auto& t){
        auto ten_0 = std::get<0>(t);
        auto ten_1 = std::get<1>(t);
        auto relative_tolerance = std::get<2>(t);
        auto absolute_tolerance = std::get<3>(t);
        auto equal_nan = std::get<4>(t);
        auto expected = std::get<5>(t);

        auto result = is_close(ten_0,ten_1,relative_tolerance,absolute_tolerance,equal_nan);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//test math element wise functions
TEST_CASE("test_tensor_math_comparison_functions_semantic","[test_math]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;
    SECTION("test_isgreater")
    {
        auto result = gtensor::isgreater(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,4.0}}, tensor_type{2.0,-3.0,4.0});
        auto expected = bool_tensor_type{{false,true,false},{true,false,false}};
        REQUIRE(result == expected);
    }
    SECTION("test_isgreaterequal")
    {
        auto result = gtensor::isgreaterequal(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,4.0}}, tensor_type{2.0,-3.0,4.0});
        auto expected = bool_tensor_type{{false,true,false},{true,false,true}};
        REQUIRE(result == expected);
    }
    SECTION("test_isless")
    {
        auto result = gtensor::isless(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,4.0}}, tensor_type{2.0,-3.0,4.0});
        auto expected = bool_tensor_type{{true,false,true},{false,true,false}};
        REQUIRE(result == expected);
    }
    SECTION("test_islessequal")
    {
        auto result = gtensor::islessequal(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,4.0}}, tensor_type{2.0,-3.0,4.0});
        auto expected = bool_tensor_type{{true,false,true},{false,true,true}};
        REQUIRE(result == expected);
    }
    SECTION("test_islessgreater")
    {
        auto result = gtensor::islessgreater(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,4.0}}, tensor_type{2.0,-3.0,4.0});
        auto expected = bool_tensor_type{{true,true,true},{true,true,false}};
        REQUIRE(result == expected);
    }
}

TEST_CASE("test_tensor_math_basic_functions_semantic","[test_math]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    SECTION("test_abs")
    {
        auto result = gtensor::abs(tensor_type{{0.0,1.1,-2.2},{-4.4,5.5,-6.6}});
        auto expected = tensor_type{{0.0,1.1,2.2},{4.4,5.5,6.6}};
        REQUIRE(result == expected);
    }
    SECTION("test_fmod")
    {
        auto result = gtensor::fmod(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,-6.6}}, tensor_type{2.0,-3.0,4.0});
        auto expected = tensor_type{{0.0,1.1,-2.2},{0.4,-2.5,-2.6}};
        REQUIRE(tensor_close(result,expected));
    }
    SECTION("test_remainder")
    {
        auto result = gtensor::remainder(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,-6.6}}, tensor_type{2.0,-3.0,4.0});
        auto expected = tensor_type{{0.0,1.1,1.8},{0.4,0.5,1.4}};
        REQUIRE(tensor_close(result,expected));
    }
    SECTION("test_fma")
    {
        auto result = gtensor::fma(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,-6.6}}, tensor_type{2.0,-3.0,4.0}, 1.0);
        auto expected = tensor_type{{1.0,-2.3,-7.8},{9.8,17.5,-25.4}};
        REQUIRE(tensor_close(result,expected));
    }
    SECTION("test_fmax")
    {
        auto result = gtensor::fmax(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,-6.6}}, tensor_type{2.0,-3.0,4.0});
        auto expected = tensor_type{{2.0,1.1,4.0},{4.4,-3.0,4.0}};
        REQUIRE(result == expected);
    }
    SECTION("test_fdim")
    {
        auto result = gtensor::fdim(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,-6.6}}, tensor_type{2.0,-3.0,4.0});
        auto expected = tensor_type{{0.0,4.1,0.0},{2.4,0.0,0.0}};
        REQUIRE(tensor_close(result,expected));
    }
    SECTION("test_clip")
    {
        auto result = gtensor::clip(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,-6.6}}, tensor_type{0.0,1.0,2.0}, 3.0);
        auto expected = tensor_type{{0.0,1.1,2.0},{3.0,1.0,2.0}};
        REQUIRE(result == expected);
    }
}

TEST_CASE("test_tensor_math_exponential_functions_semantic","[test_math]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    SECTION("test_exp")
    {
        auto result = gtensor::exp(tensor_type{0.0,1.0,-1.0});
        auto expected = tensor_type{1.0,2.718,0.3678};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
    SECTION("test_exp2")
    {
        auto result = gtensor::exp2(tensor_type{0.0,1.0,-1.0});
        auto expected = tensor_type{1.0,2.0,0.5};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
    SECTION("test_expm1")
    {
        auto result = gtensor::expm1(tensor_type{0.0,1.0,-1.0});
        auto expected = tensor_type{0.0,1.718,-0.6321};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
    SECTION("test_log")
    {
        auto result = gtensor::log(tensor_type{1.0,2.718,0.3678});
        auto expected = tensor_type{0.0,1.0,-1.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
    SECTION("test_log10")
    {
        auto result = gtensor::log10(tensor_type{1.0,10.0,0.1});
        auto expected = tensor_type{0.0,1.0,-1.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
    SECTION("test_log2")
    {
        auto result = gtensor::log2(tensor_type{1.0,2.0,0.5});
        auto expected = tensor_type{0.0,1.0,-1.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
    SECTION("test_log1p")
    {
        auto result = gtensor::log1p(tensor_type{0.0,1.718,-0.6321});
        auto expected = tensor_type{0.0,1.0,-1.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
}

TEST_CASE("test_tensor_math_power_functions_semantic","[test_math]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;
    SECTION("test_pow")
    {
        auto result = gtensor::pow(tensor_type{{0.0,1.0,-1.0},{0.1,-2.0,1.2}}, tensor_type{0.0,-1.0,2.0});
        auto expected = tensor_type{{1.0,1.0,1.0},{1.0,-0.5,1.44}};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
    SECTION("test_sqrt")
    {
        auto result = gtensor::sqrt(tensor_type{0.0,1.0,2.0});
        auto expected = tensor_type{0.0,1.0,1.414};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
    SECTION("test_cbrt")
    {
        auto result = gtensor::cbrt(tensor_type{0.0,1.0,2.0});
        auto expected = tensor_type{0.0,1.0,1.259};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
    SECTION("test_hypot")
    {
        auto result = gtensor::hypot(tensor_type{{0.0,1.0,2.0},{3.0,4.0,5.0}}, tensor_type{0.0,1.0,2.0});
        auto expected = tensor_type{{0.0,1.414,2.828},{3.0,4.123,5.385}};
        REQUIRE(tensor_close(result,expected,1E-3,1E-10));
    }
}

TEST_CASE("test_tensor_math_trigonometric_functions_semantic","[test_math]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;

    SECTION("test_sin")
    {
        auto result = gtensor::sin(tensor_type{-3.141,-1.571,0.0,1.571,3.141});
        auto expected = tensor_type{0.0,-1.0,0.0,1.0,0.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_cos")
    {
        auto result = gtensor::cos(tensor_type{-3.141,-1.571,0.0,1.571,3.141});
        auto expected = tensor_type{-1.0,0.0,1.0,0.0,-1.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_tan")
    {
        auto result = gtensor::tan(tensor_type{-0.7854,0.0,0.7854});
        auto expected = tensor_type{-1.0,0.0,1.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_asin")
    {
        auto result = gtensor::asin(tensor_type{-1.0,0.0,1.0});
        auto expected = tensor_type{-1.571,0.0,1.571};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_acos")
    {
        auto result = gtensor::acos(tensor_type{-1.0,0.0,1.0});
        auto expected = tensor_type{3.141,1.571,0.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_atan")
    {
        auto result = gtensor::atan(tensor_type{-1.0,0.0,1.0});
        auto expected = tensor_type{-0.7854,0.0,0.7854};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_atan2")
    {
        auto result = gtensor::atan2(tensor_type{-1.0,0.0,2.0},tensor_type{1.0,1.0,2.0});
        auto expected = tensor_type{-0.7854,0.0,0.7854};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
}

TEST_CASE("test_tensor_math_hyperbolic_functions_semantic","[test_math]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;

    SECTION("test_sinh")
    {
        auto result = gtensor::sinh(tensor_type{-1.0,0.0,1.0});
        auto expected = tensor_type{-1.175,0.0,1.175};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_cosh")
    {
        auto result = gtensor::cosh(tensor_type{-1.0,0.0,1.0});
        auto expected = tensor_type{1.543,1.0,1.543};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_tanh")
    {
        auto result = gtensor::tanh(tensor_type{-1.0,0.0,1.0});
        auto expected = tensor_type{-0.761,0.0,0.761};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_asinh")
    {
        auto result = gtensor::asinh(tensor_type{-1.175,0.0,1.175});
        auto expected = tensor_type{-1.0,0.0,1.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_acosh")
    {
        auto result = gtensor::acosh(tensor_type{1.0,1.543});
        auto expected = tensor_type{0.0,1.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
    SECTION("test_atanh")
    {
        auto result = gtensor::atanh(tensor_type{-0.761,0.0,0.761});
        auto expected = tensor_type{-1.0,0.0,1.0};
        REQUIRE(tensor_close(result,expected,1E-3,1E-3));
    }
}

TEST_CASE("test_tensor_math_nearest_functions_semantic","[test_math]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;

    SECTION("test_ceil")
    {
        auto result = gtensor::ceil(tensor_type{-2.4,0.0,1.0,2.4});
        auto expected = tensor_type{-2.0,0.0,1.0,3.0};
        REQUIRE(tensor_close(result,expected,1E-10,1E-10));
    }
    SECTION("test_floor")
    {
        auto result = gtensor::floor(tensor_type{-2.4,0.0,1.0,2.4});
        auto expected = tensor_type{-3.0,0.0,1.0,2.0};
        REQUIRE(tensor_close(result,expected,1E-10,1E-10));
    }
    SECTION("test_trunc")
    {
        auto result = gtensor::trunc(tensor_type{-2.4,0.0,1.0,2.4});
        auto expected = tensor_type{-2.0,0.0,1.0,2.0};
        REQUIRE(tensor_close(result,expected,1E-10,1E-10));
    }
    SECTION("test_round")
    {
        auto result = gtensor::round(tensor_type{-4.5,-3.7,-2.4,0.0,1.0,2.4,3.7,4.5});
        auto expected = tensor_type{-5.0,-4.0,-2.0,0.0,1.0,2.0,4.0,5.0};
        REQUIRE(tensor_close(result,expected,1E-10,1E-10));
    }
}

TEST_CASE("test_tensor_math_floating_point_manipulation_functions_semantic","[test_math]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;

    SECTION("test_frexp")
    {
        auto result = gtensor::frexp(tensor_type{-1.0,0.0,1.0,2.0,3.0});
        auto expected = tensor<std::pair<value_type,int>>{
            std::make_pair(-0.5,1),
            std::make_pair(0.0,0),
            std::make_pair(0.5,1),
            std::make_pair(0.5,2),
            std::make_pair(0.75,2)
        };
        REQUIRE(result == expected);
    }
    SECTION("test_ldexp")
    {
        auto result = gtensor::ldexp(tensor_type{-1.0,0.0,1.0,2.0}, tensor_type{-1.0,0.0,1.0,2.0});
        auto expected = tensor_type{-0.5,0.0,2.0,8.0};
        REQUIRE(tensor_close(result,expected,1E-10,1E-10));
    }
    SECTION("test_modf")
    {
        auto result = gtensor::modf(tensor_type{-1.5,-0.5,0.0,0.5,1.5,2.0});
        auto expected = tensor<std::pair<value_type,value_type>>{
            std::make_pair(-1.0,-0.5),
            std::make_pair(0.0,-0.5),
            std::make_pair(0.0,0.0),
            std::make_pair(0.0,0.5),
            std::make_pair(1.0,0.5),
            std::make_pair(2.0,0.0)
        };
        REQUIRE(result == expected);
    }
    SECTION("test_nextafter")
    {
        static constexpr value_type e = std::numeric_limits<value_type>::epsilon();
        auto result = gtensor::nextafter(tensor_type{1.0,2.0,4.0}, tensor_type(5));
        auto expected = tensor_type{1.0+e,2.0+(e+e),4.0+(e+e+e+e)};
        REQUIRE(result == expected);
    }
    SECTION("test_copysign")
    {
        auto result = gtensor::copysign(tensor_type{-1.0,0.0,1.0,2.0}, tensor_type{-1.0,0.0,1.0,-2.0});
        auto expected = tensor_type{-1.0,0.0,1.0,-2.0};
        REQUIRE(tensor_close(result,expected,1E-10,1E-10));
    }
    SECTION("test_nan_to_num")
    {
        static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
        static constexpr value_type inf = std::numeric_limits<value_type>::infinity();
        auto result = gtensor::nan_to_num(tensor_type{0.0,nan,1.0,2.2,-3.3,inf,nan,-inf}, value_type{1E-100},value_type{1E50},value_type{-1E50});
        auto expected = tensor_type{0.0,1E-100,1.0,2.2,-3.3,1E50,1E-100,-1E50};
        REQUIRE(result == expected);
    }
    SECTION("test_nan_to_num_defaults")
    {
        static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
        static constexpr value_type inf = std::numeric_limits<value_type>::infinity();

        static constexpr value_type nan_num = 0.0;
        static constexpr value_type pos_inf_num = std::numeric_limits<value_type>::max();
        static constexpr value_type neg_inf_num = std::numeric_limits<value_type>::min();

        auto result = gtensor::nan_to_num(tensor_type{0.0,nan,1.0,2.2,-3.3,inf,nan,-inf});
        auto expected = tensor_type{0.0,nan_num,1.0,2.2,-3.3,pos_inf_num,nan_num,neg_inf_num};
        REQUIRE(result == expected);
    }
}

TEST_CASE("test_tensor_math_classification_functions_semantic","[test_math]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using bool_tensor_type = gtensor::tensor<bool>;

    SECTION("test_isfinite")
    {
        auto result = gtensor::isfinite(tensor_type{-1.0/0.0,-1.0,0.0/0.0,std::numeric_limits<value_type>::min()/2.0,0.0,1.0,1.0/0.0});
        auto expected = bool_tensor_type{false,true,false,true,true,true,false};
        REQUIRE(result == expected);
    }
    SECTION("test_isinf")
    {
        auto result = gtensor::isinf(tensor_type{-1.0/0.0,-1.0,0.0/0.0,std::numeric_limits<value_type>::min()/2.0,0.0,1.0,1.0/0.0});
        auto expected = bool_tensor_type{true,false,false,false,false,false,true};
        REQUIRE(result == expected);
    }
    SECTION("test_isnan")
    {
        auto result = gtensor::isnan(tensor_type{-1.0/0.0,-1.0,0.0/0.0,std::numeric_limits<value_type>::min()/2.0,0.0,1.0,1.0/0.0});
        auto expected = bool_tensor_type{false,false,true,false,false,false,false};
        REQUIRE(result == expected);
    }
    SECTION("test_isnormal")
    {
        auto result = gtensor::isnormal(tensor_type{-1.0/0.0,-1.0,0.0/0.0,std::numeric_limits<value_type>::min()/2.0,0.0,1.0,1.0/0.0});
        auto expected = bool_tensor_type{false,true,false,false,false,true,false};
        REQUIRE(result == expected);
    }
}

TEST_CASE("test_tensor_math_rotines_in_rational_domain_functions_semantic","[test_math]")
{
    using value_type = std::int64_t;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::tensor_close;

    SECTION("test_gcd")
    {
        auto result = gtensor::gcd(tensor_type{6,24,18,0}, tensor_type{10,0,-15,0});
        auto expected = tensor_type{2,24,3,0};
        REQUIRE(result == expected);
    }
    SECTION("test_lcm")
    {
        auto result = gtensor::lcm(tensor_type{6,24,18,0}, tensor_type{10,0,-15,0});
        auto expected = tensor_type{30,0,90,0};
        REQUIRE(result == expected);
    }
}

//test math functions along axes
//amin
TEST_CASE("test_math_amin","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amin;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(amin(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(amin(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>())),tensor_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{},std::vector<int>{},false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,tensor_type{value_type{},value_type{}}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,tensor_type(-1)),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},false,tensor_type(-1)),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},false,tensor_type(-1)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},0,false,tensor_type{{1,4,3},{1,0,-1}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,false,tensor_type{{1,0,-1},{1,4,2}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,false,tensor_type{{1,-1},{4,1}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{0,1},false,tensor_type{1,0,-1}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},false,tensor_type{-1,1}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},false,tensor_type(-1)),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{},std::vector<int>{0},true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,tensor_type{{{value_type{}},{value_type{}}}}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,true,tensor_type{-1}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},true,tensor_type{-1}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},true,tensor_type{-1}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},0,true,tensor_type{{{1,4,3},{1,0,-1}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,true,tensor_type{{{1,0,-1}},{{1,4,2}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,true,tensor_type{{{1},{-1}},{{4},{1}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{0,1},true,tensor_type{{{1,0,-1}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},true,tensor_type{{{-1}},{{1}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},true,tensor_type{{{-1}}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = amin(ten,axes,keep_dims);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_math_amin_initializer_list_axes_all_axes","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amin;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(amin(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>())),tensor_type>);

    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{1},false) == tensor_type{{1,0,-1},{1,4,2}});
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1},false) == tensor_type{-1,1});
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true) == tensor_type{{{1,0,-1}}});
    //all axes
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false) == tensor_type(-1));
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{},false) == tensor_type(-1));
    REQUIRE(amin(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true) == tensor_type{{{-1}}});
}

//amax
TEST_CASE("test_math_amax","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amax;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(amax(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(amax(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>())),tensor_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{},std::vector<int>{},false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,tensor_type{value_type{},value_type{}}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,false,tensor_type(5)),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},false,tensor_type(5)),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},false,tensor_type(5)),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},0,false,tensor_type{{7,5,9},{2,11,2}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,false,tensor_type{{2,5,3},{7,11,9}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,false,tensor_type{{5,2},{9,11}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{0,1},false,tensor_type{7,11,9}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},false,tensor_type{5,11}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},false,tensor_type(11)),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{},std::vector<int>{0},true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,tensor_type{{{value_type{}},{value_type{}}}}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},0,true,tensor_type{5}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{0},true,tensor_type{5}),
        std::make_tuple(tensor_type{5,2,1,-1,4,4},std::vector<int>{},true,tensor_type{5}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},0,true,tensor_type{{{7,5,9},{2,11,2}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},1,true,tensor_type{{{2,5,3}},{{7,11,9}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},2,true,tensor_type{{{5},{2}},{{9},{11}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{0,1},true,tensor_type{{{7,11,9}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{2,1},true,tensor_type{{{5}},{{11}}}),
        std::make_tuple(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},std::vector<int>{},true,tensor_type{{{11}}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = amax(ten,axes,keep_dims);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_math_amax_initializer_list_axes_all_axes","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::amax;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(amax(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>())),tensor_type>);

    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{1},false) == tensor_type{{2,5,3},{7,11,9}});
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{2,1},false) == tensor_type{5,11});
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{0,1},true) == tensor_type{{{7,11,9}}});
    //all axes
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},false) == tensor_type(11));
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},{},false) == tensor_type(11));
    REQUIRE(amax(tensor_type{{{1,5,3},{2,0,-1}},{{7,4,9},{1,11,2}}},true) == tensor_type{{{11}}});
}

//sum
TEST_CASE("test_math_sum","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::sum;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(sum(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(sum(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>())),tensor_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{},std::vector<int>{},false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,tensor_type{value_type{},value_type{}}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,tensor_type(15)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,tensor_type(15)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,tensor_type(15)),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,tensor_type{{8,10,12},{14,16,18}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,tensor_type{{5,7,9},{17,19,21}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,tensor_type{{6,15},{24,33}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,tensor_type{22,26,30}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,tensor_type{21,57}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,tensor_type(78)),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{},std::vector<int>{0},true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,tensor_type{{{value_type{}},{value_type{}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,true,tensor_type{15}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,tensor_type{15}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,tensor_type{15}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,tensor_type{{{8,10,12},{14,16,18}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,tensor_type{{{5,7,9}},{{17,19,21}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,tensor_type{{{6},{15}},{{24},{33}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,tensor_type{{{22,26,30}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,tensor_type{{{21}},{{57}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,tensor_type{{{78}}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = sum(ten,axes,keep_dims);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_math_sum_initializer_list_axes_all_axes","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::sum;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(sum(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>())),tensor_type>);

    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{8,10,12},{14,16,18}});
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{22,26,30});
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{21}},{{57}}});
    //all axes
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{},false) == tensor_type(78));
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(78));
    REQUIRE(sum(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{78}}});
}

//prod
TEST_CASE("test_math_prod","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::prod;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(prod(std::declval<tensor_type>(),std::declval<int>(),std::declval<bool>())),tensor_type>);
    REQUIRE(std::is_same_v<decltype(prod(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<bool>())),tensor_type>);

    //0tensor,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{},std::vector<int>{0},false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{},std::vector<int>{},false,tensor_type(value_type{})),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},false,tensor_type{value_type{},value_type{}}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,false,tensor_type(120)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},false,tensor_type(120)),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},false,tensor_type(120)),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,false,tensor_type{{7,16,27},{40,55,72}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,false,tensor_type{{4,10,18},{70,88,108}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,false,tensor_type{{6,120},{504,1320}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},false,tensor_type{280,880,1944}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},false,tensor_type{720,665280}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},false,tensor_type(479001600)),
        //keep_dim true
        std::make_tuple(tensor_type{},0,true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{},std::vector<int>{0},true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{},std::vector<int>{},true,tensor_type{value_type{}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3),std::vector<int>{0,2},true,tensor_type{{{value_type{}},{value_type{}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,true,tensor_type{120}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{0},true,tensor_type{120}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::vector<int>{},true,tensor_type{120}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,true,tensor_type{{{7,16,27},{40,55,72}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,true,tensor_type{{{4,10,18}},{{70,88,108}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,true,tensor_type{{{6},{120}},{{504},{1320}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{0,1},true,tensor_type{{{280,880,1944}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{2,1},true,tensor_type{{{720}},{{665280}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<int>{},true,tensor_type{{{479001600}}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = prod(ten,axes,keep_dims);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_math_prod_initializer_list_axes_all_axes","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::prod;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(prod(std::declval<tensor_type>(),std::declval<std::initializer_list<int>>(),std::declval<bool>())),tensor_type>);

    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0},false) == tensor_type{{7,16,27},{40,55,72}});
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{0,1},false) == tensor_type{280,880,1944});
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{2,1},true) == tensor_type{{{720}},{{665280}}});
    //all axes
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},{},false) == tensor_type(479001600));
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},false) == tensor_type(479001600));
    REQUIRE(prod(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},true) == tensor_type{{{479001600}}});
}

//cumsum
TEST_CASE("test_math_cumsum","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumsum;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(cumsum(std::declval<tensor_type>(),std::declval<int>())),tensor_type>);

    //0tensor,1axes,2expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,tensor_type{{{1,2,3},{4,5,6}},{{8,10,12},{14,16,18}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,tensor_type{{{1,2,3},{5,7,9}},{{7,8,9},{17,19,21}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,tensor_type{{{1,3,6},{4,9,15}},{{7,15,24},{10,21,33}}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = cumsum(ten,axes);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_math_cumsum_all_axes","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumsum;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(cumsum(std::declval<tensor_type>())),tensor_type>);

    //0tensor,1expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},tensor_type{1,3,6,10,15,21,28,36,45,55,66,78})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto result = cumsum(ten);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//cumprod
TEST_CASE("test_math_cumprod","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumprod;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(cumprod(std::declval<tensor_type>(),std::declval<int>())),tensor_type>);

    //0tensor,1axes,2expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},0,tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},0,tensor_type{1,2,6,24,120}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},0,tensor_type{{{1,2,3},{4,5,6}},{{7,16,27},{40,55,72}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},1,tensor_type{{{1,2,3},{4,10,18}},{{7,8,9},{70,88,108}}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},2,tensor_type{{{1,2,6},{4,20,120}},{{7,56,504},{10,110,1320}}})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = cumprod(ten,axes);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_math_cumprod_all_axes","test_math")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::cumprod;
    using helpers_for_testing::apply_by_element;

    REQUIRE(std::is_same_v<decltype(cumprod(std::declval<tensor_type>())),tensor_type>);

    //0tensor,1expected
    auto test_data = std::make_tuple(
        //keep_dim false
        std::make_tuple(tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{1,2,6,24,120}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{1,2,3},{0,4,5}}},tensor_type{1,2,6,24,120,720,720,1440,4320,0,0,0})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto result = cumprod(ten);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

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
