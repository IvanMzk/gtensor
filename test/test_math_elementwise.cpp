#include <limits>
#include <iomanip>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "tensor_math.hpp"
#include "tensor.hpp"

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
    using gtensor::tensor;
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
    SECTION("test_fmin")
    {
        auto result = gtensor::fmin(tensor_type{{0.0,1.1,-2.2},{4.4,-5.5,-6.6}}, tensor_type{2.0,-3.0,4.0});
        auto expected = tensor_type{{0.0,-3.0,-2.2},{2.0,-5.5,-6.6}};
        REQUIRE(result == expected);
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
    SECTION("test_divmod")
    {
        auto result = gtensor::divmod(tensor_type{-3.0,-2.0,0.0,1.0,5.0}, tensor_type{1.2,-1.6,1.0,2.0,-2.0});
        auto expected = tensor<std::pair<value_type,value_type>>{
            std::make_pair(-3.0,0.6),
            std::make_pair(1.0,-0.4),
            std::make_pair(0.0,0.0),
            std::make_pair(0.0,1.0),
            std::make_pair(-3.0,-1.0)
        };
        REQUIRE(
            std::equal(
                result.begin(),
                result.end(),
                expected.begin(),
                expected.end(),
                [](auto res, auto exp){
                    return gtensor::math::isclose(res.first, exp.first, 1E-10, 1E-10) && gtensor::math::isclose(res.second, exp.second, 1E-10, 1E-10);
                }
            )
        );
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
        auto result = gtensor::ldexp(tensor_type{-1.0,0.0,1.0,2.0}, tensor<int>{-1.0,0.0,1.0,2.0});
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
        static constexpr value_type neg_inf_num = std::numeric_limits<value_type>::lowest();

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
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();

    SECTION("test_isfinite")
    {
        auto result = gtensor::isfinite(tensor_type{neg_inf,-1.0,nan,std::numeric_limits<value_type>::min()/2.0,0.0,1.0,pos_inf});
        auto expected = bool_tensor_type{false,true,false,true,true,true,false};
        REQUIRE(result == expected);
    }
    SECTION("test_isinf")
    {
        auto result = gtensor::isinf(tensor_type{neg_inf,-1.0,nan,std::numeric_limits<value_type>::min()/2.0,0.0,1.0,pos_inf});
        auto expected = bool_tensor_type{true,false,false,false,false,false,true};
        REQUIRE(result == expected);
    }
    SECTION("test_isnan")
    {
        auto result = gtensor::isnan(tensor_type{neg_inf,-1.0,nan,std::numeric_limits<value_type>::min()/2.0,0.0,1.0,pos_inf});
        auto expected = bool_tensor_type{false,false,true,false,false,false,false};
        REQUIRE(result == expected);
    }
    SECTION("test_isnormal")
    {
        auto result = gtensor::isnormal(tensor_type{neg_inf,-1.0,nan,std::numeric_limits<value_type>::min()/2.0,0.0,1.0,pos_inf});
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
