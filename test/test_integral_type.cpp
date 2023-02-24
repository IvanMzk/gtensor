#include <tuple>
#include "catch.hpp"
#include "integral_type.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_integral_construction", "[test_integral_type]")
{
    using gtensor::integral;
    using gtensor::integral_exception;
    using helpers_for_testing::apply_by_element;

    //0denote_value_type,1constructor_parameter
    auto test_data = std::make_tuple(
        std::tuple<char, char>{0, 0},
        std::tuple<char, char>{0, 100},
        std::tuple<char, unsigned char>{0, 100},
        std::tuple<char, char>{0, -100},
        std::tuple<char, int>{0, 0},
        std::tuple<char, int>{0, 100},
        std::tuple<char, unsigned int>{0, 100},
        std::tuple<char, int>{0, -100},
        std::tuple<char, std::size_t>{0, 0},
        std::tuple<char, std::size_t>{0, 100},
        std::tuple<char, std::ptrdiff_t>{0, 100},
        std::tuple<char, std::ptrdiff_t>{0, -100},
        std::tuple<unsigned char, char>{0, 0},
        std::tuple<unsigned char, char>{0, 100},
        std::tuple<unsigned char, unsigned char>{0, 100},
        std::tuple<unsigned char, int>{0, 0},
        std::tuple<unsigned char, int>{0, 100},
        std::tuple<unsigned char, unsigned int>{0, 100},
        std::tuple<unsigned char, std::size_t>{0, 0},
        std::tuple<unsigned char, std::size_t>{0, 100},
        std::tuple<unsigned char, std::ptrdiff_t>{0, 100},
        std::tuple<int, char>{0, 0},
        std::tuple<int, char>{0, 100},
        std::tuple<int, unsigned char>{0, 100},
        std::tuple<int, char>{0, -100},
        std::tuple<int, int>{0, 0},
        std::tuple<int, int>{0, 100},
        std::tuple<int, unsigned int>{0, 100},
        std::tuple<int, int>{0, -100},
        std::tuple<int, std::size_t>{0, 0},
        std::tuple<int, std::size_t>{0, 100},
        std::tuple<int, std::ptrdiff_t>{0, 100},
        std::tuple<int, std::ptrdiff_t>{0, -100},
        std::tuple<unsigned int, char>{0, 0},
        std::tuple<unsigned int, char>{0, 100},
        std::tuple<unsigned int, unsigned char>{0, 100},
        std::tuple<unsigned int, int>{0, 0},
        std::tuple<unsigned int, int>{0, 100},
        std::tuple<unsigned int, unsigned int>{0, 100},
        std::tuple<unsigned int, std::size_t>{0, 0},
        std::tuple<unsigned int, std::size_t>{0, 100},
        std::tuple<unsigned int, std::ptrdiff_t>{0, 100},
        std::tuple<std::ptrdiff_t, char>{0, 0},
        std::tuple<std::ptrdiff_t, char>{0, 100},
        std::tuple<std::ptrdiff_t, unsigned char>{0, 100},
        std::tuple<std::ptrdiff_t, char>{0, -100},
        std::tuple<std::ptrdiff_t, int>{0, 0},
        std::tuple<std::ptrdiff_t, int>{0, 100},
        std::tuple<std::ptrdiff_t, unsigned int>{0, 100},
        std::tuple<std::ptrdiff_t, int>{0, -100},
        std::tuple<std::ptrdiff_t, std::size_t>{0, 0},
        std::tuple<std::ptrdiff_t, std::size_t>{0, 100},
        std::tuple<std::ptrdiff_t, std::ptrdiff_t>{0, 100},
        std::tuple<std::ptrdiff_t, std::ptrdiff_t>{0, -100},
        std::tuple<std::size_t, char>{0, 0},
        std::tuple<std::size_t, char>{0, 100},
        std::tuple<std::size_t, unsigned char>{0, 100},
        std::tuple<std::size_t, int>{0, 0},
        std::tuple<std::size_t, int>{0, 100},
        std::tuple<std::size_t, unsigned int>{0, 100},
        std::tuple<std::size_t, std::size_t>{0, 0},
        std::tuple<std::size_t, std::size_t>{0, 100},
        std::tuple<std::size_t, std::ptrdiff_t>{0, 100}
    );

    auto test_data_exception = std::make_tuple(
        std::tuple<char, unsigned char>{0, 200},
        std::tuple<char, unsigned int>{0, 200},
        std::tuple<char, std::size_t>{0, 200},
        std::tuple<char, int>{0, 200},
        std::tuple<char, int>{0, -200},
        std::tuple<char, std::ptrdiff_t>{0, 200},
        std::tuple<char, std::ptrdiff_t>{0, -200},
        std::tuple<unsigned char, unsigned int>{0, 500},
        std::tuple<unsigned char, std::size_t>{0, 500},
        std::tuple<unsigned char, int>{0, 500},
        std::tuple<unsigned char, int>{0, -500},
        std::tuple<unsigned char, std::ptrdiff_t>{0, 500},
        std::tuple<unsigned char, std::ptrdiff_t>{0, -500},
        std::tuple<int, unsigned int>{0, std::numeric_limits<int>::max()+static_cast<unsigned int>(1)},
        std::tuple<int, std::size_t>{0, std::numeric_limits<int>::max()+static_cast<std::size_t>(1)},
        std::tuple<int, std::ptrdiff_t>{0, std::numeric_limits<int>::max()+static_cast<std::ptrdiff_t>(1)},
        std::tuple<int, std::ptrdiff_t>{0, std::numeric_limits<int>::min()-static_cast<std::ptrdiff_t>(1)},
        std::tuple<unsigned int, std::size_t>{0, std::numeric_limits<unsigned int>::max()+static_cast<std::size_t>(1)},
        std::tuple<unsigned int, int>{0, -1},
        std::tuple<unsigned int, std::ptrdiff_t>{0, std::numeric_limits<unsigned int>::max()+static_cast<std::ptrdiff_t>(1)},
        std::tuple<unsigned int, std::ptrdiff_t>{0, -1}
    );

    auto test = [](const auto& t){
        using value_type = std::decay_t<decltype(std::get<0>(t))>;
        auto param = std::get<1>(t);
        REQUIRE_NOTHROW(integral<value_type>(param));
    };
    auto test_exception = [](const auto& t){
        using value_type = std::decay_t<decltype(std::get<0>(t))>;
        auto param = std::get<1>(t);
        REQUIRE_THROWS_AS(integral<value_type>(param), integral_exception);
    };
    apply_by_element(test,test_data);
    apply_by_element(test_exception,test_data_exception);
}
