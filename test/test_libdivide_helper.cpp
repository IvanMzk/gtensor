/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include "catch.hpp"
#include <iostream>
#include "config_for_testing.hpp"
#include "libdivide_helper.hpp"

TEMPLATE_TEST_CASE("test_make_dividers", "[test_libdivide_helper]", std::size_t, std::ptrdiff_t)
{
    using native_config_type = test_config::config_div_mode_selector_t<gtensor::config::mode_div_native>;
    using libdivide_config_type = test_config::config_div_mode_selector_t<gtensor::config::mode_div_libdivide>;
    using value_type = TestType;
    using libdivide_divider_type =  gtensor::detail::libdivide_divider<value_type>;
    using gtensor::detail::make_dividers;

    REQUIRE(make_dividers<native_config_type>(std::vector<value_type>{}) == std::vector<value_type>{});
    REQUIRE(make_dividers<native_config_type>(std::vector<value_type>{1}) == std::vector<value_type>{1});
    REQUIRE(make_dividers<native_config_type>(std::vector<value_type>{3,4,5}) == std::vector<value_type>{3,4,5});

    REQUIRE(make_dividers<libdivide_config_type>(std::vector<value_type>{}) == std::vector<libdivide_divider_type>{});
    //REQUIRE(make_dividers<libdivide_config_type>(std::vector<value_type>{0}) == std::vector<libdivide_divider_type>{0});
    REQUIRE(make_dividers<libdivide_config_type>(std::vector<value_type>{1}) == std::vector<libdivide_divider_type>{libdivide_divider_type{1}});
    REQUIRE(make_dividers<libdivide_config_type>(std::vector<value_type>{3,4,5}) == std::vector<libdivide_divider_type>{libdivide_divider_type{3},libdivide_divider_type{4},libdivide_divider_type{5}});
}

TEMPLATE_TEST_CASE("test_divide", "[test_libdivide_helper]", std::size_t, std::ptrdiff_t)
{
    using value_type = TestType;
    using libdivide_divider_type =  gtensor::detail::libdivide_divider<value_type>;
    using gtensor::detail::divide;

    //0divident,1native_divider,2libdivide_divider,3expected_quotient,4expected_reminder
    using test_type = std::tuple<value_type, value_type, libdivide_divider_type, value_type, value_type>;
    auto test_data = GENERATE(
        test_type{value_type{0}, value_type{1}, libdivide_divider_type{1}, value_type{0}, value_type{0}},
        test_type{value_type{1}, value_type{1}, libdivide_divider_type{1}, value_type{1}, value_type{0}},
        test_type{value_type{2}, value_type{1}, libdivide_divider_type{1}, value_type{2}, value_type{0}},
        test_type{value_type{2}, value_type{2}, libdivide_divider_type{2}, value_type{1}, value_type{0}},
        test_type{value_type{3}, value_type{2}, libdivide_divider_type{2}, value_type{1}, value_type{1}},
        test_type{value_type{2}, value_type{3}, libdivide_divider_type{3}, value_type{0}, value_type{2}}
    );
    auto divident = std::get<0>(test_data);
    auto native_divider = std::get<1>(test_data);
    auto libdivide_divider = std::get<2>(test_data);
    auto expected_quotient = std::get<3>(test_data);
    auto expected_remainder = std::get<4>(test_data);

    SECTION("native_divider"){
        auto result_quotient = divide(divident,native_divider);
        auto result_remainder = divident;
        REQUIRE(result_quotient == expected_quotient);
        REQUIRE(result_remainder == expected_remainder);
    }

    SECTION("libdivide_divider"){
        auto result_quotient = divide(divident,libdivide_divider);
        auto result_remainder = divident;
        REQUIRE(result_quotient == expected_quotient);
        REQUIRE(result_remainder == expected_remainder);
    }
}
