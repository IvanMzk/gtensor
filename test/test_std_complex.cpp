/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <tuple>
#include <iostream>
#include <complex>
#include "catch.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_std_complex_strict_equality","[test_std_complex]")
{
    using gtensor::tensor;
    using value_type = std::complex<double>;
    using tensor_type = tensor<value_type>;
    using namespace std::complex_literals;
    static constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    REQUIRE(tensor_type{{1.0+1.0i,2.0+2.0i},{3.0+3.0i,4.0+4.0i}} == tensor_type{{1.0+1.0i,2.0+2.0i},{3.0+3.0i,4.0+4.0i}});
    REQUIRE(tensor_type{{1.0+1.0i,2.0+2.0i},{3.0+3.0i,4.2+4.0i}} != tensor_type{{1.0+1.0i,2.0+2.0i},{3.0+3.0i,4.0+4.0i}});
    REQUIRE(tensor_type{{1.0+1.0i,2.0+2.0i},{3.0+3.0i,4.0+4.2i}} != tensor_type{{1.0+1.0i,2.0+2.0i},{3.0+3.0i,4.0+4.0i}});

    REQUIRE(!tensor_equal(
        tensor<double>{{1.0,nan},{3.0,4.0}},
        tensor<double>{{1.0,nan},{3.0,4.0}}
    ));
    REQUIRE(tensor_equal(
        tensor<double>{{1.0,nan},{3.0,4.0}},
        tensor<double>{{1.0,nan},{3.0,4.0}},
        true
    ));
    REQUIRE(!tensor_equal(
        tensor_type{{1.0+1.0i,nan+2.0i},{3.0+3.0i,4.0+4.0i}},
        tensor_type{{1.0+1.0i,nan+2.0i},{3.0+3.0i,4.0+4.0i}}
    ));
    REQUIRE(tensor_equal(
        tensor_type{{1.0+1.0i,nan+2.0i},{3.0+3.0i,4.0+4.0i}},
        tensor_type{{1.0+1.0i,nan+2.0i},{3.0+3.0i,4.0+4.0i}},
        true
    ));

}

TEST_CASE("test_std_complex_close_equality","[test_std_complex]")
{
    using value_type = std::complex<double>;
    using tensor_type = gtensor::tensor<value_type>;
    using namespace std::complex_literals;
    static constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    REQUIRE(tensor_close(
        tensor_type{{1.12345+1.12345i,2.12345+2.12345i},{3.12345+3.12345i,4.12345+4.12345i}},
        tensor_type{{1.12345+1.12345i,2.12345+2.12345i},{3.12345+3.12345i,4.12345+4.12345i}}
    ));
    REQUIRE(!tensor_close(
        tensor_type{{1.12345+1.12345i,2.12545+2.12345i},{3.12345+3.12345i,4.12345+4.12345i}},
        tensor_type{{1.12345+1.12345i,2.12345+2.12345i},{3.12345+3.12345i,4.12345+4.12345i}}
    ));
    REQUIRE(!tensor_close(
        tensor_type{{1.12345+1.12345i,2.12345+2.12545i},{3.12345+3.12345i,4.12345+4.12345i}},
        tensor_type{{1.12345+1.12345i,2.12345+2.12345i},{3.12345+3.12345i,4.12345+4.12345i}}
    ));
    REQUIRE(tensor_close(
        tensor_type{{1.12345+1.12345i,2.12345+2.12545i},{3.12345+3.12345i,4.12345+4.12345i}},
        tensor_type{{1.12345+1.12345i,2.12345+2.12345i},{3.12345+3.12345i,4.12345+4.12345i}},
        1E-2,
        1E-2
    ));
    REQUIRE(!tensor_close(
        tensor_type{{1.12345+1.12345i,2.12345+2.12545i},{nan+3.12345i,4.12345+4.12345i}},
        tensor_type{{1.12345+1.12345i,2.12345+2.12345i},{nan+3.12345i,4.12345+4.12345i}},
        1E-2,
        1E-2
    ));
    REQUIRE(tensor_close(
        tensor_type{{1.12345+1.12345i,2.12345+2.12545i},{nan+3.12345i,4.12345+4.12345i}},
        tensor_type{{1.12345+1.12345i,2.12345+2.12345i},{nan+3.12345i,4.12345+4.12345i}},
        1E-2,
        1E-2,
        true
    ));
}
