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

TEST_CASE("test_std_complex_operators","[test_std_complex]")
{
    using value_type = std::complex<double>;
    using tensor_type = gtensor::tensor<value_type>;
    using namespace std::complex_literals;

    const tensor_type a{{1.1+2.2i,2.2+1.1i},{3.3+4.4i,4.4+3.3i}};
    const tensor_type b{5.1+6.2i,6.2+5.1i};

    REQUIRE(+a == a);
    REQUIRE(-a == tensor_type{{-1.1-2.2i,-2.2-1.1i},{-3.3-4.4i,-4.4-3.3i}});
    REQUIRE(tensor_close((a+b)*1.1,tensor_type{{6.82+9.24i,9.24+6.82i},{9.24+11.66i,11.66+9.24i}}));
    REQUIRE(tensor_close((a+b)*(a-b),tensor_type{{8.8-58.4i,-8.8-58.4i},{3.96-34.2i,-3.96-34.2i}}));
    REQUIRE(tensor_close((a+b)/(a-b),tensor_type{{-1.825-0.275i,-1.825+0.275i},{-5.27777778-0.61111111i,-5.27777778+0.61111111i}},1E-3,1E-3));
}

TEST_CASE("test_std_complex_assign","[test_std_complex]")
{
    using value_type = std::complex<double>;
    using tensor_type = gtensor::tensor<value_type>;
    using namespace std::complex_literals;

    const tensor_type a{{1.1+2.2i,2.2+1.1i},{3.3+4.4i,4.4+3.3i}};
    const tensor_type b{5.1+6.2i,6.2+5.1i};

    SECTION("value_assign")
    {
        tensor_type c{1,2,3};
        c = a;
        REQUIRE(c==a);
    }
    SECTION("broadcast_assign")
    {
        auto c = a.copy();
        c.assign(b);
        REQUIRE(c == tensor_type{{5.1+6.2i,6.2+5.1i},{5.1+6.2i,6.2+5.1i}});
    }
    SECTION("broadcast_assign_plus")
    {
        auto c = a.copy();
        c+=b;
        REQUIRE(tensor_close(c,tensor_type{{6.2+8.4i,8.4+6.2i},{8.4+10.6i,10.6+8.4i}}));
    }
    SECTION("broadcast_assign_minus")
    {
        auto c = a.copy();
        c-=b;
        REQUIRE(tensor_close(c,tensor_type{{-4.0-4.0i,-4.0-4.0i},{-1.8-1.8i,-1.8-1.8i}}));
    }
    SECTION("broadcast_assign_mul")
    {
        auto c = a.copy();
        c*=b;
        REQUIRE(tensor_close(c,tensor_type{{-8.03+18.04i,8.03+18.04i},{-10.45+42.9i,10.45+42.9i}}));
    }
    SECTION("broadcast_assign_div")
    {
        auto c = a.copy();
        c/=b;
        REQUIRE(tensor_close(c,tensor_type{{0.29868115+0.06826998i,0.29868115-0.06826998i},{0.68440652+0.03072149i,0.68440652-0.03072149i}},1E-6,1E-6));
    }
}


