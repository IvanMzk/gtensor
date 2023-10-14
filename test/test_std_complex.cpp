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
#include "tensor_math.hpp"
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

TEST_CASE("test_std_complex_math_routines","[test_std_complex]")
{
    using gtensor::tensor;
    using value_type = std::complex<double>;
    using tensor_type = gtensor::tensor<value_type>;
    using namespace std::complex_literals;

    const tensor_type a{{1.1+2.2i,2.2+1.1i},{3.3+4.4i,4.4+3.3i}};
    const tensor_type b{5.1+6.2i,6.2+5.1i};

    REQUIRE(real(a) == tensor<double>{{1.1,2.2},{3.3,4.4}});
    REQUIRE(imag(a) == tensor<double>{{2.2,1.1},{4.4,3.3}});
    REQUIRE(conj(a) == tensor_type{{1.1-2.2i,2.2-1.1i},{3.3-4.4i,4.4-3.3i}});
    REQUIRE(conjugate(a) == tensor_type{{1.1-2.2i,2.2-1.1i},{3.3-4.4i,4.4-3.3i}});
    REQUIRE(tensor_close(angle(a),tensor<double>{{1.10714872,0.46364761},{0.92729522,0.64350111}},1E-6,1E-6));
    REQUIRE(tensor_close(abs(a),tensor<double>{{2.45967478,2.45967478},{5.5,5.5}},1E-2,1E-2));
    REQUIRE(tensor_close(exp(a),tensor_type{{-1.76795506+2.42885743i,4.09371112+8.04315846i},{-8.33260513-25.80044343i,-80.43108505-12.84852382i}},1E-6,1E-6));
    REQUIRE(tensor_close(log(a),tensor_type{{0.90002914+1.10714872i,0.90002914+0.46364761i},{1.70474809+0.92729522i,1.70474809+0.64350111i}},1E-6,1E-6));
    REQUIRE(tensor_close(log10(a),tensor_type{{0.39087769+0.48082858i,0.39087769+0.2013596i},{0.74036269+0.4027192i,0.74036269+0.27946898i}},1E-6,1E-6));
    REQUIRE(tensor_close(power(a,b),tensor_type{{2.35614152e-02-1.00147626e-01i,9.45538714e+00+2.30539961e+01i},{-1.74399641e+01+7.56584666e+00i,1.45194217e+03+1.71468557e+02i}},1E-6,1E-6));
    REQUIRE(tensor_close(sqrt(a),tensor_type{{1.33410546+0.82452252i,1.52638049+0.36032955i},{2.0976177+1.04880885i,2.22485955+0.74161985i}},1E-6,1E-6));
    //trigonometric
    REQUIRE(tensor_close(sin(a),tensor_type{{4.07095352+2.02172562i,1.34899125-0.78603003i},{-6.42523026-40.20948071i,-12.91777076-4.16063486i}},1E-6,1E-6));
    REQUIRE(tensor_close(cos(a),tensor_type{{2.0719855-3.97220493i,-0.98192503-1.07986618i},{-40.22160434+6.42329356i,-4.17197027+12.88267266i}},1E-6,1E-6));
    REQUIRE(tensor_close(tan(a),tensor_type{{2.01403721e-02+1.01435425i,-2.23350597e-01+1.0461275i},{9.38922767e-05+0.99971357i,1.59492198e-03+1.00220794i}},1E-6,1E-6));
    REQUIRE(tensor_close(arcsin(a),tensor_type{{0.43295487+1.61848776i,1.07156621+1.56934164i},{0.63562351+2.40029428i,0.91930753+2.39566901i}},1E-6,1E-6));
    REQUIRE(tensor_close(arccos(a),tensor_type{{1.13784146-1.61848776i,0.49923012-1.56934164i},{0.93517282-2.40029428i,0.65148879-2.39566901i}},1E-6,1E-6));
    REQUIRE(tensor_close(arctan(a),tensor_type{{1.36537073+0.36585752i,1.2124347+0.16141121i},{1.45983408+0.14470951i,1.42467596+0.10721352i}},1E-6,1E-6));
    //hyperbolic
    REQUIRE(tensor_close(sinh(a),tensor_type{{-0.78603003+1.34899125i,2.02172562+4.07095352i},{-4.16063486-12.91777076i,-40.20948071-6.42523026i}},1E-6,1E-6));
    REQUIRE(tensor_close(cosh(a),tensor_type{{-0.98192503+1.07986618i,2.0719855+3.97220493i},{-4.17197027-12.88267266i,-40.22160434-6.42329356i}},1E-6,1E-6));
    REQUIRE(tensor_close(tanh(a),tensor_type{{1.0461275-2.23350597e-01i,1.01435425+2.01403721e-02i},{1.00220794+1.59492198e-03i,0.99971357+9.38922767e-05i}},1E-6,1E-6));
    REQUIRE(tensor_close(arcsinh(a),tensor_type{{1.56934164+1.07156621i,1.61848776+0.43295487i},{2.39566901+0.91930753i,2.40029428+0.63562351i}},1E-6,1E-6));
    REQUIRE(tensor_close(arccosh(a),tensor_type{{1.61848776+1.13784146i,1.56934164+0.49923012i},{2.40029428+0.93517282i,2.39566901+0.65148879i}},1E-6,1E-6));
    REQUIRE(tensor_close(arctanh(a),tensor_type{{0.16141121+1.2124347i,0.36585752+1.36537073i},{0.10721352+1.42467596i,0.14470951+1.45983408i}},1E-6,1E-6));
}

