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
#include "statistic.hpp"
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
    SECTION("broadcast_assign_view")
    {
        auto c = a.copy();
        c({{},{1}}) = -1.1;
        REQUIRE(tensor_close(c,tensor_type{{1.1+2.2i,-1.1+0.0i},{3.3+4.4i,-1.1+0.0i}},1E-6,1E-6));
    }
}

TEST_CASE("test_std_complex_broadcast_routines","[test_std_complex]")
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

TEST_CASE("test_std_complex_routines","[test_std_complex]")
{
    using gtensor::tensor;
    using value_type = std::complex<double>;
    using tensor_type = gtensor::tensor<value_type>;
    using namespace std::complex_literals;

    const tensor_type a{{{1.1+2.2i,2.2+1.1i,1.5+0.3i},{3.3+4.4i,4.4+3.3i,0.2+0.7i}},{{1.6+2.3i,2.1+1.2i,1.9+0.3i},{3.5+4.1i,4.4+3.0i,0.2+1.7i}}};

    //sum
    REQUIRE(tensor_close(a.sum(),tensor_type(26.400000000000002+24.6i)));
    REQUIRE(tensor_close(a.sum(-1),tensor_type{{4.8+3.6i,7.9+8.4i},{5.6+3.8i,8.1+8.8i}}));
    REQUIRE(tensor_close(a.sum({0,-1}),tensor_type{10.4+7.4i,16.0+17.2i}));
    //cumsum
    REQUIRE(tensor_close(a.cumsum(),tensor_type{1.1+2.2i,3.3+3.3i,4.8+3.6i,8.1+8.0i,12.5+11.3i,12.7+12.0i,14.3+14.3i,16.4+15.5i,18.3+15.8i,21.8+19.9i,26.2+22.9i,26.4+24.6i}));
    REQUIRE(tensor_close(a.cumsum(-1),tensor_type{{{1.1+2.2i,3.3+3.3i,4.8+3.6i},{3.3+4.4i,7.7+7.7i,7.9+8.4i}},{{1.6+2.3i,3.7+3.5i,5.6+3.8i},{3.5+4.1i,7.9+7.1i,8.1+8.8i}}}));
    //prod
    REQUIRE(tensor_close(a.prod(),tensor_type(-126861.3528684975+30811.523198692543i)));
    REQUIRE(tensor_close(a.prod(-1),tensor_type{{-1.815+9.075i,-21.175+6.05i},{-0.885+13.005i,-47.898+10.978i}}));
    REQUIRE(tensor_close(a.prod({0,-1}),tensor_type{-116.4141-31.63545i,947.82325-522.24205i}));
    //cumprod
    REQUIRE(tensor_close(
        a.cumprod(),
        tensor_type{1.10000000e+00+2.20000000e+00i,0.00000000e+00+6.05000000e+00i,-1.81500000e+00+9.07500000e+00i,-4.59195000e+01+2.19615000e+01i,-2.74518750e+02-5.49037500e+01i,-1.64711250e+01-2.03143875e+02i,4.40877113e+02-3.62913788e+02i,1.36133848e+03-2.33066419e+02i,2.65646304e+03-3.44246513e+01i,9.43876171e+03+1.07710122e+04i,9.21751497e+03+7.57087387e+04i,-1.26861353e+05+3.08115232e+04i},
        1E-6,
        1E-6
    ));
    REQUIRE(tensor_close(a.cumprod(-1),tensor_type{{{1.1+2.2i,0.0+6.05i,-1.815+9.075i},{3.3+4.4i,0.0+30.25i,-21.175+6.05i}},{{1.6+2.3i,0.6+6.75i,-0.885+13.005i},{3.5+4.1i,3.1+28.54i,-47.898+10.978i}}}));
    //diff
    REQUIRE(tensor_close(diff(a),tensor_type{{{1.1-1.1i,-0.7-0.8i},{1.1-1.1i,-4.2-2.6i}},{{0.5-1.1i,-0.2-0.9i},{0.9-1.1i,-4.2-1.3i}}}));
    //gradient
    REQUIRE(tensor_close(gradient(a,-1),tensor_type{{{1.1-1.1i,0.2-0.95i,-0.7-0.8i},{1.1-1.1i,-1.55-1.85i,-4.2-2.6i}},{{0.5-1.1i,0.15-1.0i,-0.2-0.9i},{0.9-1.1i,-1.65-1.2i,-4.2-1.3i}}}));
    //matmul
    REQUIRE(tensor_close(matmul(a,tensor_type{{1.1+2.2i,2.2+1.1i},{3.3+4.4i,4.4+3.3i},{2.1+2.2i,2.1+1.1i}}),
        tensor_type{{{1.28+22.08i,8.87+20.43i},{-7.17+44.26i,10.54+44.04i}},{{1.68+24.06i,9.93+21.75i},{-7.17+45.48i,11.2+44.38i}}}
    ));

    //mean
    REQUIRE(tensor_close(a.mean(),tensor_type(2.2+2.05i)));
    REQUIRE(tensor_close(a.mean(-1),tensor_type{{1.6+1.2i,2.63333333+2.8i},{1.86666667+1.26666667i,2.7+2.93333333i}},1E-6,1E-6));
    REQUIRE(tensor_close(a.mean({0,-1}),tensor_type{1.73333333+1.23333333i,2.66666667+2.86666667i},1E-6,1E-6));
    //var
    REQUIRE(tensor_close(a.var(),tensor<double>(3.7258333333333336),1E-6,1E-6));
    REQUIRE(tensor_close(a.var(-1),tensor<double>{{0.81333333,5.56888889},{0.71111111,4.22222222}},1E-6,1E-6));
    REQUIRE(tensor_close(a.var({0,-1}),tensor<double>{0.78111111,4.90111111},1E-6,1E-6));
    //stdev
    REQUIRE(tensor_close(a.stdev(),tensor<double>(1.9302417810557655),1E-6,1E-6));
    REQUIRE(tensor_close(a.stdev(-1),tensor<double>{{0.90184995,2.35984934},{0.84327404,2.05480467}},1E-6,1E-6));
    REQUIRE(tensor_close(a.stdev({0,-1}),tensor<double>{0.88380491,2.21384532},1E-6,1E-6));
}

TEMPLATE_TEST_CASE("test_std_complex_routines_policy","[test_std_complex]",
    multithreading::exec_pol<4>
)
{
    using policy = TestType;
    using gtensor::tensor;
    using value_type = std::complex<double>;
    using tensor_type = gtensor::tensor<value_type>;
    using namespace std::complex_literals;

    const tensor_type a{{{1.1+2.2i,2.2+1.1i,1.5+0.3i},{3.3+4.4i,4.4+3.3i,0.2+0.7i}},{{1.6+2.3i,2.1+1.2i,1.9+0.3i},{3.5+4.1i,4.4+3.0i,0.2+1.7i}}};

    //sum
    REQUIRE(tensor_close(a.sum(policy{}),tensor_type(26.400000000000002+24.6i)));
    REQUIRE(tensor_close(a.sum(policy{},-1),tensor_type{{4.8+3.6i,7.9+8.4i},{5.6+3.8i,8.1+8.8i}}));
    REQUIRE(tensor_close(a.sum(policy{},{0,-1}),tensor_type{10.4+7.4i,16.0+17.2i}));
    //cumsum
    REQUIRE(tensor_close(a.cumsum(policy{}),tensor_type{1.1+2.2i,3.3+3.3i,4.8+3.6i,8.1+8.0i,12.5+11.3i,12.7+12.0i,14.3+14.3i,16.4+15.5i,18.3+15.8i,21.8+19.9i,26.2+22.9i,26.4+24.6i}));
    REQUIRE(tensor_close(a.cumsum(policy{},-1),tensor_type{{{1.1+2.2i,3.3+3.3i,4.8+3.6i},{3.3+4.4i,7.7+7.7i,7.9+8.4i}},{{1.6+2.3i,3.7+3.5i,5.6+3.8i},{3.5+4.1i,7.9+7.1i,8.1+8.8i}}}));
    //prod
    REQUIRE(tensor_close(a.prod(policy{}),tensor_type(-126861.3528684975+30811.523198692543i)));
    REQUIRE(tensor_close(a.prod(policy{},-1),tensor_type{{-1.815+9.075i,-21.175+6.05i},{-0.885+13.005i,-47.898+10.978i}}));
    REQUIRE(tensor_close(a.prod(policy{},{0,-1}),tensor_type{-116.4141-31.63545i,947.82325-522.24205i}));
    //cumprod
    REQUIRE(tensor_close(
        a.cumprod(policy{}),
        tensor_type{1.10000000e+00+2.20000000e+00i,0.00000000e+00+6.05000000e+00i,-1.81500000e+00+9.07500000e+00i,-4.59195000e+01+2.19615000e+01i,-2.74518750e+02-5.49037500e+01i,-1.64711250e+01-2.03143875e+02i,4.40877113e+02-3.62913788e+02i,1.36133848e+03-2.33066419e+02i,2.65646304e+03-3.44246513e+01i,9.43876171e+03+1.07710122e+04i,9.21751497e+03+7.57087387e+04i,-1.26861353e+05+3.08115232e+04i},
        1E-6,
        1E-6
    ));
    REQUIRE(tensor_close(a.cumprod(policy{},-1),tensor_type{{{1.1+2.2i,0.0+6.05i,-1.815+9.075i},{3.3+4.4i,0.0+30.25i,-21.175+6.05i}},{{1.6+2.3i,0.6+6.75i,-0.885+13.005i},{3.5+4.1i,3.1+28.54i,-47.898+10.978i}}}));
    //diff
    REQUIRE(tensor_close(diff(policy{},a),tensor_type{{{1.1-1.1i,-0.7-0.8i},{1.1-1.1i,-4.2-2.6i}},{{0.5-1.1i,-0.2-0.9i},{0.9-1.1i,-4.2-1.3i}}}));
    //gradient
    REQUIRE(tensor_close(gradient(policy{},a,-1),tensor_type{{{1.1-1.1i,0.2-0.95i,-0.7-0.8i},{1.1-1.1i,-1.55-1.85i,-4.2-2.6i}},{{0.5-1.1i,0.15-1.0i,-0.2-0.9i},{0.9-1.1i,-1.65-1.2i,-4.2-1.3i}}}));

    //mean
    REQUIRE(tensor_close(a.mean(policy{}),tensor_type(2.2+2.05i)));
    REQUIRE(tensor_close(a.mean(policy{},-1),tensor_type{{1.6+1.2i,2.63333333+2.8i},{1.86666667+1.26666667i,2.7+2.93333333i}},1E-6,1E-6));
    REQUIRE(tensor_close(a.mean(policy{},{0,-1}),tensor_type{1.73333333+1.23333333i,2.66666667+2.86666667i},1E-6,1E-6));
    //var
    REQUIRE(tensor_close(a.var(policy{}),tensor<double>(3.7258333333333336),1E-6,1E-6));
    REQUIRE(tensor_close(a.var(policy{},-1),tensor<double>{{0.81333333,5.56888889},{0.71111111,4.22222222}},1E-6,1E-6));
    REQUIRE(tensor_close(a.var(policy{},{0,-1}),tensor<double>{0.78111111,4.90111111},1E-6,1E-6));
    //stdev
    REQUIRE(tensor_close(a.stdev(policy{}),tensor<double>(1.9302417810557655),1E-6,1E-6));
    REQUIRE(tensor_close(a.stdev(policy{},-1),tensor<double>{{0.90184995,2.35984934},{0.84327404,2.05480467}},1E-6,1E-6));
    REQUIRE(tensor_close(a.stdev(policy{},{0,-1}),tensor<double>{0.88380491,2.21384532},1E-6,1E-6));
}

TEST_CASE("test_std_complex_routines_nan_values","[test_std_complex]")
{
    using gtensor::tensor;
    using value_type = std::complex<double>;
    using tensor_type = gtensor::tensor<value_type>;
    using namespace std::complex_literals;
    static constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    const tensor_type a{{{1.1+2.2i,2.2+1.1i,3.2+0.1i},{nan+2.2i,2.1+2.3i,0.2+2.3i},{0.1+1.2i,1.1+2.2i,1.2+2.1i},{2.1+2.2i,1.2+1.1i,1.2+0.1i}},
        {{0.1+1.2i,1.1+nan*2.2i,1.2+2.1i},{1.1+2.2i,2.2+1.1i,3.2+0.1i},{0.2+2.2i,2.1+2.3i,0.2+2.3i},{1.1+2.2i,2.2+1.1i,nan+0.1i}},
        {{2.1+2.2i,1.2+1.1i,1.2+0.1i},{0.1+1.2i,1.1+2.2i,1.2+2.1i},{nan+nan*2.2i,nan+nan*1.1i,nan+nan*0.1i},{0.2+2.2i,2.1+2.3i,0.2+2.3i}}
    };

    //sum
    REQUIRE(tensor_equal(a.sum(),tensor_type(nan+nan*1.0i),true));
    REQUIRE(tensor_close(a.sum(-1),tensor_type{{6.5+3.4i,nan+6.8i,2.4+5.5i,4.5+3.4i},{nan+nan*1.0i,6.5+3.4i,2.5+6.8i,nan+3.4i},{4.5+3.4i,2.4+5.5i,nan+nan*1.0i,2.5+6.8i}},true));
    REQUIRE(tensor_close(a.sum({0,-1}),tensor_type{nan+nan*1.0i, nan+15.7i, nan+nan*1.0i, nan+13.6i},true));
    //cumsum
    REQUIRE(tensor_close(a.cumsum(),tensor_type{1.1+2.2i,3.3+3.3i,6.5+3.4i,nan+5.6i,nan+7.9i,nan+10.2i,nan+11.4i,nan+13.6i,nan+15.7i,nan+17.9i,nan+19.0i,nan+19.1i,nan+20.3i,
        nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,
        nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},true)
    );
    REQUIRE(tensor_close(a.cumsum(-1),tensor_type{{{1.1+2.2i,3.3+3.3i,6.5+3.4i},{nan+2.2i,nan+4.5i,nan+6.8i},{0.1+1.2i,1.2+3.4i,2.4+5.5i},{2.1+2.2i,3.3+3.3i,4.5+3.4i}},
        {{0.1+1.2i,nan+nan*1.0i,nan+nan*1.0i},{1.1+2.2i,3.3+3.3i,6.5+3.4i},{0.2+2.2i,2.3+4.5i,2.5+6.8i},{1.1+2.2i,3.3+3.3i,nan+3.4i}},
        {{2.1+2.2i,3.3+3.3i,4.5+3.4i},{0.1+1.2i,1.2+3.4i,2.4+5.5i},{nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},{0.2+2.2i,2.3+4.5i,2.5+6.8i}}},true)
    );
    //prod
    REQUIRE(tensor_equal(a.prod(),tensor_type(nan+nan*1.0i),true));
    REQUIRE(tensor_close(a.prod(-1),tensor_type{{-0.605+19.36i,nan+nan*1.0i,-6.27-3.465i,-0.375+5.95i},{nan+nan*1.0i,-0.605+19.36i,-12.612-9.656i,nan+nan*1.0i},
        {-0.375+5.95i,-6.27-3.465i,nan+nan*1.0i,-12.612-9.656i}},true)
    );
    REQUIRE(tensor_close(a.prod({0,-1}),tensor_type{nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},true));
    //cumprod
    REQUIRE(tensor_close(a.cumprod(),tensor_type{1.1+2.2i,0.0+6.05i,-0.605+19.36i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,
        nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,
        nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},true)
    );
    REQUIRE(tensor_close(a.cumprod(-1),tensor_type{{{1.1+2.2i,0.0+6.05i,-0.605+19.36i},{nan+2.2i,nan+nan*1.0i,nan+nan*1.0i},{0.1+1.2i,-2.53+1.54i,-6.27-3.465i},
        {2.1+2.2i,0.1+4.95i,-0.375+5.95i}},{{0.1+1.2i,nan+nan*1.0i,nan+nan*1.0i},{1.1+2.2i,0.0+6.05i,-0.605+19.36i},{0.2+2.2i,-4.64+5.08i,-12.612-9.656i},
        {1.1+2.2i,0.0+6.05i,nan+nan*1.0i}},{{2.1+2.2i,0.1+4.95i,-0.375+5.95i},{0.1+1.2i,-2.53+1.54i,-6.27-3.465i},{nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},
        {0.2+2.2i,-4.64+5.08i,-12.612-9.656i}}},true)
    );
    //mean
    REQUIRE(tensor_equal(a.mean(),tensor_type(nan+nan*1.0i),true));
    REQUIRE(tensor_close(a.mean(-1),tensor_type{{2.16666667+1.13333333i,nan+nan*1.0i,0.8+1.83333333i,1.5+1.13333333i},
        {nan+nan*1.0i,2.16666667+1.13333333i,0.83333333+2.26666667i,nan+nan*1.0i},{1.5+1.13333333i,0.8+1.83333333i,nan+nan*1.0i,0.83333333+2.26666667i}},1E-6,1E-6,true)
    );
    REQUIRE(tensor_close(a.mean({0,-1}),tensor_type{nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},1E-6,1E-6,true));
    //var
    REQUIRE(tensor_equal(a.var(),tensor<double>(nan),true));
    REQUIRE(tensor_close(a.var(-1),tensor<double>{{1.47111111,nan,0.44888889,0.91555556},{nan,1.47111111,0.80444444,nan},{0.91555556,0.44888889,nan,0.80444444}},1E-6,1E-6,true));
    REQUIRE(tensor_equal(a.var({0,-1}),tensor<double>{nan,nan,nan,nan},true));
    REQUIRE(std::is_same_v<decltype(var(a)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(var(a,-1)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(var(a,{0,-1})),tensor<double>>);
    //stdev
    REQUIRE(tensor_equal(a.stdev(),tensor<double>(nan),true));
    REQUIRE(tensor_close(a.stdev(-1),tensor<double>{{1.21289369,nan,0.66999171,0.95684667},{nan,1.21289369,0.89690827,nan},{0.95684667,0.66999171,nan,0.89690827}},1E-6,1E-6,true));
    REQUIRE(tensor_equal(a.stdev({0,-1}),tensor<double>{nan,nan,nan,nan},true));
    REQUIRE(std::is_same_v<decltype(stdev(a)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(stdev(a,-1)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(stdev(a,{0,-1})),tensor<double>>);

    //nan versions
    //nansum
    REQUIRE(tensor_close(nansum(a),tensor_type(38.7+49.4i)));
    REQUIRE(tensor_close(nansum(a,-1),tensor_type{{6.5+3.4i,2.3+4.6i,2.4+5.5i,4.5+3.4i},{1.3+3.3i,6.5+3.4i,2.5+6.8i,3.3+3.3i},{4.5+3.4i,2.4+5.5i,0.0+0.0i,2.5+6.8i}}));
    REQUIRE(tensor_close(nansum(a,{0,-1}),tensor_type{12.3+10.1i,11.2+13.5i,4.9+12.3i,10.3+13.5i}));
    //nancumsum
    REQUIRE(tensor_close(nancumsum(a),tensor_type{1.1+2.2i,3.3+3.3i,6.5+3.4i,6.5+3.4i,8.6+5.7i,8.8+8.0i,8.9+9.2i,10.0+11.4i,11.2+13.5i,13.3+15.7i,14.5+16.8i,15.7+16.9i,15.8+18.1i,
        15.8+18.1i,17.0+20.2i,18.1+22.4i,20.3+23.5i,23.5+23.6i,23.7+25.8i,25.8+28.1i,26.0+30.4i,27.1+32.6i,29.3+33.7i,29.3+33.7i,31.4+35.9i,32.6+37.0i,33.8+37.1i,33.9+38.3i,
        35.0+40.5i,36.2+42.6i,36.2+42.6i,36.2+42.6i,36.2+42.6i,36.4+44.8i,38.5+47.1i,38.7+49.4i},1E-6,1E-6)
    );
    REQUIRE(tensor_close(nancumsum(a,-1),tensor_type{{{1.1+2.2i,3.3+3.3i,6.5+3.4i},{0.0+0.0i,2.1+2.3i,2.3+4.6i},{0.1+1.2i,1.2+3.4i,2.4+5.5i},{2.1+2.2i,3.3+3.3i,4.5+3.4i}},
        {{0.1+1.2i,0.1+1.2i,1.3+3.3i},{1.1+2.2i,3.3+3.3i,6.5+3.4i},{0.2+2.2i,2.3+4.5i,2.5+6.8i},{1.1+2.2i,3.3+3.3i,3.3+3.3i}},
        {{2.1+2.2i,3.3+3.3i,4.5+3.4i},{0.1+1.2i,1.2+3.4i,2.4+5.5i},{0.0+0.0i,0.0+0.0i,0.0+0.0i},{0.2+2.2i,2.3+4.5i,2.5+6.8i}}},1E-6,1E-6)
    );
    //nanprod
    REQUIRE(tensor_close(nanprod(a),tensor_type(-19251035287.499603+10388868193.627138i),1E-6,1E-6));
    REQUIRE(tensor_close(nanprod(a,-1),tensor_type{{-0.605+19.36i,-4.87+5.29i,-6.27-3.465i,-0.375+5.95i},{-2.4+1.65i,-0.605+19.36i,-12.612-9.656i,0.0+6.05i},
        {-0.375+5.95i,-6.27-3.465i,1.0+0.0i,-12.612-9.656i}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(nanprod(a,{0,-1}),tensor_type{293.8348875-163.62905625i,285.88382625+955.87927875i,45.6192+104.2437i,432.09342+376.205335i},1E-6,1E-6));
    //nancumprod
    REQUIRE(tensor_close(nancumprod(a),tensor_type{1.10000000e+00+2.20000000e+00i,0.00000000e+00+6.05000000e+00i,-6.05000000e-01+1.93600000e+01i,-6.05000000e-01+1.93600000e+01i,
        -4.57985000e+01+3.92645000e+01i,-9.94680500e+01-9.74836500e+01i,1.07033575e+02-1.29110025e+02i,4.01778988e+02+9.34528375e+01i,2.85883826e+02+9.55879279e+02i,
        -1.50257838e+03+2.63629090e+03i,-4.70301405e+03+1.51071287e+03i,-5.79468814e+03+1.34255404e+03i,-2.19053366e+03-6.81937037e+03i,-2.19053366e+03-6.81937037e+03i,
        1.16920374e+04-1.27833651e+04i,4.09846444e+04+1.16607806e+04i,7.73393590e+04+7.07368262e+04i,2.40412266e+05+2.34091780e+05i,-4.66919462e+05+5.75725342e+05i,
        -2.30469916e+06+1.35108455e+05i,-7.71689277e+05-5.27378637e+06i,1.07534718e+07-7.49888141e+06i,3.19064075e+07-4.66872013e+06i,3.19064075e+07-4.66872013e+06i,
        7.72746401e+07+6.03897843e+07i,2.63008054e+07+1.57469845e+08i,1.58139819e+07+1.91593895e+08i,-2.28331276e+08+3.81361678e+07i,-3.35063972e+08-4.60379022e+08i,
        5.64719179e+08-1.25608917e+09i,5.64719179e+08-1.25608917e+09i,5.64719179e+08-1.25608917e+09i,5.64719179e+08-1.25608917e+09i,2.87634000e+09+9.91164360e+08i,
        3.76063598e+09+8.69702717e+09i,-1.92510353e+10+1.03888682e+10i},1E-6,1E-6)
    );
    REQUIRE(tensor_close(nancumprod(a,-1),tensor_type{{{1.1+2.2i,0.0+6.05i,-0.605+19.36i},{1.0+0.0i,2.1+2.3i,-4.87+5.29i},{0.1+1.2i,-2.53+1.54i,-6.27-3.465i},
        {2.1+2.2i,0.1+4.95i,-0.375+5.95i}},{{0.1+1.2i,0.1+1.2i,-2.4+1.65i},{1.1+2.2i,0.0+6.05i,-0.605+19.36i},{0.2+2.2i,-4.64+5.08i,-12.612-9.656i},{1.1+2.2i,0.0+6.05i,0.0+6.05i}},
        {{2.1+2.2i,0.1+4.95i,-0.375+5.95i},{0.1+1.2i,-2.53+1.54i,-6.27-3.465i},{1.0+0.0i,1.0+0.0i,1.0+0.0i},{0.2+2.2i,-4.64+5.08i,-12.612-9.656i}}},1E-6,1E-6)
    );
    //nanmean
    REQUIRE(tensor_close(nanmean(a),tensor_type(1.29+1.6466666666666665i),1E-6,1E-6));
    REQUIRE(tensor_close(nanmean(a,-1),tensor_type{{2.16666667+1.13333333i,1.15+2.3i,0.8+1.83333333i,1.5+1.13333333i},
        {0.65+1.65i,2.16666667+1.13333333i,0.83333333+2.26666667i,1.65+1.65i},{1.5+1.13333333i,0.8+1.83333333i,nan+nan*1.0i,0.83333333+2.26666667i}},1E-6,1E-6,true)
    );
    REQUIRE(tensor_close(nanmean(a,{0,-1}),tensor_type{1.5375+1.2625i,1.4+1.6875i,0.81666667+2.05i,1.2875+1.6875i},1E-6,1E-6));
    //nanvar
    REQUIRE(tensor_close(nanvar(a),tensor<double>(1.3693888888888888),1E-6,1E-6));
    REQUIRE(tensor_close(nanvar(a,-1),tensor<double>{{1.47111111,0.9025,0.44888889,0.91555556},{0.505,1.47111111,0.80444444,0.605},{0.91555556,0.44888889,nan,0.80444444}},1E-6,1E-6,true));
    REQUIRE(tensor_close(nanvar(a,{0,-1}),tensor<double>{1.4171875,1.53359375,0.67388889,1.1646875},1E-6,1E-6));
    REQUIRE(std::is_same_v<decltype(nanvar(a)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(nanvar(a,-1)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(nanvar(a,{0,-1})),tensor<double>>);
    //nanstdev
    REQUIRE(tensor_close(nanstdev(a),tensor<double>(1.1702089082248899),1E-6,1E-6));
    REQUIRE(tensor_close(nanstdev(a,-1),tensor<double>{{1.21289369,0.95,0.66999171,0.95684667},{0.71063352,1.21289369,0.89690827,0.77781746},{0.95684667,0.66999171,nan,0.89690827}},1E-6,1E-6,true));
    REQUIRE(tensor_close(nanstdev(a,{0,-1}),tensor<double>{1.19045685,1.23838352,0.82090736,1.07920688},1E-6,1E-6));
    REQUIRE(std::is_same_v<decltype(nanstdev(a)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(nanstdev(a,-1)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(nanstdev(a,{0,-1})),tensor<double>>);


    //diff
    REQUIRE(tensor_close(diff(a),tensor_type{{{1.1-1.1i,1.0-1.0i},{nan+0.1i,-1.9+0.0i},{1.0+1.0i,0.1-0.1i},{-0.9-1.1i,0.0-1.0i}},
        {{nan+nan*1.0i,nan+nan*1.0i},{1.1-1.1i,1.0-1.0i},{1.9+0.1i,-1.9+0.0i},{1.1-1.1i,nan-1.0i}},
        {{-0.9-1.1i,0.0-1.0i},{1.0+1.0i,0.1-0.1i},{nan+nan*1.0i,nan+nan*1.0i},{1.9+0.1i,-1.9+0.0i}}},true)
    );
    //gradient
    REQUIRE(tensor_close(gradient(a,-1),tensor_type{{{1.1-1.1i,1.05-1.05i,1.0-1.0i},{nan+nan*1.0i,nan+nan*1.0i,-1.9+0.0i},{1.0+1.0i,0.55+0.45i,0.1-0.1i},{-0.9-1.1i,-0.45-1.05i,0.0-1.0i}},
        {{nan+nan*1.0i,0.55+0.45i,nan+nan*1.0i},{1.1-1.1i,1.05-1.05i,1.0-1.0i},{1.9+0.1i,0.0+0.05i,-1.9+0.0i},{1.1-1.1i,nan+nan*1.0i,nan+nan*1.0i}},
        {{-0.9-1.1i,-0.45-1.05i,0.0-1.0i},{1.0+1.0i,0.55+0.45i,0.1-0.1i},{nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},{1.9+0.1i,0.0+0.05i,-1.9+0.0i}}},true)
    );
    //matmul
    REQUIRE(tensor_close(matmul(a,tensor_type{{1.1+2.2i,2.2+1.1i},{3.3+4.4i,4.4+3.3i},{2.1+2.2i,2.1+1.1i}}),tensor_type{{{5.29+25.4i,12.66+21.88i},{nan+nan*1.0i,nan+nan*1.0i},
        {-10.68+20.69i,-3.31+21.79i},{-1.11+18.8i,6.26+17.48i}},{{nan+nan*1.0i,nan+nan*1.0i},{5.29+25.4i,12.66+21.88i},{-12.45+24.96i,-2.44+27.16i},{nan+nan*1.0i,nan+nan*1.0i}},
        {{-1.11+18.8i,6.26+17.48i},{-10.68+20.69i,-3.31+21.79i},{nan+nan*1.0i,nan+nan*1.0i},{-12.45+24.96i,-2.44+27.16i}}},true)
    );
}

TEMPLATE_TEST_CASE("test_std_complex_routines_nan_values_policy","[test_std_complex]",
    multithreading::exec_pol<4>
)
{
    using policy = TestType;
    using gtensor::tensor;
    using value_type = std::complex<double>;
    using tensor_type = gtensor::tensor<value_type>;
    using namespace std::complex_literals;
    static constexpr auto nan = std::numeric_limits<double>::quiet_NaN();

    const tensor_type a{{{1.1+2.2i,2.2+1.1i,3.2+0.1i},{nan+2.2i,2.1+2.3i,0.2+2.3i},{0.1+1.2i,1.1+2.2i,1.2+2.1i},{2.1+2.2i,1.2+1.1i,1.2+0.1i}},
        {{0.1+1.2i,1.1+nan*2.2i,1.2+2.1i},{1.1+2.2i,2.2+1.1i,3.2+0.1i},{0.2+2.2i,2.1+2.3i,0.2+2.3i},{1.1+2.2i,2.2+1.1i,nan+0.1i}},
        {{2.1+2.2i,1.2+1.1i,1.2+0.1i},{0.1+1.2i,1.1+2.2i,1.2+2.1i},{nan+nan*2.2i,nan+nan*1.1i,nan+nan*0.1i},{0.2+2.2i,2.1+2.3i,0.2+2.3i}}
    };

    //sum
    REQUIRE(tensor_equal(a.sum(policy{}),tensor_type(nan+nan*1.0i),true));
    REQUIRE(tensor_close(a.sum(policy{},-1),tensor_type{{6.5+3.4i,nan+6.8i,2.4+5.5i,4.5+3.4i},{nan+nan*1.0i,6.5+3.4i,2.5+6.8i,nan+3.4i},{4.5+3.4i,2.4+5.5i,nan+nan*1.0i,2.5+6.8i}},true));
    REQUIRE(tensor_close(a.sum(policy{},{0,-1}),tensor_type{nan+nan*1.0i, nan+15.7i, nan+nan*1.0i, nan+13.6i},true));
    //cumsum
    REQUIRE(tensor_close(a.cumsum(policy{}),tensor_type{1.1+2.2i,3.3+3.3i,6.5+3.4i,nan+5.6i,nan+7.9i,nan+10.2i,nan+11.4i,nan+13.6i,nan+15.7i,nan+17.9i,nan+19.0i,nan+19.1i,nan+20.3i,
        nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,
        nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},true)
    );
    REQUIRE(tensor_close(a.cumsum(policy{},-1),tensor_type{{{1.1+2.2i,3.3+3.3i,6.5+3.4i},{nan+2.2i,nan+4.5i,nan+6.8i},{0.1+1.2i,1.2+3.4i,2.4+5.5i},{2.1+2.2i,3.3+3.3i,4.5+3.4i}},
        {{0.1+1.2i,nan+nan*1.0i,nan+nan*1.0i},{1.1+2.2i,3.3+3.3i,6.5+3.4i},{0.2+2.2i,2.3+4.5i,2.5+6.8i},{1.1+2.2i,3.3+3.3i,nan+3.4i}},
        {{2.1+2.2i,3.3+3.3i,4.5+3.4i},{0.1+1.2i,1.2+3.4i,2.4+5.5i},{nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},{0.2+2.2i,2.3+4.5i,2.5+6.8i}}},true)
    );
    //prod
    REQUIRE(tensor_equal(a.prod(policy{}),tensor_type(nan+nan*1.0i),true));
    REQUIRE(tensor_close(a.prod(policy{},-1),tensor_type{{-0.605+19.36i,nan+nan*1.0i,-6.27-3.465i,-0.375+5.95i},{nan+nan*1.0i,-0.605+19.36i,-12.612-9.656i,nan+nan*1.0i},
        {-0.375+5.95i,-6.27-3.465i,nan+nan*1.0i,-12.612-9.656i}},true)
    );
    REQUIRE(tensor_close(a.prod(policy{},{0,-1}),tensor_type{nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},true));
    //cumprod
    REQUIRE(tensor_close(a.cumprod(policy{}),tensor_type{1.1+2.2i,0.0+6.05i,-0.605+19.36i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,
        nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,
        nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},true)
    );
    REQUIRE(tensor_close(a.cumprod(policy{},-1),tensor_type{{{1.1+2.2i,0.0+6.05i,-0.605+19.36i},{nan+2.2i,nan+nan*1.0i,nan+nan*1.0i},{0.1+1.2i,-2.53+1.54i,-6.27-3.465i},
        {2.1+2.2i,0.1+4.95i,-0.375+5.95i}},{{0.1+1.2i,nan+nan*1.0i,nan+nan*1.0i},{1.1+2.2i,0.0+6.05i,-0.605+19.36i},{0.2+2.2i,-4.64+5.08i,-12.612-9.656i},
        {1.1+2.2i,0.0+6.05i,nan+nan*1.0i}},{{2.1+2.2i,0.1+4.95i,-0.375+5.95i},{0.1+1.2i,-2.53+1.54i,-6.27-3.465i},{nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},
        {0.2+2.2i,-4.64+5.08i,-12.612-9.656i}}},true)
    );
    //mean
    REQUIRE(tensor_equal(a.mean(policy{}),tensor_type(nan+nan*1.0i),true));
    REQUIRE(tensor_close(a.mean(policy{},-1),tensor_type{{2.16666667+1.13333333i,nan+nan*1.0i,0.8+1.83333333i,1.5+1.13333333i},
        {nan+nan*1.0i,2.16666667+1.13333333i,0.83333333+2.26666667i,nan+nan*1.0i},{1.5+1.13333333i,0.8+1.83333333i,nan+nan*1.0i,0.83333333+2.26666667i}},1E-6,1E-6,true)
    );
    REQUIRE(tensor_close(a.mean(policy{},{0,-1}),tensor_type{nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i,nan+nan*1.0i},1E-6,1E-6,true));
    //var
    REQUIRE(tensor_equal(a.var(policy{}),tensor<double>(nan),true));
    REQUIRE(tensor_close(a.var(policy{},-1),tensor<double>{{1.47111111,nan,0.44888889,0.91555556},{nan,1.47111111,0.80444444,nan},{0.91555556,0.44888889,nan,0.80444444}},1E-6,1E-6,true));
    REQUIRE(tensor_equal(a.var(policy{},{0,-1}),tensor<double>{nan,nan,nan,nan},true));
    REQUIRE(std::is_same_v<decltype(var(policy{},a)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(var(policy{},a,-1)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(var(policy{},a,{0,-1})),tensor<double>>);
    //stdev
    REQUIRE(tensor_equal(a.stdev(policy{}),tensor<double>(nan),true));
    REQUIRE(tensor_close(a.stdev(policy{},-1),tensor<double>{{1.21289369,nan,0.66999171,0.95684667},{nan,1.21289369,0.89690827,nan},{0.95684667,0.66999171,nan,0.89690827}},1E-6,1E-6,true));
    REQUIRE(tensor_equal(a.stdev(policy{},{0,-1}),tensor<double>{nan,nan,nan,nan},true));
    REQUIRE(std::is_same_v<decltype(stdev(policy{},a)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(stdev(policy{},a,-1)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(stdev(policy{},a,{0,-1})),tensor<double>>);

    //nan versions
    //nansum
    REQUIRE(tensor_close(nansum(policy{},a),tensor_type(38.7+49.4i)));
    REQUIRE(tensor_close(nansum(policy{},a,-1),tensor_type{{6.5+3.4i,2.3+4.6i,2.4+5.5i,4.5+3.4i},{1.3+3.3i,6.5+3.4i,2.5+6.8i,3.3+3.3i},{4.5+3.4i,2.4+5.5i,0.0+0.0i,2.5+6.8i}}));
    REQUIRE(tensor_close(nansum(policy{},a,{0,-1}),tensor_type{12.3+10.1i,11.2+13.5i,4.9+12.3i,10.3+13.5i}));
    //nancumsum
    REQUIRE(tensor_close(nancumsum(policy{},a),tensor_type{1.1+2.2i,3.3+3.3i,6.5+3.4i,6.5+3.4i,8.6+5.7i,8.8+8.0i,8.9+9.2i,10.0+11.4i,11.2+13.5i,13.3+15.7i,14.5+16.8i,15.7+16.9i,15.8+18.1i,
        15.8+18.1i,17.0+20.2i,18.1+22.4i,20.3+23.5i,23.5+23.6i,23.7+25.8i,25.8+28.1i,26.0+30.4i,27.1+32.6i,29.3+33.7i,29.3+33.7i,31.4+35.9i,32.6+37.0i,33.8+37.1i,33.9+38.3i,
        35.0+40.5i,36.2+42.6i,36.2+42.6i,36.2+42.6i,36.2+42.6i,36.4+44.8i,38.5+47.1i,38.7+49.4i},1E-6,1E-6)
    );
    REQUIRE(tensor_close(nancumsum(policy{},a,-1),tensor_type{{{1.1+2.2i,3.3+3.3i,6.5+3.4i},{0.0+0.0i,2.1+2.3i,2.3+4.6i},{0.1+1.2i,1.2+3.4i,2.4+5.5i},{2.1+2.2i,3.3+3.3i,4.5+3.4i}},
        {{0.1+1.2i,0.1+1.2i,1.3+3.3i},{1.1+2.2i,3.3+3.3i,6.5+3.4i},{0.2+2.2i,2.3+4.5i,2.5+6.8i},{1.1+2.2i,3.3+3.3i,3.3+3.3i}},
        {{2.1+2.2i,3.3+3.3i,4.5+3.4i},{0.1+1.2i,1.2+3.4i,2.4+5.5i},{0.0+0.0i,0.0+0.0i,0.0+0.0i},{0.2+2.2i,2.3+4.5i,2.5+6.8i}}},1E-6,1E-6)
    );
    //nanprod
    REQUIRE(tensor_close(nanprod(policy{},a),tensor_type(-19251035287.499603+10388868193.627138i),1E-6,1E-6));
    REQUIRE(tensor_close(nanprod(policy{},a,-1),tensor_type{{-0.605+19.36i,-4.87+5.29i,-6.27-3.465i,-0.375+5.95i},{-2.4+1.65i,-0.605+19.36i,-12.612-9.656i,0.0+6.05i},
        {-0.375+5.95i,-6.27-3.465i,1.0+0.0i,-12.612-9.656i}},1E-6,1E-6)
    );
    REQUIRE(tensor_close(nanprod(policy{},a,{0,-1}),tensor_type{293.8348875-163.62905625i,285.88382625+955.87927875i,45.6192+104.2437i,432.09342+376.205335i},1E-6,1E-6));
    //nancumprod
    REQUIRE(tensor_close(nancumprod(policy{},a),tensor_type{1.10000000e+00+2.20000000e+00i,0.00000000e+00+6.05000000e+00i,-6.05000000e-01+1.93600000e+01i,-6.05000000e-01+1.93600000e+01i,
        -4.57985000e+01+3.92645000e+01i,-9.94680500e+01-9.74836500e+01i,1.07033575e+02-1.29110025e+02i,4.01778988e+02+9.34528375e+01i,2.85883826e+02+9.55879279e+02i,
        -1.50257838e+03+2.63629090e+03i,-4.70301405e+03+1.51071287e+03i,-5.79468814e+03+1.34255404e+03i,-2.19053366e+03-6.81937037e+03i,-2.19053366e+03-6.81937037e+03i,
        1.16920374e+04-1.27833651e+04i,4.09846444e+04+1.16607806e+04i,7.73393590e+04+7.07368262e+04i,2.40412266e+05+2.34091780e+05i,-4.66919462e+05+5.75725342e+05i,
        -2.30469916e+06+1.35108455e+05i,-7.71689277e+05-5.27378637e+06i,1.07534718e+07-7.49888141e+06i,3.19064075e+07-4.66872013e+06i,3.19064075e+07-4.66872013e+06i,
        7.72746401e+07+6.03897843e+07i,2.63008054e+07+1.57469845e+08i,1.58139819e+07+1.91593895e+08i,-2.28331276e+08+3.81361678e+07i,-3.35063972e+08-4.60379022e+08i,
        5.64719179e+08-1.25608917e+09i,5.64719179e+08-1.25608917e+09i,5.64719179e+08-1.25608917e+09i,5.64719179e+08-1.25608917e+09i,2.87634000e+09+9.91164360e+08i,
        3.76063598e+09+8.69702717e+09i,-1.92510353e+10+1.03888682e+10i},1E-6,1E-6)
    );
    REQUIRE(tensor_close(nancumprod(policy{},a,-1),tensor_type{{{1.1+2.2i,0.0+6.05i,-0.605+19.36i},{1.0+0.0i,2.1+2.3i,-4.87+5.29i},{0.1+1.2i,-2.53+1.54i,-6.27-3.465i},
        {2.1+2.2i,0.1+4.95i,-0.375+5.95i}},{{0.1+1.2i,0.1+1.2i,-2.4+1.65i},{1.1+2.2i,0.0+6.05i,-0.605+19.36i},{0.2+2.2i,-4.64+5.08i,-12.612-9.656i},{1.1+2.2i,0.0+6.05i,0.0+6.05i}},
        {{2.1+2.2i,0.1+4.95i,-0.375+5.95i},{0.1+1.2i,-2.53+1.54i,-6.27-3.465i},{1.0+0.0i,1.0+0.0i,1.0+0.0i},{0.2+2.2i,-4.64+5.08i,-12.612-9.656i}}},1E-6,1E-6)
    );
    //nanmean
    REQUIRE(tensor_close(nanmean(policy{},a),tensor_type(1.29+1.6466666666666665i),1E-6,1E-6));
    REQUIRE(tensor_close(nanmean(policy{},a,-1),tensor_type{{2.16666667+1.13333333i,1.15+2.3i,0.8+1.83333333i,1.5+1.13333333i},
        {0.65+1.65i,2.16666667+1.13333333i,0.83333333+2.26666667i,1.65+1.65i},{1.5+1.13333333i,0.8+1.83333333i,nan+nan*1.0i,0.83333333+2.26666667i}},1E-6,1E-6,true)
    );
    REQUIRE(tensor_close(nanmean(policy{},a,{0,-1}),tensor_type{1.5375+1.2625i,1.4+1.6875i,0.81666667+2.05i,1.2875+1.6875i},1E-6,1E-6));
    //nanvar
    REQUIRE(tensor_close(nanvar(policy{},a),tensor<double>(1.3693888888888888),1E-6,1E-6));
    REQUIRE(tensor_close(nanvar(policy{},a,-1),tensor<double>{{1.47111111,0.9025,0.44888889,0.91555556},{0.505,1.47111111,0.80444444,0.605},{0.91555556,0.44888889,nan,0.80444444}},1E-6,1E-6,true));
    REQUIRE(tensor_close(nanvar(policy{},a,{0,-1}),tensor<double>{1.4171875,1.53359375,0.67388889,1.1646875},1E-6,1E-6));
    REQUIRE(std::is_same_v<decltype(nanvar(policy{},a)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(nanvar(policy{},a,-1)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(nanvar(policy{},a,{0,-1})),tensor<double>>);
    //nanstdev
    REQUIRE(tensor_close(nanstdev(policy{},a),tensor<double>(1.1702089082248899),1E-6,1E-6));
    REQUIRE(tensor_close(nanstdev(policy{},a,-1),tensor<double>{{1.21289369,0.95,0.66999171,0.95684667},{0.71063352,1.21289369,0.89690827,0.77781746},{0.95684667,0.66999171,nan,0.89690827}},1E-6,1E-6,true));
    REQUIRE(tensor_close(nanstdev(policy{},a,{0,-1}),tensor<double>{1.19045685,1.23838352,0.82090736,1.07920688},1E-6,1E-6));
    REQUIRE(std::is_same_v<decltype(nanstdev(policy{},a)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(nanstdev(policy{},a,-1)),tensor<double>>);
    REQUIRE(std::is_same_v<decltype(nanstdev(policy{},a,{0,-1})),tensor<double>>);
}
