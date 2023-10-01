/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <limits>
#include <vector>
#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "statistic.hpp"
#include "tensor.hpp"

namespace test_statistic_big_{

//make pseudo random sequence on doubles in range [0,9]
template<typename Layout>
auto make_test_tensor(Layout){
    auto res = gtensor::tensor<double,Layout>(std::vector<int>{100,4,115,10,115});
    helpers_for_testing::generate_lehmer(res.begin(),res.end(),[](const auto& e){return e%10;},123);
    return res;
}

//make pseudo random sequence on doubles in range [0,9], with 10% nans
template<typename Layout>
auto make_nan_test_tensor(Layout){
    auto res = gtensor::tensor<double,Layout>(std::vector<int>{100,4,115,10,115});
    helpers_for_testing::generate_lehmer(res.begin(),res.end(),
        [](const auto& e){
            const auto e_ = e%10;
            return e_ == 3 ? gtensor::math::numeric_traits<double>::nan() : e_;
        }
        ,123
    );
    return res;
}

//make pseudo random sequence on doubles in range [0,2] of size n
template<typename Size>
auto make_average_weights(const Size& n){
    auto res = std::vector<double>(static_cast<const std::size_t&>(n));
    helpers_for_testing::generate_lehmer(res.begin(),res.end(),[](const auto& e){return e%3;},456);
    return res;
}

}   //end of namespace test_statistic_big_

//default policy
TEMPLATE_TEST_CASE("test_statistic_big_flatten_default_policy","[test_statistic]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using layout = TestType;
    using tensor_type = gtensor::tensor<double>;
    using test_statistic_big_::make_test_tensor;
    using test_statistic_big_::make_average_weights;

    const auto test_tensor = make_test_tensor(layout{});

    REQUIRE(tensor_close(ptp(test_tensor), tensor_type(9.0),1E-3,1E-3));
    REQUIRE(tensor_close(mean(test_tensor), tensor_type(4.499614),1E-7,1E-7));
    REQUIRE(tensor_close(nanmean(test_tensor), tensor_type(4.499614),1E-7,1E-7));
    REQUIRE(tensor_close(var(test_tensor), tensor_type(8.248117),1E-7,1E-7));
    REQUIRE(tensor_close(nanvar(test_tensor), tensor_type(8.248117),1E-7,1E-7));
    REQUIRE(tensor_close(stdev(test_tensor), tensor_type(2.871953),1E-7,1E-7));
    REQUIRE(tensor_close(nanstdev(test_tensor), tensor_type(2.871953),1E-7,1E-7));
    REQUIRE(tensor_close(median(test_tensor), tensor_type(4.0),1E-3,1E-3));
    REQUIRE(tensor_close(nanmedian(test_tensor), tensor_type(4.0),1E-3,1E-3));
    REQUIRE(tensor_close(quantile(test_tensor,0.3), tensor_type(2.0),1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(test_tensor,0.3), tensor_type(2.0),1E-3,1E-3));
    REQUIRE(tensor_close(quantile(test_tensor,0.7), tensor_type(6.0),1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(test_tensor,0.7), tensor_type(6.0),1E-3,1E-3));

    auto w = make_average_weights(test_tensor.size());
    REQUIRE(tensor_close(average(test_tensor,w), tensor_type(4.499629),1E-7,1E-7));
}

TEMPLATE_TEST_CASE("test_statistic_big_over_axes_default_policy","[test_statistic]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using layout = TestType;
    using tensor_type = gtensor::tensor<double>;
    using test_statistic_big_::make_test_tensor;
    using test_statistic_big_::make_average_weights;
    //{100,4,115,10,115}
    const auto test_tensor = make_test_tensor(layout{});

    REQUIRE(tensor_close(ptp(test_tensor,{0,2,4}), tensor_type{{9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0},{9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0},{9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0},{9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0}},1E-3,1E-3));
    REQUIRE(tensor_close(mean(test_tensor,{0,2,4}), tensor_type{{4.49864348,4.49797278,4.50191078,4.49879698,4.49532174,4.49552968,4.49910775,4.49906767,4.50333233,4.49570888},{4.50212476,4.5019259,4.49804008,4.50218223,4.49724612,4.50229792,4.49809149,4.49957202,4.49795992,4.50103214},{4.49988355,4.49463138,4.50227902,4.49682798,4.5029225,4.50279471,4.49780567,4.50180038,4.49875388,4.49728922},{4.50297089,4.5021603,4.50185558,4.49803403,4.49885066,4.50279093,4.50239924,4.50147977,4.49670548,4.49646276}},1E-7,1E-7));
    REQUIRE(tensor_close(nanmean(test_tensor,{0,2,4}), tensor_type{{4.49864348,4.49797278,4.50191078,4.49879698,4.49532174,4.49552968,4.49910775,4.49906767,4.50333233,4.49570888},{4.50212476,4.5019259,4.49804008,4.50218223,4.49724612,4.50229792,4.49809149,4.49957202,4.49795992,4.50103214},{4.49988355,4.49463138,4.50227902,4.49682798,4.5029225,4.50279471,4.49780567,4.50180038,4.49875388,4.49728922},{4.50297089,4.5021603,4.50185558,4.49803403,4.49885066,4.50279093,4.50239924,4.50147977,4.49670548,4.49646276}},1E-7,1E-7));
    REQUIRE(tensor_close(var(test_tensor,{0,2,4}), tensor_type{{8.25021593,8.25200874,8.25059975,8.25155621,8.23534371,8.25987642,8.24542605,8.24083769,8.24584372,8.24809274},{8.24226618,8.2459237,8.25222225,8.23699562,8.23707748,8.24523858,8.25085836,8.25672647,8.25614329,8.24323599},{8.24643401,8.25203091,8.25018082,8.24959221,8.24835214,8.2502644,8.24934339,8.25344477,8.24855724,8.24016883},{8.25098475,8.25171178,8.25349145,8.24530653,8.24944216,8.24741981,8.24859084,8.2571721,8.24474605,8.24073078}},1E-7,1E-7));
    REQUIRE(tensor_close(nanvar(test_tensor,{0,2,4}), tensor_type{{8.25021593,8.25200874,8.25059975,8.25155621,8.23534371,8.25987642,8.24542605,8.24083769,8.24584372,8.24809274},{8.24226618,8.2459237,8.25222225,8.23699562,8.23707748,8.24523858,8.25085836,8.25672647,8.25614329,8.24323599},{8.24643401,8.25203091,8.25018082,8.24959221,8.24835214,8.2502644,8.24934339,8.25344477,8.24855724,8.24016883},{8.25098475,8.25171178,8.25349145,8.24530653,8.24944216,8.24741981,8.24859084,8.2571721,8.24474605,8.24073078}},1E-7,1E-7));
    REQUIRE(tensor_close(stdev(test_tensor,{0,2,4}), tensor_type{{2.87231891,2.87263098,2.87238572,2.87255221,2.86972886,2.87400007,2.87148499,2.87068593,2.87155772,2.87194929},{2.87093472,2.87157164,2.87266814,2.87001666,2.87003092,2.87145235,2.87243074,2.87345201,2.87335053,2.87110362},{2.8716605,2.87263484,2.8723128,2.87221033,2.87199445,2.87232735,2.87216702,2.87288092,2.87203016,2.87056943},{2.87245274,2.87257929,2.87288904,2.87146418,2.87218421,2.87183213,2.87203601,2.87352955,2.87136658,2.87066731}},1E-7,1E-7));
    REQUIRE(tensor_close(nanstdev(test_tensor,{0,2,4}), tensor_type{{2.87231891,2.87263098,2.87238572,2.87255221,2.86972886,2.87400007,2.87148499,2.87068593,2.87155772,2.87194929},{2.87093472,2.87157164,2.87266814,2.87001666,2.87003092,2.87145235,2.87243074,2.87345201,2.87335053,2.87110362},{2.8716605,2.87263484,2.8723128,2.87221033,2.87199445,2.87232735,2.87216702,2.87288092,2.87203016,2.87056943},{2.87245274,2.87257929,2.87288904,2.87146418,2.87218421,2.87183213,2.87203601,2.87352955,2.87136658,2.87066731}},1E-7,1E-7));
    REQUIRE(tensor_close(median(test_tensor,{0,2,4}), tensor_type{{4.0,4.0,5.0,5.0,4.0,4.0,4.0,4.0,5.0,4.0},{5.0,5.0,4.0,5.0,4.0,5.0,5.0,4.0,4.0,5.0},{5.0,4.0,5.0,4.0,5.0,5.0,4.0,5.0,4.0,4.0},{5.0,5.0,5.0,4.0,4.0,5.0,5.0,5.0,4.0,4.0}},1E-3,1E-3));
    REQUIRE(tensor_close(nanmedian(test_tensor,{0,2,4}), tensor_type{{4.0,4.0,5.0,5.0,4.0,4.0,4.0,4.0,5.0,4.0},{5.0,5.0,4.0,5.0,4.0,5.0,5.0,4.0,4.0,5.0},{5.0,4.0,5.0,4.0,5.0,5.0,4.0,5.0,4.0,4.0},{5.0,5.0,5.0,4.0,4.0,5.0,5.0,5.0,4.0,4.0}},1E-3,1E-3));
    REQUIRE(tensor_close(quantile(test_tensor,{0,2,4},0.3), tensor_type{{2.0,2.0,3.0,2.0,2.0,2.0,3.0,3.0,3.0,2.0},{3.0,3.0,2.0,3.0,2.0,3.0,2.0,2.0,2.0,3.0},{3.0,2.0,3.0,2.0,3.0,3.0,2.0,3.0,2.0,2.0},{3.0,3.0,2.0,3.0,2.0,3.0,3.0,2.0,2.0,2.0}},1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(test_tensor,{0,2,4},0.3), tensor_type{{2.0,2.0,3.0,2.0,2.0,2.0,3.0,3.0,3.0,2.0},{3.0,3.0,2.0,3.0,2.0,3.0,2.0,2.0,2.0,3.0},{3.0,2.0,3.0,2.0,3.0,3.0,2.0,3.0,2.0,2.0},{3.0,3.0,2.0,3.0,2.0,3.0,3.0,2.0,2.0,2.0}},1E-3,1E-3));
    REQUIRE(tensor_close(quantile(test_tensor,{0,2,4},0.7), tensor_type{{6.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,7.0,6.0},{7.0,7.0,6.0,6.0,6.0,7.0,6.0,7.0,7.0,7.0},{6.0,6.0,7.0,6.0,6.0,7.0,6.0,7.0,7.0,6.0},{7.0,7.0,7.0,7.0,6.0,7.0,7.0,7.0,6.0,6.0}},1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(test_tensor,{0,2,4},0.7), tensor_type{{6.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,7.0,6.0},{7.0,7.0,6.0,6.0,6.0,7.0,6.0,7.0,7.0,7.0},{6.0,6.0,7.0,6.0,6.0,7.0,6.0,7.0,7.0,6.0},{7.0,7.0,7.0,7.0,6.0,7.0,7.0,7.0,6.0,6.0}},1E-3,1E-3));

    auto w = make_average_weights(100*115*115);
    REQUIRE(tensor_close(average(test_tensor,{0,2,4},w), tensor_type{{4.50213151,4.49986914,4.50302934,4.50049771,4.4945729,4.49695553,4.49550174,4.50198553,4.49875498,4.4978632},{4.5010771,4.5034862,4.50020952,4.50246659,4.49736322,4.50174878,4.49959457,4.49663104,4.49610232,4.49344134},{4.49832157,4.49558722,4.5017934,4.49740634,4.5057637,4.50440673,4.49840326,4.50419419,4.50148858,4.49798573},{4.50242877,4.50273813,4.50089708,4.49392921,4.5004788,4.4985326,4.50366017,4.50070949,4.49723464,4.4950464}},1E-7,1E-7));
}

TEMPLATE_TEST_CASE("test_statistic_nan_big_flatten_default_policy","[test_statistic]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using layout = TestType;
    using tensor_type = gtensor::tensor<double>;
    using test_statistic_big_::make_nan_test_tensor;
    using test_statistic_big_::make_average_weights;

    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto nan_test_tensor = make_nan_test_tensor(layout{});

    REQUIRE(tensor_equal(ptp(nan_test_tensor), tensor_type(nan),true));
    REQUIRE(tensor_equal(mean(nan_test_tensor), tensor_type(nan),true));
    REQUIRE(tensor_equal(var(nan_test_tensor), tensor_type(nan),true));
    REQUIRE(tensor_equal(stdev(nan_test_tensor), tensor_type(nan),true));
    REQUIRE(tensor_equal(median(nan_test_tensor), tensor_type(nan),true));
    REQUIRE(tensor_equal(quantile(nan_test_tensor,0.3), tensor_type(nan),true));
    REQUIRE(tensor_equal(quantile(nan_test_tensor,0.7), tensor_type(nan),true));

    auto w = make_average_weights(nan_test_tensor.size());
    REQUIRE(tensor_equal(average(nan_test_tensor,w), tensor_type(nan),true));

    REQUIRE(tensor_close(nanmean(nan_test_tensor), tensor_type(4.666207),1E-7,1E-7,true));
    REQUIRE(tensor_close(nanvar(nan_test_tensor), tensor_type(8.886827),1E-7,1E-7,true));
    REQUIRE(tensor_close(nanstdev(nan_test_tensor), tensor_type(2.981078),1E-7,1E-7,true));
    REQUIRE(tensor_close(nanmedian(nan_test_tensor), tensor_type(5.0),1E-3,1E-3,true));
    REQUIRE(tensor_close(nanquantile(nan_test_tensor,0.3), tensor_type(2.0),1E-3,1E-3,true));
    REQUIRE(tensor_close(nanquantile(nan_test_tensor,0.7), tensor_type(7.0),1E-3,1E-3,true));
}

TEMPLATE_TEST_CASE("test_statistic_nan_big_over_axes_default_policy","[test_statistic]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using layout = TestType;
    using tensor_type = gtensor::tensor<double>;
    using test_statistic_big_::make_nan_test_tensor;
    using test_statistic_big_::make_average_weights;
    //{100,4,115,10,115}
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto nan_test_tensor = make_nan_test_tensor(layout{});

    REQUIRE(tensor_equal(ptp(nan_test_tensor,{0,2,4}), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(mean(nan_test_tensor,{0,2,4}), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(var(nan_test_tensor,{0,2,4}), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(stdev(nan_test_tensor,{0,2,4}), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(median(nan_test_tensor,{0,2,4}), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(quantile(nan_test_tensor,{0,2,4},0.3), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(quantile(nan_test_tensor,{0,2,4},0.7), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));

    auto w = make_average_weights(100*115*115);
    REQUIRE(tensor_equal(average(nan_test_tensor,{0,2,4},w), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));

    REQUIRE(tensor_close(nanmean(nan_test_tensor,{0,2,4}), tensor_type{{4.66497757,4.66457223,4.66906039,4.66460274,4.66144766,4.66201103,4.66536186,4.6657693,4.67025418,4.66150511},{4.66985666,4.66914031,4.66473513,4.66878011,4.66393533,4.66944854,4.66392681,4.66569293,4.66456074,4.66712436},{4.66677954,4.66091503,4.66920171,4.66329172,4.66988021,4.6690245,4.66529745,4.66890545,4.66569492,4.6632243},{4.66979089,4.66898287,4.6679983,4.66483613,4.66578989,4.67026306,4.6689497,4.66706013,4.66309132,4.66254504}},1E-7,1E-7));
    REQUIRE(tensor_close(nanvar(nan_test_tensor,{0,2,4}), tensor_type{{8.88896295,8.89245235,8.88983699,8.8883913,8.87425706,8.90266643,8.88298739,8.87956154,8.88261549,8.88690542},{8.88253488,8.88486554,8.89298873,8.8724954,8.87675562,8.88358163,8.88827175,8.89469096,8.89706177,8.87847025},{8.88585761,8.89391898,8.88825533,8.89015988,8.88585153,8.88541568,8.8929007,8.89292161,8.88926284,8.87738728},{8.88823591,8.88968248,8.88940788,8.88570131,8.89016416,8.88679585,8.88503495,8.89172504,8.8845836,8.87919447}},1E-7,1E-7));
    REQUIRE(tensor_close(nanstdev(nan_test_tensor,{0,2,4}), tensor_type{{2.98143639,2.98202152,2.98158297,2.98134052,2.97896913,2.98373364,2.98043409,2.97985931,2.9803717,2.98109131},{2.98035818,2.98074916,2.98211146,2.97867343,2.97938846,2.98053378,2.98132047,2.98239685,2.98279429,2.9796762},{2.98091557,2.98226742,2.98131772,2.98163711,2.98091455,2.98084144,2.9820967,2.9821002,2.98148668,2.97949447},{2.98131446,2.98155706,2.98151101,2.98088935,2.98163783,2.98107294,2.98077757,2.98189957,2.98070186,2.97979772}},1E-7,1E-7));
    REQUIRE(tensor_close(nanmedian(nan_test_tensor,{0,2,4}), tensor_type{{5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0},{5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0},{5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0},{5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0}},1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(nan_test_tensor,{0,2,4},0.3), tensor_type{{2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0},{2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0},{2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0},{2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0}},1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(nan_test_tensor,{0,2,4},0.7), tensor_type{{7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0},{7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0},{7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0},{7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0}},1E-3,1E-3));
}

//exec_pol<4>
TEMPLATE_TEST_CASE("test_statistic_big_flatten_policy","[test_statistic]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using layout = TestType;
    using tensor_type = gtensor::tensor<double>;
    using test_statistic_big_::make_test_tensor;
    using test_statistic_big_::make_average_weights;

    const auto test_tensor = make_test_tensor(layout{});
    const auto pol = multithreading::exec_pol<4>{};

    REQUIRE(tensor_close(ptp(pol,test_tensor), tensor_type(9.0),1E-3,1E-3));
    REQUIRE(tensor_close(mean(pol,test_tensor), tensor_type(4.499614),1E-7,1E-7));
    REQUIRE(tensor_close(nanmean(pol,test_tensor), tensor_type(4.499614),1E-7,1E-7));
    REQUIRE(tensor_close(var(pol,test_tensor), tensor_type(8.248117),1E-7,1E-7));
    REQUIRE(tensor_close(nanvar(pol,test_tensor), tensor_type(8.248117),1E-7,1E-7));
    REQUIRE(tensor_close(stdev(pol,test_tensor), tensor_type(2.871953),1E-7,1E-7));
    REQUIRE(tensor_close(nanstdev(pol,test_tensor), tensor_type(2.871953),1E-7,1E-7));
    REQUIRE(tensor_close(median(pol,test_tensor), tensor_type(4.0),1E-3,1E-3));
    REQUIRE(tensor_close(nanmedian(pol,test_tensor), tensor_type(4.0),1E-3,1E-3));
    REQUIRE(tensor_close(quantile(pol,test_tensor,0.3), tensor_type(2.0),1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(pol,test_tensor,0.3), tensor_type(2.0),1E-3,1E-3));
    REQUIRE(tensor_close(quantile(pol,test_tensor,0.7), tensor_type(6.0),1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(pol,test_tensor,0.7), tensor_type(6.0),1E-3,1E-3));

    auto w = make_average_weights(test_tensor.size());
    REQUIRE(tensor_close(average(pol,test_tensor,w), tensor_type(4.499629),1E-7,1E-7));
}

TEMPLATE_TEST_CASE("test_statistic_big_over_axes_policy","[test_statistic]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using layout = TestType;
    using tensor_type = gtensor::tensor<double>;
    using test_statistic_big_::make_test_tensor;
    using test_statistic_big_::make_average_weights;
    //{100,4,115,10,115}
    const auto test_tensor = make_test_tensor(layout{});
    const auto pol = multithreading::exec_pol<4>{};

    REQUIRE(tensor_close(ptp(pol,test_tensor,{0,2,4}), tensor_type{{9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0},{9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0},{9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0},{9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0,9.0}},1E-3,1E-3));
    REQUIRE(tensor_close(mean(pol,test_tensor,{0,2,4}), tensor_type{{4.49864348,4.49797278,4.50191078,4.49879698,4.49532174,4.49552968,4.49910775,4.49906767,4.50333233,4.49570888},{4.50212476,4.5019259,4.49804008,4.50218223,4.49724612,4.50229792,4.49809149,4.49957202,4.49795992,4.50103214},{4.49988355,4.49463138,4.50227902,4.49682798,4.5029225,4.50279471,4.49780567,4.50180038,4.49875388,4.49728922},{4.50297089,4.5021603,4.50185558,4.49803403,4.49885066,4.50279093,4.50239924,4.50147977,4.49670548,4.49646276}},1E-7,1E-7));
    REQUIRE(tensor_close(nanmean(pol,test_tensor,{0,2,4}), tensor_type{{4.49864348,4.49797278,4.50191078,4.49879698,4.49532174,4.49552968,4.49910775,4.49906767,4.50333233,4.49570888},{4.50212476,4.5019259,4.49804008,4.50218223,4.49724612,4.50229792,4.49809149,4.49957202,4.49795992,4.50103214},{4.49988355,4.49463138,4.50227902,4.49682798,4.5029225,4.50279471,4.49780567,4.50180038,4.49875388,4.49728922},{4.50297089,4.5021603,4.50185558,4.49803403,4.49885066,4.50279093,4.50239924,4.50147977,4.49670548,4.49646276}},1E-7,1E-7));
    REQUIRE(tensor_close(var(pol,test_tensor,{0,2,4}), tensor_type{{8.25021593,8.25200874,8.25059975,8.25155621,8.23534371,8.25987642,8.24542605,8.24083769,8.24584372,8.24809274},{8.24226618,8.2459237,8.25222225,8.23699562,8.23707748,8.24523858,8.25085836,8.25672647,8.25614329,8.24323599},{8.24643401,8.25203091,8.25018082,8.24959221,8.24835214,8.2502644,8.24934339,8.25344477,8.24855724,8.24016883},{8.25098475,8.25171178,8.25349145,8.24530653,8.24944216,8.24741981,8.24859084,8.2571721,8.24474605,8.24073078}},1E-7,1E-7));
    REQUIRE(tensor_close(nanvar(pol,test_tensor,{0,2,4}), tensor_type{{8.25021593,8.25200874,8.25059975,8.25155621,8.23534371,8.25987642,8.24542605,8.24083769,8.24584372,8.24809274},{8.24226618,8.2459237,8.25222225,8.23699562,8.23707748,8.24523858,8.25085836,8.25672647,8.25614329,8.24323599},{8.24643401,8.25203091,8.25018082,8.24959221,8.24835214,8.2502644,8.24934339,8.25344477,8.24855724,8.24016883},{8.25098475,8.25171178,8.25349145,8.24530653,8.24944216,8.24741981,8.24859084,8.2571721,8.24474605,8.24073078}},1E-7,1E-7));
    REQUIRE(tensor_close(stdev(pol,test_tensor,{0,2,4}), tensor_type{{2.87231891,2.87263098,2.87238572,2.87255221,2.86972886,2.87400007,2.87148499,2.87068593,2.87155772,2.87194929},{2.87093472,2.87157164,2.87266814,2.87001666,2.87003092,2.87145235,2.87243074,2.87345201,2.87335053,2.87110362},{2.8716605,2.87263484,2.8723128,2.87221033,2.87199445,2.87232735,2.87216702,2.87288092,2.87203016,2.87056943},{2.87245274,2.87257929,2.87288904,2.87146418,2.87218421,2.87183213,2.87203601,2.87352955,2.87136658,2.87066731}},1E-7,1E-7));
    REQUIRE(tensor_close(nanstdev(pol,test_tensor,{0,2,4}), tensor_type{{2.87231891,2.87263098,2.87238572,2.87255221,2.86972886,2.87400007,2.87148499,2.87068593,2.87155772,2.87194929},{2.87093472,2.87157164,2.87266814,2.87001666,2.87003092,2.87145235,2.87243074,2.87345201,2.87335053,2.87110362},{2.8716605,2.87263484,2.8723128,2.87221033,2.87199445,2.87232735,2.87216702,2.87288092,2.87203016,2.87056943},{2.87245274,2.87257929,2.87288904,2.87146418,2.87218421,2.87183213,2.87203601,2.87352955,2.87136658,2.87066731}},1E-7,1E-7));
    REQUIRE(tensor_close(median(pol,test_tensor,{0,2,4}), tensor_type{{4.0,4.0,5.0,5.0,4.0,4.0,4.0,4.0,5.0,4.0},{5.0,5.0,4.0,5.0,4.0,5.0,5.0,4.0,4.0,5.0},{5.0,4.0,5.0,4.0,5.0,5.0,4.0,5.0,4.0,4.0},{5.0,5.0,5.0,4.0,4.0,5.0,5.0,5.0,4.0,4.0}},1E-3,1E-3));
    REQUIRE(tensor_close(nanmedian(pol,test_tensor,{0,2,4}), tensor_type{{4.0,4.0,5.0,5.0,4.0,4.0,4.0,4.0,5.0,4.0},{5.0,5.0,4.0,5.0,4.0,5.0,5.0,4.0,4.0,5.0},{5.0,4.0,5.0,4.0,5.0,5.0,4.0,5.0,4.0,4.0},{5.0,5.0,5.0,4.0,4.0,5.0,5.0,5.0,4.0,4.0}},1E-3,1E-3));
    REQUIRE(tensor_close(quantile(pol,test_tensor,{0,2,4},0.3), tensor_type{{2.0,2.0,3.0,2.0,2.0,2.0,3.0,3.0,3.0,2.0},{3.0,3.0,2.0,3.0,2.0,3.0,2.0,2.0,2.0,3.0},{3.0,2.0,3.0,2.0,3.0,3.0,2.0,3.0,2.0,2.0},{3.0,3.0,2.0,3.0,2.0,3.0,3.0,2.0,2.0,2.0}},1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(pol,test_tensor,{0,2,4},0.3), tensor_type{{2.0,2.0,3.0,2.0,2.0,2.0,3.0,3.0,3.0,2.0},{3.0,3.0,2.0,3.0,2.0,3.0,2.0,2.0,2.0,3.0},{3.0,2.0,3.0,2.0,3.0,3.0,2.0,3.0,2.0,2.0},{3.0,3.0,2.0,3.0,2.0,3.0,3.0,2.0,2.0,2.0}},1E-3,1E-3));
    REQUIRE(tensor_close(quantile(pol,test_tensor,{0,2,4},0.7), tensor_type{{6.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,7.0,6.0},{7.0,7.0,6.0,6.0,6.0,7.0,6.0,7.0,7.0,7.0},{6.0,6.0,7.0,6.0,6.0,7.0,6.0,7.0,7.0,6.0},{7.0,7.0,7.0,7.0,6.0,7.0,7.0,7.0,6.0,6.0}},1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(pol,test_tensor,{0,2,4},0.7), tensor_type{{6.0,7.0,7.0,6.0,6.0,6.0,6.0,6.0,7.0,6.0},{7.0,7.0,6.0,6.0,6.0,7.0,6.0,7.0,7.0,7.0},{6.0,6.0,7.0,6.0,6.0,7.0,6.0,7.0,7.0,6.0},{7.0,7.0,7.0,7.0,6.0,7.0,7.0,7.0,6.0,6.0}},1E-3,1E-3));

    auto w = make_average_weights(100*115*115);
    REQUIRE(tensor_close(average(pol,test_tensor,{0,2,4},w), tensor_type{{4.50213151,4.49986914,4.50302934,4.50049771,4.4945729,4.49695553,4.49550174,4.50198553,4.49875498,4.4978632},{4.5010771,4.5034862,4.50020952,4.50246659,4.49736322,4.50174878,4.49959457,4.49663104,4.49610232,4.49344134},{4.49832157,4.49558722,4.5017934,4.49740634,4.5057637,4.50440673,4.49840326,4.50419419,4.50148858,4.49798573},{4.50242877,4.50273813,4.50089708,4.49392921,4.5004788,4.4985326,4.50366017,4.50070949,4.49723464,4.4950464}},1E-7,1E-7));
}

TEMPLATE_TEST_CASE("test_statistic_nan_big_flatten_policy","[test_statistic]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using layout = TestType;
    using tensor_type = gtensor::tensor<double>;
    using test_statistic_big_::make_nan_test_tensor;
    using test_statistic_big_::make_average_weights;

    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto nan_test_tensor = make_nan_test_tensor(layout{});
    const auto pol = multithreading::exec_pol<4>{};

    REQUIRE(tensor_equal(ptp(pol,nan_test_tensor), tensor_type(nan),true));
    REQUIRE(tensor_equal(mean(pol,nan_test_tensor), tensor_type(nan),true));
    REQUIRE(tensor_equal(var(pol,nan_test_tensor), tensor_type(nan),true));
    REQUIRE(tensor_equal(stdev(pol,nan_test_tensor), tensor_type(nan),true));
    REQUIRE(tensor_equal(median(pol,nan_test_tensor), tensor_type(nan),true));
    REQUIRE(tensor_equal(quantile(pol,nan_test_tensor,0.3), tensor_type(nan),true));
    REQUIRE(tensor_equal(quantile(pol,nan_test_tensor,0.7), tensor_type(nan),true));

    auto w = make_average_weights(nan_test_tensor.size());
    REQUIRE(tensor_equal(average(pol,nan_test_tensor,w), tensor_type(nan),true));

    REQUIRE(tensor_close(nanmean(pol,nan_test_tensor), tensor_type(4.666207),1E-7,1E-7,true));
    REQUIRE(tensor_close(nanvar(pol,nan_test_tensor), tensor_type(8.886827),1E-7,1E-7,true));
    REQUIRE(tensor_close(nanstdev(pol,nan_test_tensor), tensor_type(2.981078),1E-7,1E-7,true));
    REQUIRE(tensor_close(nanmedian(pol,nan_test_tensor), tensor_type(5.0),1E-3,1E-3,true));
    REQUIRE(tensor_close(nanquantile(pol,nan_test_tensor,0.3), tensor_type(2.0),1E-3,1E-3,true));
    REQUIRE(tensor_close(nanquantile(pol,nan_test_tensor,0.7), tensor_type(7.0),1E-3,1E-3,true));
}

TEMPLATE_TEST_CASE("test_statistic_nan_big_over_axes_policy","[test_statistic]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using layout = TestType;
    using tensor_type = gtensor::tensor<double>;
    using test_statistic_big_::make_nan_test_tensor;
    using test_statistic_big_::make_average_weights;
    //{100,4,115,10,115}
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const auto nan_test_tensor = make_nan_test_tensor(layout{});
    const auto pol = multithreading::exec_pol<4>{};

    REQUIRE(tensor_equal(ptp(pol,nan_test_tensor,{0,2,4}), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(mean(pol,nan_test_tensor,{0,2,4}), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(var(pol,nan_test_tensor,{0,2,4}), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(stdev(pol,nan_test_tensor,{0,2,4}), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(median(pol,nan_test_tensor,{0,2,4}), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(quantile(pol,nan_test_tensor,{0,2,4},0.3), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));
    REQUIRE(tensor_equal(quantile(pol,nan_test_tensor,{0,2,4},0.7), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));

    auto w = make_average_weights(100*115*115);
    REQUIRE(tensor_equal(average(pol,nan_test_tensor,{0,2,4},w), tensor_type{{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan},{nan,nan,nan,nan,nan,nan,nan,nan,nan,nan}},true));

    REQUIRE(tensor_close(nanmean(pol,nan_test_tensor,{0,2,4}), tensor_type{{4.66497757,4.66457223,4.66906039,4.66460274,4.66144766,4.66201103,4.66536186,4.6657693,4.67025418,4.66150511},{4.66985666,4.66914031,4.66473513,4.66878011,4.66393533,4.66944854,4.66392681,4.66569293,4.66456074,4.66712436},{4.66677954,4.66091503,4.66920171,4.66329172,4.66988021,4.6690245,4.66529745,4.66890545,4.66569492,4.6632243},{4.66979089,4.66898287,4.6679983,4.66483613,4.66578989,4.67026306,4.6689497,4.66706013,4.66309132,4.66254504}},1E-7,1E-7));
    REQUIRE(tensor_close(nanvar(pol,nan_test_tensor,{0,2,4}), tensor_type{{8.88896295,8.89245235,8.88983699,8.8883913,8.87425706,8.90266643,8.88298739,8.87956154,8.88261549,8.88690542},{8.88253488,8.88486554,8.89298873,8.8724954,8.87675562,8.88358163,8.88827175,8.89469096,8.89706177,8.87847025},{8.88585761,8.89391898,8.88825533,8.89015988,8.88585153,8.88541568,8.8929007,8.89292161,8.88926284,8.87738728},{8.88823591,8.88968248,8.88940788,8.88570131,8.89016416,8.88679585,8.88503495,8.89172504,8.8845836,8.87919447}},1E-7,1E-7));
    REQUIRE(tensor_close(nanstdev(pol,nan_test_tensor,{0,2,4}), tensor_type{{2.98143639,2.98202152,2.98158297,2.98134052,2.97896913,2.98373364,2.98043409,2.97985931,2.9803717,2.98109131},{2.98035818,2.98074916,2.98211146,2.97867343,2.97938846,2.98053378,2.98132047,2.98239685,2.98279429,2.9796762},{2.98091557,2.98226742,2.98131772,2.98163711,2.98091455,2.98084144,2.9820967,2.9821002,2.98148668,2.97949447},{2.98131446,2.98155706,2.98151101,2.98088935,2.98163783,2.98107294,2.98077757,2.98189957,2.98070186,2.97979772}},1E-7,1E-7));
    REQUIRE(tensor_close(nanmedian(pol,nan_test_tensor,{0,2,4}), tensor_type{{5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0},{5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0},{5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0},{5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0,5.0}},1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(pol,nan_test_tensor,{0,2,4},0.3), tensor_type{{2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0},{2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0},{2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0},{2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0}},1E-3,1E-3));
    REQUIRE(tensor_close(nanquantile(pol,nan_test_tensor,{0,2,4},0.7), tensor_type{{7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0},{7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0},{7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0},{7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0,7.0}},1E-3,1E-3));
}

