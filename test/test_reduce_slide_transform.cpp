/*
* GTensor - matrix computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include <algorithm>
#include "catch.hpp"
#include "builder.hpp"
#include "reduce.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

namespace test_slide_{

struct cumsum{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        auto cumsum_ = *first;
        *dfirst = cumsum_;
        for(++dfirst,++first; dfirst!=dlast; ++dfirst,++first){
            cumsum_+=*first;
            *dfirst = cumsum_;
        }
    }
};

struct cumprod_reverse{
    template<typename It, typename DstIt>
    void operator()(It, It last, DstIt dfirst, DstIt dlast){
        auto cumprod_ = *--last;
        *--dlast = cumprod_;
        while(dlast!=dfirst){
            cumprod_*=*--last;
            *--dlast = cumprod_;
        }
    }
};

struct moving_avarage{
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It, DstIt dfirst, DstIt dlast, const IdxT& window_size, const IdxT& window_step, const typename std::iterator_traits<It>::value_type& denom){
        using index_type = IdxT;
        using value_type = typename std::iterator_traits<It>::value_type;
        value_type sum{0};
        auto it = first;
        for (index_type i{0}; i!=window_size; ++i, ++it){
            sum+=*it;
        }
        for(;dfirst!=dlast;++dfirst){
            *dfirst = sum/denom;
            for (index_type i{0}; i!=window_step; ++i,++first){
                sum-=*first;
            }
            for (index_type i{0}; i!=window_step; ++i, ++it){
                sum+=*it;
            }
        }
    }
};

struct diff_1{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        for (;dfirst!=dlast;++dfirst){
            auto prev = *first;
            *dfirst = *(++first) - prev;
        }
    }
};

struct diff_2{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        for (;dfirst!=dlast;++dfirst){
            auto v0 = *first;
            auto v1 = *(++first);
            auto v2 = *(++first);
            *dfirst = v2-v1-v1+v0;
            --first;
        }
    }
};

struct sort{
    template<typename It>
    void operator()(It first, It last){
        std::sort(first,last);
    }
};

}   //end of namespace test_slide_

TEST_CASE("test_slide","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using test_slide_::cumprod_reverse;
    using test_slide_::cumsum;
    using test_slide_::diff_1;
    using test_slide_::diff_2;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}};

    //0tensor,1axis,2functor,3window_size,4window_step,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, dim_type{0}, cumsum{}, index_type{0}, index_type{1}, tensor_type{}),
        std::make_tuple(tensor_type{}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), dim_type{1}, cumsum{}, index_type{5}, index_type{1}, tensor_type{}.reshape(0,2,3)),
        std::make_tuple(tensor_type{1}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{120,120,60,20,5}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, dim_type{0}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, dim_type{0}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{1,2,3},{5,7,9},{12,15,18}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{1,3,6},{4,9,15},{7,15,24}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{28,80,162},{28,40,54},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{6,6,3},{120,30,6},{504,72,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{-2}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{28,80,162},{28,40,54},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{-1}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{6,6,3},{120,30,6},{504,72,9}}),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{{8,8,28,20,20},{0,0,12,28,4},{32,8,20,0,4},{0,28,28,20,32}},{{12,28,52,48,20},{24,16,16,44,12},{40,12,20,4,8},{24,52,40,44,60}},{{40,52,56,60,48},{32,28,48,44,24},{52,44,44,16,36},{44,84,56,76,80}}}),
        std::make_tuple((test_ten+1)*(test_ten-1), dim_type{1}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{{3,3,48,24,24},{2,2,56,72,24},{65,5,80,71,24},{64,53,128,95,87}},{{0,24,35,48,-1},{35,39,35,63,2},{38,39,34,63,2},{73,74,42,98,50}},{{48,35,0,8,48},{51,43,63,7,56},{59,106,98,15,104},{83,169,113,78,128}}}),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,2)+test_ten(2,3), dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{{11,13,18,19,16},{7,9,10,23,8},{23,13,14,9,8},{7,23,18,19,22}},{{21,29,35,40,27},{20,22,18,43,17},{40,25,23,19,16},{20,45,32,39,43}},{{37,46,47,57,45},{29,34,33,59,27},{58,44,38,31,30},{32,69,47,61,62}}}),
        std::make_tuple((test_ten-test_ten(0))*(test_ten(1,2)+test_ten(2,3)), dim_type{2}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}},{{-7,20,16,34,4},{42,78,70,43,49},{-42,-51,-71,-62,-62},{42,33,17,26,20}},{{35,71,47,29,41},{14,41,61,-2,10},{-35,19,23,50,86},{35,44,32,59,41}}})
    );
    auto test_slide = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto tensor = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto window_size = std::get<3>(t);
            auto window_step = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto result = slide<value_type>(policy..., tensor, axis, functor, window_size, window_step);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    };

    SECTION("default_policy")
    {
        test_slide();
    }
    SECTION("exec_pol<4>")
    {
        test_slide(multithreading::exec_pol<4>{});
    }
}

TEMPLATE_TEST_CASE("test_slide_flatten","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type,order>;
    using index_type = typename tensor_type::index_type;
    using test_slide_::cumprod_reverse;
    using test_slide_::cumsum;
    using test_slide_::diff_1;
    using test_slide_::diff_2;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}};

    //0tensor,1functor,2window_size,3window_step,4expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{1,2,3,4,5}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{120,120,60,20,5}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15,21,28,36,45}),
        std::make_tuple(tensor_type{{1,3,2},{5,7,4},{6,7,8}}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{{1,3,2},{5,7,4},{6,7,8}}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten, cumsum{}, index_type{1}, index_type{1}, tensor_type{8,16,44,64,84,84,84,96,124,128,160,168,188,188,192,192,220,248,268,300,304,324,348,376,376,400,416,420,436,444,452,456,456,460,464,488,512,524,548,576,604,628,632,644,672,680,692,724,724,736,748,780,804,816,844,864,896,912,944,964}),
        std::make_tuple((test_ten+1)*(test_ten-1), cumsum{}, index_type{1}, index_type{1}, tensor_type{3,6,54,78,102,101,100,108,156,156,219,222,246,245,245,244,292,340,364,427,427,451,486,534,533,568,583,583,598,601,604,604,603,603,603,638,673,681,716,764,812,847,847,855,903,906,914,977,976,984,992,1055,1090,1098,1146,1170,1233,1248,1311,1335}),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,2)+test_ten(2,3), cumsum{}, index_type{1}, index_type{1}, tensor_type{11,24,42,61,77,84,93,103,126,134,157,170,184,193,201,208,231,249,268,290,300,316,333,354,365,378,391,399,419,428,445,457,466,476,484,497,519,533,553,574,590,607,619,636,654,663,675,690,706,716,734,753,768,780,794,806,830,845,867,886}),
        std::make_tuple((test_ten-test_ten(0))*(test_ten(1,2)+test_ten(2,3)), cumsum{}, index_type{1}, index_type{1}, tensor_type{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-7,20,16,34,4,46,82,74,47,53,11,2,-18,-9,-9,33,24,8,17,11,46,82,58,40,52,66,93,113,50,62,27,81,85,112,148,183,192,180,207,189})
    );
    auto test_slide_flatten = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto tensor = std::get<0>(t);
            auto functor = std::get<1>(t);
            auto window_size = std::get<2>(t);
            auto window_step = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = slide<value_type>(policy..., tensor, gtensor::detail::no_value{}, functor, window_size, window_step);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    };

    SECTION("default_policy")
    {
        test_slide_flatten();
    }
    SECTION("exec_pol<4>")
    {
        test_slide_flatten(multithreading::exec_pol<4>{});
    }
}

TEST_CASE("test_slide_custom_arg","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using test_slide_::moving_avarage;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2functor,3window_size,4window_step,5denom,6expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10}, dim_type{0}, moving_avarage{}, index_type{3}, index_type{1}, value_type{3}, tensor_type{2,3,4,5,6,7,8,9}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10}, dim_type{0}, moving_avarage{}, index_type{3}, index_type{2}, value_type{3}, tensor_type{2,4,6,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, moving_avarage{}, index_type{2}, index_type{1}, value_type{2}, tensor_type{{2,3,4},{5,6,7}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, moving_avarage{}, index_type{2}, index_type{1}, value_type{2}, tensor_type{{1,2},{4,5},{7,8}})
    );
    auto test_slide_custom_arg = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto tensor = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto window_size = std::get<3>(t);
            auto window_step = std::get<4>(t);
            auto denom = std::get<5>(t);
            auto expected = std::get<6>(t);
            auto result = slide<value_type>(policy..., tensor, axis, functor, window_size, window_step, window_size, window_step, denom);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    };

    SECTION("default_policy")
    {
        test_slide_custom_arg();
    }
    SECTION("exec_pol<4>")
    {
        test_slide_custom_arg(multithreading::exec_pol<4>{});
    }
}

TEST_CASE("test_slide_exception","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using gtensor::value_error;
    using test_slide_::cumsum;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2functor,3window_size,4window_step
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(0), dim_type{0}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), dim_type{3}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{1}, dim_type{0}, cumsum{}, index_type{2}, index_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumsum{}, index_type{6}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{2}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, cumsum{}, index_type{3}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, cumsum{}, index_type{4}, index_type{1})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto window_size = std::get<3>(t);
        auto window_step = std::get<4>(t);
        REQUIRE_THROWS_AS(slide<value_type>(multithreading::exec_pol<1>{}, tensor, axis, functor, window_size, window_step), value_error);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_slide_flatten_exception","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using gtensor::value_error;
    using test_slide_::cumsum;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    //0tensor,1functor,2window_size,3window_step
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(0), cumsum{}, index_type{2}, index_type{1}),
        std::make_tuple(tensor_type{1}, cumsum{}, index_type{2}, index_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, cumsum{}, index_type{6}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, cumsum{}, index_type{7}, index_type{1})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto functor = std::get<1>(t);
        auto window_size = std::get<2>(t);
        auto window_step = std::get<3>(t);
        REQUIRE_THROWS_AS(slide<value_type>(tensor, gtensor::detail::no_value{}, functor, window_size, window_step), value_error);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_transform","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_slide_::sort;
    using gtensor::transform;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2functor,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, dim_type{0}, sort{}, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), dim_type{0}, sort{}, tensor_type{}.reshape(0,2,3)),
        std::make_tuple(tensor_type{1,2,3,3,2,1,0}, dim_type{0}, sort{}, tensor_type{0,1,1,2,2,3,3}),
        std::make_tuple(tensor_type{{2,1,3},{3,0,1}}, dim_type{0}, sort{}, tensor_type{{2,0,1},{3,1,3}}),
        std::make_tuple(tensor_type{{2,1,3},{3,0,1}}, dim_type{1}, sort{}, tensor_type{{1,2,3},{0,1,3}}),
        std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{0}, sort{}, tensor_type{{{0,1,1},{3,0,1}},{{2,2,3},{3,0,1}}}),
        std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{1}, sort{}, tensor_type{{{2,0,1},{3,1,3}},{{0,0,1},{3,2,1}}}),
        std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{2}, sort{}, tensor_type{{{1,2,3},{0,1,3}},{{0,1,2},{0,1,3}}})
    );
    auto test_transform = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto tensor = std::get<0>(t);
            auto axis = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto expected = std::get<3>(t);
            transform(policy..., tensor, axis, functor);
            REQUIRE(tensor == expected);
        };
        apply_by_element(test, test_data);
    };
    SECTION("default_policy")
    {
        test_transform();
    }
    SECTION("exec_policy<4>")
    {
        test_transform(multithreading::exec_pol<4>{});
    }
}

