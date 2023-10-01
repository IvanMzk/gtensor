/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include "catch.hpp"
#include <iostream>
#include <initializer_list>
#include <array>
#include <tuple>
#include "slice.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("slice_item","[test_slice]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using index_type = typename config_type::index_type;
    using slice_item_type =  gtensor::detail::slice_item<index_type>;
    SECTION("use_init_list"){
        using slice_init_type = std::initializer_list<slice_item_type>;

        slice_init_type l1 = {1};
        REQUIRE(l1.begin()[0].item() == index_type(1));
        REQUIRE(!l1.begin()[0].is_nop());

        slice_init_type l2 = {0,-1,1};
        REQUIRE(l2.begin()[0].item() == index_type(0));
        REQUIRE(!l2.begin()[0].is_nop());
        REQUIRE(l2.begin()[1].item() == index_type(-1));
        REQUIRE(!l2.begin()[1].is_nop());
        REQUIRE(l2.begin()[2].item() == index_type(1));
        REQUIRE(!l2.begin()[2].is_nop());

        slice_init_type l3 = {{},{},-1};
        REQUIRE(l3.begin()[0].is_nop());
        REQUIRE(l3.begin()[1].is_nop());
        REQUIRE(l3.begin()[2].item() == index_type(-1));
        REQUIRE(!l3.begin()[2].is_nop());
    }
    SECTION("use_array"){
        using slice_init_type = slice_item_type[3];
        /*
        slice_init_type l_{{},{},{},{}};
        */

        slice_init_type l = {};
        REQUIRE(l[0].is_nop());
        REQUIRE(l[1].is_nop());
        REQUIRE(l[2].is_nop());

        slice_init_type l1 = {1};
        REQUIRE(l1[0].item() == index_type(1));
        REQUIRE(!l1[0].is_nop());
        REQUIRE(l1[1].is_nop());
        REQUIRE(l1[2].is_nop());

        slice_init_type l2 = {0,-1,1};
        REQUIRE(l2[0].item() == index_type(0));
        REQUIRE(!l2[0].is_nop());
        REQUIRE(l2[1].item() == index_type(-1));
        REQUIRE(!l2[1].is_nop());
        REQUIRE(l2[2].item() == index_type(1));
        REQUIRE(!l2[2].is_nop());

        slice_init_type l3 = {{},{},-1};
        REQUIRE(l3[0].is_nop());
        REQUIRE(l3[1].is_nop());
        REQUIRE(l3[2].item() == index_type(-1));
        REQUIRE(!l3[2].is_nop());
    }
    SECTION("use_std_array"){
        using slice_init_type = typename std::array<slice_item_type,3>;
        slice_init_type l = {};
        REQUIRE(l[0].is_nop());
        REQUIRE(l[1].is_nop());
        REQUIRE(l[2].is_nop());

        slice_init_type l1 = {1};
        REQUIRE(l1[0].item() == index_type(1));
        REQUIRE(!l1[0].is_nop());
        REQUIRE(l1[1].is_nop());
        REQUIRE(l1[2].is_nop());

        slice_init_type l2 = {0,-1,1};
        REQUIRE(l2[0].item() == index_type(0));
        REQUIRE(!l2[0].is_nop());
        REQUIRE(l2[1].item() == index_type(-1));
        REQUIRE(!l2[1].is_nop());
        REQUIRE(l2[2].item() == index_type(1));
        REQUIRE(!l2[2].is_nop());

        /*
        slice_init_type l3 = {{},{},-1};
        REQUIRE(l3[0].is_nop());
        REQUIRE(l3[1].is_nop());
        REQUIRE(l3[2].item() == index_type(-1));
        REQUIRE(!l3[2].is_nop());
        */
    }
    SECTION("use_tuple"){
        using slice_init_type = typename std::tuple<slice_item_type,slice_item_type,slice_item_type>;
        slice_init_type l = {};
        REQUIRE(std::get<0>(l).is_nop());
        REQUIRE(std::get<1>(l).is_nop());
        REQUIRE(std::get<2>(l).is_nop());

        /*
        slice_init_type l1 = {1};
        */

        slice_init_type l1 = {1,{},{}};
        REQUIRE(!std::get<0>(l1).is_nop());
        REQUIRE(std::get<0>(l1).item() == index_type(1));
        REQUIRE(std::get<1>(l1).is_nop());
        REQUIRE(std::get<2>(l1).is_nop());
    }
}

TEST_CASE("test_slice","[test_slice]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using index_type = config_type::index_type;
    using slice_type = gtensor::slice<index_type>;
    using rtag_type = typename slice_type::reduce_tag_type;
    using nop_type = typename slice_type::nop_type;
    using helpers_for_testing::apply_by_element;
    //0items,1expected_is_start,2expected_is_stop,3expected_is_step,4expected_is_reduce,5expected_start,6expected_stop,7expected_step
    auto test_data = std::make_tuple(
        //nop,nop,nop
        std::make_tuple(std::make_tuple(), false,false,true,false, index_type{},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{}), false,false,true,false, index_type{},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{},nop_type{}), false,false,true,false, index_type{},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{},nop_type{},nop_type{}), false,false,true,false, index_type{},index_type{},index_type{1}),
        //nop,nop,step
        std::make_tuple(std::make_tuple(nop_type{},nop_type{},3), false,false,true,false, index_type{},index_type{},index_type{3}),
        std::make_tuple(std::make_tuple(nop_type{},nop_type{},-3), false,false,true,false, index_type{},index_type{},index_type{-3}),
        //nop,stop,step
        std::make_tuple(std::make_tuple(nop_type{},1,3), false,true,true,false, index_type{},index_type{1},index_type{3}),
        std::make_tuple(std::make_tuple(nop_type{},2,-3), false,true,true,false, index_type{},index_type{2},index_type{-3}),
        //start,nop,nop
        std::make_tuple(std::make_tuple(0), true,false,true,false, index_type{0},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(3), true,false,true,false, index_type{3},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(-3), true,false,true,false, index_type{-3},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(0,nop_type{}), true,false,true,false, index_type{0},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(3,nop_type{}), true,false,true,false, index_type{3},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(-3,nop_type{}), true,false,true,false, index_type{-3},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(0,nop_type{},nop_type{}), true,false,true,false, index_type{0},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(3,nop_type{},nop_type{}), true,false,true,false, index_type{3},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(-3,nop_type{},nop_type{}), true,false,true,false, index_type{-3},index_type{},index_type{1}),
        //nop,stop,nop
        std::make_tuple(std::make_tuple(nop_type{},0), false,true,true,false, index_type{},index_type{0},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{},3), false,true,true,false, index_type{},index_type{3},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{},-3), false,true,true,false, index_type{},index_type{-3},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{},0,nop_type{}), false,true,true,false, index_type{},index_type{0},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{},3,nop_type{}), false,true,true,false, index_type{},index_type{3},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{},-3,nop_type{}), false,true,true,false, index_type{},index_type{-3},index_type{1}),
        //start,stop,nop
        std::make_tuple(std::make_tuple(-3,3), true,true,true,false, index_type{-3},index_type{3},index_type{1}),
        std::make_tuple(std::make_tuple(3,-3,nop_type{}), true,true,true,false, index_type{3},index_type{-3},index_type{1}),
        //start,nop,step
        std::make_tuple(std::make_tuple(-3,nop_type{},3), true,false,true,false, index_type{-3},index_type{},index_type{3}),
        std::make_tuple(std::make_tuple(3,nop_type{},-3), true,false,true,false, index_type{3},index_type{},index_type{-3}),
        //start,stop,step
        std::make_tuple(std::make_tuple(1,2,3), true,true,true,false, index_type{1},index_type{2},index_type{3}),
        //reduce
        std::make_tuple(std::make_tuple(0,rtag_type{}), true,false,true,true, index_type{0},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(3,rtag_type{}), true,false,true,true, index_type{3},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(-3,rtag_type{}), true,false,true,true, index_type{-3},index_type{},index_type{1})
    );
    auto test_slice = [](const auto& slice_, const auto& t)
    {
        auto expected_is_start = std::get<1>(t);
        auto expected_is_stop = std::get<2>(t);
        auto expected_is_step = std::get<3>(t);
        auto expected_is_reduce = std::get<4>(t);
        auto expected_start = std::get<5>(t);
        auto expected_stop = std::get<6>(t);
        auto expected_step = std::get<7>(t);

        auto result_is_start = slice_.is_start();
        auto result_is_stop = slice_.is_stop();
        auto result_is_step = slice_.is_step();
        auto result_is_reduce = slice_.is_reduce();
        auto result_start = slice_.start();
        auto result_stop = slice_.stop();
        auto result_step = slice_.step();
        REQUIRE(result_is_start == expected_is_start);
        REQUIRE(result_is_stop == expected_is_stop);
        REQUIRE(result_is_step == expected_is_step);
        REQUIRE(result_is_reduce == expected_is_reduce);
        REQUIRE(result_start == expected_start);
        REQUIRE(result_stop == expected_stop);
        REQUIRE(result_step == expected_step);
    };
    SECTION("test_slice_slice_constructor")
    {
        auto test = [test_slice](const auto& t){
            auto items = std::get<0>(t);
            auto make_slice = [](const auto&...items_){
                return slice_type(items_...);
            };
            auto slice_ = std::apply(make_slice, items);
            test_slice(slice_, t);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_slice_init_list_constructor")
    {
        auto test = [test_slice](const auto& t){
            auto items = std::get<0>(t);
            auto make_slice = [](const auto&...items_){
                return slice_type{items_...};
            };
            auto slice_ = std::apply(make_slice, items);
            test_slice(slice_, t);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_slice_init_list_constructor_exception")
    {
        using gtensor::value_error;
        using slice_item_type = slice_type::slice_item_type;
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{1,1,1,1}),value_error);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{rtag_type{}}),value_error);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{rtag_type{},1}),value_error);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{rtag_type{},1,1}),value_error);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{nop_type{},rtag_type{}}),value_error);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{1,rtag_type{},1}),value_error);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{1,1,rtag_type{}}),value_error);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{1,1,1,rtag_type{}}),value_error);
    }
}
