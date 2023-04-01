#include "catch.hpp"
#include <iostream>
#include <initializer_list>
#include <array>
#include <tuple>
#include "slice.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("slice_item","[slice_item]"){
    using config_type = gtensor::config::default_config;
    using index_type = typename config_type::index_type;
    using slice_item_type =  typename gtensor::slice_traits<config_type>::slice_item_type;
    SECTION("use_init_list"){
        using slice_init_type = typename gtensor::slice_traits<config_type>::slice_init_type;

        slice_init_type l1 = {1};
        REQUIRE(l1.begin()[0].i == index_type(1));
        REQUIRE(!l1.begin()[0].nop);

        slice_init_type l2 = {0,-1,1};
        REQUIRE(l2.begin()[0].i == index_type(0));
        REQUIRE(!l2.begin()[0].nop);
        REQUIRE(l2.begin()[1].i == index_type(-1));
        REQUIRE(!l2.begin()[1].nop);
        REQUIRE(l2.begin()[2].i == index_type(1));
        REQUIRE(!l2.begin()[2].nop);

        slice_init_type l3 = {{},{},-1};
        REQUIRE(l3.begin()[0].nop);
        REQUIRE(l3.begin()[1].nop);
        REQUIRE(l3.begin()[2].i == index_type(-1));
        REQUIRE(!l3.begin()[2].nop);
    }
    SECTION("use_array"){
        using slice_init_type = slice_item_type[3];
        /*
        slice_init_type l_{{},{},{},{}};
        */

        slice_init_type l = {};
        REQUIRE(l[0].nop);
        REQUIRE(l[1].nop);
        REQUIRE(l[2].nop);

        slice_init_type l1 = {1};
        REQUIRE(l1[0].i == index_type(1));
        REQUIRE(!l1[0].nop);
        REQUIRE(l1[1].nop);
        REQUIRE(l1[2].nop);

        slice_init_type l2 = {0,-1,1};
        REQUIRE(l2[0].i == index_type(0));
        REQUIRE(!l2[0].nop);
        REQUIRE(l2[1].i == index_type(-1));
        REQUIRE(!l2[1].nop);
        REQUIRE(l2[2].i == index_type(1));
        REQUIRE(!l2[2].nop);

        slice_init_type l3 = {{},{},-1};
        REQUIRE(l3[0].nop);
        REQUIRE(l3[1].nop);
        REQUIRE(l3[2].i == index_type(-1));
        REQUIRE(!l3[2].nop);
    }
    SECTION("use_std_array"){
        using slice_init_type = typename std::array<slice_item_type,3>;
        slice_init_type l = {};
        REQUIRE(l[0].nop);
        REQUIRE(l[1].nop);
        REQUIRE(l[2].nop);

        slice_init_type l1 = {1};
        REQUIRE(l1[0].i == index_type(1));
        REQUIRE(!l1[0].nop);
        REQUIRE(l1[1].nop);
        REQUIRE(l1[2].nop);

        slice_init_type l2 = {0,-1,1};
        REQUIRE(l2[0].i == index_type(0));
        REQUIRE(!l2[0].nop);
        REQUIRE(l2[1].i == index_type(-1));
        REQUIRE(!l2[1].nop);
        REQUIRE(l2[2].i == index_type(1));
        REQUIRE(!l2[2].nop);

        /*
        slice_init_type l3 = {{},{},-1};
        REQUIRE(l3[0].nop);
        REQUIRE(l3[1].nop);
        REQUIRE(l3[2].i == index_type(-1));
        REQUIRE(!l3[2].nop);
        */
    }
    SECTION("use_tuple"){
        using slice_init_type = typename std::tuple<slice_item_type,slice_item_type,slice_item_type>;
        slice_init_type l = {};
        REQUIRE(std::get<0>(l).nop);
        REQUIRE(std::get<1>(l).nop);
        REQUIRE(std::get<2>(l).nop);

        /*
        slice_init_type l1 = {1};
        */

        slice_init_type l1 = {1,{},{}};
        REQUIRE(!std::get<0>(l1).nop);
        REQUIRE(std::get<0>(l1).i == index_type(1));
        REQUIRE(std::get<1>(l1).nop);
        REQUIRE(std::get<2>(l1).nop);
    }
}

TEST_CASE("test_slice","[test_slice]"){
    using config_type = gtensor::config::default_config;
    using index_type = typename config_type::index_type;
    using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
    using nop_type = typename gtensor::slice_traits<config_type>::nop_type;
    using helpers_for_testing::apply_by_element;
    //nop_type nop{};

    //0items,1expected_is_start,2expected_is_stop,3expected_is_step,4expected_start,5expected_stop,6expected_step
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(), false,false,true,index_type{},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{}), false,false,true,index_type{},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{},nop_type{}), false,false,true,index_type{},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(nop_type{},nop_type{},nop_type{}), false,false,true,index_type{},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(0), true,false,true,index_type{0},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(0,nop_type{}), true,false,true,index_type{0},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(0,nop_type{},nop_type{}), true,false,true,index_type{0},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(1), true,false,true,index_type{1},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(1,nop_type{}), true,false,true,index_type{1},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(1,nop_type{},nop_type{}), true,false,true,index_type{1},index_type{},index_type{1}),
        std::make_tuple(std::make_tuple(1,2), true,true,true,index_type{1},index_type{2},index_type{1}),
        std::make_tuple(std::make_tuple(1,2,nop_type{}), true,true,true,index_type{1},index_type{2},index_type{1}),
        std::make_tuple(std::make_tuple(1,2,3), true,true,true,index_type{1},index_type{2},index_type{3})
    );
    auto test_slice = [](const auto& slice_, const auto& t)
    {
        auto expected_is_start = std::get<1>(t);
        auto expected_is_stop = std::get<2>(t);
        auto expected_is_step = std::get<3>(t);
        auto expected_start = std::get<4>(t);
        auto expected_stop = std::get<5>(t);
        auto expected_step = std::get<6>(t);

        auto result_is_start = slice_.is_start();
        auto result_is_stop = slice_.is_stop();
        auto result_is_step = slice_.is_step();
        auto result_start = slice_.start;
        auto result_stop = slice_.stop;
        auto result_step = slice_.step;
        REQUIRE(result_is_start == expected_is_start);
        REQUIRE(result_is_stop == expected_is_stop);
        REQUIRE(result_is_step == expected_is_step);
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
}

TEST_CASE("test_fill_slice","[test_slice]"){
    using config_type = gtensor::config::default_config;
    using index_type = typename gtensor::config::default_config::index_type;
    using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
    using nop_type = typename gtensor::slice_traits<config_type>::nop_type;
    using gtensor::detail::fill_slice;
    //0slice,1expected,2shape_element
    using test_type = std::tuple<slice_type,slice_type,index_type>;
    auto test_data = GENERATE(
        test_type{slice_type(),slice_type(0,11,1),11},
        test_type{slice_type(3),slice_type(3,11,1),11},
        test_type{slice_type(2),slice_type(2,5,1),5},
        test_type{slice_type(0,3),slice_type(0,3,1),15},
        test_type{slice_type(1,9,2),slice_type(1,9,2),15},
        test_type{slice_type(-10,nop_type{},2),slice_type(5,15,2),15},
        test_type{slice_type(nop_type{},nop_type{},-1),slice_type(10,-1,-1),11},
        test_type{slice_type(-1,-4,-1),slice_type(10,7,-1),11},
        test_type{slice_type(7,nop_type{},-2),slice_type(7,-1,-2),11},
        test_type{slice_type(4,0,-1),slice_type(4,0,-1),11},
        test_type{slice_type(4,0,-1),slice_type(4,0,-1),11}
        );
    auto sl = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    auto shape_element = std::get<2>(test_data);
    auto result = fill_slice(sl, shape_element);
    REQUIRE(result == expected);
}

TEST_CASE("test_fill_slices","[test_slice]"){
    using config_type = gtensor::config::default_config;
    using shape_type = typename config_type::shape_type;
    using slices_container_type = typename gtensor::slice_traits<config_type>::slices_container_type;
    using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
    using nop_type = typename gtensor::slice_traits<config_type>::nop_type;
    using gtensor::detail::fill_slices;
    nop_type nop{};

    SECTION("test_fill_slices_init_list")
    {
        REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4},{}) == slices_container_type{} );
        REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4},{{}}) == slices_container_type{{0,2,1}} );
        REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4},{{},{},{}}) == slices_container_type{{0,2,1},{0,3,1},{0,4,1}} );
        REQUIRE(fill_slices<slices_container_type>(shape_type{4,3,2},{{1,3,{nop}},{},{}}) == slices_container_type{{1,3,1},{0,3,1},{0,2,1}} );
        REQUIRE(fill_slices<slices_container_type>(shape_type{3,4},{{nop,nop,-1},{-4,nop,-1}}) == slices_container_type{{2,-1,-1},{0,-1,-1}});
        REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4},{{},{1},{1,3}}) == slices_container_type{{0,2,1},{1,3,1},{1,3,1}});
    }
    SECTION("test_fill_slices_variadic")
    {
        REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4}) == slices_container_type{} );
        REQUIRE(fill_slices<slices_container_type>(shape_type{4,2,3},slice_type{}) == slices_container_type{slice_type{0,4,1}} );
        REQUIRE(fill_slices<slices_container_type>(shape_type{4,3,2},slice_type{},slice_type{},slice_type{}) == slices_container_type{slice_type{0,4,1},slice_type{0,3,1},slice_type{0,2,1}} );
        REQUIRE(fill_slices<slices_container_type>(shape_type{4,3,2},slice_type{1,3,{nop}},slice_type{},slice_type{}) == slices_container_type{slice_type{1,3,1},slice_type{0,3,1},slice_type{0,2,1}} );
        REQUIRE(fill_slices<slices_container_type>(shape_type{3,4},slice_type{nop,nop,-1},slice_type{-4,nop,-1}) == slices_container_type{slice_type{2,-1,-1},slice_type{0,-1,-1}});
    }
    SECTION("test_fill_slices_container")
    {
        REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4}, slices_container_type{}) == slices_container_type{});
        REQUIRE(fill_slices<slices_container_type>(shape_type{4,2,3},slices_container_type{slice_type{}}) == slices_container_type{slice_type{0,4,1}});
        REQUIRE(fill_slices<slices_container_type>(shape_type{4,3,2},slices_container_type{slice_type{},slice_type{},slice_type{}}) == slices_container_type{slice_type{0,4,1},slice_type{0,3,1},slice_type{0,2,1}} );
        REQUIRE(fill_slices<slices_container_type>(shape_type{4,3,2},slices_container_type{slice_type{1,3,{nop}},slice_type{},slice_type{}}) == slices_container_type{slice_type{1,3,1},slice_type{0,3,1},slice_type{0,2,1}} );
        REQUIRE(fill_slices<slices_container_type>(shape_type{3,4},slices_container_type{slice_type{nop,nop,-1},slice_type{-4,nop,-1}}) == slices_container_type{slice_type{2,-1,-1},slice_type{0,-1,-1}});
    }
}

TEST_CASE("test_check_slice","[test_check_slice]"){
    using config_type = gtensor::config::default_config;
    using index_type = typename config_type::index_type;
    using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
    using gtensor::subscript_exception;
    using gtensor::detail::check_slice;
    //0slice,1shape_element
    using test_type = std::tuple<slice_type, index_type>;

    SECTION("test_check_slice_nothrow")
    {
        auto test_data = GENERATE(
            test_type(slice_type{0,1,1}, index_type(1)),
            test_type(slice_type{0,5,1}, index_type(5)),
            test_type(slice_type{0,5,1}, index_type(5)),
            test_type(slice_type{0,5,1}, index_type(5)),
            test_type(slice_type{0,5,1}, index_type(5)),
            test_type(slice_type{4,-1,-1}, index_type(5)),
            test_type(slice_type{0,-1,-1}, index_type(5)),
            test_type(slice_type{0,3,1}, index_type(5)),
            test_type(slice_type{2,5,1}, index_type(5)),
            test_type(slice_type{1,4,1}, index_type(5)),
            test_type(slice_type{1,4,1}, index_type(5)),
            test_type(slice_type{3,4,1}, index_type(5)),
            test_type(slice_type{0,-1,-1}, index_type(5))
        );
        auto slice = std::get<0>(test_data);
        auto shape_element = std::get<1>(test_data);
        REQUIRE_NOTHROW(check_slice(slice,shape_element));
    }
    SECTION("test_check_slice_exception")
    {
        auto test_data = GENERATE(
            test_type(slice_type{0,1,1},index_type(0)),
            test_type(slice_type{1,0,-1},index_type(0)),
            test_type(slice_type{5,5,1},index_type(5)),
            test_type(slice_type{6,5,1},index_type(5)),
            test_type(slice_type{-1,5,1},index_type(5)),
            test_type(slice_type{1,-1,1},index_type(5)),
            test_type(slice_type{0,0,1},index_type(5)),
            test_type(slice_type{0,6,1},index_type(5)),
            test_type(slice_type{2,0,1},index_type(5)),
            test_type(slice_type{5,5,-1},index_type(5)),
            test_type(slice_type{-1,5,-1},index_type(5)),
            test_type(slice_type{0,4,-1},index_type(5)),
            test_type(slice_type{0,5,-1},index_type(5)),
            test_type(slice_type{1,4,-1},index_type(5))
        );
        auto slice = std::get<0>(test_data);
        auto shape_element = std::get<1>(test_data);
        REQUIRE_THROWS_AS(check_slice(slice,shape_element), subscript_exception);
    }
}

TEST_CASE("test_is_slices", "[test_is_slices]"){
    using config_type = gtensor::config::default_config;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
    using gtensor::detail::is_slice;
    using gtensor::detail::is_slices;


    REQUIRE(is_slice<slice_type>);
    REQUIRE(!is_slice<shape_type>);
    REQUIRE(!is_slice<index_type>);

    REQUIRE(is_slices<>);
    REQUIRE(is_slices<slice_type>);
    REQUIRE(is_slices<slice_type,slice_type,slice_type>);
    REQUIRE(!is_slices<shape_type>);
    REQUIRE(!is_slices<index_type>);
    REQUIRE(!is_slices<slice_type,index_type,slice_type>);
}

TEST_CASE("test_is_slices_container", "[test_is_slices]"){
    using config_type = gtensor::config::default_config;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
    using gtensor::detail::is_slices_container;

    REQUIRE(is_slices_container<std::vector<slice_type>>);
    REQUIRE(is_slices_container<std::initializer_list<slice_type>>);
    REQUIRE(is_slices_container<std::array<slice_type, 3>>);
    REQUIRE(is_slices_container<slice_type[3]>);
    REQUIRE(!is_slices_container<index_type>);
    REQUIRE(!is_slices_container<shape_type>);
    REQUIRE(!is_slices_container<std::vector<int>>);
    REQUIRE(!is_slices_container<int>);
}

TEST_CASE("test_check_slices","[test_check_slices]"){
    using config_type = gtensor::config::default_config;
    using shape_type = typename config_type::shape_type;
    using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
    using gtensor::detail::check_slices;
    REQUIRE_NOTHROW(check_slices(shape_type{5},std::vector<slice_type>{}));
    REQUIRE_NOTHROW(check_slices(shape_type{5},std::vector{slice_type{0,5,1}}));
    REQUIRE_NOTHROW(check_slices(shape_type{5,3,4},std::vector{slice_type{0,5,1}}));
    REQUIRE_NOTHROW(check_slices(shape_type{5,4,3},std::vector{slice_type{0,5,1}, slice_type{0,4,1}}));
    REQUIRE_NOTHROW(check_slices(shape_type{5,4,3},std::vector{slice_type{0,5,1}, slice_type{0,4,1}, slice_type{2,-1,-1}}));

    REQUIRE_THROWS_AS(check_slices(shape_type{5},std::vector{slice_type{0,5,1}, slice_type{0,5,1}}), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slices(shape_type{5},std::vector{slice_type{0,-1,1}}), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slices(shape_type{3,4,5},std::vector{slice_type{0,5,1}, slice_type{0,5,1}}), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slices(shape_type{3,4,5},std::vector{slice_type{0,5,1}, slice_type{1,1,1}, slice_type{0,3,1}}), gtensor::subscript_exception);
}

TEST_CASE("test_check_slices_number","[test_check_slices]"){
    using config_type = gtensor::config::default_config;
    using shape_type = typename config_type::shape_type;
    using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
    using slices_init_type = typename gtensor::slice_traits<config_type>::slices_init_type;
    using gtensor::detail::check_slices_number;

    SECTION("test_check_slices_number_variadic")
    {
        REQUIRE_NOTHROW(check_slices_number(shape_type{5},slice_type{}));
        REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slice_type{}));
        REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slice_type{},slice_type{}));
        REQUIRE_THROWS_AS(check_slices_number(shape_type{5},slice_type{},slice_type{}), gtensor::subscript_exception);
        REQUIRE_THROWS_AS(check_slices_number(shape_type{5,6},slice_type{},slice_type{},slice_type{},slice_type{}), gtensor::subscript_exception);
    }
    SECTION("test_check_slices_number_slices_init_type")
    {
        REQUIRE_NOTHROW(check_slices_number(shape_type{5},slices_init_type{{}}));
        REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slices_init_type{{},{}}));
        REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slices_init_type{{}}));
        REQUIRE_THROWS_AS(check_slices_number(shape_type{5,6},slices_init_type{{},{},{}}), gtensor::subscript_exception);
        REQUIRE_THROWS_AS(check_slices_number(shape_type{5},slices_init_type{{},{}}), gtensor::subscript_exception);
    }
    SECTION("test_check_slices_number_slices_container")
    {
        using slice_container_type = std::vector<slice_type>;
        REQUIRE_NOTHROW(check_slices_number(shape_type{5},slice_container_type(1)));
        REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slice_container_type(1)));
        REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slice_container_type(2)));
        REQUIRE_THROWS_AS(check_slices_number(shape_type{5,6},slice_container_type(3)), gtensor::subscript_exception);
        REQUIRE_THROWS_AS(check_slices_number(shape_type{5},slice_container_type(2)), gtensor::subscript_exception);
    }
}

