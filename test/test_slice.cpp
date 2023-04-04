#include "catch.hpp"
#include <iostream>
#include <initializer_list>
#include <array>
#include <tuple>
#include "slice.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("slice_item","[test_slice]"){
    using config_type = gtensor::config::default_config;
    using index_type = typename config_type::index_type;
    using slice_item_type =  typename gtensor::slice_traits<config_type>::slice_item_type;
    SECTION("use_init_list"){
        using slice_init_type = typename gtensor::slice_traits<config_type>::slice_init_type;

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
    using config_type = gtensor::config::default_config;
    using index_type = config_type::index_type;
    using slice_type = gtensor::slice_traits<config_type>::slice_type;
    using rtag_type = gtensor::slice_traits<config_type>::rtag_type;
    using nop_type = gtensor::slice_traits<config_type>::nop_type;
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
        std::make_tuple(std::make_tuple(0,rtag_type{}), true,true,true,true, index_type{0},index_type{1},index_type{1}),
        std::make_tuple(std::make_tuple(3,rtag_type{}), true,true,true,true, index_type{3},index_type{4},index_type{1}),
        std::make_tuple(std::make_tuple(-3,rtag_type{}), true,true,true,true, index_type{-3},index_type{-2},index_type{1})
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
        using gtensor::slice_exception;
        using slice_item_type = slice_type::slice_item_type;
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{1,1,1,1}),slice_exception);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{rtag_type{}}),slice_exception);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{rtag_type{},1}),slice_exception);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{rtag_type{},1,1}),slice_exception);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{nop_type{},rtag_type{}}),slice_exception);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{1,rtag_type{},1}),slice_exception);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{1,1,rtag_type{}}),slice_exception);
        REQUIRE_THROWS_AS(slice_type(std::initializer_list<slice_item_type>{1,1,1,rtag_type{}}),slice_exception);
    }
}

// TEST_CASE("test_fill_slice","[test_slice]"){
//     using config_type = gtensor::config::default_config;
//     using index_type = typename gtensor::config::default_config::index_type;
//     using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
//     using nop_type = typename gtensor::slice_traits<config_type>::nop_type;
//     using gtensor::detail::fill_slice;
//     //0slice,1expected,2shape_element
//     using test_type = std::tuple<slice_type,slice_type,index_type>;
//     auto test_data = GENERATE(
//         test_type{slice_type(),slice_type(0,11,1),11},
//         test_type{slice_type(3),slice_type(3,11,1),11},
//         test_type{slice_type(2),slice_type(2,5,1),5},
//         test_type{slice_type(0,3),slice_type(0,3,1),15},
//         test_type{slice_type(1,9,2),slice_type(1,9,2),15},
//         test_type{slice_type(-10,nop_type{},2),slice_type(5,15,2),15},
//         test_type{slice_type(nop_type{},nop_type{},-1),slice_type(10,-1,-1),11},
//         test_type{slice_type(-1,-4,-1),slice_type(10,7,-1),11},
//         test_type{slice_type(7,nop_type{},-2),slice_type(7,-1,-2),11},
//         test_type{slice_type(4,0,-1),slice_type(4,0,-1),11},
//         test_type{slice_type(4,0,-1),slice_type(4,0,-1),11}
//         );
//     auto sl = std::get<0>(test_data);
//     auto expected = std::get<1>(test_data);
//     auto shape_element = std::get<2>(test_data);
//     auto result = fill_slice(sl, shape_element);
//     REQUIRE(result == expected);
// }

// TEST_CASE("test_fill_slices","[test_slice]"){
//     using config_type = gtensor::config::default_config;
//     using shape_type = typename config_type::shape_type;
//     using slices_container_type = typename gtensor::slice_traits<config_type>::slices_container_type;
//     using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
//     using nop_type = typename gtensor::slice_traits<config_type>::nop_type;
//     using gtensor::detail::fill_slices;
//     nop_type nop{};

//     SECTION("test_fill_slices_init_list")
//     {
//         REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4},{}) == slices_container_type{} );
//         REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4},{{}}) == slices_container_type{{0,2,1}} );
//         REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4},{{},{},{}}) == slices_container_type{{0,2,1},{0,3,1},{0,4,1}} );
//         REQUIRE(fill_slices<slices_container_type>(shape_type{4,3,2},{{1,3,{nop}},{},{}}) == slices_container_type{{1,3,1},{0,3,1},{0,2,1}} );
//         REQUIRE(fill_slices<slices_container_type>(shape_type{3,4},{{nop,nop,-1},{-4,nop,-1}}) == slices_container_type{{2,-1,-1},{0,-1,-1}});
//         REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4},{{},{1},{1,3}}) == slices_container_type{{0,2,1},{1,3,1},{1,3,1}});
//     }
//     SECTION("test_fill_slices_variadic")
//     {
//         REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4}) == slices_container_type{} );
//         REQUIRE(fill_slices<slices_container_type>(shape_type{4,2,3},slice_type{}) == slices_container_type{slice_type{0,4,1}} );
//         REQUIRE(fill_slices<slices_container_type>(shape_type{4,3,2},slice_type{},slice_type{},slice_type{}) == slices_container_type{slice_type{0,4,1},slice_type{0,3,1},slice_type{0,2,1}} );
//         REQUIRE(fill_slices<slices_container_type>(shape_type{4,3,2},slice_type{1,3,{nop}},slice_type{},slice_type{}) == slices_container_type{slice_type{1,3,1},slice_type{0,3,1},slice_type{0,2,1}} );
//         REQUIRE(fill_slices<slices_container_type>(shape_type{3,4},slice_type{nop,nop,-1},slice_type{-4,nop,-1}) == slices_container_type{slice_type{2,-1,-1},slice_type{0,-1,-1}});
//     }
//     SECTION("test_fill_slices_container")
//     {
//         REQUIRE(fill_slices<slices_container_type>(shape_type{2,3,4}, slices_container_type{}) == slices_container_type{});
//         REQUIRE(fill_slices<slices_container_type>(shape_type{4,2,3},slices_container_type{slice_type{}}) == slices_container_type{slice_type{0,4,1}});
//         REQUIRE(fill_slices<slices_container_type>(shape_type{4,3,2},slices_container_type{slice_type{},slice_type{},slice_type{}}) == slices_container_type{slice_type{0,4,1},slice_type{0,3,1},slice_type{0,2,1}} );
//         REQUIRE(fill_slices<slices_container_type>(shape_type{4,3,2},slices_container_type{slice_type{1,3,{nop}},slice_type{},slice_type{}}) == slices_container_type{slice_type{1,3,1},slice_type{0,3,1},slice_type{0,2,1}} );
//         REQUIRE(fill_slices<slices_container_type>(shape_type{3,4},slices_container_type{slice_type{nop,nop,-1},slice_type{-4,nop,-1}}) == slices_container_type{slice_type{2,-1,-1},slice_type{0,-1,-1}});
//     }
// }

// TEST_CASE("test_check_slice_item_list","[test_slice]"){
//     using config_type = gtensor::config::default_config;
//     using slice_item_type = typename gtensor::slice_traits<config_type>::slice_item_type;
//     using rtag_type = typename gtensor::slice_traits<config_type>::rtag_type;
//     using gtensor::slice_exception;
//     using gtensor::detail::check_slice_item_list;
//     using helpers_for_testing::apply_by_element;

//     SECTION("test_check_slice_item_list_nothrow")
//     {
//         //items
//         auto test_data = std::make_tuple(
//             std::make_tuple(slice_item_type{}),
//             std::make_tuple(slice_item_type{0}),
//             std::make_tuple(slice_item_type{},slice_item_type{}),
//             std::make_tuple(slice_item_type{0},slice_item_type{}),
//             std::make_tuple(slice_item_type{},slice_item_type{0}),
//             std::make_tuple(slice_item_type{0},slice_item_type{1}),
//             std::make_tuple(slice_item_type{0},slice_item_type{rtag_type{}}),
//             std::make_tuple(slice_item_type{},slice_item_type{},slice_item_type{}),
//             std::make_tuple(slice_item_type{0},slice_item_type{},slice_item_type{}),
//             std::make_tuple(slice_item_type{0},slice_item_type{1},slice_item_type{}),
//             std::make_tuple(slice_item_type{0},slice_item_type{1},slice_item_type{2}),
//             std::make_tuple(slice_item_type{},slice_item_type{1},slice_item_type{2}),
//             std::make_tuple(slice_item_type{},slice_item_type{},slice_item_type{2}),
//             std::make_tuple(slice_item_type{0},slice_item_type{},slice_item_type{2}),
//             std::make_tuple(slice_item_type{},slice_item_type{1},slice_item_type{})
//         );
//         auto test = [](const auto& items){
//             auto apply_items = [](const auto&...items_){
//                 check_slice_item_list({items_...});
//             };
//             REQUIRE_NOTHROW(std::apply(apply_items, items));
//         };
//         apply_by_element(test, test_data);
//     }
//     SECTION("test_check_slice_item_list_exception")
//     {
//         //items
//         auto test_data = std::make_tuple(
//             std::make_tuple(slice_item_type{rtag_type{}}),
//             std::make_tuple(slice_item_type{rtag_type{}},slice_item_type{}),
//             std::make_tuple(slice_item_type{rtag_type{}},slice_item_type{1}),
//             std::make_tuple(slice_item_type{},slice_item_type{rtag_type{}}),
//             std::make_tuple(slice_item_type{rtag_type{}},slice_item_type{},slice_item_type{}),
//             std::make_tuple(slice_item_type{rtag_type{}},slice_item_type{0},slice_item_type{1}),
//             std::make_tuple(slice_item_type{},slice_item_type{rtag_type{}},slice_item_type{1}),
//             std::make_tuple(slice_item_type{},slice_item_type{},slice_item_type{rtag_type{}}),
//             std::make_tuple(slice_item_type{},slice_item_type{},slice_item_type{},slice_item_type{})
//         );
//         auto test = [](const auto& items){
//             auto apply_items = [](const auto&...items_){
//                 check_slice_item_list({items_...});
//             };
//             REQUIRE_THROWS_AS(std::apply(apply_items, items), slice_exception);
//         };
//         apply_by_element(test, test_data);
//     }
// }

// TEST_CASE("test_check_slice","[test_slice]"){
//     using config_type = gtensor::config::default_config;
//     using index_type = typename config_type::index_type;
//     using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
//     using gtensor::subscript_exception;
//     using gtensor::detail::check_slice;
//     //0slice,1shape_element
//     using test_type = std::tuple<slice_type, index_type>;

//     SECTION("test_check_slice_nothrow")
//     {
//         auto test_data = GENERATE(
//             test_type(slice_type{0,1,1}, index_type(1)),
//             test_type(slice_type{0,5,1}, index_type(5)),
//             test_type(slice_type{0,5,1}, index_type(5)),
//             test_type(slice_type{0,5,1}, index_type(5)),
//             test_type(slice_type{0,5,1}, index_type(5)),
//             test_type(slice_type{4,-1,-1}, index_type(5)),
//             test_type(slice_type{0,-1,-1}, index_type(5)),
//             test_type(slice_type{0,3,1}, index_type(5)),
//             test_type(slice_type{2,5,1}, index_type(5)),
//             test_type(slice_type{1,4,1}, index_type(5)),
//             test_type(slice_type{1,4,1}, index_type(5)),
//             test_type(slice_type{3,4,1}, index_type(5)),
//             test_type(slice_type{0,-1,-1}, index_type(5))
//         );
//         auto slice = std::get<0>(test_data);
//         auto shape_element = std::get<1>(test_data);
//         REQUIRE_NOTHROW(check_slice(slice,shape_element));
//     }
//     SECTION("test_check_slice_exception")
//     {
//         auto test_data = GENERATE(
//             test_type(slice_type{0,1,1},index_type(0)),
//             test_type(slice_type{1,0,-1},index_type(0)),
//             test_type(slice_type{5,5,1},index_type(5)),
//             test_type(slice_type{6,5,1},index_type(5)),
//             test_type(slice_type{-1,5,1},index_type(5)),
//             test_type(slice_type{1,-1,1},index_type(5)),
//             test_type(slice_type{0,0,1},index_type(5)),
//             test_type(slice_type{0,6,1},index_type(5)),
//             test_type(slice_type{2,0,1},index_type(5)),
//             test_type(slice_type{5,5,-1},index_type(5)),
//             test_type(slice_type{-1,5,-1},index_type(5)),
//             test_type(slice_type{0,4,-1},index_type(5)),
//             test_type(slice_type{0,5,-1},index_type(5)),
//             test_type(slice_type{1,4,-1},index_type(5))
//         );
//         auto slice = std::get<0>(test_data);
//         auto shape_element = std::get<1>(test_data);
//         REQUIRE_THROWS_AS(check_slice(slice,shape_element), subscript_exception);
//     }
// }

// TEST_CASE("test_is_slices", "[test_slice]"){
//     using config_type = gtensor::config::default_config;
//     using index_type = typename config_type::index_type;
//     using shape_type = typename config_type::shape_type;
//     using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
//     using gtensor::detail::is_slice;
//     using gtensor::detail::is_slices;


//     REQUIRE(is_slice<slice_type>);
//     REQUIRE(!is_slice<shape_type>);
//     REQUIRE(!is_slice<index_type>);

//     REQUIRE(is_slices<>);
//     REQUIRE(is_slices<slice_type>);
//     REQUIRE(is_slices<slice_type,slice_type,slice_type>);
//     REQUIRE(!is_slices<shape_type>);
//     REQUIRE(!is_slices<index_type>);
//     REQUIRE(!is_slices<slice_type,index_type,slice_type>);
// }

// TEST_CASE("test_is_slices_container", "[test_slice]"){
//     using config_type = gtensor::config::default_config;
//     using index_type = typename config_type::index_type;
//     using shape_type = typename config_type::shape_type;
//     using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
//     using gtensor::detail::is_slices_container;

//     REQUIRE(is_slices_container<std::vector<slice_type>>);
//     REQUIRE(is_slices_container<std::initializer_list<slice_type>>);
//     REQUIRE(is_slices_container<std::array<slice_type, 3>>);
//     REQUIRE(is_slices_container<slice_type[3]>);
//     REQUIRE(!is_slices_container<index_type>);
//     REQUIRE(!is_slices_container<shape_type>);
//     REQUIRE(!is_slices_container<std::vector<int>>);
//     REQUIRE(!is_slices_container<int>);
// }

// TEST_CASE("test_check_slices","[test_slice]"){
//     using config_type = gtensor::config::default_config;
//     using shape_type = typename config_type::shape_type;
//     using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
//     using gtensor::detail::check_slices;
//     REQUIRE_NOTHROW(check_slices(shape_type{5},std::vector<slice_type>{}));
//     REQUIRE_NOTHROW(check_slices(shape_type{5},std::vector{slice_type{0,5,1}}));
//     REQUIRE_NOTHROW(check_slices(shape_type{5,3,4},std::vector{slice_type{0,5,1}}));
//     REQUIRE_NOTHROW(check_slices(shape_type{5,4,3},std::vector{slice_type{0,5,1}, slice_type{0,4,1}}));
//     REQUIRE_NOTHROW(check_slices(shape_type{5,4,3},std::vector{slice_type{0,5,1}, slice_type{0,4,1}, slice_type{2,-1,-1}}));

//     REQUIRE_THROWS_AS(check_slices(shape_type{5},std::vector{slice_type{0,5,1}, slice_type{0,5,1}}), gtensor::subscript_exception);
//     REQUIRE_THROWS_AS(check_slices(shape_type{5},std::vector{slice_type{0,-1,1}}), gtensor::subscript_exception);
//     REQUIRE_THROWS_AS(check_slices(shape_type{3,4,5},std::vector{slice_type{0,5,1}, slice_type{0,5,1}}), gtensor::subscript_exception);
//     REQUIRE_THROWS_AS(check_slices(shape_type{3,4,5},std::vector{slice_type{0,5,1}, slice_type{1,1,1}, slice_type{0,3,1}}), gtensor::subscript_exception);
// }

// TEST_CASE("test_check_slices_number","[test_slice]"){
//     using config_type = gtensor::config::default_config;
//     using shape_type = typename config_type::shape_type;
//     using slice_type = typename gtensor::slice_traits<config_type>::slice_type;
//     using slices_init_type = typename gtensor::slice_traits<config_type>::slices_init_type;
//     using gtensor::detail::check_slices_number;

//     SECTION("test_check_slices_number_variadic")
//     {
//         REQUIRE_NOTHROW(check_slices_number(shape_type{5},slice_type{}));
//         REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slice_type{}));
//         REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slice_type{},slice_type{}));
//         REQUIRE_THROWS_AS(check_slices_number(shape_type{5},slice_type{},slice_type{}), gtensor::subscript_exception);
//         REQUIRE_THROWS_AS(check_slices_number(shape_type{5,6},slice_type{},slice_type{},slice_type{},slice_type{}), gtensor::subscript_exception);
//     }
//     SECTION("test_check_slices_number_slices_init_type")
//     {
//         REQUIRE_NOTHROW(check_slices_number(shape_type{5},slices_init_type{{}}));
//         REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slices_init_type{{},{}}));
//         REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slices_init_type{{}}));
//         REQUIRE_THROWS_AS(check_slices_number(shape_type{5,6},slices_init_type{{},{},{}}), gtensor::subscript_exception);
//         REQUIRE_THROWS_AS(check_slices_number(shape_type{5},slices_init_type{{},{}}), gtensor::subscript_exception);
//     }
//     SECTION("test_check_slices_number_slices_container")
//     {
//         using slice_container_type = std::vector<slice_type>;
//         REQUIRE_NOTHROW(check_slices_number(shape_type{5},slice_container_type(1)));
//         REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slice_container_type(1)));
//         REQUIRE_NOTHROW(check_slices_number(shape_type{5,6},slice_container_type(2)));
//         REQUIRE_THROWS_AS(check_slices_number(shape_type{5,6},slice_container_type(3)), gtensor::subscript_exception);
//         REQUIRE_THROWS_AS(check_slices_number(shape_type{5},slice_container_type(2)), gtensor::subscript_exception);
//     }
// }

