#include <tuple>
#include <vector>
#include <iostream>
#include "catch.hpp"
#include "test_config.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"

//test helpers
//test slice view helpers
TEST_CASE("test_make_slice_view_shape_element","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using index_type = config_type::index_type;
    using slice_type = gtensor::slice<index_type>;
    using nop_type = typename slice_type::nop_type;
    using rtag_type = typename slice_type::reduce_tag_type;
    using gtensor::detail::make_slice_view_shape_element;
    //0slice,1pshape_element,2expected
    using test_type = std::tuple<index_type, slice_type, index_type>;
    auto test_data = GENERATE(
        //pshape_element == 0
        //nop,nop,nop
        test_type{index_type{0},slice_type{},index_type{0}},
        //nop,nop,step
        test_type{index_type{0},slice_type{nop_type{},nop_type{},2},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},nop_type{},-1},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},nop_type{},-2},index_type{0}},
        //nop,stop,nop
        test_type{index_type{0},slice_type{nop_type{},0},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},3},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},-3},index_type{0}},
        //nop,stop,step
        test_type{index_type{0},slice_type{nop_type{},0,2},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},3,2},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},-3,2},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},0,-1},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},3,-1},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},-3,-1},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},0,-2},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},3,-2},index_type{0}},
        test_type{index_type{0},slice_type{nop_type{},-3,-2},index_type{0}},
        //start,nop,nop
        test_type{index_type{0},slice_type{0},index_type{0}},
        test_type{index_type{0},slice_type{3},index_type{0}},
        test_type{index_type{0},slice_type{-3},index_type{0}},
        //start,nop,step
        test_type{index_type{0},slice_type{0,nop_type{},2},index_type{0}},
        test_type{index_type{0},slice_type{3,nop_type{},2},index_type{0}},
        test_type{index_type{0},slice_type{-3,nop_type{},2},index_type{0}},
        test_type{index_type{0},slice_type{0,nop_type{},-1},index_type{0}},
        test_type{index_type{0},slice_type{3,nop_type{},-1},index_type{0}},
        test_type{index_type{0},slice_type{-3,nop_type{},-1},index_type{0}},
        test_type{index_type{0},slice_type{0,nop_type{},-2},index_type{0}},
        test_type{index_type{0},slice_type{3,nop_type{},-2},index_type{0}},
        test_type{index_type{0},slice_type{-3,nop_type{},-2},index_type{0}},
        //start,stop,nop
        test_type{index_type{0},slice_type{0,0},index_type{0}},
        test_type{index_type{0},slice_type{0,3},index_type{0}},
        test_type{index_type{0},slice_type{3,0},index_type{0}},
        test_type{index_type{0},slice_type{0,-3},index_type{0}},
        test_type{index_type{0},slice_type{-3,0},index_type{0}},
        test_type{index_type{0},slice_type{-3,3},index_type{0}},
        test_type{index_type{0},slice_type{3,-3},index_type{0}},
        //start,stop,step
        test_type{index_type{0},slice_type{0,0,2},index_type{0}},
        test_type{index_type{0},slice_type{0,3,2},index_type{0}},
        test_type{index_type{0},slice_type{3,0,2},index_type{0}},
        test_type{index_type{0},slice_type{0,-3,2},index_type{0}},
        test_type{index_type{0},slice_type{-3,0,2},index_type{0}},
        test_type{index_type{0},slice_type{-3,3,2},index_type{0}},
        test_type{index_type{0},slice_type{3,-3,2},index_type{0}},
        test_type{index_type{0},slice_type{0,0,-1},index_type{0}},
        test_type{index_type{0},slice_type{0,3,-1},index_type{0}},
        test_type{index_type{0},slice_type{3,0,-1},index_type{0}},
        test_type{index_type{0},slice_type{0,-3,-1},index_type{0}},
        test_type{index_type{0},slice_type{-3,0,-1},index_type{0}},
        test_type{index_type{0},slice_type{-3,3,-1},index_type{0}},
        test_type{index_type{0},slice_type{3,-3,-1},index_type{0}},
        test_type{index_type{0},slice_type{0,0,-2},index_type{0}},
        test_type{index_type{0},slice_type{0,3,-2},index_type{0}},
        test_type{index_type{0},slice_type{3,0,-2},index_type{0}},
        test_type{index_type{0},slice_type{0,-3,-2},index_type{0}},
        test_type{index_type{0},slice_type{-3,0,-2},index_type{0}},
        test_type{index_type{0},slice_type{-3,3,-2},index_type{0}},
        test_type{index_type{0},slice_type{3,-3,-2},index_type{0}},
        //reduce slice
        test_type{index_type{0},slice_type{0,rtag_type{}},index_type{0}},
        test_type{index_type{0},slice_type{3,rtag_type{}},index_type{0}},
        test_type{index_type{0},slice_type{-3,rtag_type{}},index_type{0}},
        //pshape_element>0
        //nop,nop,nop
        test_type{index_type{10},slice_type{},index_type{10}},
        //nop,nop,step
        test_type{index_type{10},slice_type{nop_type{},nop_type{},2},index_type{5}},
        test_type{index_type{10},slice_type{nop_type{},nop_type{},3},index_type{4}},
        test_type{index_type{10},slice_type{nop_type{},nop_type{},-1},index_type{10}},
        test_type{index_type{10},slice_type{nop_type{},nop_type{},-2},index_type{5}},
        test_type{index_type{10},slice_type{nop_type{},nop_type{},-3},index_type{4}},
        //nop,stop,nop
        test_type{index_type{10},slice_type{nop_type{},0},index_type{0}},
        test_type{index_type{10},slice_type{nop_type{},3},index_type{3}},
        test_type{index_type{10},slice_type{nop_type{},-3},index_type{7}},
        //nop,stop,step
        test_type{index_type{10},slice_type{nop_type{},0,2},index_type{0}},
        test_type{index_type{10},slice_type{nop_type{},3,2},index_type{2}},
        test_type{index_type{10},slice_type{nop_type{},-3,2},index_type{4}},
        test_type{index_type{10},slice_type{nop_type{},0,-1},index_type{9}},
        test_type{index_type{10},slice_type{nop_type{},3,-1},index_type{6}},
        test_type{index_type{10},slice_type{nop_type{},-3,-1},index_type{2}},
        test_type{index_type{10},slice_type{nop_type{},0,-2},index_type{5}},
        test_type{index_type{10},slice_type{nop_type{},3,-2},index_type{3}},
        test_type{index_type{10},slice_type{nop_type{},-3,-2},index_type{1}},
        test_type{index_type{10},slice_type{nop_type{},0,-3},index_type{3}},
        test_type{index_type{10},slice_type{nop_type{},3,-3},index_type{2}},
        test_type{index_type{10},slice_type{nop_type{},-3,-3},index_type{1}},
        //start,nop,nop
        test_type{index_type{10},slice_type{0},index_type{10}},
        test_type{index_type{10},slice_type{3},index_type{7}},
        test_type{index_type{10},slice_type{-3},index_type{3}},
        //start,nop,step
        test_type{index_type{10},slice_type{0,nop_type{},2},index_type{5}},
        test_type{index_type{10},slice_type{3,nop_type{},2},index_type{4}},
        test_type{index_type{10},slice_type{-3,nop_type{},2},index_type{2}},
        test_type{index_type{10},slice_type{0,nop_type{},3},index_type{4}},
        test_type{index_type{10},slice_type{3,nop_type{},3},index_type{3}},
        test_type{index_type{10},slice_type{-3,nop_type{},3},index_type{1}},
        test_type{index_type{10},slice_type{0,nop_type{},-1},index_type{1}},
        test_type{index_type{10},slice_type{3,nop_type{},-1},index_type{4}},
        test_type{index_type{10},slice_type{-3,nop_type{},-1},index_type{8}},
        test_type{index_type{10},slice_type{0,nop_type{},-2},index_type{1}},
        test_type{index_type{10},slice_type{3,nop_type{},-2},index_type{2}},
        test_type{index_type{10},slice_type{-3,nop_type{},-2},index_type{4}},
        test_type{index_type{10},slice_type{0,nop_type{},-3},index_type{1}},
        test_type{index_type{10},slice_type{3,nop_type{},-3},index_type{2}},
        test_type{index_type{10},slice_type{-3,nop_type{},-3},index_type{3}},
        //start,stop,nop
        test_type{index_type{10},slice_type{0,0},index_type{0}},
        test_type{index_type{10},slice_type{0,3},index_type{3}},
        test_type{index_type{10},slice_type{3,0},index_type{0}},
        test_type{index_type{10},slice_type{0,-3},index_type{7}},
        test_type{index_type{10},slice_type{-3,0},index_type{0}},
        test_type{index_type{10},slice_type{-3,3},index_type{0}},
        test_type{index_type{10},slice_type{3,-3},index_type{4}},
        //start,stop,step
        test_type{index_type{10},slice_type{0,0,2},index_type{0}},
        test_type{index_type{10},slice_type{0,3,2},index_type{2}},
        test_type{index_type{10},slice_type{3,0,2},index_type{0}},
        test_type{index_type{10},slice_type{0,-3,2},index_type{4}},
        test_type{index_type{10},slice_type{-3,0,2},index_type{0}},
        test_type{index_type{10},slice_type{-3,3,2},index_type{0}},
        test_type{index_type{10},slice_type{3,-3,2},index_type{2}},
        test_type{index_type{10},slice_type{0,0,3},index_type{0}},
        test_type{index_type{10},slice_type{0,3,3},index_type{1}},
        test_type{index_type{10},slice_type{3,0,3},index_type{0}},
        test_type{index_type{10},slice_type{0,-3,3},index_type{3}},
        test_type{index_type{10},slice_type{-3,0,3},index_type{0}},
        test_type{index_type{10},slice_type{-3,3,3},index_type{0}},
        test_type{index_type{10},slice_type{3,-3,3},index_type{2}},
        test_type{index_type{10},slice_type{0,0,-1},index_type{0}},
        test_type{index_type{10},slice_type{0,3,-1},index_type{0}},
        test_type{index_type{10},slice_type{3,0,-1},index_type{3}},
        test_type{index_type{10},slice_type{0,-3,-1},index_type{0}},
        test_type{index_type{10},slice_type{-3,0,-1},index_type{7}},
        test_type{index_type{10},slice_type{-3,3,-1},index_type{4}},
        test_type{index_type{10},slice_type{3,-3,-1},index_type{0}},
        test_type{index_type{10},slice_type{0,0,-2},index_type{0}},
        test_type{index_type{10},slice_type{0,3,-2},index_type{0}},
        test_type{index_type{10},slice_type{3,0,-2},index_type{2}},
        test_type{index_type{10},slice_type{0,-3,-2},index_type{0}},
        test_type{index_type{10},slice_type{-3,0,-2},index_type{4}},
        test_type{index_type{10},slice_type{-3,3,-2},index_type{2}},
        test_type{index_type{10},slice_type{3,-3,-2},index_type{0}},
        test_type{index_type{10},slice_type{0,0,-3},index_type{0}},
        test_type{index_type{10},slice_type{0,3,-3},index_type{0}},
        test_type{index_type{10},slice_type{3,0,-3},index_type{1}},
        test_type{index_type{10},slice_type{0,-3,-3},index_type{0}},
        test_type{index_type{10},slice_type{-3,0,-3},index_type{3}},
        test_type{index_type{10},slice_type{-3,3,-3},index_type{2}},
        test_type{index_type{10},slice_type{3,-3,-3},index_type{0}},
        //reduce slice
        test_type{index_type{10},slice_type{0,rtag_type{}},index_type{1}},
        test_type{index_type{10},slice_type{3,rtag_type{}},index_type{1}},
        test_type{index_type{10},slice_type{-3,rtag_type{}},index_type{1}},
        test_type{index_type{10},slice_type{10,rtag_type{}},index_type{0}},
        test_type{index_type{10},slice_type{9,rtag_type{}},index_type{1}},
        test_type{index_type{10},slice_type{-10,rtag_type{}},index_type{1}},
        test_type{index_type{10},slice_type{-11,rtag_type{}},index_type{0}}
    );
    auto pshape_element = std::get<0>(test_data);
    auto slice = std::get<1>(test_data);
    auto expected = std::get<2>(test_data);
    auto result = make_slice_view_shape_element(pshape_element,slice);
    REQUIRE(result == expected);
}

TEST_CASE("test_make_slice_view_shape","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using index_type = config_type::index_type;
    using dim_type = config_type::dim_type;
    using slice_type = gtensor::slice<index_type>;
    using nop_type = typename slice_type::nop_type;
    using rtag_type = typename slice_type::reduce_tag_type;
    using gtensor::detail::make_slice_view_shape;
    using helpers_for_testing::apply_by_element;
    //0pshape,1res_dim,2subs,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{}, shape_type{11}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{-20,0,1}}, shape_type{0}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{0,-20,1}}, shape_type{0}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{20,0,1}}, shape_type{0}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{7,3,1}}, shape_type{0}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{20,11,-1}}, shape_type{0}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{-12,-20,-1}}, shape_type{0}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{10,20,-1}}, shape_type{0}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{-12,0,-1}}, shape_type{0}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{7,15,1}}, shape_type{4}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{5,20,1}}, shape_type{6}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{-15,5,1}}, shape_type{5}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{-20,5,1}}, shape_type{5}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{7,-20,-1}}, shape_type{8}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{7,-8,-1}}, shape_type{4}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{15,5,-1}}, shape_type{5}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{15,-5,-1}}, shape_type{4}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{0,20,1}}, shape_type{11}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{0,11,1}}, shape_type{11}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{0,11,2}}, shape_type{6}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{3,11,1}}, shape_type{8}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{3,9,3}}, shape_type{2}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{3,11,2}}, shape_type{4}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{5,5,1}}, shape_type{0}),
        std::make_tuple(shape_type{11},dim_type{1},std::vector<slice_type>{slice_type{5,5,2}}, shape_type{0}),
        std::make_tuple(shape_type{2,4,3},dim_type{3},std::vector<slice_type>{}, shape_type{2,4,3}),
        std::make_tuple(shape_type{2,4,3},dim_type{3},std::array<slice_type,2>{slice_type{0,2},slice_type{0,4}}, shape_type{2,4,3}),
        std::make_tuple(shape_type{2,4,3},dim_type{3},std::initializer_list<slice_type>{slice_type{1},slice_type{2}}, shape_type{1,2,3}),
        std::make_tuple(shape_type{2,4,3},dim_type{3},std::vector<slice_type>{slice_type{0,2},slice_type{0,4},slice_type{0,3}}, shape_type{2,4,3}),
        std::make_tuple(shape_type{1,10,10},dim_type{3},std::vector<slice_type>{slice_type{0,1,1},slice_type{9,-1,-1}}, shape_type{1,0,10}),
        std::make_tuple(shape_type{1,10,10},dim_type{3},std::vector<slice_type>{slice_type{0,1,1},slice_type{9,-11,-1}}, shape_type{1,10,10}),
        std::make_tuple(shape_type{1,10,10},dim_type{3},std::vector<slice_type>{slice_type{0,1,1},slice_type{9,nop_type{},-1}}, shape_type{1,10,10}),
        std::make_tuple(shape_type{10,1,10},dim_type{3},std::vector<slice_type>{slice_type{9,3,-2}}, shape_type{3,1,10}),
        //reduce
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{0,rtag_type{}},slice_type{}}, shape_type{4}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{1,rtag_type{}},slice_type{}}, shape_type{4}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{2,rtag_type{}},slice_type{}}, shape_type{4}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{0,rtag_type{}},slice_type{nop_type{},nop_type{},-1}}, shape_type{4}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{1,rtag_type{}},slice_type{1,3}}, shape_type{2}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{2,rtag_type{}},slice_type{nop_type{},nop_type{},3}}, shape_type{2}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{},slice_type{0,rtag_type{}}}, shape_type{3}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{},slice_type{1,rtag_type{}}}, shape_type{3}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{},slice_type{2,rtag_type{}}}, shape_type{3}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{},slice_type{3,rtag_type{}}}, shape_type{3}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{1},slice_type{0,rtag_type{}}}, shape_type{2}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{0,1},slice_type{1,rtag_type{}}}, shape_type{1}),
        std::make_tuple(shape_type{3,4},dim_type{1},std::vector<slice_type>{slice_type{nop_type{},nop_type{},-2},slice_type{2,rtag_type{}}}, shape_type{2}),
        std::make_tuple(shape_type{2,4,3},dim_type{2},std::vector<slice_type>{slice_type{1,rtag_type{}}}, shape_type{4,3}),
        std::make_tuple(shape_type{2,4,3},dim_type{2},std::vector<slice_type>{slice_type{}, slice_type{1,rtag_type{}}}, shape_type{2,3}),
        std::make_tuple(shape_type{2,4,3},dim_type{2},std::vector<slice_type>{slice_type{},slice_type{},slice_type{1,rtag_type{}}}, shape_type{2,4}),
        std::make_tuple(shape_type{2,4,3},dim_type{1},std::vector<slice_type>{slice_type{1,rtag_type{}},slice_type{2,rtag_type{}}}, shape_type{3}),
        std::make_tuple(shape_type{2,4,3},dim_type{1},std::vector<slice_type>{slice_type{1,rtag_type{}},slice_type{},slice_type{2,rtag_type{}}}, shape_type{4}),
        std::make_tuple(shape_type{2,4,3},dim_type{1},std::vector<slice_type>{slice_type{1,rtag_type{}},slice_type{nop_type{},nop_type{},3},slice_type{2,rtag_type{}}}, shape_type{2})
    );
    auto test = [](const auto& t){
        auto pshape = std::get<0>(t);
        auto res_dim = std::get<1>(t);
        auto subs = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = make_slice_view_shape(pshape,res_dim,subs);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_make_slice_view_offset","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using index_type = config_type::index_type;
    using slice_type = gtensor::slice<index_type>;
    using nop_type = typename slice_type::nop_type;
    using rtag_type = typename slice_type::reduce_tag_type;
    using gtensor::detail::make_slice_view_offset;
    using helpers_for_testing::apply_by_element;
    //0pshape,1pstrides,2subs,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{}, index_type{0}),
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{slice_type{}}, index_type{0}),
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{slice_type{3}}, index_type{3}),
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{slice_type{-3}}, index_type{8}),
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{slice_type{-20}}, index_type{0}),
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{slice_type{20}}, index_type{20}),
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{slice_type{20,30,-1}}, index_type{10}),
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{slice_type{-20,30,-1}}, index_type{-9}),
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{slice_type{nop_type{},nop_type{},-1}}, index_type{10}),
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{slice_type{3,nop_type{},-1}}, index_type{3}),
        std::make_tuple(shape_type{11},shape_type{1},std::vector<slice_type>{slice_type{-3,nop_type{},-1}}, index_type{8}),
        std::make_tuple(shape_type{3,4},shape_type{4,1},std::vector<slice_type>{}, index_type{0}),
        std::make_tuple(shape_type{3,4},shape_type{4,1},std::vector<slice_type>{slice_type{}}, index_type{0}),
        std::make_tuple(shape_type{3,4},shape_type{4,1},std::vector<slice_type>{slice_type{3}}, index_type{12}),
        std::make_tuple(shape_type{3,4},shape_type{4,1},std::vector<slice_type>{slice_type{-3}}, index_type{0}),
        std::make_tuple(shape_type{3,4},shape_type{4,1},std::array<slice_type,2>{slice_type{},slice_type{}}, index_type{0}),
        std::make_tuple(shape_type{3,4},shape_type{4,1},std::initializer_list<slice_type>{slice_type{},slice_type{3}}, index_type{3}),
        std::make_tuple(shape_type{3,4},shape_type{4,1},std::vector<slice_type>{slice_type{},slice_type{-3}}, index_type{1}),
        std::make_tuple(shape_type{3,4},shape_type{4,1},std::vector<slice_type>{slice_type{1},slice_type{1}}, index_type{5}),
        std::make_tuple(shape_type{3,4},shape_type{4,1},std::vector<slice_type>{slice_type{-1},slice_type{-1}}, index_type{11}),
        std::make_tuple(shape_type{2,3,4},shape_type{12,4,1},std::vector<slice_type>{}, index_type{0}),
        std::make_tuple(shape_type{2,3,4},shape_type{12,4,1},std::vector<slice_type>{slice_type{1}}, index_type{12}),
        std::make_tuple(shape_type{2,3,4},shape_type{12,4,1},std::vector<slice_type>{slice_type{1},slice_type{2}}, index_type{20}),
        std::make_tuple(shape_type{2,3,4},shape_type{12,4,1},std::vector<slice_type>{slice_type{1},slice_type{2},slice_type{3}}, index_type{23}),
        std::make_tuple(shape_type{2,3,4},shape_type{12,4,1},std::vector<slice_type>{slice_type{0,rtag_type{}}}, index_type{0}),
        std::make_tuple(shape_type{2,3,4},shape_type{12,4,1},std::vector<slice_type>{slice_type{1,rtag_type{}}}, index_type{12}),
        std::make_tuple(shape_type{2,3,4},shape_type{12,4,1},std::vector<slice_type>{slice_type{1,rtag_type{}},slice_type{2,rtag_type{}}}, index_type{20}),
        std::make_tuple(shape_type{2,3,4},shape_type{12,4,1},std::vector<slice_type>{slice_type{},slice_type{1,rtag_type{}},slice_type{2,rtag_type{}}}, index_type{6})
    );
    auto test = [](const auto& t){
        auto pshape = std::get<0>(t);
        auto pstrides = std::get<1>(t);
        auto subs = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = make_slice_view_offset(pshape,pstrides,subs);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_make_slice_view_cstrides","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using slice_type = gtensor::slice<index_type>;
    using nop_type = typename slice_type::nop_type;
    using rtag_type = typename slice_type::reduce_tag_type;
    using gtensor::detail::make_slice_view_cstrides;
    using helpers_for_testing::apply_by_element;
    //0pstrides,1res_dim,2subs,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{1},dim_type{1},std::vector<slice_type>{}, shape_type{1}),
        std::make_tuple(shape_type{1},dim_type{1},std::vector<slice_type>{slice_type{nop_type{},nop_type{},2}}, shape_type{2}),
        std::make_tuple(shape_type{1},dim_type{1},std::vector<slice_type>{slice_type{nop_type{},nop_type{},-2}}, shape_type{-2}),
        std::make_tuple(shape_type{12,4,1},dim_type{3},std::vector<slice_type>{}, shape_type{12,4,1}),
        std::make_tuple(shape_type{12,4,1},dim_type{3},std::vector<slice_type>{slice_type{nop_type{},nop_type{},-1}}, shape_type{-12,4,1}),
        std::make_tuple(shape_type{12,4,1},dim_type{3},std::vector<slice_type>{slice_type{nop_type{},nop_type{},2}}, shape_type{24,4,1}),
        std::make_tuple(shape_type{12,4,1},dim_type{3},std::vector<slice_type>{slice_type{nop_type{},nop_type{},-2}}, shape_type{-24,4,1}),
        std::make_tuple(shape_type{12,4,1},dim_type{3},std::vector<slice_type>{slice_type{nop_type{},nop_type{},2},slice_type{nop_type{},nop_type{},2}}, shape_type{24,8,1}),
        std::make_tuple(
            shape_type{12,4,1},
            dim_type{3},
            std::vector<slice_type>{slice_type{nop_type{},nop_type{},2},slice_type{nop_type{},nop_type{},2},slice_type{nop_type{},nop_type{},2}},
            shape_type{24,8,2}
        ),
        std::make_tuple(
            shape_type{12,4,1},
            dim_type{3},
            std::vector<slice_type>{slice_type{nop_type{},nop_type{},-2},slice_type{nop_type{},nop_type{},-2},slice_type{nop_type{},nop_type{},-2}},
            shape_type{-24,-8,-2}
        ),
        std::make_tuple(
            shape_type{12,4,1},
            dim_type{2},
            std::vector<slice_type>{slice_type{1,rtag_type{}},slice_type{nop_type{},nop_type{},2},slice_type{nop_type{},nop_type{},2}},
            shape_type{8,2}
        ),
        std::make_tuple(
            shape_type{12,4,1},
            dim_type{1},
            std::vector<slice_type>{slice_type{0,rtag_type{}},slice_type{nop_type{},nop_type{},2},slice_type{0,rtag_type{}}},
            shape_type{8}
        )
    );
    auto test = [](const auto& t){
        auto pstrides = std::get<0>(t);
        auto res_dim = std::get<1>(t);
        auto subs = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = make_slice_view_cstrides(pstrides,res_dim,subs);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_check_slice_view_args","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using index_type = config_type::index_type;
    using slice_type = gtensor::slice<index_type>;
    using rtag_type = typename slice_type::reduce_tag_type;
    using gtensor::subscript_exception;
    using gtensor::detail::check_slice_view_args;
    using helpers_for_testing::apply_by_element;

    SECTION("test_check_slice_view_args_exception")
    {
        //0pshape,2subs
        auto test_data = std::make_tuple(
            std::make_tuple(shape_type{}, std::vector<slice_type>{slice_type{}}),
            std::make_tuple(shape_type{}, std::vector<slice_type>{slice_type{0,rtag_type{}}}),
            std::make_tuple(shape_type{}, std::vector<slice_type>{slice_type{-1,rtag_type{}}}),
            std::make_tuple(shape_type{0}, std::vector<slice_type>{slice_type{0,rtag_type{}}}),
            std::make_tuple(shape_type{0}, std::vector<slice_type>{slice_type{-1,rtag_type{}}}),
            std::make_tuple(shape_type{0}, std::vector<slice_type>{slice_type{},slice_type{}}),
            std::make_tuple(shape_type{3}, std::vector<slice_type>{slice_type{3,rtag_type{}}}),
            std::make_tuple(shape_type{3}, std::vector<slice_type>{slice_type{-4,rtag_type{}}}),
            std::make_tuple(shape_type{3}, std::vector<slice_type>{slice_type{},slice_type{}}),
            std::make_tuple(shape_type{3,4}, std::vector<slice_type>{slice_type{},slice_type{},slice_type{}}),
            std::make_tuple(shape_type{3,4}, std::vector<slice_type>{slice_type{4,rtag_type{}}}),
            std::make_tuple(shape_type{3,4}, std::vector<slice_type>{slice_type{-4,rtag_type{}}}),
            std::make_tuple(shape_type{3,4}, std::vector<slice_type>{slice_type{}, slice_type{5,rtag_type{}}}),
            std::make_tuple(shape_type{3,4}, std::vector<slice_type>{slice_type{}, slice_type{-5,rtag_type{}}}),
            std::make_tuple(shape_type{2,3,4}, std::vector<slice_type>{slice_type{3,rtag_type{}},slice_type{}, slice_type{0,rtag_type{}}}),
            std::make_tuple(shape_type{2,3,4}, std::vector<slice_type>{slice_type{-3,rtag_type{}},slice_type{}, slice_type{0,rtag_type{}}}),
            std::make_tuple(shape_type{2,3,4}, std::vector<slice_type>{slice_type{0,rtag_type{}},slice_type{}, slice_type{5,rtag_type{}}}),
            std::make_tuple(shape_type{2,3,4}, std::vector<slice_type>{slice_type{0,rtag_type{}},slice_type{}, slice_type{-5,rtag_type{}}}),
            std::make_tuple(shape_type{2,3,4}, std::vector<slice_type>{slice_type{},slice_type{4,rtag_type{}}, slice_type{}}),
            std::make_tuple(shape_type{2,3,4}, std::vector<slice_type>{slice_type{},slice_type{-4,rtag_type{}}, slice_type{}})
        );
        auto test = [](const auto& t){
            auto pshape = std::get<0>(t);
            auto subs = std::get<1>(t);
            REQUIRE_THROWS_AS(check_slice_view_args(pshape,subs),subscript_exception);
        };
        apply_by_element(test, test_data);
    }
}

//test subdim view helpers
TEST_CASE("test_make_subdim_view_shape","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using dim_type = config_type::dim_type;
    using gtensor::detail::make_subdim_view_shape;
    using test_type = std::tuple<shape_type, dim_type, shape_type>;
    //0pshape,1subs_number,2expected
    auto test_data = GENERATE(
        test_type{shape_type{11,1},dim_type{0}, shape_type{11,1}},
        test_type{shape_type{11,1},dim_type{1}, shape_type{1}},
        test_type{shape_type{1,11},dim_type{1}, shape_type{11}},
        test_type{shape_type{3,4,10,2},dim_type{2}, shape_type{10,2}}
    );
    auto pshape = std::get<0>(test_data);
    auto subs_number = std::get<1>(test_data);
    auto expected = std::get<2>(test_data);
    auto result = make_subdim_view_shape(pshape, subs_number);
    REQUIRE(result == expected);
}

TEST_CASE("test_make_subdim_view_offset","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using index_type = config_type::index_type;
    using gtensor::detail::make_subdim_view_offset;
    using helpers_for_testing::apply_by_element;
    //0pshape,1pstrides,2subs,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{0},shape_type{1},std::make_tuple(),index_type{0}),
        std::make_tuple(shape_type{1},shape_type{1},std::make_tuple(),index_type{0}),
        std::make_tuple(shape_type{2,4},shape_type{4,1},std::make_tuple(),index_type{0}),
        std::make_tuple(shape_type{2,4},shape_type{4,1},std::make_tuple(index_type{0}),index_type{0}),
        std::make_tuple(shape_type{2,4},shape_type{4,1},std::make_tuple(index_type{1}),index_type{4}),
        std::make_tuple(shape_type{2,4},shape_type{4,1},std::make_tuple(index_type{2}),index_type{8}),
        std::make_tuple(shape_type{3,2,4},shape_type{8,4,1},std::make_tuple(index_type{0}),index_type{0}),
        std::make_tuple(shape_type{3,2,4},shape_type{8,4,1},std::make_tuple(index_type{1}),index_type{8}),
        std::make_tuple(shape_type{3,2,4},shape_type{8,4,1},std::make_tuple(index_type{0},index_type{0}),index_type{0}),
        std::make_tuple(shape_type{3,2,4},shape_type{8,4,1},std::make_tuple(index_type{0},index_type{1}),index_type{4}),
        std::make_tuple(shape_type{3,2,4},shape_type{8,4,1},std::make_tuple(index_type{1},index_type{0}),index_type{8}),
        std::make_tuple(shape_type{3,2,4},shape_type{8,4,1},std::make_tuple(index_type{1},index_type{1}),index_type{12})
    );
    using container_type = std::vector<index_type>;
    auto test = [](const auto& t){
        auto pshape = std::get<0>(t);
        auto pstrides = std::get<1>(t);
        auto subs = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto make_container = [](const auto&...subs_){
            return container_type{subs_...};
        };
        auto subs_container = std::apply(make_container, subs);
        auto result = make_subdim_view_offset(pshape, pstrides, subs_container);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_check_subdim_args","[test_check_subdim_subs]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using index_type = config_type::index_type;
    using gtensor::subscript_exception;
    using gtensor::detail::check_subdim_args;
    using helpers_for_testing::apply_by_element;
    SECTION("test_check_subdim_args_nothrow")
    {
        //0pshape,1subs
        auto test_data = std::make_tuple(
            std::make_tuple(shape_type{}, std::make_tuple()),
            std::make_tuple(shape_type{0}, std::make_tuple()),
            std::make_tuple(shape_type{1}, std::make_tuple()),
            std::make_tuple(shape_type{1}, std::make_tuple(0)),
            std::make_tuple(shape_type{1}, std::make_tuple(-1)),
            std::make_tuple(shape_type{2}, std::make_tuple(0)),
            std::make_tuple(shape_type{2}, std::make_tuple(1)),
            std::make_tuple(shape_type{2}, std::make_tuple(-1)),
            std::make_tuple(shape_type{2}, std::make_tuple(-2)),
            std::make_tuple(shape_type{1,1}, std::make_tuple()),
            std::make_tuple(shape_type{1,1}, std::make_tuple(index_type{0})),
            std::make_tuple(shape_type{1,1}, std::make_tuple(index_type{-1})),
            std::make_tuple(shape_type{1,1}, std::make_tuple(index_type{0},index_type{0})),
            std::make_tuple(shape_type{1,1}, std::make_tuple(index_type{-1},index_type{-1})),
            std::make_tuple(shape_type{1,1}, std::make_tuple(index_type{0},index_type{-1})),
            std::make_tuple(shape_type{1,1}, std::make_tuple(index_type{-1},index_type{0})),
            std::make_tuple(shape_type{1,1,1}, std::make_tuple(index_type{0})),
            std::make_tuple(shape_type{1,1,1}, std::make_tuple(index_type{-1})),
            std::make_tuple(shape_type{1,1,1}, std::make_tuple(index_type{0},index_type{0})),
            std::make_tuple(shape_type{1,1,1}, std::make_tuple(index_type{0},index_type{-1})),
            std::make_tuple(shape_type{1,1,1}, std::make_tuple(index_type{-1},index_type{0})),
            std::make_tuple(shape_type{1,1,1}, std::make_tuple(index_type{0},index_type{0},index_type{0})),
            std::make_tuple(shape_type{1,1,1}, std::make_tuple(index_type{-1},index_type{-1},index_type{-1})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{0})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{1})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{-1})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{-2})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{-1},index_type{0})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{-1},index_type{1})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{-1},index_type{-1})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{-1},index_type{-2})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{0},index_type{0})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{-1},index_type{-2},index_type{-3})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{1},index_type{2}))
        );
        using container_type = shape_type;
        auto test = [](const auto& t){
            auto pshape = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto make_container = [](const auto&...subs_){
                return container_type{subs_...};
            };
            auto subs_container = std::apply(make_container, subs);
            REQUIRE_NOTHROW(check_subdim_args(pshape,subs_container));
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_subdim_args_exception")
    {
        //0pshape,1subs
        auto test_data = std::make_tuple(
            std::make_tuple(shape_type{}, std::make_tuple(index_type{0})),
            std::make_tuple(shape_type{0}, std::make_tuple(index_type{0})),
            std::make_tuple(shape_type{0}, std::make_tuple(index_type{-1})),
            std::make_tuple(shape_type{0}, std::make_tuple(index_type{1})),
            std::make_tuple(shape_type{1}, std::make_tuple(index_type{1})),
            std::make_tuple(shape_type{1}, std::make_tuple(index_type{2})),
            std::make_tuple(shape_type{1}, std::make_tuple(index_type{-2})),
            std::make_tuple(shape_type{2}, std::make_tuple(index_type{2})),
            std::make_tuple(shape_type{2}, std::make_tuple(index_type{3})),
            std::make_tuple(shape_type{2}, std::make_tuple(index_type{-3})),
            std::make_tuple(shape_type{1}, std::make_tuple(index_type{0},index_type{0})),
            std::make_tuple(shape_type{1,2}, std::make_tuple(index_type{1})),
            std::make_tuple(shape_type{1,2}, std::make_tuple(index_type{-2})),
            std::make_tuple(shape_type{3,4}, std::make_tuple(index_type{3})),
            std::make_tuple(shape_type{3,4}, std::make_tuple(index_type{-4})),
            std::make_tuple(shape_type{1,2}, std::make_tuple(index_type{0},index_type{2})),
            std::make_tuple(shape_type{1,2}, std::make_tuple(index_type{0},index_type{-3})),
            std::make_tuple(shape_type{1,2}, std::make_tuple(index_type{1},index_type{0})),
            std::make_tuple(shape_type{1,2}, std::make_tuple(index_type{-2},index_type{0})),
            std::make_tuple(shape_type{1,2}, std::make_tuple(index_type{0},index_type{0},index_type{0})),
            std::make_tuple(shape_type{1,1,1}, std::make_tuple(index_type{1})),
            std::make_tuple(shape_type{1,1,1}, std::make_tuple(index_type{0},index_type{-2})),
            std::make_tuple(shape_type{1,1,1}, std::make_tuple(index_type{-2},index_type{0})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{2})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{-3})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{0},index_type{3})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{0},index_type{-4})),
            std::make_tuple(shape_type{1,2,3}, std::make_tuple(index_type{0},index_type{0},index_type{0},index_type{}))
        );
        using container_type = shape_type;
        auto test = [](const auto& t){
            auto pshape = std::get<0>(t);
            auto subs = std::get<1>(t);
            auto make_container = [](const auto&...subs_){
                return container_type{subs_...};
            };
            auto subs_container = std::apply(make_container, subs);
            REQUIRE_THROWS_AS(check_subdim_args(pshape,subs_container), subscript_exception);
        };
        apply_by_element(test, test_data);
    }
}

//test reshape view helpers
TEST_CASE("test_make_reshape_view_shape","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using index_type = config_type::index_type;
    using gtensor::detail::make_reshape_view_shape;
    using helpers_for_testing::apply_by_element;
    //0pshape,1psize,2subs,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{0},index_type{0}, shape_type{}, shape_type{0}),
        std::make_tuple(shape_type{0},index_type{0}, shape_type{0}, shape_type{0}),
        std::make_tuple(shape_type{0},index_type{0}, shape_type{-1}, shape_type{0}),
        std::make_tuple(shape_type{0},index_type{0}, shape_type{1,0}, shape_type{1,0}),
        std::make_tuple(shape_type{0},index_type{0}, shape_type{1,-1}, shape_type{1,0}),
        std::make_tuple(shape_type{0},index_type{0}, shape_type{-1,1}, shape_type{0,1}),
        std::make_tuple(shape_type{0},index_type{0}, std::vector<int>{1,2,3,-1}, shape_type{1,2,3,0}),
        std::make_tuple(shape_type{0},index_type{0}, std::array<int,4>{1,-1,2,3}, shape_type{1,0,2,3}),
        std::make_tuple(shape_type{1,2,3,0},index_type{0}, shape_type{}, shape_type{1,2,3,0}),
        std::make_tuple(shape_type{1,2,3,0},index_type{0}, std::initializer_list<std::size_t>{2,0}, shape_type{2,0}),
        std::make_tuple(shape_type{1,2,3,0},index_type{0}, shape_type{2,-1}, shape_type{2,0}),
        std::make_tuple(shape_type{1,2,3,0},index_type{0}, shape_type{4,-1,2,8}, shape_type{4,0,2,8}),
        std::make_tuple(shape_type{11},index_type{11}, shape_type{}, shape_type{11}),
        std::make_tuple(shape_type{11},index_type{11}, shape_type{-1}, shape_type{11}),
        std::make_tuple(shape_type{11},index_type{11}, shape_type{1,-1}, shape_type{1,11}),
        std::make_tuple(shape_type{11},index_type{11}, shape_type{-1,1}, shape_type{11,1}),
        std::make_tuple(shape_type{11,1},index_type{11}, shape_type{}, shape_type{11,1}),
        std::make_tuple(shape_type{11,1},index_type{11}, shape_type{11}, shape_type{11}),
        std::make_tuple(shape_type{1,11},index_type{11}, shape_type{11,1}, shape_type{11,1}),
        std::make_tuple(shape_type{3,4,10,2},index_type{240}, shape_type{}, shape_type{3,4,10,2}),
        std::make_tuple(shape_type{3,4,10,2},index_type{240}, shape_type{-1}, shape_type{240}),
        std::make_tuple(shape_type{3,4,10,2},index_type{240}, shape_type{20,12}, shape_type{20,12}),
        std::make_tuple(shape_type{3,4,10,2},index_type{240}, shape_type{20,-1}, shape_type{20,12}),
        std::make_tuple(shape_type{3,4,10,2},index_type{240}, shape_type{-1,12}, shape_type{20,12}),
        std::make_tuple(shape_type{3,4,10,2},index_type{240}, shape_type{5,-1,2}, shape_type{5,24,2})
    );
    auto test = [](const auto& t){
        auto pshape = std::get<0>(t);
        auto psize = std::get<1>(t);
        auto subs = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = make_reshape_view_shape(pshape, psize, subs);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_check_reshape_args","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using index_type = config_type::index_type;
    using gtensor::subscript_exception;
    using gtensor::detail::check_reshape_args;
    using helpers_for_testing::apply_by_element;

    SECTION("test_check_reshape_args_nothrow")
    {
        //0psize,1subs
        auto test_data = std::make_tuple(
            std::make_tuple(index_type{0}, shape_type{}),
            std::make_tuple(index_type{0}, shape_type{0}),
            std::make_tuple(index_type{0}, shape_type{1,0}),
            std::make_tuple(index_type{0}, shape_type{5,0}),
            std::make_tuple(index_type{0}, shape_type{2,3,0}),
            std::make_tuple(index_type{0}, shape_type{-1}),
            std::make_tuple(index_type{0}, shape_type{1,-1}),
            std::make_tuple(index_type{0}, shape_type{2,-1,3}),
            std::make_tuple(index_type{5}, shape_type{}),
            std::make_tuple(index_type{5}, shape_type{5}),
            std::make_tuple(index_type{5}, shape_type{-1}),
            std::make_tuple(index_type{5}, shape_type{5,-1}),
            std::make_tuple(index_type{5}, shape_type{-1,5}),
            std::make_tuple(index_type{20}, shape_type{4,5}),
            std::make_tuple(index_type{20}, shape_type{-1,5}),
            std::make_tuple(index_type{20}, shape_type{4,-1}),
            std::make_tuple(index_type{20}, shape_type{1,20,-1}),
            std::make_tuple(index_type{20}, shape_type{2,10}),
            std::make_tuple(index_type{20}, shape_type{2,5,2,1}),
            std::make_tuple(index_type{20}, shape_type{2,-1,2,1})
        );
        auto test = [](const auto& t){
            auto psize = std::get<0>(t);
            auto subs = std::get<1>(t);
            REQUIRE_NOTHROW(check_reshape_args(psize,subs));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_check_reshape_args_exception")
    {
        //0psize,1subs
        auto test_data = std::make_tuple(
            std::make_tuple(index_type{0}, shape_type{5}),
            std::make_tuple(index_type{0}, shape_type{0,-1}),
            std::make_tuple(index_type{0}, shape_type{2,3,0,-1}),
            std::make_tuple(index_type{0}, shape_type{-1,0}),
            std::make_tuple(index_type{0}, shape_type{-1,0,2,3}),
            std::make_tuple(index_type{0}, shape_type{-1,3,2,-1}),
            std::make_tuple(index_type{5}, shape_type{0}),
            std::make_tuple(index_type{5}, shape_type{1,-1,5,-1}),
            std::make_tuple(index_type{5}, shape_type{5,0}),
            std::make_tuple(index_type{5}, shape_type{3,2}),
            std::make_tuple(index_type{5}, shape_type{3,-1}),
            std::make_tuple(index_type{60}, shape_type{70}),
            std::make_tuple(index_type{60}, shape_type{2,3,2,4}),
            std::make_tuple(index_type{60}, shape_type{2,-1,2,4})
        );
        auto test = [](const auto& t){
            auto psize = std::get<0>(t);
            auto subs = std::get<1>(t);
            REQUIRE_THROWS_AS(check_reshape_args(psize,subs), subscript_exception);
        };
        apply_by_element(test,test_data);
    }
}

//test transpose view helpers
TEST_CASE("test_make_transpose_view_shape","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_transpose_view_shape;
    using helpers_for_testing::apply_by_element;
    //0pshape,1subs,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{3}, std::vector<int>{}, shape_type{3}),
        std::make_tuple(shape_type{3}, std::vector<int>{0}, shape_type{3}),
        std::make_tuple(shape_type{3,2}, std::vector<int>{}, shape_type{2,3}),
        std::make_tuple(shape_type{3,2}, std::array<int,2>{0,1}, shape_type{3,2}),
        std::make_tuple(shape_type{3,2}, std::vector<std::size_t>{1,0}, shape_type{2,3}),
        std::make_tuple(shape_type{4,3,2,2}, std::vector<int>{}, shape_type{2,2,3,4}),
        std::make_tuple(shape_type{4,3,2,2}, std::vector<int>{3,1,0,2}, shape_type{2,3,4,2}),
        std::make_tuple(shape_type{4,3,2,2}, std::vector<int>{3,1,2,0}, shape_type{2,3,2,4})
    );
    auto test = [](const auto& t){
        auto pshape = std::get<0>(t);
        auto subs = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_transpose_view_shape(pshape, subs);
        REQUIRE(expected == result);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_check_transpose_args","[test_view_factory]"){
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using gtensor::subscript_exception;
    using gtensor::subscript_exception;
    using gtensor::detail::check_transpose_args;
    using gtensor::detail::check_transpose_args_variadic;
    using helpers_for_testing::apply_by_element;

    SECTION("test_check_transpose_args_nothrow")
    {
        //0pdim,1subs
        auto test_data = std::make_tuple(
            std::make_tuple(dim_type{1},std::vector<int>{}),
            std::make_tuple(dim_type{1},std::vector<int>{0}),
            std::make_tuple(dim_type{3},std::vector<int>{0,1,2}),
            std::make_tuple(dim_type{3},std::vector<int>{0,2,1}),
            std::make_tuple(dim_type{3},std::vector<int>{2,1,0}),
            std::make_tuple(dim_type{3},std::vector<int>{2,0,1}),
            std::make_tuple(dim_type{3},std::vector<int>{1,0,2}),
            std::make_tuple(dim_type{3},std::vector<int>{1,0,2})
        );
        auto test = [](const auto& t){
            auto pdim = std::get<0>(t);
            auto subs = std::get<1>(t);
            REQUIRE_NOTHROW(check_transpose_args(pdim,subs));
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_transpose_args_exception")
    {
        //0pdim,1subs
        auto test_data = std::make_tuple(
            std::make_tuple(dim_type{1},std::vector<int>{-1}),
            std::make_tuple(dim_type{1},std::vector<int>{1,1,-1}),
            std::make_tuple(dim_type{1},std::vector<int>{0,1}),
            std::make_tuple(dim_type{2},std::vector<int>{0,0}),
            std::make_tuple(dim_type{3},std::vector<int>{0,1,2,3}),
            std::make_tuple(dim_type{3},std::vector<int>{0,1,1}),
            std::make_tuple(dim_type{3},std::vector<int>{0,2,2}),
            std::make_tuple(dim_type{3},std::vector<int>{0,0,1}),
            std::make_tuple(dim_type{3},std::vector<int>{2,1,1})
        );
        auto test = [](const auto& t){
            auto pdim = std::get<0>(t);
            auto subs = std::get<1>(t);
            REQUIRE_THROWS_AS(check_transpose_args(pdim,subs), subscript_exception);
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_transpose_args_variadic"){
        REQUIRE_NOTHROW(check_transpose_args_variadic());
        REQUIRE_NOTHROW(check_transpose_args_variadic(0));
        REQUIRE_NOTHROW(check_transpose_args_variadic(0,0));
        REQUIRE_NOTHROW(check_transpose_args_variadic(0,1));
        REQUIRE_NOTHROW(check_transpose_args_variadic(1,2,3));
        REQUIRE_THROWS_AS(check_transpose_args_variadic(-1), subscript_exception);
        REQUIRE_THROWS_AS(check_transpose_args_variadic(0,-1), subscript_exception);
        REQUIRE_THROWS_AS(check_transpose_args_variadic(2,-1,3), subscript_exception);
        REQUIRE_THROWS_AS(check_transpose_args_variadic(-2,-1,3), subscript_exception);
    }
}

//test mapping view helpers
TEST_CASE("test_check_index_mapping_view_subs","[test_view_factory]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = typename config_type::shape_type;
    using gtensor::subscript_exception;
    using gtensor::detail::check_index_mapping_view_subs;
    using helpers_for_testing::apply_by_element;

    SECTION("test_check_index_mapping_view_subs_nothrow")
    {
        //0pshape,1subs_shapes
        auto test_data = std::make_tuple(
            std::make_tuple(shape_type{0},std::make_tuple(shape_type{0})),
            std::make_tuple(shape_type{0},std::make_tuple(shape_type{1,2,3,0})),
            std::make_tuple(shape_type{0},std::make_tuple(shape_type{0,1,2,3})),
            std::make_tuple(shape_type{0},std::make_tuple(shape_type{1,2,0,3})),
            std::make_tuple(shape_type{1},std::make_tuple(shape_type{0})),
            std::make_tuple(shape_type{1},std::make_tuple(shape_type{1,2,3,0})),
            std::make_tuple(shape_type{1},std::make_tuple(shape_type{0,1,2,3})),
            std::make_tuple(shape_type{1},std::make_tuple(shape_type{5})),
            std::make_tuple(shape_type{0,1},std::make_tuple(shape_type{0})),
            std::make_tuple(shape_type{0,1},std::make_tuple(shape_type{1,2,3,0})),
            std::make_tuple(shape_type{1,0},std::make_tuple(shape_type{1,2,3,0})),
            std::make_tuple(shape_type{1,0},std::make_tuple(shape_type{1})),
            std::make_tuple(shape_type{1,0},std::make_tuple(shape_type{5})),
            std::make_tuple(shape_type{1,0},std::make_tuple(shape_type{5,4,3})),
            std::make_tuple(shape_type{0,1},std::make_tuple(shape_type{0},shape_type{1})),
            std::make_tuple(shape_type{0,1},std::make_tuple(shape_type{3,0},shape_type{3,1})),
            std::make_tuple(shape_type{1,0,1,0},std::make_tuple(shape_type{5})),
            std::make_tuple(shape_type{1,0,1,0},std::make_tuple(shape_type{3,1},shape_type{3,0})),
            std::make_tuple(shape_type{1,0,1,0},std::make_tuple(shape_type{3,1},shape_type{3,0},shape_type{1})),
            std::make_tuple(shape_type{1,0,1,0},std::make_tuple(shape_type{3,1},shape_type{3,0},shape_type{1},shape_type{1,0}))
        );
        auto test = [](const auto& t){
            auto pshape = std::get<0>(t);
            auto subs_shapes = std::get<1>(t);
            auto apply_subs_shapes = [&pshape](const auto&...subs_shapes_){
                check_index_mapping_view_subs(pshape, subs_shapes_...);
            };
            REQUIRE_NOTHROW(std::apply(apply_subs_shapes, subs_shapes));
        };
        apply_by_element(test, test_data);
    }
    SECTION("test_check_index_mapping_view_subs_exception")
    {
        //0pshape,1subs_shapes
        auto test_data = std::make_tuple(
            std::make_tuple(shape_type{},std::make_tuple(shape_type{})),
            std::make_tuple(shape_type{},std::make_tuple(shape_type{0})),
            std::make_tuple(shape_type{},std::make_tuple(shape_type{1})),
            std::make_tuple(shape_type{0},std::make_tuple(shape_type{1})),
            std::make_tuple(shape_type{0},std::make_tuple(shape_type{1,2,3})),
            std::make_tuple(shape_type{1},std::make_tuple(shape_type{0},shape_type{0})),
            std::make_tuple(shape_type{1},std::make_tuple(shape_type{1},shape_type{1})),
            std::make_tuple(shape_type{0,1},std::make_tuple(shape_type{1})),
            std::make_tuple(shape_type{0,1},std::make_tuple(shape_type{1},shape_type{0})),
            std::make_tuple(shape_type{0,1},std::make_tuple(shape_type{1},shape_type{1,2,3})),
            std::make_tuple(shape_type{1,0},std::make_tuple(shape_type{0},shape_type{1})),
            std::make_tuple(shape_type{1,0},std::make_tuple(shape_type{0},shape_type{1,2,3})),
            std::make_tuple(shape_type{1,0},std::make_tuple(shape_type{1},shape_type{1})),
            std::make_tuple(shape_type{1,0},std::make_tuple(shape_type{1},shape_type{1,2,3})),
            std::make_tuple(shape_type{1,0},std::make_tuple(shape_type{1},shape_type{0},shape_type{0})),
            std::make_tuple(shape_type{1,0,1,0},std::make_tuple(shape_type{3,1},shape_type{1,3})),
            std::make_tuple(shape_type{1,0,1,0},std::make_tuple(shape_type{3,1},shape_type{1,3},shape_type{1})),
            std::make_tuple(shape_type{1,0,1,0},std::make_tuple(shape_type{3,1},shape_type{3,0},shape_type{1},shape_type{1})),
            std::make_tuple(shape_type{1,0,1,0},std::make_tuple(shape_type{3,1},shape_type{3,0},shape_type{1},shape_type{1,0},shape_type{1}))
        );
        auto test = [](const auto& t){
            auto pshape = std::get<0>(t);
            auto subs_shapes = std::get<1>(t);
            auto apply_subs_shapes = [&pshape](const auto&...subs_shapes_){
                check_index_mapping_view_subs(pshape, subs_shapes_...);
            };
            REQUIRE_THROWS_AS(std::apply(apply_subs_shapes, subs_shapes), subscript_exception);
        };
        apply_by_element(test, test_data);
    }
}

TEST_CASE("test_make_index_mapping_view_shape","[test_view_factory]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = typename config_type::dim_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::subscript_exception;
    using gtensor::detail::make_index_mapping_view_shape;
    using gtensor::detail::make_broadcast_shape;
    using helpers_for_testing::apply_by_element;
    //0parent_shape,1subs_shapes,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{0}, std::make_tuple(shape_type{0}), shape_type{0}),
        std::make_tuple(shape_type{0}, std::make_tuple(shape_type{1,2,3,0}), shape_type{1,2,3,0}),
        std::make_tuple(shape_type{1}, std::make_tuple(shape_type{}), shape_type{}),
        std::make_tuple(shape_type{1}, std::make_tuple(shape_type{0}), shape_type{0}),
        std::make_tuple(shape_type{1}, std::make_tuple(shape_type{1,2,3,0}), shape_type{1,2,3,0}),
        std::make_tuple(shape_type{10}, std::make_tuple(shape_type{}), shape_type{}),
        std::make_tuple(shape_type{10}, std::make_tuple(shape_type{4}), shape_type{4}),
        std::make_tuple(shape_type{10}, std::make_tuple(shape_type{2,2}), shape_type{2,2}),
        std::make_tuple(shape_type{4,0}, std::make_tuple(shape_type{}), shape_type{0}),
        std::make_tuple(shape_type{4,0}, std::make_tuple(shape_type{3}), shape_type{3,0}),
        std::make_tuple(shape_type{4,0}, std::make_tuple(shape_type{3,3}), shape_type{3,3,0}),
        std::make_tuple(shape_type{0,4}, std::make_tuple(shape_type{0}), shape_type{0,4}),
        std::make_tuple(shape_type{0,4}, std::make_tuple(shape_type{1,2,3,0}), shape_type{1,2,3,0,4}),
        std::make_tuple(shape_type{4,0}, std::make_tuple(shape_type{}, shape_type{1,0}), shape_type{1,0}),
        std::make_tuple(shape_type{4,0}, std::make_tuple(shape_type{3,1}, shape_type{1,0}), shape_type{3,0}),
        std::make_tuple(shape_type{4,5}, std::make_tuple(shape_type{3,1}, shape_type{1,0}), shape_type{3,0}),
        std::make_tuple(shape_type{4,5}, std::make_tuple(shape_type{}), shape_type{5}),
        std::make_tuple(shape_type{4,5}, std::make_tuple(shape_type{},shape_type{}), shape_type{}),
        std::make_tuple(shape_type{5,4,3,2}, std::make_tuple(shape_type{},shape_type{}), shape_type{3,2}),
        std::make_tuple(shape_type{5,4,3,2}, std::make_tuple(shape_type{},shape_type{8}), shape_type{8,3,2}),
        std::make_tuple(shape_type{5,4,3,2}, std::make_tuple(shape_type{},shape_type{},shape_type{8}), shape_type{8,2}),
        std::make_tuple(shape_type{5,4,3,2}, std::make_tuple(shape_type{8},shape_type{8}), shape_type{8,3,2}),
        std::make_tuple(shape_type{5,4,3,2}, std::make_tuple(shape_type{3,4},shape_type{1},shape_type{3,1}), shape_type{3,4,2}),
        std::make_tuple(shape_type{5,4,3,2}, std::make_tuple(shape_type{3,4},shape_type{1},shape_type{3,1}, shape_type{1,4}), shape_type{3,4})
    );
    auto test = [](const auto& t){
        auto parent_shape = std::get<0>(t);
        auto subs_shapes = std::get<1>(t);
        auto expected = std::get<2>(t);
        dim_type subs_number = std::tuple_size_v<decltype(subs_shapes)>;
        auto make_subs_shape = [](const auto&...subs_shapes_){
            return make_broadcast_shape<shape_type>(subs_shapes_...);
        };
        auto subs_shape = std::apply(make_subs_shape, subs_shapes);
        auto result = make_index_mapping_view_shape(parent_shape, subs_shape, subs_number);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_fill_index_map","[test_view_factory]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using index_tensor_type = gtensor::tensor<index_type,TestType,config_type>;
    using index_map_type = std::vector<index_type>;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::walker_forward_traverser;
    using gtensor::detail::fill_index_map;
    using gtensor::detail::make_index_mapping_view_shape;
    using gtensor::detail::make_size;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_strides_div;
    using gtensor::detail::make_broadcast_shape;
    using helpers_for_testing::apply_by_element;

    //0pshape,1subs,2expected
    auto test_data = std::make_tuple(
        //parent's layout c_order
        std::make_tuple(tensor<value_type,c_order,config_type>{0}, std::make_tuple(index_tensor_type{0,0,0}),index_map_type{0,0,0}),
        std::make_tuple(tensor<value_type,c_order,config_type>{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{0,2,0,1,0}),index_map_type{0,2,0,1,0}),
        std::make_tuple(tensor<value_type,c_order,config_type>{0,1,2,3,4}, std::make_tuple(index_tensor_type{{4,3,2,1,0}}),index_map_type{4,3,2,1,0}),
        std::make_tuple(tensor<value_type,c_order,config_type>{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{{0,2,4,6,8},{1,3,5,7,9}}),index_map_type{0,2,4,6,8,1,3,5,7,9}),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{1,3,0,1}),index_map_type{3,4,5,9,10,11,0,1,2,3,4,5}),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1},{3}}),index_map_type{3,4,5,9,10,11}),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{0,1,2}),index_map_type{0,4,8}),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1,2},{0,1}}, index_tensor_type{{0,1},{1,2}}),index_map_type{3,7,1,5}),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{1}),index_map_type{1,4,7}),
        std::make_tuple(
            tensor<value_type,c_order,config_type>{{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},{{12,13,14},{15,16,17},{18,19,20},{21,22,23}}},
            std::make_tuple(index_tensor_type{0,1}, index_tensor_type{1}),
            index_map_type{3,4,5,15,16,17}
        ),
        std::make_tuple(
            tensor<value_type,c_order,config_type>{{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},{{12,13,14},{15,16,17},{18,19,20},{21,22,23}}},
            std::make_tuple(index_tensor_type{0,1}, index_tensor_type{3}, index_tensor_type{2}),
            index_map_type{11,23}
        ),
        //parent's layout f_order
        std::make_tuple(tensor<value_type,f_order,config_type>{0}, std::make_tuple(index_tensor_type{0,0,0}),index_map_type{0,0,0}),
        std::make_tuple(tensor<value_type,f_order,config_type>{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{0,2,0,1,0}),index_map_type{0,2,0,1,0}),
        std::make_tuple(tensor<value_type,f_order,config_type>{0,1,2,3,4}, std::make_tuple(index_tensor_type{{4,3,2,1,0}}),index_map_type{4,3,2,1,0}),
        std::make_tuple(tensor<value_type,f_order,config_type>{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{{0,2,4,6,8},{1,3,5,7,9}}),index_map_type{0,1,2,3,4,5,6,7,8,9}),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{1,3,0,1}),index_map_type{1,3,0,1,5,7,4,5,9,11,8,9}),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1},{3}}),index_map_type{1,3,5,7,9,11}),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{0,1,2}),index_map_type{0,5,10}),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1,2},{0,1}}, index_tensor_type{{0,1},{1,2}}),index_map_type{1,4,6,9}),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{1}),index_map_type{4,5,6}),
        std::make_tuple(
            tensor<value_type,f_order,config_type>{{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},{{12,13,14},{15,16,17},{18,19,20},{21,22,23}}},
            std::make_tuple(index_tensor_type{0,1}, index_tensor_type{1}),
            index_map_type{2,3,10,11,18,19}
        ),
        std::make_tuple(
            tensor<value_type,f_order,config_type>{{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},{{12,13,14},{15,16,17},{18,19,20},{21,22,23}}},
            std::make_tuple(index_tensor_type{0,1}, index_tensor_type{3}, index_tensor_type{2}),
            index_map_type{22,23}
        )
    );

    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto subs = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto broadcast_shape_maker = [](const auto&...subs){
            return make_broadcast_shape<shape_type>(subs.shape()...);
        };
        auto subs_shape = std::apply(broadcast_shape_maker, subs);
        auto subs_size = make_size(subs_shape);
        auto result_shape = make_index_mapping_view_shape(parent.shape(), subs_shape, std::tuple_size_v<std::decay_t<decltype(subs)>>);
        auto result_size = make_size(result_shape);
        index_map_type result(result_size);
        using order = typename decltype(parent)::order;
        auto result_filler = [&parent,&result,&subs_shape,&subs_size](const auto&...subs){
            return fill_index_map<order>(
                parent.shape(),
                parent.strides(),
                subs_size,
                result,
                walker_forward_traverser<config_type, decltype(subs.create_walker())>{subs_shape, subs.create_walker()}...
            );
        };
        std::apply(result_filler, subs);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_fill_index_map_exception","[test_view_factory]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using index_tensor_type = gtensor::tensor<index_type,TestType,config_type>;
    using index_map_type = std::vector<index_type>;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::walker_forward_traverser;
    using gtensor::subscript_exception;
    using gtensor::detail::fill_index_map;
    using gtensor::detail::make_index_mapping_view_shape;
    using gtensor::detail::make_broadcast_shape;
    using gtensor::detail::make_size;
    using gtensor::detail::make_strides;
    using gtensor::detail::make_strides_div;
    using gtensor::detail::make_broadcast_shape;
    using helpers_for_testing::apply_by_element;

    //0pshape,1subs
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(tensor<value_type,c_order,config_type>{0}, std::make_tuple(index_tensor_type{0,0,3})),
        std::make_tuple(tensor<value_type,c_order,config_type>{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{0,20,0,1,0})),
        std::make_tuple(tensor<value_type,c_order,config_type>{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{{0,2,4,16,8},{1,3,5,7,9}})),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{1,3,0,4})),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{11},{3}})),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1,2},{5,1}}, index_tensor_type{{0,1},{1,2}})),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1,2},{0,1}}, index_tensor_type{{0,1},{1,4}})),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,5}, index_tensor_type{1})),
        std::make_tuple(tensor<value_type,c_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{3})),
        std::make_tuple(
            tensor<value_type,c_order,config_type>{{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},{{12,13,14},{15,16,17},{18,19,20},{21,22,23}}},
            std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{4}, index_tensor_type{2})
        ),
        //f_order
        std::make_tuple(tensor<value_type,f_order,config_type>{0}, std::make_tuple(index_tensor_type{0,0,3})),
        std::make_tuple(tensor<value_type,f_order,config_type>{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{0,20,0,1,0})),
        std::make_tuple(tensor<value_type,f_order,config_type>{0,1,2,3,4,5,6,7,8,9}, std::make_tuple(index_tensor_type{{0,2,4,16,8},{1,3,5,7,9}})),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{1,3,0,4})),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{11},{3}})),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1,2},{5,1}}, index_tensor_type{{0,1},{1,2}})),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{{1,2},{0,1}}, index_tensor_type{{0,1},{1,4}})),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,5}, index_tensor_type{1})),
        std::make_tuple(tensor<value_type,f_order,config_type>{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{3})),
        std::make_tuple(
            tensor<value_type,f_order,config_type>{{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},{{12,13,14},{15,16,17},{18,19,20},{21,22,23}}},
            std::make_tuple(index_tensor_type{0,1,2}, index_tensor_type{4}, index_tensor_type{2})
        )
    );

    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto subs = std::get<1>(t);
        auto broadcast_shape_maker = [](const auto&...subs){
            return make_broadcast_shape<shape_type>(subs.shape()...);
        };
        auto subs_shape = std::apply(broadcast_shape_maker, subs);
        auto subs_size = make_size(subs_shape);
        auto res_shape = make_index_mapping_view_shape(parent.shape(), subs_shape, std::tuple_size_v<std::decay_t<decltype(subs)>>);
        auto res_size = make_size(res_shape);
        index_map_type index_map(res_size);
        using order = typename decltype(parent)::order;
        auto elements_filler = [&parent,&index_map,&subs_shape,&subs_size](const auto&...subs){
            return fill_index_map<order>(
                parent.shape(),
                parent.strides(),
                subs_size,
                index_map,
                walker_forward_traverser<config_type, decltype(subs.create_walker())>{subs_shape, subs.create_walker()}...
            );
        };
        REQUIRE_THROWS_AS(std::apply(elements_filler, subs), subscript_exception);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_check_bool_mapping_view_subs","[test_view_factory]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = typename config_type::shape_type;
    using gtensor::subscript_exception;
    using gtensor::detail::check_bool_mapping_view_subs;
    //0pshape,1subs_shape
    using test_type = std::tuple<shape_type,shape_type>;
    SECTION("test_check_bool_mapping_view_subs_nothrow"){
        auto test_data = GENERATE(
            test_type(shape_type{0}, shape_type{0}),
            test_type(shape_type{1}, shape_type{0}),
            test_type(shape_type{1}, shape_type{1}),
            test_type(shape_type{10}, shape_type{0}),
            test_type(shape_type{10}, shape_type{5}),
            test_type(shape_type{10}, shape_type{10}),
            test_type(shape_type{4,3,2}, shape_type{0}),
            test_type(shape_type{4,3,2}, shape_type{0,3}),
            test_type(shape_type{4,3,0}, shape_type{4,3,0}),
            test_type(shape_type{4,0,2}, shape_type{4,0,2}),
            test_type(shape_type{4,0,2}, shape_type{3,0,2}),
            test_type(shape_type{4,3,2}, shape_type{4,3,2}),
            test_type(shape_type{4,3,2}, shape_type{2,2,2}),
            test_type(shape_type{4,3,2}, shape_type{2,1}),
            test_type(shape_type{4,3,2}, shape_type{2})
        );
        auto pshape = std::get<0>(test_data);
        auto subs_shape = std::get<1>(test_data);
        REQUIRE_NOTHROW(check_bool_mapping_view_subs(pshape,subs_shape));
    }
    SECTION("test_check_bool_mapping_view_subs_exception"){
        auto test_data = GENERATE(
            //0-dim
            test_type(shape_type{}, shape_type{}),
            test_type(shape_type{}, shape_type{0}),
            test_type(shape_type{}, shape_type{1}),
            //subs dim > parent dim
            test_type(shape_type{0}, shape_type{0,0}),
            test_type(shape_type{10}, shape_type{10,10}),
            test_type(shape_type{3,2,4}, shape_type{2,2,2,2}),
            //subs direction size > parent direction size
            test_type(shape_type{0}, shape_type{1}),
            test_type(shape_type{10}, shape_type{20}),
            test_type(shape_type{3,2,4}, shape_type{4}),
            test_type(shape_type{3,2,4}, shape_type{0,4}),
            test_type(shape_type{3,2,4}, shape_type{4,0}),
            test_type(shape_type{3,2,4}, shape_type{3,3}),
            test_type(shape_type{3,0,4}, shape_type{1,4}),
            test_type(shape_type{3,0,4}, shape_type{1,4})
        );
        auto pshape = std::get<0>(test_data);
        auto subs_shape = std::get<1>(test_data);
        REQUIRE_THROWS_AS(check_bool_mapping_view_subs(pshape,subs_shape), subscript_exception);
    }
}

TEST_CASE("test_make_bool_mapping_view_shape","[test_view_factory]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;
    using gtensor::detail::make_bool_mapping_view_shape;
    using test_type = std::tuple<shape_type,index_type,dim_type,shape_type>;
    //0pshape,1trues_number,2subs_dim,3expected
    auto test_data = GENERATE(
        test_type{shape_type{0},index_type{0},dim_type{0},shape_type{0,0}},
        test_type{shape_type{0},index_type{1},dim_type{0},shape_type{1,0}},
        test_type{shape_type{0},index_type{0},dim_type{1},shape_type{0}},
        test_type{shape_type{1,0},index_type{0},dim_type{1},shape_type{0,0}},
        test_type{shape_type{1,0},index_type{0},dim_type{2},shape_type{0}},
        test_type{shape_type{10},index_type{0},dim_type{0},shape_type{0,10}},
        test_type{shape_type{10},index_type{1},dim_type{0},shape_type{1,10}},
        test_type{shape_type{10},index_type{10},dim_type{1},shape_type{10}},
        test_type{shape_type{10},index_type{3},dim_type{1},shape_type{3}},
        test_type{shape_type{10},index_type{0},dim_type{1},shape_type{0}},
        test_type{shape_type{2,3,4},index_type{0},dim_type{1},shape_type{0,3,4}},
        test_type{shape_type{2,3,4},index_type{1},dim_type{1},shape_type{1,3,4}},
        test_type{shape_type{2,3,4},index_type{2},dim_type{1},shape_type{2,3,4}},
        test_type{shape_type{2,3,4},index_type{0},dim_type{2},shape_type{0,4}},
        test_type{shape_type{2,3,4},index_type{1},dim_type{2},shape_type{1,4}},
        test_type{shape_type{2,3,4},index_type{6},dim_type{2},shape_type{6,4}},
        test_type{shape_type{2,3,4},index_type{0},dim_type{3},shape_type{0}},
        test_type{shape_type{2,3,4},index_type{1},dim_type{3},shape_type{1}},
        test_type{shape_type{2,3,4},index_type{5},dim_type{3},shape_type{5}},
        test_type{shape_type{0,2,3,4},index_type{0},dim_type{2},shape_type{0,3,4}}
    );
    auto pshape = std::get<0>(test_data);
    auto trues_number = std::get<1>(test_data);
    auto subs_dim = std::get<2>(test_data);
    auto expected = std::get<3>(test_data);
    auto result = make_bool_mapping_view_shape(pshape,trues_number,subs_dim);
    REQUIRE(result == expected);
}

// TEST_CASE("test_fill_bool_map","[test_view_factory]"){
//     using value_type = float;
//     using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
//     using tensor_type = gtensor::tensor<value_type, config_type>;
//     using index_tensor_type = gtensor::tensor<bool, config_type>;
//     using index_type = typename config_type::index_type;
//     using index_container_type = std::vector<index_type>;
//     using gtensor::walker_forward_traverser;
//     using gtensor::detail::fill_bool_map;
//     using gtensor::detail::make_bool_mapping_view_shape;
//     using helpers_for_testing::apply_by_element;

//     //0parent,1subs,2expected_trues_number,3expected_elements
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{0}, index_tensor_type{false}, index_type{0}, index_container_type{}),
//         std::make_tuple(tensor_type{0}, index_tensor_type{true}, index_type{1}, index_container_type{0}),
//         std::make_tuple(tensor_type{{0}}, index_tensor_type{false}, index_type{0}, index_container_type{}),
//         std::make_tuple(tensor_type{{0}}, index_tensor_type{{false}}, index_type{0}, index_container_type{}),
//         std::make_tuple(tensor_type{0,1,2,3,4,5}, index_tensor_type{false,false,false,false,false}, index_type{0}, index_container_type{}),
//         std::make_tuple(tensor_type{0,1,2,3,4,5}, index_tensor_type{false,true,false,true,false}, index_type{2}, index_container_type{1,3}),
//         std::make_tuple(tensor_type{0,1,2,3,4,5,6,7,8,9}, index_tensor_type{true,false,true,false,false}, index_type{2}, index_container_type{0,2}),
//         std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, index_tensor_type{false,false,false,false}, index_type{0}, index_container_type{}),
//         std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, index_tensor_type{true,false,true,false}, index_type{2}, index_container_type{0,1,2,6,7,8}),
//         std::make_tuple(
//             tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},
//             index_tensor_type{{false,false,false},{false,false,false},{false,false,false},{false,false,false}},
//             index_type{0},
//             index_container_type{}
//         ),
//         std::make_tuple(
//             tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},
//             index_tensor_type{{false,true,true},{false,false,false},{false,false,false},{false,false,false}},
//             index_type{2},
//             index_container_type{1,2}
//         ),
//         std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, index_tensor_type{{false,true,true}}, index_type{2}, index_container_type{1,2}),
//         std::make_tuple(
//             tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}},
//             index_tensor_type{{false,false,false},{true,false,false},{true,false,false},{false,false,false}},
//             index_type{2},
//             index_container_type{3,6}
//         ),
//         std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, index_tensor_type{{false},{true},{true}}, index_type{2}, index_container_type{3,6}),
//         std::make_tuple(tensor_type{{0,1,2},{3,4,5},{6,7,8},{9,10,11}}, index_tensor_type{{false,true},{true,false}}, index_type{2}, index_container_type{1,3}),
//         std::make_tuple(
//             tensor_type{{{0,1},{2,3},{4,5},{6,7}},{{8,9},{10,11},{12,13},{14,15}},{{16,17},{18,19},{20,21},{22,23}}},
//             index_tensor_type{false,false,false},
//             index_type{0},
//             index_container_type{}
//         ),
//         std::make_tuple(
//             tensor_type{{{0,1},{2,3},{4,5},{6,7}},{{8,9},{10,11},{12,13},{14,15}},{{16,17},{18,19},{20,21},{22,23}}},
//             index_tensor_type{{false,false,true,false},{false,false,false,true},{false,true,false,true}},
//             index_type{4},
//             index_container_type{4,5,14,15,18,19,22,23}
//         ),
//         std::make_tuple(
//             tensor_type{{{0,1},{2,3},{4,5},{6,7}},{{8,9},{10,11},{12,13},{14,15}},{{16,17},{18,19},{20,21},{22,23}}},
//             index_tensor_type{{{true,false},{false,true},{false,false}}},
//             index_type{2},
//             index_container_type{0,3}
//         )
//     );
//     auto test = [](const auto& t){
//         auto parent = std::get<0>(t);
//         auto subs = std::get<1>(t);
//         auto expected_trues_number = std::get<2>(t);
//         auto expected_index = std::get<3>(t);
//         index_container_type result_index{};
//         result_index.reserve(parent.size());
//         auto result_trues_number = fill_bool_map(
//             parent.shape(),
//             parent.strides(),
//             result_index,
//             subs,
//             walker_forward_traverser<config_type, decltype(subs.create_walker())>{subs.shape(), subs.create_walker()}
//         );
//         REQUIRE(result_trues_number == expected_trues_number);
//         REQUIRE(result_index == expected_index);
//     };
//     apply_by_element(test,test_data);
// }

//test view_factory
// //test create_reshape_view
// TEST_CASE("test_create_reshape_view","[test_view_factory]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using shape_type = typename tensor_type::shape_type;
//     using gtensor::basic_tensor;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using helpers_for_testing::apply_by_element;
//     //0parent,1order,2subs,3expected
//     auto test_data = std::make_tuple(
//         //c_order
//         std::make_tuple(tensor_type(2),c_order{},std::make_tuple(),tensor_type{shape_type{},2}),
//         std::make_tuple(tensor_type(2),c_order{},std::make_tuple(1),tensor_type{shape_type{1},2}),
//         std::make_tuple(tensor_type(2),c_order{},std::make_tuple(-1),tensor_type{shape_type{1},2}),
//         std::make_tuple(tensor_type(3),c_order{},std::make_tuple(1,1),tensor_type{shape_type{1,1},3}),
//         std::make_tuple(tensor_type(3),c_order{},std::make_tuple(1,-1),tensor_type{shape_type{1,1},3}),
//         std::make_tuple(tensor_type(3),c_order{},std::make_tuple(-1,1),tensor_type{shape_type{1,1},3}),
//         std::make_tuple(tensor_type(4),c_order{},std::make_tuple(1,1,1),tensor_type{shape_type{1,1,1},4}),
//         std::make_tuple(tensor_type(4),c_order{},std::make_tuple(1,-1,1),tensor_type{shape_type{1,1,1},4}),
//         std::make_tuple(tensor_type{},c_order{},std::make_tuple(),tensor_type{}),
//         std::make_tuple(tensor_type{},c_order{},std::make_tuple(1,-1),tensor_type{}.reshape(1,0)),
//         std::make_tuple(tensor_type{},c_order{},std::make_tuple(-1,1),tensor_type{}.reshape(0,1)),
//         std::make_tuple(tensor_type{},c_order{},std::make_tuple(2,-1,1),tensor_type{}.reshape(2,0,1)),
//         std::make_tuple(tensor_type{1},c_order{},std::make_tuple(),tensor_type{1}),
//         std::make_tuple(tensor_type{1},c_order{},std::make_tuple(-1),tensor_type{1}),
//         std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(),tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(-1),tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(1,5),tensor_type{{1,2,3,4,5}}),
//         std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(-1,5),tensor_type{{1,2,3,4,5}}),
//         std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(5,1),tensor_type{{1},{2},{3},{4},{5}}),
//         std::make_tuple(tensor_type{1,2,3,4,5},c_order{},std::make_tuple(5,-1),tensor_type{{1},{2},{3},{4},{5}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},c_order{},std::make_tuple(), tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},c_order{},std::make_tuple(-1), tensor_type{1,2,3,4,5,6,7,8,9,10,11,12}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},c_order{},std::make_tuple(6,2), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},c_order{},std::make_tuple(6,-1), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},c_order{},std::make_tuple(-1,2), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}),
//         //f_order
//         std::make_tuple(tensor_type(2),f_order{},std::make_tuple(),tensor_type{shape_type{},2}),
//         std::make_tuple(tensor_type(2),f_order{},std::make_tuple(1),tensor_type{shape_type{1},2}),
//         std::make_tuple(tensor_type(2),f_order{},std::make_tuple(-1),tensor_type{shape_type{1},2}),
//         std::make_tuple(tensor_type(3),f_order{},std::make_tuple(1,1),tensor_type{shape_type{1,1},3}),
//         std::make_tuple(tensor_type(3),f_order{},std::make_tuple(1,-1),tensor_type{shape_type{1,1},3}),
//         std::make_tuple(tensor_type(3),f_order{},std::make_tuple(-1,1),tensor_type{shape_type{1,1},3}),
//         std::make_tuple(tensor_type(4),f_order{},std::make_tuple(1,1,1),tensor_type{shape_type{1,1,1},4}),
//         std::make_tuple(tensor_type(4),f_order{},std::make_tuple(1,-1,1),tensor_type{shape_type{1,1,1},4}),
//         std::make_tuple(tensor_type{},f_order{},std::make_tuple(),tensor_type{}),
//         std::make_tuple(tensor_type{},f_order{},std::make_tuple(1,-1),tensor_type{}.reshape(1,0)),
//         std::make_tuple(tensor_type{},f_order{},std::make_tuple(-1,1),tensor_type{}.reshape(0,1)),
//         std::make_tuple(tensor_type{},f_order{},std::make_tuple(2,-1,1),tensor_type{}.reshape(2,0,1)),
//         std::make_tuple(tensor_type{1},f_order{},std::make_tuple(),tensor_type{1}),
//         std::make_tuple(tensor_type{1},f_order{},std::make_tuple(-1),tensor_type{1}),
//         std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(),tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(-1),tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(1,5),tensor_type{{1,2,3,4,5}}),
//         std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(-1,5),tensor_type{{1,2,3,4,5}}),
//         std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(5,1),tensor_type{{1},{2},{3},{4},{5}}),
//         std::make_tuple(tensor_type{1,2,3,4,5},f_order{},std::make_tuple(5,-1),tensor_type{{1},{2},{3},{4},{5}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},f_order{},std::make_tuple(), tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},f_order{},std::make_tuple(-1), tensor_type{1,7,3,9,5,11,2,8,4,10,6,12}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},f_order{},std::make_tuple(6,2), tensor_type{{1,2},{7,8},{3,4},{9,10},{5,6},{11,12}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},f_order{},std::make_tuple(6,-1), tensor_type{{1,2},{7,8},{3,4},{9,10},{5,6},{11,12}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}},f_order{},std::make_tuple(-1,2), tensor_type{{1,2},{7,8},{3,4},{9,10},{5,6},{11,12}})
//     );
//     SECTION("test_create_reshape_view_variadic")
//     {
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto order = std::get<1>(t);
//             auto subs = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             using order_type = decltype(order);
//             auto apply_subs = [&parent](const auto&...subs_){
//                 return basic_tensor{view_factory_type::create_reshape_view<order_type>(parent, subs_...)};
//             };
//             auto result = std::apply(apply_subs, subs);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_create_reshape_view_container")
//     {
//         using container_type = std::vector<int>;
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto order = std::get<1>(t);
//             auto subs = std::get<2>(t);
//             auto expected = std::get<3>(t);
//             auto make_container = [](const auto&...subs_){
//                 return container_type{subs_...};
//             };
//             auto container = std::apply(make_container, subs);
//             using order_type = decltype(order);
//             auto result = basic_tensor{view_factory_type::template create_reshape_view<order_type>(parent, container)};
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEMPLATE_TEST_CASE("test_create_reshape_view_exception","[test_view_factory]",
//     gtensor::config::c_order,
//     gtensor::config::f_order
// )
// {
//     using order = TestType;
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using gtensor::subscript_exception;
//     using helpers_for_testing::apply_by_element;

//     //0parent,1subs
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type(0),std::make_tuple(0)),
//         std::make_tuple(tensor_type(0),std::make_tuple(1,0)),
//         std::make_tuple(tensor_type(0),std::make_tuple(-1,0)),
//         std::make_tuple(tensor_type(0),std::make_tuple(2)),
//         std::make_tuple(tensor_type(0),std::make_tuple(1,2)),
//         std::make_tuple(tensor_type{},std::make_tuple(-1,-1)),
//         std::make_tuple(tensor_type{},std::make_tuple(-1,0)),
//         std::make_tuple(tensor_type{},std::make_tuple(0,-1)),
//         std::make_tuple(tensor_type{1},std::make_tuple(0)),
//         std::make_tuple(tensor_type{1},std::make_tuple(2)),
//         std::make_tuple(tensor_type{1},std::make_tuple(-1,0)),
//         std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(10)),
//         std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(3,3)),
//         std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(-1,-1)),
//         std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(-1,4)),
//         std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(4,-1)),
//         std::make_tuple(tensor_type{{1,2},{3,4},{5,6}},std::make_tuple(0,2))
//     );
//     SECTION("test_create_reshape_view_exception_variadic")
//     {
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto subs = std::get<1>(t);
//             auto apply_subs = [&parent](const auto&...subs_){
//                 return view_factory_type::template create_reshape_view<order>(parent, subs_...);
//             };
//             REQUIRE_THROWS_AS(std::apply(apply_subs, subs), subscript_exception);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_create_reshape_view_exception_container")
//     {
//         using container_type = std::vector<int>;
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto subs = std::get<1>(t);
//             auto make_container = [](const auto&...subs_){
//                 return container_type{subs_...};
//             };
//             auto container = std::apply(make_container, subs);
//             REQUIRE_THROWS_AS(view_factory_type::template create_reshape_view<order>(parent, container), subscript_exception);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// //test create_transpose_view
// TEST_CASE("test_create_transpose_view","[test_view_factory]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using gtensor::basic_tensor;
//     using helpers_for_testing::apply_by_element;
//     //0parent,1subs,2expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type(2),std::make_tuple(),tensor_type(2)),
//         std::make_tuple(tensor_type{},std::make_tuple(),tensor_type{}),
//         std::make_tuple(tensor_type{},std::make_tuple(0),tensor_type{}),
//         std::make_tuple(tensor_type{1},std::make_tuple(),tensor_type{1}),
//         std::make_tuple(tensor_type{1},std::make_tuple(0),tensor_type{1}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(),tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(0),tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{{1,2,3,4,5}},std::make_tuple(),tensor_type{{1},{2},{3},{4},{5}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5}},std::make_tuple(1,0),tensor_type{{1},{2},{3},{4},{5}}),
//         std::make_tuple(tensor_type{{1,2,3,4,5}},std::make_tuple(0,1),tensor_type{{1,2,3,4,5}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}},std::make_tuple(),tensor_type{{{1},{3},{5}},{{2},{4},{6}}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}},std::make_tuple(2,1,0),tensor_type{{{1},{3},{5}},{{2},{4},{6}}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}},std::make_tuple(2,0,1),tensor_type{{{1,3,5}},{{2,4,6}}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}},std::make_tuple(1,0,2),tensor_type{{{1,2}},{{3,4}},{{5,6}}})
//     );
//     SECTION("test_create_transpose_view_variadic")
//     {
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto subs = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto apply_subs = [&parent](const auto&...subs_){
//                 return basic_tensor{view_factory_type::create_transpose_view(parent, subs_...)};
//             };
//             auto result = std::apply(apply_subs, subs);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_create_transpose_view_container")
//     {
//         using container_type = std::vector<int>;
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto subs = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto make_container = [](const auto&...subs_){
//                 return container_type{subs_...};
//             };
//             auto container = std::apply(make_container, subs);
//             auto result = basic_tensor{view_factory_type::create_transpose_view(parent, container)};
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// TEST_CASE("test_create_transpose_view_exception","[test_view_factory]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using gtensor::basic_tensor;
//     using helpers_for_testing::apply_by_element;
//     using gtensor::subscript_exception;
//     //0parent,1subs
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type(2),std::make_tuple(0)),
//         std::make_tuple(tensor_type(2),std::make_tuple(1)),
//         std::make_tuple(tensor_type(2),std::make_tuple(0,1)),
//         std::make_tuple(tensor_type{},std::make_tuple(0,0)),
//         std::make_tuple(tensor_type{},std::make_tuple(1)),
//         std::make_tuple(tensor_type{1},std::make_tuple(0,1)),
//         std::make_tuple(tensor_type{1},std::make_tuple(1)),
//         std::make_tuple(tensor_type{{1,2},{3,4}},std::make_tuple(0,2,1)),
//         std::make_tuple(tensor_type{{1,2},{3,4}},std::make_tuple(0)),
//         std::make_tuple(tensor_type{{1,2},{3,4}},std::make_tuple(1,1))
//     );
//     SECTION("test_create_transpose_view_exception_variadic")
//     {
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto subs = std::get<1>(t);
//             auto apply_subs = [&parent](const auto&...subs_){
//                 return basic_tensor{view_factory_type::create_transpose_view(parent, subs_...)};
//             };
//             REQUIRE_THROWS_AS(std::apply(apply_subs, subs), subscript_exception);
//         };
//         apply_by_element(test,test_data);
//     }
//     SECTION("test_create_transpose_view_exception_container")
//     {
//         using container_type = std::vector<int>;
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto subs = std::get<1>(t);
//             auto make_container = [](const auto&...subs_){
//                 return container_type{subs_...};
//             };
//             auto container = std::apply(make_container, subs);
//             REQUIRE_THROWS_AS(view_factory_type::create_transpose_view(parent, container), subscript_exception);
//         };
//         apply_by_element(test,test_data);
//     }
// }

// //test create_slice_view
// TEST_CASE("test_create_slice_view","[test_view_factory]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using slice_type = typename tensor_type::slice_type;
//     using nop_type = typename slice_type::nop_type;
//     using rtag_type = typename slice_type::reduce_tag_type;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using gtensor::basic_tensor;
//     using helpers_for_testing::apply_by_element;
//     //0parent,1subs,2expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type(2),std::make_tuple(),tensor_type(2)),
//         std::make_tuple(tensor_type{},std::make_tuple(),tensor_type{}),
//         std::make_tuple(tensor_type{},std::make_tuple(slice_type{}),tensor_type{}),
//         std::make_tuple(tensor_type{},std::make_tuple(slice_type{1,-1}),tensor_type{}),
//         std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,-1}),tensor_type{}.reshape(2,3,0)),
//         std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{nop_type{},nop_type{},2}),tensor_type{}.reshape(2,3,0)),
//         std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,-1},slice_type{nop_type{},nop_type{},2}),tensor_type{}.reshape(2,2,0)),
//         std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,-1},slice_type{nop_type{},nop_type{},-3},slice_type{1,-1}),tensor_type{}.reshape(2,1,0)),
//         std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,rtag_type{}}),tensor_type{}.reshape(3,0)),
//         std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,-1},slice_type{1,rtag_type{}}),tensor_type{}.reshape(2,0)),
//         std::make_tuple(tensor_type{}.reshape(4,3,0),std::make_tuple(slice_type{1,rtag_type{}},slice_type{1,rtag_type{}}),tensor_type{}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(),tensor_type{1,2,3,4,5,6,7,8,9,10}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{0,rtag_type{}}),tensor_type(1)),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{1,rtag_type{}}),tensor_type(2)),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-1,rtag_type{}}),tensor_type(10)),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{}),tensor_type{1,2,3,4,5,6,7,8,9,10}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-20,20}),tensor_type{1,2,3,4,5,6,7,8,9,10}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-20,5}),tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-20,-5}),tensor_type{1,2,3,4,5}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{5,20}),tensor_type{6,7,8,9,10}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{20,-20,-1}),tensor_type{10,9,8,7,6,5,4,3,2,1}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{20,5,-1}),tensor_type{10,9,8,7}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{20,-5,-1}),tensor_type{10,9,8,7}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{5,-20,-1}),tensor_type{6,5,4,3,2,1}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{nop_type{},nop_type{},-1}),tensor_type{10,9,8,7,6,5,4,3,2,1}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{nop_type{},nop_type{},-3}),tensor_type{10,7,4,1}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{2,-2}),tensor_type{3,4,5,6,7,8}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-2,2,-1}),tensor_type{9,8,7,6,5,4}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{2,-2,2}),tensor_type{3,5,7}),
//         std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10},std::make_tuple(slice_type{-2,2,-2}),tensor_type{9,7,5}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(),tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{}),tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{1}),tensor_type{{4,5,6},{7,8,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-10,10}),tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-10,2}),tensor_type{{1,2,3},{4,5,6}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{10,-3,-1}),tensor_type{{7,8,9},{4,5,6}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{nop_type{},nop_type{},-1}),tensor_type{{7,8,9},{4,5,6},{1,2,3}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{}),tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-10,10},slice_type{-10,10}),tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{1}),tensor_type{{2,3},{5,6},{8,9}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{1,2}),tensor_type{{2},{5},{8}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{0,1},slice_type{1,2}),tensor_type{{2}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{0,rtag_type{}}),tensor_type{1,2,3}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{1,rtag_type{}}),tensor_type{4,5,6}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{2,rtag_type{}}),tensor_type{7,8,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-1,rtag_type{}}),tensor_type{7,8,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-2,rtag_type{}}),tensor_type{4,5,6}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-3,rtag_type{}}),tensor_type{1,2,3}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{0,rtag_type{}},slice_type{0,rtag_type{}}),tensor_type(1)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{1,rtag_type{}},slice_type{1,rtag_type{}}),tensor_type(5)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{-1,rtag_type{}},slice_type{-1,rtag_type{}}),tensor_type(9)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{2,rtag_type{}}),tensor_type{3,6,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{1,rtag_type{}}),tensor_type{2,5,8}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{0,rtag_type{}}),tensor_type{1,4,7}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{-3,rtag_type{}}),tensor_type{1,4,7}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{-2,rtag_type{}}),tensor_type{2,5,8}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},slice_type{-1,rtag_type{}}),tensor_type{3,6,9}),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(),
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{}),
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{nop_type{},nop_type{},-1}),
//             tensor_type{{{10,11,12},{13,14,15},{16,17,18}},{{1,2,3},{4,5,6},{7,8,9}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{1}),
//             tensor_type{{{10,11,12},{13,14,15},{16,17,18}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{1,rtag_type{}}),
//             tensor_type{{10,11,12},{13,14,15},{16,17,18}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{-2,rtag_type{}}),
//             tensor_type{{1,2,3},{4,5,6},{7,8,9}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{},slice_type{}),
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{},slice_type{nop_type{},nop_type{},-1}),
//             tensor_type{{{7,8,9},{4,5,6},{1,2,3}},{{16,17,18},{13,14,15},{10,11,12}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{1},slice_type{1}),
//             tensor_type{{{13,14,15},{16,17,18}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{-2},slice_type{-1,1,-1}),
//             tensor_type{{{7,8,9}},{{16,17,18}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{1,rtag_type{}},slice_type{}),
//             tensor_type{{10,11,12},{13,14,15},{16,17,18}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{1,rtag_type{}},slice_type{0,rtag_type{}}),
//             tensor_type{10,11,12}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{},slice_type{2,rtag_type{}}),
//             tensor_type{{7,8,9},{16,17,18}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{},slice_type{-1,rtag_type{}}),
//             tensor_type{{7,8,9},{16,17,18}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{},slice_type{},slice_type{}),
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{},slice_type{},slice_type{nop_type{},nop_type{},-2}),
//             tensor_type{{{3,1},{6,4},{9,7}},{{12,10},{15,13},{18,16}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{},slice_type{1,-1},slice_type{1,-1}),
//             tensor_type{{{5}},{{14}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{1},slice_type{1,-1},slice_type{1,-1}),
//             tensor_type{{{14}}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{1,rtag_type{}},slice_type{1,-1},slice_type{1,-1}),
//             tensor_type{{14}}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{1,rtag_type{}},slice_type{1,rtag_type{}}),
//             tensor_type{13,14,15}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(slice_type{-2,rtag_type{}},slice_type{-1,rtag_type{}},slice_type{nop_type{},nop_type{},-1}),
//             tensor_type{9,8,7}
//         )
//     );
//     SECTION("test_create_slice_view_variadic")
//     {
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto subs = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto apply_subs = [&parent](const auto&...subs_){
//                 return basic_tensor{view_factory_type::create_slice_view(parent, subs_...)};
//             };
//             auto result = std::apply(apply_subs, subs);
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test, test_data);
//     }
//     SECTION("test_create_slice_view_container")
//     {
//         using container_type = std::vector<slice_type>;
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto subs = std::get<1>(t);
//             auto expected = std::get<2>(t);
//             auto make_container = [](const auto&...subs_){
//                 return container_type{subs_...};
//             };
//             auto container = std::apply(make_container, subs);
//             auto result = basic_tensor{view_factory_type::create_slice_view(parent, container)};
//             REQUIRE(result == expected);
//         };
//         apply_by_element(test, test_data);
//     }
// }

// TEST_CASE("test_create_slice_view_mixed_subs","[test_view_factory]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using slice_type = typename tensor_type::slice_type;
//     using nop_type = typename slice_type::nop_type;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using gtensor::basic_tensor;
//     using helpers_for_testing::apply_by_element;
//     //0parent,1subs,2expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{1,2,3,4,5,6},std::make_tuple(0),tensor_type(1)),
//         std::make_tuple(tensor_type{1,2,3,4,5,6},std::make_tuple(1),tensor_type(2)),
//         std::make_tuple(tensor_type{1,2,3,4,5,6},std::make_tuple(-1),tensor_type(6)),
//         std::make_tuple(tensor_type{1,2,3,4,5,6},std::make_tuple(-2),tensor_type(5)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(0),tensor_type{1,2,3}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(1),tensor_type{4,5,6}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(2),tensor_type{7,8,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(-3),tensor_type{1,2,3}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(-2),tensor_type{4,5,6}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(-1),tensor_type{7,8,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(0,0),tensor_type(1)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(1,1),tensor_type(5)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(-1,-1),tensor_type(9)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},0),tensor_type{1,4,7}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},1),tensor_type{2,5,8}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},-1),tensor_type{3,6,9}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(slice_type{},-2),tensor_type{2,5,8}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(0,slice_type{}),tensor_type{1,2,3}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(1,slice_type{nop_type{},nop_type{},-1}),tensor_type{6,5,4}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(-1,slice_type{nop_type{},nop_type{},-2}),tensor_type{9,7}),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(-2,-1,slice_type{nop_type{},nop_type{},-1}),
//             tensor_type{9,8,7}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(1,slice_type{1},0),
//             tensor_type{13,16}
//         ),
//         std::make_tuple(
//             tensor_type{{{1,2,3},{4,5,6},{7,8,9}},{{10,11,12},{13,14,15},{16,17,18}}},
//             std::make_tuple(1,slice_type{1},1),
//             tensor_type{14,17}
//         )
//     );
//     auto test = [](const auto& t){
//         auto parent = std::get<0>(t);
//         auto subs = std::get<1>(t);
//         auto expected = std::get<2>(t);
//         auto apply_subs = [&parent](const auto&...subs_){
//             return basic_tensor{view_factory_type::create_slice_view(parent, subs_...)};
//         };
//         auto result = std::apply(apply_subs, subs);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_create_slice_view_init_list_interface","[test_view_factory]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using slice_type = typename tensor_type::slice_type;
//     using rtag_type = typename slice_type::reduce_tag_type;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using slice_item_type = typename slice_type::slice_item_type;
//     using list_type = std::initializer_list<std::initializer_list<slice_item_type>>;
//     using gtensor::basic_tensor;
//     using helpers_for_testing::apply_by_element;
//     //0result,1expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{},list_type{})},tensor_type{}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{},list_type{{-3,3}})},tensor_type{}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{})},tensor_type{1,2,3,4,5,6}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{0,10}})},tensor_type{1,2,3,4,5,6}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{-10,10}})},tensor_type{1,2,3,4,5,6}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{3,10}})},tensor_type{4,5,6}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{10,-10,-1}})},tensor_type{6,5,4,3,2,1}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{3,-10,-1}})},tensor_type{4,3,2,1}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{1,-1}})},tensor_type{2,3,4,5}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{-1,{},-1}})},tensor_type{6,5,4,3,2,1}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{1,2,3,4,5,6},list_type{{-1,2,-1}})},tensor_type{6,5,4}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{}})},tensor_type{{1,2,3},{4,5,6},{7,8,9}}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{{},{},-1}})},tensor_type{{3,2,1},{6,5,4},{9,8,7}}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{{},2}})},tensor_type{{1,2},{4,5},{7,8}}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{{},{},-1},{}})},tensor_type{{7,8,9},{4,5,6},{1,2,3}}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{0,rtag_type{}}})},tensor_type{1,4,7}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{-1,rtag_type{}}})},tensor_type{3,6,9}),
//         std::make_tuple(basic_tensor{view_factory_type::create_slice_view(tensor_type{{1,2,3},{4,5,6},{7,8,9}},list_type{{},{1,rtag_type{}}})},tensor_type{2,5,8})
//     );
//     auto test = [](const auto& t){
//         auto result = std::get<0>(t);
//         auto expected = std::get<1>(t);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_create_slice_exception","[test_view_factory]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using slice_type = typename tensor_type::slice_type;
//     using rtag_type = typename slice_type::reduce_tag_type;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using gtensor::basic_tensor;
//     using gtensor::subscript_exception;
//     using helpers_for_testing::apply_by_element;
//     //0parent,1subs
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type(2),std::make_tuple(slice_type{})),
//         std::make_tuple(tensor_type(2),std::make_tuple(slice_type{0,rtag_type{}})),
//         std::make_tuple(tensor_type(2),std::make_tuple(slice_type{},slice_type{})),
//         std::make_tuple(tensor_type{},std::make_tuple(slice_type{0,rtag_type{}})),
//         std::make_tuple(tensor_type{},std::make_tuple(slice_type{},slice_type{})),
//         std::make_tuple(tensor_type{}.reshape(4,0),std::make_tuple(slice_type{},slice_type{0,rtag_type{}})),
//         std::make_tuple(tensor_type{}.reshape(4,0),std::make_tuple(slice_type{},slice_type{1,rtag_type{}})),
//         std::make_tuple(tensor_type{1,2,3,4,5,6},std::make_tuple(slice_type{},slice_type{})),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(slice_type{},slice_type{},slice_type{})),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(slice_type{},slice_type{3,rtag_type{}})),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(slice_type{},slice_type{-4,rtag_type{}}))
//     );
//     SECTION("test_create_slice_view_variadic_exception")
//     {
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto subs = std::get<1>(t);
//             auto apply_subs = [&parent](const auto&...subs_){
//                 return basic_tensor{view_factory_type::create_slice_view(parent, subs_...)};
//             };
//             REQUIRE_THROWS_AS(std::apply(apply_subs, subs), subscript_exception);
//         };
//         apply_by_element(test, test_data);
//     }
//     SECTION("test_create_slice_view_container_exception")
//     {
//         using container_type = std::vector<slice_type>;
//         auto test = [](const auto& t){
//             auto parent = std::get<0>(t);
//             auto subs = std::get<1>(t);
//             auto make_container = [](const auto&...subs_){
//                 return container_type{subs_...};
//             };
//             auto container = std::apply(make_container, subs);
//             REQUIRE_THROWS_AS(view_factory_type::create_slice_view(parent, container), subscript_exception);
//         };
//         apply_by_element(test, test_data);
//     }
// }

// TEST_CASE("test_create_slice_exception_mixed_subs_exception","[test_view_factory]")
// {
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type>;
//     using config_type = typename tensor_type::config_type;
//     using slice_type = typename tensor_type::slice_type;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using gtensor::basic_tensor;
//     using gtensor::subscript_exception;
//     using helpers_for_testing::apply_by_element;
//     //0parent,1subs
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type(2),std::make_tuple(0)),
//         std::make_tuple(tensor_type(2),std::make_tuple(slice_type{},0)),
//         std::make_tuple(tensor_type(2),std::make_tuple(0,slice_type{})),
//         std::make_tuple(tensor_type{},std::make_tuple(0)),
//         std::make_tuple(tensor_type{},std::make_tuple(0,0)),
//         std::make_tuple(tensor_type{}.reshape(4,0),std::make_tuple(0,0)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(2)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(-3)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(2,slice_type{})),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(-3,slice_type{})),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(slice_type{},3)),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(slice_type{},-4))
//     );
//     auto test = [](const auto& t){
//         auto parent = std::get<0>(t);
//         auto subs = std::get<1>(t);
//         auto apply_subs = [&parent](const auto&...subs_){
//             return basic_tensor{view_factory_type::create_slice_view(parent, subs_...)};
//         };
//         REQUIRE_THROWS_AS(std::apply(apply_subs, subs), subscript_exception);
//     };
//     apply_by_element(test, test_data);
// }

// //test create_index_mapping_view
// TEMPLATE_TEST_CASE("test_create_index_mapping_view","[test_view_factory]",
//     //parent's order, subs order
//     (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
//     (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
//     (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
//     (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
// )
// {
//     using parent_order = std::tuple_element_t<0,TestType>;
//     using subs_order = std::tuple_element_t<1,TestType>;
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type,parent_order>;
//     using config_type = typename tensor_type::config_type;
//     using index_tensor_type = gtensor::tensor<int, subs_order, config_type>;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using gtensor::basic_tensor;
//     using helpers_for_testing::apply_by_element;
//     //0parent,1subs,2expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type{}),tensor_type{}),
//         std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type{}.reshape(2,3,0)),tensor_type{}.reshape(2,3,0)),
//         std::make_tuple(tensor_type{}.reshape(1,0),std::make_tuple(index_tensor_type(0)),tensor_type{}),
//         std::make_tuple(tensor_type{}.reshape(1,0),std::make_tuple(index_tensor_type{0}),tensor_type{}.reshape(1,0)),
//         std::make_tuple(tensor_type{}.reshape(1,0),std::make_tuple(index_tensor_type{0,0,0}),tensor_type{}.reshape(3,0)),
//         std::make_tuple(tensor_type{}.reshape(1,0),std::make_tuple(index_tensor_type{0},index_tensor_type{}),tensor_type{}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type(1)),tensor_type{}.reshape(3,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{0,1,0,1,0}),tensor_type{}.reshape(5,3,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{4,1,2,1,3}),tensor_type{}.reshape(5,3,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{0,1},index_tensor_type{2}),tensor_type{}.reshape(2,0)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{0,1},index_tensor_type{2},index_tensor_type{}.reshape(0,3,1)),tensor_type{}.reshape(0,3,2)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{{0,1}},index_tensor_type{{0,2}},index_tensor_type{}.reshape(0,3,1)),tensor_type{}.reshape(0,3,2)),
//         std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{{0,1}},index_tensor_type{4},index_tensor_type{}.reshape(0,3,1)),tensor_type{}.reshape(0,3,2)),
//         std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type(0)), tensor_type(1)),
//         std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type{0}), tensor_type{1}),
//         std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type{0,0,0}), tensor_type{1,1,1}),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(index_tensor_type(3)), tensor_type(4)),
//         std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(index_tensor_type{1,1,0,0}), tensor_type{2,2,1,1}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}},std::make_tuple(index_tensor_type{{1,2},{0,1}}), tensor_type{{{4,5,6},{7,8,9}},{{1,2,3},{4,5,6}}}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},std::make_tuple(index_tensor_type{{0,0},{3,3}}, index_tensor_type{{0,2},{0,2}}), tensor_type{{1,3},{10,12}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1}), tensor_type{{{5,6},{7,8}}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type(1)), tensor_type{{5,6},{7,8}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type(1),index_tensor_type(0)), tensor_type{5,6}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1,3}), tensor_type{{{5,6},{7,8}},{{13,14},{15,16}}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1,3}, index_tensor_type{0,1}), tensor_type{{5,6},{15,16}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1,3}, index_tensor_type(1)), tensor_type{{7,8},{15,16}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type(2), index_tensor_type{1,0}), tensor_type{{11,12},{9,10}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type(2), index_tensor_type{{1,0},{0,1}}), tensor_type{{{11,12},{9,10}},{{9,10},{11,12}}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1,3}, index_tensor_type{1}), tensor_type{{7,8},{15,16}}),
//         std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}},std::make_tuple(index_tensor_type{1,3}, index_tensor_type{{1,0},{0,1}}), tensor_type{{{7,8},{13,14}},{{5,6},{15,16}}})
//     );
//     auto test = [](const auto& t){
//         auto parent = std::get<0>(t);
//         auto subs = std::get<1>(t);
//         auto expected = std::get<2>(t);
//         auto apply_subs = [&parent](const auto&...subs_){
//             return basic_tensor{view_factory_type::create_index_mapping_view(parent, subs_...)};
//         };
//         auto result = std::apply(apply_subs, subs);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_create_index_mapping_view_exception","[test_view_factory]",
//     //parent's order, subs order
//     (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
//     (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
//     (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
//     (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
// )
// {
//     using parent_order = std::tuple_element_t<0,TestType>;
//     using subs_order = std::tuple_element_t<1,TestType>;
//     using value_type = double;
//     using tensor_type = gtensor::tensor<value_type,parent_order>;
//     using index_tensor_type = gtensor::tensor<int,subs_order>;
//     using config_type = typename tensor_type::config_type;
//     using view_factory_type = gtensor::view_factory_selector_t<config_type>;
//     using gtensor::basic_tensor;
//     using gtensor::subscript_exception;
//     using gtensor::broadcast_exception;
//     using helpers_for_testing::apply_by_element;
//     //0parent,1subs,2exception
//     auto test_data = std::make_tuple(
//         //0-dim tensor
//         std::make_tuple(tensor_type(2),std::make_tuple(index_tensor_type{}),subscript_exception{""}),
//         //exception, parent zero size direction and non zero size subs
//         std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type(0)),subscript_exception{""}),
//         std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type{0}),subscript_exception{""}),
//         std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type{1}),subscript_exception{""}),
//         std::make_tuple(tensor_type{}.reshape(2,3,0),std::make_tuple(index_tensor_type{1},index_tensor_type{2},index_tensor_type{0}),subscript_exception{""}),
//         //exception, subs number more than parent dim
//         std::make_tuple(tensor_type{},std::make_tuple(index_tensor_type{},index_tensor_type{}),subscript_exception{""}),
//         std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type{0},index_tensor_type{0,0,0}),subscript_exception{""}),
//         std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type{0,1},index_tensor_type{0,1}),subscript_exception{""}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(index_tensor_type{0,1},index_tensor_type{1,1},index_tensor_type{}),subscript_exception{""}),
//         //exception, subs shapes not broadcast
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(index_tensor_type{0,0},index_tensor_type{0,0,0}),broadcast_exception{""}),
//         //exception, subs out of bounds
//         std::make_tuple(tensor_type{1},std::make_tuple(index_tensor_type{0,4,0}),subscript_exception{""}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(index_tensor_type{3}),subscript_exception{""}),
//         std::make_tuple(tensor_type{{1,2,3},{4,5,6}},std::make_tuple(index_tensor_type{0},index_tensor_type{1,2,3}),subscript_exception{""})
//     );
//     auto test = [](const auto& t){
//         auto parent = std::get<0>(t);
//         auto subs = std::get<1>(t);
//         auto exception = std::get<2>(t);
//         auto apply_subs = [&parent](const auto&...subs_){
//             return basic_tensor{view_factory_type::create_index_mapping_view(parent, subs_...)};
//         };
//         REQUIRE_THROWS_AS(std::apply(apply_subs, subs), decltype(exception));
//     };
//     apply_by_element(test,test_data);
// }

//test create_bool_mapping_view
TEMPLATE_TEST_CASE("test_create_bool_mapping_view","[test_view_factory]",
    //parent's order, subs order
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using parent_order = std::tuple_element_t<0,TestType>;
    using subs_order = std::tuple_element_t<1,TestType>;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,parent_order>;
    using bool_tensor_type = gtensor::tensor<bool,subs_order>;
    using config_type = typename tensor_type::config_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, bool_tensor_type{}, tensor_type{}),
        std::make_tuple(tensor_type{}, bool_tensor_type(false), tensor_type{}.reshape(0,0)),
        std::make_tuple(tensor_type{}, bool_tensor_type(true), tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type(false), tensor_type{}.reshape(0,2,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type(true), tensor_type{}.reshape(1,2,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{}.reshape(2,0), tensor_type{}.reshape(0,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{}.reshape(2,3,0), tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{false,false}, tensor_type{}.reshape(0,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{true,false}, tensor_type{}.reshape(1,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{true,true}, tensor_type{}.reshape(2,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{{true,true,false},{false,true,true}}, tensor_type{}.reshape(4,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{}.reshape(2,3,0), tensor_type{}),
        std::make_tuple(tensor_type{1}, bool_tensor_type(false), tensor_type{}.reshape(0,1)),
        std::make_tuple(tensor_type{1}, bool_tensor_type(true), tensor_type{{1}}),
        std::make_tuple(tensor_type{1}, bool_tensor_type{}, tensor_type{}),
        std::make_tuple(tensor_type{1}, bool_tensor_type{false}, tensor_type{}),
        std::make_tuple(tensor_type{1}, bool_tensor_type{true}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type(false), tensor_type{}.reshape(0,5)),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type(true), tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type{false,true,false,true,false}, tensor_type{2,4}),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type{true,true,true,true,true}, tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type{false,false,false,false,false}, tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5}, bool_tensor_type{true,true}, tensor_type{1,2}),
        std::make_tuple(tensor_type{{1,2,3,4,5}}, bool_tensor_type{{false,false,true,false,true}}, tensor_type{3,5}),
        std::make_tuple(tensor_type{{1,2,3,4,5}}, bool_tensor_type{false}, tensor_type{}.reshape(0,5)),
        std::make_tuple(tensor_type{{1,2,3,4,5}}, bool_tensor_type{true}, tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}}, bool_tensor_type{{true,false},{false,true}}, tensor_type{1,6}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}}, bool_tensor_type{true,true}, tensor_type{{1,2,3,4},{5,6,7,8}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}, bool_tensor_type{true}, tensor_type{{{1,2},{3,4}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}, bool_tensor_type{true,true}, tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}, bool_tensor_type{false,true,false,true}, tensor_type{{{5,6},{7,8}},{{13,14},{15,16}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}, bool_tensor_type{{false,true},{true,false}}, tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}, bool_tensor_type{{{false,true}},{{true,false}}}, tensor_type{2,5})
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto subs = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = basic_tensor{view_factory_type::create_bool_mapping_view(parent, subs)};
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_create_bool_mapping_view_exception","[test_view_factory]",
    //parent's order, subs order
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using parent_order = std::tuple_element_t<0,TestType>;
    using subs_order = std::tuple_element_t<1,TestType>;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,parent_order>;
    using bool_tensor_type = gtensor::tensor<bool,subs_order>;
    using config_type = typename tensor_type::config_type;
    using view_factory_type = gtensor::view_factory_selector_t<config_type>;
    using gtensor::basic_tensor;
    using gtensor::subscript_exception;
    using helpers_for_testing::apply_by_element;
    //0parent,1subs
    auto test_data = std::make_tuple(
        //0-dim tensor
        std::make_tuple(tensor_type(2), bool_tensor_type{}),
        //exception, subs dim > parent dim
        std::make_tuple(tensor_type{}, bool_tensor_type{}.reshape(0,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{}.reshape(1,2,3,0)),
        std::make_tuple(tensor_type{1}, bool_tensor_type{{true}}),
        std::make_tuple(tensor_type{1}, bool_tensor_type{{false}}),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}}, bool_tensor_type{{{true}}}),
        //exception, subs out of bounds
        std::make_tuple(tensor_type{}, bool_tensor_type{true}),
        std::make_tuple(tensor_type{}.reshape(1,0), bool_tensor_type{{true}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), bool_tensor_type{}.reshape(3,3,0)),
        std::make_tuple(tensor_type{1}, bool_tensor_type{true,true}),
        std::make_tuple(tensor_type{1}, bool_tensor_type{false,false}),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}}, bool_tensor_type{{true,false,true}})
    );
    auto test = [](const auto& t){
        auto parent = std::get<0>(t);
        auto subs = std::get<1>(t);
        REQUIRE_THROWS_AS(view_factory_type::create_bool_mapping_view(parent, subs), subscript_exception);
    };
    apply_by_element(test,test_data);
}


