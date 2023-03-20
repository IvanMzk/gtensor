#include <tuple>
#include "catch.hpp"
#include "test_config.hpp"
#include "gtensor.hpp"
#include "helpers_for_testing.hpp"

TEMPLATE_TEST_CASE("test_view","[test_view]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using index_type = typename config_type::index_type;
    using tensor_type = gtensor::tensor<value_type,config_type>;
    using index_tensor_type =  gtensor::tensor<index_type, config_type>;
    using bool_tensor_type =  gtensor::tensor<bool, config_type>;
    auto nop = typename gtensor::slice_traits<config_type>::nop_type{};
    using helpers_for_testing::apply_by_element;

    //0result,1expected
    auto test_data = std::make_tuple(
        //slice view slice-direction interface
        std::make_tuple(tensor_type{1}({},0),tensor_type{1}),
        std::make_tuple(tensor_type{1}({0,1},0),tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}({},0),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5}({2,nop},0),tensor_type{3,4,5}),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}}({1,-1},0),tensor_type{{3,4}}),
        std::make_tuple(tensor_type{{1,2},{3,4},{5,6}}({{},{},-1},0),tensor_type{{5,6},{3,4},{1,2}}),
        //slice view init-list interface
        std::make_tuple(tensor_type{1}({{}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{0}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{0,1}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{{},1}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{{},{},1}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{nop,1}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{nop,nop,1}}),tensor_type{1}),
        std::make_tuple(tensor_type{1}({{}})({{}}),tensor_type{1}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{}}),tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{},{}}),tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{1},{},{}}),tensor_type{{{7,8},{9,10},{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{-1},{}}),tensor_type{{{5,6}},{{11,12}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{0,1},{0,-1},{}}),tensor_type{{{1,2},{3,4}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{},{{},{},-1}}),tensor_type{{{2,1},{4,3},{6,5}},{{8,7},{10,9},{12,11}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{},{{},{},-1}})({{0,1},{0,-1}}),tensor_type{{{2,1},{4,3}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}({{},{{},{},2},{{},{},-1}}),tensor_type{{{2,1},{6,5}},{{8,7},{12,11}}}),
        //transpose view
        std::make_tuple(tensor_type{1}.transpose(),tensor_type{1}),
        std::make_tuple(tensor_type{1}.transpose(0),tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}.transpose(),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5}.transpose(0),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1,2,3,4,5}}.transpose(),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{1,2,3,4,5}}.transpose(1,0),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{1,2,3,4,5}}.transpose(0,1),tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}}.transpose().transpose(),tensor_type{{{1,2},{3,4},{5,6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}}.transpose(),tensor_type{{{1},{3},{5}},{{2},{4},{6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}}.transpose(2,1,0),tensor_type{{{1},{3},{5}},{{2},{4},{6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}}.transpose(2,0,1),tensor_type{{{1,3,5}},{{2,4,6}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}}}.transpose(1,0,2),tensor_type{{{1,2}},{{3,4}},{{5,6}}}),
        //subdim view
        std::make_tuple(tensor_type{1}(),tensor_type{1}),
        std::make_tuple(tensor_type{1}()(),tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}(),tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}}}(0),tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}}}(0,0),tensor_type{1,2,3}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}}}(0,1),tensor_type{4,5,6}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(1),tensor_type{{5,6},{7,8}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(2,0),tensor_type{9,10}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(2)(0),tensor_type{9,10}),
        //reshape view
        std::make_tuple(tensor_type{1}.reshape(),tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}.reshape(1,5),tensor_type{{1,2,3,4,5}}),
        std::make_tuple(tensor_type{1,2,3,4,5}.reshape(5,1),tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}.reshape(6,2), tensor_type{{1,2},{3,4},{5,6},{7,8},{9,10},{11,12}}),
        //index mapping view
        std::make_tuple(tensor_type{1}(index_tensor_type{0}), tensor_type{1}),
        std::make_tuple(tensor_type{1}(index_tensor_type{0,0,0}), tensor_type{1,1,1}),
        std::make_tuple(tensor_type{1}(index_tensor_type{0,0,0})(index_tensor_type{0,0,0,0}), tensor_type{1,1,1,1}),
        std::make_tuple(tensor_type{1,2,3,4,5}(index_tensor_type{1,1,0,0}), tensor_type{2,2,1,1}),
        std::make_tuple(tensor_type{1,2,3,4,5}(index_tensor_type{1,1,0,0})(index_tensor_type{0,3,3,0}), tensor_type{2,1,1,2}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(index_tensor_type{1,3}), tensor_type{{{5,6},{7,8}},{{13,14},{15,16}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(index_tensor_type{1,3}, index_tensor_type{0,1}), tensor_type{{5,6},{15,16}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(index_tensor_type{1,3}, index_tensor_type{1}), tensor_type{{7,8},{15,16}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}(index_tensor_type{{0,0},{3,3}}, index_tensor_type{{0,2},{0,2}}), tensor_type{{1,3},{10,12}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}(index_tensor_type{3,2,1,0})(index_tensor_type{{0,0},{3,3}}, index_tensor_type{{0,2},{0,2}}), tensor_type{{10,12},{1,3}}),
        //bool mapping view
        std::make_tuple(tensor_type{}(bool_tensor_type{}), tensor_type{}),
        std::make_tuple(tensor_type{}(bool_tensor_type{true}), tensor_type{}),
        std::make_tuple(tensor_type{1}(bool_tensor_type{}), tensor_type{}),
        std::make_tuple(tensor_type{1}(bool_tensor_type{false}), tensor_type{}),
        std::make_tuple(tensor_type{1}(bool_tensor_type{true}), tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}(bool_tensor_type{false,true,false,true,false}), tensor_type{2,4}),
        std::make_tuple(tensor_type{1,2,3,4,5}(bool_tensor_type{true,true,true,true,true}), tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5}(bool_tensor_type{false,false,false,false,false}), tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5}(bool_tensor_type{true,true}), tensor_type{1,2}),
        std::make_tuple(tensor_type{{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}}(bool_tensor_type{{true,false},{false,true}}), tensor_type{1,6}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{true}),tensor_type{{{1,2},{3,4}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{true,true}),tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{false,true,false,true}),tensor_type{{{5,6},{7,8}},{{13,14},{15,16}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{{false,true},{true,false}}),tensor_type{{3,4},{5,6}}),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}(bool_tensor_type{{{false,true}},{{true,false}}}),tensor_type{2,5}),
        std::make_tuple([](){auto x = tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}; return x(x>tensor_type{10});}(),tensor_type{11,12,13,14,15,16}),
        std::make_tuple([](){auto x = tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}; return x(x<tensor_type{5});}(),tensor_type{1,2,3,4}),
        std::make_tuple([](){auto x = tensor_type{{{1,2},{3,4}},{{5,6},{7,8}},{{9,10},{11,12}},{{13,14},{15,16}}}; return x(x>tensor_type{5} && x<tensor_type{10});}(),tensor_type{6,7,8,9}),
        //view composition
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose().reshape(2,3), tensor_type{{1,4,2},{5,3,6}}),
        std::make_tuple((tensor_type{{1,2,3},{4,5,6}}.transpose().reshape(2,3) + tensor_type{{0},{1}})(bool_tensor_type{{false,true,false},{true,false,true}}), tensor_type{4,6,7}),
        std::make_tuple((tensor_type{{1,2,3},{4,5,6}}.transpose().reshape(2,3) + tensor_type{{0},{1}})(index_tensor_type{1,0,1}), tensor_type{{6,4,7},{1,4,2},{6,4,7}})

    );

    auto test = [](auto& t){
        auto result = std::get<0>(t);
        auto expected = std::get<1>(t);
        //REQUIRE(result.shape() == expected.shape());
        REQUIRE(std::equal(result.begin(), result.end(), expected.begin()));
        //REQUIRE(result.equals(expected));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_prev","[test_view]",
    test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using value_type = double;
    using config_type = TestType;
    using tensor_type = gtensor::tensor<value_type,config_type>;

    tensor_type t{1,2,3,4,5,6};
    auto e = t+t;
    //auto it = e.end();
    //auto prev_it = std::prev(it);
}