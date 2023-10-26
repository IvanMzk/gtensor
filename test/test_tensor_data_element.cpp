#include <vector>
#include <tuple>
#include "catch.hpp"
#include "tensor.hpp"
#include "config_for_testing.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_tensor_data","[test_tensor]")
{
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<gtensor::basic_storage>,value_type>;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using tensor_type_c = tensor<value_type,c_order,config_type>;
    using tensor_type_f = tensor<value_type,f_order,config_type>;

    const auto elements = std::vector<value_type>{1,2,3,4,5,6,1,2,3,3,2,1};

    //0ten,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type_c(2),std::vector<value_type>{2}),
        std::make_tuple(tensor_type_c{3},std::vector<value_type>{3}),
        std::make_tuple(tensor_type_c{1,2,3,4,5},std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(tensor_type_c{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12}),
        std::make_tuple(tensor_type_f(2),std::vector<value_type>{2}),
        std::make_tuple(tensor_type_f{3},std::vector<value_type>{3}),
        std::make_tuple(tensor_type_f{1,2,3,4,5},std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(tensor_type_f{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(tensor_type_c(12,elements.begin(),elements.end()),std::vector<value_type>(elements.begin(),elements.end())),
        std::make_tuple(tensor_type_f(12,elements.begin(),elements.end()),std::vector<value_type>(elements.begin(),elements.end())),
        std::make_tuple(tensor_type_c({2,3,2},elements.begin(),elements.end()),std::vector<value_type>(elements.begin(),elements.end())),
        std::make_tuple(tensor_type_f({2,2,3},elements.begin(),elements.end()),std::vector<value_type>(elements.begin(),elements.end()))
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);
        auto size = static_cast<std::ptrdiff_t>(ten.size());
        REQUIRE(std::equal(ten.data(),ten.data()+size,expected.begin(),expected.end()));
    };
    helpers_for_testing::apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_data_result_type","[test_tensor]")
{
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<gtensor::basic_storage>,double>;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;

    REQUIRE(std::is_same_v<decltype(std::declval<tensor<double,c_order,config_type>>().data()), double*>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor<double,c_order,config_type>>().data()), const double*>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor<int,c_order,config_type>>().data()), int*>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor<int,c_order,config_type>>().data()), const int*>);
}

TEMPLATE_TEST_CASE("test_tensor_element","[test_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type,TestType>;

    const auto a = tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}};
    const auto b = tensor_type{3,2,1};

    //0ten,1indxes,2expected
    auto test_data = std::make_tuple(
        //tensor
        std::make_tuple(tensor_type(2),std::make_tuple(),value_type{2}),
        std::make_tuple(tensor_type{3},std::make_tuple(0),value_type{3}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(0),value_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(1),value_type{2}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(2),value_type{3}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(3),value_type{4}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::make_tuple(4),value_type{5}),
        std::make_tuple(a,std::make_tuple(0,0,0),value_type{1}),
        std::make_tuple(a,std::make_tuple(1,1,1),value_type{11}),
        std::make_tuple(a,std::make_tuple(0,0,2),value_type{3}),
        std::make_tuple(a,std::make_tuple(1,0,0),value_type{7}),
        std::make_tuple(a,std::make_tuple(0,1,2),value_type{6}),
        std::make_tuple(a,std::make_tuple(1,1,2),value_type{12}),
        //expression view
        std::make_tuple(2*(a-1),std::make_tuple(0,0,0),value_type{0}),
        std::make_tuple(2*(a-1),std::make_tuple(1,1,1),value_type{20}),
        std::make_tuple(2*(a-1),std::make_tuple(0,0,2),value_type{4}),
        std::make_tuple(2*(a-1),std::make_tuple(1,0,0),value_type{12}),
        std::make_tuple(2*(a-1),std::make_tuple(0,1,2),value_type{10}),
        std::make_tuple(2*(a-1),std::make_tuple(1,1,2),value_type{22}),
        std::make_tuple((a+b)*(b-1),std::make_tuple(0,0,0),value_type{8}),
        std::make_tuple((a+b)*(b-1),std::make_tuple(1,1,1),value_type{13}),
        std::make_tuple((a+b)*(b-1),std::make_tuple(0,0,2),value_type{0}),
        std::make_tuple((a+b)*(b-1),std::make_tuple(1,0,0),value_type{20}),
        std::make_tuple((a+b)*(b-1),std::make_tuple(0,1,2),value_type{0}),
        std::make_tuple((a+b)*(b-1),std::make_tuple(1,1,2),value_type{0}),
        //transpose view
        std::make_tuple(a.transpose(),std::make_tuple(0,0,0),value_type{1}),
        std::make_tuple(a.transpose(),std::make_tuple(1,1,1),value_type{11}),
        std::make_tuple(a.transpose(),std::make_tuple(2,1,0),value_type{6}),
        std::make_tuple(a.transpose(),std::make_tuple(1,0,1),value_type{8}),
        std::make_tuple(a.transpose(),std::make_tuple(2,1,1),value_type{12}),
        //slice view
        std::make_tuple(a({{},{1},{1}}),std::make_tuple(0,0,0),value_type{5}),
        std::make_tuple(a({{},{1},{1}}),std::make_tuple(0,0,1),value_type{6}),
        std::make_tuple(a({{},{1},{1}}),std::make_tuple(1,0,0),value_type{11}),
        std::make_tuple(a({{},{1},{1}}),std::make_tuple(1,0,1),value_type{12}),
        std::make_tuple(a(1,0,1),std::make_tuple(),value_type{8}),
        std::make_tuple(a(0,1,1),std::make_tuple(),value_type{5}),
        //reshape view
        std::make_tuple(a.reshape(3,4),std::make_tuple(0,0),value_type{1}),
        std::make_tuple(a.reshape(3,4),std::make_tuple(1,1),value_type{6}),
        std::make_tuple(a.reshape(3,4),std::make_tuple(2,1),value_type{10}),
        std::make_tuple(a.reshape(3,4),std::make_tuple(1,2),value_type{7}),
        std::make_tuple(a.reshape(3,4),std::make_tuple(2,3),value_type{12}),
        //mapping view
        std::make_tuple(a(tensor<int>{1,0},tensor<int>{0,1}),std::make_tuple(0,0),value_type{7}),
        std::make_tuple(a(tensor<int>{1,0},tensor<int>{0,1}),std::make_tuple(1,1),value_type{5}),
        std::make_tuple(a(tensor<int>{1,0},tensor<int>{0,1}),std::make_tuple(1,2),value_type{6}),
        std::make_tuple(a(tensor<int>{1,0},tensor<int>{0,1}),std::make_tuple(0,2),value_type{9}),
        std::make_tuple(a(gtensor::cast<int>(a)%2>0),std::make_tuple(0),value_type{1}),
        std::make_tuple(a(gtensor::cast<int>(a)%2>0),std::make_tuple(1),value_type{3}),
        std::make_tuple(a(gtensor::cast<int>(a)%2>0),std::make_tuple(2),value_type{5}),
        std::make_tuple(a(gtensor::cast<int>(a)%2>0),std::make_tuple(3),value_type{7}),
        std::make_tuple(a(gtensor::cast<int>(a)%2>0),std::make_tuple(4),value_type{9}),
        std::make_tuple(a(gtensor::cast<int>(a)%2>0),std::make_tuple(5),value_type{11})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto indexes = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto apply_indexes = [&ten](auto...indexes_){
            return ten.element(indexes_...);
        };
        auto result = std::apply(apply_indexes,indexes);
        REQUIRE(result == expected);
    };
    helpers_for_testing::apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_element_result_type","[test_tensor]")
{
    using gtensor::tensor;
    //tensor
    REQUIRE(std::is_same_v<decltype(std::declval<tensor<double>>().element(std::declval<int>())), double&>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor<double>>().element(std::declval<int>())), const double&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor<int>>().element(std::declval<int>())), int&>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor<int>>().element(std::declval<int>())), const int&>);
    //expression view
    REQUIRE(std::is_same_v<decltype((std::declval<tensor<double>>()+std::declval<tensor<double>>()+std::declval<tensor<double>>()).element(std::declval<int>())), double>);
    REQUIRE(std::is_same_v<decltype((2*(std::declval<tensor<double>>()-1)).element(std::declval<int>())), double>);
    //transpose view
    REQUIRE(std::is_same_v<decltype(std::declval<tensor<double>>().transpose().element(std::declval<int>())), double&>);
    //slice view
    REQUIRE(std::is_same_v<decltype(std::declval<tensor<double>>()({{},{1},{1}}).element(std::declval<int>())), double&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor<double>>()(std::declval<int>()).element()), double&>);
    //reshape view
    REQUIRE(std::is_same_v<decltype(std::declval<tensor<double>>().reshape(1,-1).element(std::declval<int>())), double&>);
    //mapping view
    REQUIRE(std::is_same_v<decltype(std::declval<tensor<double>>()(std::declval<tensor<int>>()).element()), double&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor<double>>()(std::declval<tensor<bool>>()).element()), double&>);
}

