#include <tuple>
#include <vector>
#include "catch.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEST_CASE("test_tensor_meta_data_interface","[test_tensor]")
{
    using value_type = double;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,value_type>;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using helpers_for_testing::apply_by_element;
    //0tensor,1expected_dim,2expected_size,3expected_shape,4expected_strides
    auto test_data = std::make_tuple(
        //c_order layout
        std::make_tuple(tensor<value_type,c_order>(1),dim_type{0},index_type{1},shape_type{},shape_type{}),
        std::make_tuple(tensor<value_type,c_order>{},dim_type{1},index_type{0},shape_type{0},shape_type{1}),
        std::make_tuple(tensor<value_type,c_order>{1},dim_type{1},index_type{1},shape_type{1},shape_type{1}),
        std::make_tuple(tensor<value_type,c_order>{1,2,3},dim_type{1},index_type{3},shape_type{3},shape_type{1}),
        std::make_tuple(tensor<value_type,c_order>{{{1,2},{3,4}},{{5,6},{7,8}}},dim_type{3},index_type{8},shape_type{2,2,2},shape_type{4,2,1}),
        //f_order layout
        std::make_tuple(tensor<value_type,f_order>(1),dim_type{0},index_type{1},shape_type{},shape_type{}),
        std::make_tuple(tensor<value_type,f_order>{},dim_type{1},index_type{0},shape_type{0},shape_type{1}),
        std::make_tuple(tensor<value_type,f_order>{1},dim_type{1},index_type{1},shape_type{1},shape_type{1}),
        std::make_tuple(tensor<value_type,f_order>{1,2,3},dim_type{1},index_type{3},shape_type{3},shape_type{1}),
        std::make_tuple(tensor<value_type,f_order>{{{1,2},{3,4}},{{5,6},{7,8}}},dim_type{3},index_type{8},shape_type{2,2,2},shape_type{1,2,4})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected_dim = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_shape = std::get<3>(t);
        auto expected_strides = std::get<4>(t);

        auto result_dim = ten.dim();
        auto result_size = ten.size();
        auto result_shape = ten.shape();
        auto result_strides = ten.strides();

        REQUIRE(result_dim == expected_dim);
        REQUIRE(result_size == expected_size);
        REQUIRE(result_shape == expected_shape);
        REQUIRE(result_strides == expected_strides);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_data_interface","[test_tensor]",
    (gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
    (gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>)
)
{
    using tensor_type = TestType;
    using config_type = typename tensor_type::config_type;
    using traverse_order = typename config_type::order;
    using index_type = typename tensor_type::index_type;
    using value_type = typename tensor_type::value_type;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;
    //0tensor,1elements_c_traverse,2elements_f_traverse
    auto test_data = std::make_tuple(
        //tensor
        std::make_tuple(tensor_type(1),std::vector<value_type>{1},std::vector<value_type>{1}),
        std::make_tuple(tensor_type{},std::vector<value_type>{},std::vector<value_type>{}),
        std::make_tuple(tensor_type{2},std::vector<value_type>{2},std::vector<value_type>{2}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::vector<value_type>{1,2,3,4,5,6},std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        //trivial view
        std::make_tuple(tensor_type(1)+tensor_type(2)+tensor_type(3),std::vector<value_type>{6},std::vector<value_type>{6}),
        std::make_tuple((tensor_type(1)+tensor_type(2)+tensor_type(3))*4,std::vector<value_type>{24},std::vector<value_type>{24}),
        std::make_tuple((tensor_type{1}+tensor_type{2}+tensor_type{3})*4,std::vector<value_type>{24},std::vector<value_type>{24}),
        std::make_tuple((tensor_type{1,2,3,4,5,6} + 1)*tensor_type{2,3,4,5,6,7},std::vector<value_type>{4,9,16,25,36,49},std::vector<value_type>{4,9,16,25,36,49}),
        std::make_tuple((tensor_type{{1,2,3},{4,5,6}} + 1)*tensor_type{{2,3,4},{5,6,7}}-tensor_type{{1,1,1},{2,2,2}},std::vector<value_type>{3,8,15,23,34,47},std::vector<value_type>{3,23,8,34,15,47})
    );
    //data interface
    SECTION("test_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto first = ten.begin();
            auto last = ten.end();
            auto first_trivial = ten.begin_trivial();
            auto last_trivial = ten.end_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
                REQUIRE(std::equal(first_trivial,last_trivial,elements_c_traverse.begin(),elements_c_traverse.end()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
                REQUIRE(std::equal(first_trivial,last_trivial,elements_f_traverse.begin(),elements_f_traverse.end()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto first = ten.rbegin();
            auto last = ten.rend();
            auto first_trivial = ten.rbegin_trivial();
            auto last_trivial = ten.rend_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::is_same_v<decltype(first_trivial),decltype(last_trivial)>);
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
                REQUIRE(std::equal(first_trivial,last_trivial,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
                REQUIRE(std::equal(first_trivial,last_trivial,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_indexer")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto indexer = ten.create_indexer();
            std::vector<value_type> traverse_result{};
            for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
                traverse_result.push_back(indexer[i]);
            }
            auto first = traverse_result.begin();
            auto last = traverse_result.end();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_walker_c_order_traverse")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto walker = ten.create_walker();
            using walker_type = decltype(walker);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,gtensor::config::c_order>;
            using walker_iterator_type = gtensor::walker_iterator<config_type,traverser_type>;
            walker_iterator_type first{walker, ten.shape(), ten.descriptor().strides_div(c_order{}), index_type{0}};
            walker_iterator_type last{walker, ten.shape(), ten.descriptor().strides_div(c_order{}), ten.size()};
            REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_walker_f_order_traverse")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto walker = ten.create_walker();
            using walker_type = decltype(walker);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,gtensor::config::f_order>;
            using walker_iterator_type = gtensor::walker_iterator<config_type,traverser_type>;
            walker_iterator_type first{walker, ten.shape(), ten.descriptor().strides_div(f_order{}), index_type{0}};
            walker_iterator_type last{walker, ten.shape(), ten.descriptor().strides_div(f_order{}), ten.size()};
            REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
        };
        apply_by_element(test,test_data);
    }

    //traverse adapter
    SECTION("test_traverse_adapter_c_order_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto first = a.begin();
            auto last = a.end();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto first = a.begin();
            auto last = a.end();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_c_order_reverse_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto first = a.rbegin();
            auto last = a.rend();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_reverse_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto first = a.rbegin();
            auto last = a.rend();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_c_order_indexer")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto indexer = a.create_indexer();
            std::vector<value_type> traverse_result{};
            for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
                traverse_result.push_back(indexer[i]);
            }
            auto first = traverse_result.begin();
            auto last = traverse_result.end();
            REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_indexer")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto indexer = a.create_indexer();
            std::vector<value_type> traverse_result{};
            for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
                traverse_result.push_back(indexer[i]);
            }
            auto first = traverse_result.begin();
            auto last = traverse_result.end();
            REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_tensor_broadcast_iterator","[test_tensor]",
    (gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
    (gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>)
)
{
    using tensor_type = TestType;
    using config_type = typename tensor_type::config_type;
    using traverse_order = typename config_type::order;
    using shape_type = typename tensor_type::shape_type;
    using value_type = typename tensor_type::value_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;
    //0tensor,1broadcast_shape,2elements_c_traverse,3elements_f_traverse
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(2),shape_type{1},std::vector<value_type>{2},std::vector<value_type>{2}),
        std::make_tuple(tensor_type(2),shape_type{1,1},std::vector<value_type>{2},std::vector<value_type>{2}),
        std::make_tuple(tensor_type(2),shape_type{5},std::vector<value_type>{2,2,2,2,2},std::vector<value_type>{2,2,2,2,2}),
        std::make_tuple(tensor_type(2),shape_type{2,3},std::vector<value_type>{2,2,2,2,2,2},std::vector<value_type>{2,2,2,2,2,2}),
        std::make_tuple(tensor_type{1},shape_type{1},std::vector<value_type>{1},std::vector<value_type>{1}),
        std::make_tuple(tensor_type{1},shape_type{1,1},std::vector<value_type>{1},std::vector<value_type>{1}),
        std::make_tuple(tensor_type{1},shape_type{5},std::vector<value_type>{1,1,1,1,1},std::vector<value_type>{1,1,1,1,1}),
        std::make_tuple(tensor_type{1},shape_type{2,3},std::vector<value_type>{1,1,1,1,1,1},std::vector<value_type>{1,1,1,1,1,1}),
        std::make_tuple(tensor_type{1,2,3,4,5},shape_type{5},std::vector<value_type>{1,2,3,4,5},std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5},shape_type{2,5},std::vector<value_type>{1,2,3,4,5,1,2,3,4,5},std::vector<value_type>{1,1,2,2,3,3,4,4,5,5}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},shape_type{2,3},std::vector<value_type>{1,2,3,4,5,6},std::vector<value_type>{1,4,2,5,3,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},shape_type{1,2,2,3},std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6},std::vector<value_type>{1,1,4,4,2,2,5,5,3,3,6,6})
    );

    SECTION("test_broadcast_iterator")
    {
        auto test = [](const auto& t){
            tensor_type ten = std::get<0>(t);
            auto broadcast_shape = std::get<1>(t);
            auto elements_c_traverse = std::get<2>(t);
            auto elements_f_traverse = std::get<3>(t);
            auto first = ten.begin(broadcast_shape);
            auto last = ten.end(broadcast_shape);
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_reverse_broadcast_iterator")
    {
        auto test = [](const auto& t){
            tensor_type ten = std::get<0>(t);
            auto broadcast_shape = std::get<1>(t);
            auto elements_c_traverse = std::get<2>(t);
            auto elements_f_traverse = std::get<3>(t);
            auto first = ten.rbegin(broadcast_shape);
            auto last = ten.rend(broadcast_shape);
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_c_order_broadcast_iterator")
    {
        auto test = [](const auto& t){
            tensor_type ten = std::get<0>(t);
            auto broadcast_shape = std::get<1>(t);
            auto elements_c_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto first = a.begin(broadcast_shape);
            auto last = a.end(broadcast_shape);
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_broadcast_iterator")
    {
        auto test = [](const auto& t){
            tensor_type ten = std::get<0>(t);
            auto broadcast_shape = std::get<1>(t);
            auto elements_f_traverse = std::get<3>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto first = a.begin(broadcast_shape);
            auto last = a.end(broadcast_shape);
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_c_order_reverse_broadcast_iterator")
    {
        auto test = [](const auto& t){
            tensor_type ten = std::get<0>(t);
            auto broadcast_shape = std::get<1>(t);
            auto elements_c_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto first = a.rbegin(broadcast_shape);
            auto last = a.rend(broadcast_shape);
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_reverse_broadcast_iterator")
    {
        auto test = [](const auto& t){
            tensor_type ten = std::get<0>(t);
            auto broadcast_shape = std::get<1>(t);
            auto elements_f_traverse = std::get<3>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto first = a.rbegin(broadcast_shape);
            auto last = a.rend(broadcast_shape);
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
        };
        apply_by_element(test,test_data);
    }
}

TEST_CASE("test_tensor_data_interface_result_type","[test_tensor]")
{
    using value_type = int;
    using gtensor::config::c_order;
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
    using tensor_type = gtensor::tensor<int,c_order,config_type>;
    using dim_type = tensor_type::dim_type;
    using index_type = tensor_type::index_type;
    using shape_type = tensor_type::shape_type;

    //non const instance
    //tensor
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().begin()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().end()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().rbegin()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().rend()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().begin(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().end(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().rbegin(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().rend(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_type>().create_indexer()[std::declval<index_type>()]),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().create_walker(std::declval<dim_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().create_walker()),value_type&>);
    //adapter
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().traverse_order_adapter(c_order{}).begin()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().traverse_order_adapter(c_order{}).end()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().traverse_order_adapter(c_order{}).rbegin()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().traverse_order_adapter(c_order{}).rend()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().traverse_order_adapter(c_order{}).begin(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().traverse_order_adapter(c_order{}).end(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().traverse_order_adapter(c_order{}).rbegin(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_type>().traverse_order_adapter(c_order{}).rend(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_type>().traverse_order_adapter(c_order{}).create_indexer()[std::declval<index_type>()]),value_type&>);
    //const instance
    //tensor
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().begin()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().end()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().rbegin()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().rend()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().begin(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().end(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().rbegin(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().rend(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor_type>().create_indexer()[std::declval<index_type>()]),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().create_walker(std::declval<dim_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().create_walker()),const value_type&>);
    //adapter
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().traverse_order_adapter(c_order{}).begin()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().traverse_order_adapter(c_order{}).end()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().traverse_order_adapter(c_order{}).rbegin()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().traverse_order_adapter(c_order{}).rend()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().traverse_order_adapter(c_order{}).begin(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().traverse_order_adapter(c_order{}).end(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().traverse_order_adapter(c_order{}).rbegin(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_type>().traverse_order_adapter(c_order{}).rend(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor_type>().traverse_order_adapter(c_order{}).create_indexer()[std::declval<index_type>()]),const value_type&>);
}

TEMPLATE_TEST_CASE("test_tensor_is_trivial_same_layout","[test_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using order = TestType;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type,order>;
    using slice_type = typename tensor_type::slice_type;
    using helpers_for_testing::apply_by_element;

    //0tensor,1expected
    auto test_data = std::make_tuple(
        //tensor
        std::make_tuple(tensor_type(1), true),
        std::make_tuple(tensor_type{}, true),
        std::make_tuple(tensor_type{1,2,3,4,5}, true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, true),
        //view
        //reshape, trivial parent
        std::make_tuple(tensor_type(1).reshape({},order{}), true),
        std::make_tuple(tensor_type(1).reshape({1},order{}), true),
        std::make_tuple(tensor_type(1).reshape({1,1,1,1},order{}), true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}.reshape({},order{}), true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}.reshape({2,3},order{}), true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}.reshape({1,3,1,2,1},order{}), true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.reshape({6},order{}), true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.reshape({3,2},order{}), true),
        //reshape, not trivial parent
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose().reshape({3,2},order{}), false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(0).reshape({1,3,1},order{}), false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(slice_type{},slice_type{{},{},{2}}).reshape({4,1},order{}), false),
        //transpose, trivial parent
        std::make_tuple(tensor_type(1).transpose(), true),
        std::make_tuple(tensor_type{}.transpose(), true),
        std::make_tuple(tensor_type{}.transpose(0), true),
        std::make_tuple(tensor_type{1}.transpose(), true),
        std::make_tuple(tensor_type{1}.transpose(0), true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}.transpose(), true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}.transpose(0), true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(), false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(1,0), false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose(0,1), true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}.transpose(), false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}.transpose(1,0), false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}.transpose(0,1), true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(0,2,1), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(0,1,2), true),
        //transpose, not trivial parent
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}(slice_type{},slice_type{{},{},{2}}).transpose(), false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}.transpose().reshape({-1},order{}).transpose(), false),
        //slice, trivial parent
        std::make_tuple(tensor_type(1)(), true),
        std::make_tuple(tensor_type{}(slice_type{}), true),
        std::make_tuple(tensor_type{}(slice_type{0,0}), true),
        std::make_tuple(tensor_type{1}(slice_type{}), true),
        std::make_tuple(tensor_type{1}(slice_type{1}), false),
        std::make_tuple(tensor_type{1}(slice_type{{},{},2}), false),
        std::make_tuple(tensor_type{1,2,3,4,5,6}(slice_type{}), true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}(slice_type{0,{},{1}}), true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}(slice_type{{},{10},{1}}), true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}(slice_type{{-10},{10},{1}}), true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}(slice_type{{1},{-1}}), false),
        std::make_tuple(tensor_type{1,2,3,4,5,6}(slice_type{{},{},{-1}}), false),
        std::make_tuple(tensor_type{1,2,3,4,5,6}(0), false),
        std::make_tuple(tensor_type{1,2,3,4,5,6}(3), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(0), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(1), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(1,1), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(), true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(slice_type{}), true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(slice_type{},slice_type{}), true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(slice_type{},slice_type{},slice_type{}), true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(slice_type{},0), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(0,slice_type{}), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(1,slice_type{}), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(slice_type{1}), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(slice_type{},slice_type{},slice_type{{},{},2}), false),
        //slice, not trivial parent
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose()(), false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(0)(), false),
        //map, trivial parent
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}(tensor<int>{1,0,0,1}),true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.reshape({-1},order{})(tensor<int>{1,0,4,5,1,3}),true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose(0,1,2)(tensor<int>{1,0,0,1}),true),
        //map, not trivial parent
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}.transpose()(tensor<int>{1,0,0,1}),false),
        //expression view
        //trivial
        std::make_tuple(tensor_type(1)+tensor_type(2)+tensor_type(3),true),
        std::make_tuple((tensor_type(1)+2)*(tensor_type(2)+tensor_type(3)),true),
        std::make_tuple(((tensor_type(1)+2)+3)+4,true),
        std::make_tuple(tensor_type{1}+tensor_type{2}+tensor_type{3},true),
        std::make_tuple((tensor_type{1}+tensor_type(2)+tensor_type{3}+4+tensor_type(5))*6,true),
        std::make_tuple((3+tensor_type{6,5,4,3,2,1})*(tensor_type{4,5,6,1,2,3}-2),true),
        std::make_tuple((tensor_type{1,2,3,4,5,6}+tensor_type{6,5,4,3,2,1})*(tensor_type{4,5,6,1,2,3}-tensor_type{1,1,1,2,2,2}),true),
        std::make_tuple((tensor_type{{1,2,3},{4,5,6}}+tensor_type{6,5,4,3,2,1}.reshape({2,3},order{}))*(tensor_type{{4,5,6},{1,2,3}}.transpose(0,1)-tensor_type{1,1,1,2,2,2}(tensor<int>{{0,1,0},{3,2,2}})),true),
        //not trivial
        std::make_tuple(tensor_type(1)+tensor_type{1,2,3,4,5,6}, false),
        std::make_tuple(tensor_type{1}+tensor_type{1,2,3,4,5,6}, false),
        std::make_tuple(tensor_type{1,2,3}+tensor_type{{1,2,3},{4,5,6}}, false),
        std::make_tuple(tensor_type{{1},{2},{3}}+tensor_type{4,5,6}, false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{{1,2,3},{1,2,3}}+tensor_type(7), false),
        std::make_tuple((tensor_type{{1,2,3},{4,5,6}}+tensor_type{{1,2,3},{1,2,3}})*(tensor_type{{1,2,3},{4,5,6}}+tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}), false),
        std::make_tuple(tensor_type{1}+tensor_type{1,2,3}(0)+tensor_type{3},false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}+tensor_type{{1,2},{3,4},{5,6}}.transpose(), false)
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);

        auto result = ten.is_trivial();
        REQUIRE(result==expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_tensor_is_trivial_different_layouts","[test_tensor]")
{
    using value_type = double;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using helpers_for_testing::apply_by_element;

    //0tensor,1expected
    auto test_data = std::make_tuple(
        //view
        //reshape, trivial parent
        std::make_tuple(tensor<value_type,c_order>(1).reshape({},f_order{}), false),
        std::make_tuple(tensor<value_type,f_order>(1).reshape({},c_order{}), false),
        std::make_tuple(tensor<value_type,c_order>(1).reshape({1},f_order{}), false),
        std::make_tuple(tensor<value_type,f_order>(1).reshape({1},c_order{}), false),
        std::make_tuple(tensor<value_type,c_order>{{1,2,3},{4,5,6}}.reshape({6},f_order{}), false),
        std::make_tuple(tensor<value_type,f_order>{{1,2,3},{4,5,6}}.reshape({6},c_order{}), false),
        //expression view
        std::make_tuple(tensor<value_type,c_order>{1,2,3,4,5,6}+tensor<value_type,f_order>{1,2,3,4,5,6}, true),
        std::make_tuple(tensor<value_type,c_order>{1,2,3,4,5,6}+tensor<value_type,f_order>{1,2,3,4,5,6}+tensor<value_type,c_order>{1,2,3,4,5,6}, true),
        std::make_tuple(tensor<value_type,c_order>{{1,2,3},{4,5,6}}+tensor<value_type,f_order>{{1,2,3},{4,5,6}}, false)
    );

    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);

        auto result = ten.is_trivial();
        REQUIRE(result==expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_trivial_data_interface_same_layout","[test_tensor]",
    (gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
    (gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::c_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::c_order>,double>>),
    (const gtensor::tensor<double, gtensor::config::f_order, gtensor::config::extend_config_t<test_config::config_order_selector_t<gtensor::config::f_order>,double>>)
)
{
    using tensor_type = TestType;
    using slice_type = typename tensor_type::slice_type;
    using config_type = typename tensor_type::config_type;
    using traverse_order = typename config_type::order;
    using index_type = typename tensor_type::index_type;
    using value_type = typename tensor_type::value_type;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;
    //0tensor,1elements_c_traverse,2elements_f_traverse
    auto test_data = std::make_tuple(
        //tensor
        std::make_tuple(tensor_type(1),std::vector<value_type>{1},std::vector<value_type>{1}),
        std::make_tuple(tensor_type{},std::vector<value_type>{},std::vector<value_type>{}),
        std::make_tuple(tensor_type{2},std::vector<value_type>{2},std::vector<value_type>{2}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},std::vector<value_type>{1,2,3,4,5,6},std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}},std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        //view
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.transpose(0,1,2),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}(),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}(slice_type{},slice_type{}),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}(tensor<int>{0,1,0}),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6},std::vector<value_type>{1,7,1,4,10,4,2,8,2,5,11,5,3,9,3,6,12,6}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}(tensor<int>{0,1,0}).transpose(0,1,2),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6},std::vector<value_type>{1,7,1,4,10,4,2,8,2,5,11,5,3,9,3,6,12,6}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.transpose(0,1,2)(slice_type{}),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.transpose(0,1,2)(),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}(slice_type{}).transpose(0,1,2),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}().transpose(0,1,2),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        //expression view
        std::make_tuple((tensor_type(1)+tensor_type(2)+tensor_type(3))*4,std::vector<value_type>{24},std::vector<value_type>{24}),
        std::make_tuple((tensor_type{1}+tensor_type{2}+tensor_type{3})*4,std::vector<value_type>{24},std::vector<value_type>{24}),
        std::make_tuple((tensor_type{1,2,3,4,5,6} + 1)*tensor_type{2,3,4,5,6,7},std::vector<value_type>{4,9,16,25,36,49},std::vector<value_type>{4,9,16,25,36,49}),
        std::make_tuple((tensor_type{{1,2,3},{4,5,6}} + 1)*tensor_type{{2,3,4},{5,6,7}}-tensor_type{{1,1,1},{2,2,2}},std::vector<value_type>{3,8,15,23,34,47},std::vector<value_type>{3,23,8,34,15,47}),
        std::make_tuple(((tensor_type{{1,2,3},{4,5,6}} + 1)*tensor_type{{2,3,4},{5,6,7}}-tensor_type{{1,1,1},{2,2,2}}).transpose(0,1)(tensor<int>{1,1,0,0},tensor<int>{2,2,1,1}),std::vector<value_type>{47,47,8,8},std::vector<value_type>{47,47,8,8}),
        std::make_tuple((tensor_type{{1,2,3},{4,5,6}}(tensor<int>{{1},{0}},tensor<int>{2,1})+1).transpose(0,1)*tensor_type{{3,4},{6,7}}(slice_type{})-tensor_type{{1,1},{2,2}}(),std::vector<value_type>{20,23,22,19},std::vector<value_type>{20,22,23,19})
    );

    //trivial data interface
    SECTION("test_trivial_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto first = ten.begin_trivial();
            auto last = ten.end_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(ten.is_trivial());
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_trivial_reverse_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto first = ten.rbegin_trivial();
            auto last = ten.rend_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_trivial_indexer")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto indexer = ten.create_trivial_indexer();
            std::vector<value_type> traverse_result{};
            for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
                traverse_result.push_back(indexer[i]);
            }
            auto first = traverse_result.begin();
            auto last = traverse_result.end();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_trivial_walker_c_order_traverse")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto walker = ten.create_trivial_walker();
            using walker_type = decltype(walker);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,gtensor::config::c_order>;
            using walker_iterator_type = gtensor::walker_iterator<config_type,traverser_type>;
            walker_iterator_type first{walker, ten.shape(), ten.descriptor().strides_div(c_order{}), index_type{0}};
            walker_iterator_type last{walker, ten.shape(), ten.descriptor().strides_div(c_order{}), ten.size()};
            REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_trivial_walker_f_order_traverse")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto walker = ten.create_trivial_walker();
            using walker_type = decltype(walker);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,gtensor::config::f_order>;
            using walker_iterator_type = gtensor::walker_iterator<config_type,traverser_type>;
            walker_iterator_type first{walker, ten.shape(), ten.descriptor().strides_div(f_order{}), index_type{0}};
            walker_iterator_type last{walker, ten.shape(), ten.descriptor().strides_div(f_order{}), ten.size()};
            REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
        };
        apply_by_element(test,test_data);
    }

    //traverse adapter trivial data interface
    SECTION("test_traverse_adapter_c_order_trivial_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto first = a.begin_trivial();
            auto last = a.end_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_trivial_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto first = a.begin_trivial();
            auto last = a.end_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_c_order_trivial_reverse_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto first = a.rbegin_trivial();
            auto last = a.rend_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_trivial_reverse_iterator")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto first = a.rbegin_trivial();
            auto last = a.rend_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_c_order_trivial_indexer")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto indexer = a.create_trivial_indexer();
            std::vector<value_type> traverse_result{};
            for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
                traverse_result.push_back(indexer[i]);
            }
            auto first = traverse_result.begin();
            auto last = traverse_result.end();
            REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_trivial_indexer")
    {
        auto test = [](const auto& t){
            using ten_type = std::tuple_element_t<0, std::remove_cv_t<std::remove_reference_t<decltype(t)>>>;
            std::conditional_t<std::is_const_v<tensor_type>,const ten_type,ten_type> ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto indexer = a.create_trivial_indexer();
            std::vector<value_type> traverse_result{};
            for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
                traverse_result.push_back(indexer[i]);
            }
            auto first = traverse_result.begin();
            auto last = traverse_result.end();
            REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_tensor_trivial_data_accessor_different_layout","[test_tensor]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using traverse_order = TestType;
    using config_type = gtensor::config::extend_config_t<test_config::config_order_selector_t<traverse_order>,value_type>;
    using gtensor::tensor;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using c_tensor_type = tensor<value_type,c_order,config_type>;
    using f_tensor_type = tensor<value_type,f_order,config_type>;
    using slice_type = typename c_tensor_type::slice_type;
    using index_type = typename c_tensor_type::index_type;
    using helpers_for_testing::apply_by_element;
    //0tensor,1elements_c_traverse,2elements_f_traverse
    auto test_data = std::make_tuple(
        //view
        std::make_tuple(c_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.reshape({-1},c_order{}),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12}),
        std::make_tuple(f_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.reshape({-1},f_order{}),std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(c_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.reshape({4,3},c_order{}).transpose(0,1),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,4,7,10,2,5,8,11,3,6,9,12}),
        std::make_tuple(f_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.reshape({4,3},f_order{}).transpose(0,1),std::vector<value_type>{{1,2,3,7,8,9,4,5,6,10,11,12}},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(c_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.reshape({4,3},c_order{})(),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,4,7,10,2,5,8,11,3,6,9,12}),
        std::make_tuple(f_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.reshape({4,3},f_order{})(),std::vector<value_type>{{1,2,3,7,8,9,4,5,6,10,11,12}},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(c_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.reshape({4,3},c_order{})(slice_type{}),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,4,7,10,2,5,8,11,3,6,9,12}),
        std::make_tuple(f_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.reshape({4,3},f_order{})(slice_type{}),std::vector<value_type>{{1,2,3,7,8,9,4,5,6,10,11,12}},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(c_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.reshape({4,3},c_order{})(tensor<int>{{0},{1},{0}},tensor<int>{1,0,1}),std::vector<value_type>{2,1,2,5,4,5,2,1,2},std::vector<value_type>{2,5,2,1,4,1,2,5,2}),
        std::make_tuple(f_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.reshape({4,3},f_order{})(tensor<int>{{0},{1},{0}},tensor<int>{1,0,1}),std::vector<value_type>{2,1,2,8,7,8,2,1,2},std::vector<value_type>{2,8,2,1,7,1,2,8,2}),
        std::make_tuple(c_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.transpose(0,1,2).reshape({4,3},c_order{}),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,4,7,10,2,5,8,11,3,6,9,12}),
        std::make_tuple(f_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}.transpose(0,1,2).reshape({4,3},f_order{}),std::vector<value_type>{{1,2,3,7,8,9,4,5,6,10,11,12}},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(c_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}().reshape({4,3},c_order{}),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,4,7,10,2,5,8,11,3,6,9,12}),
        std::make_tuple(f_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}().reshape({4,3},f_order{}),std::vector<value_type>{{1,2,3,7,8,9,4,5,6,10,11,12}},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(c_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}(slice_type{}).reshape({4,3},c_order{}),std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12},std::vector<value_type>{1,4,7,10,2,5,8,11,3,6,9,12}),
        std::make_tuple(f_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}(slice_type{}).reshape({4,3},f_order{}),std::vector<value_type>{{1,2,3,7,8,9,4,5,6,10,11,12}},std::vector<value_type>{1,7,4,10,2,8,5,11,3,9,6,12}),
        std::make_tuple(c_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}(tensor<int>{{1},{0}},tensor<int>{0,1}).reshape({4,3},c_order{}),std::vector<value_type>{7,8,9,10,11,12,1,2,3,4,5,6},std::vector<value_type>{7,10,1,4,8,11,2,5,9,12,3,6}),
        std::make_tuple(f_tensor_type{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}}(tensor<int>{{1},{0}},tensor<int>{0,1}).reshape({4,3},f_order{}),std::vector<value_type>{7,8,9,1,2,3,10,11,12,4,5,6},std::vector<value_type>{7,1,10,4,8,2,11,5,9,3,12,6}),
        //expression view
        std::make_tuple((c_tensor_type{1,2,3,4,5,6} + 1)*f_tensor_type{2,3,4,5,6,7}-c_tensor_type{1,2,3,4,5,6},std::vector<value_type>{3,7,13,21,31,43},std::vector<value_type>{3,7,13,21,31,43})
    );
    //trivial data interface
    SECTION("test_trivial_iterator")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto first = ten.begin_trivial();
            auto last = ten.end_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(ten.is_trivial());
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_trivial_reverse_iterator")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto first = ten.rbegin_trivial();
            auto last = ten.rend_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_trivial_indexer")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto indexer = ten.create_trivial_indexer();
            std::vector<value_type> traverse_result{};
            for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
                traverse_result.push_back(indexer[i]);
            }
            auto first = traverse_result.begin();
            auto last = traverse_result.end();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            if constexpr (std::is_same_v<traverse_order,c_order>){
                REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
            }else{
                REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_trivial_walker_c_order_traverse")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto walker = ten.create_trivial_walker();
            using walker_type = decltype(walker);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,gtensor::config::c_order>;
            using walker_iterator_type = gtensor::walker_iterator<config_type,traverser_type>;
            walker_iterator_type first{walker, ten.shape(), ten.descriptor().strides_div(c_order{}), index_type{0}};
            walker_iterator_type last{walker, ten.shape(), ten.descriptor().strides_div(c_order{}), ten.size()};
            REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_trivial_walker_f_order_traverse")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto walker = ten.create_trivial_walker();
            using walker_type = decltype(walker);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,gtensor::config::f_order>;
            using walker_iterator_type = gtensor::walker_iterator<config_type,traverser_type>;
            walker_iterator_type first{walker, ten.shape(), ten.descriptor().strides_div(f_order{}), index_type{0}};
            walker_iterator_type last{walker, ten.shape(), ten.descriptor().strides_div(f_order{}), ten.size()};
            REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
        };
        apply_by_element(test,test_data);
    }

    //traverse adapter trivial data interface
    SECTION("test_traverse_adapter_c_order_trivial_iterator")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto first = a.begin_trivial();
            auto last = a.end_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_trivial_iterator")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto first = a.begin_trivial();
            auto last = a.end_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_c_order_trivial_reverse_iterator")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto first = a.rbegin_trivial();
            auto last = a.rend_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_c_traverse.rbegin(),elements_c_traverse.rend()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_trivial_reverse_iterator")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto first = a.rbegin_trivial();
            auto last = a.rend_trivial();
            REQUIRE(std::is_same_v<decltype(first),decltype(last)>);
            REQUIRE(std::equal(first,last,elements_f_traverse.rbegin(),elements_f_traverse.rend()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_c_order_trivial_indexer")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_c_traverse = std::get<1>(t);
            auto a = ten.traverse_order_adapter(c_order{});
            auto indexer = a.create_trivial_indexer();
            std::vector<value_type> traverse_result{};
            for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
                traverse_result.push_back(indexer[i]);
            }
            auto first = traverse_result.begin();
            auto last = traverse_result.end();
            REQUIRE(std::equal(first,last,elements_c_traverse.begin(),elements_c_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_traverse_adapter_f_order_trivial_indexer")
    {
        auto test = [](const auto& t){
            auto ten = std::get<0>(t);
            auto elements_f_traverse = std::get<2>(t);
            auto a = ten.traverse_order_adapter(f_order{});
            auto indexer = a.create_trivial_indexer();
            std::vector<value_type> traverse_result{};
            for(index_type i=0, i_last = ten.size(); i!=i_last; ++i){
                traverse_result.push_back(indexer[i]);
            }
            auto first = traverse_result.begin();
            auto last = traverse_result.end();
            REQUIRE(std::equal(first,last,elements_f_traverse.begin(),elements_f_traverse.end()));
        };
        apply_by_element(test,test_data);
    }
}

