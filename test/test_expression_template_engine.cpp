/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#include "catch.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"
#include "expression_template_operator.hpp"
#include "tensor.hpp"

namespace test_expression_template_engine_{

struct unary_ident_ref{
    template<typename T>
    T& operator()(T& t)const{
        return t;
    }
};
struct unary_ident_const_ref{
    template<typename T>
    const T& operator()(const T& t)const{
        return t;
    }
};
struct unary_square{
    template<typename T>
    auto operator()(const T& t)const{
        return t*t;
    }
};
struct binary_mul{
    template<typename T1, typename T2>
    auto operator()(const T1& t1, const T2& t2)const{
        return t1*t2;
    }
};
struct binary_sub{
    template<typename T1, typename T2>
    auto operator()(const T1& t1, const T2& t2)const{
        return t1-t2;
    }
};
struct ternary_add_mul{
    template<typename T1, typename T2, typename T3>
    auto operator()(const T1& t1, const T2& t2, const T3& t3)const{
        return (t1+t2)*t3;
    }
};

struct assign{
    template<typename T1, typename T2>
    void operator()(T1& t1, const T2& t2)const{
        t1 = t2;
    }
};
struct assign_add{
    template<typename T1, typename T2>
    void operator()(T1& t1, const T2& t2)const{
        t1 += t2;
    }
};

}

TEST_CASE("test_expression_template_walker","[test_expression_template_engine]")
{
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<test_config::config_engine_selector_t<gtensor::config::engine_expression_template>,value_type>;
    using dim_type = typename config_type::dim_type;
    using tensor_type = gtensor::tensor<value_type>;
    using test_expression_template_engine_::unary_square;
    using test_expression_template_engine_::binary_mul;
    using gtensor::expression_template_walker;
    using helpers_for_testing::apply_by_element;
    //0operation,1operands,2max_dim,3mover,4expected
    auto test_data = std::make_tuple(
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type(2)), dim_type{0}, [](auto&){}, value_type{4}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type(3)), dim_type{0}, [](auto& w){w.reset_back();}, value_type{9}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{2}), dim_type{1}, [](auto&){}, value_type{4}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{3}), dim_type{1}, [](auto& w){w.reset_back();}, value_type{9}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto&){}, value_type{1}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.reset(0);}, value_type{9}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.step(0);}, value_type{4}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2);}, value_type{9}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.step_back(0);}, value_type{4}),
        std::make_tuple(unary_square{}, std::make_tuple(tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.reset_back();}, value_type{1}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type(1)), dim_type{0}, [](auto&){}, value_type{2}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type(3)), dim_type{0}, [](auto&){}, value_type{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{1,2,3}), dim_type{1}, [](auto&){}, value_type{2}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.step(0);}, value_type{4}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2);}, value_type{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.step_back(0);}, value_type{4}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type(2),tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.reset_back();}, value_type{2}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto&){}, value_type{4}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.reset(0);}, value_type{18}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.step(0);}, value_type{10}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2);}, value_type{18}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.step_back(0);}, value_type{10}),
        std::make_tuple(binary_mul{}, std::make_tuple(tensor_type{4,5,6},tensor_type{1,2,3}), dim_type{1}, [](auto& w){w.walk(0,2); w.reset_back();}, value_type{4})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        using F = decltype(f);
        auto operands = std::get<1>(t);
        auto max_dim = std::get<2>(t);
        auto mover = std::get<3>(t);
        auto expected = std::get<4>(t);
        auto make_walker_helper = [](auto f_, auto...walkers_){
            return expression_template_walker<config_type,F,decltype(walkers_)...>{f_,walkers_...};
        };
        auto make_walker = [max_dim,f,make_walker_helper](auto...operands){
            return make_walker_helper(f,operands.create_walker(max_dim)...);
        };
        auto result_walker = std::apply(make_walker, operands);
        mover(result_walker);
        auto result = *result_walker;
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_expression_template_walker_result_type","[test_expression_template_engine]")
{
    using gtensor::tensor;
    using gtensor::config::extend_config_t;
    using test_expression_template_engine_::unary_ident_ref;
    using test_expression_template_engine_::unary_ident_const_ref;
    using test_expression_template_engine_::unary_square;
    using test_expression_template_engine_::binary_mul;
    using gtensor::expression_template_walker;

    using order = gtensor::config::c_order;
    using config_type_ = test_config::config_engine_selector_t<gtensor::config::engine_expression_template>;
    using config_type_int = extend_config_t<config_type_,int>;
    using config_type_double = extend_config_t<config_type_,double>;
    using tensor_int_walker_type = decltype(std::declval<tensor<int,order,config_type_int>>().create_walker());
    using tensor_double_walker_type = decltype(std::declval<tensor<double,order,config_type_double>>().create_walker());

    REQUIRE(std::is_same_v<int, decltype(*std::declval<expression_template_walker<config_type_int, unary_square, tensor_int_walker_type>>())>);
    REQUIRE(std::is_same_v<double, decltype(*std::declval<expression_template_walker<config_type_double, unary_square, tensor_double_walker_type>>())>);
    REQUIRE(std::is_same_v<int&, decltype(*std::declval<expression_template_walker<config_type_int, unary_ident_ref, tensor_int_walker_type>>())>);
    REQUIRE(std::is_same_v<const int&, decltype(*std::declval<expression_template_walker<config_type_int, unary_ident_const_ref, tensor_int_walker_type>>())>);
    REQUIRE(std::is_same_v<const double&, decltype(*std::declval<expression_template_walker<config_type_double, unary_ident_const_ref, tensor_double_walker_type>>())>);
    REQUIRE(std::is_same_v<int, decltype(*std::declval<expression_template_walker<config_type_int, binary_mul, tensor_int_walker_type, tensor_int_walker_type>>())>);
    REQUIRE(std::is_same_v<double, decltype(*std::declval<expression_template_walker<config_type_double, binary_mul, tensor_double_walker_type, tensor_int_walker_type>>())>);
    REQUIRE(std::is_same_v<double, decltype(*std::declval<expression_template_walker<config_type_double, binary_mul, tensor_int_walker_type, tensor_double_walker_type>>())>);
}

TEST_CASE("test_expression_template_core","[test_expression_template_engine]")
{
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<test_config::config_engine_selector_t<gtensor::config::engine_expression_template>,value_type>;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using c_tensor_type = gtensor::tensor<value_type,c_order,config_type>;
    using f_tensor_type = gtensor::tensor<value_type,f_order,config_type>;
    using gtensor::walker_iterator;
    using gtensor::expression_template_core;
    using test_expression_template_engine_::unary_square;
    using test_expression_template_engine_::binary_mul;
    using helpers_for_testing::apply_by_element;
    //0operation,1operands,2expected_dim,3expected_size,4expected_shape,5expected_elements
    auto test_data = std::make_tuple(
        //c_order operands
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type(2)), dim_type{0}, index_type{1}, shape_type{}, std::vector<value_type>{4}),
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type{3}), dim_type{1}, index_type{1}, shape_type{1}, std::vector<value_type>{9}),
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type{1,2,3,4,5}), dim_type{1}, index_type{5}, shape_type{5}, std::vector<value_type>{1,4,9,16,25}),
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type{{1},{2},{3},{4},{5}}), dim_type{2}, index_type{5}, shape_type{5,1}, std::vector<value_type>{1,4,9,16,25}),
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}), dim_type{3}, index_type{8}, shape_type{2,2,2}, std::vector<value_type>{1,4,9,16,25,36,49,64}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{},c_tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{2},c_tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{},c_tensor_type(1)), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type(2),c_tensor_type(3)), dim_type{0}, index_type{1}, shape_type{}, std::vector<value_type>{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type(2),c_tensor_type{3}), dim_type{1}, index_type{1}, shape_type{1}, std::vector<value_type>{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type(2),c_tensor_type{3,4,5}), dim_type{1}, index_type{3}, shape_type{3}, std::vector<value_type>{6,8,10}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type(2),c_tensor_type{{3},{4},{5}}), dim_type{2}, index_type{3}, shape_type{3,1}, std::vector<value_type>{6,8,10}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{1,2},c_tensor_type{{3},{4}}), dim_type{2}, index_type{4}, shape_type{2,2}, std::vector<value_type>{3,6,4,8}),
        //f_order operands
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type(2)), dim_type{0}, index_type{1}, shape_type{}, std::vector<value_type>{4}),
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type{3}), dim_type{1}, index_type{1}, shape_type{1}, std::vector<value_type>{9}),
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type{1,2,3,4,5}), dim_type{1}, index_type{5}, shape_type{5}, std::vector<value_type>{1,4,9,16,25}),
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type{{1},{2},{3},{4},{5}}), dim_type{2}, index_type{5}, shape_type{5,1}, std::vector<value_type>{1,4,9,16,25}),
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}), dim_type{3}, index_type{8}, shape_type{2,2,2}, std::vector<value_type>{1,4,9,16,25,36,49,64}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{},f_tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{2},f_tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{},f_tensor_type(1)), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type(2),f_tensor_type(3)), dim_type{0}, index_type{1}, shape_type{}, std::vector<value_type>{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type(2),f_tensor_type{3}), dim_type{1}, index_type{1}, shape_type{1}, std::vector<value_type>{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type(2),f_tensor_type{3,4,5}), dim_type{1}, index_type{3}, shape_type{3}, std::vector<value_type>{6,8,10}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type(2),f_tensor_type{{3},{4},{5}}), dim_type{2}, index_type{3}, shape_type{3,1}, std::vector<value_type>{6,8,10}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{1,2},f_tensor_type{{3},{4}}), dim_type{2}, index_type{4}, shape_type{2,2}, std::vector<value_type>{3,6,4,8}),
        //mixed operands
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{},c_tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{2},f_tensor_type{}), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{},f_tensor_type(1)), dim_type{1}, index_type{0}, shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type(2),c_tensor_type(3)), dim_type{0}, index_type{1}, shape_type{}, std::vector<value_type>{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type(2),c_tensor_type{3}), dim_type{1}, index_type{1}, shape_type{1}, std::vector<value_type>{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type(2),f_tensor_type{3,4,5}), dim_type{1}, index_type{3}, shape_type{3}, std::vector<value_type>{6,8,10}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type(2),c_tensor_type{{3},{4},{5}}), dim_type{2}, index_type{3}, shape_type{3,1}, std::vector<value_type>{6,8,10}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{1,2},f_tensor_type{{3},{4}}), dim_type{2}, index_type{4}, shape_type{2,2}, std::vector<value_type>{3,6,4,8})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        using F = decltype(f);
        auto operands = std::get<1>(t);
        auto expected_dim = std::get<2>(t);
        auto expected_size = std::get<3>(t);
        auto expected_shape = std::get<4>(t);
        auto expected_elements = std::get<5>(t);
        auto make_core = [f](auto&&...operands){
            return expression_template_core<config_type,F,std::remove_reference_t<decltype(operands)>...>{f,std::forward<decltype(operands)>(operands)...};
        };
        auto result_core = std::apply(make_core, operands);
        auto result_dim = result_core.descriptor().dim();
        auto result_size = result_core.descriptor().size();
        auto result_shape = result_core.descriptor().shape();
        REQUIRE(result_dim == expected_dim);
        REQUIRE(result_size == expected_size);
        REQUIRE(result_shape == expected_shape);
        using walker_type = decltype(result_core.create_walker(result_dim));
        using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,gtensor::config::c_order>;
        using iterator_type = walker_iterator<config_type,traverser_type>;
        auto result_first = iterator_type{
            result_core.create_walker(result_dim),
            result_core.descriptor().shape(),
            result_core.descriptor().strides_div(),
            index_type{0}
        };
        auto result_last = iterator_type{
            result_core.create_walker(result_dim),
            result_core.descriptor().shape(),
            result_core.descriptor().strides_div(),
            index_type{result_size}
        };
        REQUIRE(std::equal(result_first,result_last,expected_elements.begin(),expected_elements.end()));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_expression_template_core_trivial_indexer","[test_expression_template_engine]")
{
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<test_config::config_engine_selector_t<gtensor::config::engine_expression_template>,value_type>;
    using index_type = typename config_type::index_type;
    using c_tensor_type = gtensor::tensor<value_type,c_order,config_type>;
    using f_tensor_type = gtensor::tensor<value_type,f_order,config_type>;
    using gtensor::expression_template_core;
    using test_expression_template_engine_::unary_square;
    using test_expression_template_engine_::binary_mul;
    using helpers_for_testing::apply_by_element;
    //0operation,1operands,2expected_elements
    auto test_data = std::make_tuple(
        //c_order operands
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type(2)), std::vector<value_type>{4}),
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type{3}), std::vector<value_type>{9}),
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type{1,2,3,4,5}), std::vector<value_type>{1,4,9,16,25}),
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type{{{1,2,3},{3,2,1}},{{4,5,6},{6,5,4}}}), std::vector<value_type>{1,4,9,9,4,1,16,25,36,36,25,16}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type(2),c_tensor_type(3)), std::vector<value_type>{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{3},c_tensor_type{4}), std::vector<value_type>{12}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{1,2,3,4,5},c_tensor_type{5,4,3,2,1}), std::vector<value_type>{5,8,9,8,5}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{{{1,2,3},{3,2,1}},{{4,5,6},{6,5,4}}},c_tensor_type{{{1,1,1},{2,2,2}},{{3,3,3},{4,4,4}}}),
            std::vector<value_type>{1,2,3,6,4,2,12,15,18,24,20,16}
        ),
        std::make_tuple(std::minus<void>{}, std::make_tuple(c_tensor_type{{1,2,3},{4,5,6}},c_tensor_type{{1,1,1},{2,2,2}}),
            std::vector<value_type>{0,1,2,2,3,4}
        ),
        //f_order operands
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type(2)), std::vector<value_type>{4}),
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type{3}), std::vector<value_type>{9}),
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type{1,2,3,4,5}), std::vector<value_type>{1,4,9,16,25}),
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type{{{1,2,3},{3,2,1}},{{4,5,6},{6,5,4}}}), std::vector<value_type>{1,16,9,36,4,25,4,25,9,36,1,16}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type(2),f_tensor_type(3)), std::vector<value_type>{6}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{3},f_tensor_type{4}), std::vector<value_type>{12}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{1,2,3,4,5},f_tensor_type{5,4,3,2,1}), std::vector<value_type>{5,8,9,8,5}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{{{1,2,3},{3,2,1}},{{4,5,6},{6,5,4}}},f_tensor_type{{{1,1,1},{2,2,2}},{{3,3,3},{4,4,4}}}),
            std::vector<value_type>{1,12,6,24,2,15,4,20,3,18,2,16}
        ),
        std::make_tuple(std::minus<void>{}, std::make_tuple(f_tensor_type{{1,2,3},{4,5,6}},f_tensor_type{{1,1,1},{2,2,2}}),
            std::vector<value_type>{0,2,1,3,2,4}
        ),
        //mixed operands
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{1,2,3,4,5},f_tensor_type{5,4,3,2,1}), std::vector<value_type>{5,8,9,8,5}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{1,2,3,4,5},c_tensor_type{5,4,3,2,1}), std::vector<value_type>{5,8,9,8,5})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        using F = decltype(f);
        auto operands = std::get<1>(t);
        auto expected_elements = std::get<2>(t);
        auto make_core = [f](auto&&...operands){
            return expression_template_core<config_type,F,std::remove_reference_t<decltype(operands)>...>{f,std::forward<decltype(operands)>(operands)...};
        };
        auto result_core = std::apply(make_core, operands);
        using indexer_type = decltype(result_core.create_trivial_indexer());
        using iterator_type = gtensor::indexer_iterator<config_type,indexer_type>;
        auto result_first = iterator_type{result_core.create_trivial_indexer(),index_type{0}};
        auto result_last = iterator_type{result_core.create_trivial_indexer(),result_core.descriptor().size()};
        REQUIRE(std::equal(result_first,result_last,expected_elements.begin(),expected_elements.end()));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_expression_template_core_order","[test_expression_template_engine]")
{
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<test_config::config_engine_selector_t<gtensor::config::engine_expression_template>,value_type>;
    using c_tensor_type = gtensor::tensor<value_type,c_order,config_type>;
    using f_tensor_type = gtensor::tensor<value_type,f_order,config_type>;
    using gtensor::expression_template_core;
    using test_expression_template_engine_::unary_square;
    using test_expression_template_engine_::binary_mul;
    using helpers_for_testing::apply_by_element;
    //0operation,1operands,2expected_order
    auto test_data = std::make_tuple(
        //c_order operands
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type(2)), c_order{}),
        std::make_tuple(unary_square{}, std::make_tuple(c_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}), c_order{}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{},c_tensor_type{}), c_order{}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{1,2},c_tensor_type{{3},{4}}), c_order{}),
        //f_order operands
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type(2)), f_order{}),
        std::make_tuple(unary_square{}, std::make_tuple(f_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}), f_order{}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{},f_tensor_type{}), f_order{}),
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{1,2},f_tensor_type{{3},{4}}), f_order{}),
        //mixed operands
        std::make_tuple(binary_mul{}, std::make_tuple(f_tensor_type{},c_tensor_type{}), typename config_type::order{}),
        std::make_tuple(binary_mul{}, std::make_tuple(c_tensor_type{1,2},f_tensor_type{{3},{4}}), typename config_type::order{})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        using F = decltype(f);
        auto operands = std::get<1>(t);
        auto expected_order = std::get<2>(t);
        using expected_order_type = decltype(expected_order);
        auto make_core = [f](auto&&...operands){
            return expression_template_core<config_type,F,std::remove_reference_t<decltype(operands)>...>{f,std::forward<decltype(operands)>(operands)...};
        };
        auto result_core = std::apply(make_core, operands);
        using result_order_type = typename decltype(result_core)::order;
        REQUIRE(std::is_same_v<result_order_type,expected_order_type>);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_expression_template_operator_n_operator","[test_expression_template_engine]")
{
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<test_config::config_engine_selector_t<gtensor::config::engine_expression_template>,value_type>;
    using tensor_type = gtensor::tensor<value_type>;
    using c_tensor_type = gtensor::tensor<value_type,c_order,config_type>;
    using f_tensor_type = gtensor::tensor<value_type,f_order,config_type>;
    using gtensor::expression_template_operator;
    using test_expression_template_engine_::unary_square;
    using test_expression_template_engine_::binary_mul;
    using test_expression_template_engine_::binary_sub;
    using test_expression_template_engine_::ternary_add_mul;
    using helpers_for_testing::apply_by_element;
    //0operation,1operands,2expected
    auto test_data = std::make_tuple(
        //c_order operands
        std::make_tuple(unary_square{},std::make_tuple(c_tensor_type{}),tensor_type{}),
        std::make_tuple(unary_square{},std::make_tuple(c_tensor_type(2)),tensor_type(4)),
        std::make_tuple(unary_square{},std::make_tuple(c_tensor_type{1,2,3,4,5}),tensor_type{1,4,9,16,25}),
        std::make_tuple(binary_mul{},std::make_tuple(c_tensor_type(5),2),tensor_type(10)),
        std::make_tuple(binary_mul{},std::make_tuple(3,c_tensor_type(5)),tensor_type(15)),
        std::make_tuple(binary_mul{},std::make_tuple(c_tensor_type(5),c_tensor_type(4)),tensor_type(20)),
        std::make_tuple(binary_mul{},std::make_tuple(c_tensor_type{1,2,3,4,5},2),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(3,c_tensor_type{1,2,3,4,5}),tensor_type{3,6,9,12,15}),
        std::make_tuple(binary_sub{},std::make_tuple(3,c_tensor_type{1,2,3,4,5}),tensor_type{2,1,0,-1,-2}),
        std::make_tuple(binary_sub{},std::make_tuple(c_tensor_type{1,2,3,4,5},3),tensor_type{-2,-1,0,1,2}),
        std::make_tuple(binary_mul{},std::make_tuple(c_tensor_type{1,2,3,4,5},c_tensor_type(2)),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(c_tensor_type{1,2,3,4,5},c_tensor_type{2}),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(c_tensor_type{1,2,3,4,5},c_tensor_type{5,4,3,2,1}),tensor_type{5,8,9,8,5}),
        std::make_tuple(binary_mul{},std::make_tuple(4,c_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),tensor_type{{{4,8},{12,16}},{{20,24},{28,32}}}),
        std::make_tuple(binary_mul{},std::make_tuple(c_tensor_type(4), c_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),tensor_type{{{4,8},{12,16}},{{20,24},{28,32}}}),
        std::make_tuple(binary_mul{},std::make_tuple(c_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},c_tensor_type{-1,2}),tensor_type{{{-1,4},{-3,8}},{{-5,12},{-7,16}}}),
        std::make_tuple(binary_mul{},std::make_tuple(c_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},c_tensor_type{{-1},{2}}),tensor_type{{{-1,-2},{6,8}},{{-5,-6},{14,16}}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(c_tensor_type(1),c_tensor_type(2),c_tensor_type(3)),tensor_type(9)),
        std::make_tuple(ternary_add_mul{},std::make_tuple(c_tensor_type(1),c_tensor_type{1,2,3,4,5},c_tensor_type(3)),tensor_type{6,9,12,15,18}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(c_tensor_type(1),c_tensor_type{1,2,3},c_tensor_type{{1},{2},{3}}),tensor_type{{2,3,4},{4,6,8},{6,9,12}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(-1,c_tensor_type{1,2,3},c_tensor_type{{1},{2},{3}}),tensor_type{{0,1,2},{0,2,4},{0,3,6}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(c_tensor_type{{4,5,6},{7,8,9}},c_tensor_type{1,2,3},2),tensor_type{{10,14,18},{16,20,24}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(-1,c_tensor_type{{1,2,3},{4,5,6}},2),tensor_type{{0,2,4},{6,8,10}}),
        //f_order operands
        std::make_tuple(unary_square{},std::make_tuple(f_tensor_type{}),tensor_type{}),
        std::make_tuple(unary_square{},std::make_tuple(f_tensor_type(2)),tensor_type(4)),
        std::make_tuple(unary_square{},std::make_tuple(f_tensor_type{1,2,3,4,5}),tensor_type{1,4,9,16,25}),
        std::make_tuple(binary_mul{},std::make_tuple(f_tensor_type(5),2),tensor_type(10)),
        std::make_tuple(binary_mul{},std::make_tuple(3,f_tensor_type(5)),tensor_type(15)),
        std::make_tuple(binary_mul{},std::make_tuple(f_tensor_type(5),f_tensor_type(4)),tensor_type(20)),
        std::make_tuple(binary_mul{},std::make_tuple(f_tensor_type{1,2,3,4,5},2),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(3,f_tensor_type{1,2,3,4,5}),tensor_type{3,6,9,12,15}),
        std::make_tuple(binary_sub{},std::make_tuple(3,f_tensor_type{1,2,3,4,5}),tensor_type{2,1,0,-1,-2}),
        std::make_tuple(binary_sub{},std::make_tuple(f_tensor_type{1,2,3,4,5},3),tensor_type{-2,-1,0,1,2}),
        std::make_tuple(binary_mul{},std::make_tuple(f_tensor_type{1,2,3,4,5},f_tensor_type(2)),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(f_tensor_type{1,2,3,4,5},f_tensor_type{2}),tensor_type{2,4,6,8,10}),
        std::make_tuple(binary_mul{},std::make_tuple(f_tensor_type{1,2,3,4,5},f_tensor_type{5,4,3,2,1}),tensor_type{5,8,9,8,5}),
        std::make_tuple(binary_mul{},std::make_tuple(4,f_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),tensor_type{{{4,8},{12,16}},{{20,24},{28,32}}}),
        std::make_tuple(binary_mul{},std::make_tuple(f_tensor_type(4), f_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}),tensor_type{{{4,8},{12,16}},{{20,24},{28,32}}}),
        std::make_tuple(binary_mul{},std::make_tuple(f_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},f_tensor_type{-1,2}),tensor_type{{{-1,4},{-3,8}},{{-5,12},{-7,16}}}),
        std::make_tuple(binary_mul{},std::make_tuple(f_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},f_tensor_type{{-1},{2}}),tensor_type{{{-1,-2},{6,8}},{{-5,-6},{14,16}}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(f_tensor_type(1),f_tensor_type(2),f_tensor_type(3)),tensor_type(9)),
        std::make_tuple(ternary_add_mul{},std::make_tuple(f_tensor_type(1),f_tensor_type{1,2,3,4,5},f_tensor_type(3)),tensor_type{6,9,12,15,18}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(f_tensor_type(1),f_tensor_type{1,2,3},f_tensor_type{{1},{2},{3}}),tensor_type{{2,3,4},{4,6,8},{6,9,12}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(-1,f_tensor_type{1,2,3},f_tensor_type{{1},{2},{3}}),tensor_type{{0,1,2},{0,2,4},{0,3,6}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(f_tensor_type{{4,5,6},{7,8,9}},f_tensor_type{1,2,3},2),tensor_type{{10,14,18},{16,20,24}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(-1,f_tensor_type{{1,2,3},{4,5,6}},2),tensor_type{{0,2,4},{6,8,10}}),
        //mixed operands
        std::make_tuple(binary_mul{},std::make_tuple(c_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},f_tensor_type{-1,2}),tensor_type{{{-1,4},{-3,8}},{{-5,12},{-7,16}}}),
        std::make_tuple(binary_mul{},std::make_tuple(f_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},c_tensor_type{{-1},{2}}),tensor_type{{{-1,-2},{6,8}},{{-5,-6},{14,16}}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(f_tensor_type(1),c_tensor_type(2),f_tensor_type(3)),tensor_type(9)),
        std::make_tuple(ternary_add_mul{},std::make_tuple(f_tensor_type(1),f_tensor_type{1,2,3,4,5},c_tensor_type(3)),tensor_type{6,9,12,15,18}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(f_tensor_type(1),c_tensor_type{1,2,3},f_tensor_type{{1},{2},{3}}),tensor_type{{2,3,4},{4,6,8},{6,9,12}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(-1,f_tensor_type{1,2,3},c_tensor_type{{1},{2},{3}}),tensor_type{{0,1,2},{0,2,4},{0,3,6}}),
        std::make_tuple(ternary_add_mul{},std::make_tuple(f_tensor_type{{4,5,6},{7,8,9}},c_tensor_type{1,2,3},2),tensor_type{{10,14,18},{16,20,24}})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        using F = decltype(f);
        auto operands = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto apply_n_operator = [f](auto&&...operands){
            return expression_template_operator<F>::n_operator(f,operands...);
        };
        auto result = std::apply(apply_n_operator, operands);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_expression_template_operator_a_operator","[test_expression_template_engine]",
    //lhs order, rhs order
    (std::tuple<gtensor::config::c_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::c_order,gtensor::config::f_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::c_order>),
    (std::tuple<gtensor::config::f_order,gtensor::config::f_order>)
)
{
    using lhs_order = std::tuple_element_t<0,TestType>;
    using rhs_order = std::tuple_element_t<1,TestType>;
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<test_config::config_engine_selector_t<gtensor::config::engine_expression_template>,value_type>;
    using tensor_type = gtensor::tensor<value_type>;
    using lhs_tensor_type = gtensor::tensor<value_type,lhs_order,config_type>;
    using rhs_tensor_type = gtensor::tensor<value_type,rhs_order,config_type>;
    using gtensor::expression_template_operator;
    using test_expression_template_engine_::assign;
    using test_expression_template_engine_::assign_add;
    using helpers_for_testing::apply_by_element;
    //0operation,1lhs,2rhs,3expected
    auto test_data = std::make_tuple(
        //rhs scalar
        std::make_tuple(assign{},lhs_tensor_type{},2,tensor_type{}),
        std::make_tuple(assign{},lhs_tensor_type(2),1,tensor_type(1)),
        std::make_tuple(assign{},lhs_tensor_type{1,2,3,4,5},3,tensor_type{3,3,3,3,3}),
        std::make_tuple(assign{},lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},4,tensor_type{{{4,4},{4,4}},{{4,4},{4,4}}}),
        std::make_tuple(assign_add{},lhs_tensor_type{},2,tensor_type{}),
        std::make_tuple(assign_add{},lhs_tensor_type(2),1,tensor_type(3)),
        std::make_tuple(assign_add{},lhs_tensor_type{1,2,3,4,5},3,tensor_type{4,5,6,7,8}),
        std::make_tuple(assign_add{},lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},4,tensor_type{{{5,6},{7,8}},{{9,10},{11,12}}}),
        //rhs tensor
        std::make_tuple(assign{},lhs_tensor_type{},rhs_tensor_type{},tensor_type{}),
        std::make_tuple(assign{},lhs_tensor_type{},rhs_tensor_type(1),tensor_type{}),
        std::make_tuple(assign{},lhs_tensor_type{},rhs_tensor_type{1},tensor_type{}),
        std::make_tuple(assign{},lhs_tensor_type(1),rhs_tensor_type{},tensor_type(1)),
        std::make_tuple(assign{},lhs_tensor_type(1),rhs_tensor_type(2),tensor_type(2)),
        std::make_tuple(assign{},lhs_tensor_type(2),rhs_tensor_type{3},tensor_type(3)),
        std::make_tuple(assign{},lhs_tensor_type{1,2,3,4,5},rhs_tensor_type{6},tensor_type{6,6,6,6,6}),
        std::make_tuple(assign{},lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},rhs_tensor_type{{-1},{1}},tensor_type{{{-1,-1},{1,1}},{{-1,-1},{1,1}}}),
        std::make_tuple(assign_add{},lhs_tensor_type{},rhs_tensor_type{},tensor_type{}),
        std::make_tuple(assign_add{},lhs_tensor_type{},rhs_tensor_type(1),tensor_type{}),
        std::make_tuple(assign_add{},lhs_tensor_type{},rhs_tensor_type{1},tensor_type{}),
        std::make_tuple(assign_add{},lhs_tensor_type(1),rhs_tensor_type{},tensor_type(1)),
        std::make_tuple(assign_add{},lhs_tensor_type(1),rhs_tensor_type(2),tensor_type(3)),
        std::make_tuple(assign_add{},lhs_tensor_type(2),rhs_tensor_type{3},tensor_type(5)),
        std::make_tuple(assign_add{},lhs_tensor_type{1,2,3,4,5},rhs_tensor_type{6},tensor_type{7,8,9,10,11}),
        std::make_tuple(assign_add{},lhs_tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}},rhs_tensor_type{{-1},{1}},tensor_type{{{0,1},{4,5}},{{4,5},{8,9}}}),
        //assign multiple times to lhs
        std::make_tuple(assign{},lhs_tensor_type(3),rhs_tensor_type{1,2,3,4,5},tensor_type(5)),
        std::make_tuple(assign{},lhs_tensor_type{0},rhs_tensor_type{1,2,3,4,5},tensor_type{5}),
        std::make_tuple(assign{},lhs_tensor_type{0,0},rhs_tensor_type{{1,2},{3,4},{5,6}},tensor_type{5,6}),
        std::make_tuple(assign_add{},lhs_tensor_type(3),rhs_tensor_type{1,2,3,4,5},tensor_type(18)),
        std::make_tuple(assign_add{},lhs_tensor_type{0},rhs_tensor_type{1,2,3,4,5},tensor_type{15}),
        std::make_tuple(assign_add{},lhs_tensor_type{-1,1},rhs_tensor_type{{1,2},{3,4},{5,6}},tensor_type{8,13})
    );
    auto test = [](const auto& t){
        auto f = std::get<0>(t);
        using F = decltype(f);
        auto lhs = std::get<1>(t);
        auto rhs = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto& result = expression_template_operator<F>::a_operator(f,lhs,rhs);
        REQUIRE(&result == &lhs);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

