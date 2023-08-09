#include <vector>
#include <string>
#include <list>
#include "catch.hpp"
#include "tensor_implementation.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"


namespace test_tensor_implementation_{

template<typename Config, typename T, typename Order>
class test_core_base{
protected:
    using extended_config_type = gtensor::config::extend_config_t<Config,T>;
    using descriptor_type = gtensor::basic_descriptor<extended_config_type, Order>;
    using storage_type = typename extended_config_type::template storage<T>;
public:
    using order = Order;
    using config_type = extended_config_type;
    using value_type = T;
    using shape_type = typename config_type::shape_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    template<typename It>
    test_core_base(const shape_type& shape__, It first, It last):
        descriptor_{shape__},
        elements_{first,last}
    {}
    const auto& descriptor()const{return descriptor_;}
protected:
    descriptor_type descriptor_;
    storage_type elements_;
};

template<typename Config, typename T, typename Order>
class test_core_subscriptable : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    decltype(auto) operator[](index_type i){return elements_[i];}
};
template<typename Config, typename T, typename Order>
class test_core_indexible : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    auto create_indexer(){return gtensor::basic_indexer<storage_type&>{elements_};}
};
template<typename Config, typename T, typename Order>
class test_core_walkable : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    using test_core_base_type::descriptor;
    auto create_walker(dim_type max_dim){
        using indexer_type = gtensor::basic_indexer<storage_type&>;
        return gtensor::axes_correction_walker<gtensor::indexer_walker<config_type,indexer_type>>{
            max_dim,
            descriptor().adapted_strides(),
            descriptor().reset_strides(),
            index_type{0},
            indexer_type{elements_}
        };
    }
    auto create_walker(){
        using indexer_type = gtensor::basic_indexer<storage_type&>;
        return gtensor::indexer_walker<config_type,indexer_type>{
            descriptor().adapted_strides(),
            descriptor().reset_strides(),
            index_type{0},
            indexer_type{elements_}
        };
    }
};
template<typename Config, typename T, typename Order>
class test_core_iterable : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    auto begin(){return elements_.begin();}
    auto end(){return elements_.end();}
};

template<typename Config, typename T, typename Order>
class test_core_const_subscriptable : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    decltype(auto) operator[](index_type i)const{return elements_[i];}
};
template<typename Config, typename T, typename Order>
class test_core_const_indexible : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    auto create_indexer()const{return gtensor::basic_indexer<const storage_type&>{elements_};}
};
template<typename Config, typename T, typename Order>
class test_core_const_walkable : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    using test_core_base_type::descriptor;
    auto create_walker(dim_type max_dim)const{
        using indexer_type = gtensor::basic_indexer<const storage_type&>;
        return gtensor::axes_correction_walker<gtensor::indexer_walker<config_type,indexer_type>>{
            max_dim,
            descriptor().adapted_strides(),
            descriptor().reset_strides(),
            index_type{0},
            indexer_type{elements_}
        };
    }
    auto create_walker()const{
        using indexer_type = gtensor::basic_indexer<const storage_type&>;
        return gtensor::indexer_walker<config_type,indexer_type>{
            descriptor().adapted_strides(),
            descriptor().reset_strides(),
            index_type{0},
            indexer_type{elements_}
        };
    }
};
template<typename Config, typename T, typename Order>
class test_core_const_iterable : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    auto begin()const{return elements_.begin();}
    auto end()const{return elements_.end();}
};

template<typename Config, typename T, typename Order>
class test_core_full_subscriptable : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    decltype(auto) operator[](index_type i)const{return elements_[i];}
    decltype(auto) operator[](index_type i){return elements_[i];}
};
template<typename Config, typename T, typename Order>
class test_core_full_indexible : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    auto create_indexer()const{return gtensor::basic_indexer<const storage_type&>{elements_};}
    auto create_indexer(){return gtensor::basic_indexer<storage_type&>{elements_};}
};
template<typename Config, typename T, typename Order>
class test_core_full_walkable : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    using test_core_base_type::descriptor;
    auto create_walker(dim_type max_dim)const{return create_walker_helper(*this,max_dim);}
    auto create_walker(dim_type max_dim){return create_walker_helper(*this,max_dim);}
    auto create_walker()const{return create_walker_helper(*this);}
    auto create_walker(){return create_walker_helper(*this);}
private:
    template<typename U>
    static auto create_walker_helper(U& instance, dim_type max_dim){
        using indexer_type = std::conditional_t<std::is_const_v<U>, gtensor::basic_indexer<const storage_type&>, gtensor::basic_indexer<storage_type&>>;
        return gtensor::axes_correction_walker<gtensor::indexer_walker<config_type,indexer_type>>{
            max_dim,
            instance.descriptor().adapted_strides(),
            instance.descriptor().reset_strides(),
            index_type{0},
            indexer_type{instance.elements_}
        };
    }
    template<typename U>
    static auto create_walker_helper(U& instance){
        using indexer_type = std::conditional_t<std::is_const_v<U>, gtensor::basic_indexer<const storage_type&>, gtensor::basic_indexer<storage_type&>>;
        return gtensor::indexer_walker<config_type,indexer_type>{
            instance.descriptor().adapted_strides(),
            instance.descriptor().reset_strides(),
            index_type{0},
            indexer_type{instance.elements_}
        };
    }
};
template<typename Config, typename T, typename Order>
class test_core_full_iterable : public test_core_base<Config,T,Order>{
    using test_core_base_type = test_core_base<Config,T,Order>;
public:
    using typename test_core_base_type::order;
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::elements_;
    auto begin()const{return elements_.begin();}
    auto end()const{return elements_.end();}
    auto begin(){return elements_.begin();}
    auto end(){return elements_.end();}
};

template<template<typename...> typename Core>
struct test_core_template{
    template<typename Config, typename T, typename Order> using core = Core<Config,T,Order>;
};

}   //end of namespace test_tensor_implementation_

TEMPLATE_TEST_CASE("test_tensor_implementation","[test_tensor_implementation]",
    //non const accessible core
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_subscriptable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_indexible>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_walkable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_iterable>),
    //const accessible core
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_const_subscriptable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_const_indexible>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_const_walkable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_const_iterable>),
    //full accessible core
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_full_subscriptable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_full_indexible>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_full_walkable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_full_iterable>)
)
{
    using value_type = int;
    using config_type = typename gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,int>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::tensor_implementation;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;

    //0core_order,1traverse_order,2shape,3elements,4expected
    auto test_data = std::make_tuple(
        //core c_order, traverse c_order
        std::make_tuple(c_order{}, c_order{}, shape_type{}, std::vector<value_type>{1}, std::vector<value_type>{1}),
        std::make_tuple(c_order{}, c_order{}, shape_type{}, std::vector<value_type>{2}, std::vector<value_type>{2}),
        std::make_tuple(c_order{}, c_order{}, shape_type{0}, std::vector<value_type>{}, std::vector<value_type>{}),
        std::make_tuple(c_order{}, c_order{}, shape_type{5}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(c_order{}, c_order{}, shape_type{1,5}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(c_order{}, c_order{}, shape_type{5,1}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(c_order{}, c_order{}, shape_type{1,2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(c_order{}, c_order{}, shape_type{2,3,2,1}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12}),
        std::make_tuple(
            c_order{},
            c_order{},
            shape_type{2,3,4},
            std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24},
            std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}
        ),
        //core c_order, traverse f_order
        std::make_tuple(c_order{}, f_order{}, shape_type{}, std::vector<value_type>{1}, std::vector<value_type>{1}),
        std::make_tuple(c_order{}, f_order{}, shape_type{}, std::vector<value_type>{2}, std::vector<value_type>{2}),
        std::make_tuple(c_order{}, f_order{}, shape_type{0}, std::vector<value_type>{}, std::vector<value_type>{}),
        std::make_tuple(c_order{}, f_order{}, shape_type{5}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(c_order{}, f_order{}, shape_type{1,5}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(c_order{}, f_order{}, shape_type{5,1}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(c_order{}, f_order{}, shape_type{1,2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::vector<value_type>{1,4,2,5,3,6}),
        std::make_tuple(c_order{}, f_order{}, shape_type{2,3,2,1}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12}, std::vector<value_type>{1,7,3,9,5,11,2,8,4,10,6,12}),
        std::make_tuple(
            c_order{},
            f_order{},
            shape_type{2,3,4},
            std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24},
            std::vector<value_type>{1,13,5,17,9,21,2,14,6,18,10,22,3,15,7,19,11,23,4,16,8,20,12,24}
        ),
        //core f_order, traverse c_order
        std::make_tuple(f_order{}, c_order{}, shape_type{}, std::vector<value_type>{1}, std::vector<value_type>{1}),
        std::make_tuple(f_order{}, c_order{}, shape_type{}, std::vector<value_type>{2}, std::vector<value_type>{2}),
        std::make_tuple(f_order{}, c_order{}, shape_type{0}, std::vector<value_type>{}, std::vector<value_type>{}),
        std::make_tuple(f_order{}, c_order{}, shape_type{5}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(f_order{}, c_order{}, shape_type{1,5}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(f_order{}, c_order{}, shape_type{5,1}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(f_order{}, c_order{}, shape_type{1,2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::vector<value_type>{1,3,5,2,4,6}),
        std::make_tuple(f_order{}, c_order{}, shape_type{2,3,2,1}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12}, std::vector<value_type>{1,7,3,9,5,11,2,8,4,10,6,12}),
        std::make_tuple(
            f_order{},
            c_order{},
            shape_type{2,3,4},
            std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24},
            std::vector<value_type>{1,7,13,19,3,9,15,21,5,11,17,23,2,8,14,20,4,10,16,22,6,12,18,24}
        ),
        //core f_order, traverse f_order
        std::make_tuple(f_order{}, f_order{}, shape_type{}, std::vector<value_type>{1}, std::vector<value_type>{1}),
        std::make_tuple(f_order{}, f_order{}, shape_type{}, std::vector<value_type>{2}, std::vector<value_type>{2}),
        std::make_tuple(f_order{}, f_order{}, shape_type{0}, std::vector<value_type>{}, std::vector<value_type>{}),
        std::make_tuple(f_order{}, f_order{}, shape_type{5}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(f_order{}, f_order{}, shape_type{1,5}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(f_order{}, f_order{}, shape_type{5,1}, std::vector<value_type>{1,2,3,4,5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(f_order{}, f_order{}, shape_type{1,2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(f_order{}, f_order{}, shape_type{2,3,2,1}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12}),
        std::make_tuple(
            f_order{},
            f_order{},
            shape_type{2,3,4},
            std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24},
            std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}
        )
    );

    SECTION("test_iterator")
    {
        auto test_equal = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
            REQUIRE(std::equal(res_first,res_last,expected_first,expected_last));
        };
        auto test_backward_traverse = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
            std::vector<value_type> result;
            while(res_last!=res_first){
                result.push_back(*--res_last);
            }
            REQUIRE(std::equal(result.rbegin(),result.rend(),expected_first,expected_last));
        };
        auto test_subscript = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
            std::vector<value_type> result;
            for(index_type i{0}, i_last = res_last-res_first; i!=i_last; ++i){
                result.push_back(res_first[i]);
            }
            REQUIRE(std::equal(result.begin(),result.end(),expected_first,expected_last));
        };

        auto test = [test_equal,test_backward_traverse,test_subscript](const auto& t){
            auto core_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto shape = std::get<2>(t);
            auto expected_shape = shape;
            auto elements = std::get<3>(t);
            auto expected = std::get<4>(t);
            using core_order_type = decltype(core_order);
            using tensor_implementation_type = tensor_implementation<typename TestType::template core<config_type,value_type,core_order_type>>;
            tensor_implementation_type tensor_implementation{shape, elements.begin(), elements.end()};
            auto result_shape = tensor_implementation.shape();
            REQUIRE(result_shape == expected_shape);
            using traverse_order_type = decltype(traverse_order);
            auto first = tensor_implementation.template begin<traverse_order_type>();
            auto last = tensor_implementation.template end<traverse_order_type>();
            auto rfirst = tensor_implementation.template rbegin<traverse_order_type>();
            auto rlast = tensor_implementation.template rend<traverse_order_type>();

            test_equal(first,last,expected.begin(),expected.end());
            test_equal(rfirst,rlast,expected.rbegin(),expected.rend());
            test_backward_traverse(first,last,expected.begin(),expected.end());
            test_backward_traverse(rfirst,rlast,expected.rbegin(),expected.rend());
            test_subscript(first,last,expected.begin(),expected.end());
            test_subscript(rfirst,rlast,expected.rbegin(),expected.rend());
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_indexer")
    {
        auto test = [](const auto& t){
            auto core_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto shape = std::get<2>(t);
            auto expected_shape = shape;
            auto elements = std::get<3>(t);
            auto expected = std::get<4>(t);
            using core_order_type = decltype(core_order);
            using tensor_implementation_type = tensor_implementation<typename TestType::template core<config_type,value_type,core_order_type>>;
            tensor_implementation_type tensor_implementation{shape, elements.begin(), elements.end()};
            auto result_shape = tensor_implementation.shape();
            REQUIRE(result_shape == expected_shape);
            using traverse_order_type = decltype(traverse_order);
            auto indexer = tensor_implementation.template create_indexer<traverse_order_type>();
            std::vector<value_type> result;
            for (index_type i{0}; i!=tensor_implementation.size(); ++i){
                result.push_back(indexer[i]);
            }
            REQUIRE(std::equal(result.begin(),result.end(),expected.begin(),expected.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_walker")
    {
        auto test = [](const auto& t){
            auto core_order = std::get<0>(t);
            auto traverse_order = std::get<1>(t);
            auto shape = std::get<2>(t);
            auto expected_shape = shape;
            auto elements = std::get<3>(t);
            auto expected = std::get<4>(t);
            using core_order_type = decltype(core_order);
            using tensor_implementation_type = tensor_implementation<typename TestType::template core<config_type,value_type,core_order_type>>;
            tensor_implementation_type tensor_implementation{shape, elements.begin(), elements.end()};
            auto result_shape = tensor_implementation.shape();
            REQUIRE(result_shape == expected_shape);
            using walker_type = decltype(tensor_implementation.create_walker());
            using traverse_order_type = decltype(traverse_order);
            using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,traverse_order_type>;
            using walker_iterator_type = gtensor::walker_iterator<config_type,traverser_type>;
            walker_iterator_type first{
                tensor_implementation.create_walker(),
                tensor_implementation.shape(),
                tensor_implementation.descriptor().strides_div(traverse_order),
                index_type{0}
            };
            walker_iterator_type last{
                tensor_implementation.create_walker(),
                tensor_implementation.shape(),
                tensor_implementation.descriptor().strides_div(traverse_order),
                tensor_implementation.size()
            };
            REQUIRE(std::equal(first,last,expected.begin(),expected.end()));
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("test_tensor_implementation_broadcast_iterator","[test_tensor_implementation]",
    //non const accessible core
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_subscriptable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_indexible>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_walkable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_iterable>),
    //const accessible core
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_const_subscriptable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_const_indexible>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_const_walkable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_const_iterable>),
    //full accessible core
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_full_subscriptable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_full_indexible>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_full_walkable>),
    (test_tensor_implementation_::test_core_template<test_tensor_implementation_::test_core_full_iterable>)
)
{
    using value_type = int;
    using config_type = typename gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,int>;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::tensor_implementation;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using helpers_for_testing::apply_by_element;

    auto elements_c = std::vector<value_type>{1,2,3,4,5,6};
    auto elements_f = std::vector<value_type>{1,4,2,5,3,6};

    //0core_order,1traverse_order,2elements,3shape,4broadcast_shape,5expected
    auto test_data = std::make_tuple(
        //elements in c_order
        //traverse c_order
        std::make_tuple(c_order{}, c_order{}, elements_c, shape_type{6}, shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(c_order{}, c_order{}, elements_c, shape_type{6}, shape_type{1,2,6}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
        std::make_tuple(c_order{}, c_order{}, elements_c, shape_type{2,3}, shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(c_order{}, c_order{}, elements_c, shape_type{2,3}, shape_type{2,2,3}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
        //traverse f_order
        std::make_tuple(c_order{}, f_order{}, elements_c, shape_type{6}, shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(c_order{}, f_order{}, elements_c, shape_type{6}, shape_type{1,2,6}, std::vector<value_type>{1,1,2,2,3,3,4,4,5,5,6,6}),
        std::make_tuple(c_order{}, f_order{}, elements_c, shape_type{2,3}, shape_type{2,3}, std::vector<value_type>{1,4,2,5,3,6}),
        std::make_tuple(c_order{}, f_order{}, elements_c, shape_type{2,3}, shape_type{2,2,3}, std::vector<value_type>{1,1,4,4,2,2,5,5,3,3,6,6}),
        //elements in f_order
        //traverse c_order
        std::make_tuple(f_order{}, c_order{}, elements_f, shape_type{6}, shape_type{6}, std::vector<value_type>{1,4,2,5,3,6}),
        std::make_tuple(f_order{}, c_order{}, elements_f, shape_type{6}, shape_type{1,2,6}, std::vector<value_type>{1,4,2,5,3,6,1,4,2,5,3,6}),
        std::make_tuple(f_order{}, c_order{}, elements_f, shape_type{2,3}, shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(f_order{}, c_order{}, elements_f, shape_type{2,3}, shape_type{2,2,3}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
        // //traverse f_order
        std::make_tuple(f_order{}, f_order{}, elements_f, shape_type{6}, shape_type{6}, std::vector<value_type>{1,4,2,5,3,6}),
        std::make_tuple(f_order{}, f_order{}, elements_f, shape_type{6}, shape_type{1,2,6}, std::vector<value_type>{1,1,4,4,2,2,5,5,3,3,6,6}),
        std::make_tuple(f_order{}, f_order{}, elements_f, shape_type{2,3}, shape_type{2,3}, std::vector<value_type>{1,4,2,5,3,6}),
        std::make_tuple(f_order{}, f_order{}, elements_f, shape_type{2,3}, shape_type{2,2,3}, std::vector<value_type>{1,1,4,4,2,2,5,5,3,3,6,6})
    );

    auto test_equal = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
        REQUIRE(std::equal(res_first,res_last,expected_first,expected_last));
    };
    auto test_backward_traverse = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
        std::vector<value_type> result;
        while(res_last!=res_first){
            result.push_back(*--res_last);
        }
        REQUIRE(std::equal(result.rbegin(),result.rend(),expected_first,expected_last));
    };
    auto test_subscript = [](auto res_first, auto res_last, auto expected_first, auto expected_last){
        std::vector<value_type> result;
        for(index_type i{0}, i_last = res_last-res_first; i!=i_last; ++i){
            result.push_back(res_first[i]);
        }
        REQUIRE(std::equal(result.begin(),result.end(),expected_first,expected_last));
    };

    auto test = [test_equal,test_backward_traverse,test_subscript](const auto& t){
        auto core_order = std::get<0>(t);
        auto traverse_order = std::get<1>(t);
        auto elements = std::get<2>(t);
        auto shape = std::get<3>(t);
        auto broadcast_shape = std::get<4>(t);
        auto expected = std::get<5>(t);
        using core_order_type = decltype(core_order);
        using tensor_implementation_type = tensor_implementation<typename TestType::template core<config_type,value_type,core_order_type>>;
        tensor_implementation_type tensor_implementation{shape, elements.begin(), elements.end()};
        auto result_shape = tensor_implementation.shape();
        using traverse_order_type = decltype(traverse_order);
        auto first = tensor_implementation.template begin<traverse_order_type>(broadcast_shape);
        auto last = tensor_implementation.template end<traverse_order_type>(broadcast_shape);
        auto rfirst = tensor_implementation.template rbegin<traverse_order_type>(broadcast_shape);
        auto rlast = tensor_implementation.template rend<traverse_order_type>(broadcast_shape);

        test_equal(first,last,expected.begin(),expected.end());
        test_equal(rfirst,rlast,expected.rbegin(),expected.rend());
        test_backward_traverse(first,last,expected.begin(),expected.end());
        test_backward_traverse(rfirst,rlast,expected.rbegin(),expected.rend());
        test_subscript(first,last,expected.begin(),expected.end());
        test_subscript(rfirst,rlast,expected.rbegin(),expected.rend());
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_tensor_implementation_data_accesor_result_type_non_const_accessible_core","[test_tensor_implementation]",
    (test_tensor_implementation_::test_core_subscriptable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_indexible<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_walkable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_iterable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_subscriptable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>),
    (test_tensor_implementation_::test_core_indexible<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>),
    (test_tensor_implementation_::test_core_walkable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>),
    (test_tensor_implementation_::test_core_iterable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>)
)
{
    using core_type = TestType;
    using config_type = typename core_type::config_type;
    using value_type = typename core_type::value_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_implementation_type = gtensor::tensor_implementation<core_type>;
    //non const instance
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().create_walker()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<c_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<c_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<c_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<c_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<c_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<c_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<c_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<c_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().template create_indexer<c_order>()[std::declval<index_type>()]),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<f_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<f_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<f_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<f_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<f_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<f_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<f_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<f_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().template create_indexer<f_order>()[std::declval<index_type>()]),value_type&>);
    //const instance
    //core doesn't provide const access must not compile
}

TEMPLATE_TEST_CASE("test_tensor_implementation_data_accesor_result_type_const_accessible_core","[test_tensor_implementation]",
    (test_tensor_implementation_::test_core_const_subscriptable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_const_indexible<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_const_walkable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_const_iterable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_const_subscriptable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>),
    (test_tensor_implementation_::test_core_const_indexible<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>),
    (test_tensor_implementation_::test_core_const_walkable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>),
    (test_tensor_implementation_::test_core_const_iterable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>)
)
{
    using core_type = TestType;
    using config_type = typename core_type::config_type;
    using value_type = typename core_type::value_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_implementation_type = gtensor::tensor_implementation<core_type>;
    //non const instance
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().create_walker()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().template create_indexer<c_order>()[std::declval<index_type>()]),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().template create_indexer<f_order>()[std::declval<index_type>()]),const value_type&>);
    //const instance
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().create_walker()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template begin<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template end<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rbegin<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rend<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template begin<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template end<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rbegin<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rend<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor_implementation_type>().template create_indexer<c_order>()[std::declval<index_type>()]),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template begin<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template end<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rbegin<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rend<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template begin<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template end<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rbegin<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rend<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor_implementation_type>().template create_indexer<f_order>()[std::declval<index_type>()]),const value_type&>);
}

TEMPLATE_TEST_CASE("test_tensor_implementation_data_accesor_result_type_full_accessible_core","[test_tensor_implementation]",
    (test_tensor_implementation_::test_core_full_subscriptable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_full_indexible<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_full_walkable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_full_iterable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::c_order>),
    (test_tensor_implementation_::test_core_full_subscriptable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>),
    (test_tensor_implementation_::test_core_full_indexible<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>),
    (test_tensor_implementation_::test_core_full_walkable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>),
    (test_tensor_implementation_::test_core_full_iterable<test_config::config_storage_selector_t<std::vector>,int,gtensor::config::f_order>)
)
{
    using core_type = TestType;
    using config_type = typename core_type::config_type;
    using value_type = typename core_type::value_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using tensor_implementation_type = gtensor::tensor_implementation<core_type>;
    //non const instance
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().create_walker()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<c_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<c_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<c_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<c_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<c_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<c_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<c_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<c_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().template create_indexer<c_order>()[std::declval<index_type>()]),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<f_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<f_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<f_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<f_order>()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template begin<f_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template end<f_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rbegin<f_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().template rend<f_order>(std::declval<shape_type>())),value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().template create_indexer<f_order>()[std::declval<index_type>()]),value_type&>);
    //const instance
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().create_walker()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template begin<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template end<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rbegin<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rend<c_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template begin<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template end<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rbegin<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rend<c_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor_implementation_type>().template create_indexer<c_order>()[std::declval<index_type>()]),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template begin<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template end<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rbegin<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rend<f_order>()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template begin<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template end<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rbegin<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().template rend<f_order>(std::declval<shape_type>())),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor_implementation_type>().template create_indexer<f_order>()[std::declval<index_type>()]),const value_type&>);
}
