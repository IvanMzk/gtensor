#include <vector>
#include <string>
#include <list>
#include "catch.hpp"
#include "integral_type.hpp"
#include "tensor_implementation.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"


namespace test_tensor_implementation_{

// template<typename T>
// class subscriptable_storage
// {
//     using inner_storage_type = std::vector<T>;
//     inner_storage_type impl_;
// public:
//     using value_type = T;
//     using size_type = typename inner_storage_type::size_type;
//     using difference_type = typename inner_storage_type::difference_type;
//     subscriptable_storage(size_type n):
//         impl_(n)
//     {}
//     template<typename It>
//     subscriptable_storage(It first, It last):
//         impl_(first,last)
//     {}
//     size_type size()const{return impl_.size();}
//     decltype(auto) operator[](size_type i){return impl_[i];}
// };
// template<typename T>
// class subscriptable_storage_integral
// {
//     using inner_storage_type = std::vector<T>;
//     using inner_size_type = typename inner_storage_type::size_type;
//     using inner_difference_type = typename inner_storage_type::difference_type;
//     inner_storage_type impl_;
// public:
//     using value_type = T;
//     using size_type = integral_type::integral<inner_size_type>;
//     using difference_type = integral_type::integral<inner_difference_type>;
//     subscriptable_storage_integral(size_type n):
//         impl_(n.value())
//     {}
//     template<typename It>
//     subscriptable_storage_integral(It first, It last):
//         impl_(first,last)
//     {}
//     size_type size()const{return impl_.size();}
//     decltype(auto) operator[](size_type i){return impl_[i.value()];}
// };
// template<typename T>
// class iterable_storage
// {
//     using inner_storage_type = std::vector<T>;
//     inner_storage_type impl_;
// public:
//     using value_type = T;
//     using size_type = typename inner_storage_type::size_type;
//     using difference_type = typename inner_storage_type::difference_type;
//     iterable_storage(size_type n):
//         impl_(n)
//     {}
//     template<typename It>
//     iterable_storage(It first, It last):
//         impl_(first,last)
//     {}
//     size_type size()const{return impl_.size();}
//     auto begin(){return impl_.begin();}
//     auto end(){return impl_.end();}
// };
// template<typename T>
// class iterable_storage_integral
// {
//     using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<subscriptable_storage_integral>,T>;
//     using inner_storage_type = typename config_type::template storage<T>;
//     using indexer_type = gtensor::basic_indexer<inner_storage_type&>;
//     using iterator = gtensor::indexer_iterator<config_type,indexer_type>;
//     inner_storage_type impl_;
// public:
//     using value_type = T;
//     using size_type = typename inner_storage_type::size_type;
//     using difference_type = typename inner_storage_type::difference_type;
//     iterable_storage_integral(size_type n):
//         impl_(n)
//     {}
//     template<typename It>
//     iterable_storage_integral(It first, It last):
//         impl_(first,last)
//     {}
//     size_type size()const{return impl_.size();}
//     iterator begin(){return iterator{indexer_type{impl_},0};}
//     iterator end(){return iterator{indexer_type{impl_},size()};}
// };

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
        return gtensor::walker<config_type,indexer_type>{descriptor().adapted_strides(),descriptor().reset_strides(),descriptor().offset(),indexer_type{elements_},max_dim};
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
        return gtensor::walker<config_type,indexer_type>{descriptor().adapted_strides(),descriptor().reset_strides(),descriptor().offset(),indexer_type{elements_},max_dim};
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
    auto create_walker(dim_type max_dim)const{
        using indexer_type = gtensor::basic_indexer<const storage_type&>;
        return gtensor::walker<config_type,indexer_type>{descriptor().adapted_strides(),descriptor().reset_strides(),descriptor().offset(),indexer_type{elements_},max_dim};
    }
    auto create_walker(dim_type max_dim){
        using indexer_type = gtensor::basic_indexer<storage_type&>;
        return gtensor::walker<config_type,indexer_type>{descriptor().adapted_strides(),descriptor().reset_strides(),descriptor().offset(),indexer_type{elements_},max_dim};
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
            SECTION("test_equal")
            {
                REQUIRE(std::equal(tensor_implementation.template begin<traverse_order_type>(),tensor_implementation.template end<traverse_order_type>(),expected.begin(),expected.end()));
                REQUIRE(std::equal(tensor_implementation.template rbegin<traverse_order_type>(),tensor_implementation.template rend<traverse_order_type>(),expected.rbegin(),expected.rend()));
            }
            SECTION("test_iterator_backward_traverse")
            {
                std::vector<value_type> result;
                for(auto it = tensor_implementation.template end<traverse_order_type>(), it_first = tensor_implementation.template begin<traverse_order_type>(); it!=it_first;){
                    result.push_back(*--it);
                }
                REQUIRE(std::equal(result.rbegin(),result.rend(),expected.begin(),expected.end()));
            }
            SECTION("test_reverse_iterator_backward_traverse")
            {
                std::vector<value_type> result;
                for(auto it = tensor_implementation.template rend<traverse_order_type>(), it_first = tensor_implementation.template rbegin<traverse_order_type>(); it!=it_first;){
                    result.push_back(*--it);
                }
                REQUIRE(std::equal(result.begin(),result.end(),expected.begin(),expected.end()));
            }
            SECTION("test_iterator_subscript")
            {
                auto first = tensor_implementation.template begin<traverse_order_type>();
                auto last = tensor_implementation.template end<traverse_order_type>();
                std::vector<value_type> result;
                for(index_type i{0}, i_last = last - first; i!=i_last; ++i){
                    result.push_back(first[i]);
                }
                REQUIRE(std::equal(result.begin(),result.end(),expected.begin(),expected.end()));
            }
            SECTION("test_reverse_iterator_subscript")
            {
                auto first = tensor_implementation.template rbegin<traverse_order_type>();
                auto last = tensor_implementation.template rend<traverse_order_type>();
                std::vector<value_type> result;
                for(index_type i{0}, i_last = last - first; i!=i_last; ++i){
                    result.push_back(first[i]);
                }
                REQUIRE(std::equal(result.rbegin(),result.rend(),expected.begin(),expected.end()));
            }
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
            using traverse_order_type = decltype(traverse_order);
            using walker_iterator_type = gtensor::walker_iterator<config_type,decltype(tensor_implementation.create_walker()),traverse_order_type>;
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

// TEMPLATE_TEST_CASE("test_tensor_implementation_broadcast_iterator_c_layout","[test_tensor_implementation]",
//     //non const accessible core
//     (test_tensor_implementation_::test_core_subscriptable<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_subscriptable<test_tensor_implementation_::test_config_c_layout<test_tensor_implementation_::subscriptable_storage_integral>,int>),
//     (test_tensor_implementation_::test_core_indexible<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_walkable<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_iterable<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_iterable<test_tensor_implementation_::test_config_c_layout<test_tensor_implementation_::iterable_storage_integral>,int>),
//     //const accessible core
//     (test_tensor_implementation_::test_core_const_subscriptable<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_const_indexible<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_const_walkable<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_const_iterable<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     //full accessible core
//     (test_tensor_implementation_::test_core_full_subscriptable<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_full_indexible<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_full_walkable<test_tensor_implementation_::test_config_c_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_full_iterable<test_tensor_implementation_::test_config_c_layout<std::vector>,int>)
// )
// {
//     using core_type = TestType;
//     using config_type = typename core_type::config_type;
//     using value_type = typename core_type::value_type;
//     using shape_type = typename config_type::shape_type;
//     using tensor_implementation_type = gtensor::tensor_implementation<core_type>;
//     using helpers_for_testing::apply_by_element;

//     //0shape,1elements,2broadcast_shape,3expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(shape_type{}, std::vector<value_type>{1}, shape_type{}, std::vector<value_type>{1}),
//         std::make_tuple(shape_type{}, std::vector<value_type>{2}, shape_type{1}, std::vector<value_type>{2}),
//         std::make_tuple(shape_type{}, std::vector<value_type>{3}, shape_type{5}, std::vector<value_type>{3,3,3,3,3}),
//         std::make_tuple(shape_type{1}, std::vector<value_type>{2}, shape_type{1}, std::vector<value_type>{2}),
//         std::make_tuple(shape_type{1}, std::vector<value_type>{1}, shape_type{5}, std::vector<value_type>{1,1,1,1,1}),
//         std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{1}, std::vector<value_type>{1}),
//         std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}),
//         std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{1,6}, std::vector<value_type>{1,2,3,4,5,6}),
//         std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{6,1}, std::vector<value_type>{1,1,1,1,1,1}),
//         std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{2,6}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
//         std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::array<int,2>{2,3}, std::vector<value_type>{1,2,3,4,5,6}),
//         std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::list<std::size_t>{2,2,3}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
//         std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{1,3}, std::vector<value_type>{1,2,3}),
//         std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{2,1,3}, std::vector<value_type>{1,2,3,1,2,3})
//     );

//     auto test = [](const auto& t){
//         const auto shape = std::get<0>(t);
//         const auto elements = std::get<1>(t);
//         const auto broadcast_shape = std::get<2>(t);
//         const auto expected = std::get<3>(t);
//         tensor_implementation_type result_tensor_implementation{shape, elements.begin(), elements.end()};
//         REQUIRE(std::equal(result_tensor_implementation.begin(broadcast_shape),result_tensor_implementation.end(broadcast_shape),expected.begin(),expected.end()));
//         REQUIRE(std::equal(result_tensor_implementation.rbegin(broadcast_shape),result_tensor_implementation.rend(broadcast_shape),expected.rbegin(),expected.rend()));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_tensor_implementation_broadcast_iterator_f_layout","[test_tensor_implementation]",
//     //non const accessible core
//     (test_tensor_implementation_::test_core_subscriptable<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_subscriptable<test_tensor_implementation_::test_config_f_layout<test_tensor_implementation_::subscriptable_storage_integral>,int>),
//     (test_tensor_implementation_::test_core_indexible<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_walkable<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_iterable<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_iterable<test_tensor_implementation_::test_config_f_layout<test_tensor_implementation_::iterable_storage_integral>,int>),
//     //const accessible core
//     (test_tensor_implementation_::test_core_const_subscriptable<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_const_indexible<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_const_walkable<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_const_iterable<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     //full accessible core
//     (test_tensor_implementation_::test_core_full_subscriptable<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_full_indexible<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_full_walkable<test_tensor_implementation_::test_config_f_layout<std::vector>,int>),
//     (test_tensor_implementation_::test_core_full_iterable<test_tensor_implementation_::test_config_f_layout<std::vector>,int>)
// )
// {
//     using core_type = TestType;
//     using config_type = typename core_type::config_type;
//     using value_type = typename core_type::value_type;
//     using shape_type = typename config_type::shape_type;
//     using tensor_implementation_type = gtensor::tensor_implementation<core_type>;
//     using helpers_for_testing::apply_by_element;

//     //0shape,1elements,2broadcast_shape,3expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(shape_type{}, std::vector<value_type>{1}, shape_type{}, std::vector<value_type>{1}),
//         std::make_tuple(shape_type{}, std::vector<value_type>{2}, shape_type{1}, std::vector<value_type>{2}),
//         std::make_tuple(shape_type{}, std::vector<value_type>{3}, shape_type{5}, std::vector<value_type>{3,3,3,3,3}),
//         std::make_tuple(shape_type{1}, std::vector<value_type>{2}, shape_type{1}, std::vector<value_type>{2}),
//         std::make_tuple(shape_type{1}, std::vector<value_type>{1}, shape_type{5}, std::vector<value_type>{1,1,1,1,1}),
//         std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{1}, std::vector<value_type>{1}),
//         std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}),
//         std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{1,6}, std::vector<value_type>{1,2,3,4,5,6}),
//         std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{6,1}, std::vector<value_type>{1,1,1,1,1,1}),
//         std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{2,6}, std::vector<value_type>{1,1,2,2,3,3,4,4,5,5,6,6}),
//         std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::array<int,2>{2,3}, std::vector<value_type>{1,2,3,4,5,6}),
//         std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::list<std::size_t>{2,2,3}, std::vector<value_type>{1,1,2,2,3,3,4,4,5,5,6,6}),
//         std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{1,3}, std::vector<value_type>{1,3,5}),
//         std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{2,1,3}, std::vector<value_type>{1,1,3,3,5,5})
//     );

//     auto test = [](const auto& t){
//         const auto shape = std::get<0>(t);
//         const auto elements = std::get<1>(t);
//         const auto broadcast_shape = std::get<2>(t);
//         const auto expected = std::get<3>(t);
//         tensor_implementation_type result_tensor_implementation{shape, elements.begin(), elements.end()};
//         REQUIRE(std::equal(result_tensor_implementation.begin(broadcast_shape),result_tensor_implementation.end(broadcast_shape),expected.begin(),expected.end()));
//         REQUIRE(std::equal(result_tensor_implementation.rbegin(broadcast_shape),result_tensor_implementation.rend(broadcast_shape),expected.rbegin(),expected.rend()));
//     };
//     apply_by_element(test,test_data);
// }

// TEMPLATE_TEST_CASE("test_tensor_implementation_data_accesor_result_type_non_const_accessible_core","[test_tensor_implementation]",
//     (test_tensor_implementation_::test_core_subscriptable<test_config::config_storage_selector_t<std::vector>,int>),
//     (test_tensor_implementation_::test_core_subscriptable<test_config::config_storage_selector_t<test_tensor_implementation_::subscriptable_storage_integral>,int>),
//     (test_tensor_implementation_::test_core_indexible<test_config::config_storage_selector_t<std::vector>,int>),
//     (test_tensor_implementation_::test_core_walkable<test_config::config_storage_selector_t<std::vector>,int>),
//     (test_tensor_implementation_::test_core_iterable<test_config::config_storage_selector_t<std::vector>,int>),
//     (test_tensor_implementation_::test_core_iterable<test_config::config_storage_selector_t<test_tensor_implementation_::iterable_storage_integral>,int>)
// )
// {
//     using core_type = TestType;
//     using config_type = typename core_type::config_type;
//     using value_type = typename core_type::value_type;
//     using index_type = typename config_type::index_type;
//     using shape_type = typename config_type::shape_type;
//     using tensor_implementation_type = gtensor::tensor_implementation<core_type>;

//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().begin()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().end()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rbegin()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rend()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().begin(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().end(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rbegin(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rend(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().create_walker()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().create_indexer()[std::declval<index_type>()]),value_type&>);
// }

// TEMPLATE_TEST_CASE("test_tensor_implementation_data_accesor_result_type_const_accessible_core","[test_tensor_implementation]",
//     (test_tensor_implementation_::test_core_const_subscriptable<test_config::config_storage_selector_t<std::vector>,int>),
//     (test_tensor_implementation_::test_core_const_indexible<test_config::config_storage_selector_t<std::vector>,int>),
//     (test_tensor_implementation_::test_core_const_walkable<test_config::config_storage_selector_t<std::vector>,int>),
//     (test_tensor_implementation_::test_core_const_iterable<test_config::config_storage_selector_t<std::vector>,int>)
// )
// {
//     using core_type = TestType;
//     using config_type = typename core_type::config_type;
//     using value_type = typename core_type::value_type;
//     using index_type = typename config_type::index_type;
//     using shape_type = typename config_type::shape_type;
//     using tensor_implementation_type = gtensor::tensor_implementation<core_type>;

//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().begin()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().end()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rbegin()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rend()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().begin(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().end(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rbegin(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rend(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().create_walker()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().create_indexer()[std::declval<index_type>()]),const value_type&>);

//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().begin()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().end()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().rbegin()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().rend()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().create_walker()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(std::declval<const tensor_implementation_type>().create_indexer()[std::declval<index_type>()]),const value_type&>);
// }

// TEMPLATE_TEST_CASE("test_tensor_implementation_data_accesor_result_type_full_accessible_core","[test_tensor_implementation]",
//     (test_tensor_implementation_::test_core_full_subscriptable<test_config::config_storage_selector_t<std::vector>,int>),
//     (test_tensor_implementation_::test_core_full_indexible<test_config::config_storage_selector_t<std::vector>,int>),
//     (test_tensor_implementation_::test_core_full_walkable<test_config::config_storage_selector_t<std::vector>,int>),
//     (test_tensor_implementation_::test_core_full_iterable<test_config::config_storage_selector_t<std::vector>,int>)
// )
// {
//     using core_type = TestType;
//     using config_type = typename core_type::config_type;
//     using value_type = typename core_type::value_type;
//     using index_type = typename config_type::index_type;
//     using shape_type = typename config_type::shape_type;
//     using tensor_implementation_type = gtensor::tensor_implementation<core_type>;

//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().begin()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().end()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rbegin()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rend()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().begin(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().end(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rbegin(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rend(std::declval<shape_type>())),value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().create_walker()),value_type&>);
//     REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().create_indexer()[std::declval<index_type>()]),value_type&>);

//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().begin()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().end()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().rbegin()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().rend()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().begin(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().end(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().rbegin(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().rend(std::declval<shape_type>())),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().create_walker()),const value_type&>);
//     REQUIRE(std::is_same_v<decltype(std::declval<const tensor_implementation_type>().create_indexer()[std::declval<index_type>()]),const value_type&>);
// }

