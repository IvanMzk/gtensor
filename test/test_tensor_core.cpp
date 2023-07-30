#include <vector>
#include <string>
#include <list>
#include "catch.hpp"
#include "tensor_core.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_has_mutating_iterator","[test_tensor_implementation]")
{
    struct const_iterable{
        using value_type = int;
        const value_type* begin()const{return nullptr;}
        const value_type* end()const{return nullptr;}
    };
    struct non_const_iterable{
        using value_type = int;
        value_type* begin(){return nullptr;}
        value_type* end(){return nullptr;}
    };
    struct iterable{
        using value_type = int;
        const value_type* begin()const{return nullptr;}
        const value_type* end()const{return nullptr;}
        value_type* begin(){return nullptr;}
        value_type* end(){return nullptr;}
    };
    using gtensor::detail::has_mutating_iterator_v;
    REQUIRE(!has_mutating_iterator_v<void>);
    REQUIRE(!has_mutating_iterator_v<void*>);
    REQUIRE(!has_mutating_iterator_v<int>);
    REQUIRE(!has_mutating_iterator_v<int*>);
    REQUIRE(!has_mutating_iterator_v<int[]>);
    REQUIRE(!has_mutating_iterator_v<const_iterable>);
    REQUIRE(!has_mutating_iterator_v<const std::list<int>>);
    REQUIRE(!has_mutating_iterator_v<const std::vector<int>>);
    REQUIRE(!has_mutating_iterator_v<const std::string>);
    REQUIRE(has_mutating_iterator_v<iterable>);
    REQUIRE(has_mutating_iterator_v<non_const_iterable>);
    REQUIRE(has_mutating_iterator_v<std::list<int>>);
    REQUIRE(has_mutating_iterator_v<std::vector<int>>);
    REQUIRE(has_mutating_iterator_v<std::vector<bool>>);
    REQUIRE(has_mutating_iterator_v<std::string>);
}

TEST_CASE("test_has_mutating_subscript_operator","[test_tensor_implementation]")
{
    struct const_subscriptable{
        using size_type = std::size_t;
        using difference_type = std::int64_t;
        using value_type = std::string;
        const value_type& operator[](size_type)const;
    };
    struct non_const_subscriptable{
        using size_type = std::size_t;
        using difference_type = std::int64_t;
        using value_type = std::string;
        value_type& operator[](size_type);
    };
    struct subscriptable{
        using size_type = std::size_t;
        using difference_type = std::int64_t;
        using value_type = std::string;
        const value_type& operator[](size_type)const;
        value_type& operator[](size_type);
    };

    using gtensor::detail::has_mutating_subscript_operator_v;
    REQUIRE(!has_mutating_subscript_operator_v<void>);
    REQUIRE(!has_mutating_subscript_operator_v<void*>);
    REQUIRE(!has_mutating_subscript_operator_v<int>);
    REQUIRE(!has_mutating_subscript_operator_v<int*>);
    REQUIRE(!has_mutating_subscript_operator_v<int[]>);
    REQUIRE(!has_mutating_subscript_operator_v<const_subscriptable>);
    REQUIRE(!has_mutating_subscript_operator_v<const std::list<int>>);
    REQUIRE(!has_mutating_subscript_operator_v<const std::vector<int>>);
    REQUIRE(!has_mutating_subscript_operator_v<const std::string>);
    REQUIRE(!has_mutating_subscript_operator_v<std::list<int>>);

    REQUIRE(has_mutating_subscript_operator_v<subscriptable>);
    REQUIRE(has_mutating_subscript_operator_v<non_const_subscriptable>);
    REQUIRE(has_mutating_subscript_operator_v<std::vector<int>>);
    REQUIRE(has_mutating_subscript_operator_v<std::vector<bool>>);
    REQUIRE(has_mutating_subscript_operator_v<std::string>);
}

namespace test_storage_core{

template<typename T>
class subscriptable_storage
{
    using inner_storage_type = std::vector<T>;
    inner_storage_type impl_;
public:
    using value_type = T;
    using size_type = typename inner_storage_type::size_type;
    using difference_type = typename inner_storage_type::difference_type;
    subscriptable_storage(size_type n):
        impl_(n)
    {}
    template<typename It>
    subscriptable_storage(It first, It last):
        impl_(first,last)
    {}
    size_type size()const{return impl_.size();}
    decltype(auto) operator[](size_type i){return impl_[i];}
};
template<typename T>
class iterable_storage
{
    using inner_storage_type = std::vector<T>;
    inner_storage_type impl_;
public:
    using value_type = T;
    using size_type = typename inner_storage_type::size_type;
    using difference_type = typename inner_storage_type::difference_type;
    iterable_storage(size_type n):
        impl_(n)
    {}
    template<typename It>
    iterable_storage(It first, It last):
        impl_(first,last)
    {}
    size_type size()const{return impl_.size();}
    auto begin(){return impl_.begin();}
    auto end(){return impl_.end();}
};
}   //end of namespace test_storage_core

TEMPLATE_TEST_CASE("test_storage_core","[test_tensor_implementation]",
    //0test_config,1is_iterable,2is_subscriptable
    (std::tuple<test_config::config_storage_selector_t<std::vector>, std::true_type, std::true_type>),
    (std::tuple<test_config::config_storage_selector_t<std::list>, std::true_type, std::false_type>),
    (std::tuple<test_config::config_storage_selector_t<test_storage_core::subscriptable_storage>, std::false_type, std::true_type>),
    (std::tuple<test_config::config_storage_selector_t<test_storage_core::iterable_storage>, std::true_type, std::false_type>)
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<std::tuple_element_t<0,TestType>,value_type>;
    using is_iterable =std::tuple_element_t<1,TestType>;
    using is_subscriptable =std::tuple_element_t<2,TestType>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::storage_core;
    using helpers_for_testing::apply_by_element;

    SECTION("test_storage_core_shape_value_constructor"){
        //0layout,1shape,2value,3expected_shape,4expected_elements
        auto test_data = std::make_tuple(
            //c_order
            std::make_tuple(c_order{}, shape_type{}, value_type{3}, shape_type{}, std::vector<value_type>{3}),
            std::make_tuple(c_order{}, shape_type{0}, value_type{0}, shape_type{0}, std::vector<value_type>{}),
            std::make_tuple(c_order{}, shape_type{3,0}, value_type{2}, shape_type{3,0}, std::vector<value_type>{}),
            std::make_tuple(c_order{}, shape_type{1}, value_type{1}, shape_type{1}, std::vector<value_type>{1}),
            std::make_tuple(c_order{}, shape_type{5}, value_type{2}, shape_type{5}, std::vector<value_type>{2,2,2,2,2}),
            std::make_tuple(c_order{}, shape_type{2,3}, value_type{4}, shape_type{2,3}, std::vector<value_type>{4,4,4,4,4,4}),
            //f_order
            std::make_tuple(f_order{}, shape_type{}, value_type{3}, shape_type{}, std::vector<value_type>{3}),
            std::make_tuple(f_order{}, shape_type{0}, value_type{0}, shape_type{0}, std::vector<value_type>{}),
            std::make_tuple(f_order{}, shape_type{3,0}, value_type{2}, shape_type{3,0}, std::vector<value_type>{}),
            std::make_tuple(f_order{}, shape_type{1}, value_type{1}, shape_type{1}, std::vector<value_type>{1}),
            std::make_tuple(f_order{}, shape_type{5}, value_type{2}, shape_type{5}, std::vector<value_type>{2,2,2,2,2}),
            std::make_tuple(f_order{}, shape_type{2,3}, value_type{4}, shape_type{2,3}, std::vector<value_type>{4,4,4,4,4,4})
        );
        auto test = [](const auto& t){
            auto layout = std::get<0>(t);
            auto shape = std::get<1>(t);
            auto value = std::get<2>(t);
            auto expected_shape = std::get<3>(t);
            auto expected_elements = std::get<4>(t);
            using layout_type = decltype(layout);
            using core_type = storage_core<config_type,value_type,layout_type>;
            core_type core(shape,value);
            auto result_shape = core.descriptor().shape();
            REQUIRE(result_shape == expected_shape);
            if constexpr (is_iterable::value){
                REQUIRE(std::equal(core.begin(),core.end(),expected_elements.begin(),expected_elements.end()));
            }
            if constexpr (is_subscriptable::value){
                std::vector<value_type> result_elements{};
                const index_type result_size = core.descriptor().size();
                for (index_type i{0}; i!=result_size; ++i){
                    result_elements.push_back(core[i]);
                }
                REQUIRE(result_elements == expected_elements);
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_storage_core_init_list_constructor")
    {
        //0core,1expected_shape,2expected_elements
        auto test_data = std::make_tuple(
            //c_order
            std::make_tuple(storage_core<config_type,value_type,c_order>(std::initializer_list<value_type>{}), shape_type{0}, std::vector<value_type>{}),
            std::make_tuple(storage_core<config_type,value_type,c_order>(std::initializer_list<value_type>{1}), shape_type{1}, std::vector<value_type>{1}),
            std::make_tuple(storage_core<config_type,value_type,c_order>(std::initializer_list<value_type>{1,2,3,4,5}), shape_type{5}, std::vector<value_type>{1,2,3,4,5}),
            std::make_tuple(
                storage_core<config_type,value_type,c_order>(std::initializer_list<std::initializer_list<value_type>>{{1,2,3,4,5}}),
                shape_type{1,5},
                std::vector<value_type>{1,2,3,4,5}
            ),
            std::make_tuple(
                storage_core<config_type,value_type,c_order>(std::initializer_list<std::initializer_list<value_type>>{{1,2,3},{4,5,6}}),
                shape_type{2,3},
                std::vector<value_type>{1,2,3,4,5,6}
            ),
            std::make_tuple(
                storage_core<config_type,value_type,c_order>(std::initializer_list<std::initializer_list<std::initializer_list<value_type>>>{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}),
                shape_type{2,3,2},
                std::vector<value_type>{1,2,3,4,5,6,7,8,9,10,11,12}
            ),
            //f_order
            std::make_tuple(storage_core<config_type,value_type,f_order>(std::initializer_list<value_type>{}), shape_type{0}, std::vector<value_type>{}),
            std::make_tuple(storage_core<config_type,value_type,f_order>(std::initializer_list<value_type>{1}), shape_type{1}, std::vector<value_type>{1}),
            std::make_tuple(storage_core<config_type,value_type,f_order>(std::initializer_list<value_type>{1,2,3,4,5}), shape_type{5}, std::vector<value_type>{1,2,3,4,5}),
            std::make_tuple(
                storage_core<config_type,value_type,f_order>(std::initializer_list<std::initializer_list<value_type>>{{1,2,3,4,5}}),
                shape_type{1,5},
                std::vector<value_type>{1,2,3,4,5}
            ),
            std::make_tuple(
                storage_core<config_type,value_type,f_order>(std::initializer_list<std::initializer_list<value_type>>{{1,2,3},{4,5,6}}),
                shape_type{2,3},
                std::vector<value_type>{1,4,2,5,3,6}
            ),
            std::make_tuple(
                storage_core<config_type,value_type,f_order>(std::initializer_list<std::initializer_list<std::initializer_list<value_type>>>{{{1,2},{3,4},{5,6}},{{7,8},{9,10},{11,12}}}),
                shape_type{2,3,2},
                std::vector<value_type>{1,7,3,9,5,11,2,8,4,10,6,12}
            )
        );
        auto test = [](const auto& t){
            auto core = std::get<0>(t);
            auto expected_shape = std::get<1>(t);
            auto expected_elements = std::get<2>(t);
            auto result_shape = core.descriptor().shape();
            REQUIRE(result_shape == expected_shape);
            if constexpr (is_iterable::value){
                REQUIRE(std::equal(core.begin(),core.end(),expected_elements.begin(),expected_elements.end()));
            }
            if constexpr (is_subscriptable::value){
                std::vector<value_type> result_elements{};
                const index_type result_size = core.descriptor().size();
                for (index_type i{0}; i!=result_size; ++i){
                    result_elements.push_back(core[i]);
                }
                REQUIRE(result_elements == expected_elements);
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_storage_core_range_constructor"){
        //0layout,1shape,2elements,3expected_shape,4expected_elements
        auto test_data = std::make_tuple(
            //c_order
            std::make_tuple(c_order{}, shape_type{0}, std::vector<value_type>{}, shape_type{0}, std::vector<value_type>{}),
            std::make_tuple(c_order{}, shape_type{0,3}, std::vector<value_type>{}, shape_type{0,3}, std::vector<value_type>{}),
            std::make_tuple(c_order{}, shape_type{0}, std::vector<value_type>{2,2,2,2}, shape_type{0}, std::vector<value_type>{}),
            std::make_tuple(c_order{}, shape_type{1}, std::vector<value_type>{2}, shape_type{1}, std::vector<value_type>{2}),
            std::make_tuple(c_order{}, shape_type{1}, std::vector<value_type>{3,4,5}, shape_type{1}, std::vector<value_type>{3}),
            std::make_tuple(c_order{}, shape_type{5}, std::vector<value_type>{1,2,3,4,5}, shape_type{5}, std::vector<value_type>{1,2,3,4,5}),
            std::make_tuple(c_order{}, shape_type{5}, std::vector<value_type>{1,2,3,4,5,6,7,8,9}, shape_type{5}, std::vector<value_type>{1,2,3,4,5}),
            std::make_tuple(c_order{}, shape_type{5}, std::vector<value_type>{1,2,3}, shape_type{5}, std::vector<value_type>{1,2,3,0,0}),
            std::make_tuple(c_order{}, shape_type{3,2}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{3,2}, std::vector<value_type>{1,2,3,4,5,6}),
            std::make_tuple(c_order{}, shape_type{3,2}, std::vector<value_type>{1,2}, shape_type{3,2}, std::vector<value_type>{1,2,0,0,0,0}),
            std::make_tuple(c_order{}, shape_type{3,2}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10}, shape_type{3,2}, std::vector<value_type>{1,2,3,4,5,6}),
            //f_order
            std::make_tuple(f_order{}, shape_type{0}, std::vector<value_type>{}, shape_type{0}, std::vector<value_type>{}),
            std::make_tuple(f_order{}, shape_type{0,3}, std::vector<value_type>{}, shape_type{0,3}, std::vector<value_type>{}),
            std::make_tuple(f_order{}, shape_type{0}, std::vector<value_type>{2,2,2,2}, shape_type{0}, std::vector<value_type>{}),
            std::make_tuple(f_order{}, shape_type{1}, std::vector<value_type>{2}, shape_type{1}, std::vector<value_type>{2}),
            std::make_tuple(f_order{}, shape_type{1}, std::vector<value_type>{3,4,5}, shape_type{1}, std::vector<value_type>{3}),
            std::make_tuple(f_order{}, shape_type{5}, std::vector<value_type>{1,2,3,4,5}, shape_type{5}, std::vector<value_type>{1,2,3,4,5}),
            std::make_tuple(f_order{}, shape_type{5}, std::vector<value_type>{1,2,3,4,5,6,7,8,9}, shape_type{5}, std::vector<value_type>{1,2,3,4,5}),
            std::make_tuple(f_order{}, shape_type{5}, std::vector<value_type>{1,2,3}, shape_type{5}, std::vector<value_type>{1,2,3,0,0}),
            std::make_tuple(f_order{}, shape_type{3,2}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{3,2}, std::vector<value_type>{1,2,3,4,5,6}),
            std::make_tuple(f_order{}, shape_type{3,2}, std::vector<value_type>{1,2}, shape_type{3,2}, std::vector<value_type>{1,2,0,0,0,0}),
            std::make_tuple(f_order{}, shape_type{3,2}, std::vector<value_type>{1,2,3,4,5,6,7,8,9,10}, shape_type{3,2}, std::vector<value_type>{1,2,3,4,5,6})
        );
        auto test = [](const auto& t){
            auto layout = std::get<0>(t);
            auto shape = std::get<1>(t);
            auto elements = std::get<2>(t);
            auto expected_shape = std::get<3>(t);
            auto expected_elements = std::get<4>(t);
            using layout_type = decltype(layout);
            using core_type = storage_core<config_type,value_type,layout_type>;
            core_type core(shape,elements.begin(),elements.end());
            auto result_shape = core.descriptor().shape();
            REQUIRE(result_shape == expected_shape);
            if constexpr (is_iterable::value){
                REQUIRE(std::equal(core.begin(),core.end(),expected_elements.begin(),expected_elements.end()));
            }
            if constexpr (is_subscriptable::value){
                std::vector<value_type> result_elements{};
                const index_type result_size = core.descriptor().size();
                for (index_type i{0}; i!=result_size; ++i){
                    result_elements.push_back(core[i]);
                }
                REQUIRE(result_elements == expected_elements);
            }
        };
        apply_by_element(test,test_data);
    }
}
