#include <vector>
#include <string>
#include <list>
#include "catch.hpp"
#include "tensor_implementation.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_has_iterator","[test_tensor_implementation]")
{
    struct const_iterable{
        const int* begin()const{return nullptr;}
        const int* end()const{return nullptr;}
    };
    struct non_const_iterable{
        int* begin(){return nullptr;}
        int* end(){return nullptr;}
    };
    struct iterable{
        const int* begin()const{return nullptr;}
        const int* end()const{return nullptr;}
        int* begin(){return nullptr;}
        int* end(){return nullptr;}
    };

    SECTION("test_has_iterator_v")
    {
        using gtensor::detail::has_iterator_v;
        REQUIRE(!has_iterator_v<void>);
        REQUIRE(!has_iterator_v<void*>);
        REQUIRE(!has_iterator_v<int>);
        REQUIRE(!has_iterator_v<int*>);
        REQUIRE(!has_iterator_v<int[]>);
        REQUIRE(!has_iterator_v<const_iterable>);

        REQUIRE(has_iterator_v<iterable>);
        REQUIRE(has_iterator_v<non_const_iterable>);
        REQUIRE(has_iterator_v<std::list<int>>);
        REQUIRE(has_iterator_v<std::vector<int>>);
        REQUIRE(has_iterator_v<std::string>);
    }
    SECTION("test_has_const_iterator_v")
    {
        using gtensor::detail::has_const_iterator_v;
        REQUIRE(!has_const_iterator_v<void>);
        REQUIRE(!has_const_iterator_v<void*>);
        REQUIRE(!has_const_iterator_v<int>);
        REQUIRE(!has_const_iterator_v<int*>);
        REQUIRE(!has_const_iterator_v<int[]>);
        REQUIRE(!has_const_iterator_v<non_const_iterable>);

        REQUIRE(has_const_iterator_v<iterable>);
        REQUIRE(has_const_iterator_v<const_iterable>);
        REQUIRE(has_const_iterator_v<std::list<int>>);
        REQUIRE(has_const_iterator_v<std::vector<int>>);
        REQUIRE(has_const_iterator_v<std::string>);
    }
}

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

TEST_CASE("test_has_reverse_iterator","[test_tensor_implementation]")
{
    struct const_reverse_iterable{
        const int* rbegin()const{return nullptr;}
        const int* rend()const{return nullptr;}
    };
    struct non_const_reverse_iterable{
        int* rbegin(){return nullptr;}
        int* rend(){return nullptr;}
    };
    struct reverse_iterable{
        const int* rbegin()const{return nullptr;}
        const int* rend()const{return nullptr;}
        int* rbegin(){return nullptr;}
        int* rend(){return nullptr;}
    };

    SECTION("test_has_reverse_iterator_v")
    {
        using gtensor::detail::has_reverse_iterator_v;
        REQUIRE(!has_reverse_iterator_v<void>);
        REQUIRE(!has_reverse_iterator_v<void*>);
        REQUIRE(!has_reverse_iterator_v<int>);
        REQUIRE(!has_reverse_iterator_v<int*>);
        REQUIRE(!has_reverse_iterator_v<int[]>);
        REQUIRE(!has_reverse_iterator_v<const_reverse_iterable>);

        REQUIRE(has_reverse_iterator_v<reverse_iterable>);
        REQUIRE(has_reverse_iterator_v<non_const_reverse_iterable>);
        REQUIRE(has_reverse_iterator_v<std::list<int>>);
        REQUIRE(has_reverse_iterator_v<std::vector<int>>);
        REQUIRE(has_reverse_iterator_v<std::string>);
    }
    SECTION("test_has_const_reverse_iterator_v")
    {
        using gtensor::detail::has_const_reverse_iterator_v;
        REQUIRE(!has_const_reverse_iterator_v<void>);
        REQUIRE(!has_const_reverse_iterator_v<void*>);
        REQUIRE(!has_const_reverse_iterator_v<int>);
        REQUIRE(!has_const_reverse_iterator_v<int*>);
        REQUIRE(!has_const_reverse_iterator_v<int[]>);
        REQUIRE(!has_const_reverse_iterator_v<non_const_reverse_iterable>);

        REQUIRE(has_const_reverse_iterator_v<reverse_iterable>);
        REQUIRE(has_const_reverse_iterator_v<const_reverse_iterable>);
        REQUIRE(has_const_reverse_iterator_v<std::list<int>>);
        REQUIRE(has_const_reverse_iterator_v<std::vector<int>>);
        REQUIRE(has_const_reverse_iterator_v<std::string>);
    }
}

TEST_CASE("test_has_subscript_operator","[test_tensor_implementation]")
{
    struct const_subscriptable{
        using size_type = std::size_t;
        int* p{};
        int operator[](size_type)const{return *p;}
    };
    struct non_const_subscriptable{
        using size_type = std::size_t;
        int* p{};
        int& operator[](size_type){return *p;}
    };
    struct subscriptable{
        using size_type = std::size_t;
        int* p{};
        const int& operator[](size_type)const{return *p;}
        int& operator[](size_type){return *p;}
    };

    SECTION("test_has_subscript_operator")
    {
        using gtensor::detail::has_subscript_operator;
        REQUIRE(!has_subscript_operator<void>());
        REQUIRE(!has_subscript_operator<void*>());
        REQUIRE(!has_subscript_operator<int>());
        REQUIRE(!has_subscript_operator<int*>());
        REQUIRE(!has_subscript_operator<int[]>());
        REQUIRE(!has_subscript_operator<const_subscriptable>());
        REQUIRE(!has_subscript_operator<std::list<int>>());

        REQUIRE(has_subscript_operator<subscriptable>());
        REQUIRE(has_subscript_operator<non_const_subscriptable>());
        REQUIRE(has_subscript_operator<std::vector<int>>());
        REQUIRE(has_subscript_operator<std::string>());
    }
    SECTION("test_has_const_subscript_operator")
    {
        using gtensor::detail::has_subscript_operator_const;
        REQUIRE(!has_subscript_operator_const<void>());
        REQUIRE(!has_subscript_operator_const<void*>());
        REQUIRE(!has_subscript_operator_const<int>());
        REQUIRE(!has_subscript_operator_const<int*>());
        REQUIRE(!has_subscript_operator_const<int[]>());
        REQUIRE(!has_subscript_operator_const<non_const_subscriptable>());
        REQUIRE(!has_subscript_operator_const<std::list<int>>());

        REQUIRE(has_subscript_operator_const<subscriptable>());
        REQUIRE(has_subscript_operator_const<const_subscriptable>());
        REQUIRE(has_subscript_operator_const<std::vector<int>>());
        REQUIRE(has_subscript_operator_const<std::string>());
    }
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

namespace test_storage_engine{
template<typename T>
class minimal_storage
{
    using inner_storage_type = std::vector<T>;
    inner_storage_type impl_;
public:
    using value_type = T;
    using size_type = typename inner_storage_type::size_type;
    using difference_type = typename inner_storage_type::difference_type;
    minimal_storage(size_type n):
        impl_(n)
    {}
    decltype(std::declval<inner_storage_type&>()[std::declval<size_type&>()]) operator[](size_type i){return impl_[i];}
};
}   //end of namespace test_storage_engine

TEMPLATE_TEST_CASE("test_storage_engine","[test_tensor_implementation]",
    //0config_selector,1has_iter,2has_const_iter,3has_reverse_iter,4has_const_reverse_iter,5has_subscript_operator,6has_subscript_operator_const
    (std::tuple<test_config::config_storage_selector<std::vector>, std::true_type,std::true_type,std::true_type,std::true_type,std::true_type,std::true_type>),
    (std::tuple<test_config::config_storage_selector<std::list>, std::true_type,std::true_type,std::true_type,std::true_type,std::false_type,std::false_type>),
    (std::tuple<test_config::config_storage_selector<test_storage_engine::minimal_storage>, std::false_type,std::false_type,std::false_type,std::false_type,std::true_type,std::false_type>)
)
{
    using value_type = int;
    using config_selector = std::tuple_element_t<0, TestType>;
    using config_type = gtensor::config::extend_config_t<typename config_selector::config_type,value_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using engine_type = gtensor::storage_engine<config_type,value_type>;
    using helpers_for_testing::apply_by_element;

    constexpr static bool has_iterator_expected = std::tuple_element_t<1, TestType>::value;
    constexpr static bool has_const_iterator_expected = std::tuple_element_t<2, TestType>::value;
    constexpr static bool has_reverse_iterator_expected = std::tuple_element_t<3, TestType>::value;
    constexpr static bool has_const_reverse_iterator_expected = std::tuple_element_t<4, TestType>::value;
    constexpr static bool has_subscript_operator_expected = std::tuple_element_t<5, TestType>::value;
    constexpr static bool has_subscript_operator_const_expected = std::tuple_element_t<6, TestType>::value;

    constexpr static bool has_iterator_result = gtensor::detail::has_iterator_v<engine_type>;
    constexpr static bool has_const_iterator_result = gtensor::detail::has_const_iterator_v<engine_type>;
    constexpr static bool has_reverse_iterator_result = gtensor::detail::has_reverse_iterator_v<engine_type>;
    constexpr static bool has_const_reverse_iterator_result = gtensor::detail::has_const_reverse_iterator_v<engine_type>;
    constexpr static bool has_subscript_operator_result = gtensor::detail::has_subscript_operator<engine_type>();
    constexpr static bool has_subscript_operator_const_result = gtensor::detail::has_subscript_operator_const<engine_type>();

    REQUIRE(gtensor::detail::has_descriptor_const<engine_type>::value);
    REQUIRE(has_iterator_result == has_iterator_expected);
    REQUIRE(has_const_iterator_result == has_const_iterator_expected);
    REQUIRE(has_reverse_iterator_result == has_reverse_iterator_expected);
    REQUIRE(has_const_reverse_iterator_result == has_const_reverse_iterator_expected);
    REQUIRE(has_subscript_operator_result == has_subscript_operator_expected);
    REQUIRE(has_subscript_operator_const_result == has_subscript_operator_const_expected);

    SECTION("test_storage_engine_size_value_constructor"){
        //0shape,1value,2expected_shape,3expected_elements
        auto test_data = std::make_tuple(
            std::make_tuple(shape_type{0},0,shape_type{0},std::vector<value_type>{}),
            std::make_tuple(shape_type{3,0},2,shape_type{3,0},std::vector<value_type>{}),
            std::make_tuple(shape_type{1},1,shape_type{1},std::vector<value_type>{1}),
            std::make_tuple(shape_type{5},2,shape_type{5},std::vector<value_type>{2,2,2,2,2}),
            std::make_tuple(shape_type{2,3},4,shape_type{2,3},std::vector<value_type>{4,4,4,4,4,4})
        );
        auto test = [](const auto& t){
            auto shape = std::get<0>(t);
            auto value = std::get<1>(t);
            auto expected_elements = std::get<3>(t);
            auto expected_shape = std::get<2>(t);
            engine_type engine(shape,value);
            auto result_shape = engine.descriptor().shape();
            REQUIRE(result_shape == expected_shape);
            if constexpr (has_iterator_expected || has_const_iterator_expected){
                std::vector<value_type> result_elements(engine.begin(),engine.end());
                REQUIRE(result_elements == expected_elements);
            }
            if constexpr (has_subscript_operator_expected || has_subscript_operator_const_expected){
                std::vector<value_type> result_elements{};
                const index_type result_size = engine.descriptor().size();
                for (index_type i{0}; i!=result_size; ++i){
                    result_elements.push_back(engine[i]);
                }
                REQUIRE(result_elements == expected_elements);
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_storage_engine_init_list_constructor"){
        //0engine,1expected_shape,2expected_elements
        auto test_data = std::make_tuple(
            std::make_tuple(engine_type(std::initializer_list<value_type>{}),shape_type{0},std::vector<value_type>{}),
            std::make_tuple(engine_type(std::initializer_list<value_type>{1}),shape_type{1},std::vector<value_type>{1}),
            std::make_tuple(engine_type(std::initializer_list<value_type>{1,2,3,4,5}),shape_type{5},std::vector<value_type>{1,2,3,4,5}),
            std::make_tuple(engine_type(std::initializer_list<std::initializer_list<value_type>>{{1,2,3},{4,5,6}}),shape_type{2,3},std::vector<value_type>{1,2,3,4,5,6}),
            std::make_tuple(
                engine_type(std::initializer_list<std::initializer_list<std::initializer_list<value_type>>>{{{1},{2},{3}},{{4},{5},{6}}}),
                shape_type{2,3,1},
                std::vector<value_type>{1,2,3,4,5,6}
            )
        );
        auto test = [](const auto& t){
            auto engine = std::get<0>(t);
            auto expected_shape = std::get<1>(t);
            auto expected_elements = std::get<2>(t);
            auto result_shape = engine.descriptor().shape();
            REQUIRE(result_shape == expected_shape);
            if constexpr (has_iterator_expected || has_const_iterator_expected){
                std::vector<value_type> result_elements(engine.begin(),engine.end());
                REQUIRE(result_elements == expected_elements);
            }
            if constexpr (has_subscript_operator_expected || has_subscript_operator_const_expected){
                std::vector<value_type> result_elements{};
                const index_type result_size = engine.descriptor().size();
                for (index_type i{0}; i!=result_size; ++i){
                    result_elements.push_back(engine[i]);
                }
                REQUIRE(result_elements == expected_elements);
            }
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_storage_engine_range_constructor"){
        //0shape,1elements,2expected_shape,3expected_elements
        auto test_data = std::make_tuple(
            std::make_tuple(shape_type{0},std::vector<value_type>{},shape_type{0},std::vector<value_type>{}),
            std::make_tuple(shape_type{0,3},std::vector<value_type>{},shape_type{0,3},std::vector<value_type>{}),
            std::make_tuple(shape_type{0},std::vector<value_type>{2,2,2,2},shape_type{0},std::vector<value_type>{}),
            std::make_tuple(shape_type{1},std::vector<value_type>{2},shape_type{1},std::vector<value_type>{2}),
            std::make_tuple(shape_type{1},std::vector<value_type>{3,4,5},shape_type{1},std::vector<value_type>{3}),
            std::make_tuple(shape_type{5},std::vector<value_type>{1,2,3,4,5},shape_type{5},std::vector<value_type>{1,2,3,4,5}),
            std::make_tuple(shape_type{5},std::vector<value_type>{1,2,3,4,5,6,7,8,9},shape_type{5},std::vector<value_type>{1,2,3,4,5}),
            std::make_tuple(shape_type{5},std::vector<value_type>{1,2,3},shape_type{5},std::vector<value_type>{1,2,3,0,0}),
            std::make_tuple(shape_type{3,2},std::vector<value_type>{1,2,3,4,5,6},shape_type{3,2},std::vector<value_type>{1,2,3,4,5,6}),
            std::make_tuple(shape_type{3,2},std::vector<value_type>{1,2},shape_type{3,2},std::vector<value_type>{1,2,0,0,0,0}),
            std::make_tuple(shape_type{3,2},std::vector<value_type>{1,2,3,4,5,6,7,8,9,10},shape_type{3,2},std::vector<value_type>{1,2,3,4,5,6})
        );
        auto test = [](const auto& t){
            auto shape = std::get<0>(t);
            auto elements = std::get<1>(t);
            auto expected_shape = std::get<2>(t);
            auto expected_elements = std::get<3>(t);
            engine_type engine(shape,elements.begin(),elements.end());
            auto result_shape = engine.descriptor().shape();
            REQUIRE(result_shape == expected_shape);
            if constexpr (has_iterator_expected || has_const_iterator_expected){
                std::vector<value_type> result_elements(engine.begin(),engine.end());
                REQUIRE(result_elements == expected_elements);
            }
            if constexpr (has_subscript_operator_expected || has_subscript_operator_const_expected){
                std::vector<value_type> result_elements{};
                const index_type result_size = engine.descriptor().size();
                for (index_type i{0}; i!=result_size; ++i){
                    result_elements.push_back(engine[i]);
                }
                REQUIRE(result_elements == expected_elements);
            }
        };
        apply_by_element(test,test_data);
    }
}
