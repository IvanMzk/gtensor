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
        using gtensor::detail::has_iterator;
        REQUIRE(!has_iterator<void>::value);
        REQUIRE(!has_iterator<void*>::value);
        REQUIRE(!has_iterator<int>::value);
        REQUIRE(!has_iterator<int*>::value);
        REQUIRE(!has_iterator<int[]>::value);
        REQUIRE(!has_iterator<const_iterable>::value);

        REQUIRE(has_iterator<iterable>::value);
        REQUIRE(has_iterator<non_const_iterable>::value);
        REQUIRE(has_iterator<std::list<int>>::value);
        REQUIRE(has_iterator<std::vector<int>>::value);
        REQUIRE(has_iterator<std::string>::value);
    }
    SECTION("test_has_const_iterator_v")
    {
        using gtensor::detail::has_const_iterator;
        REQUIRE(!has_const_iterator<void>::value);
        REQUIRE(!has_const_iterator<void*>::value);
        REQUIRE(!has_const_iterator<int>::value);
        REQUIRE(!has_const_iterator<int*>::value);
        REQUIRE(!has_const_iterator<int[]>::value);
        REQUIRE(!has_const_iterator<non_const_iterable>::value);

        REQUIRE(has_const_iterator<iterable>::value);
        REQUIRE(has_const_iterator<const_iterable>::value);
        REQUIRE(has_const_iterator<std::list<int>>::value);
        REQUIRE(has_const_iterator<std::vector<int>>::value);
        REQUIRE(has_const_iterator<std::string>::value);
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
        using gtensor::detail::has_reverse_iterator;
        REQUIRE(!has_reverse_iterator<void>::value);
        REQUIRE(!has_reverse_iterator<void*>::value);
        REQUIRE(!has_reverse_iterator<int>::value);
        REQUIRE(!has_reverse_iterator<int*>::value);
        REQUIRE(!has_reverse_iterator<int[]>::value);
        REQUIRE(!has_reverse_iterator<const_reverse_iterable>::value);

        REQUIRE(has_reverse_iterator<reverse_iterable>::value);
        REQUIRE(has_reverse_iterator<non_const_reverse_iterable>::value);
        REQUIRE(has_reverse_iterator<std::list<int>>::value);
        REQUIRE(has_reverse_iterator<std::vector<int>>::value);
        REQUIRE(has_reverse_iterator<std::string>::value);
    }
    SECTION("test_has_const_reverse_iterator_v")
    {
        using gtensor::detail::has_const_reverse_iterator;
        REQUIRE(!has_const_reverse_iterator<void>::value);
        REQUIRE(!has_const_reverse_iterator<void*>::value);
        REQUIRE(!has_const_reverse_iterator<int>::value);
        REQUIRE(!has_const_reverse_iterator<int*>::value);
        REQUIRE(!has_const_reverse_iterator<int[]>::value);
        REQUIRE(!has_const_reverse_iterator<non_const_reverse_iterable>::value);

        REQUIRE(has_const_reverse_iterator<reverse_iterable>::value);
        REQUIRE(has_const_reverse_iterator<const_reverse_iterable>::value);
        REQUIRE(has_const_reverse_iterator<std::list<int>>::value);
        REQUIRE(has_const_reverse_iterator<std::vector<int>>::value);
        REQUIRE(has_const_reverse_iterator<std::string>::value);
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

namespace test_storage_core{
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
}   //end of namespace test_storage_core

TEMPLATE_TEST_CASE("test_storage_core","[test_tensor_implementation]",
    //0config_selector,1has_iter,2has_const_iter,3has_reverse_iter,4has_const_reverse_iter,5has_subscript_operator,6has_subscript_operator_const
    (std::tuple<test_config::config_storage_selector<std::vector>, std::true_type,std::true_type,std::true_type,std::true_type,std::true_type,std::true_type>),
    (std::tuple<test_config::config_storage_selector<std::list>, std::true_type,std::true_type,std::true_type,std::true_type,std::false_type,std::false_type>),
    (std::tuple<test_config::config_storage_selector<test_storage_core::minimal_storage>, std::false_type,std::false_type,std::false_type,std::false_type,std::true_type,std::false_type>)
)
{
    using value_type = int;
    using config_selector = std::tuple_element_t<0, TestType>;
    using config_type = gtensor::config::extend_config_t<typename config_selector::config_type,value_type>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using core_type = gtensor::storage_core<config_type,value_type>;
    using helpers_for_testing::apply_by_element;

    constexpr static bool has_iterator_expected = std::tuple_element_t<1, TestType>::value;
    constexpr static bool has_const_iterator_expected = std::tuple_element_t<2, TestType>::value;
    constexpr static bool has_reverse_iterator_expected = std::tuple_element_t<3, TestType>::value;
    constexpr static bool has_const_reverse_iterator_expected = std::tuple_element_t<4, TestType>::value;
    constexpr static bool has_subscript_operator_expected = std::tuple_element_t<5, TestType>::value;
    constexpr static bool has_subscript_operator_const_expected = std::tuple_element_t<6, TestType>::value;

    constexpr static bool has_iterator_result = gtensor::detail::has_iterator<core_type>::value;
    constexpr static bool has_const_iterator_result = gtensor::detail::has_const_iterator<core_type>::value;
    constexpr static bool has_reverse_iterator_result = gtensor::detail::has_reverse_iterator<core_type>::value;
    constexpr static bool has_const_reverse_iterator_result = gtensor::detail::has_const_reverse_iterator<core_type>::value;
    constexpr static bool has_subscript_operator_result = gtensor::detail::has_subscript_operator<core_type>::value;
    constexpr static bool has_subscript_operator_const_result = gtensor::detail::has_subscript_operator_const<core_type>::value;

    REQUIRE(gtensor::detail::has_descriptor_const<core_type>::value);
    REQUIRE(has_iterator_result == has_iterator_expected);
    REQUIRE(has_const_iterator_result == has_const_iterator_expected);
    REQUIRE(has_reverse_iterator_result == has_reverse_iterator_expected);
    REQUIRE(has_const_reverse_iterator_result == has_const_reverse_iterator_expected);
    REQUIRE(has_subscript_operator_result == has_subscript_operator_expected);
    REQUIRE(has_subscript_operator_const_result == has_subscript_operator_const_expected);

    SECTION("test_storage_core_shape_value_constructor"){
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
            core_type core(shape,value);
            auto result_shape = core.descriptor().shape();
            REQUIRE(result_shape == expected_shape);
            if constexpr (has_iterator_expected || has_const_iterator_expected){
                std::vector<value_type> result_elements(core.begin(),core.end());
                REQUIRE(result_elements == expected_elements);
            }
            if constexpr (has_subscript_operator_expected || has_subscript_operator_const_expected){
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
    SECTION("test_storage_core_init_list_constructor"){
        //0core,1expected_shape,2expected_elements
        auto test_data = std::make_tuple(
            std::make_tuple(core_type(std::initializer_list<value_type>{}),shape_type{0},std::vector<value_type>{}),
            std::make_tuple(core_type(std::initializer_list<value_type>{1}),shape_type{1},std::vector<value_type>{1}),
            std::make_tuple(core_type(std::initializer_list<value_type>{1,2,3,4,5}),shape_type{5},std::vector<value_type>{1,2,3,4,5}),
            std::make_tuple(core_type(std::initializer_list<std::initializer_list<value_type>>{{1,2,3},{4,5,6}}),shape_type{2,3},std::vector<value_type>{1,2,3,4,5,6}),
            std::make_tuple(
                core_type(std::initializer_list<std::initializer_list<std::initializer_list<value_type>>>{{{1},{2},{3}},{{4},{5},{6}}}),
                shape_type{2,3,1},
                std::vector<value_type>{1,2,3,4,5,6}
            )
        );
        auto test = [](const auto& t){
            auto core = std::get<0>(t);
            auto expected_shape = std::get<1>(t);
            auto expected_elements = std::get<2>(t);
            auto result_shape = core.descriptor().shape();
            REQUIRE(result_shape == expected_shape);
            if constexpr (has_iterator_expected || has_const_iterator_expected){
                std::vector<value_type> result_elements(core.begin(),core.end());
                REQUIRE(result_elements == expected_elements);
            }
            if constexpr (has_subscript_operator_expected || has_subscript_operator_const_expected){
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
            core_type core(shape,elements.begin(),elements.end());
            auto result_shape = core.descriptor().shape();
            REQUIRE(result_shape == expected_shape);
            if constexpr (has_iterator_expected || has_const_iterator_expected){
                std::vector<value_type> result_elements(core.begin(),core.end());
                REQUIRE(result_elements == expected_elements);
            }
            if constexpr (has_subscript_operator_expected || has_subscript_operator_const_expected){
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

namespace test_tensor_implementation_{

template<typename Config, typename T>
class test_core_base{
protected:
    using extended_config_type = gtensor::config::extend_config_t<Config,T>;
    using descriptor_type = gtensor::basic_descriptor<extended_config_type>;
    using storage_type = typename extended_config_type::template storage<T>;
public:
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

template<typename Config, typename T>
class test_core_subscriptable : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::descriptor;
    using test_core_base_type::elements_;
    gtensor::detail::subscript_operator_result_t<storage_type> operator[](index_type i){return elements_[i];}
};
template<typename Config, typename T>
class test_core_indexible : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::descriptor;
    using test_core_base_type::elements_;
    auto create_indexer(){return gtensor::basic_indexer<storage_type&>{elements_};}
};
template<typename Config, typename T>
class test_core_walkable : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::descriptor;
    using test_core_base_type::elements_;
    auto create_walker(dim_type max_dim){
        using indexer_type = gtensor::basic_indexer<storage_type&>;
        return gtensor::walker<config_type,indexer_type>{descriptor().adapted_strides(),descriptor().reset_strides(),descriptor().offset(),indexer_type{elements_},max_dim};
    }
};
template<typename Config, typename T>
class test_core_iterable : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::descriptor;
    using test_core_base_type::elements_;
    auto begin(){return elements_.begin();}
    auto end(){return elements_.end();}
};

template<typename Config, typename T>
class test_core_const_subscriptable : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::descriptor;
    using test_core_base_type::elements_;
    gtensor::detail::subscript_operator_const_result_t<storage_type> operator[](index_type i)const{return elements_[i];}
};
template<typename Config, typename T>
class test_core_const_indexible : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::descriptor;
    using test_core_base_type::elements_;
    auto create_indexer()const{return gtensor::basic_indexer<const storage_type&>{elements_};}
};
template<typename Config, typename T>
class test_core_const_walkable : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::descriptor;
    using test_core_base_type::elements_;
    auto create_walker(dim_type max_dim)const{
        using indexer_type = gtensor::basic_indexer<const storage_type&>;
        return gtensor::walker<config_type,indexer_type>{descriptor().adapted_strides(),descriptor().reset_strides(),descriptor().offset(),indexer_type{elements_},max_dim};
    }
};
template<typename Config, typename T>
class test_core_const_iterable : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
    using typename test_core_base_type::config_type;
    using typename test_core_base_type::value_type;
    using typename test_core_base_type::shape_type;
    using typename test_core_base_type::dim_type;
    using typename test_core_base_type::index_type;
    using typename test_core_base_type::storage_type;
    using test_core_base_type::test_core_base_type;
    using test_core_base_type::descriptor;
    using test_core_base_type::elements_;
    auto begin()const{return elements_.begin();}
    auto end()const{return elements_.end();}
};

}   //end of namespace test_tensor_implementation_

TEMPLATE_TEST_CASE("test_tensor_implementation","[test_tensor_implementation]",
    //0config_selector,1has_data_accessor,2has_const_data_accessor
    (std::tuple<test_tensor_implementation_::test_core_subscriptable<test_config::config_storage_selector<std::vector>::config_type,int> ,std::true_type,std::false_type>),
    (std::tuple<test_tensor_implementation_::test_core_indexible<test_config::config_storage_selector<std::vector>::config_type,int> ,std::true_type,std::false_type>),
    (std::tuple<test_tensor_implementation_::test_core_walkable<test_config::config_storage_selector<std::vector>::config_type,int> ,std::true_type,std::false_type>),
    (std::tuple<test_tensor_implementation_::test_core_iterable<test_config::config_storage_selector<std::vector>::config_type,int> ,std::true_type,std::false_type>),

    (std::tuple<test_tensor_implementation_::test_core_const_subscriptable<test_config::config_storage_selector<std::vector>::config_type,int> ,std::false_type,std::true_type>),
    (std::tuple<test_tensor_implementation_::test_core_const_indexible<test_config::config_storage_selector<std::vector>::config_type,int> ,std::false_type,std::true_type>),
    (std::tuple<test_tensor_implementation_::test_core_const_walkable<test_config::config_storage_selector<std::vector>::config_type,int> ,std::false_type,std::true_type>),
    (std::tuple<test_tensor_implementation_::test_core_const_iterable<test_config::config_storage_selector<std::vector>::config_type,int> ,std::false_type,std::true_type>)
)
{
    using core_type = std::tuple_element_t<0, TestType>;
    using config_type = typename core_type::config_type;
    using value_type = typename core_type::value_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using tensor_implementation_type = gtensor::tensor_implementation<core_type>;
    using helpers_for_testing::apply_by_element;

    constexpr static bool has_data_accessor_v = std::tuple_element_t<1, TestType>::value;
    constexpr static bool has_const_data_accessor_v = std::tuple_element_t<2, TestType>::value;

    REQUIRE(has_data_accessor_v == gtensor::detail::has_iterator<tensor_implementation_type>::value);
    REQUIRE(has_data_accessor_v == gtensor::detail::has_reverse_iterator<tensor_implementation_type>::value);
    REQUIRE(has_data_accessor_v == gtensor::detail::has_create_indexer<tensor_implementation_type>::value);
    REQUIRE(has_data_accessor_v == gtensor::detail::has_create_walker<tensor_implementation_type>::value);

    REQUIRE(has_const_data_accessor_v == gtensor::detail::has_const_iterator<tensor_implementation_type>::value);
    REQUIRE(has_const_data_accessor_v == gtensor::detail::has_const_reverse_iterator<tensor_implementation_type>::value);
    REQUIRE(has_const_data_accessor_v == gtensor::detail::has_create_indexer_const<tensor_implementation_type>::value);
    REQUIRE(has_const_data_accessor_v == gtensor::detail::has_create_walker_const<tensor_implementation_type>::value);
    //0shape,1expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{}, std::vector<value_type>{1}),
        std::make_tuple(shape_type{}, std::vector<value_type>{2}),
        std::make_tuple(shape_type{0}, std::vector<value_type>{}),
        std::make_tuple(shape_type{5}, std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(shape_type{2,2,2}, std::vector<value_type>{1,2,3,4,5,6,7,8})
    );

    SECTION("test_iterator")
    {
        auto test = [](const auto& t){
            const auto shape = std::get<0>(t);
            const auto expected_elements = std::get<1>(t);
            const auto expected_shape = shape;
            tensor_implementation_type result_tensor_implementation{shape, expected_elements.begin(), expected_elements.end()};
            auto result_shape = result_tensor_implementation.shape();
            REQUIRE(result_shape == expected_shape);
            REQUIRE(std::equal(result_tensor_implementation.begin(),result_tensor_implementation.end(),expected_elements.begin(),expected_elements.end()));
            REQUIRE(std::equal(result_tensor_implementation.rbegin(),result_tensor_implementation.rend(),expected_elements.rbegin(),expected_elements.rend()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_indexer")
    {
        auto test = [](const auto& t){
            const auto shape = std::get<0>(t);
            const auto expected_elements = std::get<1>(t);
            const auto expected_shape = shape;
            tensor_implementation_type result_tensor_implementation{shape, expected_elements.begin(), expected_elements.end()};
            auto result_shape = result_tensor_implementation.shape();
            REQUIRE(result_shape == expected_shape);
            using indexer_iterator_type = gtensor::indexer_iterator<config_type,decltype(result_tensor_implementation.create_indexer())>;
            indexer_iterator_type result_first{
                result_tensor_implementation.create_indexer(),
                index_type{0}
            };
            indexer_iterator_type result_last{
                result_tensor_implementation.create_indexer(),
                result_tensor_implementation.size()
            };
            REQUIRE(std::equal(result_first,result_last,expected_elements.begin(),expected_elements.end()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_walker")
    {
        auto test = [](const auto& t){
            const auto shape = std::get<0>(t);
            const auto expected_elements = std::get<1>(t);
            const auto expected_shape = shape;
            tensor_implementation_type result_tensor_implementation{shape, expected_elements.begin(), expected_elements.end()};
            auto result_shape = result_tensor_implementation.shape();
            REQUIRE(result_shape == expected_shape);
            using walker_iterator_type = gtensor::walker_iterator<config_type,decltype(result_tensor_implementation.create_walker())>;
            walker_iterator_type result_first{
                result_tensor_implementation.create_walker(),
                result_tensor_implementation.shape(),
                result_tensor_implementation.descriptor().strides_div(),
                index_type{0}
            };
            walker_iterator_type result_last{
                result_tensor_implementation.create_walker(),
                result_tensor_implementation.shape(),
                result_tensor_implementation.descriptor().strides_div(),
                result_tensor_implementation.size()
            };
            REQUIRE(std::equal(result_first,result_last,expected_elements.begin(),expected_elements.end()));
        };
        apply_by_element(test,test_data);
    }
}