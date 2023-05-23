#include <vector>
#include <string>
#include <list>
#include "catch.hpp"
#include "integral_type.hpp"
#include "tensor_implementation.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"

TEST_CASE("test_has_callable_iterator","[test_tensor_implementation]")
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

    using gtensor::detail::has_callable_iterator;
    REQUIRE(!has_callable_iterator<void>::value);
    REQUIRE(!has_callable_iterator<void*>::value);
    REQUIRE(!has_callable_iterator<int>::value);
    REQUIRE(!has_callable_iterator<int*>::value);
    REQUIRE(!has_callable_iterator<int[]>::value);
    REQUIRE(!has_callable_iterator<const non_const_iterable>::value);

    REQUIRE(has_callable_iterator<non_const_iterable>::value);
    REQUIRE(has_callable_iterator<const_iterable>::value);
    REQUIRE(has_callable_iterator<const const_iterable>::value);
    REQUIRE(has_callable_iterator<iterable>::value);
    REQUIRE(has_callable_iterator<const iterable>::value);
    REQUIRE(has_callable_iterator<std::list<int>>::value);
    REQUIRE(has_callable_iterator<std::vector<int>>::value);
    REQUIRE(has_callable_iterator<std::string>::value);
}

TEST_CASE("test_has_callable_reverse_iterator","[test_tensor_implementation]")
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

    using gtensor::detail::has_callable_reverse_iterator;
    REQUIRE(!has_callable_reverse_iterator<void>::value);
    REQUIRE(!has_callable_reverse_iterator<void*>::value);
    REQUIRE(!has_callable_reverse_iterator<int>::value);
    REQUIRE(!has_callable_reverse_iterator<int*>::value);
    REQUIRE(!has_callable_reverse_iterator<int[]>::value);
    REQUIRE(!has_callable_reverse_iterator<const non_const_reverse_iterable>::value);

    REQUIRE(has_callable_reverse_iterator<non_const_reverse_iterable>::value);
    REQUIRE(has_callable_reverse_iterator<const_reverse_iterable>::value);
    REQUIRE(has_callable_reverse_iterator<const const_reverse_iterable>::value);
    REQUIRE(has_callable_reverse_iterator<reverse_iterable>::value);
    REQUIRE(has_callable_reverse_iterator<const reverse_iterable>::value);
    REQUIRE(has_callable_reverse_iterator<std::list<int>>::value);
    REQUIRE(has_callable_reverse_iterator<std::vector<int>>::value);
    REQUIRE(has_callable_reverse_iterator<std::string>::value);
}

TEST_CASE("test_has_callable_subscript_operator","[test_tensor_implementation]")
{
    struct const_subscriptable{
        using size_type = std::size_t;
        int operator[](size_type)const;
    };
    struct non_const_subscriptable{
        using difference_type = std::ptrdiff_t;
        int& operator[](difference_type);
    };
    struct subscriptable{
        using size_type = std::size_t;
        const int& operator[](size_type)const;
        int& operator[](size_type);
    };

    using gtensor::detail::has_callable_subscript_operator;
    REQUIRE(!has_callable_subscript_operator<void>());
    REQUIRE(!has_callable_subscript_operator<void*>());
    REQUIRE(!has_callable_subscript_operator<int>());
    REQUIRE(!has_callable_subscript_operator<int*>());
    REQUIRE(!has_callable_subscript_operator<int[]>());
    REQUIRE(!has_callable_subscript_operator<std::list<int>>());
    REQUIRE(!has_callable_subscript_operator<const non_const_subscriptable>());

    REQUIRE(has_callable_subscript_operator<non_const_subscriptable>());
    REQUIRE(has_callable_subscript_operator<const_subscriptable>());
    REQUIRE(has_callable_subscript_operator<const const_subscriptable>());
    REQUIRE(has_callable_subscript_operator<subscriptable>());
    REQUIRE(has_callable_subscript_operator<const subscriptable>());
    REQUIRE(has_callable_subscript_operator<std::vector<int>>());
    REQUIRE(has_callable_subscript_operator<std::string>());
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
class subscriptable_storage_integral
{
    using inner_storage_type = std::vector<T>;
    using inner_size_type = typename inner_storage_type::size_type;
    using inner_difference_type = typename inner_storage_type::difference_type;
    inner_storage_type impl_;
public:
    using value_type = T;
    using size_type = integral_type::integral<inner_size_type>;
    using difference_type = integral_type::integral<inner_difference_type>;
    subscriptable_storage_integral(size_type n):
        impl_(n.value())
    {}
    template<typename It>
    subscriptable_storage_integral(It first, It last):
        impl_(first,last)
    {}
    size_type size()const{return impl_.size();}
    decltype(auto) operator[](size_type i){return impl_[i.value()];}
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
template<typename T>
class iterable_storage_integral
{
    using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<subscriptable_storage_integral>,T>;
    using inner_storage_type = typename config_type::template storage<T>;
    using indexer_type = gtensor::basic_indexer<inner_storage_type&>;
    using iterator = gtensor::indexer_iterator<config_type,indexer_type>;
    inner_storage_type impl_;
public:
    using value_type = T;
    using size_type = typename inner_storage_type::size_type;
    using difference_type = typename inner_storage_type::difference_type;
    iterable_storage_integral(size_type n):
        impl_(n)
    {}
    template<typename It>
    iterable_storage_integral(It first, It last):
        impl_(first,last)
    {}
    size_type size()const{return impl_.size();}
    iterator begin(){return iterator{indexer_type{impl_},0};}
    iterator end(){return iterator{indexer_type{impl_},size()};}
};
}   //end of namespace test_storage_core

TEMPLATE_TEST_CASE("test_storage_core","[test_tensor_implementation]",
    //0test_config,1is_iterable,2is_subscriptable
    (std::tuple<test_config::config_storage_selector_t<std::vector>, std::true_type, std::true_type>),
    (std::tuple<test_config::config_storage_selector_t<std::list>, std::true_type, std::false_type>),
    (std::tuple<test_config::config_storage_selector_t<test_storage_core::subscriptable_storage>, std::false_type, std::true_type>),
    (std::tuple<test_config::config_storage_selector_t<test_storage_core::subscriptable_storage_integral>, std::false_type, std::true_type>),
    (std::tuple<test_config::config_storage_selector_t<test_storage_core::iterable_storage>, std::true_type, std::false_type>),
    (std::tuple<test_config::config_storage_selector_t<test_storage_core::iterable_storage_integral>, std::true_type, std::false_type>)
)
{
    using value_type = int;
    using config_type = gtensor::config::extend_config_t<std::tuple_element_t<0,TestType>,value_type>;
    using is_iterable =std::tuple_element_t<1,TestType>;
    using is_subscriptable =std::tuple_element_t<2,TestType>;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using core_type = gtensor::storage_core<config_type,value_type>;
    using helpers_for_testing::apply_by_element;

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
    using test_core_base_type::elements_;
    decltype(auto) operator[](index_type i){return elements_[i];}
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
    using test_core_base_type::elements_;
    using test_core_base_type::descriptor;
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
    using test_core_base_type::elements_;
    decltype(auto) operator[](index_type i)const{return elements_[i];}
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
    using test_core_base_type::elements_;
    using test_core_base_type::descriptor;
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
    using test_core_base_type::elements_;
    auto begin()const{return elements_.begin();}
    auto end()const{return elements_.end();}
};

template<typename Config, typename T>
class test_core_full_subscriptable : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
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
template<typename Config, typename T>
class test_core_full_indexible : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
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
template<typename Config, typename T>
class test_core_full_walkable : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
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
template<typename Config, typename T>
class test_core_full_iterable : public test_core_base<Config,T>{
    using test_core_base_type = test_core_base<Config,T>;
public:
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

}   //end of namespace test_tensor_implementation_

TEMPLATE_TEST_CASE("test_tensor_implementation","[test_tensor_implementation]",
    //non const accessible core
    (test_tensor_implementation_::test_core_subscriptable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_subscriptable<test_config::config_storage_selector_t<test_storage_core::subscriptable_storage_integral>,int>),
    (test_tensor_implementation_::test_core_indexible<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_walkable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_iterable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_iterable<test_config::config_storage_selector_t<test_storage_core::iterable_storage_integral>,int>),
    //const accessible core
    (test_tensor_implementation_::test_core_const_subscriptable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_const_indexible<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_const_walkable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_const_iterable<test_config::config_storage_selector_t<std::vector>,int>),
    //full accessible core
    (test_tensor_implementation_::test_core_full_subscriptable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_full_indexible<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_full_walkable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_full_iterable<test_config::config_storage_selector_t<std::vector>,int>)
)
{
    using core_type = TestType;
    using config_type = typename core_type::config_type;
    using value_type = typename core_type::value_type;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using tensor_implementation_type = gtensor::tensor_implementation<core_type>;
    using helpers_for_testing::apply_by_element;

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

TEMPLATE_TEST_CASE("test_tensor_implementation_data_accesor_result_type_non_const_accessible_core","[test_tensor_implementation]",
    (test_tensor_implementation_::test_core_subscriptable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_subscriptable<test_config::config_storage_selector_t<test_storage_core::subscriptable_storage_integral>,int>),
    (test_tensor_implementation_::test_core_indexible<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_walkable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_iterable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_iterable<test_config::config_storage_selector_t<test_storage_core::iterable_storage_integral>,int>)
)
{
    using core_type = TestType;
    using config_type = typename core_type::config_type;
    using value_type = typename core_type::value_type;
    using index_type = typename config_type::index_type;
    using tensor_implementation_type = gtensor::tensor_implementation<core_type>;

    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().begin()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().end()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rbegin()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rend()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().create_walker()),value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().create_indexer()[std::declval<index_type>()]),value_type&>);
}

TEMPLATE_TEST_CASE("test_tensor_implementation_data_accesor_result_type_const_accessible_core","[test_tensor_implementation]",
    (test_tensor_implementation_::test_core_const_subscriptable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_const_indexible<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_const_walkable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_const_iterable<test_config::config_storage_selector_t<std::vector>,int>)
)
{
    using core_type = TestType;
    using config_type = typename core_type::config_type;
    using value_type = typename core_type::value_type;
    using index_type = typename config_type::index_type;
    using tensor_implementation_type = gtensor::tensor_implementation<core_type>;

    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().begin()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().end()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rbegin()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rend()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().create_walker()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().create_indexer()[std::declval<index_type>()]),const value_type&>);

    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().begin()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().end()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().rbegin()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().rend()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().create_walker()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor_implementation_type>().create_indexer()[std::declval<index_type>()]),const value_type&>);
}

TEMPLATE_TEST_CASE("test_tensor_implementation_data_accesor_result_type_full_accessible_core","[test_tensor_implementation]",
    (test_tensor_implementation_::test_core_full_subscriptable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_full_indexible<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_full_walkable<test_config::config_storage_selector_t<std::vector>,int>),
    (test_tensor_implementation_::test_core_full_iterable<test_config::config_storage_selector_t<std::vector>,int>)
)
{
    using core_type = TestType;
    using config_type = typename core_type::config_type;
    using value_type = typename core_type::value_type;
    using index_type = typename config_type::index_type;
    using tensor_implementation_type = gtensor::tensor_implementation<core_type>;

    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().begin()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().end()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rbegin()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().rend()),value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<tensor_implementation_type>().create_walker()),value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<tensor_implementation_type>().create_indexer()[std::declval<index_type>()]),value_type&>);

    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().begin()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().end()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().rbegin()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().rend()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(*std::declval<const tensor_implementation_type>().create_walker()),const value_type&>);
    REQUIRE(std::is_same_v<decltype(std::declval<const tensor_implementation_type>().create_indexer()[std::declval<index_type>()]),const value_type&>);
}

TEMPLATE_TEST_CASE("test_tensor_implementation_broadcast_iterator","[test_tensor_implementation]",
    test_config::config_storage_selector_t<std::vector>
)
{
    using value_type = double;
    using config_type = gtensor::config::extend_config_t<TestType,value_type>;
    using core_type = gtensor::storage_core<config_type,value_type>;
    using shape_type = typename config_type::shape_type;
    using tensor_implementation_type = gtensor::tensor_implementation<core_type>;
    using helpers_for_testing::apply_by_element;

    //0shape,1elements,2broadcast_shape,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{}, std::vector<value_type>{1}, shape_type{}, std::vector<value_type>{1}),
        std::make_tuple(shape_type{}, std::vector<value_type>{2}, shape_type{1}, std::vector<value_type>{2}),
        std::make_tuple(shape_type{}, std::vector<value_type>{3}, shape_type{5}, std::vector<value_type>{3,3,3,3,3}),
        std::make_tuple(shape_type{1}, std::vector<value_type>{2}, shape_type{1}, std::vector<value_type>{2}),
        std::make_tuple(shape_type{1}, std::vector<value_type>{1}, shape_type{5}, std::vector<value_type>{1,1,1,1,1}),
        std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{1}, std::vector<value_type>{1}),
        std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{1,6}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{6,1}, std::vector<value_type>{1,1,1,1,1,1}),
        std::make_tuple(shape_type{6}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{2,6}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
        std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::array<int,2>{2,3}, std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, std::list<std::size_t>{2,2,3}, std::vector<value_type>{1,2,3,4,5,6,1,2,3,4,5,6}),
        std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{1,3}, std::vector<value_type>{1,2,3}),
        std::make_tuple(shape_type{2,3}, std::vector<value_type>{1,2,3,4,5,6}, shape_type{2,1,3}, std::vector<value_type>{1,2,3,1,2,3})
    );

    SECTION("test_broadcast_iterator")
    {
        auto test = [](const auto& t){
            const auto shape = std::get<0>(t);
            const auto elements = std::get<1>(t);
            const auto broadcast_shape = std::get<2>(t);
            const auto expected = std::get<3>(t);
            tensor_implementation_type result_tensor_implementation{shape, elements.begin(), elements.end()};
            REQUIRE(std::equal(result_tensor_implementation.begin(broadcast_shape),result_tensor_implementation.end(broadcast_shape),expected.begin(),expected.end()));
            REQUIRE(std::equal(result_tensor_implementation.rbegin(broadcast_shape),result_tensor_implementation.rend(broadcast_shape),expected.rbegin(),expected.rend()));
        };
        apply_by_element(test,test_data);
    }
    SECTION("test_const_broadcast_iterator")
    {
        auto test = [](const auto& t){
            const auto shape = std::get<0>(t);
            const auto elements = std::get<1>(t);
            const auto broadcast_shape = std::get<2>(t);
            const auto expected = std::get<3>(t);
            const tensor_implementation_type result_tensor_implementation{shape, elements.begin(), elements.end()};
            REQUIRE(std::equal(result_tensor_implementation.begin(broadcast_shape),result_tensor_implementation.end(broadcast_shape),expected.begin(),expected.end()));
            REQUIRE(std::equal(result_tensor_implementation.rbegin(broadcast_shape),result_tensor_implementation.rend(broadcast_shape),expected.rbegin(),expected.rend()));
        };
        apply_by_element(test,test_data);
    }

}