#include <vector>
#include <list>
#include <tuple>
#include "catch.hpp"
#include "storage.hpp"
#include "helpers_for_testing.hpp"


TEST_CASE("test_default_constructor","[test_basic_storage]")
{
    using value_type = double;
    using storage_type = gtensor::basic_storage<value_type>;
    storage_type s{};
    REQUIRE(s.empty());
    REQUIRE(s.size() == 0);
    REQUIRE(s.begin() == s.end());
    REQUIRE(s.rbegin() == s.rend());
    REQUIRE(static_cast<const storage_type&>(s).begin() == static_cast<const storage_type&>(s).end());
    REQUIRE(static_cast<const storage_type&>(s).rbegin() == static_cast<const storage_type&>(s).rend());
}

TEMPLATE_TEST_CASE("test_n_constructor_size","[test_basic_storage]",
    double,
    std::string
)
{
    using value_type = TestType;
    using storage_type = gtensor::basic_storage<value_type>;
    using helpers_for_testing::apply_by_element;
    //0n,1expected_size,2expected_empty
    auto test_data = std::make_tuple(
        std::make_tuple(0,0,true),
        std::make_tuple(1,1,false),
        std::make_tuple(10,10,false),
        std::make_tuple(120,120,false)
    );

    auto test = [](const auto& t){
        auto n = std::get<0>(t);
        auto expected_size = std::get<1>(t);
        auto expected_empty = std::get<2>(t);

        storage_type stor(n);
        auto result_size = stor.size();
        auto result_empty = stor.empty();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_empty == expected_empty);
        REQUIRE(std::distance(stor.begin(),stor.end()) == expected_size);
        REQUIRE(std::distance(stor.rbegin(),stor.rend()) == expected_size);
    };
    apply_by_element(test,test_data);
}

namespace test_basic_storage{

template<typename T>
struct not_trivial{
    T t;
    virtual ~not_trivial(){}
    not_trivial() = default;
    not_trivial(T t_):
        t{t_}
    {}
    bool operator==(const not_trivial& other)const{
        return t==other.t;
    }
};

}


TEST_CASE("test_n_constructor_elements","[test_basic_storage]")
{
    using value_type = test_basic_storage::not_trivial<double>;
    using storage_type = gtensor::basic_storage<value_type>;
    using helpers_for_testing::apply_by_element;
    REQUIRE(!std::is_trivially_copyable_v<value_type>);
    //0n,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(0,std::vector<value_type>{}),
        std::make_tuple(1,std::vector<value_type>(1)),
        std::make_tuple(5,std::vector<value_type>(5)),
        std::make_tuple(10,std::vector<value_type>(10))
    );

    auto test = [](const auto& t){
        auto n = std::get<0>(t);
        auto expected = std::get<1>(t);

        storage_type stor(n);
        REQUIRE(std::equal(stor.begin(),stor.end(),expected.begin(),expected.end()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_n_value_constructor","[test_basic_storage]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using storage_type = gtensor::basic_storage<value_type>;
    using helpers_for_testing::apply_by_element;
    //0n,1value,2expected_size,3expected_empty,4expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(0,value_type{2},0,true,std::vector<value_type>{}),
        std::make_tuple(1,value_type{3},1,false,std::vector<value_type>{value_type{3}}),
        std::make_tuple(5,value_type{4},5,false,std::vector<value_type>{value_type{4},value_type{4},value_type{4},value_type{4},value_type{4}})
    );

    auto test = [](const auto& t){
        auto n = std::get<0>(t);
        auto value = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_empty = std::get<3>(t);
        auto expected_elements = std::get<4>(t);

        storage_type stor(n,value);
        auto result_size = stor.size();
        auto result_empty = stor.empty();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_empty == expected_empty);
        REQUIRE(std::equal(stor.begin(),stor.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(stor.rbegin(),stor.rend(),expected_elements.rbegin(),expected_elements.rend()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_iterators_range_constructor","[test_basic_storage]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using storage_type = gtensor::basic_storage<value_type>;
    using helpers_for_testing::apply_by_element;
    //0elements,1expected_size,2expected_empty,3expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(std::vector<value_type>{},0,true,std::vector<value_type>{}),
        std::make_tuple(std::vector<value_type>{value_type{3}},1,false,std::vector<value_type>{value_type{3}}),
        std::make_tuple(
            std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{5},value_type{6}},
            6,
            false,
            std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{5},value_type{6}}
        ),
        std::make_tuple(
            std::list<value_type>{value_type{7},value_type{8},value_type{9},value_type{10},value_type{11}},
            5,
            false,
            std::vector<value_type>{value_type{7},value_type{8},value_type{9},value_type{10},value_type{11}}
        )
    );

    auto test = [](const auto& t){
        auto elements = std::get<0>(t);
        auto expected_size = std::get<1>(t);
        auto expected_empty = std::get<2>(t);
        auto expected_elements = std::get<3>(t);

        storage_type stor(elements.begin(),elements.end());
        auto result_size = stor.size();
        auto result_empty = stor.empty();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_empty == expected_empty);
        REQUIRE(std::equal(stor.begin(),stor.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(stor.rbegin(),stor.rend(),expected_elements.rbegin(),expected_elements.rend()));
    };
    apply_by_element(test,test_data);
}