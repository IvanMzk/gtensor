#include <vector>
#include <list>
#include <tuple>
#include <iostream>
#include "catch.hpp"
#include "storage.hpp"
#include "helpers_for_testing.hpp"


TEST_CASE("test_basic_storage_default_constructor","[test_basic_storage]")
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

TEMPLATE_TEST_CASE("test_basic_storage_n_constructor_size","[test_basic_storage]",
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
struct counter{
    inline static int dtr_counter{0};
    inline static int ctr_counter{0};
    virtual ~counter()
    {++dtr_counter;}
    counter()
    {++ctr_counter;}
    counter(const counter&)
    {++ctr_counter;}
    counter(counter&&)
    {++ctr_counter;}
    counter& operator=(const counter&) = default;
    counter& operator=(counter&&) = default;
};

template<typename T>
struct not_trivial : counter<not_trivial<T>>{
    T t;
    not_trivial() = default;
    not_trivial(T t_):
        t{t_}
    {}
    bool operator==(const not_trivial& other)const{
        return t==other.t;
    }
};

template<typename T>
struct test_allocator : public std::allocator<T>
{
    using base_type = std::allocator<T>;
    using pointer = typename std::allocator_traits<base_type>::pointer;
    using typename base_type::size_type;
    using is_always_equal = std::false_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::false_type;

    inline static int alloc_counter{0};
    inline static int dealloc_counter{0};

    int state_{};

    using base_type::base_type;
    test_allocator() = default;
    test_allocator(int state__):
        state_{state__}
    {}

    pointer allocate(size_type n){
        ++alloc_counter;
        return base_type::allocate(n);
    }
    void deallocate(pointer p, size_type n){
        ++dealloc_counter;
        base_type::deallocate(p,n);
    }

    bool operator==(const test_allocator& other){
        return state_==other.state_;
    }
    bool operator!=(const test_allocator& other){
        return state_!=other.state_;
    }
};

}

TEST_CASE("test_basic_storage_n_constructor_elements","[test_basic_storage]")
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

TEMPLATE_TEST_CASE("test_basic_storage_n_value_constructor","[test_basic_storage]",
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

TEMPLATE_TEST_CASE("test_basic_storage_iterators_range_constructor","[test_basic_storage]",
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

TEMPLATE_TEST_CASE("test_basic_storage_initializer_list_constructor","[test_basic_storage]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using storage_type = gtensor::basic_storage<value_type>;

    storage_type stor{value_type{1},value_type{2},value_type{3},value_type{4},value_type{5}};
    std::vector<value_type> expected_elements{value_type{1},value_type{2},value_type{3},value_type{4},value_type{5}};
    REQUIRE(stor.size() == 5);
    REQUIRE(stor.empty() == false);
    REQUIRE(std::equal(stor.begin(),stor.end(),expected_elements.begin(),expected_elements.end()));
    REQUIRE(std::equal(stor.rbegin(),stor.rend(),expected_elements.rbegin(),expected_elements.rend()));
}

TEMPLATE_TEST_CASE("test_basic_storage_copy_constructor","[test_basic_storage]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using storage_type = gtensor::basic_storage<value_type>;

    storage_type stor{value_type{1},value_type{2},value_type{3},value_type{4},value_type{5}};
    storage_type stor_copy{stor};

    REQUIRE(stor.size() == 5);
    REQUIRE(stor.empty() == false);
    REQUIRE(stor_copy.size() == 5);
    REQUIRE(stor_copy.empty() == false);
    REQUIRE(std::equal(stor.begin(),stor.end(),stor_copy.begin(),stor_copy.end()));
    REQUIRE(std::equal(stor.rbegin(),stor.rend(),stor_copy.rbegin(),stor_copy.rend()));
}

TEMPLATE_TEST_CASE("test_basic_storage_move_constructor","[test_basic_storage]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using storage_type = gtensor::basic_storage<value_type>;

    storage_type stor{value_type{1},value_type{2},value_type{3},value_type{4},value_type{5}};
    storage_type stor_copy{stor};
    storage_type stor_move{std::move(stor)};

    REQUIRE(stor.size() == 0);
    REQUIRE(stor.empty() == true);
    REQUIRE(stor_move.size() == 5);
    REQUIRE(stor_move.empty() == false);
    REQUIRE(std::equal(stor_move.begin(),stor_move.end(),stor_copy.begin(),stor_copy.end()));
    REQUIRE(std::equal(stor_move.rbegin(),stor_move.rend(),stor_copy.rbegin(),stor_copy.rend()));
}

TEST_CASE("test_basic_storage_operator==","[test_basic_storage]")
{
    using value_type = double;
    using storage_type = gtensor::basic_storage<value_type>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(storage_type{},storage_type{},true),
        std::make_tuple(storage_type{1},storage_type{1},true),
        std::make_tuple(storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},true),
        std::make_tuple(storage_type{},storage_type{1,2,3,4,5},false),
        std::make_tuple(storage_type{1,2,3,4,5},storage_type{},false),
        std::make_tuple(storage_type{6,7,8},storage_type{1,2,3,4,5},false),
        std::make_tuple(storage_type{1,2,3,4,5},storage_type{6,7,8},false),
        std::make_tuple(storage_type{1,2,3,4,5},storage_type{1,2,0,4,5},false)
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto result = lhs==rhs;
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_basic_storage_copy_assign","[test_basic_storage]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using storage_type = gtensor::basic_storage<value_type>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        //always_equal_allocator
        std::make_tuple(storage_type{},storage_type{},storage_type{},storage_type{}),
        std::make_tuple(storage_type{},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5}),
        std::make_tuple(storage_type{6,7,8,9,10},storage_type{},storage_type{},storage_type{}),
        std::make_tuple(storage_type{6,7,8},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5}),
        std::make_tuple(storage_type{6,7,8,9,10},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5}),
        std::make_tuple(storage_type{6,7,8,9,10,12,13},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5})
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);

        auto& result = lhs=rhs;
        REQUIRE(&result == &lhs);
        REQUIRE(lhs == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_basic_storage_copy_assign_propagate_allocator","[test_basic_storage]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using allocator_type = test_basic_storage::test_allocator<value_type>;
    using storage_type = gtensor::basic_storage<value_type,allocator_type>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected_lhs,3expected_rhs,4expected_allocator
    auto test_data = std::make_tuple(
        //always_equal_allocator
        std::make_tuple(storage_type(allocator_type{1}),storage_type{allocator_type{2}},storage_type{allocator_type{2}},storage_type{allocator_type{2}},allocator_type{2}),
        std::make_tuple(storage_type(allocator_type{1}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{2}),allocator_type{2}),
        std::make_tuple(storage_type({6,7,8,9,10},allocator_type{1}),storage_type({},allocator_type{2}),storage_type({},allocator_type{2}),storage_type({},allocator_type{2}),allocator_type{2}),
        std::make_tuple(storage_type({6,7,8,9,10},allocator_type{1}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{2}),allocator_type{2}),
        std::make_tuple(storage_type({6,7,8},allocator_type{1}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{2}),allocator_type{2}),
        std::make_tuple(storage_type({6,7,8,9,10,11,12},allocator_type{1}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{2}),allocator_type{2})
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);
        auto expected_allocator = std::get<4>(t);

        auto& result = lhs=rhs;
        REQUIRE(&result == &lhs);
        REQUIRE(lhs == expected_lhs);
        REQUIRE(rhs == expected_rhs);
        REQUIRE(lhs.get_allocator() == expected_allocator);
        REQUIRE(rhs.get_allocator() == expected_allocator);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_basic_storage_move_assign","[test_basic_storage]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using storage_type = gtensor::basic_storage<value_type>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected_lhs,3expected_rhs_empty,4expected_rhs_size
    auto test_data = std::make_tuple(
        std::make_tuple(storage_type{},storage_type{},storage_type{},true,0),
        std::make_tuple(storage_type{},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},true,0),
        std::make_tuple(storage_type{6,7,8,9,10},storage_type{},storage_type{},true,0),
        std::make_tuple(storage_type{6,7,8},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},true,0),
        std::make_tuple(storage_type{6,7,8,9,10},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},true,0),
        std::make_tuple(storage_type{6,7,8,9,10,12,13},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},true,0)
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs_empty = std::get<3>(t);
        auto expected_rhs_size = std::get<4>(t);

        auto& result = lhs=std::move(rhs);
        REQUIRE(&result == &lhs);
        REQUIRE(lhs == expected_lhs);
        REQUIRE(rhs.empty() == expected_rhs_empty);
        REQUIRE(rhs.size() == expected_rhs_size);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_basic_storage_move_assign_not_propagate_allocator","[test_basic_storage]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using allocator_type = test_basic_storage::test_allocator<value_type>;
    using storage_type = gtensor::basic_storage<value_type,allocator_type>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected_lhs,3expected_lhs_allocator,4expected_rhs_empty,5expected_rhs_size
    auto test_data = std::make_tuple(
        std::make_tuple(storage_type(allocator_type{1}),storage_type(allocator_type{2}),storage_type(allocator_type{1}),allocator_type{1},true,0),
        std::make_tuple(storage_type({6,7,8,9,10},allocator_type{1}),storage_type(allocator_type{2}),storage_type(allocator_type{1}),allocator_type{1},true,0),
        std::make_tuple(storage_type({6,7,8,9,10},allocator_type{1}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{1}),allocator_type{1},false,5),
        std::make_tuple(storage_type({6,7,8},allocator_type{1}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{1}),allocator_type{1},false,5),
        std::make_tuple(storage_type({6,7,8,9,10,11,12},allocator_type{1}),storage_type({1,2,3,4,5},allocator_type{2}),storage_type({1,2,3,4,5},allocator_type{1}),allocator_type{1},false,5)
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_lhs_allocator = std::get<3>(t);
        auto expected_rhs_empty = std::get<4>(t);
        auto expected_rhs_size = std::get<5>(t);

        auto& result = lhs=std::move(rhs);
        REQUIRE(&result == &lhs);
        REQUIRE(lhs == expected_lhs);
        REQUIRE(lhs.get_allocator() == expected_lhs_allocator);
        REQUIRE(rhs.empty() == expected_rhs_empty);
        REQUIRE(rhs.size() == expected_rhs_size);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_basic_storage_swap","[test_basic_storage]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using storage_type = gtensor::basic_storage<value_type>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected_lhs,3expected_rhs
    auto test_data = std::make_tuple(
        std::make_tuple(storage_type{},storage_type{},storage_type{},storage_type{}),
        std::make_tuple(storage_type{},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},storage_type{}),
        std::make_tuple(storage_type{6,7,8,9,10},storage_type{},storage_type{},storage_type{6,7,8,9,10}),
        std::make_tuple(storage_type{6,7,8,9,10},storage_type{1,2,3,4,5},storage_type{1,2,3,4,5},storage_type{6,7,8,9,10})
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs = std::get<2>(t);
        auto expected_rhs = std::get<3>(t);

        swap(lhs,rhs);
        REQUIRE(lhs == expected_lhs);
        REQUIRE(rhs == expected_rhs);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_basic_storage_destructor","[test_basic_storage]")
{
    using value_type = test_basic_storage::not_trivial<double>;
    using allocator_type = test_basic_storage::test_allocator<value_type>;
    using storage_type = gtensor::basic_storage<value_type,allocator_type>;
    value_type::ctr_counter = 0;
    value_type::dtr_counter = 0;
    allocator_type::alloc_counter = 0;
    allocator_type::dealloc_counter = 0;

    {
        storage_type stor{};
    }
    REQUIRE(value_type::ctr_counter == 0);
    REQUIRE(value_type::dtr_counter == 0);
    REQUIRE(allocator_type::alloc_counter == 0);
    REQUIRE(allocator_type::dealloc_counter == 0);
    {
        storage_type stor{1,2,3,4,5};
    }
    REQUIRE(value_type::ctr_counter >= 5);
    REQUIRE(value_type::ctr_counter == value_type::dtr_counter);
    REQUIRE(allocator_type::alloc_counter == 1);
    REQUIRE(allocator_type::dealloc_counter == 1);
}



// TEST_CASE("test_std_vector_alloc_propagating","[test_std_vector_alloc_propagating]")
// {
//     using test_basic_storage::test_allocator;
//     using test_basic_storage::test_allocator1;
//     using vector_type = std::vector<int,test_allocator<int>>;

//     vector_type vec{1,2,3,4,5,6,7,8,9,10};
//     std::cout<<std::endl<<vec.size()<<" "<<vec.capacity()<<" "<<vec.data()<<" "<<vec.get_allocator().state_;
//     vector_type vec1(3,0,test_allocator<int>{1});
//     std::cout<<std::endl<<vec1.size()<<" "<<vec1.capacity()<<" "<<vec1.data()<<" "<<vec1.get_allocator().state_;
//     //vec=vec1;
//     vec=std::move(vec1);
//     std::cout<<std::endl<<vec.size()<<" "<<vec.capacity()<<" "<<vec.data()<<" "<<vec.get_allocator().state_;
//     std::cout<<std::endl<<vec1.size()<<" "<<vec1.capacity()<<" "<<vec1.data()<<" "<<vec1.get_allocator().state_;
// }


TEST_CASE("test_stack_prealloc_vector_default_constructor","[test_stack_prealloc_vector]")
{
    using value_type = double;
    using gtensor::stack_prealloc_vector;
    using helpers_for_testing::apply_by_element;

    //0vec,1expected_size,2expected_capacity,3expected_empty
    auto test_data = std::make_tuple(
        std::make_tuple(stack_prealloc_vector<value_type,1>{},0,1,true),
        std::make_tuple(stack_prealloc_vector<value_type,8>{},0,8,true)
    );

    auto test = [](const auto& t){
        auto vec = std::get<0>(t);
        auto expected_size = std::get<1>(t);
        auto expected_capacity = std::get<2>(t);
        auto expected_empty = std::get<3>(t);

        auto result_size = vec.size();
        auto result_capacity = vec.capacity();
        auto result_empty = vec.empty();

        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(result_empty == expected_empty);
        REQUIRE(vec.begin() == vec.end());
        REQUIRE(vec.rbegin() == vec.rend());
        using vec_type = decltype(vec);
        REQUIRE(static_cast<const vec_type&>(vec).begin() == static_cast<const vec_type&>(vec).end());
        REQUIRE(static_cast<const vec_type&>(vec).rbegin() == static_cast<const vec_type&>(vec).rend());
    };
    apply_by_element(test,test_data);
}