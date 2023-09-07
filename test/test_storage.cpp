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
    static void reset(){
        ctr_counter=0;
        dtr_counter=0;
    }
};

template<typename T>
struct not_trivial : counter<not_trivial<T>>{
    T t;
    std::vector<T> vec_t;
    not_trivial() = default;
    not_trivial(T t_):
        t{t_},
        vec_t(static_cast<std::size_t>(t_),t_)
    {}
    bool operator==(const not_trivial& other)const{
        return t==other.t && vec_t==other.vec_t;
    }
};

template<typename T, typename AlwaysEqual=std::false_type, typename PropOnCopyAssign=std::true_type, typename PropOnMoveAssign=std::false_type>
struct test_allocator : public std::allocator<T>
{
    using base_type = std::allocator<T>;
    using pointer = typename std::allocator_traits<base_type>::pointer;
    using typename base_type::size_type;
    using is_always_equal = AlwaysEqual;
    using propagate_on_container_copy_assignment = PropOnCopyAssign;
    using propagate_on_container_move_assignment = PropOnMoveAssign;

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

    static void reset(){
        alloc_counter=0;
        dealloc_counter=0;
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
//     //using test_basic_storage::test_allocator1;
//     using vector_type = std::vector<int,test_allocator<int>>;

//     vector_type vec{1,2,3,4,5,6,7,8,9,10};
//     std::cout<<std::endl<<vec.size()<<" "<<vec.capacity()<<" "<<vec.data()<<" "<<vec.get_allocator().state_;
//     vector_type vec1(3,0,test_allocator<int>{1});
//     std::cout<<std::endl<<vec1.size()<<" "<<vec1.capacity()<<" "<<vec1.data()<<" "<<vec1.get_allocator().state_;
//     //vec=vec1;
//     //vec=std::move(vec1);
//     vec.swap(vec1);
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

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_n_constructor_size","[test_stack_prealloc_vector]",
    double,
    std::string
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using helpers_for_testing::apply_by_element;
    //0size_constant,1n,2expected_size,3expected_capacity,4expected_empty
    auto test_data = std::make_tuple(
        std::make_tuple(std::integral_constant<int,1>{},0,0,1,true),
        std::make_tuple(std::integral_constant<int,1>{},1,1,1,false),
        std::make_tuple(std::integral_constant<int,1>{},2,2,2,false),
        std::make_tuple(std::integral_constant<int,4>{},0,0,4,true),
        std::make_tuple(std::integral_constant<int,4>{},1,1,4,false),
        std::make_tuple(std::integral_constant<int,4>{},4,4,4,false),
        std::make_tuple(std::integral_constant<int,4>{},5,5,5,false),
        std::make_tuple(std::integral_constant<int,4>{},8,8,8,false),
        std::make_tuple(std::integral_constant<int,8>{},0,0,8,true),
        std::make_tuple(std::integral_constant<int,8>{},1,1,8,false),
        std::make_tuple(std::integral_constant<int,8>{},5,5,8,false),
        std::make_tuple(std::integral_constant<int,8>{},8,8,8,false),
        std::make_tuple(std::integral_constant<int,8>{},10,10,10,false)
    );

    auto test = [](const auto& t){
        auto size_constant = std::get<0>(t);
        auto n = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_capacity = std::get<3>(t);
        auto expected_empty = std::get<4>(t);
        using vector_type = stack_prealloc_vector<value_type,decltype(size_constant)::value>;
        vector_type vec(n);
        auto result_size = vec.size();
        auto result_capacity = vec.capacity();
        auto result_empty = vec.empty();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(result_empty == expected_empty);
        REQUIRE(std::distance(vec.begin(),vec.end()) == expected_size);
        REQUIRE(std::distance(vec.rbegin(),vec.rend()) == expected_size);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_n_constructor_elements","[test_stack_prealloc_vector]",
    (std::integral_constant<int,1>),
    (std::integral_constant<int,4>),
    (std::integral_constant<int,8>)
)
{
    using value_type = test_basic_storage::not_trivial<double>;
    using vector_type = gtensor::stack_prealloc_vector<value_type,TestType::value>;
    using helpers_for_testing::apply_by_element;
    REQUIRE(!std::is_trivially_copyable_v<value_type>);
    //0n,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(0,std::vector<value_type>{}),
        std::make_tuple(1,std::vector<value_type>(1)),
        std::make_tuple(5,std::vector<value_type>(5)),
        std::make_tuple(8,std::vector<value_type>(8)),
        std::make_tuple(10,std::vector<value_type>(10))
    );

    auto test = [](const auto& t){
        auto n = std::get<0>(t);
        auto expected = std::get<1>(t);

        vector_type vec(n);
        REQUIRE(std::equal(vec.begin(),vec.end(),expected.begin(),expected.end()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_n_value_constructor","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using helpers_for_testing::apply_by_element;
    //0size_constant,1n,2value,3expected_size,4expected_capacity,5expected_empty,6expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(std::integral_constant<int,1>{},0,value_type{2},0,1,true,std::vector<value_type>{}),
        std::make_tuple(std::integral_constant<int,1>{},1,value_type{3},1,1,false,std::vector<value_type>(1,value_type{3})),
        std::make_tuple(std::integral_constant<int,1>{},5,value_type{4},5,5,false,std::vector<value_type>(5,value_type{4})),
        std::make_tuple(std::integral_constant<int,4>{},0,value_type{2},0,4,true,std::vector<value_type>{}),
        std::make_tuple(std::integral_constant<int,4>{},1,value_type{3},1,4,false,std::vector<value_type>(1,value_type{3})),
        std::make_tuple(std::integral_constant<int,4>{},4,value_type{3},4,4,false,std::vector<value_type>(4,value_type{3})),
        std::make_tuple(std::integral_constant<int,4>{},5,value_type{4},5,5,false,std::vector<value_type>(5,value_type{4})),
        std::make_tuple(std::integral_constant<int,4>{},10,value_type{5},10,10,false,std::vector<value_type>(10,value_type{5}))
    );

    auto test = [](const auto& t){
        auto size_constant = std::get<0>(t);
        auto n = std::get<1>(t);
        auto value = std::get<2>(t);
        auto expected_size = std::get<3>(t);
        auto expected_capacity = std::get<4>(t);
        auto expected_empty = std::get<5>(t);
        auto expected_elements = std::get<6>(t);
        using vector_type = stack_prealloc_vector<value_type,decltype(size_constant)::value>;
        vector_type vec(n,value);
        auto result_size = vec.size();
        auto result_capacity = vec.capacity();
        auto result_empty = vec.empty();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(result_empty == expected_empty);
        REQUIRE(std::equal(vec.begin(),vec.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(vec.rbegin(),vec.rend(),expected_elements.rbegin(),expected_elements.rend()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_iterators_range_constructor","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using helpers_for_testing::apply_by_element;
    //0size_constant,1elements,2expected_size,3expected_capacity,4expected_empty
    auto test_data = std::make_tuple(
        std::make_tuple(std::integral_constant<int,1>{},std::vector<value_type>{},0,1,true),
        std::make_tuple(std::integral_constant<int,1>{},std::vector<value_type>{value_type{3}},1,1,false),
        std::make_tuple(std::integral_constant<int,1>{},std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{5},value_type{6}},6,6,false),
        std::make_tuple(std::integral_constant<int,1>{},std::list<value_type>{value_type{7},value_type{8},value_type{9},value_type{10},value_type{11}},5,5,false),
        std::make_tuple(std::integral_constant<int,8>{},std::vector<value_type>{},0,8,true),
        std::make_tuple(std::integral_constant<int,8>{},std::vector<value_type>{value_type{3}},1,8,false),
        std::make_tuple(std::integral_constant<int,8>{},std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{5},value_type{6}},6,8,false),
        std::make_tuple(std::integral_constant<int,8>{},std::list<value_type>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{5},value_type{6},value_type{7},value_type{8},value_type{9},value_type{10},value_type{11}},11,11,false)
    );

    auto test = [](const auto& t){
        auto size_constant = std::get<0>(t);
        auto elements = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_capacity = std::get<3>(t);
        auto expected_empty = std::get<4>(t);
        auto expected_elements = elements;
        using vector_type = stack_prealloc_vector<value_type,decltype(size_constant)::value>;
        vector_type vec(elements.begin(),elements.end());
        auto result_size = vec.size();
        auto result_capacity = vec.capacity();
        auto result_empty = vec.empty();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(result_empty == expected_empty);
        REQUIRE(std::equal(vec.begin(),vec.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(vec.rbegin(),vec.rend(),expected_elements.rbegin(),expected_elements.rend()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_initializer_list_constructor","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using helpers_for_testing::apply_by_element;

    //0vec,1expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(stack_prealloc_vector<value_type,4>{},std::vector<value_type>{}),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{3}},std::vector<value_type>{value_type{3}}),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{1},value_type{2},value_type{3},value_type{4}},std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4}}),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{1},value_type{2},value_type{3},value_type{4}},
            std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{1},value_type{2},value_type{3},value_type{4}}
        )
    );

    auto test = [](const auto& t){
        auto vec = std::get<0>(t);
        auto expected_elements = std::get<1>(t);
        REQUIRE(std::equal(vec.begin(),vec.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(vec.rbegin(),vec.rend(),expected_elements.rbegin(),expected_elements.rend()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_copy_constructor","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using helpers_for_testing::apply_by_element;

    //0vec,1expected_size,2expected_capacity,3expected_empty,4expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(stack_prealloc_vector<value_type,4>{},0,4,true,std::vector<value_type>{}),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{1}},1,4,false,std::vector<value_type>{value_type{1}}),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{1},value_type{2}},2,4,false,std::vector<value_type>{value_type{1},value_type{2}}),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{1},value_type{2},value_type{3},value_type{4}},4,4,false,
            std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4}}
        ),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{1},value_type{2},value_type{3},value_type{4}},8,8,false,
            std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{1},value_type{2},value_type{3},value_type{4}}
        ),
        std::make_tuple(stack_prealloc_vector<value_type,10>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{1},value_type{2},value_type{3},value_type{4}},8,10,false,
            std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{1},value_type{2},value_type{3},value_type{4}}
        )
    );

    auto test = [](const auto& t){
        auto vec = std::get<0>(t);
        auto expected_size = std::get<1>(t);
        auto expected_capacity = std::get<2>(t);
        auto expected_empty = std::get<3>(t);
        auto expected_elements = std::get<4>(t);

        auto result = vec;
        auto result_size = result.size();
        auto result_capacity = result.capacity();
        auto result_empty = result.empty();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(result_empty == expected_empty);
        REQUIRE(std::equal(result.begin(),result.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(result.rbegin(),result.rend(),expected_elements.rbegin(),expected_elements.rend()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_move_constructor","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using helpers_for_testing::apply_by_element;

    //0vec,1expected_size,2expected_capacity,3expected_empty,4expected_elements
    auto test_data = std::make_tuple(
        std::make_tuple(stack_prealloc_vector<value_type,4>{},0,4,true,std::vector<value_type>{}),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{1}},1,4,false,std::vector<value_type>{value_type{1}}),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{1},value_type{2}},2,4,false,std::vector<value_type>{value_type{1},value_type{2}}),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{1},value_type{2},value_type{3},value_type{4}},4,4,false,
            std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4}}
        ),
        std::make_tuple(stack_prealloc_vector<value_type,4>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{1},value_type{2},value_type{3},value_type{4}},8,8,false,
            std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{1},value_type{2},value_type{3},value_type{4}}
        ),
        std::make_tuple(stack_prealloc_vector<value_type,10>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{1},value_type{2},value_type{3},value_type{4}},8,10,false,
            std::vector<value_type>{value_type{1},value_type{2},value_type{3},value_type{4},value_type{1},value_type{2},value_type{3},value_type{4}}
        )
    );

    auto test = [](const auto& t){
        auto vec = std::get<0>(t);
        auto expected_size = std::get<1>(t);
        auto expected_capacity = std::get<2>(t);
        auto expected_empty = std::get<3>(t);
        auto expected_elements = std::get<4>(t);

        auto result = std::move(vec);
        auto result_size = result.size();
        auto result_capacity = result.capacity();
        auto result_empty = result.empty();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(result_empty == expected_empty);
        REQUIRE(std::equal(result.begin(),result.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(result.rbegin(),result.rend(),expected_elements.rbegin(),expected_elements.rend()));

        REQUIRE(vec.empty());
        REQUIRE(vec.size()==0);
        REQUIRE(vec.begin()==vec.end());
        REQUIRE(vec.rbegin()==vec.rend());
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_stack_prealloc_vector_operator==","[test_stack_prealloc_vector]")
{
    using value_type = double;
    using vector_type = gtensor::stack_prealloc_vector<value_type>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(vector_type{},vector_type{},true),
        std::make_tuple(vector_type{1},vector_type{1},true),
        std::make_tuple(vector_type{1,2,3,4,5},vector_type{1,2,3,4,5},true),
        std::make_tuple(vector_type{},vector_type{1,2,3,4,5},false),
        std::make_tuple(vector_type{1,2,3,4,5},vector_type{},false),
        std::make_tuple(vector_type{6,7,8},vector_type{1,2,3,4,5},false),
        std::make_tuple(vector_type{1,2,3,4,5},vector_type{6,7,8},false),
        std::make_tuple(vector_type{1,2,3,4,5},vector_type{1,2,0,4,5},false)
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

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_copy_assign","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using vector_type = stack_prealloc_vector<value_type,4>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected_size,3expected_capacity,4expected_elements
    auto test_data = std::make_tuple(
        //always_equal_allocator
        std::make_tuple(vector_type{},vector_type{},0,4,std::vector<value_type>{}),
        std::make_tuple(vector_type{},vector_type{1,2,3},3,4,std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type{},vector_type{1,2,3,4,5},5,5,std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(vector_type{6,7,8},vector_type{},0,4,std::vector<value_type>{}),
        std::make_tuple(vector_type{6,7,8,9},vector_type{10,11,12},3,4,std::vector<value_type>{10,11,12}),
        std::make_tuple(vector_type{6,7,8},vector_type{1,2,3,4,5,6},6,6,std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type{6,7,8,9,10,11,12},vector_type{},0,7,std::vector<value_type>{}),
        std::make_tuple(vector_type{6,7,8,9,10,11,12},vector_type{1,2,3},3,7,std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type{6,7,8,9,10,11,12},vector_type{4,5,6,7,8,9,10,11,12},9,9,std::vector<value_type>{4,5,6,7,8,9,10,11,12})
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_capacity = std::get<3>(t);
        auto expected_elements = std::get<4>(t);

        auto& result = lhs=rhs;
        REQUIRE(&result == &lhs);
        auto result_size = result.size();
        auto result_capacity = result.capacity();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(std::equal(result.begin(),result.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(result.rbegin(),result.rend(),expected_elements.rbegin(),expected_elements.rend()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_copy_assign_propagate_allocator","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using allocator_type = test_basic_storage::test_allocator<value_type>;
    using gtensor::stack_prealloc_vector;
    using vector_type = stack_prealloc_vector<value_type,4,allocator_type>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected_size,3expected_capacity,4expected_allocator,5expected_elements
    auto test_data = std::make_tuple(
        //equal allocators
        std::make_tuple(vector_type(allocator_type{2}),vector_type(allocator_type{2}),0,4,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type(allocator_type{2}),vector_type({1,2,3},allocator_type{2}),3,4,allocator_type{2},std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type(allocator_type{2}),vector_type({1,2,3,4,5},allocator_type{2}),5,5,allocator_type{2},std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(vector_type({6,7,8},allocator_type{2}),vector_type({},allocator_type{2}),0,4,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type({6,7,8,9},allocator_type{2}),vector_type({10,11,12},allocator_type{2}),3,4,allocator_type{2},std::vector<value_type>{10,11,12}),
        std::make_tuple(vector_type({6,7,8},allocator_type{2}),vector_type({1,2,3,4,5,6},allocator_type{2}),6,6,allocator_type{2},std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{2}),vector_type(allocator_type{2}),0,7,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{2}),vector_type({1,2,3},allocator_type{2}),3,7,allocator_type{2},std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{2}),vector_type({4,5,6,7,8,9,10,11,12},allocator_type{2}),9,9,allocator_type{2},std::vector<value_type>{4,5,6,7,8,9,10,11,12}),
        //not equal allocators
        std::make_tuple(vector_type(allocator_type{1}),vector_type(allocator_type{2}),0,4,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type(allocator_type{1}),vector_type({1,2,3},allocator_type{2}),3,4,allocator_type{2},std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type(allocator_type{1}),vector_type({1,2,3,4,5},allocator_type{2}),5,5,allocator_type{2},std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(vector_type({6,7,8},allocator_type{1}),vector_type({},allocator_type{2}),0,4,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type({6,7,8,9},allocator_type{1}),vector_type({10,11,12},allocator_type{2}),3,4,allocator_type{2},std::vector<value_type>{10,11,12}),
        std::make_tuple(vector_type({6,7,8},allocator_type{1}),vector_type({1,2,3,4,5,6},allocator_type{2}),6,6,allocator_type{2},std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{1}),vector_type(allocator_type{2}),0,0,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{1}),vector_type({1,2,3},allocator_type{2}),3,3,allocator_type{2},std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{1}),vector_type({4,5,6,7,8,9,10,11,12},allocator_type{2}),9,9,allocator_type{2},std::vector<value_type>{4,5,6,7,8,9,10,11,12})
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_capacity = std::get<3>(t);
        auto expected_allocator = std::get<4>(t);
        auto expected_elements = std::get<5>(t);

        auto& result = lhs=rhs;
        REQUIRE(&result == &lhs);
        auto result_size = result.size();
        auto result_capacity = result.capacity();
        auto result_allocator = result.get_allocator();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(result_allocator == expected_allocator);
        REQUIRE(std::equal(result.begin(),result.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(result.rbegin(),result.rend(),expected_elements.rbegin(),expected_elements.rend()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_move_assign","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using vector_type = stack_prealloc_vector<value_type,4>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected_size,3expected_capacity,4expected_elements
    auto test_data = std::make_tuple(
        //always_equal_allocator
        std::make_tuple(vector_type{},vector_type{},0,4,std::vector<value_type>{}),
        std::make_tuple(vector_type{},vector_type{1,2,3},3,4,std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type{},vector_type{1,2,3,4,5},5,5,std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(vector_type{6,7,8},vector_type{},0,4,std::vector<value_type>{}),
        std::make_tuple(vector_type{6,7,8,9},vector_type{10,11,12},3,4,std::vector<value_type>{10,11,12}),
        std::make_tuple(vector_type{6,7,8},vector_type{1,2,3,4,5,6},6,6,std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type{6,7,8,9,10,11,12},vector_type{},0,7,std::vector<value_type>{}),
        std::make_tuple(vector_type{6,7,8,9,10,11,12},vector_type{1,2,3},3,7,std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type{6,7,8,9,10,11,12},vector_type{4,5,6,7,8,9,10,11,12},9,9,std::vector<value_type>{4,5,6,7,8,9,10,11,12})
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_capacity = std::get<3>(t);
        auto expected_elements = std::get<4>(t);

        auto& result = lhs=std::move(rhs);
        REQUIRE(&result == &lhs);
        auto result_size = result.size();
        auto result_capacity = result.capacity();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(std::equal(result.begin(),result.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(result.rbegin(),result.rend(),expected_elements.rbegin(),expected_elements.rend()));

        REQUIRE(rhs.empty());
        REQUIRE(rhs.size()==0);
        REQUIRE(rhs.begin()==rhs.end());
        REQUIRE(rhs.rbegin()==rhs.rend());
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_move_assign_not_propagate_allocator","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using allocator_type = test_basic_storage::test_allocator<value_type>;
    using gtensor::stack_prealloc_vector;
    using vector_type = stack_prealloc_vector<value_type,4,allocator_type>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected_size,3expected_capacity,4expected_allocator,5expected_elements
    auto test_data = std::make_tuple(
        //equal allocators
        std::make_tuple(vector_type(allocator_type{2}),vector_type(allocator_type{2}),0,4,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type(allocator_type{2}),vector_type({1,2,3},allocator_type{2}),3,4,allocator_type{2},std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type(allocator_type{2}),vector_type({1,2,3,4,5},allocator_type{2}),5,5,allocator_type{2},std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(vector_type({6,7,8},allocator_type{2}),vector_type({},allocator_type{2}),0,4,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type({6,7,8,9},allocator_type{2}),vector_type({10,11,12},allocator_type{2}),3,4,allocator_type{2},std::vector<value_type>{10,11,12}),
        std::make_tuple(vector_type({6,7,8},allocator_type{2}),vector_type({1,2,3,4,5,6},allocator_type{2}),6,6,allocator_type{2},std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{2}),vector_type(allocator_type{2}),0,7,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{2}),vector_type({1,2,3},allocator_type{2}),3,7,allocator_type{2},std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{2}),vector_type({4,5,6,7,8,9,10,11,12},allocator_type{2}),9,9,allocator_type{2},std::vector<value_type>{4,5,6,7,8,9,10,11,12}),
        //not equal allocators
        std::make_tuple(vector_type(allocator_type{1}),vector_type(allocator_type{2}),0,4,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type(allocator_type{1}),vector_type({1,2,3},allocator_type{2}),3,4,allocator_type{2},std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type(allocator_type{1}),vector_type({1,2,3,4,5},allocator_type{2}),5,5,allocator_type{2},std::vector<value_type>{1,2,3,4,5}),
        std::make_tuple(vector_type({6,7,8},allocator_type{1}),vector_type({},allocator_type{2}),0,4,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type({6,7,8,9},allocator_type{1}),vector_type({10,11,12},allocator_type{2}),3,4,allocator_type{2},std::vector<value_type>{10,11,12}),
        std::make_tuple(vector_type({6,7,8},allocator_type{1}),vector_type({1,2,3,4,5,6},allocator_type{2}),6,6,allocator_type{2},std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{1}),vector_type(allocator_type{2}),0,7,allocator_type{2},std::vector<value_type>{}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{1}),vector_type({1,2,3},allocator_type{2}),3,7,allocator_type{2},std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type({6,7,8,9,10,11,12},allocator_type{1}),vector_type({4,5,6,7,8,9,10,11,12},allocator_type{2}),9,9,allocator_type{2},std::vector<value_type>{4,5,6,7,8,9,10,11,12})
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_capacity = std::get<3>(t);
        auto expected_allocator = std::get<4>(t);
        auto expected_elements = std::get<5>(t);

        auto& result = lhs=std::move(rhs);
        REQUIRE(&result == &lhs);
        auto result_size = result.size();
        auto result_capacity = result.capacity();
        auto result_allocator = result.get_allocator();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(result_allocator == expected_allocator);
        REQUIRE(std::equal(result.begin(),result.end(),expected_elements.begin(),expected_elements.end()));
        REQUIRE(std::equal(result.rbegin(),result.rend(),expected_elements.rbegin(),expected_elements.rend()));

        REQUIRE(rhs.empty());
        REQUIRE(rhs.size()==0);
        REQUIRE(rhs.begin()==rhs.end());
        REQUIRE(rhs.rbegin()==rhs.rend());
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_swap","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using vector_type = stack_prealloc_vector<value_type,4>;
    using helpers_for_testing::apply_by_element;

    //0lhs,1rhs,2expected_lhs_size,3expected_lhs_capacity,4expected_rhs_size,5expected_rhs_capacity,6expected_lhs_elements,7expected_rhs_elements
    auto test_data = std::make_tuple(
        //some on stack
        //lhs.size()==rhs.size()
        std::make_tuple(vector_type{},vector_type{},0,4,0,4,std::vector<value_type>{},std::vector<value_type>{}),
        std::make_tuple(vector_type{4,5,6},vector_type{1,2,3},3,4,3,4,std::vector<value_type>{1,2,3},std::vector<value_type>{4,5,6}),
        //lhs.capacity()<rhs.size()
        std::make_tuple(vector_type{},vector_type{4,5,6,7,8,9},6,6,0,6,std::vector<value_type>{4,5,6,7,8,9},std::vector<value_type>{}),
        std::make_tuple(vector_type{1,2,3},vector_type{4,5,6,7,8,9},6,6,3,6,std::vector<value_type>{4,5,6,7,8,9},std::vector<value_type>{1,2,3}),
        //rhs.capacity()<lhs.size()
        std::make_tuple(vector_type{4,5,6,7,8,9},vector_type{},0,6,6,6,std::vector<value_type>{},std::vector<value_type>{4,5,6,7,8,9}),
        std::make_tuple(vector_type{4,5,6,7,8,9},vector_type{1,2,3},3,6,6,6,std::vector<value_type>{1,2,3},std::vector<value_type>{4,5,6,7,8,9}),
        //lhs.size()<rhs.size()
        std::make_tuple(vector_type{},vector_type{1,2,3},3,4,0,4,std::vector<value_type>{1,2,3},std::vector<value_type>{}),
        std::make_tuple(vector_type{1,2},vector_type{3,4,5,6},4,4,2,4,std::vector<value_type>{3,4,5,6},std::vector<value_type>{1,2}),
        std::make_tuple(vector_type{},vector_type{1,2,3,4,5,6},6,6,0,6,std::vector<value_type>{1,2,3,4,5,6},std::vector<value_type>{}),
        std::make_tuple(vector_type{1,2,3},vector_type{4,5,6,7,8,9},6,6,3,6,std::vector<value_type>{4,5,6,7,8,9},std::vector<value_type>{1,2,3}),
        //rhs.size()<lhs.size()
        std::make_tuple(vector_type{1,2,3},vector_type{},0,4,3,4,std::vector<value_type>{},std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type{3,4,5,6},vector_type{1,2},2,4,4,4,std::vector<value_type>{1,2},std::vector<value_type>{3,4,5,6}),
        std::make_tuple(vector_type{1,2,3,4,5,6},vector_type{},0,6,6,6,std::vector<value_type>{},std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type{4,5,6,7,8,9},vector_type{1,2,3},3,6,6,6,std::vector<value_type>{1,2,3},std::vector<value_type>{4,5,6,7,8,9}),
        //not on stack
        std::make_tuple(vector_type{7,8,9,10,11,12},vector_type{1,2,3,4,5,6},6,6,6,6,std::vector<value_type>{1,2,3,4,5,6},std::vector<value_type>{7,8,9,10,11,12}),
        std::make_tuple(vector_type{1,2,3,4,5,6},vector_type{7,8,9,10,11,12,13,14,15},9,9,6,6,std::vector<value_type>{7,8,9,10,11,12,13,14,15},std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type{7,8,9,10,11,12,13,14,15},vector_type{1,2,3,4,5,6},6,6,9,9,std::vector<value_type>{1,2,3,4,5,6},std::vector<value_type>{7,8,9,10,11,12,13,14,15})
    );

    auto test = [](const auto& t){
        auto lhs = std::get<0>(t);
        auto rhs = std::get<1>(t);
        auto expected_lhs_size = std::get<2>(t);
        auto expected_lhs_capacity = std::get<3>(t);
        auto expected_rhs_size = std::get<4>(t);
        auto expected_rhs_capacity = std::get<5>(t);
        auto expected_lhs_elements = std::get<6>(t);
        auto expected_rhs_elements = std::get<7>(t);

        lhs.swap(rhs);
        auto result_lhs_size = lhs.size();
        auto result_lhs_capacity = lhs.capacity();
        auto result_rhs_size = rhs.size();
        auto result_rhs_capacity = rhs.capacity();
        REQUIRE(result_lhs_size == expected_lhs_size);
        REQUIRE(result_lhs_capacity == expected_lhs_capacity);
        REQUIRE(result_rhs_size == expected_rhs_size);
        REQUIRE(result_rhs_capacity == expected_rhs_capacity);
        REQUIRE(std::equal(lhs.begin(),lhs.end(),expected_lhs_elements.begin(),expected_lhs_elements.end()));
        REQUIRE(std::equal(rhs.begin(),rhs.end(),expected_rhs_elements.begin(),expected_rhs_elements.end()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_reserve","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using vector_type = stack_prealloc_vector<value_type,4>;
    using helpers_for_testing::apply_by_element;

    //0vec,1command,2expected_size,3expected_capacity,4expected_elements
    auto test_data = std::make_tuple(
        //empty
        std::make_tuple(vector_type{},[](auto& v){v.reserve(0);},0,4,std::vector<value_type>{}),
        std::make_tuple(vector_type{},[](auto& v){v.reserve(3);},0,4,std::vector<value_type>{}),
        std::make_tuple(vector_type{},[](auto& v){v.reserve(2); v.reserve(3);},0,4,std::vector<value_type>{}),
        std::make_tuple(vector_type{},[](auto& v){v.reserve(10);},0,10,std::vector<value_type>{}),
        std::make_tuple(vector_type{},[](auto& v){v.reserve(10); v.reserve(8);},0,10,std::vector<value_type>{}),
        std::make_tuple(vector_type{},[](auto& v){v.reserve(10); v.reserve(18);},0,18,std::vector<value_type>{}),
        //on stack
        std::make_tuple(vector_type{1,2,3},[](auto& v){v.reserve(0);},3,4,std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type{1,2,3},[](auto& v){v.reserve(3);},3,4,std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type{1,2,3},[](auto& v){v.reserve(2); v.reserve(3);},3,4,std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type{1,2,3},[](auto& v){v.reserve(10);},3,10,std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type{1,2,3},[](auto& v){v.reserve(10); v.reserve(8);},3,10,std::vector<value_type>{1,2,3}),
        std::make_tuple(vector_type{1,2,3},[](auto& v){v.reserve(10); v.reserve(18);},3,18,std::vector<value_type>{1,2,3}),
        //allocated
        std::make_tuple(vector_type{1,2,3,4,5,6},[](auto& v){v.reserve(0);},6,6,std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type{1,2,3,4,5,6},[](auto& v){v.reserve(3);},6,6,std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type{1,2,3,4,5,6},[](auto& v){v.reserve(2); v.reserve(3);},6,6,std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type{1,2,3,4,5,6},[](auto& v){v.reserve(10);},6,10,std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type{1,2,3,4,5,6},[](auto& v){v.reserve(10); v.reserve(8);},6,10,std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type{1,2,3,4,5,6},[](auto& v){v.reserve(10); v.reserve(18);},6,18,std::vector<value_type>{1,2,3,4,5,6})
    );

    auto test = [](const auto& t){
        auto vec = std::get<0>(t);
        auto command = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_capacity = std::get<3>(t);
        auto expected_elements = std::get<4>(t);

        command(vec);
        auto result_size = vec.size();
        auto result_capacity = vec.capacity();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(std::equal(vec.begin(),vec.end(),expected_elements.begin(),expected_elements.end()));
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_stack_prealloc_vector_push_back","[test_stack_prealloc_vector]",
    double,
    test_basic_storage::not_trivial<double>
)
{
    using value_type = TestType;
    using gtensor::stack_prealloc_vector;
    using vector_type = stack_prealloc_vector<value_type,4>;
    using helpers_for_testing::apply_by_element;

    //0vec,1command,2expected_size,3expected_capacity,4expected_elements
    auto test_data = std::make_tuple(

        std::make_tuple(vector_type{},[](auto& v){v.push_back(1);},1,4,std::vector<value_type>{1}),
        std::make_tuple(vector_type{},[](auto& v){v.push_back(1); v.push_back(2);},2,4,std::vector<value_type>{1,2}),
        std::make_tuple(vector_type{},[](auto& v){v.push_back(1); v.push_back(2);},2,4,std::vector<value_type>{1,2}),
        std::make_tuple(vector_type{},[](auto& v){v.push_back(1); v.push_back(2); v.push_back(3); v.push_back(4);},4,4,std::vector<value_type>{1,2,3,4}),
        std::make_tuple(vector_type{},[](auto& v){v.push_back(1); v.push_back(2); v.push_back(3); v.push_back(4); v.push_back(5); v.push_back(6);},6,8,std::vector<value_type>{1,2,3,4,5,6}),
        std::make_tuple(vector_type{},
            [](auto& v){v.push_back(1); v.push_back(2); v.push_back(3); v.push_back(4); v.push_back(5); v.push_back(6); v.push_back(7); v.push_back(8); v.push_back(9); v.push_back(10);},
            10,16,std::vector<value_type>{1,2,3,4,5,6,7,8,9,10}
        ),
        std::make_tuple(vector_type{1,2,3},[](auto& v){v.push_back(1); v.push_back(2); v.push_back(3); v.push_back(4);},7,8,std::vector<value_type>{1,2,3,1,2,3,4}),
        std::make_tuple(vector_type{1,2,3,4,5,6},[](auto& v){v.push_back(1); v.push_back(2); v.push_back(3); v.push_back(4);},10,12,std::vector<value_type>{1,2,3,4,5,6,1,2,3,4})
    );

    auto test = [](const auto& t){
        auto vec = std::get<0>(t);
        auto command = std::get<1>(t);
        auto expected_size = std::get<2>(t);
        auto expected_capacity = std::get<3>(t);
        auto expected_elements = std::get<4>(t);

        command(vec);
        auto result_size = vec.size();
        auto result_capacity = vec.capacity();
        REQUIRE(result_size == expected_size);
        REQUIRE(result_capacity == expected_capacity);
        REQUIRE(std::equal(vec.begin(),vec.end(),expected_elements.begin(),expected_elements.end()));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_stack_prealloc_vector_destructor","[test_stack_prealloc_vector]")
{
    using value_type = test_basic_storage::not_trivial<double>;
    using AlwaysEqual = std::true_type;
    using PropOnCopyAssign = std::false_type;
    using PropOnMoveAssign = std::true_type;
    using allocator_type = test_basic_storage::test_allocator<value_type,AlwaysEqual,PropOnCopyAssign,PropOnMoveAssign>;
    static constexpr std::size_t Size=4;
    using vector_type = gtensor::stack_prealloc_vector<value_type,Size,allocator_type>;
    value_type::reset();
    allocator_type::reset();
    SECTION("on_stack_default")
    {
        {
            vector_type vec{};
        }
        REQUIRE(value_type::ctr_counter == 0);
        REQUIRE(value_type::dtr_counter == 0);
        REQUIRE(allocator_type::alloc_counter == 0);
        REQUIRE(allocator_type::dealloc_counter == 0);
    }
    SECTION("on_stack_from_range")
    {
        std::vector<value_type> v{2,3};
        value_type::reset();
        {
            vector_type vec(v.begin(),v.end());
        }
        REQUIRE(value_type::ctr_counter == 2);
        REQUIRE(value_type::dtr_counter == value_type::ctr_counter);
        REQUIRE(allocator_type::alloc_counter == 0);
        REQUIRE(allocator_type::dealloc_counter == 0);
    }
    SECTION("on_stack_n_value")
    {
        value_type v{2};
        value_type::reset();
        {
            vector_type vec(3,v);
        }
        REQUIRE(value_type::ctr_counter == 3);
        REQUIRE(value_type::dtr_counter == value_type::ctr_counter);
        REQUIRE(allocator_type::alloc_counter == 0);
        REQUIRE(allocator_type::dealloc_counter == 0);
    }
    SECTION("allocate_from_range")
    {
        std::vector<value_type> v(Size+10,value_type{4});
        value_type::reset();
        {
            vector_type vec(v.begin(),v.end());
        }
        REQUIRE(value_type::ctr_counter == Size+10);
        REQUIRE(value_type::dtr_counter == value_type::ctr_counter);
        REQUIRE(allocator_type::alloc_counter == 1);
        REQUIRE(allocator_type::dealloc_counter == 1);
    }
    SECTION("allocate_n_value")
    {
        value_type v{2};
        value_type::reset();
        {
            vector_type vec(Size+5,v);
        }
        REQUIRE(value_type::ctr_counter == Size+5);
        REQUIRE(value_type::dtr_counter == value_type::ctr_counter);
        REQUIRE(allocator_type::alloc_counter == 1);
        REQUIRE(allocator_type::dealloc_counter == 1);
    }
    SECTION("on_stack_reserve")
    {
        {
            vector_type vec{};
            vec.reserve(Size-1);
            vec.reserve(Size);
        }
        REQUIRE(value_type::ctr_counter == 0);
        REQUIRE(value_type::dtr_counter == 0);
        REQUIRE(allocator_type::alloc_counter == 0);
        REQUIRE(allocator_type::dealloc_counter == 0);
    }
    SECTION("allocate_reserve")
    {
        {
            vector_type vec{};
            vec.reserve(Size+10);
        }
        REQUIRE(value_type::ctr_counter == 0);
        REQUIRE(value_type::dtr_counter == 0);
        REQUIRE(allocator_type::alloc_counter == 1);
        REQUIRE(allocator_type::dealloc_counter == 1);
    }
    SECTION("on_stack_push_back_1")
    {
        value_type v{2};
        value_type::reset();
        {
            vector_type vec{};
            vec.push_back(v);
        }
        REQUIRE(value_type::ctr_counter == 1);
        REQUIRE(value_type::dtr_counter == 1);
        REQUIRE(allocator_type::alloc_counter == 0);
        REQUIRE(allocator_type::dealloc_counter == 0);
    }
    SECTION("on_stack_push_back_2")
    {
        value_type v{2};
        value_type::reset();
        {
            vector_type vec{};
            vec.push_back(v);
            vec.push_back(v);
            vec.push_back(v);
        }
        REQUIRE(value_type::ctr_counter == 3);
        REQUIRE(value_type::dtr_counter == 3);
        REQUIRE(allocator_type::alloc_counter == 0);
        REQUIRE(allocator_type::dealloc_counter == 0);
    }
    SECTION("on_stack_push_back_3")
    {
        std::vector<value_type> v1{1,2};
        value_type v2{3};
        value_type::reset();
        {
            vector_type vec(v1.begin(),v1.end());
            vec.push_back(v2);
            vec.push_back(v2);
        }
        REQUIRE(value_type::ctr_counter == 4);
        REQUIRE(value_type::dtr_counter == 4);
        REQUIRE(allocator_type::alloc_counter == 0);
        REQUIRE(allocator_type::dealloc_counter == 0);
    }
    SECTION("allocate_push_back")
    {
        std::vector<value_type> v1{1,2,3};
        value_type v2{4};
        value_type::reset();
        {
            vector_type vec(v1.begin(),v1.end());
            vec.push_back(v2);
            vec.push_back(v2);
            vec.push_back(v2);
        }
        REQUIRE(value_type::ctr_counter == 10); //init 3, push_back 1, reallocate 3+1, push_back 2
        REQUIRE(value_type::dtr_counter == 10);
        REQUIRE(allocator_type::alloc_counter == 1);
        REQUIRE(allocator_type::dealloc_counter == 1);
    }
}
