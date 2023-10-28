#include <tuple>
#include "catch.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"

namespace test_tensor_cloning_semantics_{

template<typename T>
class iterable_storage
{
    using inner_storage_type = std::vector<T>;
    inner_storage_type impl_;

    inline static std::size_t copy_ctr_counter_{0};
    inline static std::size_t move_ctr_counter_{0};
    inline static std::size_t custom_ctr_counter_{0};

public:
    using value_type = T;
    using size_type = typename inner_storage_type::size_type;
    using difference_type = typename inner_storage_type::difference_type;

    iterable_storage(const iterable_storage& other):
        impl_(other.impl_)
    {
        copy_ctr_counter_++;
    }
    iterable_storage(iterable_storage&& other):
        impl_(std::move(other.impl_))
    {
        move_ctr_counter_++;
    }

    iterable_storage(size_type n):
        impl_(n)
    {
        custom_ctr_counter_++;
    }
    template<typename It>
    iterable_storage(It first, It last):
        impl_(first,last)
    {
        custom_ctr_counter_++;
    }
    auto begin(){return impl_.begin();}
    auto end(){return impl_.end();}
    auto begin()const{return impl_.begin();}
    auto end()const{return impl_.end();}

    static void reset_counters(){
        copy_ctr_counter_=0;
        move_ctr_counter_=0;
        custom_ctr_counter_=0;
    }
    static auto custom_ctr_counter(){
        return custom_ctr_counter_;
    }
    static auto copy_ctr_counter(){
        return copy_ctr_counter_;
    }
    static auto move_ctr_counter(){
        return move_ctr_counter_;
    }
};

template<typename Semantics>
struct test_config : gtensor::config::default_config{
    using semantics = Semantics;
    template<typename T> using storage = iterable_storage<T>;
};


}   //end of namespace test_tensor_cloning_semantics_

TEST_CASE("test_tensor_cloning_semantics_shallow","[test_tensor_cloning_semantics]")
{
    using value_type = double;
    using gtensor::config::c_order;
    using storage_type = test_tensor_cloning_semantics_::iterable_storage<value_type>;
    using semantics = gtensor::config::shallow_semantics;
    using config_type = gtensor::config::extend_config_t<test_tensor_cloning_semantics_::test_config<semantics>,value_type>;
    using tensor_type = gtensor::tensor<value_type,c_order,config_type>;

    tensor_type t{1,2,3,4,5};
    storage_type::reset_counters();
    REQUIRE(storage_type::copy_ctr_counter()==0);
    REQUIRE(storage_type::move_ctr_counter()==0);
    REQUIRE(storage_type::custom_ctr_counter()==0);

    SECTION("copy construction")
    {
        auto t1 = t;
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(t.is_same(t1));
        auto t2 = t1;
        REQUIRE(t.is_same(t1));
        REQUIRE(t.is_same(t2));
        REQUIRE(t1.is_same(t2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("move construction")
    {
        auto t1 = std::move(t);
        REQUIRE(t.empty());
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
}

TEST_CASE("test_tensor_cloning_semantics_deep","[test_tensor_cloning_semantics]")
{
    using value_type = double;
    using gtensor::config::c_order;
    using storage_type = test_tensor_cloning_semantics_::iterable_storage<value_type>;
    using semantics = gtensor::config::deep_semantics;
    using config_type = gtensor::config::extend_config_t<test_tensor_cloning_semantics_::test_config<semantics>,value_type>;
    using tensor_type = gtensor::tensor<value_type,c_order,config_type>;

    tensor_type t{1,2,3,4,5};
    storage_type::reset_counters();
    REQUIRE(storage_type::copy_ctr_counter()==0);
    REQUIRE(storage_type::move_ctr_counter()==0);
    REQUIRE(storage_type::custom_ctr_counter()==0);

    SECTION("copy_construction")
    {
        auto t1 = t;
        REQUIRE(storage_type::copy_ctr_counter()==1);
        REQUIRE(!t.is_same(t1));
        auto t2 = t1;
        REQUIRE(storage_type::copy_ctr_counter()==2);
        REQUIRE(!t.is_same(t1));
        REQUIRE(!t.is_same(t2));
        REQUIRE(!t1.is_same(t2));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("move construction")
    {
        auto t1 = std::move(t);
        REQUIRE(t.empty());
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
}

TEMPLATE_TEST_CASE("test_tensor_cloning_semantics_clone","[test_tensor_cloning_semantics]",
    gtensor::config::deep_semantics,
    gtensor::config::shallow_semantics
)
{
    using value_type = double;
    using gtensor::config::c_order;
    using gtensor::config::deep_semantics;
    using gtensor::config::shallow_semantics;
    using storage_type = test_tensor_cloning_semantics_::iterable_storage<value_type>;
    using semantics = TestType;
    using config_type = gtensor::config::extend_config_t<test_tensor_cloning_semantics_::test_config<semantics>,value_type>;
    using tensor_type = gtensor::tensor<value_type,c_order,config_type>;

    tensor_type t{1,2,3,4,5};
    storage_type::reset_counters();
    REQUIRE(storage_type::copy_ctr_counter()==0);
    REQUIRE(storage_type::move_ctr_counter()==0);
    REQUIRE(storage_type::custom_ctr_counter()==0);

    SECTION("making_shallow_copy_with_shallow_semantic")
    {
        auto t1 = t.clone(shallow_semantics::value,shallow_semantics::value);
        auto t2 = t1;
        REQUIRE(t.is_same(t1));
        REQUIRE(t.is_same(t2));
        REQUIRE(t1.is_same(t2));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("making_deep_copy_with_deep_semantic")
    {
        auto t1 = t.clone(deep_semantics::value,deep_semantics::value);
        REQUIRE(storage_type::copy_ctr_counter()==1);
        REQUIRE(!t.is_same(t1));
        auto t2 = t1;
        REQUIRE(storage_type::copy_ctr_counter()==2);
        REQUIRE(!t.is_same(t2));
        REQUIRE(!t1.is_same(t2));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("making_shallow_copy_with_deep_semantic")
    {
        auto t1 = t.clone(shallow_semantics::value,deep_semantics::value);
        REQUIRE(t.is_same(t1));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        auto t2 = t1;
        REQUIRE(storage_type::copy_ctr_counter()==1);
        REQUIRE(!t.is_same(t2));
        REQUIRE(!t1.is_same(t2));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("making_deep_copy_with_shallow_semantic")
    {
        auto t1 = t.clone(deep_semantics::value,shallow_semantics::value);
        REQUIRE(!t.is_same(t1));
        REQUIRE(storage_type::copy_ctr_counter()==1);
        auto t2 = t1;
        REQUIRE(storage_type::copy_ctr_counter()==1);
        REQUIRE(!t.is_same(t2));
        REQUIRE(t1.is_same(t2));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
}

TEMPLATE_TEST_CASE("test_tensor_cloning_semantics_view","[test_tensor_cloning_semantics]",
    gtensor::config::deep_semantics,
    gtensor::config::shallow_semantics
)
{
    using value_type = double;
    using gtensor::config::c_order;
    using gtensor::config::deep_semantics;
    using gtensor::config::shallow_semantics;
    using storage_type = test_tensor_cloning_semantics_::iterable_storage<value_type>;
    using semantics = TestType;
    using config_type = gtensor::config::extend_config_t<test_tensor_cloning_semantics_::test_config<semantics>,value_type>;
    using tensor_type = gtensor::tensor<value_type,c_order,config_type>;

    tensor_type t1{{1,2,3},{4,5,6}};
    tensor_type t2{7,8,9};
    storage_type::reset_counters();
    REQUIRE(storage_type::copy_ctr_counter()==0);
    REQUIRE(storage_type::move_ctr_counter()==0);
    REQUIRE(storage_type::custom_ctr_counter()==0);

    SECTION("expression_view")
    {
        auto v = (1+t1+t2)*2*(t1-t2);
        auto v1 = v;
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("expression_view1")
    {
        auto v = n_operator([](auto a, auto b, auto c){return a+b+c;},t1,value_type{1.1},t2);
        REQUIRE(storage_type::custom_ctr_counter()==1);
        auto v1 = v;
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
    }
    SECTION("transpose_view")
    {
        auto v = t1.transpose();
        auto v1 = v;
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("slice_view1")
    {
        auto v = t1({{},{1}});
        auto v1 = v;
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("slice_view2")
    {
        auto v = t1(0,1);
        auto v1 = v;
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("reshape_view")
    {
        auto v = t1.reshape(-1,1);
        auto v1 = v;
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("mapping_view1")
    {
        auto v = t1(gtensor::tensor<int>{0,1,0});
        auto v1 = v;
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("mapping_view2")
    {
        auto v = t1(std::vector<gtensor::tensor<int>>{gtensor::tensor<int>{0,1,0}});
        auto v1 = v;
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("mapping_view3")
    {
        auto v = t1(gtensor::tensor<bool>{{true,false,false},{false,true,true}});
        auto v1 = v;
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("composition")
    {
        auto v = (((t1+t2)*(t1-t2)).transpose() + t1.reshape(3,2))({{},{1}});
        auto v1 = v;
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
}

TEMPLATE_TEST_CASE("test_tensor_cloning_semantics_view_clone","[test_tensor_cloning_semantics]",
    gtensor::config::deep_semantics,
    gtensor::config::shallow_semantics
)
{
    using value_type = double;
    using gtensor::config::c_order;
    using gtensor::config::deep_semantics;
    using gtensor::config::shallow_semantics;
    using storage_type = test_tensor_cloning_semantics_::iterable_storage<value_type>;
    using semantics = TestType;
    using config_type = gtensor::config::extend_config_t<test_tensor_cloning_semantics_::test_config<semantics>,value_type>;
    using tensor_type = gtensor::tensor<value_type,c_order,config_type>;

    tensor_type t1{{1,2,3},{4,5,6}};
    tensor_type t2{7,8,9};
    auto v = (t1+t2)*(t1-t2);
    storage_type::reset_counters();
    REQUIRE(storage_type::copy_ctr_counter()==0);
    REQUIRE(storage_type::move_ctr_counter()==0);
    REQUIRE(storage_type::custom_ctr_counter()==0);

    SECTION("making_shallow_copy_with_shallow_semantic")
    {
        auto v1 = v.clone(shallow_semantics::value,shallow_semantics::value);
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("making_deep_copy_with_deep_semantic")
    {
        auto v1 = v.clone(deep_semantics::value,deep_semantics::value);
        auto v2 = v1;
        REQUIRE(!v.is_same(v1));
        REQUIRE(!v.is_same(v2));
        REQUIRE(!v1.is_same(v2));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("making_shallow_copy_with_deep_semantic")
    {
        auto v1 = v.clone(shallow_semantics::value,deep_semantics::value);
        auto v2 = v1;
        REQUIRE(v.is_same(v1));
        REQUIRE(!v.is_same(v2));
        REQUIRE(!v1.is_same(v2));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
    SECTION("making_deep_copy_with_shallow_semantic")
    {
        auto v1 = v.clone(deep_semantics::value,shallow_semantics::value);
        auto v2 = v1;
        REQUIRE(!v.is_same(v1));
        REQUIRE(!v.is_same(v2));
        REQUIRE(v1.is_same(v2));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
}

TEMPLATE_TEST_CASE("test_tensor_cloning_semantics_copy_eval","[test_tensor_cloning_semantics]",
    gtensor::config::deep_semantics,
    gtensor::config::shallow_semantics
)
{
    using value_type = double;
    using gtensor::config::c_order;
    using gtensor::config::deep_semantics;
    using gtensor::config::shallow_semantics;
    using storage_type = test_tensor_cloning_semantics_::iterable_storage<value_type>;
    using semantics = TestType;
    using config_type = gtensor::config::extend_config_t<test_tensor_cloning_semantics_::test_config<semantics>,value_type>;
    using tensor_type = gtensor::tensor<value_type,c_order,config_type>;

    tensor_type t{{1,2,3},{4,5,6}};
    storage_type::reset_counters();
    REQUIRE(storage_type::copy_ctr_counter()==0);
    REQUIRE(storage_type::move_ctr_counter()==0);
    REQUIRE(storage_type::custom_ctr_counter()==0);

    SECTION("copy_cloning_semantics")
    {
        auto t1 = t.copy();
        REQUIRE(!t1.is_same(t));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==1);
    }
    SECTION("eval_cloning_semantics")
    {
        auto t1 = t.eval();
        REQUIRE(t1.is_same(t));
        REQUIRE(storage_type::move_ctr_counter()==0);
        REQUIRE(storage_type::copy_ctr_counter()==0);
        REQUIRE(storage_type::custom_ctr_counter()==0);
    }
}

