#include "catch.hpp"
#include "expression_template_engine.hpp"
#include "tensor.hpp"

namespace test_cross_product{
struct A{};
struct B{};
struct C{};
struct D{};
struct E{};
struct F{};
template<typename F, typename S> struct type_pair{};

}   //end of namespace test_cross_product

namespace test_walker_types{
using gtensor::tensor;    

template<typename ValT, typename CfgT, typename ImplT>
class test_tensor : public tensor<ValT, CfgT, ImplT>
{
public:
    test_tensor(const tensor& base):
        tensor{base}
    {}
    auto impl(){return tensor::impl();}
};

template<typename ValT, typename CfgT, typename ImplT>
auto make_test_tensor(const tensor<ValT, CfgT, ImplT>& t){return test_tensor<ValT,CfgT,ImplT>{t};}

}



TEST_CASE("test_cross_product","[test_expression_template_engine]"){
    using gtensor::detail::cross_product;
    using gtensor::detail::type_list;
    using test_cross_product::type_pair;
    using test_cross_product::A;
    using test_cross_product::B;
    using test_cross_product::C;
    using test_cross_product::D;
    using test_cross_product::E;
    using test_cross_product::F;
    
            
    REQUIRE(std::is_same_v<
        cross_product<type_list, type_list<A,B,C>, type_list<D,E,F>>::type , 
        type_list<type_list<A,D>,type_list<A,E>,type_list<A,F>,type_list<B,D>,type_list<B,E>,type_list<B,F>,type_list<C,D>,type_list<C,E>,type_list<C,F>> >
    );
    REQUIRE(std::is_same_v<
        cross_product<type_pair, type_list<A,B,C>, type_list<D,E,F>>::type , 
        type_list<type_pair<A,D>,type_pair<A,E>,type_pair<A,F>,type_pair<B,D>,type_pair<B,E>,type_pair<B,F>,type_pair<C,D>,type_pair<C,E>,type_pair<C,F>> >
    );
    REQUIRE(std::is_same_v<
        cross_product<type_list, type_list<A,A,C>, type_list<D,E,E>>::type , 
        type_list<type_list<A,D>,type_list<A,E>,type_list<A,E>,type_list<A,D>,type_list<A,E>,type_list<A,E>,type_list<C,D>,type_list<C,E>,type_list<C,E>> >
    );
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<A>, type_list<B>>::type , type_list<type_list<A,B>>>);
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<>, type_list<B>>::type , type_list<>>);
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<>, type_list<>>::type , type_list<>>);
    REQUIRE(std::is_same_v<cross_product<type_list, type_list<A,B,C>, type_list<>>::type , type_list<>>);
}

TEST_CASE("test_walker_types","[test_expression_template_engine]"){
    using value_type = float;
    using gtensor::storage_walker;
    using gtensor::evaluating_walker;
    using gtensor::walker;
    using gtensor::detail::type_list;
    using gtensor::binary_operations::add;
    using gtensor::binary_operations::mul;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using test_walker_types::make_test_tensor;

    tensor_type t{1,2,3};
    auto test_t = make_test_tensor(t);
    REQUIRE(std::is_same_v<typename std::decay_t<decltype(test_t.impl()->engine())>::walker_types, type_list<storage_walker<value_type, config_type>>>);
        
    auto e1 = t+t;
    auto test_e1 = make_test_tensor(e1);
    REQUIRE(std::decay_t<decltype(test_e1.impl()->engine())>::walker_types::size == 2);
    REQUIRE(std::is_same_v<
        std::decay_t<decltype(test_e1.impl()->engine())>::walker_types, 
        type_list< 
            storage_walker<value_type, config_type>,
            evaluating_walker<value_type, config_type, add, storage_walker<value_type, config_type>, storage_walker<value_type, config_type> >>>
    );

    auto e2 = e1+e1;
    auto test_e2 = make_test_tensor(e2);
    REQUIRE(std::decay_t<decltype(test_e2.impl()->engine())>::walker_types::size == 5);
    
    auto e3 = e2+e2;
    auto test_e3 = make_test_tensor(e3);
    REQUIRE(std::decay_t<decltype(test_e3.impl()->engine())>::walker_types::size == 26);
    
    auto e4 = e3+e3;
    auto test_e4 = make_test_tensor(e4);
    REQUIRE(std::decay_t<decltype(test_e4.impl()->engine())>::walker_types::size == 1);
    REQUIRE(std::is_same_v<std::decay_t<decltype(test_e4.impl()->engine())>::walker_types, type_list<walker<value_type, config_type>>>);
    
    auto e5 = e4*e4;
    auto test_e5 = make_test_tensor(e5);
    REQUIRE(std::decay_t<decltype(test_e5.impl()->engine())>::walker_types::size == 2);
    REQUIRE(std::is_same_v<
        std::decay_t<decltype(test_e5.impl()->engine())>::walker_types, 
        type_list<
            storage_walker<value_type, config_type>,
            evaluating_walker<value_type, config_type, mul, walker<value_type, config_type>, walker<value_type, config_type>> >>
        );
}