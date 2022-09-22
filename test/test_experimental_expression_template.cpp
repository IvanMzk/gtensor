
#include "catch.hpp"
#include "experimental_expression_template.hpp"
#include "expression_template_variant_dispatch.hpp"
#include "expression_template_polytensor.hpp"

namespace test_cross_product{
struct A{};
struct B{};
struct C{};
struct D{};
struct E{};
struct F{};
template<typename F, typename S> struct type_pair{};

}   //end of namespace test_cross_product


TEST_CASE("test_cross_product","[test_experimental_expression_template]"){
    using expression_template_variant_dispatch::detail::cross_product;
    using expression_template_variant_dispatch::detail::type_list;
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

TEST_CASE("test_expression_template_polytensor","[test_experimental_expression_template]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using test_tensor_type = expression_template_polytensor::test_tensor<value_type,config_type>;
    using test_double_tensor_type = expression_template_polytensor::test_tensor<double,config_type>;

    test_double_tensor_type d1{1,2,3};
    test_tensor_type t1{{1,2,3}};
    test_tensor_type t2{{1},{2},{3}};
    test_tensor_type t3{-2};

    auto e_homo = t1+t2;    //float
    auto e_hetero = e_homo+d1;  //double
    e_hetero.begin();

    auto e = t2+t1+t2+t3;
    REQUIRE(!e.engine().is_trivial());
    REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));

    auto e_trivial_tree = t1+t1+t1+t1;
    REQUIRE(e_trivial_tree.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_tree.begin(), e_trivial_tree.end(), std::vector<float>{4,8,12}.begin()));

    auto e_trivial_tree1 = e_trivial_tree + e_trivial_tree;
    REQUIRE(e_trivial_tree1.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_tree1.begin(), e_trivial_tree1.end(), std::vector<float>{8,16,24}.begin()));

    auto e_trivial_subtree = e_trivial_tree + t2 + t3;
    REQUIRE(!e_trivial_subtree.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_subtree.begin(), e_trivial_subtree.end(), std::vector<float>{3,7,11,4,8,12,5,9,13}.begin()));
}

TEST_CASE("test_expression_template_polywalker","[test_experimental_expression_template]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using test_tensor_type = expression_template_polywalker::test_tensor<value_type,config_type>;

    test_tensor_type t1{{1,2,3}};
    test_tensor_type t2{{1},{2},{3}};
    test_tensor_type t3{-2};

    auto e = t2+t1+t2+t3;
    REQUIRE(!e.engine().is_trivial());
    REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));

    auto e_trivial_tree = t1+t1+t1+t1;
    REQUIRE(e_trivial_tree.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_tree.begin(), e_trivial_tree.end(), std::vector<float>{4,8,12}.begin()));

    auto e_trivial_tree1 = e_trivial_tree + e_trivial_tree;
    REQUIRE(e_trivial_tree1.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_tree1.begin(), e_trivial_tree1.end(), std::vector<float>{8,16,24}.begin()));

    auto e_trivial_subtree = e_trivial_tree + t2 + t3;
    REQUIRE(!e_trivial_subtree.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_subtree.begin(), e_trivial_subtree.end(), std::vector<float>{3,7,11,4,8,12,5,9,13}.begin()));
}

TEST_CASE("test_expression_template_without_dispatching","[test_experimental_expression_template]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using test_tensor_type = expression_template_without_dispatching::test_tensor<value_type,config_type>;

    test_tensor_type t1{{1,2,3}};
    test_tensor_type t2{{1},{2},{3}};
    test_tensor_type t3{-2};
    auto e = t2+t1+t2+t3;
    REQUIRE(!e.engine().is_trivial());
    REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));

    auto e_trivial_tree = t1+t1+t1+t1;
    REQUIRE(e_trivial_tree.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_tree.begin(), e_trivial_tree.end(), std::vector<float>{4,8,12}.begin()));

    auto e_trivial_tree1 = e_trivial_tree + e_trivial_tree;
    REQUIRE(e_trivial_tree1.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_tree1.begin(), e_trivial_tree1.end(), std::vector<float>{8,16,24}.begin()));

    auto e_trivial_subtree = e_trivial_tree + t2 + t3;
    REQUIRE(!e_trivial_subtree.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_subtree.begin(), e_trivial_subtree.end(), std::vector<float>{3,7,11,4,8,12,5,9,13}.begin()));
}

TEST_CASE("expression_template_dispatch_in_walker","[test_experimental_expression_template]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using test_tensor_type = expression_template_dispatch_in_walker::test_tensor<value_type,config_type>;

    test_tensor_type t1{{1,2,3}};
    test_tensor_type t2{{1},{2},{3}};
    test_tensor_type t3{-2};
    auto e = t2+t1+t2+t3;
    REQUIRE(!e.engine().is_trivial());
    REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));

    auto e_trivial_tree = t1+t1+t1+t1;
    REQUIRE(e_trivial_tree.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_tree.begin(), e_trivial_tree.end(), std::vector<float>{4,8,12}.begin()));

    auto e_trivial_tree1 = e_trivial_tree + e_trivial_tree;
    REQUIRE(e_trivial_tree1.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_tree1.begin(), e_trivial_tree1.end(), std::vector<float>{8,16,24}.begin()));

    auto e_trivial_subtree = e_trivial_tree + t2 + t3;
    REQUIRE(!e_trivial_subtree.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_subtree.begin(), e_trivial_subtree.end(), std::vector<float>{3,7,11,4,8,12,5,9,13}.begin()));
}

TEST_CASE("test_expression_template_variant_dispatch","[test_experimental_expression_template]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using test_tensor_type = expression_template_variant_dispatch::test_tensor<std::integral_constant<std::size_t,3>,value_type,config_type>;
    using gtensor::multiindex_iterator;
    using expression_template_variant_dispatch::detail::type_list;
    using gtensor::binary_operations::add;
    using gtensor::evaluating_walker;
    using gtensor::evaluating_trivial_root_walker;
    using gtensor::storage_walker;
    using gtensor::walker;
    using gtensor::storage_trivial_walker;

    SECTION("test_simple_not_trivial_tree"){
        test_tensor_type t1{1,2,3};
        test_tensor_type t2{1};
        auto e = t1+t2;
        REQUIRE(!e.engine().is_trivial());
        REQUIRE(decltype(e.engine())::walker_types::size == 2);
        std::visit(
            [&e](auto&& w){
                using iterator_type = multiindex_iterator<value_type,config_type,std::decay_t<decltype(w)>>;
                auto it_begin = e.begin(w);
                REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{2,3,4}.begin()));
            },
            e.engine().create_walker()
        );
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{2,3,4}.begin()));
    }
    SECTION("test_not_trivial_tree"){
        test_tensor_type t1{{1,2,3}};
        test_tensor_type t2{{1},{2},{3}};
        test_tensor_type t3{-2};

        auto e1 = t2+t1+t2;     //{3,4,5,5,6,7,7,8,9}
        auto e2 = e1+e1;    //{6,8,10,10,12,14,14,16,18}
        auto e3 = e2+e2;     //{12,16,20,20,24,28,28,32,36}
        auto e4 = e3+e3;     //{24,32,40,40,48,56,56,64,72}
        auto e5 = e4+e4;     //{24,32,40,40,48,56,56,64,72}
        auto e = e5;
        REQUIRE(!e.engine().is_trivial());
        std::visit(
            [&e](auto&& w){
                auto it_begin = e.begin(w);
                REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{48,64,80,80,96,112,112,128,144}.begin()));
                //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{24,32,40,40,48,56,56,64,72}.begin()));
                //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{12,16,20,20,24,28,28,32,36}.begin()));
                //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{6,8,10,10,12,14,14,16,18}.begin()));
                //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
            },
            e.engine().create_walker()
        );
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{48,64,80,80,96,112,112,128,144}.begin()));
    }

    SECTION("trivial_tree"){
        test_tensor_type t1{{1,2,3}};
        auto e1 = t1+t1+t1;     //{3,6,9}
        auto e2 = e1+e1;    //{6,12,18}
        auto e3 = e2+e2;    //{12,24,36}
        auto e4 = e3+e3;    //{24,48,72}
        auto e5 = e4+e4;    //{48,96,144}
        auto e = e5;
        REQUIRE(e.engine().is_trivial());
        std::visit(
            [&e](auto&& w){
                auto it_begin = e.begin(w);
                REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{48,96,144}.begin()));
                //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{12,24,36}.begin()));
            },
            e.engine().create_walker()
        );
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{48,96,144}.begin()));
    }

    SECTION("trivial_subtree"){
        test_tensor_type t1{{1,2,3}};
        test_tensor_type t2{{1},{2},{3}};
        test_tensor_type t3{-2};
        auto e1 = t1+t1+t1;     //{3,6,9}
        auto e2 = e1+e1;    //{6,12,18}
        auto e3 = e2+e2;    //{12,24,36}
        auto e4 = e3+e3;    //{24,48,72}
        auto e5 = e4+e4;    //{48,96,144}
        REQUIRE(e5.engine().is_trivial());
        auto e6 = e5+(t2+t3);    //{{47,95,143},{48,96,144},{49,97,145}}
        auto e = e6;
        REQUIRE(!e.engine().is_trivial());
        std::visit(
            [&e](auto&& w){
                auto it_begin = e.begin(w);
                REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{47,95,143,48,96,144,49,97,145}.begin()));
                //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{48,96,144}.begin()));
                //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{12,24,36}.begin()));
            },
            e.engine().create_walker()
        );
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{47,95,143,48,96,144,49,97,145}.begin()));
    }

}