#include <tuple>
#include "catch.hpp"
#include "gtensor.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"
#include "expression_template_test_helpers.hpp"


TEST_CASE("test_is_trivial","[test_expression_template_engine]"){
    using value_type = float;
    using tensor_type = gtensor::tensor<value_type, test_expression_template_helpers::test_default_config_type>;
    using test_expression_template_helpers::make_test_tensor;
    using test_type = std::tuple<bool, bool>;
    //0result is_trivial,1expected_is_trivial
    auto test_data = GENERATE(
                                test_type(make_test_tensor(tensor_type{1}+tensor_type{1}).is_trivial(), true),
                                test_type(make_test_tensor(tensor_type{1,2,3,4,5}+tensor_type{1}).is_trivial(), false),
                                test_type(make_test_tensor(tensor_type{1,2,3,4,5}+tensor_type{1,2,3,4,5}).is_trivial(), true),
                                test_type(make_test_tensor(tensor_type{{1},{2},{3},{4},{5}}+tensor_type{{1,2,3,4,5}}).is_trivial(), false),
                                test_type(make_test_tensor(tensor_type{1,2,3}+tensor_type{1,2,3}+tensor_type{3,4,5}+tensor_type{3,4,5}).is_trivial(), true),
                                test_type(make_test_tensor((tensor_type{1,2,3}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5})).is_trivial(), true),
                                test_type(make_test_tensor((tensor_type{1}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5})).is_trivial(), false)
                            );

    auto result_is_trivial = std::get<0>(test_data);
    auto expected_is_trivial = std::get<1>(test_data);
    REQUIRE(result_is_trivial == expected_is_trivial);
}

TEMPLATE_LIST_TEST_CASE("test_broadcast_walker","[test_expression_template_engine]",
    typename test_expression_template_helpers::makers_type_list<test_expression_template_helpers::test_tensor>::type
)
{
    using value_type = typename TestType::value_type;
    using test_type = std::tuple<value_type,value_type>;

    auto test_data = GENERATE(
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); return *w; }(), value_type{1}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(2,1); return *w;}(), value_type{1}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(1,1); return *w;}(), value_type{4}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(0,1); return *w;}(), value_type{2}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(1,1); w.walk(0,2); return *w;}(), value_type{6}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(3,1); w.walk(0,2); return *w;}(), value_type{3}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(1,1); w.reset(1); return *w;}(), value_type{1}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(1,1); w.walk(0,2); w.reset(2); return *w;}(), value_type{6}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(1,1); w.walk(0,2); w.reset(3); return *w;}(), value_type{6}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(1,1); w.walk(0,2); w.reset(1); return *w;}(), value_type{3}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(1,1); w.walk(0,2); w.reset(0); return *w;}(), value_type{4}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_broadcast_walker(); w.walk(1,1); w.walk(0,2); w.reset(); return *w;}(), value_type{1}}
    );
    auto deref = std::get<0>(test_data);
    auto expected_deref = std::get<1>(test_data);
    REQUIRE(deref == expected_deref);
}

TEMPLATE_LIST_TEST_CASE("test_trivial_walker","[test_expression_template_engine]",
    typename test_expression_template_helpers::makers_trivial_type_list<test_expression_template_helpers::test_tensor>::type
)
{
    using value_type = typename TestType::value_type;
    using test_type = std::tuple<value_type,value_type>;
    REQUIRE(TestType{}().is_trivial());
    auto test_data = GENERATE(
        test_type{[](){auto t = TestType{}(); auto w = t.create_trivial_walker(); return w[0]; }(), value_type{1}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_trivial_walker(); return w[4];}(), value_type{5}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_trivial_walker(); return w[0];}(), value_type{1}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_trivial_walker(); return w[3];}(), value_type{4}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_trivial_walker(); return w[1];}(), value_type{2}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_trivial_walker(); return w[2];}(), value_type{3}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_trivial_walker(); return w[5];}(), value_type{6}},
        test_type{[](){auto t = TestType{}(); auto w = t.create_trivial_walker(); return w[5];}(), value_type{6}}
    );
    auto deref = std::get<0>(test_data);
    auto expected_deref = std::get<1>(test_data);
    REQUIRE(deref == expected_deref);
}

TEMPLATE_TEST_CASE("test_indexer","[test_expression_template_engine]",
    test_expression_template_helpers::storage_tensor_maker<>,
    test_expression_template_helpers::notrivial_tensor_maker<>,
    test_expression_template_helpers::trivial_subtree_tensor_maker<>,
    test_expression_template_helpers::trivial_tensor_maker<>
)
{
    using value_type = typename TestType::value_type;
    using test_type = std::tuple<value_type,value_type>;

    //0result,1expected
    auto test_data = GENERATE(
        test_type{TestType{}().create_indexer()[0],value_type{1}},
        test_type{TestType{}().create_indexer()[5],value_type{6}},
        test_type{TestType{}().create_indexer()[1],value_type{2}},
        test_type{TestType{}().create_indexer()[2],value_type{3}},
        test_type{TestType{}().create_indexer()[4],value_type{5}},
        test_type{TestType{}().create_indexer()[3],value_type{4}}
    );
    auto result = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    REQUIRE(result == expected);
}

TEMPLATE_TEST_CASE("test_viewing_indexer","[test_expression_template_engine]",
    test_expression_template_helpers::view_slice_of_storage_maker<>,
    test_expression_template_helpers::view_slice_of_eval_maker<>,
    test_expression_template_helpers::view_view_slice_of_eval_maker<>
)
{
    using value_type = typename TestType::value_type;
    using test_type = std::tuple<value_type,value_type>;

    //0result,1expected
    auto test_data = GENERATE(
        test_type{TestType{}().create_indexer()[0],value_type{0}},
        test_type{TestType{}().create_indexer()[1],value_type{0}},
        test_type{TestType{}().create_indexer()[2],value_type{0}},
        test_type{TestType{}().create_indexer()[3],value_type{0}},
        test_type{TestType{}().create_indexer()[4],value_type{0}},
        test_type{TestType{}().create_indexer()[5],value_type{0}},
        test_type{TestType{}().create_indexer()[6],value_type{0}},
        test_type{TestType{}().create_indexer()[7],value_type{0}},
        test_type{TestType{}().create_indexer()[8],value_type{0}},
        test_type{TestType{}().create_indexer()[9],value_type{0}},
        test_type{TestType{}().create_indexer()[10],value_type{0}},
        test_type{TestType{}().create_indexer()[11],value_type{0}},
        test_type{TestType{}().create_indexer()[12],value_type{1}},
        test_type{TestType{}().create_indexer()[13],value_type{0}},
        test_type{TestType{}().create_indexer()[14],value_type{2}},
        test_type{TestType{}().create_indexer()[15],value_type{0}},
        test_type{TestType{}().create_indexer()[16],value_type{3}},
        test_type{TestType{}().create_indexer()[17],value_type{0}},
        test_type{TestType{}().create_indexer()[18],value_type{4}},
        test_type{TestType{}().create_indexer()[19],value_type{0}},
        test_type{TestType{}().create_indexer()[20],value_type{5}},
        test_type{TestType{}().create_indexer()[21],value_type{0}},
        test_type{TestType{}().create_indexer()[22],value_type{6}},
        test_type{TestType{}().create_indexer()[23],value_type{0}}
    );
    auto result = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    REQUIRE(result == expected);
}

TEST_CASE("test_result_type","[test_expression_template_engine]"){
    using value_type = float;
    using reference_type = value_type&;
    using const_reference_type = const value_type&;
    using test_expression_template_helpers::test_tensor;
    using test_expression_template_helpers::make_test_tensor;
    using test_config_type = typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type;
    using index_type = typename test_config_type::index_type;
    using tensor_type = gtensor::tensor<value_type, test_config_type>;

    auto t = make_test_tensor<test_tensor>(tensor_type{1,2,3});
    SECTION("storage"){
        auto& re = t.engine();
        const auto& cre = re;
        REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),reference_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),reference_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),reference_type>);
        REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(*re.begin()),reference_type>);
        REQUIRE(std::is_same_v<decltype(*re.end()),reference_type>);
        REQUIRE(std::is_same_v<decltype(*cre.begin()),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(*cre.end()),const_reference_type>);
    }
    SECTION("view_of_storage"){
        auto vt = make_test_tensor<test_tensor>(t({{}}));
        auto& re = vt.engine();
        const auto& cre = re;
        REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),reference_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),reference_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),reference_type>);
        REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(*re.begin()),reference_type>);
        REQUIRE(std::is_same_v<decltype(*re.end()),reference_type>);
        REQUIRE(std::is_same_v<decltype(*cre.begin()),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(*cre.end()),const_reference_type>);
    }
    SECTION("view_view_of_storage"){
        auto vvt = make_test_tensor<test_tensor>(t({{}}).transpose().reshape());
        auto& re = vvt.engine();
        const auto& cre = re;
        REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),reference_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),reference_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),reference_type>);
        REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(*re.begin()),reference_type>);
        REQUIRE(std::is_same_v<decltype(*re.end()),reference_type>);
        REQUIRE(std::is_same_v<decltype(*cre.begin()),const_reference_type>);
        REQUIRE(std::is_same_v<decltype(*cre.end()),const_reference_type>);
    }
    SECTION("evaluating"){
        auto e = make_test_tensor<test_tensor>(t+t+t);
        auto& re = e.engine();
        const auto& cre = re;
        REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),value_type>);
        REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),value_type>);
        REQUIRE(std::is_same_v<decltype(*re.begin()),value_type>);
        REQUIRE(std::is_same_v<decltype(*re.end()),value_type>);
        REQUIRE(std::is_same_v<decltype(*cre.begin()),value_type>);
        REQUIRE(std::is_same_v<decltype(*cre.end()),value_type>);
    }
    SECTION("view_of_evaluating"){
        auto ve = make_test_tensor<test_tensor>((t+t+t).transpose());
        auto& re = ve.engine();
        const auto& cre = re;
        REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),value_type>);
        REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),value_type>);
        REQUIRE(std::is_same_v<decltype(*re.begin()),value_type>);
        REQUIRE(std::is_same_v<decltype(*re.end()),value_type>);
        REQUIRE(std::is_same_v<decltype(*cre.begin()),value_type>);
        REQUIRE(std::is_same_v<decltype(*cre.end()),value_type>);
    }
    SECTION("view_view_of_evaluating"){
        auto vve = make_test_tensor<test_tensor>((t+t+t).transpose().reshape()({{}})());
        auto& re = vve.engine();
        const auto& cre = re;
        REQUIRE(std::is_same_v<decltype(re.create_indexer()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_indexer()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(re.create_trivial_walker()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(cre.create_trivial_walker()[std::declval<index_type>()]),value_type>);
        REQUIRE(std::is_same_v<decltype(*re.create_broadcast_walker()),value_type>);
        REQUIRE(std::is_same_v<decltype(*cre.create_broadcast_walker()),value_type>);
        REQUIRE(std::is_same_v<decltype(*re.begin()),value_type>);
        REQUIRE(std::is_same_v<decltype(*re.end()),value_type>);
        REQUIRE(std::is_same_v<decltype(*cre.begin()),value_type>);
        REQUIRE(std::is_same_v<decltype(*cre.end()),value_type>);
    }
}

namespace test_iterator_helpers{
using test_iterator_makers = typename helpers_for_testing::list_concat<
        typename test_expression_template_helpers::makers_type_list<test_expression_template_helpers::test_multiindex_iterator_tensor>::type,
        typename test_expression_template_helpers::makers_trivial_type_list<test_expression_template_helpers::test_flatindex_iterator_tensor>::type
        >::type;
}

TEMPLATE_LIST_TEST_CASE("test_multiindex_iterator","[test_expression_template_engine]",test_iterator_helpers::test_iterator_makers)
{
    SECTION("test_iter_deref"){
        using value_type = typename TestType::value_type;
        using test_type = std::tuple<value_type, value_type>;
        //0deref,1expected_deref
        auto test_data = GENERATE(
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); return *it;}(), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; return *it;}(), value_type{6}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; --it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; --it; --it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; --it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; --it; --it; return *it;}(), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; return *it;}(), value_type{6}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; return *it;}(), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; ++it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; ++it; ++it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; ++it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; ++it; ++it; return *it;}(), value_type{6}},
            test_type{TestType{}().begin()[0], value_type{1}},
            test_type{TestType{}().begin()[1], value_type{2}},
            test_type{TestType{}().begin()[2], value_type{3}},
            test_type{TestType{}().begin()[3], value_type{4}},
            test_type{TestType{}().begin()[4], value_type{5}},
            test_type{TestType{}().begin()[5], value_type{6}},
            test_type{TestType{}().end()[-1], value_type{6}},
            test_type{TestType{}().end()[-2], value_type{5}},
            test_type{TestType{}().end()[-3], value_type{4}},
            test_type{TestType{}().end()[-4], value_type{3}},
            test_type{TestType{}().end()[-5], value_type{2}},
            test_type{TestType{}().end()[-6], value_type{1}},
            test_type{*(TestType{}().begin()+0), value_type{1}},
            test_type{*(TestType{}().begin()+1), value_type{2}},
            test_type{*(TestType{}().begin()+2), value_type{3}},
            test_type{*(TestType{}().begin()+3), value_type{4}},
            test_type{*(TestType{}().begin()+4), value_type{5}},
            test_type{*(TestType{}().begin()+5), value_type{6}},
            test_type{*(TestType{}().end()-1), value_type{6}},
            test_type{*(TestType{}().end()-2), value_type{5}},
            test_type{*(TestType{}().end()-3), value_type{4}},
            test_type{*(TestType{}().end()-4), value_type{3}},
            test_type{*(TestType{}().end()-5), value_type{2}},
            test_type{*(TestType{}().end()-6), value_type{1}},
            test_type{*(TestType{}().begin()+=0), value_type{1}},
            test_type{*(TestType{}().begin()+=1), value_type{2}},
            test_type{*(TestType{}().begin()+=2), value_type{3}},
            test_type{*(TestType{}().begin()+=3), value_type{4}},
            test_type{*(TestType{}().begin()+=4), value_type{5}},
            test_type{*(TestType{}().begin()+=5), value_type{6}},
            test_type{*(TestType{}().end()-=1), value_type{6}},
            test_type{*(TestType{}().end()-=2), value_type{5}},
            test_type{*(TestType{}().end()-=3), value_type{4}},
            test_type{*(TestType{}().end()-=4), value_type{3}},
            test_type{*(TestType{}().end()-=5), value_type{2}},
            test_type{*(TestType{}().end()-=6), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it;  return *(it-2);}(), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin(); ++it; ++it;  return *(it+2);}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it;  return *(it+1);}(), value_type{6}},
            test_type{[](){auto t = TestType{}(); auto it = t.end(); --it; --it;  return *(it-2);}(), value_type{3}}
        );
        auto deref = std::get<0>(test_data);
        auto expected_deref = std::get<1>(test_data);
        REQUIRE(deref == expected_deref);
    }
    SECTION("test_iter_cmp"){
        using test_type = std::tuple<bool,bool>;
        //0cmp_result,1expected_cmp_result
        auto test_data = GENERATE(
            test_type{[](){auto t = TestType{}(); return t.begin() > t.end();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end() > t.begin();}(), true},
            test_type{[](){auto t = TestType{}(); return t.begin() < t.end();}(), true},
            test_type{[](){auto t = TestType{}(); return t.end() < t.begin();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end() > t.end();}(), false},
            test_type{[](){auto t = TestType{}(); return t.begin() > t.begin();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end() < t.end();}(), false},
            test_type{[](){auto t = TestType{}(); return t.begin() < t.begin();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end() >= t.end();}(), true},
            test_type{[](){auto t = TestType{}(); return t.begin() >= t.begin();}(), true},
            test_type{[](){auto t = TestType{}(); return t.end() <= t.end();}(), true},
            test_type{[](){auto t = TestType{}(); return t.begin() <= t.begin();}(), true}
        );
        auto cmp_result = std::get<0>(test_data);
        auto expected_cmp_result = std::get<1>(test_data);
        REQUIRE(cmp_result == expected_cmp_result);
    }
}

TEST_CASE("has_view_with_converting_descriptor","[test_expression_template_engine]"){
    using value_type = float;
    using tensor_type = gtensor::tensor<value_type, test_expression_template_helpers::test_default_config_type>;
    using gtensor::detail::has_view_with_converting_descriptor;
    using test_expression_template_helpers::make_test_tensor;
    using test_type = std::tuple<bool,bool>;
    auto test_data = GENERATE(
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}).engine())>>::value,false},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}+tensor_type{1}).engine())>>::value,false},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}+tensor_type{1}+tensor_type{1}).engine())>>::value,false},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor((tensor_type{1}+tensor_type{1})+(tensor_type{1}+tensor_type{1})).engine())>>::value,false},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}()).engine())>>::value,false},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}()+tensor_type{1}()+tensor_type{1}()).engine())>>::value,false},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1} + tensor_type{1}()+ tensor_type{1}).engine())>>::value,false},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}.reshape() + tensor_type{1} + tensor_type{1}() + tensor_type{1}()).engine())>>::value,false},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}({{}})).engine())>>::value,true},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}({{}}) + tensor_type{1}({{}}) + tensor_type{1}({{}})).engine())>>::value,true},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}({{}}) + tensor_type{1} + tensor_type{1}).engine())>>::value,true},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor((tensor_type{1}() + tensor_type{1}({{}})) + (tensor_type{1}.reshape() + tensor_type{1})).engine())>>::value,true},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}.transpose()).engine())>>::value,true},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1}.transpose() + tensor_type{1}.transpose() + tensor_type{1}.transpose()).engine())>>::value,true},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1} + tensor_type{1}.transpose() + tensor_type{1}).engine())>>::value,true},
        test_type{has_view_with_converting_descriptor<std::decay_t<decltype(make_test_tensor(tensor_type{1} + tensor_type{1}.transpose() + tensor_type{1}({{}}) + tensor_type{1}).engine())>>::value,true}
    );

    auto result = std::get<0>(test_data);
    auto expected = std::get<1>(test_data);
    REQUIRE(result == expected);
}

TEST_CASE("test_broadcast_assignment","[test_expression_template_engine]"){
    using value_type = float;
    using reference_type = value_type&;
    using const_reference_type = const value_type&;
    using test_expression_template_helpers::test_tensor;
    using test_expression_template_helpers::make_test_tensor;
    using test_config_type = typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type;
    using index_type = typename test_config_type::index_type;
    using tensor_type = gtensor::tensor<value_type, test_config_type>;
    using gtensor::broadcast_exception;

    //tensor assignment
    auto t_lhs = tensor_type();
    t_lhs = tensor_type{1,2,3};
    REQUIRE(std::equal(t_lhs.begin(), t_lhs.end(), std::vector<float>{1,2,3}.begin()));

    //broadcast assignment
    auto lhs = tensor_type{1,2,3,4,5};
    lhs({{{},{},2}}) = tensor_type{0};
    REQUIRE(std::equal(lhs.begin(), lhs.end(), std::vector<float>{0,2,0,4,0}.begin()));

    auto lhs1 = tensor_type{{1,2,3},{4,5,6}};
    lhs1() = tensor_type{0,1,2};
    REQUIRE(std::equal(lhs1.begin(), lhs1.end(), std::vector<float>{0,1,2,0,1,2}.begin()));

    auto lhs2 = tensor_type{{1,2,3},{4,5,6}};
    lhs2(1) = tensor_type{0,1,2};
    REQUIRE(std::equal(lhs2.begin(), lhs2.end(), std::vector<float>{1,2,3,0,1,2}.begin()));

    auto lhs4 = tensor_type{1,2,3,4,5};
    lhs4({{{},{},-1}})({{{},{},2}}) = tensor_type{0,1,2};
    REQUIRE(std::equal(lhs4.begin(), lhs4.end(), std::vector<float>{2,2,1,4,0}.begin()));

    //broadcast assignment exception, every element of lhs must be assigned only once
    auto lhs3 = tensor_type{0};
    REQUIRE_THROWS_AS((lhs3() = tensor_type{0,1,2}), broadcast_exception);

    //not compile, broadcast assignment to evaluating tensor
    //lhs+lhs = tensor_type{1};

    //not compile, broadcast assignment to view of evaluating tensor
    // auto e = lhs+lhs;
    // e() = tensor_type{1};
}