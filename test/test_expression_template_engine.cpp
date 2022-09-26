#include <tuple>
#include "catch.hpp"
#include "gtensor.hpp"
#include "test_config.hpp"
#include "helpers_for_testing.hpp"

namespace test_expression_template_helpers{

using value_type = float;
using gtensor::tensor;
using test_config_div_native_type = typename test_config::config_engine_div_selector<gtensor::config::engine_expression_template,gtensor::config::mode_div_native>::config_type;
using test_config_div_libdivide_type = typename test_config::config_engine_div_selector<gtensor::config::engine_expression_template,gtensor::config::mode_div_libdivide>::config_type;
using test_default_config_type = test_config_div_native_type;
//using test_default_config_type = test_config_div_libdivide_type;
using gtensor::expression_template_storage_engine;
using gtensor::multiindex_iterator;
using gtensor::flat_index_iterator;
test_default_config_type::nop_type nop{};

template<typename EngineT> auto begin_broadcast_helper(const EngineT& engine){
    using iterator_type = multiindex_iterator<typename EngineT::value_type,typename EngineT::config_type,decltype(engine.create_broadcast_walker())>;
    return iterator_type{engine.create_broadcast_walker(), engine.host()->shape(), engine.host()->descriptor().as_descriptor_with_libdivide()->strides_libdivide()};
}
template<typename EngineT> auto end_broadcast_helper(const EngineT& engine){
    using iterator_type = multiindex_iterator<typename EngineT::value_type,typename EngineT::config_type,decltype(engine.create_broadcast_walker())>;
    return iterator_type{engine.create_broadcast_walker(), engine.host()->shape(), engine.host()->descriptor().as_descriptor_with_libdivide()->strides_libdivide(), engine.host()->size()};
}
template<typename...Ts> auto begin_broadcast_helper(const expression_template_storage_engine<Ts...>& engine){return engine.begin();}
template<typename...Ts> auto end_broadcast_helper(const expression_template_storage_engine<Ts...>& engine){return engine.end();}

template<typename EngineT> auto begin_trivial_helper(const EngineT& engine){
    using iterator_type = flat_index_iterator<typename EngineT::value_type,typename EngineT::config_type,decltype(engine.create_trivial_walker())>;
    return iterator_type{engine.create_trivial_walker()};
}
template<typename EngineT> auto end_trivial_helper(const EngineT& engine){
    using iterator_type = flat_index_iterator<typename EngineT::value_type,typename EngineT::config_type,decltype(engine.create_trivial_walker())>;
    return iterator_type{engine.create_trivial_walker(), engine.host()->size()};
}

template<typename T>
struct test_tensor : public T{
    test_tensor(const T& base):
        tensor{base}
    {}
    auto& engine()const{return impl()->engine();}
    bool is_trivial()const{return engine().is_trivial();}
    auto create_broadcast_walker()const{return engine().create_broadcast_walker();}
    auto create_trivial_walker()const{return engine().create_trivial_walker();}
    auto create_indexer()const{return engine().create_indexer();}
    auto begin_broadcast()const{return begin_broadcast_helper(engine());}
    auto end_broadcast()const{return end_broadcast_helper(engine());}
    auto begin_trivial()const{return begin_trivial_helper(engine());}
    auto end_trivial()const{return end_trivial_helper(engine());}
};

template<typename T>
auto make_test_tensor(T&& t){return test_tensor<std::decay_t<T>>{t};}

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct storage_tensor_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{{{1,2,3},{4,5,6}}}
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct notrivial_tensor_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{{{0},{3}}} + tensor_type{1,2,3}
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct trivial_subtree_tensor_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{2} * tensor_type{-1,-1,-1} + (tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{0,0,0},{3,3,3}}}) + tensor_type{5,5,5} - tensor_type{3}
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct trivial_tensor_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{{{-1,-1,-1},{-1,-1,-1}}} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{1,1,1},{4,4,4}}}
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct view_slice_of_storage_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{1,0,2,0,3,0},{4,0,5,0,6,0}}}({{1,2},{},{nop,nop,2}})
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct view_slice_of_eval_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            (tensor_type{2} * tensor_type{{1,1,1,1,1,1}} + tensor_type{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{1,0,2,0,3,0},{4,0,5,0,6,0}}} - tensor_type{{{3,3,3,3,3,3}}} + tensor_type{1})({{1,2},{},{nop,nop,2}})
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct view_view_slice_of_eval_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            (tensor_type{2} * tensor_type{{1,1,1,1,1,1}} + tensor_type{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{0,3,0,2,0,1},{0,6,0,5,0,4}}} - tensor_type{{{3,3,3,3,3,3}}} + tensor_type{1})({{},{},{nop,nop,-1}})({{1,2},{},{nop,nop,2}})
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct view_transpose_of_storage_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{{{1},{4}},{{2},{5}},{{3},{6}}}.transpose()
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct view_subdim_of_storage_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{{{{0,0,0},{0,0,0}}},{{{1,2,3},{4,5,6}}}}(1)
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct view_reshape_of_storage_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{{{1},{2},{3}},{{4},{5},{6}}}.reshape(1,2,3)
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct eval_view_operand_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            view_transpose_of_storage_maker<CfgT>{}() +
            storage_tensor_maker<CfgT>{}() -
            trivial_subtree_tensor_maker<CfgT>{}() +
            view_view_slice_of_eval_maker<CfgT>{}() -
            trivial_tensor_maker<CfgT>{}()
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct trivial_view_operand_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            (storage_tensor_maker<CfgT>{}() +
            view_transpose_of_storage_maker<CfgT>{}()) +
            (storage_tensor_maker<CfgT>{}() -
            view_view_slice_of_eval_maker<CfgT>{}() -
            trivial_tensor_maker<CfgT>{}())
        );
    }
};

template<typename CfgT = test_default_config_type, typename ValT = value_type>
struct view_eval_view_operand_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            (view_transpose_of_storage_maker<CfgT>{}() +
                storage_tensor_maker<CfgT>{}() -
                trivial_subtree_tensor_maker<CfgT>{}() +
                view_view_slice_of_eval_maker<CfgT>{}() -
                trivial_tensor_maker<CfgT>{}() +
                tensor_type{{{0,0,0},{0,0,0}},{{2,0,-2},{2,0,-2}}}
            )({{1},{},{nop,nop,-1}})
        );
    }
};


using makers_type_list = std::tuple<
    test_expression_template_helpers::storage_tensor_maker<test_config_div_native_type>,
    test_expression_template_helpers::notrivial_tensor_maker<test_config_div_native_type>,
    test_expression_template_helpers::trivial_subtree_tensor_maker<test_config_div_native_type>,
    test_expression_template_helpers::trivial_tensor_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_slice_of_storage_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_slice_of_eval_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_view_slice_of_eval_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_transpose_of_storage_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_subdim_of_storage_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_reshape_of_storage_maker<test_config_div_native_type>,
    test_expression_template_helpers::eval_view_operand_maker<test_config_div_native_type>,
    test_expression_template_helpers::trivial_view_operand_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_eval_view_operand_maker<test_config_div_native_type>,

    test_expression_template_helpers::storage_tensor_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::notrivial_tensor_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::trivial_subtree_tensor_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::trivial_tensor_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_slice_of_storage_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_slice_of_eval_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_view_slice_of_eval_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_transpose_of_storage_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_subdim_of_storage_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_reshape_of_storage_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::eval_view_operand_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::trivial_view_operand_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_eval_view_operand_maker<test_config_div_libdivide_type>
>;

using makers_trivial_type_list = std::tuple<
    test_expression_template_helpers::storage_tensor_maker<test_config_div_native_type>,
    test_expression_template_helpers::trivial_tensor_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_slice_of_storage_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_slice_of_eval_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_view_slice_of_eval_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_transpose_of_storage_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_subdim_of_storage_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_reshape_of_storage_maker<test_config_div_native_type>,
    test_expression_template_helpers::trivial_view_operand_maker<test_config_div_native_type>,
    test_expression_template_helpers::view_eval_view_operand_maker<test_config_div_native_type>,

    test_expression_template_helpers::storage_tensor_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::trivial_tensor_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_slice_of_storage_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_slice_of_eval_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_view_slice_of_eval_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_transpose_of_storage_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_subdim_of_storage_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_reshape_of_storage_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::trivial_view_operand_maker<test_config_div_libdivide_type>,
    test_expression_template_helpers::view_eval_view_operand_maker<test_config_div_libdivide_type>
>;

}   //end of namespace test_tensor

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

TEMPLATE_LIST_TEST_CASE("test_broadcast_walker","[test_expression_template_engine]", test_expression_template_helpers::makers_type_list)
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

TEMPLATE_LIST_TEST_CASE("test_trivial_walker","[test_expression_template_engine]", test_expression_template_helpers::makers_trivial_type_list)
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

TEMPLATE_LIST_TEST_CASE("test_multiindex_iterator","[test_expression_template_engine]",test_expression_template_helpers::makers_type_list)
{
    SECTION("test_iter_deref"){
        using value_type = typename TestType::value_type;
        using test_type = std::tuple<value_type, value_type>;
        //0deref,1expected_deref
        auto test_data = GENERATE(
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); return *it;}(), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); ++it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); ++it; ++it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); ++it; ++it; ++it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); ++it; ++it; ++it; ++it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); ++it; ++it; ++it; ++it; ++it; return *it;}(), value_type{6}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); ++it; ++it; ++it; ++it; ++it; --it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); ++it; ++it; ++it; ++it; ++it; --it; --it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; --it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_broadcast(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; --it; --it; return *it;}(), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; return *it;}(), value_type{6}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; --it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; --it; --it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; --it; --it; --it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; --it; --it; --it; --it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; --it; --it; --it; --it; --it; return *it;}(), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; --it; --it; --it; --it; --it; ++it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; --it; --it; --it; --it; --it; ++it; ++it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; ++it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_broadcast(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; ++it; ++it; return *it;}(), value_type{6}},
            test_type{TestType{}().begin_broadcast()[0], value_type{1}},
            test_type{TestType{}().begin_broadcast()[1], value_type{2}},
            test_type{TestType{}().begin_broadcast()[2], value_type{3}},
            test_type{TestType{}().begin_broadcast()[3], value_type{4}},
            test_type{TestType{}().begin_broadcast()[4], value_type{5}},
            test_type{TestType{}().begin_broadcast()[5], value_type{6}},
            test_type{TestType{}().end_broadcast()[-1], value_type{6}},
            test_type{TestType{}().end_broadcast()[-2], value_type{5}},
            test_type{TestType{}().end_broadcast()[-3], value_type{4}},
            test_type{TestType{}().end_broadcast()[-4], value_type{3}},
            test_type{TestType{}().end_broadcast()[-5], value_type{2}},
            test_type{TestType{}().end_broadcast()[-6], value_type{1}},
            test_type{*(TestType{}().begin_broadcast()+0), value_type{1}},
            test_type{*(TestType{}().begin_broadcast()+1), value_type{2}},
            test_type{*(TestType{}().begin_broadcast()+2), value_type{3}},
            test_type{*(TestType{}().begin_broadcast()+3), value_type{4}},
            test_type{*(TestType{}().begin_broadcast()+4), value_type{5}},
            test_type{*(TestType{}().begin_broadcast()+5), value_type{6}},
            test_type{*(TestType{}().end_broadcast()-1), value_type{6}},
            test_type{*(TestType{}().end_broadcast()-2), value_type{5}},
            test_type{*(TestType{}().end_broadcast()-3), value_type{4}},
            test_type{*(TestType{}().end_broadcast()-4), value_type{3}},
            test_type{*(TestType{}().end_broadcast()-5), value_type{2}},
            test_type{*(TestType{}().end_broadcast()-6), value_type{1}}
        );
        auto deref = std::get<0>(test_data);
        auto expected_deref = std::get<1>(test_data);
        REQUIRE(deref == expected_deref);
    }
    SECTION("test_iter_cmp"){
        using test_type = std::tuple<bool,bool>;
        //0cmp_result,1expected_cmp_result
        auto test_data = GENERATE(
            test_type{[](){auto t = TestType{}(); return t.begin_broadcast() > t.end_broadcast();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end_broadcast() > t.begin_broadcast();}(), true},
            test_type{[](){auto t = TestType{}(); return t.begin_broadcast() < t.end_broadcast();}(), true},
            test_type{[](){auto t = TestType{}(); return t.end_broadcast() < t.begin_broadcast();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end_broadcast() > t.end_broadcast();}(), false},
            test_type{[](){auto t = TestType{}(); return t.begin_broadcast() > t.begin_broadcast();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end_broadcast() < t.end_broadcast();}(), false},
            test_type{[](){auto t = TestType{}(); return t.begin_broadcast() < t.begin_broadcast();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end_broadcast() >= t.end_broadcast();}(), true},
            test_type{[](){auto t = TestType{}(); return t.begin_broadcast() >= t.begin_broadcast();}(), true},
            test_type{[](){auto t = TestType{}(); return t.end_broadcast() <= t.end_broadcast();}(), true},
            test_type{[](){auto t = TestType{}(); return t.begin_broadcast() <= t.begin_broadcast();}(), true}
        );
        auto cmp_result = std::get<0>(test_data);
        auto expected_cmp_result = std::get<1>(test_data);
        REQUIRE(cmp_result == expected_cmp_result);
    }
}

TEMPLATE_LIST_TEST_CASE("test_flat_index_iterator","[test_expression_template_engine]",test_expression_template_helpers::makers_trivial_type_list)
{
    SECTION("test_iter_deref"){
        using value_type = typename TestType::value_type;
        using test_type = std::tuple<value_type, value_type>;
        //0deref,1expected_deref
        auto test_data = GENERATE(
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); return *it;}(), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); ++it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); ++it; ++it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); ++it; ++it; ++it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); ++it; ++it; ++it; ++it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); ++it; ++it; ++it; ++it; ++it; return *it;}(), value_type{6}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); ++it; ++it; ++it; ++it; ++it; --it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); ++it; ++it; ++it; ++it; ++it; --it; --it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; --it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.begin_trivial(); ++it; ++it; ++it; ++it; ++it; --it; --it; --it; --it; --it; return *it;}(), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; return *it;}(), value_type{6}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; --it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; --it; --it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; --it; --it; --it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; --it; --it; --it; --it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; --it; --it; --it; --it; --it; return *it;}(), value_type{1}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; --it; --it; --it; --it; --it; ++it; return *it;}(), value_type{2}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; --it; --it; --it; --it; --it; ++it; ++it; return *it;}(), value_type{3}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; return *it;}(), value_type{4}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; ++it; return *it;}(), value_type{5}},
            test_type{[](){auto t = TestType{}(); auto it = t.end_trivial(); --it; --it; --it; --it; --it; --it; ++it; ++it; ++it; ++it; ++it; return *it;}(), value_type{6}},
            test_type{TestType{}().begin_trivial()[0], value_type{1}},
            test_type{TestType{}().begin_trivial()[1], value_type{2}},
            test_type{TestType{}().begin_trivial()[2], value_type{3}},
            test_type{TestType{}().begin_trivial()[3], value_type{4}},
            test_type{TestType{}().begin_trivial()[4], value_type{5}},
            test_type{TestType{}().begin_trivial()[5], value_type{6}},
            test_type{TestType{}().end_trivial()[-1], value_type{6}},
            test_type{TestType{}().end_trivial()[-2], value_type{5}},
            test_type{TestType{}().end_trivial()[-3], value_type{4}},
            test_type{TestType{}().end_trivial()[-4], value_type{3}},
            test_type{TestType{}().end_trivial()[-5], value_type{2}},
            test_type{TestType{}().end_trivial()[-6], value_type{1}},
            test_type{*(TestType{}().begin_trivial()+0), value_type{1}},
            test_type{*(TestType{}().begin_trivial()+1), value_type{2}},
            test_type{*(TestType{}().begin_trivial()+2), value_type{3}},
            test_type{*(TestType{}().begin_trivial()+3), value_type{4}},
            test_type{*(TestType{}().begin_trivial()+4), value_type{5}},
            test_type{*(TestType{}().begin_trivial()+5), value_type{6}},
            test_type{*(TestType{}().end_trivial()-1), value_type{6}},
            test_type{*(TestType{}().end_trivial()-2), value_type{5}},
            test_type{*(TestType{}().end_trivial()-3), value_type{4}},
            test_type{*(TestType{}().end_trivial()-4), value_type{3}},
            test_type{*(TestType{}().end_trivial()-5), value_type{2}},
            test_type{*(TestType{}().end_trivial()-6), value_type{1}}
        );
        auto deref = std::get<0>(test_data);
        auto expected_deref = std::get<1>(test_data);
        REQUIRE(deref == expected_deref);
    }
    SECTION("test_iter_cmp"){
        using test_type = std::tuple<bool,bool>;
        //0cmp_result,1expected_cmp_result
        auto test_data = GENERATE(
            test_type{[](){auto t = TestType{}(); return t.begin_trivial() > t.end_trivial();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end_trivial() > t.begin_trivial();}(), true},
            test_type{[](){auto t = TestType{}(); return t.begin_trivial() < t.end_trivial();}(), true},
            test_type{[](){auto t = TestType{}(); return t.end_trivial() < t.begin_trivial();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end_trivial() > t.end_trivial();}(), false},
            test_type{[](){auto t = TestType{}(); return t.begin_trivial() > t.begin_trivial();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end_trivial() < t.end_trivial();}(), false},
            test_type{[](){auto t = TestType{}(); return t.begin_trivial() < t.begin_trivial();}(), false},
            test_type{[](){auto t = TestType{}(); return t.end_trivial() >= t.end_trivial();}(), true},
            test_type{[](){auto t = TestType{}(); return t.begin_trivial() >= t.begin_trivial();}(), true},
            test_type{[](){auto t = TestType{}(); return t.end_trivial() <= t.end_trivial();}(), true},
            test_type{[](){auto t = TestType{}(); return t.begin_trivial() <= t.begin_trivial();}(), true}
        );
        auto cmp_result = std::get<0>(test_data);
        auto expected_cmp_result = std::get<1>(test_data);
        REQUIRE(cmp_result == expected_cmp_result);
    }
}
