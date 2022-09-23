#include <tuple>
#include "catch.hpp"
#include "gtensor.hpp"
#include "test_config.hpp"

namespace test_expression_template_helpers{

// using config_type = gtensor::config::default_config;
// using tensor_type = gtensor::tensor<value_type,config_type>;

using value_type = float;
using gtensor::tensor;
using test_config_type = typename test_config::config_engine_selector<gtensor::config::engine_expression_template>::config_type;


template<typename T>
struct test_tensor : public T{
    test_tensor(const T& base):
        tensor{base}
    {}
    auto& engine()const{return impl()->engine();}
    bool is_trivial()const{return engine().is_trivial();}
    auto create_walker()const{return engine().create_walker();}
};

template<typename T>
auto make_test_tensor(T&& t){return test_tensor<std::decay_t<T>>{t};}

template<typename ValT = value_type, typename CfgT = test_config_type>
struct storage_tensor_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{{{1,2,3},{4,5,6}}}
        );
    }
};

template<typename ValT = value_type, typename CfgT = test_config_type>
struct notrivial_tensor_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{{{0},{3}}} + tensor_type{1,2,3}
        );
    }
};

template<typename ValT = value_type, typename CfgT = test_config_type>
struct trivial_subtree_tensor_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{2} * tensor_type{-1,-1,-1} + (tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{0,0,0},{3,3,3}}}) + tensor_type{5,5,5} - tensor_type{3}
        );
    }
};

template<typename ValT = value_type, typename CfgT = test_config_type>
struct trivial_tensor_maker{
    using value_type = ValT;
    using tensor_type = tensor<ValT,CfgT>;
    auto operator()(){
        return make_test_tensor(
            tensor_type{{{-1,-1,-1},{-1,-1,-1}}} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{1,1,1},{4,4,4}}}
        );
    }
};

}   //end of namespace test_tensor

TEST_CASE("test_is_trivial","[test_expression_template_engine]"){
    using value_type = float;
    using tensor_type = gtensor::tensor<value_type, test_expression_template_helpers::test_config_type>;
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

TEMPLATE_TEST_CASE("test_walker","[test_expression_template_engine]",
    test_expression_template_helpers::storage_tensor_maker<>,
    test_expression_template_helpers::notrivial_tensor_maker<>,
    test_expression_template_helpers::trivial_subtree_tensor_maker<>,
    test_expression_template_helpers::trivial_tensor_maker<>
)
{
    using value_type = typename TestType::value_type;
    using test_type = std::tuple<value_type,value_type>;

    auto test_data = GENERATE(
        test_type{[](){auto w = TestType{}().create_walker(); return *w; }(), value_type{1}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(2,1); return *w;}(), value_type{1}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(1,1); return *w;}(), value_type{4}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(0,1); return *w;}(), value_type{2}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(1,1); w.walk(0,2); return *w;}(), value_type{6}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(3,1); w.walk(0,2); return *w;}(), value_type{3}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(1,1); w.reset(1); return *w;}(), value_type{1}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(1,1); w.walk(0,2); w.reset(2); return *w;}(), value_type{6}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(1,1); w.walk(0,2); w.reset(3); return *w;}(), value_type{6}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(1,1); w.walk(0,2); w.reset(1); return *w;}(), value_type{3}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(1,1); w.walk(0,2); w.reset(0); return *w;}(), value_type{4}},
        test_type{[](){auto w = TestType{}().create_walker(); w.walk(1,1); w.walk(0,2); w.reset(); return *w;}(), value_type{1}}
    );
    auto deref = std::get<0>(test_data);
    auto expected_deref = std::get<1>(test_data);
    REQUIRE(deref == expected_deref);
}