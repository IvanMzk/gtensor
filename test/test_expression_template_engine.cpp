#include "catch.hpp"
#include "gtensor.hpp"
#include <tuple>

namespace test_expression_template_helpers{

template<typename T>
struct test_tensor : public T{
    test_tensor(const T& base):
        tensor{base}
    {}
    auto& engine()const{return impl()->engine();}
    bool is_trivial()const{return engine().is_trivial();}
};

}   //end of namespace test_tensor

TEST_CASE("test_is_trivial","[test_expression_template_engine]"){
    using value_type = float;
    using gtensor::tensor_base;
    using config_type = gtensor::config::default_config;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using htensor_type = typename tensor_type::htensor_type;
    using test_tensor_type = test_expression_template_helpers::test_tensor<htensor_type>;
    using test_type = std::tuple<htensor_type, bool>;
    //tensor,expected_is_trivial
    auto test_data = GENERATE(
                                test_type(static_cast<htensor_type>(tensor_type{1}+tensor_type{1}), true),
                                test_type(static_cast<htensor_type>(tensor_type{1,2,3,4,5}+tensor_type{1}), false),
                                test_type(static_cast<htensor_type>(tensor_type{1,2,3,4,5}+tensor_type{1,2,3,4,5}), true),
                                test_type(static_cast<htensor_type>(tensor_type{{1},{2},{3},{4},{5}}+tensor_type{{1,2,3,4,5}}), false),
                                test_type(static_cast<htensor_type>(tensor_type{1,2,3}+tensor_type{1,2,3}+tensor_type{3,4,5}+tensor_type{3,4,5}), true),
                                test_type(static_cast<htensor_type>((tensor_type{1,2,3}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5})), true),
                                test_type(static_cast<htensor_type>((tensor_type{1}+tensor_type{1,2,3})+(tensor_type{3,4,5}+tensor_type{3,4,5})), false)
                            );

    auto t = test_tensor_type{std::get<0>(test_data)};
    auto expected_is_trivial = std::get<1>(test_data);
    REQUIRE(t.is_trivial() == expected_is_trivial);
}