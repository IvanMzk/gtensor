#include <tuple>
#include <vector>
#include "catch.hpp"
#include "random.hpp"
#include "helpers_for_testing.hpp"

TEMPLATE_TEST_CASE("test_random_integers","[test_random]",
    int,
    std::int64_t
)
{
    using value_type = TestType;
    using tensor_type = gtensor::tensor<value_type>;
    using config_type = typename tensor_type::config_type;
    using bit_generator_type = std::mt19937_64;
    using gtensor::default_rng;
    using gtensor::make_rng;
    using helpers_for_testing::apply_by_element;

    //0seeds,1low,2high,3size,4end_point,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::make_tuple(1,2,3),0,5,10,false,tensor_type{}),
        std::make_tuple(std::make_tuple(3,2,1),0,5,10,false,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),-3,3,10,false,tensor_type{}),
        std::make_tuple(std::make_tuple(1,2,3),-3,3,std::vector<int>{3,4},true,tensor_type{})
    );
    auto test = [](const auto& t){
        auto seeds = std::get<0>(t);
        auto low = std::get<1>(t);
        auto high = std::get<2>(t);
        auto size = std::get<3>(t);
        auto end_point = std::get<4>(t);
        auto expected = std::get<5>(t);

        auto rng_maker = [](const auto&...seeds_){
            return make_rng<bit_generator_type,config_type>(seeds_...);
        };
        auto rng = std::apply(rng_maker,seeds);
        auto result = rng.template integers<value_type>(low,high,size,end_point);
        std::cout<<std::endl<<result;
        //REQUIRE()
    };
    apply_by_element(test,test_data);
}