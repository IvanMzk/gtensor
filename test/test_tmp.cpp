#include <iostream>
#include <vector>
#include "statistic.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "benchmark_helpers.hpp"

namespace test_tmp{


struct sum
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{0};}
        const auto& init = *first;
        return std::accumulate(++first,last,init,std::plus{});
    }
};


}

TEST_CASE("test_tmp","[test_tmp]")
{

    using benchmark_helpers::timing;
    using benchmark_helpers::shapes;

    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;
    using tensor_type = tensor<int,c_order>;
    using slice_type = tensor_type::slice_type;
    using config_type = tensor_type::config_type;
    using shape_type = tensor_type::shape_type;
    using helpers_for_testing::range_to_str;
    using gtensor::detail::shape_to_str;



    //auto t = tensor<double,c_order>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}};
    auto t = tensor<double,c_order>{{1},{2},{3},{4},{5},{6}};

    auto r = gtensor::reduce(t,std::vector<int>{1},test_tmp::sum{},false,true);

    std::cout<<std::endl<<r;

}