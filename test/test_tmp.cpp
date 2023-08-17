#include <iostream>
#include <vector>
#include "statistic.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "benchmark_helpers.hpp"

namespace test_tmp{

using gtensor::basic_tensor;

}

TEST_CASE("test_tmp","[test_tmp]")
{

    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;

    using layout = f_order;

    auto e = tensor<double,layout>{{1,2,3},{4,5,6}} - tensor<double,layout>{{1,1,1},{2,2,2}};

    std::cout<<std::endl<<e;
    std::cout<<std::endl<<benchmark_helpers::order_to_str(typename decltype(e)::order{});

    auto a = e.traverse_order_adapter(f_order{});

    auto ind = a.create_trivial_indexer();

    // for (auto i=0; i!=e.size(); ++i){
    //     std::cout<<std::endl<<ind[i];
    // }

    // std::cout<<std::endl<<helpers_for_testing::range_to_str(a.begin_trivial(),a.end_trivial());
    // std::cout<<std::endl<<helpers_for_testing::range_to_str(e.begin_trivial(),e.end_trivial());


}