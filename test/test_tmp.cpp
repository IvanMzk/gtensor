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
    using tensor_type = tensor<int,c_order>;
    using slice_type = tensor_type::slice_type;
    using config_type = tensor_type::config_type;
    using shape_type = tensor_type::shape_type;
    using helpers_for_testing::range_to_str;
    using gtensor::detail::shape_to_str;
    using gtensor::detail::next_13::next_c;
    using gtensor::detail::next_13::prev_c;

    auto t = tensor<int,c_order>{{{1,2,3},{4,5,6}},{{7,8,9},{10,11,12}}};
    //auto t = tensor<int,c_order>({5,5,5},0);
    //auto t = tensor_type{{1,2,3},{4,5,6}};
    //std::cout<<std::endl<<t;
    //using walker_type = decltype(t.create_walker());
    //using travrser_type = gtensor::walker_forward_traverser<config_type,walker_type>;
    //travrser_type tr{t.shape(),t.create_walker()};

    auto w = t.create_walker();
    shape_type index(t.dim(),0);
    auto n = t.size();
    while(n!=0){
        --n;
        prev_c(w,index.begin(),t.shape().begin(),t.dim());
        std::cout<<std::endl<<shape_to_str(index)<<" "<<*w;
    }



}