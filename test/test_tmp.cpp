#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "statistic.hpp"
#include "builder.hpp"

namespace test_tmp{
}   //end of namespace test_tmp

TEST_CASE("test_tmp_copy","[test_tmp]")
{

    using gtensor::tensor;
    using value_type = double;
    tensor<value_type> a{{1,1,1,3,3,2,3,2,2,2,0,2,4,0,4,3,2,3,2,3,2,4,3,4,0,2,3,4,1,0},{0,1,2,0,4,0,2,3,2,4,4,4,2,2,4,1,1,1,4,4,1,2,3,4,0,4,3,4,3,2},{2,1,3,2,1,4,1,0,1,0,1,2,4,1,4,3,1,4,4,0,1,1,1,0,2,0,3,3,1,4},{3,1,1,3,1,1,4,2,1,2,3,1,4,1,0,3,2,1,2,0,2,2,1,3,4,4,2,1,3,4},{0,3,2,3,1,2,0,4,3,3,4,3,0,1,0,0,3,0,0,0,0,0,3,1,1,2,2,4,1,3},{0,1,2,0,3,1,3,1,3,2,4,0,4,0,2,3,2,0,4,3,0,1,4,2,0,1,1,1,2,4},{1,3,2,4,0,3,4,3,3,0,4,4,0,1,2,0,4,1,1,2,3,3,1,2,2,1,4,0,1,4},{0,2,3,4,1,3,0,0,1,3,4,4,1,4,3,1,1,1,1,2,0,2,2,1,3,4,4,0,0,3},{4,0,2,0,2,2,4,1,4,0,0,0,4,4,3,1,4,4,1,4,4,0,0,4,0,4,2,4,4,0},{4,1,1,4,3,1,0,3,2,0,0,0,3,0,0,1,1,3,2,1,0,4,3,2,3,1,2,1,4,4}};
    std::cout<<std::endl<<a;
    const auto var_axis = 1;
    const auto n = a.shape()[var_axis];
    std::cout<<std::endl<<a.mean(var_axis,true);

    using fp_type = gtensor::math::make_floating_point_t<value_type>;
    auto a_cnt = a - a.mean(var_axis,true);
    std::cout<<std::endl<<a_cnt;

    auto a_cov = matmul(a_cnt,a_cnt.transpose())/static_cast<fp_type>(n-1);
    std::cout<<std::endl<<a_cov;
    std::cout<<std::endl<<diag(a_cov);
    auto a_vars = diag(a_cov);
    auto normalizer = sqrt(a_vars.reshape(-1,1)*a_vars);
    std::cout<<std::endl<<normalizer;
    auto a_corr = a_cov/normalizer;
    std::cout<<std::endl<<a_corr;





}
