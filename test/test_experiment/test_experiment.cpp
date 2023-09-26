#include <iostream>
#include <vector>
#include "sort_search.hpp"
#include "tensor.hpp"
#include "../helpers_for_testing.hpp"
#include "../benchmark_helpers.hpp"
#include "../test_config.hpp"

TEST_CASE("test_tmp","[test_tmp]")
{


    using gtensor::tensor;
    using value_type = double;
    using tensor_type = tensor<value_type>;
    using index_type = typename tensor_type::index_type;

    using index_tensor_type = tensor<index_type>;

    // index_type n{1};

    // //index_tensor_type tt(n,0);

    // value_type v{1};
    // std::vector<int> ss{1,2,3};
    // auto tt = tensor_type(10,0);
    // std::cout<<std::endl<<tt;

    tensor_type t{{1,2},{3,4},{2,1},{3,5},{7,4},{2,3},{1,1},{7,2}};

    // auto uniq = unique(t,);
    // std::cout<<std::endl<<uniq;

    // auto uniq = unique(t,std::true_type{});
    // std::cout<<std::endl<<std::get<0>(uniq);
    // std::cout<<std::endl<<std::get<1>(uniq);

    // auto uniq = unique(t,std::true_type{},std::true_type{});
    // std::cout<<std::endl<<std::get<0>(uniq);
    // std::cout<<std::endl<<std::get<1>(uniq);
    // std::cout<<std::endl<<std::get<2>(uniq);

    auto uniq = unique(t,std::true_type{},std::true_type{},std::true_type{},0);
    std::cout<<std::endl<<std::get<0>(uniq);
    std::cout<<std::endl<<std::get<1>(uniq);
    std::cout<<std::endl<<std::get<2>(uniq);
    std::cout<<std::endl<<std::get<3>(uniq);

}