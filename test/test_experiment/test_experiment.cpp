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

    //index_type n{1};
    //index_tensor_type tt(n,0);

    value_type v{10};
    index_type n{10};
    //auto tt = tensor_type(10,0);
    //index_tensor_type tt(n,0);
    //index_tensor_type tt(10.0);
    std::cout<<std::endl<<tt;

}