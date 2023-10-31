#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"

namespace test_tmp{
}   //end of namespace test_tmp

TEST_CASE("test_tmp_copy","[test_tmp]")
{

    using gtensor::tensor;

    tensor<double> t{{1,0,3},{4,5,6}};
    std::cout<<std::endl<<t;

    tensor<bool> b{true,false,true};
    std::cout<<std::endl<<b;
    typename std::iterator_traits<decltype(b.begin())>::reference e = *b.begin();
    e = false;
    std::cout<<std::endl<<b;

    auto t1 = any(t,1);
    std::cout<<std::endl<<t1;

    auto t2 = all(t,1);
    std::cout<<std::endl<<t2;

    //REQUIRE(tensor_type{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(policy{},std::vector<int>{0,1}) == tensor<bool>{false,true,true,false});
    //std::make_tuple(tensor_type{nan,pos_inf,0.0,nan,neg_inf},std::vector<int>{},false,bool_tensor_type{true,true,false,true,true})

    //tensor<double>{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(policy{},std::vector<int>{0,1})
    std::cout<<std::endl<<tensor<double>{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(multithreading::exec_pol<1>{},std::vector<int>{0,1});
    std::cout<<std::endl<<tensor<double>{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(std::vector<int>{0,1});
    std::cout<<std::endl<<tensor<int>{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(std::vector<int>{0,1});
    std::cout<<std::endl<<tensor<int>{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(multithreading::exec_pol<1>{},std::vector<int>{0,1});
    REQUIRE(tensor<int>{{{0,1,2,3},{1,2,3,0}},{{1,2,3,4},{1,2,3,4}}}.all(multithreading::exec_pol<1>{},std::vector<int>{0,1}) == tensor<bool>{false,true,true,false});

}
