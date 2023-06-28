#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"

namespace test_tmp{

using gtensor::basic_tensor;

template<typename ResultT, typename F, typename...Ts, typename IdxT, typename...Args>
auto slide(const basic_tensor<Ts...>& t, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
    std::cout<<std::endl<<"template<typename ResultT, typename F, typename...Ts, typename IdxT, typename...Args>";
    return t+1;
}

template<typename F, typename...Ts, typename IdxT, typename...Args>
auto slide(const basic_tensor<Ts...>& t, F f, const IdxT& window_size, const IdxT& window_step, Args&&...args){
    std::cout<<std::endl<<"template<typename F, typename...Ts, typename IdxT, typename...Args>";
    return t+1;
}




}

TEST_CASE("test_tmp","[test_tmp]")
{
    using value_type = double;
    using gtensor::tensor;

    tensor<value_type> t{{1,2,3},{4,5,6}};

    auto slide_f = [](auto first, auto last, auto dfirst, auto dlast){
        for (;first!=last;++first){
            std::cout<<std::endl<<*first;
        }
    };

    // auto r = test_tmp::slide(t,slide_f,1,1);
    // auto rr = test_tmp::slide<value_type>(t,slide_f,1,1);


    //auto r = gtensor::slide(t,slide_f,1,1);


    //auto r = gtensor::slide<value_type>(t,1,slide_f,1,1);
    //auto r = gtensor::slide<value_type>(t,slide_f,1,1);

    //auto d = gtensor::diff2(t,0);
    auto g = gtensor::gradient(t,1);
    std::cout<<std::endl<<g;

}