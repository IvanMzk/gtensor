#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"

namespace test_tmp{

using gtensor::basic_tensor;

// template<typename Stream, typename T>
// void str_helper(const std::vector<T>& v, Stream& stream){
//     if (v.size()<=1){
//         //stream<<"{"<<[&]{for(const auto& e:t){stream<<e<<" ";}; return "}";}();
//         std::cout<<"q";
//     }else{
//         //str_helper(tensor<value_type,order,config_type>{},stream);
//         std::cout<<"t";
//         auto tmp = v;
//         tmp.pop_back();
//         str_helper(tmp,stream);
//         // const auto& shape = t.shape();
//         // for (auto it=shape.begin(),last=shape.end(); it!=last; ++it){
//         //     const auto axis_size = *it;
//         //     for (index_type i=0; i!=axis_size; ++i){
//         //         //str_helper(t(i),stream);
//         //     }
//         // }
//     }
// }

// template<typename P, typename T>
// auto str(const std::vector<T>& v, const P& precision){
//     std::stringstream ss{};
//     ss.precision(precision);
//     str_helper(v,ss);
//     return ss.str();
// }


template<typename Stream, typename...Ts>
void str_helper(const basic_tensor<Ts...>& t, Stream& stream){
    using index_type = typename basic_tensor<Ts...>::index_type;
    if (t.dim()>1){
        auto axis_size = t.shape()[0];
        if (axis_size > 0){
            index_type i=0;
            for (--axis_size; i!=axis_size; ++i){
                stream<<"{";
                str_helper(t(i).copy(),stream);
                stream<<"}"<<",";
            }
            stream<<"{";
            str_helper(t(i).copy(),stream);
            stream<<"}";
        }
    }else{
        if (t.size()>0){
            auto it = t.begin();
            auto last = t.end();
            for (--last; it!=last; ++it){
                stream<<*it<<",";
            }
            stream<<*it;
        }
    }
}

template<typename P, typename...Ts>
auto str(const basic_tensor<Ts...>& t, const P& precision){
    std::stringstream ss{};
    ss.precision(precision);
    ss<<"{";
    str_helper(t,ss);
    ss<<"}";
    return ss.str();
}

}

TEST_CASE("test_tmp","[test_tmp]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;

    // std::vector<int> v{1,2,3};
    // auto s = test_tmp::str(v,3);

    auto t = tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}};
    std::cout<<std::endl<<test_tmp::str(t,3);
    std::cout<<std::endl<<test_tmp::str(tensor_type{1,2,3,4,5},3);
    std::cout<<std::endl<<test_tmp::str(tensor_type{{1},{2},{3},{4},{5}},3);
    std::cout<<std::endl<<test_tmp::str(tensor_type{},3);
    std::cout<<std::endl<<test_tmp::str(tensor_type{}.reshape(2,3,0),3);
    std::cout<<std::endl<<test_tmp::str(tensor_type{}.reshape(2,0,3),3);
    std::cout<<std::endl<<test_tmp::str(tensor_type{}.reshape(0,2,3),3);


}