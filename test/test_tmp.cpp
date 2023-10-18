#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"
#include "tensor_math.hpp"

namespace test_tmp{

using gtensor::basic_tensor;
using gtensor::tensor;

template<typename T> struct copy_result{

    template<typename U, typename> struct selector_{using type = U;};
    template<typename U> struct selector_<U,std::true_type>{
        using type = tensor<typename selector_<typename U::value_type,std::bool_constant<gtensor::detail::is_tensor_v<typename U::value_type>>>::type>;
    };
    using type = typename selector_<T,std::bool_constant<gtensor::detail::is_tensor_v<T>>>::type;
};
template<typename T> using copy_result_t = typename copy_result<T>::type;

template<typename It, typename DstIt>
auto copy_helper(It first, It last, DstIt dfirst);

template<typename...Ts>
auto copy(const basic_tensor<Ts...>& t){
    //using value_type = typename basic_tensor<Ts...>::value_type;
    using res_type = copy_result_t<basic_tensor<Ts...>>;
    res_type res(t.shape());
    copy_helper(t.begin(),t.end(),res.begin());
    return res;
}

template<typename It, typename DstIt>
auto copy_helper(It first, It last, DstIt dfirst){
    using value_type = typename std::iterator_traits<It>::value_type;
    if constexpr (gtensor::detail::is_tensor_v<value_type>){
        for (;first!=last; ++first,++dfirst){
            //*dfirst = copy(*first);
            *dfirst = *first;
        }
    }else{
        std::copy(first,last,dfirst);
    }
}



}   //end of namespace test_tmp

TEST_CASE("test_tmp_copy","[test_tmp]")
{

    using gtensor::tensor;
    using test_tmp::copy_result_t;
    using test_tmp::copy;

    tensor<tensor<tensor<double>>> a{
        {tensor<tensor<double>>{tensor<double>{1,2,1},tensor<double>{4,0,3}},tensor<tensor<double>>{tensor<double>{3,0,2},tensor<double>{1,1,1}}},
        {tensor<tensor<double>>{tensor<double>{1,4,1},tensor<double>{1,1,0}},tensor<tensor<double>>{tensor<double>{1,0,1},tensor<double>{1,2,0}}}
    };
    tensor<tensor<tensor<double>>> b{
        tensor<tensor<double>>{tensor<double>{1,4,1},tensor<double>{2,1,3}},
        tensor<tensor<double>>{tensor<double>{1,0,2},tensor<double>{2,3,4}}
    };

    tensor<tensor<double>> c{tensor<double>{1,2,3},tensor<double>{4,5,6}};
    tensor<tensor<double>> d{tensor<double>{7,8,9},tensor<double>{10,11,12}};
    tensor<tensor<double>> e{{tensor<double>{2,1,0},tensor<double>{1,1,2}},{tensor<double>{1,0,1},tensor<double>{1,3,2}}};
    tensor<tensor<double>> f{{tensor<double>{2,1,0},tensor<double>{1,1,2}},{tensor<double>{1,0,1},tensor<double>{1,3,2}},{tensor<double>{0,0,1},tensor<double>{1,0,3}}};

    // std::cout<<std::endl<<tensor<double>{};
    // std::cout<<std::endl<<tensor<double>(1);
    // std::cout<<std::endl<<tensor<double>(3,2.2);
    // std::cout<<std::endl<<tensor<double>(std::vector<int>{2,2},3.3);
    // std::cout<<std::endl<<tensor<double>({2,2},1.1);
    // std::cout<<std::endl<<tensor<double>{{1,2,3},{4,5,6}};

    // std::cout<<std::endl<<tensor<tensor<double>>{};
    // std::cout<<std::endl<<tensor<tensor<double>>(1);
    // std::cout<<std::endl<<tensor<tensor<double>>(3,2.2);
    // std::cout<<std::endl<<tensor<tensor<double>>(std::vector<int>{2,2},3.3);
    // std::cout<<std::endl<<tensor<tensor<double>>({2,2},1.1);
    // std::cout<<std::endl<<tensor<tensor<double>>{{1,2,3},{4,5,6}};

    // std::cout<<std::endl<<tensor<tensor<tensor<double>>>{};
    // std::cout<<std::endl<<tensor<tensor<tensor<double>>>(1);
    // std::cout<<std::endl<<tensor<tensor<tensor<double>>>(3,2.2);
    // std::cout<<std::endl<<tensor<tensor<tensor<double>>>(std::vector<int>{2,2},0);
    // std::cout<<std::endl<<tensor<tensor<tensor<double>>>({2,2},0);

    std::cout<<std::endl<<(a.reshape(-1,1)+b.reshape(1,-1));
    std::cout<<std::endl<<c*d;
    std::cout<<std::endl<<matmul(c,d);
    std::cout<<std::endl<<matmul(c,e);
    std::cout<<std::endl<<matmul(e,c);
    std::cout<<std::endl<<matmul(f,e);
    std::cout<<std::endl<<matmul(a,b);
    std::cout<<std::endl<<matmul(b,a);
    std::cout<<std::endl<<matmul(a,a);
    std::cout<<std::endl<<matmul(tensor<double>{1,2,3},tensor<double>{4,5,6});
}

TEST_CASE("test_element_type","[test_tmp]")
{

    using gtensor::tensor;
    using tensor_type = tensor<double>;
    using tensor_type_1 = tensor<tensor<double>>;
    using tensor_type_2 = tensor<tensor<tensor<int>>>;

    REQUIRE(std::is_same_v<typename tensor_type::element_type,double>);
    REQUIRE(std::is_same_v<typename tensor_type::value_type,double>);

    REQUIRE(std::is_same_v<typename tensor_type_1::element_type,double>);
    REQUIRE(std::is_same_v<typename tensor_type_1::value_type,tensor<double>>);

    REQUIRE(std::is_same_v<typename tensor_type_2::element_type,int>);
    REQUIRE(std::is_same_v<typename tensor_type_2::value_type,tensor<tensor<int>>>);

    REQUIRE(std::is_same_v<typename decltype(std::declval<tensor_type_2>()+std::declval<tensor_type_2>()+std::declval<tensor_type_2>())::element_type,int>);
}

// TEST_CASE("test_copy_result_2","[test_tmp]")
// {

//     using gtensor::tensor;
//     using test_tmp::copy_result_t;

//     tensor<tensor<double>> a{tensor<double>{1,2,3},tensor<double>{4,5,6}};
//     tensor<tensor<double>> b{tensor<double>{7,8,9},tensor<double>{10,11,12}};

//     REQUIRE(std::is_same_v<copy_result_t<decltype(a)>,tensor<tensor<double>>>);
//     REQUIRE(std::is_same_v<copy_result_t<decltype(a+b)>,tensor<tensor<double>>>);
//     REQUIRE(std::is_same_v<copy_result_t<decltype(a+b+a+b)>,tensor<tensor<double>>>);
//     REQUIRE(std::is_same_v<copy_result_t<decltype((a+b)+(a+b)+a+b)>,tensor<tensor<double>>>);
// }

// TEST_CASE("test_copy_result_3","[test_tmp]")
// {

//     using gtensor::tensor;
//     using test_tmp::copy_result_t;

//     tensor<tensor<tensor<double>>> a{tensor<tensor<double>>{tensor<double>{1,2,3},tensor<double>{4,5,6}},tensor<tensor<double>>{tensor<double>{7,8,9},tensor<double>{10,11,12}}};
//     tensor<tensor<tensor<double>>> b{tensor<tensor<double>>{tensor<double>{13,14,15},tensor<double>{16,17,18}},tensor<tensor<double>>{tensor<double>{19,20,21},tensor<double>{22,23,24}}};

//     REQUIRE(std::is_same_v<copy_result_t<decltype(a)>,tensor<tensor<tensor<double>>>>);
//     REQUIRE(std::is_same_v<copy_result_t<decltype(a+b)>,tensor<tensor<tensor<double>>>>);
//     REQUIRE(std::is_same_v<copy_result_t<decltype(a+b+a+b)>,tensor<tensor<tensor<double>>>>);
//     REQUIRE(std::is_same_v<copy_result_t<decltype((a+b)+(a+b)+a+b)>,tensor<tensor<tensor<double>>>>);
// }