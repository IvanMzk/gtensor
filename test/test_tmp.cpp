#include <iostream>
#include <vector>
#include "tensor_math.hpp"
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

struct cumsum{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        auto cumsum_ = *first;
        *dfirst = cumsum_;
        for(++dfirst,++first; dfirst!=dlast; ++dfirst,++first){
            cumsum_+=*first;
            *dfirst = cumsum_;
        }
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

    using value_type = double;
    static constexpr value_type nan = std::numeric_limits<value_type>::quiet_NaN();
    static constexpr value_type pos_inf = std::numeric_limits<value_type>::infinity();
    static constexpr value_type neg_inf = -std::numeric_limits<value_type>::infinity();

    tensor<value_type,c_order> t{1.0,0.0,2.0,neg_inf,3.0,pos_inf};

    auto r = gtensor::nanprod(t);



    std::cout<<std::endl<<r;

    std::cout<<std::endl<<gtensor::math::isnan(pos_inf);
    std::cout<<std::endl<<gtensor::math::isnan(pos_inf*0.0);
    std::cout<<std::endl<<gtensor::math::isnan(pos_inf*pos_inf);
    std::cout<<std::endl<<gtensor::math::isnan(pos_inf*neg_inf);

}