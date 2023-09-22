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

template<typename First, typename Second>
struct pair{
    First first;
    Second second;
    pair() = default;
    template<typename First_, typename Second_>
    pair(First_&& first__, Second_&& second__):
        first{std::forward<First_>(first__)},
        second{std::forward<Second_>(second__)}
    {}
    friend std::ostream& operator<<(std::ostream& os, const pair& inst){
        return os<<inst.first<<" "<<inst.second;
    }
};

template<typename First, typename Second>
auto make_pair(First&& first, Second&& second){
    return pair<std::remove_cv_t<std::remove_reference_t<First>>,std::remove_cv_t<std::remove_reference_t<Second>>>{
        std::forward<First>(first),
        std::forward<Second>(second)
    };
}


}

TEST_CASE("test_tmp","[test_tmp]")
{

    using benchmark_helpers::timing;
    using benchmark_helpers::shapes;

    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::tensor;

    using test_tmp::pair;
    using test_tmp::make_pair;

    using value_type = double;

    auto axes = 0;
    tensor<value_type,c_order> t{{1,2,3},{4,5,6},{7,8,9}};
    auto init = *t.begin();
    auto tmp = reduce_binary(t,axes,
        [](const auto& r, const auto& e){
            return test_tmp::make_pair(std::min(r.first,e),std::max(r.second,e));
        },
        false,test_tmp::make_pair(init,init)
    );

    std::cout<<std::endl<<tmp;
    std::cout<<std::endl<<t.max(axes)-t.min(axes);
    //std::cout<<std::endl<<(*tmp.begin()).first<<" "<<(*tmp.begin()).second;

}