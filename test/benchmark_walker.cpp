#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "tensor.hpp"
#include "impl_walker_base.hpp"
#include "impl_multiindex_iterator.hpp"
#include "test_config.hpp"
#include <iterator>

namespace benchmark_walker{
using gtensor::multiindex_iterator_impl;
using gtensor::walker;
using gtensor::tensor;

template<typename ValT, template<typename> typename Cfg>
struct inline_walker_test_tensor : public tensor<ValT,Cfg>{
    using base_type = tensor<ValT,Cfg>;
    using tensor::tensor;
    inline_walker_test_tensor(const base_type& base):
        base_type{base}
    {}
    walker<ValT,Cfg> create_walker()const{
        return get_impl()->as_expression()->create_walker();
    }    
    auto begin()const{return multiindex_iterator_impl<ValT,Cfg,walker<ValT,Cfg>>{create_walker(), get_impl()->shape(), get_impl()->strides()};}
    auto end()const{return multiindex_iterator_impl<ValT,Cfg,walker<ValT,Cfg>>{create_walker(), get_impl()->shape(), get_impl()->strides(), get_impl()->size()};}
};

template<typename ValT, template<typename> typename Cfg>
struct noinline_walker_test_tensor : public tensor<ValT,Cfg>{
    using base_type = tensor<ValT,Cfg>;
    using tensor::tensor;
    noinline_walker_test_tensor(const base_type& base):
        base_type{base}
    {}
    walker<ValT,Cfg> create_walker()const{
        return get_impl()->as_walker_maker()->create_walker();
    }
};

}   //end of namespace benchmark_walker


TEMPLATE_TEST_CASE("benchmark_walker","[benchmark_walker]", gtensor::config::mode_div_native){
    using value_type = float;
    using inline_walker_tensor_type = benchmark_walker::inline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using noinline_walker_tensor_type = benchmark_walker::noinline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using walker_type = gtensor::walker<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using iterator_type = gtensor::multiindex_iterator_impl<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl, walker_type>;

    inline_walker_tensor_type t1{{1,2,3}};
    inline_walker_tensor_type t2{{1},{2},{3}};
    inline_walker_tensor_type e = t1+t2;
    auto e_begin = e.begin();
    auto e_end = e.end();
    
    // for (auto i : e){
    //     std::cout<<i;
    // }
    //REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{2,3,4,3,4,5,4,5,6}.begin()));

}