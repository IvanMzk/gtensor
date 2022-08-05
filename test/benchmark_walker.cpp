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
    using config_type = Cfg<ValT>;
    using iterator_type = multiindex_iterator_impl<ValT,Cfg,walker<ValT,Cfg>>;
    using strides_type = typename gtensor::detail::libdiv_strides_traits<inline_walker_test_tensor::config_type>::type;
    
    strides_type strides{gtensor::detail::make_dividers<inline_walker_test_tensor::config_type>(get_impl()->strides())};

    using tensor::tensor;
    inline_walker_test_tensor(const base_type& base):
        base_type{base}
    {}
    
    auto begin()const{return iterator_type{get_impl()->as_expression()->create_walker(), get_impl()->shape(), strides};}
    auto end()const{return iterator_type{get_impl()->as_expression()->create_walker(), get_impl()->shape(), strides, get_impl()->size()};}
};

template<typename ValT, template<typename> typename Cfg>
struct noinline_walker_test_tensor : public tensor<ValT,Cfg>{
    using base_type = tensor<ValT,Cfg>;
    using config_type = Cfg<ValT>;
    using iterator_type = multiindex_iterator_impl<ValT,Cfg,walker<ValT,Cfg>>;
    using strides_type = typename gtensor::detail::libdiv_strides_traits<noinline_walker_test_tensor::config_type>::type;
    
    strides_type strides{gtensor::detail::make_dividers<noinline_walker_test_tensor::config_type>(get_impl()->strides())};

    using tensor::tensor;
    noinline_walker_test_tensor(const base_type& base):
        base_type{base}
    {}
    
    auto begin()const{return iterator_type{get_impl()->as_walker_maker()->create_walker(), get_impl()->shape(), strides};}
    auto end()const{return iterator_type{get_impl()->as_walker_maker()->create_walker(), get_impl()->shape(), strides, get_impl()->size()};}
};



}   //end of namespace benchmark_walker




TEMPLATE_TEST_CASE("test_benchmark_iterators","[benchmark_walker]", gtensor::config::mode_div_native){
    using value_type = float;
    using inline_walker_tensor_type = benchmark_walker::inline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using noinline_walker_tensor_type = benchmark_walker::noinline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;

    SECTION("test_inline_walker_iterator"){
        inline_walker_tensor_type t1{{1,2,3}};
        inline_walker_tensor_type t2{{1},{2},{3}};
        inline_walker_tensor_type e = t1+t2;
        auto e_begin = e.begin();
        auto e_end = e.end();        
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{2,3,4,3,4,5,4,5,6}.begin()));
    }
    SECTION("test_noinline_walker_iterator"){
        noinline_walker_tensor_type t1{{1,2,3}};
        noinline_walker_tensor_type t2{{1},{2},{3}};
        noinline_walker_tensor_type e = t1+t2;
        auto e_begin = e.begin();
        auto e_end = e.end();        
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{2,3,4,3,4,5,4,5,6}.begin()));
    }
}

TEMPLATE_TEST_CASE("benchmark_walker","[benchmark_walker]", gtensor::config::mode_div_native){
    using value_type = float;
    using config_type = test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl<value_type>;
    using shape_type = typename config_type::shape_type;
    using inline_walker_tensor_type = benchmark_walker::inline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using noinline_walker_tensor_type = benchmark_walker::noinline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    
    auto make_inline = [](const auto& shape, const auto& v){
        return inline_walker_tensor_type(v,shape);
    };
    auto make_noinline = [](const auto& shape, const auto& v){return noinline_walker_tensor_type(v,shape);};

    auto iterate = [](const auto& t){
        auto t_it = t.begin();
        auto t_end = t.end();
        while (t_it!=t_end){
            ++t_it;
        }
        return t_it;
    };
    
    auto iterate_with_deref = [](const auto& t){
        auto t_it = t.begin();
        auto t_end = t.end();
        std::size_t c{};
        while (t_it!=t_end){
            if (*t_it > 2){
                ++c;
            }
            ++t_it;
        }
        return c;
    };

    shape_type shape1{1,1,3,1,5,1,7,1,9,1};
    shape_type shape2{1,2,1,4,1,6,1,8,1,10};
    
    // shape_type shape1{1,2,1,4,1,6,1,8,1,10};
    // shape_type shape2{1,2,3,4,5,6,7,8,9,10};

    // shape_type shape1{1, 10000};
    // shape_type shape2{10,10000};
    // shape_type shape1{1,3};
    // shape_type shape2{3,1};



    inline_walker_tensor_type t1_inline = make_inline(shape1, 0);
    inline_walker_tensor_type t2_inline = make_inline(shape2, 1);
    noinline_walker_tensor_type t1_noinline = make_noinline(shape1, 0);
    noinline_walker_tensor_type t2_noinline = make_noinline(shape2, 1);
    
    // inline_walker_tensor_type e_inline = make_inline(shape1, 0) + make_inline(shape2, 1);
    // noinline_walker_tensor_type e_noinline = make_noinline(shape1, 0) + make_noinline(shape2, 1);

    inline_walker_tensor_type e_inline = t2_inline+t1_inline+t1_inline+t1_inline+t1_inline+t1_inline+t1_inline+t1_inline+t1_inline+t1_inline+t1_inline+t1_inline;
    noinline_walker_tensor_type e_noinline = t2_noinline+t1_noinline+t1_noinline+t1_noinline+t1_noinline+t1_noinline+t1_noinline+t1_noinline+t1_noinline+t1_noinline+t1_noinline+t1_noinline;

    BENCHMARK("inline_walker_iterator"){
        return iterate_with_deref(e_inline);
    };
    BENCHMARK("noinline_walker_iterator"){
        return iterate_with_deref(e_noinline);
    };

    // BENCHMARK("inline_walker_iterator"){
    //     return iterate(e_inline);
    // };
    // BENCHMARK("noinline_walker_iterator"){
    //     return iterate(e_noinline);
    // };



}