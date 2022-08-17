#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#pragma inline_depth(255)
#include <vector>
#include <iterator>

#include "catch.hpp"
#include "tensor.hpp"
#include "walker_base.hpp"
#include "iterator.hpp"
#include "test_config.hpp"

namespace benchmark_walker{
using gtensor::multiindex_iterator;
using gtensor::walker;
using gtensor::tensor;
using gtensor::evaluating_tensor;

template<typename ValT, template<typename> typename Cfg>
struct inline_walker_test_tensor : public tensor<ValT,Cfg>{
    using config_type = Cfg<ValT>;
    using iterator_type = multiindex_iterator<ValT,Cfg,walker<ValT,Cfg>>;
    using strides_type = typename gtensor::detail::libdiv_strides_traits<inline_walker_test_tensor::config_type>::type;
    
    strides_type strides{gtensor::detail::make_dividers<inline_walker_test_tensor::config_type>(impl()->strides())};

    using tensor::tensor;
    inline_walker_test_tensor(const tensor& base):
        tensor{base}
    {}

    auto begin()const{return iterator_type{impl()->as_evaluating()->create_walker(),impl()->shape(), strides};}
    auto end()const{return iterator_type{impl()->as_evaluating()->create_walker(), impl()->shape(), strides, impl()->size()};}
};

template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_asymmetric_tree(const T1& t1, const T2& t2){
    return make_asymmetric_tree<Depth-1>(t1,t2+t1);
}
template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< Depth==1,int> = 0 >
auto make_asymmetric_tree(const T1& t1, const T2& t2){
    return t2+t1;
}

template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< (Depth>1) ,int> = 0 >
auto make_symmetric_tree(const T1& t1, const T2& t2){
    return make_symmetric_tree<Depth-1>(t2+t1,t2+t1);
}
template<std::size_t Depth, typename T1, typename T2, std::enable_if_t< Depth==1,int> = 0 >
auto make_symmetric_tree(const T1& t1, const T2& t2){
    return t2+t1;
}

}   //end of namespace benchmark_walker

namespace noinline_evaluation{
using gtensor::walker;
using gtensor::tensor;
using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::tensor_base;
using gtensor::storage_walker_polymorphic;
using gtensor::evaluating_walker_polymorphic;
using gtensor::binary_operations::add;
using gtensor::walker_base;
using gtensor::multiindex_iterator;

//noinline walker maker interface
template<typename ValT, template<typename> typename Cfg>
class noinline_walker_maker
{
    virtual walker<ValT,Cfg> create_noinline_walker()const = 0;
public:
    auto create_walker()const{return create_noinline_walker();}    
};

template<typename ValT, template<typename> typename Cfg>
class noinline_storage_tensor : 
    public storage_tensor<ValT,Cfg>,    
    public noinline_walker_maker<ValT,Cfg>
{             
    walker<ValT,Cfg> create_noinline_walker()const override{return std::make_unique<storage_walker_polymorphic<ValT,Cfg>>(shape(),strides(),data());}
public:    
    using storage_tensor::storage_tensor;
};

template<typename ValT, template<typename> typename Cfg>
class test_tensor_noinline : public tensor<ValT, Cfg>
{    
    using storage_tensor_type = noinline_storage_tensor<ValT,Cfg>;
    using iterator_type = multiindex_iterator<ValT,Cfg,walker<ValT,Cfg>>;
    using strides_type = typename gtensor::detail::libdiv_strides_traits<Cfg<ValT>>::type;

    strides_type strides{gtensor::detail::make_dividers<Cfg<ValT>>(impl()->strides())};

    template<typename Nested>
    test_tensor_noinline(std::initializer_list<Nested> init_data, int):        
        tensor(std::make_shared<storage_tensor_type>(init_data))
    {}
public:    
    test_tensor_noinline() = default;
    test_tensor_noinline(typename gtensor::detail::nested_initializer_list_type<value_type,1>::type init_data):test_tensor_noinline(init_data,0){}
    test_tensor_noinline(typename gtensor::detail::nested_initializer_list_type<value_type,2>::type init_data):test_tensor_noinline(init_data,0){}
    test_tensor_noinline(typename gtensor::detail::nested_initializer_list_type<value_type,3>::type init_data):test_tensor_noinline(init_data,0){}
    test_tensor_noinline(typename gtensor::detail::nested_initializer_list_type<value_type,4>::type init_data):test_tensor_noinline(init_data,0){}
    test_tensor_noinline(typename gtensor::detail::nested_initializer_list_type<value_type,5>::type init_data):test_tensor_noinline(init_data,0){}

    template<typename...Dims>
    test_tensor_noinline(const value_type& v, const Dims&...dims):        
        tensor(std::make_shared<storage_tensor_type>(v, dims...))
    {}

    test_tensor_noinline(std::shared_ptr<tensor_base<ValT,Cfg>>&& impl__):
        tensor(std::move(impl__))
    {}

    auto impl()const{return tensor::impl();}
    auto begin()const{return iterator_type{dynamic_cast<const noinline_walker_maker<ValT,Cfg>*>(impl()->impl().get())->create_walker(),impl()->shape(), strides};}
    auto end()const{return iterator_type{dynamic_cast<const noinline_walker_maker<ValT,Cfg>*>(impl()->impl().get())->create_walker(), impl()->shape(), strides, impl()->size()};}
    
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class noinline_evaluating_tensor : 
    public evaluating_tensor<ValT,Cfg,F,Ops...>,
    public noinline_walker_maker<ValT,Cfg>
{
    template<std::size_t...I>
    walker<ValT,Cfg> create_walker_helper(std::index_sequence<I...>)const{
        using walker_type = evaluating_walker_polymorphic<ValT,Cfg,F,decltype(dynamic_cast<noinline_walker_maker<ValT,Cfg>*>(std::declval<Ops>().get())->create_walker())...>;
        return std::make_unique<walker_type>(shape(),dynamic_cast<noinline_walker_maker<ValT,Cfg>*>(operand<I>().get())->create_walker()...);        
    }

    walker<ValT,Cfg> create_noinline_walker()const override{return create_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});}

public:
    using evaluating_tensor::evaluating_tensor;    
};

template<typename ValT1, typename ValT2, template<typename> typename Cfg>
static inline auto operator+(const test_tensor_noinline<ValT1, Cfg>& op1, const test_tensor_noinline<ValT2, Cfg>& op2){
    using operation_type = add;
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));
    using exp_operand1_type = std::shared_ptr<tensor_base<ValT1,Cfg>>;
    using exp_operand2_type = std::shared_ptr<tensor_base<ValT2,Cfg>>;
    using exp_type = noinline_evaluating_tensor<result_type, Cfg, operation_type, exp_operand1_type, exp_operand2_type>;
    return test_tensor_noinline<result_type,Cfg>{std::make_shared<exp_type>(op1.impl(),op2.impl())};
}

}   //end of namespace noinline_evaluation

namespace true_expression_template{
using gtensor::storage_tensor;
using gtensor::storage_walker;
using gtensor::evaluating_tensor;
using gtensor::evaluating_walker;
using gtensor::storage_walker_factory;
using gtensor::evaluating_walker_polymorphic;
using gtensor::storage_walker_polymorphic;
using gtensor::binary_operations::add;
using gtensor::multiindex_iterator;
using gtensor::basic_walker;

template<typename ValT, template<typename> typename Cfg>
class test_stensor : public storage_tensor<ValT,Cfg>
{ 
    using base_stensor = storage_tensor<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

public:    
    test_stensor() = default;
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,1>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,2>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,3>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,4>::type init_data):base_stensor(init_data){}
    test_stensor(typename gtensor::detail::nested_initializer_list_type<value_type,5>::type init_data):base_stensor(init_data){}

    template<typename...Dims>
    test_stensor(const value_type& v, const Dims&...dims):
        base_stensor(v, dims...)
    {}    
    auto create_concrete_walker()const{return storage_walker<ValT, Cfg>{shape(),strides(),data()};}  
    bool is_trivial()const{return storage_tensor::is_trivial();}
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class test_etensor : public evaluating_tensor<ValT,Cfg,F,Ops...>
{
    template<std::size_t...I>
    auto create_concrete_walker_helper(std::index_sequence<I...>)const{        
        using walker_type = evaluating_walker<ValT,Cfg,F, decltype(std::declval<Ops>()->create_concrete_walker())...>;
        return walker_type{shape(),operand<I>()->create_concrete_walker()...};
    }    
public:
    using evaluating_tensor::evaluating_tensor;

    bool is_trivial()const{return evaluating_tensor::is_trivial();}
    
    auto create_concrete_walker()const{return create_concrete_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});}
    auto begin()const{
        using iterator_type = multiindex_iterator<ValT,Cfg,decltype(std::declval<test_etensor>().create_concrete_walker())>;
        return iterator_type{create_concrete_walker(),shape(),gtensor::detail::strides_div(concrete_descriptor())};
    }
    auto end()const{
        using iterator_type = multiindex_iterator<ValT,Cfg,decltype(std::declval<test_etensor>().create_concrete_walker())>;
        return iterator_type{create_concrete_walker(),shape(),gtensor::detail::strides_div(concrete_descriptor()),size()};
    }
};

template<typename ValT, template<typename> typename Cfg, typename ImplT = test_stensor<ValT,Cfg>>
class static_tensor{
    using base_stensor = storage_tensor<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

    std::shared_ptr<ImplT> impl_; 

    template<typename Nested>
    static_tensor(std::initializer_list<Nested> init_data,int):
        impl_{new ImplT(init_data)}
    {}   

public:
    static_tensor() = default;
    static_tensor(std::shared_ptr<ImplT>&& impl__):
        impl_{std::move(impl__)}
    {}

    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,1>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,2>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,3>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,4>::type init_data):static_tensor(init_data,0){}
    static_tensor(typename gtensor::detail::nested_initializer_list_type<value_type,5>::type init_data):static_tensor(init_data,0){}

    template<typename...Dims>
    static_tensor(const value_type& v, const Dims&...dims):
        impl_{new ImplT(v, dims...)}
    {}
    auto impl()const{return impl_;}    
    const auto& shape()const{return impl()->shape();}
    auto create_concrete_walker()const{return impl()->create_concrete_walker();}
    auto begin()const{return impl()->begin();}
    auto end()const{return impl()->end();}
};

template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, template<typename> typename Cfg>
static inline auto operator+(const static_tensor<ValT1, Cfg, ImplT1>& op1, const static_tensor<ValT2, Cfg, ImplT2>& op2){
    using operation_type = add;
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));
    using operand1_type = std::shared_ptr<ImplT1>;
    using operand2_type = std::shared_ptr<ImplT2>;
    using exp_type = test_etensor<result_type, Cfg, operation_type, operand1_type, operand2_type>;
    return static_tensor<result_type,Cfg, exp_type>{std::make_shared<exp_type>(op1.impl(),op2.impl())};
}

}   //end of namespace true_expression_template


TEST_CASE("test_benchmark_helper_classes","[benchmark_walker]"){
    using value_type = float;
    using gtensor::config::default_config;
    using noinline_tensor_type = noinline_evaluation::test_tensor_noinline<value_type, default_config>;
    using partly_inline_tensor_type = benchmark_walker::inline_walker_test_tensor<value_type, default_config>;
    using full_inline_tensor_type = true_expression_template::static_tensor<value_type, default_config>;

    SECTION("test_noinline_tensor_iterator"){
        noinline_tensor_type t1{{1,2,3}};
        noinline_tensor_type t2{{1},{2},{3}};
        noinline_tensor_type t3{-2};
        noinline_tensor_type e = t2+t1+t2+t3;
        auto e_begin = e.begin();
        auto e_end = e.end();        
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
    }
    SECTION("test_partly_inline_tensor_iterator"){
        partly_inline_tensor_type t1{{1,2,3}};
        partly_inline_tensor_type t2{{1},{2},{3}};
        partly_inline_tensor_type t3{-2};
        partly_inline_tensor_type e = t2+t1+t2+t3;
        auto e_begin = e.begin();
        auto e_end = e.end();        
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
    }

    SECTION("test_full_inline_walker_iterator"){
        full_inline_tensor_type t1{1,2,3};
        full_inline_tensor_type t2{{1},{2},{3}};
        full_inline_tensor_type t3{-2};        
        auto e = t2+t1+t2+t3;        
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
    }
}

TEMPLATE_TEST_CASE("benchmark_walker","[benchmark_walker]", gtensor::config::mode_div_native){
    using value_type = float;
    using config_type = test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl<value_type>;
    using shape_type = typename config_type::shape_type;    
    using tensor_type = gtensor::tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;        
    using noinline_tensor_type = noinline_evaluation::test_tensor_noinline<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using partly_inline_tensor_type = benchmark_walker::inline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using full_inline_tensor_type = true_expression_template::static_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using benchmark_walker::make_asymmetric_tree;
    using benchmark_walker::make_symmetric_tree;
    
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
    
    auto just_iterate_with_deref = [](auto& it_begin, auto& it_end){        
        std::size_t c{};        
        while (it_begin!=it_end){
            if (*it_begin > 2){
                ++c;
            }
            ++it_begin;
        }
        return c;
    };
    
    auto make_iterators = [](std::size_t n, const auto& t){
        return std::vector<std::pair<decltype(t.begin()), decltype(t.end())>>(n, std::make_pair(t.begin(), t.end()));
    };

    // shape_type shape1{1,1,3,1,5,1,7,1,9,1};
    // shape_type shape2{1,2,1,4,1,6,1,8,1,10};
    
    // shape_type shape1{1,2,1,4,1,6,1,8,1,10};
    // shape_type shape2{1,2,3,4,5,6,7,8,9,10};

    shape_type shape1{1, 10000};
    shape_type shape2{10,10000};
    // shape_type shape1{1,3};
    // shape_type shape2{3,1};

    static constexpr std::size_t tree_depth = 60;
    auto make_tree = [](const auto& t1, const auto& t2){return make_asymmetric_tree<tree_depth>(t1,t2);};

    full_inline_tensor_type t1_full(0, shape1);
    full_inline_tensor_type t2_full(0, shape2);
    auto e_full = make_tree(t1_full,t2_full);
    
    tensor_type t1(0, shape1);
    tensor_type t2(0, shape2);
    partly_inline_tensor_type e_inline = make_tree(t1,t2);
    
    noinline_tensor_type t1_noinline(0, shape1);
    noinline_tensor_type t2_noinline(0, shape2);
    noinline_tensor_type e_noinline = make_tree(t1_noinline,t2_noinline);    
    
    BENCHMARK_ADVANCED("full_inline_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
        auto v = make_iterators(meter.runs(),e_full);
        meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
    };
    BENCHMARK_ADVANCED("partly_inline_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
        auto v = make_iterators(meter.runs(),e_inline);
        meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
    };
    BENCHMARK_ADVANCED("noinline_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {
        auto v = make_iterators(meter.runs(),e_noinline);
        meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
    };
    
    BENCHMARK_ADVANCED("full_inline_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
        meter.measure([&iterate_with_deref, &e_full] { return iterate_with_deref(e_full); });
    };
    BENCHMARK_ADVANCED("partly_inline_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
        meter.measure([&iterate_with_deref, &e_inline] { return iterate_with_deref(e_inline); });
    };
    BENCHMARK_ADVANCED("noinline_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
        meter.measure([&iterate_with_deref, &e_noinline] { return iterate_with_deref(e_noinline); });
    };    
}