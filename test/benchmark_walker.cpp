#define CATCH_CONFIG_ENABLE_BENCHMARKING
//#pragma inline_depth(255)
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
    
    auto begin()const{return iterator_type{get_impl()->as_expression()->create_walker(),get_impl()->shape(), strides};}
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

namespace true_expression_template{
using gtensor::stensor_impl;
using gtensor::expression_impl;
using gtensor::storage_walker_factory;
using gtensor::evaluating_walker_impl;
using gtensor::storage_walker_impl;
using gtensor::binary_operations::add;
using gtensor::multiindex_iterator_impl;
using gtensor::walker_maker;
using gtensor::basic_walker;


template<typename ValT, template<typename> typename Cfg>
class concrete_storage_walker :  private basic_walker<ValT,Cfg, const ValT*>
{   
    using base_basic_walker = basic_walker<ValT, Cfg, const ValT*>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

public:    
    concrete_storage_walker(const shape_type& shape_, const shape_type& strides_,  const value_type* data_):
        base_basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, data_}        
    {}
    void walk(const index_type& direction, const index_type& steps){base_basic_walker::walk(direction,steps);}
    void step(const index_type& direction){base_basic_walker::step(direction);}
    void step_back(const index_type& direction){base_basic_walker::step_back(direction);}
    void reset(const index_type& direction){base_basic_walker::reset(direction);}
    void reset(){base_basic_walker::reset();}
    value_type operator*() const {return *cursor();}
};


template<typename ValT, template<typename> typename Cfg, typename F, typename...Wks>
class concrete_evaluating_walker
{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    index_type dim_;
    gtensor::detail::shape_inverter<ValT,Cfg> shape;
    std::tuple<Wks...> walkers;
    F f{};
        
    template<std::size_t...I>
    void step_helper(const index_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).step(direction),...);}    
    template<std::size_t...I>
    void step_back_helper(const index_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).step_back(direction),...);}    
    template<std::size_t...I>
    void reset_helper(const index_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).reset(direction),...);}
    template<std::size_t...I>
    void reset_helper(std::index_sequence<I...>){(std::get<I>(walkers).reset(),...);}    
    template<std::size_t...I>
    value_type deref_helper(std::index_sequence<I...>) const {return f(*std::get<I>(walkers)...);}
protected:
    template<std::size_t...I>
    void walk_helper(const index_type& direction, const index_type& steps, std::index_sequence<I...>){(std::get<I>(walkers).walk(direction,steps),...);}
    index_type dim()const{return dim_;}
public:
    concrete_evaluating_walker(const shape_type& shape_, Wks&&...walkers_):
        dim_{static_cast<index_type>(shape_.size())},
        shape{shape_},
        walkers{std::move(walkers_)...}
    {}
    
    void walk(const index_type& direction, const index_type& steps){
        if (gtensor::detail::can_walk(direction,dim_,shape.element(direction))){
            walk_helper(direction,steps,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void step(const index_type& direction){
        if (gtensor::detail::can_walk(direction,dim_,shape.element(direction))){
            step_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void step_back(const index_type& direction){
        if (gtensor::detail::can_walk(direction,dim_,shape.element(direction))){
            step_back_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void reset(const index_type& direction){
        if (gtensor::detail::can_walk(direction,dim_,shape.element(direction))){
            reset_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void reset(){reset_helper(std::make_index_sequence<sizeof...(Wks)>{});}
    value_type operator*() const {return deref_helper(std::make_index_sequence<sizeof...(Wks)>{});}
};



template<typename ValT, template<typename> typename Cfg>
class test_stensor : public stensor_impl<ValT,Cfg>
{ 
    using base_stensor = stensor_impl<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;    

public:
    using walker_type = concrete_storage_walker<ValT,Cfg>;

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
    auto create_concrete_walker()const{
        return concrete_storage_walker<ValT, Cfg>{shape(),strides(),data()};
        //return storage_walker_impl<ValT,Cfg>{shape(), strides(), data()};
        //return storage_walker_factory<ValT,Cfg>::create_walker(shape(),strides(),data());
    }  
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class test_expression : public expression_impl<ValT,Cfg,F,Ops...>
{

    using base_expression = expression_impl<ValT,Cfg,F,Ops...>;
    using config_type = Cfg<ValT>;
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;  
    using test_stensor_type = test_stensor<ValT,Cfg>;  

public:
    using walker_type = concrete_evaluating_walker<ValT,Cfg,F,typename Ops::element_type::walker_type...>;
private:
    template<std::size_t...I>
    auto create_concrete_walker_helper(std::index_sequence<I...>)const{
        //using walker_type = evaluating_walker_impl<ValT,Cfg,F, decltype(std::declval<Ops>()->create_concrete_walker())...>;
        using walker_type = concrete_evaluating_walker<ValT,Cfg,F, decltype(std::declval<Ops>()->create_concrete_walker())...>;
        return walker_type{shape(),operand<I>()->create_concrete_walker()...};
    }    

    // template<std::size_t...I>
    // walker_type create_concrete_walker_helper(std::index_sequence<I...>)const{        
    //     return walker_type{shape(), operand<I>()->create_concrete_walker()...};
    // }    
    
    // template<std::size_t...I>
    // auto create_concrete_walker_helper(std::index_sequence<I...>)const{        
    //     return create_concrete_walker_helper(operand<I>()->create_concrete_walker()...);
    // }    
    // template<typename...W>
    // auto create_concrete_walker_helper(W&&...walkers)const{                
    //     return concrete_evaluating_walker<ValT,Cfg,F, W...>{shape(),std::forward<W>(walkers)...};
    // }    
public:
    template<typename...O>    
    explicit test_expression(O&&...operands_):
        base_expression{std::forward<O>(operands_)...}
    {}
    
    auto create_concrete_walker()const{return create_concrete_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});}
    auto begin()const{
        using iterator_type = multiindex_iterator_impl<ValT,Cfg,decltype(std::declval<test_expression>().create_concrete_walker())>;
        return iterator_type{create_concrete_walker(),shape(),gtensor::detail::strides_div(concrete_descriptor())};
    }
    auto end()const{
        using iterator_type = multiindex_iterator_impl<ValT,Cfg,decltype(std::declval<test_expression>().create_concrete_walker())>;
        return iterator_type{create_concrete_walker(),shape(),gtensor::detail::strides_div(concrete_descriptor()),size()};
    }
};

template<typename ValT, template<typename> typename Cfg, typename ImplT = test_stensor<ValT,Cfg>>
class static_tensor{
    using base_stensor = stensor_impl<ValT,Cfg>;
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
    const std::shared_ptr<ImplT>& impl()const{return impl_;}    
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
    using exp_type = test_expression<result_type, Cfg, operation_type, operand1_type, operand2_type>;
    return static_tensor<result_type,Cfg, exp_type>{std::make_shared<exp_type>(op1.impl(),op2.impl())};
}

}   //end of namespace true_expression_template


TEMPLATE_TEST_CASE("test_benchmark_iterators","[benchmark_walker]", gtensor::config::mode_div_native){
    using value_type = float;
    using partly_inline_tensor_type = benchmark_walker::inline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using noinline_tensor_type = benchmark_walker::noinline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using full_inline_tensor_type = true_expression_template::static_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;

    SECTION("test_storage_inline_walker_iterator"){
        partly_inline_tensor_type t1{{1,2,3}};
        partly_inline_tensor_type t2{{1},{2},{3}};
        partly_inline_tensor_type t3{-2};
        partly_inline_tensor_type e = t2+t1+t2+t3;
        auto e_begin = e.begin();
        auto e_end = e.end();        
        REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
    }
    SECTION("test_noinline_walker_iterator"){
        noinline_tensor_type t1{{1,2,3}};
        noinline_tensor_type t2{{1},{2},{3}};
        noinline_tensor_type t3{-2};
        noinline_tensor_type e = t2+t1+t2+t3;
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
    using partly_inline_tensor_type = benchmark_walker::inline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
    using noinline_tensor_type = benchmark_walker::noinline_walker_test_tensor<value_type, test_config::config_tmpl_div_mode_selector<TestType>::config_tmpl>;
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
    
    // shape_type shape1{1,1,3,1,5,1,7,1,9,1};
    // shape_type shape2{1,2,1,4,1,6,1,8,1,10};
    
    // shape_type shape1{1,2,1,4,1,6,1,8,1,10};
    // shape_type shape2{1,2,3,4,5,6,7,8,9,10};

    // shape_type shape1{1, 10000};
    // shape_type shape2{10,10000};
    shape_type shape1{1,3};
    shape_type shape2{3,1};

    static constexpr std::size_t tree_depth = 20;
    auto make_tree = [](const auto& t1, const auto& t2){return make_asymmetric_tree<tree_depth>(t1,t2);};

    tensor_type t1(0, shape1);
    tensor_type t2(0, shape2);
    partly_inline_tensor_type e_inline = make_tree(t1,t2);
    noinline_tensor_type e_noinline = make_tree(t1,t2);    

    full_inline_tensor_type t1_full(0, shape1);
    full_inline_tensor_type t2_full(0, shape2);
    auto e_full = make_tree(t1_full,t2_full);
    
    
    BENCHMARK("full_inline_iteration_and_dereference"){
        return iterate_with_deref(e_full);
    };
    BENCHMARK("partly_inline_iteration_and_dereference"){
        return iterate_with_deref(e_inline);
    };
    BENCHMARK("noinline_iteration_and_dereference"){
        return iterate_with_deref(e_noinline);
    };
}