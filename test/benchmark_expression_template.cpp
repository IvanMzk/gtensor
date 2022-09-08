
#include <variant>
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "tensor.hpp"

#define BENCHMARK_EXPRESSION_TEMPLATE_BINARY_OPERATOR(NAME,IMPL)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
auto NAME(const test_tensor<ValT1, CfgT, ImplT1>& op1, const test_tensor<ValT2, CfgT, ImplT2>& op2){return IMPL(op1,op2);}

#define BENCHMARK_EXPRESSION_TEMPLATE_BINARY_OPERATOR_IMPL(NAME,OP)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
static inline auto NAME(const test_tensor<ValT1, CfgT, ImplT1>& op1, const test_tensor<ValT2, CfgT, ImplT2>& op2){\
    using operation_type = OP;\
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));\
    using operand1_type = ImplT1;\
    using operand2_type = ImplT2;\
    using engine_type = typename engine_traits<evaluating_tensor<result_type, CfgT, operation_type, void, operand1_type, operand2_type>>::type;\
    using impl_type = evaluating_tensor<result_type, CfgT, operation_type, engine_type, operand1_type, operand2_type>;\
    return test_tensor<result_type,CfgT, impl_type>{std::make_shared<impl_type>(engine_type{}, op1.impl(),op2.impl())};\
}

namespace expression_template_without_dispatching{
using gtensor::tensor;
using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::expression_template_elementwise_engine;
using gtensor::evaluating_walker;
using gtensor::multiindex_iterator;

template<typename ValT, typename CfgT, typename ImplT = storage_tensor<ValT,CfgT>>
class test_tensor : public tensor<ValT,CfgT, ImplT>{
    
public:
    using tensor<ValT,CfgT, ImplT>::tensor;
    auto impl()const{return tensor::impl();}
    auto engine()const{return tensor::impl()->engine();}
    auto begin()const{
        using iterator_type = multiindex_iterator<ValT,CfgT,decltype(engine().create_walker())>;
        return iterator_type{engine().create_walker(), impl()->shape(), gtensor::detail::strides_div(*impl()->descriptor().as_descriptor_with_libdivide())};
    }
    auto end()const{
        using iterator_type = multiindex_iterator<ValT,CfgT,decltype(engine().create_walker())>;
        return iterator_type{engine().create_walker(), impl()->shape(), gtensor::detail::strides_div(*impl()->descriptor().as_descriptor_with_libdivide()), impl()->size()};
    }
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class no_dispatching_engine : public expression_template_elementwise_engine< no_dispatching_engine<ValT,CfgT,F,Ops...>, ValT,CfgT,F,Ops...>
{
public:
    
    template<typename...Wks>
    auto create_walker_helper(Wks&&...walkers)const{
        return evaluating_walker<ValT,CfgT,F,Wks...>{root()->shape(),std::forward<Wks>(walkers)...};
    }
    template<std::size_t...I>
    auto create_walker_helper(std::index_sequence<I...>)const{
        return create_walker_helper(static_cast<Ops*>(std::get<I>(root()->operands()).get())->engine().create_walker()...);
    }
    auto create_walker()const{
        return create_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
        
    }
};

template<typename> struct engine_traits;
template<typename ValT, typename CfgT, typename F, typename Dummy, typename...Ops> 
struct engine_traits<evaluating_tensor<ValT, CfgT, F, Dummy, Ops...>>{using type = no_dispatching_engine<ValT,CfgT,F,Ops...>;};

BENCHMARK_EXPRESSION_TEMPLATE_BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add);
BENCHMARK_EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator+, operator_add_impl);

}   //end of namespace expression_template_without_dispatching

namespace expression_template_dispatch_in_walker{
using gtensor::tensor;
using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::expression_template_elementwise_engine;
using gtensor::expression_template_storage_engine;
using gtensor::evaluating_walker;
using gtensor::storage_trivial_walker;
using gtensor::evaluating_trivial_walker;
using gtensor::multiindex_iterator;
using gtensor::basic_walker;

template<typename ValT, typename CfgT, typename F, typename...Wks>
class evaluating_dispatching_walker : private basic_walker<ValT, CfgT, typename CfgT::index_type>
{    
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    
    std::tuple<Wks...> walkers;
    F f{};
    bool is_trivial;    
        
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
    template<std::size_t...I>
    value_type subscription_helper(const index_type& idx, std::index_sequence<I...>) const {return f(std::get<I>(walkers)[idx]...);}
    template<std::size_t...I>
    void walk_helper(const index_type& direction, const index_type& steps, std::index_sequence<I...>){(std::get<I>(walkers).walk(direction,steps),...);}

public:

    evaluating_dispatching_walker(const shape_type& shape_, const shape_type& strides_, bool is_trivial_, Wks&&...walkers_):
        basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, index_type{0}},
        is_trivial{is_trivial_},
        walkers{std::move(walkers_)...}
    {}

    index_type dim()const{return dim_;}

    //walk method without check to utilize in evaluating_indexer
    void walk_without_check(const index_type& direction, const index_type& steps){
        if (is_trivial){
            basic_walker::walk(direction, steps);            
        }else{
            walk_helper(direction,steps,std::make_index_sequence<sizeof...(Wks)>{});                    
        }
    }    
    void walk(const index_type& direction, const index_type& steps){
        if (is_trivial){
            basic_walker::walk(direction, steps);
        }else{
            if (basic_walker::can_walk(direction)){
                walk_helper(direction,steps,std::make_index_sequence<sizeof...(Wks)>{});
            }
        }
    }
    void step(const index_type& direction){
        if (is_trivial){
            basic_walker::step(direction);
        }else{
            if (basic_walker::can_walk(direction)){
                step_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
            }            
        }
    }
    void step_back(const index_type& direction){
        if (is_trivial){
            basic_walker::step_back(direction);
        }else{
            if (basic_walker::can_walk(direction)){
                step_back_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
            }
        }
    }
    void reset(const index_type& direction){
        if (is_trivial){
            basic_walker::reset(direction);
        }else{
            if (basic_walker::can_walk(direction)){
                reset_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
            }            
        }
    }
    void reset(){
        if (is_trivial){
            basic_walker::reset();
        }else{
            reset_helper(std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    value_type operator[](const index_type& idx)const{return subscription_helper(idx, std::make_index_sequence<sizeof...(Wks)>{});}
    value_type operator*() const {
        if (is_trivial){
            return operator[](basic_walker::cursor());
        }else{
            return deref_helper(std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
};

template<typename ValT, typename CfgT, typename ImplT = storage_tensor<ValT,CfgT, engine_traits<storage_tensor<ValT,CfgT,void>>::type>>
class test_tensor : public tensor<ValT,CfgT, ImplT>{
    
public:
    using tensor<ValT,CfgT, ImplT>::tensor;
    auto impl()const{return tensor::impl();}
    auto engine()const{return tensor::impl()->engine();}
    auto begin()const{
        using iterator_type = multiindex_iterator<ValT,CfgT,decltype(engine().create_walker())>;
        return iterator_type{engine().create_walker(), impl()->shape(), gtensor::detail::strides_div(*impl()->descriptor().as_descriptor_with_libdivide())};
    }
    auto end()const{
        using iterator_type = multiindex_iterator<ValT,CfgT,decltype(engine().create_walker())>;
        return iterator_type{engine().create_walker(), impl()->shape(), gtensor::detail::strides_div(*impl()->descriptor().as_descriptor_with_libdivide()), impl()->size()};
    }
};

template<typename ValT, typename CfgT>
class dispatching_in_walker_storage_engine : public expression_template_storage_engine<dispatching_in_walker_storage_engine<ValT,CfgT>,ValT,CfgT>
{
public:    
    using expression_template_storage_engine::expression_template_storage_engine;
    auto create_trivial_walker()const{
        return create_walker();
    }
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class dispatching_in_walker_engine : public expression_template_elementwise_engine< dispatching_in_walker_engine<ValT,CfgT,F,Ops...>, ValT,CfgT,F,Ops...>
{
public:    
    template<typename...Wks>
    auto create_broadcast_walker_helper(Wks&&...walkers)const{
        return evaluating_dispatching_walker<ValT,CfgT,F,Wks...>{root()->shape(),root()->strides(), false ,std::forward<Wks>(walkers)...};
    }
    auto create_broadcast_walker()const{
        return std::apply(
            [this](const auto&...args){return create_broadcast_walker_helper(static_cast<Ops*>(args.get())->engine().create_walker()...);},
            root()->operands()
        );
    }
    
    template<typename...Wks>
    auto create_trivial_walker_helper(Wks&&...walkers)const{
        return evaluating_dispatching_walker<ValT,CfgT,F,Wks...>{root()->shape(),root()->strides(), true ,std::forward<Wks>(walkers)...};
    }
    auto create_trivial_walker()const{
        return std::apply(
            [this](const auto&...args){return create_trivial_walker_helper(static_cast<Ops*>(args.get())->engine().create_trivial_walker()...);},
            root()->operands()
        );
    }    
    auto create_walker()const{
        if (is_trivial()){
            return create_trivial_walker();
        }else{
            return create_broadcast_walker();
        }
    }
};

template<typename> struct engine_traits;
template<typename ValT, typename CfgT, typename EngineT> 
struct engine_traits<storage_tensor<ValT,CfgT,EngineT>>{using type = dispatching_in_walker_storage_engine<ValT,CfgT>;};
template<typename ValT, typename CfgT, typename F, typename Dummy, typename...Ops> 
struct engine_traits<evaluating_tensor<ValT, CfgT, F, Dummy, Ops...>>{using type = dispatching_in_walker_engine<ValT,CfgT,F,Ops...>;};

BENCHMARK_EXPRESSION_TEMPLATE_BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add);
BENCHMARK_EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator+, operator_add_impl);

}   //end of namespace expression_template_dispatch_in_walker

namespace expression_template_variant_dispatch{
using gtensor::tensor;
using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::expression_template_elementwise_engine;
using gtensor::expression_template_storage_engine;
using gtensor::expression_template_engine_base;
using gtensor::engine_root_accessor;
using gtensor::evaluating_walker;
using gtensor::evaluating_walker_polymorphic;
using gtensor::storage_walker;
using gtensor::storage_trivial_walker;
using gtensor::evaluating_trivial_walker;
using gtensor::evaluating_trivial_root_walker;
using gtensor::walker;
using gtensor::multiindex_iterator;
using gtensor::detail::type_list;
using gtensor::detail::list_concat;
using gtensor::detail::cross_product;
using gtensor::detail::is_trivial;

template<typename ValT, typename CfgT, typename ImplT = storage_tensor<ValT,CfgT, engine_traits<storage_tensor<ValT,CfgT,void>>::type>>
class test_tensor : public tensor<ValT,CfgT, ImplT>{
    
    struct polymorphic_walker_maker{        
        auto&& operator()(walker<ValT,CfgT>&& w){
            return std::move(w);
        }
        template<typename...Ts>
        auto operator()(evaluating_walker<ValT,CfgT,Ts...>&& w){
            using impl_type = evaluating_walker<ValT,CfgT,Ts...>;
            return walker<ValT,CfgT>{std::make_unique<evaluating_walker_polymorphic<ValT,CfgT,impl_type>>(std::move(w))};
        }
        template<typename...Ts>
        auto operator()(evaluating_trivial_root_walker<ValT,CfgT,Ts...>&& w){
            using impl_type = evaluating_trivial_root_walker<ValT,CfgT,Ts...>;
            return walker<ValT,CfgT>{std::make_unique<evaluating_walker_polymorphic<ValT,CfgT,impl_type>>(std::move(w))};
        }        
    };

public:
    using tensor<ValT,CfgT, ImplT>::tensor;
    auto impl()const{return tensor::impl();}
    auto engine()const{return tensor::impl()->engine();}
    
    auto begin()const{
        using iterator_type = multiindex_iterator<ValT,CfgT,walker<ValT,CfgT>>;
        return std::visit(
            [this](auto&& walker){
                return iterator_type{polymorphic_walker_maker{}(std::forward<decltype(walker)>(walker)), impl()->shape(), impl()->descriptor().as_descriptor_with_libdivide()->strides_libdivide()};
            },
            engine().create_walker()
        );        
    }
    auto end()const{
        using iterator_type = multiindex_iterator<ValT,CfgT,walker<ValT,CfgT>>;
        return std::visit(
            [this](auto&& walker){
                return iterator_type{polymorphic_walker_maker{}(std::forward<decltype(walker)>(walker)), impl()->shape(), impl()->descriptor().as_descriptor_with_libdivide()->strides_libdivide(), impl()->size()};
            },
            engine().create_walker()
        );        
    }    
    template<typename W>
    auto begin(W&& w)const{
        using iterator_type = multiindex_iterator<ValT,CfgT,std::decay_t<W>>;
        return iterator_type{std::forward<W>(w), impl()->shape(), gtensor::detail::strides_div(*impl()->descriptor().as_descriptor_with_libdivide())};
    }
    template<typename W>
    auto end(W&& w)const{
        using iterator_type = multiindex_iterator<ValT,CfgT,std::decay_t<W>>;
        return iterator_type{std::forward<W>(w), impl()->shape(), gtensor::detail::strides_div(*impl()->descriptor().as_descriptor_with_libdivide()), impl()->size()};
    }
};

template<typename> struct make_variant_type;
template<typename...Ts> struct make_variant_type<type_list<Ts...>>{
    using type = std::variant<Ts...>;
};

template<typename ValT, typename CfgT>
class variant_dispatch_storage_engine : 
    public expression_template_engine_base<ValT, CfgT>,
    public engine_root_accessor<storage_tensor, ValT, CfgT, variant_dispatch_storage_engine<ValT,CfgT>>
{    
public:
    using walker_types = type_list<storage_walker<ValT,CfgT>>;
    using trivial_walker_type = storage_trivial_walker<ValT,CfgT>;
    using variant_type = typename make_variant_type<walker_types>::type;
    variant_dispatch_storage_engine() = default;    
    template<typename R>
    variant_dispatch_storage_engine(R* root_):
        engine_root_accessor{root_}
    {}

    bool is_trivial()const override{return true;}
    auto create_walker()const{return variant_type{storage_walker<ValT,CfgT>{root()->shape(),root()->strides(),root()->data()}};}
    auto create_trivial_walker()const{return storage_trivial_walker<ValT,CfgT>{root()->data()};}
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class variant_dispatch_engine : 
    public expression_template_engine_base<ValT, CfgT>,
    public engine_root_accessor<evaluating_tensor, ValT, CfgT, F, variant_dispatch_engine<ValT,CfgT,F,Ops...>, Ops...>
{
private:
    static constexpr std::size_t max_walker_types_size = 30;
    static constexpr std::size_t walker_types_size = (Ops::engine_type::walker_types::size*...);
    template<typename...Us> using evaluating_walker_alias = evaluating_walker<ValT, CfgT, F, Us...>;
    
    template<bool> struct walker_types_traits{                
        using type = typename cross_product<evaluating_walker_alias, typename Ops::engine_type::walker_types...>::type;            
    };
    template<> struct walker_types_traits<false>{        
        using type = type_list<walker<ValT,CfgT>>;
    };
public:
    using value_type = ValT;
    using trivial_walker_type = evaluating_trivial_root_walker<ValT,CfgT,F,typename Ops::engine_type::trivial_walker_type...>;
    using walker_types = typename list_concat< 
            //type_list<storage_walker<ValT,CfgT>>, 
            type_list<trivial_walker_type>,
            typename walker_types_traits<(walker_types_size<max_walker_types_size)>::type >::type;
    using variant_type = typename make_variant_type<walker_types>::type;
    
    bool is_trivial()const override{return expression_template_variant_dispatch::is_trivial(root()->size(),root()->operands());}    

    // template<typename...Wks>
    // auto create_broadcast_walker_helper(std::true_type, Wks&&...walkers)const{
    //     return variant_type{evaluating_walker<ValT,CfgT,F,std::decay_t<Wks>...>{root()->shape(),std::forward<Wks>(walkers)...}};
    // }
    // template<typename...Wks>
    // auto create_broadcast_walker_helper(std::false_type, Wks&&...walkers)const{
    //     using evaluating_walker_type = evaluating_walker<ValT,CfgT,F,std::decay_t<Wks>...>;
    //     using evaluating_walker_polymorphic_type = evaluating_walker_polymorphic<ValT,CfgT,evaluating_walker_type>;
    //     return variant_type{
    //         walker<ValT,CfgT>{
    //             std::make_unique<evaluating_walker_polymorphic_type>(evaluating_walker_type{root()->shape(),std::forward<Wks>(walkers)...})
    //         }            
    //     };
    // }

    // auto create_broadcast_walker()const{
    //     return std::apply(
    //         [this](const auto&...operands){
    //             return std::visit(
    //                 [this](auto&&...walkers){
    //                     return create_broadcast_walker_helper(std::integral_constant<bool,(walker_types_size<max_walker_types_size)>{}, std::forward<decltype(walkers)>(walkers)...);
    //                 },
    //                 static_cast<Ops*>(operands.get())->engine().create_walker()...
    //             );
    //         },
    //         root()->operands()
    //     );
    // }    
    // auto create_trivial_walker()const{
    //     return std::apply(
    //         [this](const auto&...operands){
    //             return [this](auto&&...walkers){
    //                 return evaluating_trivial_root_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>{root()->shape(),root()->strides(),std::forward<decltype(walkers)>(walkers)...};
    //             }(static_cast<Ops*>(operands.get())->engine().create_trivial_walker()...);
    //         },
    //         root()->operands()
    //     );
    // }    
    
    // auto create_walker()const{
    //     if (is_trivial()){
    //         return variant_type{create_trivial_walker()};
    //     }else{
    //         return create_broadcast_walker();
    //     }
    // }
    
};

template<typename> struct engine_traits;
template<typename ValT, typename CfgT, typename Dummy> 
struct engine_traits<storage_tensor<ValT,CfgT,Dummy>>{using type = variant_dispatch_storage_engine<ValT,CfgT>;};
template<typename ValT, typename CfgT, typename F, typename Dummy, typename...Ops> 
struct engine_traits<evaluating_tensor<ValT, CfgT, F, Dummy, Ops...>>{using type = variant_dispatch_engine<ValT,CfgT,F,Ops...>;};

BENCHMARK_EXPRESSION_TEMPLATE_BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add);
BENCHMARK_EXPRESSION_TEMPLATE_BINARY_OPERATOR_IMPL(operator_mul_impl, gtensor::binary_operations::mul);
BENCHMARK_EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator+, operator_add_impl);
BENCHMARK_EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator*, operator_mul_impl);

}   //end of namespace expression_template_variant_dispatch

namespace benchmark_helpers{

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

}   //end of namespace benchmark_helpers

TEST_CASE("test_expression_template_without_dispatching","[benchmark_expression_template]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using test_tensor_type = expression_template_without_dispatching::test_tensor<value_type,config_type>;

    test_tensor_type t1{{1,2,3}};
    test_tensor_type t2{{1},{2},{3}};
    test_tensor_type t3{-2};
    auto e = t2+t1+t2+t3;         
    REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));    
}

TEST_CASE("expression_template_dispatch_in_walker","[benchmark_expression_template]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using test_tensor_type = expression_template_dispatch_in_walker::test_tensor<value_type,config_type>;
    using benchmark_helpers::make_asymmetric_tree;

    test_tensor_type t1{{1,2,3}};
    test_tensor_type t2{{1},{2},{3}};
    test_tensor_type t3{-2};
    auto e = t2+t1+t2+t3;
    REQUIRE(!e.engine().is_trivial());
    REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));    
    
    auto e_trivial_tree = t1+t1+t1+t1;
    REQUIRE(e_trivial_tree.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_tree.begin(), e_trivial_tree.end(), std::vector<float>{4,8,12}.begin()));
    
    auto e_trivial_tree1 = e_trivial_tree + e_trivial_tree;
    REQUIRE(e_trivial_tree1.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_tree1.begin(), e_trivial_tree1.end(), std::vector<float>{8,16,24}.begin()));
    
    auto e_trivial_subtree = e_trivial_tree + t2 + t3;
    REQUIRE(!e_trivial_subtree.engine().is_trivial());
    REQUIRE(std::equal(e_trivial_subtree.begin(), e_trivial_subtree.end(), std::vector<float>{3,7,11,4,8,12,5,9,13}.begin()));    
}

// TEST_CASE("test_expression_template_variant_dispatch","[benchmark_expression_template]"){
//     using value_type = float;
//     using config_type = gtensor::config::default_config;
//     using test_tensor_type = expression_template_variant_dispatch::test_tensor<value_type,config_type>;
//     using gtensor::multiindex_iterator;
//     using gtensor::detail::type_list;
//     using gtensor::binary_operations::add;
//     using gtensor::evaluating_walker;
//     using gtensor::evaluating_trivial_root_walker;
//     using gtensor::storage_walker;
//     using gtensor::walker;
//     using gtensor::storage_trivial_walker;

//     // SECTION("test_walker_types_with_trivial"){
//     //     using gtensor::detail::type_list;
//     //     using gtensor::storage_walker;
//     //     using gtensor::storage_trivial_walker;
//     //     using gtensor::evaluating_walker;
//     //     using gtensor::evaluating_trivial_root_walker;
//     //     using gtensor::walker;
//     //     using gtensor::binary_operations::add;
//     //     using gtensor::binary_operations::mul;

//     //     test_tensor_type t{1,2,3};        
//     //     REQUIRE(std::is_same_v<typename std::decay_t<decltype(t.engine())>::walker_types, type_list<storage_walker<value_type, config_type>>>);
            
//     //     auto e1 = t+t;        
//     //     REQUIRE(std::decay_t<decltype(e1.engine())>::walker_types::size == 2);
//     //     REQUIRE(std::is_same_v<
//     //         std::decay_t<decltype(e1.engine())>::walker_types, 
//     //         type_list< 
//     //             //storage_walker<value_type, config_type>,
//     //             evaluating_trivial_root_walker<value_type,config_type,add,storage_trivial_walker<value_type,config_type>,storage_trivial_walker<value_type,config_type>>,
//     //             evaluating_walker<value_type, config_type, add, storage_walker<value_type, config_type>, storage_walker<value_type, config_type> >>>
//     //     );
//     //     auto e2 = e1+e1;
//     //     REQUIRE(std::decay_t<decltype(e2.engine())>::walker_types::size == 5);        
//     //     auto e3 = e2+e2;
//     //     REQUIRE(std::decay_t<decltype(e3.engine())>::walker_types::size == 26);
//     //     auto e4 = e3+e3;
//     //     REQUIRE(std::decay_t<decltype(e4.engine())>::walker_types::size == 2);
//     //     using e4_triv_type = typename std::decay_t<decltype(e4.engine())>::trivial_walker_type;
//     //     REQUIRE(std::is_same_v<std::decay_t<decltype(e4.engine())>::walker_types, type_list<e4_triv_type, walker<value_type, config_type>>>);
//     //     auto e5 = e4*e4;
//     //     REQUIRE(std::decay_t<decltype(e5.engine())>::walker_types::size == 5);
//     //     REQUIRE(std::is_same_v<
//     //         std::decay_t<decltype(e5.engine())>::walker_types, 
//     //         type_list<
//     //             std::decay_t<decltype(e5.engine())>::trivial_walker_type,
//     //             evaluating_walker<value_type, config_type, mul, e4_triv_type, e4_triv_type>, 
//     //             evaluating_walker<value_type, config_type, mul, e4_triv_type, walker<value_type, config_type>>, 
//     //             evaluating_walker<value_type, config_type, mul, walker<value_type, config_type>, e4_triv_type>, 
//     //             evaluating_walker<value_type, config_type, mul, walker<value_type, config_type>, walker<value_type, config_type>>
//     //             >>
//     //         );
//     // }
//     // SECTION("test_walker_types"){
//     //     using gtensor::detail::type_list;
//     //     using gtensor::storage_walker;
//     //     using gtensor::evaluating_walker;
//     //     using gtensor::walker;
//     //     using gtensor::binary_operations::add;
//     //     using gtensor::binary_operations::mul;

//     //     test_tensor_type t{1,2,3};        
//     //     REQUIRE(std::is_same_v<typename std::decay_t<decltype(t.engine())>::walker_types, type_list<storage_walker<value_type, config_type>>>);
            
//     //     auto e1 = t+t;        
//     //     REQUIRE(std::decay_t<decltype(e1.engine())>::walker_types::size == 2);
//     //     REQUIRE(std::is_same_v<
//     //         std::decay_t<decltype(e1.engine())>::walker_types, 
//     //         type_list< 
//     //             storage_walker<value_type, config_type>,
//     //             evaluating_walker<value_type, config_type, add, storage_walker<value_type, config_type>, storage_walker<value_type, config_type> >>>
//     //     );
//     //     auto e2 = e1+e1;
//     //     REQUIRE(std::decay_t<decltype(e2.engine())>::walker_types::size == 5);        
//     //     auto e3 = e2+e2;
//     //     REQUIRE(std::decay_t<decltype(e3.engine())>::walker_types::size == 26);
//     //     auto e4 = e3+e3;
//     //     REQUIRE(std::decay_t<decltype(e4.engine())>::walker_types::size == 1);
//     //     REQUIRE(std::is_same_v<std::decay_t<decltype(e4.engine())>::walker_types, type_list<walker<value_type, config_type>>>);
//     //     auto e5 = e4*e4;
//     //     REQUIRE(std::decay_t<decltype(e5.engine())>::walker_types::size == 2);
//     //     REQUIRE(std::is_same_v<
//     //         std::decay_t<decltype(e5.engine())>::walker_types, 
//     //         type_list<
//     //             storage_walker<value_type, config_type>,
//     //             evaluating_walker<value_type, config_type, mul, walker<value_type, config_type>, walker<value_type, config_type>> >>
//     //         );
//     // }
    
//     SECTION("test_simple_not_trivial_tree"){
//         test_tensor_type t1{1,2,3};        
//         test_tensor_type t2{1};
//         auto e = t1+t2;        
//         REQUIRE(!e.engine().is_trivial());
//         REQUIRE(decltype(e.engine())::walker_types::size == 2);        
//         std::visit(
//             [&e](auto&& w){
//                 using iterator_type = multiindex_iterator<value_type,config_type,std::decay_t<decltype(w)>>;
//                 auto it_begin = e.begin(w);                
//                 REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{2,3,4}.begin()));
//             },
//             e.engine().create_walker()
//         );
//         REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{2,3,4}.begin()));
//     }
//     SECTION("test_not_trivial_tree"){
//         test_tensor_type t1{{1,2,3}};
//         test_tensor_type t2{{1},{2},{3}};
//         test_tensor_type t3{-2};
        
//         auto e1 = t2+t1+t2;     //{3,4,5,5,6,7,7,8,9}        
//         REQUIRE(decltype(e1.engine())::walker_types::size == 3);
//         auto e2 = e1+e1;    //{6,8,10,10,12,14,14,16,18}
//         REQUIRE(decltype(e2.engine())::walker_types::size == 10);
//         auto e3 = e2+e2;     //{12,16,20,20,24,28,28,32,36}
//         REQUIRE(decltype(e3.engine())::walker_types::size == 2);
//         auto e4 = e3+e3;     //{24,32,40,40,48,56,56,64,72}
//         REQUIRE(decltype(e4.engine())::walker_types::size == 5);
//         auto e5 = e4+e4;     //{24,32,40,40,48,56,56,64,72}
//         REQUIRE(decltype(e5.engine())::walker_types::size == 26);
//         auto e = e5;
//         REQUIRE(!e.engine().is_trivial());
//         std::visit(
//             [&e](auto&& w){
//                 auto it_begin = e.begin(w);
//                 REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{48,64,80,80,96,112,112,128,144}.begin()));
//                 //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{24,32,40,40,48,56,56,64,72}.begin()));
//                 //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{12,16,20,20,24,28,28,32,36}.begin()));
//                 //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{6,8,10,10,12,14,14,16,18}.begin()));
//                 //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{1,2,3,3,4,5,5,6,7}.begin()));
//             },
//             e.engine().create_walker()
//         );
//         REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{48,64,80,80,96,112,112,128,144}.begin()));
//     }
    
//     SECTION("trivial_tree"){
//         test_tensor_type t1{{1,2,3}};
//         auto e1 = t1+t1+t1;     //{3,6,9}
//         REQUIRE(decltype(e1.engine())::walker_types::size == 3);
//         auto e2 = e1+e1;    //{6,12,18}
//         REQUIRE(decltype(e2.engine())::walker_types::size == 10);
//         auto e3 = e2+e2;    //{12,24,36}
//         REQUIRE(decltype(e3.engine())::walker_types::size == 2);
//         auto e4 = e3+e3;    //{24,48,72}
//         REQUIRE(decltype(e4.engine())::walker_types::size == 5);
//         auto e5 = e4+e4;    //{48,96,144}
//         REQUIRE(decltype(e5.engine())::walker_types::size == 26);
//         auto e = e5;
//         REQUIRE(e.engine().is_trivial());                
//         std::visit(
//             [&e](auto&& w){
//                 auto it_begin = e.begin(w);
//                 REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{48,96,144}.begin()));
//                 //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{12,24,36}.begin()));
//             },
//             e.engine().create_walker()
//         );
//         REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{48,96,144}.begin()));        
//     }
    
//     SECTION("trivial_subtree"){
//         test_tensor_type t1{{1,2,3}};
//         test_tensor_type t2{{1},{2},{3}};
//         test_tensor_type t3{-2};
//         auto e1 = t1+t1+t1;     //{3,6,9}
//         REQUIRE(decltype(e1.engine())::walker_types::size == 3);
//         auto e2 = e1+e1;    //{6,12,18}
//         REQUIRE(decltype(e2.engine())::walker_types::size == 10);
//         auto e3 = e2+e2;    //{12,24,36}
//         REQUIRE(decltype(e3.engine())::walker_types::size == 2);
//         auto e4 = e3+e3;    //{24,48,72}
//         REQUIRE(decltype(e4.engine())::walker_types::size == 5);
//         auto e5 = e4+e4;    //{48,96,144}
//         REQUIRE(decltype(e5.engine())::walker_types::size == 26);
//         REQUIRE(e5.engine().is_trivial());
//         auto e6 = e5+(t2+t3);    //{{47,95,143},{48,96,144},{49,97,145}}
//         REQUIRE(decltype(e6.engine())::walker_types::size == 2);
//         auto e = e6;
//         REQUIRE(!e.engine().is_trivial());
//         std::visit(
//             [&e](auto&& w){
//                 auto it_begin = e.begin(w);
//                 REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{47,95,143,48,96,144,49,97,145}.begin()));
//                 //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{48,96,144}.begin()));
//                 //REQUIRE(std::equal(it_begin, e.end(std::forward<decltype(w)>(w)), std::vector<float>{12,24,36}.begin()));
//             },
//             e.engine().create_walker()
//         );
//         REQUIRE(std::equal(e.begin(), e.end(), std::vector<float>{47,95,143,48,96,144,49,97,145}.begin()));
//     }
    
// }

TEST_CASE("benchmark_expression_template","[benchmark_expression_template]"){
    using value_type = float;
    using config_type = gtensor::config::default_config;
    using shape_type = typename config_type::shape_type;
    using benchmark_helpers::make_symmetric_tree;
    using benchmark_helpers::make_asymmetric_tree;
    using tensor_no_dispatch_type = expression_template_without_dispatching::test_tensor<value_type,config_type>;
    using tensor_walker_dispatch_type = expression_template_dispatch_in_walker::test_tensor<value_type,config_type>;
    using tensor_variant_dispatch_type = expression_template_variant_dispatch::test_tensor<value_type,config_type>;

     auto iterate_without_deref = [](const auto& t){
        auto t_it = t.begin();
        auto t_end = t.end();
        std::size_t c{};
        while (t_it!=t_end){            
            ++c;            
            ++t_it;
        }
        return c;
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
    
    // shape_type shape1{1,3000};
    // shape_type shape2{3000,1};
    
    // shape_type shape1{1,10000};
    // shape_type shape2{10000,1};

    //tensor_variant_dispatch_type t1(0, shape1);
    //tensor_variant_dispatch_type t2(0, shape2);
    //auto e = t1+t2+t1+t1+t1+t1+t1+t1+t1+t1 + t1+t1+t1+t1+t1;
    //auto e = make_asymmetric_tree<20>(tensor_variant_dispatch_type(0, shape1), tensor_variant_dispatch_type(0, shape2));
    //auto e = make_asymmetric_tree<20>(tensor_walker_dispatch_type(0, shape1), tensor_walker_dispatch_type(0, shape2));
    //auto e = make_asymmetric_tree<20>(tensor_no_dispatch_type(0, shape1), tensor_no_dispatch_type(0, shape2));
    //std::cout<<std::endl<<decltype(e.engine())::walker_types::size;

    enum class benchmark_kinds {iteration_and_dereference, iterator_construction_iteration_and_dereference};
    auto benchmark_kind = benchmark_kinds::iterator_construction_iteration_and_dereference;
    //auto benchmark_kind = benchmark_kinds::iteration_and_dereference;

    // SECTION("benchmark_shape(1,10000)_shape(10,10000)_asymmetric_tree_depth_50"){

    //     shape_type shape1{1, 10000};
    //     shape_type shape2{10,10000};

    //     static constexpr std::size_t tree_depth = 50;
    //     auto make_tree = [](const auto& t1, const auto& t2){return make_asymmetric_tree<tree_depth>(t1,t2);};

    //     auto e_no_dispatch = make_tree(tensor_no_dispatch_type(0, shape1),tensor_no_dispatch_type(0, shape2));        
    //     auto e_walker_dispatch = make_tree(tensor_walker_dispatch_type(0, shape1), tensor_walker_dispatch_type(0, shape2););
    //     auto e_variant_dispatch = make_tree(tensor_variant_dispatch_type(0, shape1), tensor_variant_dispatch_type(0, shape2););        
        
    //     if (benchmark_kind == benchmark_kinds::iteration_and_dereference)
    //     {
    //         BENCHMARK_ADVANCED("no_dispatch_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
    //             auto v = make_iterators(meter.runs(),e_no_dispatch);
    //             meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
    //         };
    //         BENCHMARK_ADVANCED("walker_dispatch_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
    //             auto v = make_iterators(meter.runs(),e_walker_dispatch);
    //             meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
    //         };
    //         BENCHMARK_ADVANCED("variant_dispatch_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
    //             auto v = make_iterators(meter.runs(),e_variant_dispatch);
    //             meter.measure([&just_iterate_with_deref, &v](int i) { return just_iterate_with_deref(v[i].first, v[i].second); });
    //         };            
    //     }

    //     if (benchmark_kind == benchmark_kinds::iterator_construction_iteration_and_dereference)
    //     {
    //         BENCHMARK_ADVANCED("no_dispatch_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
    //             meter.measure([&iterate_with_deref, &e_no_dispatch] { return iterate_with_deref(e_full); });
    //         };    
    //         BENCHMARK_ADVANCED("walker_dispatch_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
    //             meter.measure([&iterate_with_deref, &e_walker_dispatch] { return iterate_with_deref(e_full_split); });
    //         };    
    //         BENCHMARK_ADVANCED("variant_dispatch_iterator_construction_iteration_and_dereference")(Catch::Benchmark::Chronometer meter) {        
    //             meter.measure([&iterate_with_deref, &e_variant_dispatch] { return iterate_with_deref(e_full_v1); });
    //         };                
    //     }        
    // }   //end of SECTION("benchmark_shape(1,10000)_shape(10,10000)_assymetric_tree_depth_50")


}