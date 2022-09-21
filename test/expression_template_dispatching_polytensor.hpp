#ifndef EXPRESSION_TEMPLATE_DISPATCHING_POLYTENSOR_HPP_
#define EXPRESSION_TEMPLATE_DISPATCHING_POLYTENSOR_HPP_

#include <variant>
#include <functional>
#include "gtensor.hpp"
#include "walker.hpp"
#include "evaluating_walker.hpp"
#include "storage_walker.hpp"
#include "iterator.hpp"
#include "engine.hpp"



#define EXPRESSION_TEMPLATE_DISPATCHING_POLYTENSOR_BINARY_OPERATOR(NAME,IMPL)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
auto NAME(const test_tensor<ValT1, CfgT, ImplT1>& op1, const test_tensor<ValT2, CfgT, ImplT2>& op2){return IMPL(op1,op2);}

#define EXPRESSION_TEMPLATE_DISPATCHING_POLYTENSOR_BINARY_OPERATOR_IMPL(NAME,OP)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
static inline auto NAME(const test_tensor<ValT1, CfgT, ImplT1>& op1, const test_tensor<ValT2, CfgT, ImplT2>& op2){\
    using operation_type = OP;\
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));\
    using operand1_type = ImplT1;\
    using operand2_type = ImplT2;\
    using engine_type = evaluating_engine_traits<result_type, CfgT, operation_type, operand1_type, operand2_type>::type;\
    using impl_type = evaluating_tensor<result_type, CfgT, engine_type>;\
    return test_tensor<result_type,CfgT, impl_type>{std::make_shared<impl_type>(operation_type{}, op1.impl(),op2.impl())};\
}


namespace visitable_walker{

template<typename ValT, typename CfgT, typename VisitorT> class walker;

template<typename VisitorT>
class walker_visitable{
    using visitor_type = VisitorT;
public:
    virtual ~walker_visitable(){}
    virtual std::unique_ptr<walker_visitable> accept(visitor_type&)const = 0;
};

template<typename ValT, typename CfgT, typename VisitorT>
class walker_base : public walker_visitable<VisitorT>
{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using visitor_type = VisitorT;
public:
    virtual ~walker_base(){}
    virtual void walk(const index_type& direction, const index_type& steps) = 0;
    virtual void step(const index_type& direction) = 0;
    virtual void step_back(const index_type& direction) = 0;
    virtual void reset(const index_type& direction) = 0;
    virtual void reset() = 0;
    virtual value_type operator*() const = 0;
    virtual std::unique_ptr<walker_base<ValT,CfgT,visitor_type>> clone()const = 0;
};

template<typename ValT, typename CfgT, typename ImplT, typename VisitorT>
class walker_polymorphic : public walker_base<ValT, CfgT, VisitorT>
{
    using impl_type = ImplT;
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using visitor_type = VisitorT;

    impl_type impl_;

    std::unique_ptr<walker_base> clone()const override{return std::make_unique<walker_polymorphic>(*this);}

public:
    walker_polymorphic(impl_type&& impl__):
        impl_{std::move(impl__)}
    {}
    std::unique_ptr<walker_visitable<visitor_type>> accept(visitor_type& visitor)const override{return visitor.visit(*this);}
    void walk(const index_type& direction, const index_type& steps)override{return impl_.walk(direction,steps);}
    void step(const index_type& direction)override{return impl_.step(direction);}
    void step_back(const index_type& direction)override{return impl_.step_back(direction);}
    void reset(const index_type& direction)override{return impl_.reset(direction);}
    void reset()override{return impl_.reset();}
    value_type operator*() const override{return impl_.operator*();}
};

template<typename ValT, typename CfgT, typename VisitorT>
class walker{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using visitor_type = VisitorT;
    using impl_base_type = walker_base<ValT, CfgT, visitor_type>;
    std::unique_ptr<impl_base_type> impl;
public:
    walker(std::unique_ptr<impl_base_type>&& impl_):
        impl{std::move(impl_)}
    {}
    walker(const walker& other):
        impl{other.impl->clone()}
    {}
    walker(walker&& other) = default;

    auto accept(visitor_type& visitor)const{return impl->accept(visitor);}
    walker& walk(const index_type& direction, const index_type& steps){
        impl->walk(direction,steps);
        return *this;
    }
    walker& step(const index_type& direction){
        impl->step(direction);
        return *this;
    }
    walker& step_back(const index_type& direction){
        impl->step_back(direction);
        return *this;
    }
    walker& reset(const index_type& direction){
        impl->reset(direction);
        return *this;
    }
    walker& reset(){
        impl->reset();
        return *this;
    }
    value_type operator*() const{return impl->operator*();}
    auto& as_trivial()const{return static_cast<const walker_trivial_base<ValT,CfgT>&>(*impl.get());}
};

template<typename> struct walker_traits;
template<typename ValT, typename CfgT, typename VisitorT> struct walker_traits<walker<ValT,CfgT,VisitorT>>{using visitor_type = VisitorT;};

}

namespace expression_template_dispatching_polytensor{
using gtensor::tensor;
using gtensor::tensor_base_base;
using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::viewing_tensor;
using gtensor::storage_engine;
using gtensor::evaluating_engine;
using gtensor::storage_walker;
using gtensor::walker_polymorphic;
using gtensor::trivial_walker_polymorphic;
using gtensor::walker;
using gtensor::indexer;
using gtensor::storage_trivial_walker;
using gtensor::evaluating_walker;
using gtensor::evaluating_trivial_root_walker;
using gtensor::evaluating_trivial_walker;
using gtensor::multiindex_iterator;
using gtensor::engine_host_accessor;

template<typename V, typename TagT> struct node_details{
    using value_type = V;
    using type_tag = TagT;
};
template<typename> struct node_type_traits;
template<typename...Ts> struct node_type_traits<storage_tensor<Ts...>>{
    using type =  node_details<typename storage_tensor<Ts...>::value_type, gtensor::detail::storage_type_tag>;
};
template<typename...Ts> struct node_type_traits<evaluating_tensor<Ts...>>{
    using type =  node_details<typename evaluating_tensor<Ts...>::value_type, gtensor::detail::evaluating_type_tag>;
};
template<typename...Ts> struct node_type_traits<viewing_tensor<Ts...>>{
    using type =  node_details<typename viewing_tensor<Ts...>::value_type, gtensor::detail::viewing_type_tag>;
};

template<typename ValT, typename CfgT, typename ImplT = storage_tensor<ValT,CfgT, typename storage_engine_traits<ValT,CfgT>::type >>
class test_tensor : public tensor<ValT,CfgT, ImplT>{

public:
    using tensor<ValT,CfgT, ImplT>::tensor;
    auto impl()const{return tensor::impl();}
    auto& engine()const{return tensor::impl()->engine();}
    auto begin()const{
        using iterator_type = multiindex_iterator<ValT,CfgT,decltype(engine().create_walker())>;
        return iterator_type{engine().create_walker(), impl()->shape(), impl()->descriptor().as_descriptor_with_libdivide()->strides_libdivide()};
    }
    auto end()const{
        using iterator_type = multiindex_iterator<ValT,CfgT,decltype(engine().create_walker())>;
        return iterator_type{engine().create_walker(), impl()->shape(), impl()->descriptor().as_descriptor_with_libdivide()->strides_libdivide(), impl()->size()};
    }
};

template<typename ValT, typename CfgT>
class polytensor_storage_engine : public storage_engine<ValT,CfgT>
{
public:
    using storage_engine::storage_engine;
    using trivial_walker_type = storage_trivial_walker<ValT,CfgT>;
    using walker_type = storage_walker<ValT,CfgT>;
    bool is_trivial()const{return true;}
    auto create_walker()const{
        return storage_walker<ValT,CfgT>{host()->shape(),host()->strides(),data()};
    }
    auto create_trivial_walker()const{
        return storage_trivial_walker<ValT,CfgT>{data()};
    }
};

template<typename ValT, typename CfgT, typename F, typename WalkerT, typename...OperandsDetails>
class polytensor_dispatching_engine : public evaluating_engine<ValT,CfgT,F,OperandsDetails...>
{
    using typename engine_host_accessor::host_type;
    using operands_container_type = std::vector<std::shared_ptr<tensor_base_base<CfgT>>>;

    struct visitor{
        template<typename U>
        auto visit(const U&){return nullptr;}
    };
    using visitor_type = visitor;

    // template<typename...Us, typename...Ws>
    // auto test_visitor(const storage_walker<Us...>&, const Ws&...)const{}
    // template<typename ValT, typename CfgT, typename VisitorT, typename...Ws>
    // auto test_visitor(const visitable_walker::walker<ValT,CfgT,VisitorT>& w, const Ws&...walkers)const{
    //     using visitor_type = VisitorT;
    //     auto v = visitor_type{};
    //     w.accept(v);
    // }

    // template<typename...Us>
    // auto test_visitor(const Us&...walkers){
    //     //auto t = std::tie(walkers);
    //     (test_visitor(walkers),...);
    // }

    std::function<visitable_walker::walker<ValT,CfgT,visitor_type>(const polytensor_dispatching_engine&)> broadcast_walker_maker;
    std::function<indexer<ValT,CfgT>(const polytensor_dispatching_engine&)> trivial_walker_maker;
    std::function<walker<ValT,CfgT>(const polytensor_dispatching_engine&)> trivial_root_walker_maker;
    std::function<bool(const polytensor_dispatching_engine&)> is_trivial_;


    template<std::size_t I, typename...Ts>
    struct walker_maker_helper{
        //0 create broadcast walker helper
        template<typename...Ts>
        visitable_walker::walker<ValT,CfgT,visitor_type> helper(std::integral_constant<std::size_t, 0>, const polytensor_dispatching_engine& outer, const Ts&...operands)const{
            return [&](auto&&...walkers){
                //(outer.test_visitor(walkers,walkers...),...);
                //(outer.test_visitor(walkers,std::tie(walkers...)),...);
                using impl_type = evaluating_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>;
                return std::make_unique<visitable_walker::walker_polymorphic<ValT,CfgT,impl_type,visitor_type>>(impl_type{outer.host()->shape(),std::forward<decltype(walkers)>(walkers)...});
            }(operands->engine().create_walker()...);
        }
        //1 create trivial walker helper
        template<typename...Ts>
        indexer<ValT,CfgT> helper(std::integral_constant<std::size_t, 1>, const polytensor_dispatching_engine& outer, const Ts&...operands)const{
            return [&](auto&&...walkers){
                using impl_type = evaluating_trivial_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>;
                return std::make_unique<trivial_walker_polymorphic<ValT,CfgT,impl_type>>(impl_type{std::forward<decltype(walkers)>(walkers)...});
            }(operands->engine().create_trivial_walker()...);
        }
        //2 create trivial root walker helper
        template<typename...Ts>
        walker<ValT,CfgT> helper(std::integral_constant<std::size_t, 2>, const polytensor_dispatching_engine& outer, const Ts&...operands)const{
            return [&](auto&&...walkers){
                using impl_type = evaluating_trivial_root_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>;
                return std::make_unique<walker_polymorphic<ValT,CfgT,impl_type>>(impl_type{outer.host()->shape(), outer.host()->strides(), std::forward<decltype(walkers)>(walkers)...});
            }(operands->engine().create_trivial_walker()...);
        }
        //3 is trivial helper
        template<typename...Ts>
        auto helper(std::integral_constant<std::size_t, 3>, const polytensor_dispatching_engine& outer, const Ts&...operands)const{
                return gtensor::detail::is_trivial(outer.host()->size(),operands...);
        }
        auto operator()(const polytensor_dispatching_engine& outer)const{
            return std::apply(
                [&](const auto&...operands){
                    return helper(std::integral_constant<std::size_t, I>{}, outer, static_cast<typename Ts::element_type*>(operands.get())...);
                },
                outer.operands()
            );
        }
    };
public:
    using walker_type = WalkerT;
    template<typename...Ts>
    polytensor_dispatching_engine(host_type* host, F&& f, Ts&&...operands):
        evaluating_engine{host, std::forward<F>(f), std::forward<Ts>(operands)...},
        broadcast_walker_maker{walker_maker_helper<0, std::decay_t<Ts>...>{}},
        trivial_walker_maker{walker_maker_helper<1, std::decay_t<Ts>...>{}},
        trivial_root_walker_maker{walker_maker_helper<2, std::decay_t<Ts>...>{}},
        is_trivial_{walker_maker_helper<3, std::decay_t<Ts>...>{}}
    {}

    auto create_walker()const{
        if (is_trivial()){
            //return trivial_root_walker_maker(*this);
            return broadcast_walker_maker(*this);
        }else{
            return broadcast_walker_maker(*this);
        }
    }
    auto create_trivial_walker()const{
        return trivial_walker_maker(*this);
    }
    auto is_trivial()const{
        return is_trivial_(*this);
    }
};

template<typename ValT, typename CfgT, typename F, typename...ValTs>
class polytensor_engine : public evaluating_engine<ValT,CfgT,F,ValTs...>
{
    using typename engine_host_accessor::host_type;
    using operands_container_type = std::vector<std::shared_ptr<tensor_base_base<CfgT>>>;

    std::function<walker<ValT,CfgT>(const polytensor_engine&)> broadcast_walker_maker;
    std::function<indexer<ValT,CfgT>(const polytensor_engine&)> trivial_walker_maker;
    std::function<walker<ValT,CfgT>(const polytensor_engine&)> trivial_root_walker_maker;
    std::function<bool(const polytensor_engine&)> is_trivial_;

    template<std::size_t I, typename...Ts>
    struct walker_maker_helper{
        //0 create broadcast walker helper
        template<typename...Ts>
        walker<ValT,CfgT> helper(std::integral_constant<std::size_t, 0>, const polytensor_engine& outer, const Ts&...operands)const{
            return [&](auto&&...walkers){
                using impl_type = evaluating_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>;
                return std::make_unique<walker_polymorphic<ValT,CfgT,impl_type>>(impl_type{outer.host()->shape(),std::forward<decltype(walkers)>(walkers)...});
            }(operands->engine().create_walker()...);
        }
        //1 create trivial walker helper
        template<typename...Ts>
        indexer<ValT,CfgT> helper(std::integral_constant<std::size_t, 1>, const polytensor_engine& outer, const Ts&...operands)const{
            return [&](auto&&...walkers){
                using impl_type = evaluating_trivial_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>;
                return std::make_unique<trivial_walker_polymorphic<ValT,CfgT,impl_type>>(impl_type{std::forward<decltype(walkers)>(walkers)...});
            }(operands->engine().create_trivial_walker()...);
        }
        //2 create trivial root walker helper
        template<typename...Ts>
        walker<ValT,CfgT> helper(std::integral_constant<std::size_t, 2>, const polytensor_engine& outer, const Ts&...operands)const{
            return [&](auto&&...walkers){
                using impl_type = evaluating_trivial_root_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>;
                return std::make_unique<walker_polymorphic<ValT,CfgT,impl_type>>(impl_type{outer.host()->shape(), outer.host()->strides(), std::forward<decltype(walkers)>(walkers)...});
            }(operands->engine().create_trivial_walker()...);
        }
        //3 is trivial helper
        template<typename...Ts>
        auto helper(std::integral_constant<std::size_t, 3>, const polytensor_engine& outer, const Ts&...operands)const{
                return gtensor::detail::is_trivial(outer.host()->size(),operands...);
        }
        auto operator()(const polytensor_engine& outer)const{
            return std::apply(
                [&](const auto&...operands){
                    return helper(std::integral_constant<std::size_t, I>{}, outer, static_cast<typename Ts::element_type*>(operands.get())...);
                },
                outer.operands()
            );
        }
    };
public:
    template<typename...Ts>
    polytensor_engine(host_type* host, F&& f, Ts&&...operands):
        evaluating_engine{host, std::forward<F>(f), std::forward<Ts>(operands)...},
        broadcast_walker_maker{walker_maker_helper<0, std::decay_t<Ts>...>{}},
        trivial_walker_maker{walker_maker_helper<1, std::decay_t<Ts>...>{}},
        trivial_root_walker_maker{walker_maker_helper<2, std::decay_t<Ts>...>{}},
        is_trivial_{walker_maker_helper<3, std::decay_t<Ts>...>{}}
    {}
    auto create_walker()const{
        if (is_trivial()){
            //std::cout<<std::endl<<"trivial_root_walker_maker";
            return trivial_root_walker_maker(*this);
        }else{
            //std::cout<<std::endl<<"broadcast_walker_maker";
            return broadcast_walker_maker(*this);
        }
    }
    auto create_trivial_walker()const{
        //std::cout<<std::endl<<"trivial_walker_maker";
        return trivial_walker_maker(*this);
    }
    auto is_trivial()const{
        return is_trivial_(*this);
    }
};


template<typename...> struct storage_engine_traits;
template<typename ValT, typename CfgT> struct storage_engine_traits<ValT,CfgT>{
    using type = polytensor_storage_engine<ValT,CfgT>;
};
template<typename...> struct evaluating_engine_traits;
template<typename ValT, typename CfgT,  typename F, typename...Ops> struct evaluating_engine_traits<ValT,CfgT,F,Ops...>{
    using type = polytensor_dispatching_engine<ValT,CfgT,F,evaluating_walker<ValT,CfgT,F,typename Ops::engine_type::walker_type...>, typename Ops::value_type...>;
    //using type = polytensor_dispatching_engine<ValT,CfgT,F,typename Ops::value_type...>;
};

EXPRESSION_TEMPLATE_DISPATCHING_POLYTENSOR_BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add);
EXPRESSION_TEMPLATE_DISPATCHING_POLYTENSOR_BINARY_OPERATOR(operator+, operator_add_impl);

}   //end of namespace expression_template_dispatching_polytensor


#endif