#ifndef EXPRESSION_TEMPLATE_VARIANT_DISPATCH_HPP_
#define EXPRESSION_TEMPLATE_VARIANT_DISPATCH_HPP_

#include <variant>
#include <functional>
#include "gtensor.hpp"
#include "walker.hpp"
#include "evaluating_walker.hpp"
#include "storage_walker.hpp"
#include "iterator.hpp"
#include "engine.hpp"



#define EXPRESSION_TEMPLATE_VARIANT_DISPATCH_BINARY_OPERATOR(NAME,IMPL)\
template<typename DispatchDepth, typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
auto NAME(const test_tensor<DispatchDepth, ValT1, CfgT, ImplT1>& op1, const test_tensor<DispatchDepth, ValT2, CfgT, ImplT2>& op2){return IMPL(op1,op2);}

#define EXPRESSION_TEMPLATE_VARIANT_DISPATCH_BINARY_OPERATOR_IMPL(NAME,OP)\
template<typename DispatchDepth, typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
static inline auto NAME(const test_tensor<DispatchDepth, ValT1, CfgT, ImplT1>& op1, const test_tensor<DispatchDepth, ValT2, CfgT, ImplT2>& op2){\
    using operation_type = OP;\
    using result_type = decltype(std::declval<operation_type>()(std::declval<ValT1>(),std::declval<ValT2>()));\
    using operand1_type = ImplT1;\
    using operand2_type = ImplT2;\
    using engine_type = evaluating_engine_traits<DispatchDepth, result_type, CfgT, operation_type, operand1_type, operand2_type>::type;\
    using impl_type = evaluating_tensor<result_type, CfgT, engine_type>;\
    return test_tensor<DispatchDepth, result_type,CfgT, impl_type>{std::make_shared<impl_type>(operation_type{}, op1.impl(),op2.impl())};\
}

namespace expression_template_variant_dispatch{
using gtensor::tensor;
using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::storage_engine;
using gtensor::evaluating_engine;
using gtensor::evaluating_walker;
using gtensor::walker_polymorphic;
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

template<typename DispatchDepth, typename ValT, typename CfgT, typename ImplT = storage_tensor<ValT,CfgT, typename storage_engine_traits<ValT,CfgT>::type >>
class test_tensor : public tensor<ValT,CfgT, ImplT>{

    struct polymorphic_walker_maker{
        auto&& operator()(walker<ValT,CfgT>&& w){
            return std::move(w);
        }
        template<typename...Ts>
        auto operator()(evaluating_walker<ValT,CfgT,Ts...>&& w){
            using impl_type = evaluating_walker<ValT,CfgT,Ts...>;
            return walker<ValT,CfgT>{std::make_unique<walker_polymorphic<ValT,CfgT,impl_type>>(std::move(w))};
        }
        template<typename...Ts>
        auto operator()(evaluating_trivial_root_walker<ValT,CfgT,Ts...>&& w){
            using impl_type = evaluating_trivial_root_walker<ValT,CfgT,Ts...>;
            return walker<ValT,CfgT>{std::make_unique<walker_polymorphic<ValT,CfgT,impl_type>>(std::move(w))};
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
        return iterator_type{std::forward<W>(w), impl()->shape(), impl()->descriptor().as_descriptor_with_libdivide()->strides_libdivide()};
    }
    template<typename W>
    auto end(W&& w)const{
        using iterator_type = multiindex_iterator<ValT,CfgT,std::decay_t<W>>;
        return iterator_type{std::forward<W>(w), impl()->shape(), impl()->descriptor().as_descriptor_with_libdivide()->strides_libdivide(), impl()->size()};
    }
};

template<typename> struct make_variant_type;
template<typename...Ts> struct make_variant_type<type_list<Ts...>>{
    using type = std::variant<Ts...>;
};

template<typename ValT, typename CfgT>
class variant_dispatch_storage_engine : public storage_engine<ValT,CfgT>
{
public:
    using walker_types = type_list<storage_walker<ValT,CfgT>>;
    using trivial_walker_type = storage_trivial_walker<ValT,CfgT>;
    using variant_type = typename make_variant_type<walker_types>::type;

    using storage_engine::storage_engine;
    bool is_trivial()const{return true;}
    auto create_walker()const{return variant_type{storage_walker<ValT,CfgT>{host()->shape(),host()->strides(),data()}};}
    auto create_trivial_walker()const{return storage_trivial_walker<ValT,CfgT>{data()};}
};

template<typename DispatchDepth, typename ValT, typename CfgT, typename F, typename...Ops>
class variant_dispatch_engine : public evaluating_engine<ValT,CfgT,F,Ops...>
{
private:
    static constexpr std::size_t max_walker_types_size = DispatchDepth::value;
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

    using evaluating_engine::evaluating_engine;
    bool is_trivial()const{
        return std::apply(
            [this](const auto&...operands){
                return gtensor::detail::is_trivial(host()->size(),static_cast<Ops*>(operands.get())...);
            },
            operands()
        );
    }
    template<typename...Wks>
    auto create_broadcast_walker_helper(std::true_type, Wks&&...walkers)const{
        return variant_type{evaluating_walker<ValT,CfgT,F,std::decay_t<Wks>...>{host()->shape(),std::forward<Wks>(walkers)...}};
    }
    template<typename...Wks>
    auto create_broadcast_walker_helper(std::false_type, Wks&&...walkers)const{
        using impl_type = evaluating_walker<ValT,CfgT,F,std::decay_t<Wks>...>;
        using walker_polymorphic_type = walker_polymorphic<ValT,CfgT,impl_type>;
        return variant_type{
            walker<ValT,CfgT>{
                std::make_unique<walker_polymorphic_type>(impl_type{host()->shape(),std::forward<Wks>(walkers)...})
            }
        };
    }

    auto create_broadcast_walker()const{
        return std::apply(
            [this](const auto&...operands){
                return std::visit(
                    [this](auto&&...walkers){
                        return create_broadcast_walker_helper(std::integral_constant<bool,(walker_types_size<max_walker_types_size)>{}, std::forward<decltype(walkers)>(walkers)...);
                    },
                    static_cast<Ops*>(operands.get())->engine().create_walker()...
                );
            },
            operands()
        );
    }
    auto create_trivial_walker()const{
        return std::apply(
            [this](const auto&...operands){
                return [this](auto&&...walkers){
                    return evaluating_trivial_root_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>{host()->shape(),host()->strides(),std::forward<decltype(walkers)>(walkers)...};
                }(static_cast<Ops*>(operands.get())->engine().create_trivial_walker()...);
            },
            operands()
        );
    }

    auto create_walker()const{
        if (is_trivial()){
            return variant_type{create_trivial_walker()};
        }else{
            return create_broadcast_walker();
        }
    }

};

template<typename...> struct storage_engine_traits;
template<typename ValT, typename CfgT> struct storage_engine_traits<ValT,CfgT>{
    using type = variant_dispatch_storage_engine<ValT,CfgT>;
};
template<typename...> struct evaluating_engine_traits;
template<typename DispatchDepth, typename ValT, typename CfgT,  typename F, typename...Ops> struct evaluating_engine_traits<DispatchDepth, ValT,CfgT,F,Ops...>{
    using type = variant_dispatch_engine<DispatchDepth, ValT,CfgT,F,Ops...>;
};

EXPRESSION_TEMPLATE_VARIANT_DISPATCH_BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add);
EXPRESSION_TEMPLATE_VARIANT_DISPATCH_BINARY_OPERATOR_IMPL(operator_mul_impl, gtensor::binary_operations::mul);
EXPRESSION_TEMPLATE_VARIANT_DISPATCH_BINARY_OPERATOR(operator+, operator_add_impl);
EXPRESSION_TEMPLATE_VARIANT_DISPATCH_BINARY_OPERATOR(operator*, operator_mul_impl);

}   //end of namespace expression_template_variant_dispatch

#endif