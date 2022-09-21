#ifndef EXPRESSION_TEMPLATE_POLYTENSOR_HPP_
#define EXPRESSION_TEMPLATE_POLYTENSOR_HPP_

#include <variant>
#include <functional>
#include "gtensor.hpp"
#include "walker.hpp"
#include "evaluating_walker.hpp"
#include "storage_walker.hpp"
#include "iterator.hpp"
#include "engine.hpp"



#define EXPRESSION_TEMPLATE_POLYTENSOR_BINARY_OPERATOR(NAME,IMPL)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
auto NAME(const test_tensor<ValT1, CfgT, ImplT1>& op1, const test_tensor<ValT2, CfgT, ImplT2>& op2){return IMPL(op1,op2);}

#define EXPRESSION_TEMPLATE_POLYTENSOR_BINARY_OPERATOR_IMPL(NAME,OP)\
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

namespace expression_template_polytensor{
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
    bool is_trivial()const{return true;}
    auto create_walker()const{
        return storage_walker<ValT,CfgT>{host()->shape(),host()->strides(),data()};
    }
    auto create_trivial_walker()const{
        return storage_trivial_walker<ValT,CfgT>{data()};
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
    using type = polytensor_engine<ValT,CfgT,F,typename Ops::value_type...>;
};

EXPRESSION_TEMPLATE_POLYTENSOR_BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add);
EXPRESSION_TEMPLATE_POLYTENSOR_BINARY_OPERATOR(operator+, operator_add_impl);

}   //end of namespace expression_template_polytensor


#endif