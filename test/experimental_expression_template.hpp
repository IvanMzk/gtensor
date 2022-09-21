#ifndef EXPERIMENTAL_EXPRESSION_TEMPLATE_HPP_
#define EXPERIMENTAL_EXPRESSION_TEMPLATE_HPP_

#include <variant>
#include <functional>
#include "gtensor.hpp"
#include "walker.hpp"
#include "evaluating_walker.hpp"
#include "storage_walker.hpp"
#include "iterator.hpp"
#include "engine.hpp"



#define EXPERIMENTAL_EXPRESSION_TEMPLATE_BINARY_OPERATOR(NAME,IMPL)\
template<typename ValT1, typename ValT2, typename ImplT1, typename ImplT2, typename CfgT>\
auto NAME(const test_tensor<ValT1, CfgT, ImplT1>& op1, const test_tensor<ValT2, CfgT, ImplT2>& op2){return IMPL(op1,op2);}

#define EXPERIMENTAL_EXPRESSION_TEMPLATE_BINARY_OPERATOR_IMPL(NAME,OP)\
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

namespace expression_template_polywalker{
using gtensor::tensor;
using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::storage_engine;
using gtensor::evaluating_engine;
using gtensor::storage_walker;
using gtensor::walker_polymorphic;
using gtensor::walker;
using gtensor::storage_trivial_walker;
using gtensor::evaluating_walker;
using gtensor::evaluating_trivial_root_walker;
using gtensor::evaluating_trivial_walker;
using gtensor::multiindex_iterator;

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
class polywalker_storage_engine : public storage_engine<ValT,CfgT>
{
public:
    using storage_engine::storage_engine;
    bool is_trivial()const{return true;}
    auto create_walker()const{
        return storage_walker<ValT,CfgT>{host()->shape(),host()->strides(),host()->reset_strides(),data()};
    }
    auto create_trivial_walker()const{
        return storage_trivial_walker<ValT,CfgT>{data()};
    }
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class polywalker_engine : public evaluating_engine<ValT,CfgT,F,Ops...>
{
public:
    using evaluating_engine::evaluating_engine;
    bool is_trivial()const{
        return std::apply(
            [this](const auto&...operands){
                return gtensor::detail::is_trivial(host()->size(),static_cast<Ops*>(operands.get())...);
            },
            operands()
        );
    }
    template<typename U>
    auto create_walker_helper(U&& walker_)const{
        return walker<ValT,CfgT>{
            std::make_unique<walker_polymorphic<ValT,CfgT,std::decay_t<U>>>(std::forward<U>(walker_))
        };
    }
    auto create_walker()const{
        if (is_trivial()){
            return create_walker_helper(create_trivial_root_walker());
        }else{
            return create_walker_helper(create_broadcast_walker());
        }
    }
    auto create_broadcast_walker()const{
        return std::apply(
            [this](const auto&...operands){
                return [this](auto&&...walkers){
                    return evaluating_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>{host()->shape(),std::forward<decltype(walkers)>(walkers)...};
                }(static_cast<Ops*>(operands.get())->engine().create_walker()...);
            },
            operands()
        );
    }
    auto create_trivial_root_walker()const{
        return std::apply(
            [this](const auto&...operands){
                return [this](auto&&...walkers){
                    return evaluating_trivial_root_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>{host()->shape(),host()->strides(),host()->reset_strides(),std::forward<decltype(walkers)>(walkers)...};
                }(static_cast<Ops*>(operands.get())->engine().create_trivial_walker()...);
            },
            operands()
        );
    }
    auto create_trivial_walker()const{
        return std::apply(
            [this](const auto&...operands){
                return [](auto&&...walkers){
                    return evaluating_trivial_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>{std::forward<decltype(walkers)>(walkers)...};
                }(static_cast<Ops*>(operands.get())->engine().create_trivial_walker()...);
            },
            operands()
        );
    }
};

template<typename...> struct storage_engine_traits;
template<typename ValT, typename CfgT> struct storage_engine_traits<ValT,CfgT>{
    using type = polywalker_storage_engine<ValT,CfgT>;
};
template<typename...> struct evaluating_engine_traits;
template<typename ValT, typename CfgT,  typename F, typename...Ops> struct evaluating_engine_traits<ValT,CfgT,F,Ops...>{
    using type = polywalker_engine<ValT,CfgT,F,Ops...>;
};

EXPERIMENTAL_EXPRESSION_TEMPLATE_BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add);
EXPERIMENTAL_EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator+, operator_add_impl);

}   //end of namespace expression_template_polywalker

namespace expression_template_without_dispatching{
using gtensor::tensor;
using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::storage_engine;
using gtensor::evaluating_engine;
using gtensor::walker;
using gtensor::walker_polymorphic;
using gtensor::storage_walker;
using gtensor::evaluating_walker;
using gtensor::evaluating_trivial_walker;
using gtensor::evaluating_trivial_root_walker;
using gtensor::storage_trivial_walker;
using gtensor::multiindex_iterator;

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
class no_dispatching_storage_engine : public storage_engine<ValT,CfgT>
{
public:
    using storage_engine::storage_engine;
    bool is_trivial()const{return true;}
    auto create_walker()const{return create_broadcast_walker();}
    auto create_broadcast_walker()const{return storage_walker<ValT,CfgT>{host()->shape(),host()->strides(),host()->reset_strides(),data()};}
    auto create_trivial_walker()const{return storage_trivial_walker<ValT,CfgT>{data()};}
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class no_dispatching_engine_root_dispatch : public evaluating_engine<ValT,CfgT,F,Ops...>
{
public:
    using evaluating_engine::evaluating_engine;
    bool is_trivial()const{
        return std::apply(
            [this](const auto&...operands){
                return gtensor::detail::is_trivial(host()->size(),static_cast<Ops*>(operands.get())...);
            },
            operands()
        );
    }
    //make result polywalker
    template<typename U>
    auto create_walker_helper(U&& walker_)const{
        return walker<ValT,CfgT>{
            std::make_unique<walker_polymorphic<ValT,CfgT,std::decay_t<U>>>(std::forward<U>(walker_))
        };
    }
    auto create_walker()const{
        if (is_trivial()){
            return create_walker_helper(create_trivial_root_walker());
        }else{
            return create_walker_helper(create_broadcast_walker());
        }
    }
    //make broadcast walker
    template<typename...Wks>
    auto create_broadcast_walker_helper(Wks&&...walkers)const{
        return evaluating_walker<ValT,CfgT,F,Wks...>{host()->shape(),std::forward<Wks>(walkers)...};
    }
    template<std::size_t...I>
    auto create_broadcast_walker_helper(std::index_sequence<I...>)const{
        return create_broadcast_walker_helper(static_cast<Ops*>(std::get<I>(operands()).get())->engine().create_broadcast_walker()...);
    }
    auto create_broadcast_walker()const{
        return create_broadcast_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
    }
    //make trivial root walker
    template<typename...Wks>
    auto create_trivial_root_walker_helper(Wks&&...walkers)const{
        return evaluating_trivial_root_walker<ValT,CfgT,F,Wks...>{host()->shape(),host()->strides(),std::forward<Wks>(walkers)...};
    }
    template<std::size_t...I>
    auto create_trivial_root_walker_helper(std::index_sequence<I...>)const{
        return create_trivial_root_walker_helper(static_cast<Ops*>(std::get<I>(operands()).get())->engine().create_trivial_walker()...);
    }
    auto create_trivial_root_walker()const{
        return create_trivial_root_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
    }
    //make trivial walker
    template<typename...Wks>
    auto create_trivial_walker_helper(Wks&&...walkers)const{
        return evaluating_trivial_walker<ValT,CfgT,F,Wks...>{std::forward<Wks>(walkers)...};
    }
    template<std::size_t...I>
    auto create_trivial_walker_helper(std::index_sequence<I...>)const{
        return create_trivial_walker_helper(static_cast<Ops*>(std::get<I>(operands()).get())->engine().create_trivial_walker()...);
    }
    auto create_trivial_walker()const{
        return create_trivial_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
    }
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class no_dispatching_engine : public evaluating_engine<ValT,CfgT,F,Ops...>
{
public:
    using evaluating_engine::evaluating_engine;
    //bool is_trivial()const{return gtensor::detail::is_trivial(host()->size(),operands());}
    bool is_trivial()const{
        return std::apply(
            [this](const auto&...operands){
                return gtensor::detail::is_trivial(host()->size(),static_cast<Ops*>(operands.get())...);
            },
            operands()
        );
    }
    template<typename...Wks>
    auto create_walker_helper(Wks&&...walkers)const{
        return evaluating_walker<ValT,CfgT,F,Wks...>{host()->shape(),std::forward<Wks>(walkers)...};
    }
    template<std::size_t...I>
    auto create_walker_helper(std::index_sequence<I...>)const{
        return create_walker_helper(static_cast<Ops*>(std::get<I>(operands()).get())->engine().create_walker()...);
    }
    auto create_walker()const{
        return create_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});

    }
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class no_dispatching_engine_with_apply : public evaluating_engine<ValT,CfgT,F,Ops...>
{
public:
    using evaluating_engine::evaluating_engine;
    bool is_trivial()const{return gtensor::detail::is_trivial(host()->size(),operands());}
    template<std::size_t...I>
    auto create_walker()const{
        return std::apply(
            [this](const auto&...operands){
                return [this](auto&&...walkers){
                    return evaluating_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>{host()->shape(),std::forward<decltype(walkers)>(walkers)...};
                }(static_cast<Ops*>(operands.get())->engine().create_walker()...);
            },
            operands()
        );
    }
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class no_dispatching_engine_without_apply : public evaluating_engine<ValT,CfgT,F,Ops...>
{
public:
    using evaluating_engine::evaluating_engine;
    bool is_trivial()const{return gtensor::detail::is_trivial(host()->size(),operands());}
    template<std::size_t...I>
    auto create_walker_helper(std::index_sequence<I...>)const{
        return [this](auto&&...walkers){
            return evaluating_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>{host()->shape(),std::forward<decltype(walkers)>(walkers)...};
        }(static_cast<Ops*>(std::get<I>(operands()).get())->engine().create_walker()...);
    }
    template<std::size_t...I>
    auto create_walker()const{
        return create_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
    }
};

template<typename...> struct storage_engine_traits;
template<typename ValT, typename CfgT> struct storage_engine_traits<ValT,CfgT>{
    using type = no_dispatching_storage_engine<ValT,CfgT>;
};
template<typename...> struct evaluating_engine_traits;
template<typename ValT, typename CfgT,  typename F, typename...Ops> struct evaluating_engine_traits<ValT,CfgT,F,Ops...>{
    //using type = no_dispatching_engine_without_apply<ValT,CfgT,F,Ops...>;
    //using type = no_dispatching_engine_with_apply<ValT,CfgT,F,Ops...>;
    //using type = no_dispatching_engine_root_dispatch<ValT,CfgT,F,Ops...>;
    using type = no_dispatching_engine<ValT,CfgT,F,Ops...>;
};

EXPERIMENTAL_EXPRESSION_TEMPLATE_BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add);
EXPERIMENTAL_EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator+, operator_add_impl);

}   //end of namespace expression_template_without_dispatching

namespace expression_template_dispatch_in_walker{
using gtensor::tensor;
using gtensor::storage_tensor;
using gtensor::evaluating_tensor;
using gtensor::multiindex_iterator;
using gtensor::storage_engine;
using gtensor::evaluating_engine;
using gtensor::basic_walker;
using gtensor::storage_walker;

template<typename ValT, typename CfgT, typename F, typename...Wks>
class evaluating_dispatching_walker : private basic_walker<ValT, CfgT, typename CfgT::index_type>
{
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    std::pair<Wks...> walkers;
    //std::tuple<Wks...> walkers;
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

    evaluating_dispatching_walker(const shape_type& shape_, const shape_type& strides_, const shape_type& reset_strides_, bool is_trivial_, Wks&&...walkers_):
        basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, reset_strides_, index_type{0}},
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
class dispatching_in_walker_storage_engine : public storage_engine<ValT,CfgT>
{
public:
    using storage_engine::storage_engine;
    bool is_trivial()const{return true;}
    auto create_walker()const{return storage_walker<ValT,CfgT>{host()->shape(),host()->strides(),host()->reset_strides(),data()};}
    auto create_trivial_walker()const{return create_walker();}
};

template<typename ValT, typename CfgT, typename F, typename...Ops>
class dispatching_in_walker_engine : public evaluating_engine<ValT,CfgT,F,Ops...>
{
public:
    using evaluating_engine::evaluating_engine;
    bool is_trivial()const{
        return std::apply(
            [this](const auto&...operands){
                return gtensor::detail::is_trivial(host()->size(),static_cast<Ops*>(operands.get())...);
            },
            operands()
        );
    }
    auto create_walker()const{
        if (is_trivial()){
            return create_trivial_walker();
        }else{
            return create_broadcast_walker();
        }
    }
    auto create_broadcast_walker()const{
        return std::apply(
            [this](const auto&...args){
                return [this](auto&&...walkers){
                    return evaluating_dispatching_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>{host()->shape(),host()->strides(), host()->reset_strides(), false ,std::forward<decltype(walkers)>(walkers)...};
                }(static_cast<Ops*>(args.get())->engine().create_walker()...);
            },
            operands()
        );
    }
    auto create_trivial_walker()const{
        return std::apply(
            [this](const auto&...args){
                return [this](auto&&...walkers){
                    return evaluating_dispatching_walker<ValT,CfgT,F,std::decay_t<decltype(walkers)>...>{host()->shape(),host()->strides(), host()->reset_strides(), true ,std::forward<decltype(walkers)>(walkers)...};
                }(static_cast<Ops*>(args.get())->engine().create_trivial_walker()...);
            },
            operands()
        );
    }
};

template<typename...> struct storage_engine_traits;
template<typename ValT, typename CfgT> struct storage_engine_traits<ValT,CfgT>{
    using type = dispatching_in_walker_storage_engine<ValT,CfgT>;
};
template<typename...> struct evaluating_engine_traits;
template<typename ValT, typename CfgT,  typename F, typename...Ops> struct evaluating_engine_traits<ValT,CfgT,F,Ops...>{
    using type = dispatching_in_walker_engine<ValT,CfgT,F,Ops...>;
};

EXPERIMENTAL_EXPRESSION_TEMPLATE_BINARY_OPERATOR_IMPL(operator_add_impl, gtensor::binary_operations::add);
EXPERIMENTAL_EXPRESSION_TEMPLATE_BINARY_OPERATOR(operator+, operator_add_impl);

}   //end of namespace expression_template_dispatch_in_walker

#endif