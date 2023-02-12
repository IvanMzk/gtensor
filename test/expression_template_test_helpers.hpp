#ifndef EXPRESSION_TEMPLATE_TEST_HELPERS_HPP_
#define EXPRESSION_TEMPLATE_TEST_HELPERS_HPP_

#include <tuple>
#include "gtensor.hpp"
#include "test_config.hpp"

namespace test_expression_template_helpers{

using value_type = float;
using gtensor::tensor;
using test_config_div_native_type = typename test_config::config_host_engine_div_selector<gtensor::config::engine_expression_template,gtensor::config::mode_div_native>::config_type;
using test_config_div_libdivide_type = typename test_config::config_host_engine_div_selector<gtensor::config::engine_expression_template,gtensor::config::mode_div_libdivide>::config_type;
using test_default_config_type = test_config_div_native_type;
//using test_default_config_type = test_config_div_libdivide_type;
using gtensor::expression_template_storage_engine;
using gtensor::broadcast_iterator;
using gtensor::trivial_broadcast_iterator;
using gtensor::detail::begin_broadcast;
using gtensor::detail::end_broadcast;
using gtensor::detail::begin_trivial;
using gtensor::detail::end_trivial;
static test_default_config_type::nop_type nop{};

template<typename T>
struct test_tensor : public T
{
    test_tensor(const T& base):
        T{base}
    {}
    using T::engine;
    bool is_trivial()const{return engine().is_trivial();}
    auto create_broadcast_walker()const{return engine().create_broadcast_walker();}
    auto create_trivial_walker()const{return engine().create_trivial_walker();}
    auto create_indexer()const{return engine().create_indexer();}
    auto create_indexer(){return engine().create_indexer();}
};

template<typename T>
struct test_broadcast_iterator_tensor : public T
{
    test_broadcast_iterator_tensor(const T& base):
        T{base}
    {}
    auto& engine()const{return impl()->engine();}
    auto begin()const{return begin_broadcast(engine());}
    auto end()const{return end_broadcast(engine());}
};

template<typename T>
struct test_trivial_iterator_tensor : public T
{
    test_trivial_iterator_tensor(const T& base):
        T{base}
    {}
    auto& engine()const{return impl()->engine();}
    auto begin()const{return begin_trivial(engine());}
    auto end()const{return end_trivial(engine());}
};


template<template<typename> typename TestT = test_tensor, typename T>
auto make_test_tensor(T&& t){return TestT<std::decay_t<T>>{std::forward<T>(t)};}

template<typename TensorT>
struct storage_tensor_maker{
    auto operator()(){
        return TensorT{{{1,2,3},{4,5,6}}};
    }
};

template<typename TensorT>
struct notrivial_tensor_maker{
    auto operator()(){
        return TensorT{{{0},{3}}} + TensorT{1,2,3};
    }
};

template<typename TensorT>
struct trivial_subtree_tensor_maker{
    auto operator()(){
        return TensorT{2} * TensorT{-1,-1,-1} + (TensorT{{{1,2,3},{1,2,3}}} + TensorT{{{0,0,0},{3,3,3}}}) + TensorT{5,5,5} - TensorT{3};
    }
};

template<typename TensorT>
struct trivial_tensor_maker{
    auto operator()(){
        return TensorT{{{-1,-1,-1},{-1,-1,-1}}} + TensorT{{{1,2,3},{1,2,3}}} + TensorT{{{1,1,1},{4,4,4}}};
    }
};

template<typename TensorT>
struct view_slice_of_storage_maker{
    auto operator()(){
        return TensorT{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{1,0,2,0,3,0},{4,0,5,0,6,0}}}({{1,2},{},{nop,nop,2}});
    }
};

template<typename TensorT>
struct view_slice_of_eval_maker{
    auto operator()(){
        return TensorT{2} * TensorT{{1,1,1,1,1,1}} + TensorT{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{1,0,2,0,3,0},{4,0,5,0,6,0}}} - TensorT{{{3,3,3,3,3,3}}} + TensorT{1})({{1,2},{},{nop,nop,2}});
    }
};

template<typename TensorT>
struct view_view_slice_of_eval_maker{
    auto operator()(){
        return (TensorT{2} * TensorT{{1,1,1,1,1,1}} + TensorT{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{0,3,0,2,0,1},{0,6,0,5,0,4}}} - TensorT{{{3,3,3,3,3,3}}} + TensorT{1})({{},{},{nop,nop,-1}})({{1,2},{},{nop,nop,2}});
    }
};

template<typename TensorT>
struct view_transpose_of_storage_maker{
    auto operator()(){
        return TensorT{{{1},{4}},{{2},{5}},{{3},{6}}}.transpose();
    }
};

template<typename TensorT>
struct view_subdim_of_storage_maker{
    auto operator()(){
        return TensorT{{{{0,0,0},{0,0,0}}},{{{1,2,3},{4,5,6}}}}(1);
    }
};

template<typename TensorT>
struct view_reshape_of_storage_maker{
    auto operator()(){
        return TensorT{{{1},{2},{3}},{{4},{5},{6}}}.reshape(1,2,3);
    }
};

template<typename TensorT>
struct eval_view_operand_maker{
    auto operator()(){
        return view_transpose_of_storage_maker<TensorT>{}() +
            storage_tensor_maker<TensorT>{}() -
            trivial_subtree_tensor_maker<TensorT>{}() +
            view_view_slice_of_eval_maker<TensorT>{}() -
            trivial_tensor_maker<TensorT>{}()
    }
};

template<typename TensorT>
struct trivial_view_operand_maker{
    auto operator()(){
        return storage_tensor_maker<TensorT>{}() +
            view_transpose_of_storage_maker<TensorT>{}()) +
            (storage_tensor_maker<TensorT>{}() -
            view_view_slice_of_eval_maker<TensorT>{}() -
            trivial_tensor_maker<TensorT>{}());
    }
};

template<typename TensorT>
struct view_eval_view_operand_maker{
    auto operator()(){
        return (view_transpose_of_storage_maker<TensorT>{}() +
                storage_tensor_maker<TensorT>{}() -
                trivial_subtree_tensor_maker<TensorT>{}() +
                view_view_slice_of_eval_maker<TensorT>{}() -
                trivial_tensor_maker<TensorT>{}() +
                TensorT{{{0,0,0},{0,0,0}},{{2,0,-2},{2,0,-2}}}
                )({{1},{},{nop,nop,-1}});
    }
};


template<template<typename> typename TestT>
struct makers_type_list{
    using type = std::tuple<
        storage_tensor_maker<TestT, test_config_div_native_type>,
        notrivial_tensor_maker<TestT, test_config_div_native_type>,
        trivial_subtree_tensor_maker<TestT, test_config_div_native_type>,
        trivial_tensor_maker<TestT, test_config_div_native_type>,
        view_slice_of_storage_maker<TestT, test_config_div_native_type>,
        view_slice_of_eval_maker<TestT, test_config_div_native_type>,
        view_view_slice_of_eval_maker<TestT, test_config_div_native_type>,
        view_transpose_of_storage_maker<TestT, test_config_div_native_type>,
        view_subdim_of_storage_maker<TestT, test_config_div_native_type>,
        view_reshape_of_storage_maker<TestT, test_config_div_native_type>,
        eval_view_operand_maker<TestT, test_config_div_native_type>,
        trivial_view_operand_maker<TestT, test_config_div_native_type>,
        view_eval_view_operand_maker<TestT, test_config_div_native_type>,

        storage_tensor_maker<TestT, test_config_div_libdivide_type>,
        notrivial_tensor_maker<TestT, test_config_div_libdivide_type>,
        trivial_subtree_tensor_maker<TestT, test_config_div_libdivide_type>,
        trivial_tensor_maker<TestT, test_config_div_libdivide_type>,
        view_slice_of_storage_maker<TestT, test_config_div_libdivide_type>,
        view_slice_of_eval_maker<TestT, test_config_div_libdivide_type>,
        view_view_slice_of_eval_maker<TestT, test_config_div_libdivide_type>,
        view_transpose_of_storage_maker<TestT, test_config_div_libdivide_type>,
        view_subdim_of_storage_maker<TestT, test_config_div_libdivide_type>,
        view_reshape_of_storage_maker<TestT, test_config_div_libdivide_type>,
        eval_view_operand_maker<TestT, test_config_div_libdivide_type>,
        trivial_view_operand_maker<TestT, test_config_div_libdivide_type>,
        view_eval_view_operand_maker<TestT, test_config_div_libdivide_type>
    >;
};

template<template<typename> typename TestT>
struct makers_trivial_type_list
{
    using type = std::tuple<
        storage_tensor_maker<TestT, test_config_div_native_type>,
        trivial_tensor_maker<TestT, test_config_div_native_type>,
        view_slice_of_storage_maker<TestT, test_config_div_native_type>,
        view_slice_of_eval_maker<TestT, test_config_div_native_type>,
        view_view_slice_of_eval_maker<TestT, test_config_div_native_type>,
        view_transpose_of_storage_maker<TestT, test_config_div_native_type>,
        view_subdim_of_storage_maker<TestT, test_config_div_native_type>,
        view_reshape_of_storage_maker<TestT, test_config_div_native_type>,
        trivial_view_operand_maker<TestT, test_config_div_native_type>,
        view_eval_view_operand_maker<TestT, test_config_div_native_type>,

        storage_tensor_maker<TestT, test_config_div_libdivide_type>,
        trivial_tensor_maker<TestT, test_config_div_libdivide_type>,
        view_slice_of_storage_maker<TestT, test_config_div_libdivide_type>,
        view_slice_of_eval_maker<TestT, test_config_div_libdivide_type>,
        view_view_slice_of_eval_maker<TestT, test_config_div_libdivide_type>,
        view_transpose_of_storage_maker<TestT, test_config_div_libdivide_type>,
        view_subdim_of_storage_maker<TestT, test_config_div_libdivide_type>,
        view_reshape_of_storage_maker<TestT, test_config_div_libdivide_type>,
        trivial_view_operand_maker<TestT, test_config_div_libdivide_type>,
        view_eval_view_operand_maker<TestT, test_config_div_libdivide_type>
    >;
};





}   //end of namespace test_expression_template_helpers

#endif