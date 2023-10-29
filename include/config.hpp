/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <vector>
#include "storage.hpp"

namespace gtensor{
namespace config{

enum class div_modes : std::size_t {native, libdivide};
enum class engines : std::size_t {expression_template};
enum class orders : std::size_t {c,f};
enum class cloning_semantics : std::size_t {deep,shallow};

using mode_div_native = std::integral_constant<div_modes, div_modes::native>;
using mode_div_libdivide = std::integral_constant<div_modes, div_modes::libdivide>;
using engine_expression_template = std::integral_constant<engines, engines::expression_template>;
using c_order = std::integral_constant<orders, orders::c>;
using f_order = std::integral_constant<orders, orders::f>;
using deep_semantics = std::integral_constant<cloning_semantics, cloning_semantics::deep>;
using shallow_semantics = std::integral_constant<cloning_semantics, cloning_semantics::shallow>;

struct default_config
{
    using engine = engine_expression_template;

    //specify whether to use optimized division
    using div_mode = mode_div_libdivide;
    //using div_mode = mode_div_native;

    //specify default traverse order of iterators
    using order = c_order;
    //using order = f_order;

    //cloning semantics - determines effect of tensor copy construction
    using semantics = deep_semantics;
    //using semantics = shallow_semantics;

    //data elements storage template
    template<typename T> using storage = gtensor::basic_storage<T>;

    //meta-data elements storage template i.e. shape, strides are specialization of shape
    //must provide std::vector like interface
    template<typename T> using shape = gtensor::stack_prealloc_vector<T,8>;

    //generally when public interface expected container parameter it may be any type providig usual container semantic and interface: iterators, aliases...
    //specialization of config_type::container uses as return type in public interface
    //it may be used by implementation as general purpose container
    //must provide std::vector like interface
    template<typename T> using container = std::vector<T>;

    //index_map specialization is used in mapping_descriptor that is descriptor type of mapping_view
    //it is natural to use storage as index_map in general, but if storage is specific e.g. map to file system or network, these should differ
    template<typename T> using index_map = storage<T>;
};

template<typename Config, typename IdxT>
struct extended_config : public Config
{
    //index_type defines data elements address space:
    //e.g. shape and strides elements are of index_type
    //slice, reshape view subscripts are of index_type
    //must have semantic of signed integral type
    using index_type = IdxT;

    //meta-data container type
    using shape_type = typename Config::template shape<index_type>;

    //dim_type used in indexed access to meta-data elements:
    //e.g. index of axis, dimensions number
    //transpose view subscripts are of dim_type, since they are axes indexes
    //must have semantic of signed integral type
    using dim_type = typename shape_type::difference_type;
};

template<typename Config, typename T>
struct extend_config
{
    static_assert(!std::is_void_v<T>);
    using type = extended_config<Config, typename Config::template storage<T>::difference_type>;
};

template<typename Config, typename IdxT, typename T>
struct extend_config<extended_config<Config,IdxT>,T>
{
    template<typename, typename> struct selector_;
    template<typename Dummy> struct selector_<std::true_type,Dummy>{
        using type = extended_config<Config,IdxT>;
    };
    template<typename Dummy> struct selector_<std::false_type,Dummy>{
        using type = extended_config<Config, typename Config::template storage<T>::difference_type>;
    };
    using type = typename selector_<typename std::is_void<T>::type,void>::type;
};

template<typename Config, typename T> using extend_config_t = typename extend_config<Config,T>::type;

template<typename T> constexpr bool is_extended_config_v = false;
template<typename...Ts> constexpr bool is_extended_config_v<extended_config<Ts...>> = true;

}   //end of namespace config
}   //end of namespace gtensor

#endif