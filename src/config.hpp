#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <vector>
#include "uvector.h"
#include "forward_decl.hpp"

namespace gtensor{

namespace config{



enum class div_modes : std::size_t {native, libdivide};
enum class engines : std::size_t {expression_template,cuda};

template<typename T, T M> struct tag{static constexpr T value = M;};

using mode_div_native = tag<div_modes, div_modes::native>;
using mode_div_libdivide = tag<div_modes, div_modes::libdivide>;

using engine_expression_template = tag<engines, engines::expression_template>;
using engine_cuda = tag<engines, engines::cuda>;

struct NOP{};


struct default_config{

    //using engine = engine_expression_template;
    using host_engine = engine_expression_template;
    using device_engine = engine_cuda;

    using div_mode = mode_div_libdivide;
    //using div_mode = mode_div_native;

    using difference_type = std::int64_t;
    using index_type = difference_type;
    //template<typename ValT> using storage = gtensor::detail::shareable_storage<std::vector<ValT>>;
    template<typename ValT> using storage = std::vector<ValT>;
    //template<typename ValT> using storage = trivial_type_vector::uvector<ValT>;
    //using storage_type = gtensor::detail::shareable_storage<std::vector<value_type>>;
    //using shape_type = trivial_type_vector::uvector<index_type>;
    using shape_type = std::vector<index_type>;

    using nop_type = NOP;
    using slice_type = slice<index_type, nop_type>;
    using slice_item_type = detail::slice_item<index_type, nop_type>;
    using slice_init_type = std::initializer_list<slice_item_type>;
    using slices_init_type = std::initializer_list<slice_init_type>;
    using slices_collection_type = std::vector<slice_type>;

};

}

namespace detail{

enum class view_kinds : std::size_t {slice, transpose, subdim, reshape};
enum class tensor_kinds : std::size_t {storage, evaluating, viewing};
struct storage_type_tag : std::integral_constant<tensor_kinds,tensor_kinds::storage>{};
struct evaluating_type_tag : std::integral_constant<tensor_kinds,tensor_kinds::evaluating>{};
struct viewing_type_tag : std::integral_constant<tensor_kinds,tensor_kinds::viewing>{};

// template<typename> struct tensor_type_tag_selector;
// template<typename...Ts> struct tensor_type_tag_selector<storage_tensor<Ts...>>{using type = storage_type_tag;};
// template<typename...Ts> struct tensor_type_tag_selector<evaluating_tensor<Ts...>>{using type = evaluating_type_tag;};
// template<typename...Ts> struct tensor_type_tag_selector<viewing_tensor<Ts...>>{using type = viewing_type_tag;};



template<typename C> inline constexpr bool is_caching_always = std::is_same_v<C::caching_mode,gtensor::config::mode_caching_always>;
template<typename C> inline constexpr bool is_mode_div_libdivide = std::is_same_v<C::div_mode,gtensor::config::mode_div_libdivide>;
template<typename C> inline constexpr bool is_mode_div_native = std::is_same_v<C::div_mode,gtensor::config::mode_div_native>;
template<typename C> inline constexpr bool is_mode_trivial_broadcast_eval_multi = std::is_same_v<C::trivial_broadcast_eval_mode,gtensor::config::mode_trivial_broadcast_eval_multi>;
template<typename C> inline constexpr bool is_mode_trivial_broadcast_eval_flat = std::is_same_v<C::trivial_broadcast_eval_mode,gtensor::config::mode_trivial_broadcast_eval_flat>;
template<typename C> inline constexpr bool is_mode_trivial_broadcast_eval_combi = std::is_same_v<C::trivial_broadcast_eval_mode,gtensor::config::mode_trivial_broadcast_eval_combi>;


template<typename C>
struct is_caching_never{constexpr static bool value = std::is_same_v<C::caching_mode,gtensor::config::mode_caching_never>;};
template<typename C>
struct is_caching_broadcast{constexpr static bool value = std::is_same_v<C::caching_mode,gtensor::config::mode_caching_broadcast>;};

}   //end of namespace detail

}   //end of namespace gtensor

#endif