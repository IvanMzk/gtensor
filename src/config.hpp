#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <vector>
#include "uvector.h"
#include "forward_decl.hpp"

namespace gtensor{

namespace config{


enum class caching_modes : std::size_t {caching_never, caching_always, caching_broadcast};
enum class deref_modes : std::size_t {deref_val, deref_variant};
enum class div_modes : std::size_t {native, libdivide};
enum class trivial_broadcast_eval_modes : std::size_t {flat, multi, combi};

template<typename T, T M> struct mode{static constexpr T value = M;};
using mode_caching_never = mode<caching_modes, caching_modes::caching_never>;
using mode_caching_always = mode<caching_modes, caching_modes::caching_always>;
using mode_caching_broadcast = mode<caching_modes, caching_modes::caching_broadcast>;
using mode_deref_value = mode<deref_modes, deref_modes::deref_val>;
using mode_deref_variant = mode<deref_modes, deref_modes::deref_variant>;
using mode_div_native = mode<div_modes, div_modes::native>;
using mode_div_libdivide = mode<div_modes, div_modes::libdivide>;
using mode_trivial_broadcast_eval_flat = mode<trivial_broadcast_eval_modes, trivial_broadcast_eval_modes::flat>;
using mode_trivial_broadcast_eval_multi = mode<trivial_broadcast_eval_modes, trivial_broadcast_eval_modes::multi>;
using mode_trivial_broadcast_eval_combi = mode<trivial_broadcast_eval_modes, trivial_broadcast_eval_modes::combi>;

struct NOP{};

/*
* ValT type of tensor element
*/
template<typename ValT>
struct default_config{    
    using value_type = ValT;
    
    using caching_mode = mode_caching_broadcast;
    //using caching_mode = typename mode_caching_always;
    //using caching_mode = typename mode_caching_never;
    
    using trivial_broadcast_eval_mode = mode_trivial_broadcast_eval_combi;
    //using trivial_broadcast_eval_mode = mode_trivial_broadcast_eval_multi;
    //using trivial_broadcast_eval_mode = mode_trivial_broadcast_eval_flat;
    
    //using div_mode = mode_div_libdivide;
    using div_mode = mode_div_native;
    
    using difference_type = std::int64_t;
    using index_type = difference_type;
    using storage_type = gtensor::detail::shareable_storage<std::vector<value_type>>;
    //using storage_type = std::vector<value_type>;
    using shape_type = trivial_type_vector::uvector<index_type>;

    using nop_type = NOP;
    using slice_type = slice<index_type, nop_type>;
    using slice_item_type = detail::slice_item<index_type, nop_type>;
    using slice_init_type = std::initializer_list<slice_item_type>;
    using slices_init_type = std::initializer_list<slice_init_type>;
    using slices_collection_type = std::vector<slice_type>;    

};

}

namespace detail{

enum class view_kind {slice, transpose, subdim, reshape};
enum class tensor_kind {stensor, expression, view};

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

}

}

#endif