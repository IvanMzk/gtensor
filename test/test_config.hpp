#ifndef TEST_CONFIG_HPP_
#define TEST_CONFIG_HPP_
#include <tuple>
#include "config.hpp"

namespace test_config{


using mode_trivial_broadcast_eval_list = std::tuple<gtensor::config::mode_trivial_broadcast_eval_multi, gtensor::config::mode_trivial_broadcast_eval_flat, gtensor::config::mode_trivial_broadcast_eval_combi>;

using mode_caching_list = std::tuple<gtensor::config::mode_caching_always,gtensor::config::mode_caching_broadcast,gtensor::config::mode_caching_never>;
using mode_caching_list_always_never = std::tuple<gtensor::config::mode_caching_always,gtensor::config::mode_caching_never>;
using mode_caching_list_broadcast_never = std::tuple<gtensor::config::mode_caching_broadcast,gtensor::config::mode_caching_never>;

using mode_caching_list_always = std::tuple<gtensor::config::mode_caching_always>;
using mode_caching_list_broadcast = std::tuple<gtensor::config::mode_caching_broadcast>;
using mode_caching_list_never = std::tuple<gtensor::config::mode_caching_never>;

template<typename M, typename Div, typename Ev>
struct config_tmpl_{    
    using caching_mode = M;
    using trivial_broadcast_eval_mode = Ev;
    using div_mode = Div;
    using difference_type = typename gtensor::config::default_config::difference_type;
    using index_type = typename gtensor::config::default_config::index_type;
    template<typename ValT> using storage = typename gtensor::config::default_config::storage<ValT>;
    
    using shape_type = typename gtensor::config::default_config::shape_type;

    using nop_type = typename gtensor::config::default_config::nop_type;;
    using slice_type = typename gtensor::config::default_config::slice_type;;
    using slice_item_type = typename gtensor::config::default_config::slice_item_type;
    using slice_init_type = typename gtensor::config::default_config::slice_init_type;
    using slices_init_type = typename gtensor::config::default_config::slices_init_type;
    using slices_collection_type = typename gtensor::config::default_config::slices_collection_type;
};    

template<typename M>
struct config_caching_mode_selector{
    using config_type = config_tmpl_<
        M, 
        typename gtensor::config::default_config::div_mode, 
        typename gtensor::config::default_config::trivial_broadcast_eval_mode
        >;
};

template<typename Div>
struct config_div_mode_selector{    
    using config_type = config_tmpl_<
        typename gtensor::config::default_config::caching_mode,
        Div, 
        typename gtensor::config::default_config::trivial_broadcast_eval_mode
        >; 
};

template<typename M, typename Ev>
struct config_tmpl_caching_eval_mode_selector{    
    using config_type = config_tmpl_<
        M,
        typename gtensor::config::default_config::div_mode,
        Ev
        >; 
};


// template<typename T> using config_caching_never = config_tmpl_<T,gtensor::config::mode_caching_never,gtensor::config::mode_deref_value, gtensor::config::mode_div_native>;
// template<typename T> using config_caching_broadcast = config_tmpl_<T,gtensor::config::mode_caching_broadcast,gtensor::config::mode_deref_value, gtensor::config::mode_div_native>;
// template<typename T> using config_caching_always = config_tmpl_<T,gtensor::config::mode_caching_always,gtensor::config::mode_deref_value, gtensor::config::mode_div_native>;

/*
* helper to parameterize tests with storage container template
*/
enum class container_id : std::size_t{vector_id, uvector_id};
template<typename T, T Id>
struct container_id_type{static constexpr T value = Id;};
using vector_id_type = typename container_id_type<container_id, container_id::vector_id>;
using uvector_id_type = typename container_id_type<container_id, container_id::uvector_id>;

template<typename, typename Idx = std::int64_t> struct container_selector;
template<typename Idx> struct container_selector<vector_id_type, Idx>{
    using index_type = Idx;
    template<typename T> using container_tmpl = std::vector<T>;    
};
template<typename Idx> struct container_selector<uvector_id_type, Idx>{
    using index_type = Idx;
    template<typename T> using container_tmpl = trivial_type_vector::uvector<T>;
};


}

#endif