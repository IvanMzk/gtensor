#ifndef TEST_CONFIG_HPP_
#define TEST_CONFIG_HPP_
#include "config.hpp"

namespace test_config{

template<typename Div, typename Engine, template<typename...> typename Storage, template<typename...> typename IndexMap>
struct config_tmpl_{
    using engine = Engine;
    using div_mode = Div;
    template<typename T> using storage = Storage<T>;
    template<typename T> using shape = typename gtensor::config::default_config::shape<T>;
    template<typename T> using container = typename gtensor::config::default_config::container<T>;
    template<typename T> using index_map = IndexMap<T>;
};

// template<typename Eng, typename Div>
// struct config_engine_div_selector{
//     using config_type = config_tmpl_<Div,Eng>;
// };

template<typename Engine>
struct config_engine_selector{
    using config_type = config_tmpl_<
        typename gtensor::config::default_config::div_mode,
        Engine,
        gtensor::config::default_config::template storage,
        gtensor::config::default_config::template index_map
    >;
};
template<typename Engine> using config_engine_selector_t = typename config_engine_selector<Engine>::config_type;

template<typename Div>
struct config_div_mode_selector{
    using config_type = config_tmpl_<
        Div,
        gtensor::config::default_config::engine,
        gtensor::config::default_config::template storage,
        gtensor::config::default_config::template index_map
    >;
};
template<typename Div> using config_div_mode_selector_t = typename config_div_mode_selector<Div>::config_type;

template<template<typename...> typename Storage>
struct config_storage_selector{
    using config_type = config_tmpl_<
        gtensor::config::default_config::div_mode,
        gtensor::config::default_config::engine,
        Storage,
        gtensor::config::default_config::template index_map
    >;
};
template<template<typename...> typename Storage> using config_storage_selector_t = typename config_storage_selector<Storage>::config_type;

template<template<typename...> typename Storage, template<typename...> typename IndexMap>
struct config_storage_index_map_selector{
    using config_type = config_tmpl_<
        gtensor::config::default_config::div_mode,
        gtensor::config::default_config::engine,
        Storage,
        IndexMap
    >;
};
template<template<typename...> typename Storage ,template<typename...> typename IndexMap>
using config_storage_index_map_selector_t = typename config_storage_index_map_selector<Storage, IndexMap>::config_type;

}   //end of namespace test_config

#endif