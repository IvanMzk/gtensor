#ifndef TEST_CONFIG_HPP_
#define TEST_CONFIG_HPP_
#include "config.hpp"

namespace test_config{

using test_default_config = gtensor::config::default_config;

template<typename Engine>
struct config_engine_selector : test_default_config{
    using engine = Engine;
};
template<typename Engine> using config_engine_selector_t = config_engine_selector<Engine>;

template<typename Div>
struct config_div_mode_selector : test_default_config{
    using div_mode = Div;
};
template<typename Div> using config_div_mode_selector_t = config_div_mode_selector<Div>;

template<typename Layout>
struct config_layout_selector : test_default_config{
    using layout = Layout;
};
template<typename Layout> using config_layout_selector_t = config_layout_selector<Layout>;

template<template<typename...> typename Storage>
struct config_storage_selector : test_default_config{
    template<typename T> using storage = Storage<T>;
};
template<template<typename...> typename Storage> using config_storage_selector_t = config_storage_selector<Storage>;

template<template<typename...> typename Storage, template<typename...> typename IndexMap>
struct config_storage_index_map_selector : test_default_config{
    template<typename T> using storage = Storage<T>;
    template<typename T> using index_map = IndexMap<T>;
};
template<template<typename...> typename Storage ,template<typename...> typename IndexMap>
using config_storage_index_map_selector_t = config_storage_index_map_selector<Storage, IndexMap>;

}   //end of namespace test_config

#endif