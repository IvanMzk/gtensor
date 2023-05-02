#ifndef TEST_CONFIG_HPP_
#define TEST_CONFIG_HPP_
#include "config.hpp"

namespace test_config{

template<typename Div, typename Engine, template<typename...> typename Storage>
struct config_tmpl_{
    using engine = Engine;
    using div_mode = Div;
    template<typename T> using storage = Storage<T>;
    template<typename T> using shape = typename gtensor::config::default_config::shape<T>;
    template<typename T> using container = typename gtensor::config::default_config::container<T>;
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
        gtensor::config::default_config::template storage
    >;
};

template<typename Div>
struct config_div_mode_selector{
    using config_type = config_tmpl_<
        Div,
        gtensor::config::default_config::engine,
        gtensor::config::default_config::template storage
    >;
};
template<template<typename...> typename Storage>
struct config_storage_selector{
    using config_type = config_tmpl_<
        gtensor::config::default_config::div_mode,
        gtensor::config::default_config::engine,
        Storage
    >;
};

}   //end of namespace test_config

#endif