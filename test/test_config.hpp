#ifndef TEST_CONFIG_HPP_
#define TEST_CONFIG_HPP_
#include "config.hpp"

namespace test_config{

template<typename Div, typename Eng>
struct config_tmpl_{
    using engine = Eng;
    using div_mode = Div;
    template<typename ValT> using storage = typename gtensor::config::default_config::storage<ValT>;
    template<typename ValT> using shape = typename gtensor::config::default_config::shape<ValT>;
    template<typename T> using container = typename gtensor::config::default_config::container<T>;
};

template<typename Eng, typename Div>
struct config_engine_div_selector{
    using config_type = config_tmpl_<Div,Eng>;
};

template<typename Eng>
struct config_engine_selector{
    using config_type = config_tmpl_<typename gtensor::config::default_config::div_mode,Eng>;
};

template<typename Div>
struct config_div_mode_selector{
    using config_type = config_tmpl_<Div,typename gtensor::config::default_config::engine>;
};

}   //end of namespace test_config

#endif