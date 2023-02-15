#ifndef TEST_CONFIG_HPP_
#define TEST_CONFIG_HPP_
#include "config.hpp"

namespace test_config{

template<typename Div, typename HEng>
struct config_tmpl_{
    using host_engine = HEng;
    //using device_engine = DEng;
    using div_mode = Div;
    using difference_type = typename gtensor::config::default_config::difference_type;
    using index_type = typename gtensor::config::default_config::index_type;
    template<typename ValT> using storage = typename gtensor::config::default_config::storage<ValT>;

    using shape_type = typename gtensor::config::default_config::shape_type;
};

template<typename HEng, typename Div>
struct config_host_engine_div_selector{
    using config_type = config_tmpl_<
        Div,
        HEng
        >;
};

template<typename HEng>
struct config_host_engine_selector{
    using config_type = config_tmpl_<
        typename gtensor::config::default_config::div_mode,
        HEng
        >;
};

template<typename Div>
struct config_div_mode_selector{
    using config_type = config_tmpl_<
        Div,
        typename gtensor::config::default_config::host_engine
        >;
};


}

#endif