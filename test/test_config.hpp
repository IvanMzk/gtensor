#ifndef TEST_CONFIG_HPP_
#define TEST_CONFIG_HPP_
#include <tuple>
#include "config.hpp"

namespace test_config{

template<typename Div, typename HEng, typename DEng>
struct config_tmpl_{
    //using engine = HEng;
    using host_engine = HEng;
    using device_engine = DEng;
    using div_mode = Div;
    using difference_type = typename gtensor::config::default_config::difference_type;
    using index_type = typename gtensor::config::default_config::index_type;
    template<typename ValT> using storage = typename gtensor::config::default_config::storage<ValT>;

    using shape_type = typename gtensor::config::default_config::shape_type;

    // using nop_type = typename gtensor::config::default_config::nop_type;;
    // using slice_type = typename gtensor::config::default_config::slice_type;;
    // using slice_item_type = typename gtensor::config::default_config::slice_item_type;
    // using slice_init_type = typename gtensor::config::default_config::slice_init_type;
    // using slices_init_type = typename gtensor::config::default_config::slices_init_type;
    // using slices_collection_type = typename gtensor::config::default_config::slices_collection_type;
};

template<typename HEng, typename Div>
struct config_host_engine_div_selector{
    using config_type = config_tmpl_<
        Div,
        HEng,
        typename gtensor::config::default_config::device_engine
        >;
};

template<typename HEng>
struct config_host_engine_selector{
    using config_type = config_tmpl_<
        typename gtensor::config::default_config::div_mode,
        HEng,
        typename gtensor::config::default_config::device_engine
        >;
};


template<typename Div>
struct config_div_mode_selector{
    using config_type = config_tmpl_<
        Div,
        typename gtensor::config::default_config::host_engine,
        typename gtensor::config::default_config::device_engine
        >;
};


}

#endif