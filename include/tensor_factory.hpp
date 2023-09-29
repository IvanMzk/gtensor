#ifndef TENSOR_FACTORY_HPP_
#define TENSOR_FACTORY_HPP_

#include "config.hpp"
#include "tensor_core.hpp"
#include "tensor_implementation.hpp"

namespace gtensor{

template<typename Config, typename T, typename Layout>
class tensor_factory
{
    using config_type = config::extend_config_t<Config,T>;
public:
    using result_type = tensor_implementation<storage_core<config_type,T,Layout>>;

    template<typename...Args>
    static auto create(Args&&...args){
        return std::make_shared<result_type>(std::forward<Args>(args)...);
    }
};

}   //end of namespace gtensor
#endif
