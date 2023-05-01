#ifndef TENSOR_FACTORY_HPP_
#define TENSOR_FACTORY_HPP_

#include "config.hpp"
#include "tensor_implementation.hpp"

namespace gtensor{

template<typename Config, typename T>
class tensor_factory
{
    using config_type = config::extend_config_t<Config,T>;
public:
    using result_type = tensor_implementation<storage_core<config_type,T>>;

    template<typename...Args>
    static auto create(Args&&...args){
        return std::make_shared<result_type>(std::forward<Args>(args)...);
    }
};

// namespace detail{
// template<typename Impl>
// auto create_tensor(std::shared_ptr<Impl>&& impl){
//     return basic_tensor<Impl>{std::forward<Impl>(impl)};
// }
// }   //end of namespace detail

// template<typename T, typename Config = config::default_config, typename ShT>
// auto create_tensor(ShT&& shape){
//     return detail::create_tensor(tensor_factory_selector_t<Config,T>::create(std::forward<ShT>(shape)));
// }
// template<typename Config = config::default_config, typename T, typename ShT>
// auto create_tensor(ShT&& shape, const T& value){
//     return detail::create_tensor(tensor_factory_selector_t<Config,T>::create(std::forward<ShT>(shape),value));
// }
// template<typename Config = config::default_config, typename T>
// auto create_tensor(std::initializer_list<T> init_list){
//     return detail::create_tensor(tensor_factory_selector_t<Config,T>::create(init_list));
// }
// template<typename Config = config::default_config, typename T>
// auto create_tensor(std::initializer_list<std::initializer_list<T>> init_list){
//     return detail::create_tensor(tensor_factory_selector_t<Config,T>::create(init_list));
// }

}   //end of namespace gtensor
#endif
