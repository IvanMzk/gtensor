#ifndef TENSOR_FACTORY_HPP_
#define TENSOR_FACTORY_HPP_

#include "type_selector.hpp"

namespace gtensor{

//tensor factory
template<typename CfgT, typename Selector> class tensor_factory
{
    using impl_base_type = typename Selector::base_type;
    using impl_type = typename Selector::type;
public:
    template<typename...Args>
    static auto make(Args&&...args){
        return tensor<typename impl_type::value_type, CfgT, impl_base_type>::template make_tensor<impl_type>(std::forward<Args>(args)...);
    }
};
template<typename CfgT, typename...Ts> class storage_tensor_factory : public tensor_factory<CfgT, storage_tensor_implementation_selector<CfgT, Ts...>>{};
template<typename CfgT, typename...Ts> class evaluating_tensor_factory : public tensor_factory<CfgT, evaluating_tensor_implementation_selector<CfgT, Ts...>>{};
template<typename CfgT, typename...Ts> class viewing_tensor_factory : public tensor_factory<CfgT, viewing_tensor_implementation_selector<CfgT, Ts...>>{};

}   //end of namespace gtensor

#endif