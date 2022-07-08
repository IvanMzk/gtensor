#ifndef IMPL_VIEW_HPP_
#define IMPL_VIEW_HPP_

#include "shareable_storage.hpp"
#include "impl_tensor_base.hpp"

namespace gtensor{

template<typename ValT, template<typename> typename Cfg, typename StorT>
class view_impl : public tensor_impl_base<ValT, Cfg> {
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::storage_type;
    using descriptor_type = stensor_descriptor<value_type, Cfg>;
    using slices_init_type = typename config_type::slices_init_type;
    using slices_collection_type = typename config_type::slices_collection_type;



};

}   //end of namespace gtensor


#endif