#ifndef IMPL_VIEW_HPP_
#define IMPL_VIEW_HPP_

#include "impl_tensor_base.hpp"

namespace gtensor{

template<typename ValT, template<typename> typename Cfg, typename StorT>
class view_impl : public tensor_impl_base<ValT, Cfg> {

};

}   //end of namespace gtensor


#endif