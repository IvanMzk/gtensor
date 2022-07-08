#ifndef FORWARD_DECL_HPP_
#define FORWARD_DECL_HPP_

namespace gtensor{

template<typename ValT, typename F, template<typename> typename Cfg, typename...Ops> class expression_impl;
template<typename ValT, template<typename> typename Cfg> class tensor_impl_base;
template<typename ValT, template<typename> typename Cfg> class stensor_impl;
template<typename ValT, template<typename> typename Cfg> class tensor;
template<typename ValT, template<typename> typename Cfg> class walker_impl_base;
template<typename DifT, typename N> struct slice;

namespace detail{
template<typename DifT, typename N> struct slice_item;
}


template<typename T> inline constexpr bool is_tensor = false;
template<typename...T> inline constexpr bool is_tensor<tensor<T...>> = true;

}   //end of namespace gtensor


#endif