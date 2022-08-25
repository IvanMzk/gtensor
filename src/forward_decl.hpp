#ifndef FORWARD_DECL_HPP_
#define FORWARD_DECL_HPP_

namespace gtensor{

template<typename ValT, typename CfgT> class tensor_base;
template<typename ValT, typename CfgT> class storing_base;
template<typename ValT, typename CfgT> class evaluating_base;
template<typename ValT, typename CfgT> class evaluating_trivial_base;
template<typename ValT, typename CfgT> class viewing_evaluating_base;
template<typename ValT, typename CfgT> class converting_base;
template<typename ValT, typename CfgT> class storage_tensor;
template<typename ValT, typename CfgT> class storage_walker;
template<typename ValT, typename CfgT> class storage_walker_polymorphic;
template<typename ValT, typename CfgT> class walker;
template<typename ValT, typename CfgT> class indexer;
template<typename ValT, typename CfgT> class walker_base;
template<typename ValT, typename CfgT> class evaluating_trivial_walker;
template<typename ValT, typename CfgT> class viewing_evaluating_walker;
template<typename ValT, typename CfgT> class tensor_wrapper;
template<typename ValT, typename CfgT, typename Wkr> class multiindex_iterator;
template<typename ValT, typename CfgT, typename F, typename FactoryT, typename...Ops> class evaluating_tensor;
template<typename ValT, typename CfgT> class tensor;
template<typename ValT, typename CfgT, typename DescT> class view_tensor;
template<typename ValT, typename CfgT> class vwalker_impl;
template<typename DifT, typename N> struct slice;

template<typename ValT, typename CfgT> class stensor_descriptor;
template<typename CfgT> class descriptor_base;

namespace detail{
template<typename DifT, typename N> struct slice_item;
template<typename ImplT> class shareable_storage;
}


template<typename T> inline constexpr bool is_tensor = false;
template<typename...T> inline constexpr bool is_tensor<tensor<T...>> = true;

}   //end of namespace gtensor


#endif