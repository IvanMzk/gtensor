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
template<typename ValT, template<typename> typename Cfg, typename F, typename FactoryT, typename...Ops> class evaluating_tensor;
template<typename ValT, template<typename> typename Cfg> class tensor_wrapper;
template<typename ValT, typename CfgT> class descriptor_base;
template<typename ValT, template<typename> typename Cfg, typename DescT> class view_tensor;
template<typename ValT, template<typename> typename Cfg> class tensor;
template<typename ValT, template<typename> typename Cfg> class walker;
template<typename ValT, template<typename> typename Cfg> class indexer;
template<typename ValT, template<typename> typename Cfg> class walker_base;
template<typename ValT, template<typename> typename Cfg> class storage_walker_polymorphic;
template<typename ValT, template<typename> typename Cfg> class storage_walker;
template<typename ValT, template<typename> typename Cfg> class evaluating_trivial_walker;
template<typename ValT, template<typename> typename Cfg> class vwalker_impl;
template<typename ValT, template<typename> typename Cfg> class viewing_evaluating_walker;
template<typename DifT, typename N> struct slice;
template<typename ValT, template<typename> typename Cfg, typename Wkr> class multiindex_iterator;

namespace detail{
template<typename DifT, typename N> struct slice_item;
template<typename ImplT> class shareable_storage;
}


template<typename T> inline constexpr bool is_tensor = false;
template<typename...T> inline constexpr bool is_tensor<tensor<T...>> = true;

}   //end of namespace gtensor


#endif