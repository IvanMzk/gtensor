#ifndef FORWARD_DECL_HPP_
#define FORWARD_DECL_HPP_

namespace gtensor{

template<typename ValT, typename CfgT> class tensor_base;
template<typename ValT, typename CfgT> class storing_base;
template<typename ValT, typename CfgT> class evaluating_base;
template<typename ValT, typename CfgT> class evaluating_trivial_base;
template<typename ValT, typename CfgT> class viewing_evaluating_base;
template<typename ValT, typename CfgT> class storage_walker;
template<typename ValT, typename CfgT> class storage_walker_polymorphic;
template<typename ValT, typename CfgT> class walker;
template<typename ValT, typename CfgT> class indexer;
template<typename ValT, typename CfgT> class walker_base;
template<typename ValT, typename CfgT, typename F, typename...Wks> class evaluating_trivial_walker;
template<typename ValT, typename CfgT> class viewing_evaluating_walker;
template<typename ValT, typename CfgT> class tensor_wrapper;
template<typename ValT, typename CfgT, typename Wkr> class broadcast_iterator;
template<typename ValT, typename CfgT> class vwalker_impl;
template<typename DifT, typename N> struct slice;

template<typename ValT, typename CfgT, typename ImplT> class tensor;
template<typename DescT, typename EngineT> class viewing_tensor;
template<typename EngineT> class storage_tensor;
template<typename EngineT> class evaluating_tensor;
template<typename CfgT> class converting_base;

template<typename CfgT> class descriptor_base;
template<typename CfgT> class basic_descriptor;
template<typename CfgT> class descriptor_with_offset;
template<typename CfgT> class converting_descriptor;

namespace detail{
template<typename DifT, typename N> struct slice_item;
template<typename ImplT> class shareable_storage;
}

}   //end of namespace gtensor


#endif