#ifndef FORWARD_DECL_HPP_
#define FORWARD_DECL_HPP_

namespace gtensor{

template<typename ValT, template<typename> typename Cfg> class tensor_base;
template<typename ValT, template<typename> typename Cfg> class storage_tensor_impl_base;
template<typename ValT, template<typename> typename Cfg> class expression_impl_base;
template<typename ValT, template<typename> typename Cfg> class trivial_impl_base;
template<typename ValT, template<typename> typename Cfg> class view_impl_base;
template<typename ValT, template<typename> typename Cfg> class view_expression_impl_base;
template<typename ValT, template<typename> typename Cfg> class view_index_converter;
template<typename ValT, template<typename> typename Cfg> class walker_maker;
template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops> class expression_tensor;
template<typename ValT, template<typename> typename Cfg> class storage_tensor;
template<typename ValT, template<typename> typename Cfg> class descriptor_base;
template<typename ValT, template<typename> typename Cfg, typename DescT> class view_tensor;
template<typename ValT, template<typename> typename Cfg> class tensor;
template<typename ValT, template<typename> typename Cfg> class walker;
template<typename ValT, template<typename> typename Cfg> class evaluating_storage;
template<typename ValT, template<typename> typename Cfg> class walker_impl_base;
template<typename ValT, template<typename> typename Cfg> class storage_walker_impl;
template<typename ValT, template<typename> typename Cfg> class storage_walker_inline_impl;
template<typename ValT, template<typename> typename Cfg> class ewalker_trivial_impl;
template<typename ValT, template<typename> typename Cfg> class vwalker_impl;
template<typename ValT, template<typename> typename Cfg> class view_expression_walker_impl;
template<typename DifT, typename N> struct slice;
template<typename ValT, template<typename> typename Cfg, typename Wkr> class multiindex_iterator_impl;

namespace detail{
template<typename DifT, typename N> struct slice_item;
template<typename ImplT> class shareable_storage;
}


template<typename T> inline constexpr bool is_tensor = false;
template<typename...T> inline constexpr bool is_tensor<tensor<T...>> = true;

}   //end of namespace gtensor


#endif