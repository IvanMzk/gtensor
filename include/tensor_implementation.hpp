/*
* GTensor - computation library
* Copyright (c) 2022 Ivan Malezhyk <ivanmzk@gmail.com>
*
* Distributed under the Boost Software License, Version 1.0.
* The full license is in the file LICENSE.txt, distributed with this software.
*/

#ifndef TENSOR_IMPLEMENTATION_HPP_
#define TENSOR_IMPLEMENTATION_HPP_

#include <type_traits>
#include <iterator>
#include <memory>
#include "common.hpp"
#include "descriptor.hpp"
#include "data_accessor.hpp"
#include "iterator.hpp"

namespace gtensor{

namespace detail{

//create walker
template<typename Config, typename Indexer, typename Order> using max_dim_indexer_walker = gtensor::axes_correction_walker<gtensor::indexer_walker<Config,Indexer,Order>>;
template<typename Config, typename Iterator, typename Order> using max_dim_iterator_walker = gtensor::axes_correction_walker<gtensor::cursor_walker<Config,Iterator,Order>>;
template<typename Core, typename Descriptor, typename DimT>
inline auto create_walker(Core& t, const Descriptor& descriptor, const DimT& max_dim){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    using order = typename Core::order;
    const auto dim_offset = max_dim-descriptor.dim();
    if constexpr(has_callable_create_walker_dim_type<Core>::value){
        return t.create_walker(max_dim);
    }else if constexpr (has_callable_create_indexer<Core>::value){
        using indexer_type = decltype(t.create_indexer());
        return max_dim_indexer_walker<config_type,indexer_type,order>{dim_offset,descriptor.adapted_strides(),descriptor.reset_strides(),index_type{0},t.create_indexer()};
    }else if constexpr (has_callable_random_access_iterator<Core>::value){
        using iterator_type = decltype(t.begin());
        return max_dim_iterator_walker<config_type,iterator_type,order>{dim_offset,descriptor.adapted_strides(),descriptor.reset_strides(),t.begin()};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return max_dim_indexer_walker<config_type,indexer_type,order>{dim_offset,descriptor.adapted_strides(),descriptor.reset_strides(),index_type{0},indexer_type{t}};
    }else if constexpr (has_callable_iterator<Core>::value){
        using iterator_type = decltype(t.begin());
        return max_dim_iterator_walker<config_type,iterator_type,order>{dim_offset,descriptor.adapted_strides(),descriptor.reset_strides(),t.begin()};
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}
template<typename Core, typename Descriptor>
inline auto create_walker(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    using order = typename Core::order;
    if constexpr(has_callable_create_walker<Core>::value){
        return t.create_walker();
    }else if constexpr (has_callable_create_indexer<Core>::value){
        using indexer_type = decltype(t.create_indexer());
        return gtensor::indexer_walker<config_type,indexer_type,order>{descriptor.adapted_strides(),descriptor.reset_strides(),index_type{0},t.create_indexer()};
    }else if constexpr (has_callable_random_access_iterator<Core>::value){
        using iterator_type = decltype(t.begin());
        return gtensor::cursor_walker<config_type,iterator_type,order>{descriptor.adapted_strides(),descriptor.reset_strides(),t.begin()};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return gtensor::indexer_walker<config_type,indexer_type,order>{descriptor.adapted_strides(),descriptor.reset_strides(),index_type{0},indexer_type{t}};
    }else if constexpr (has_callable_iterator<Core>::value){
        using iterator_type = decltype(t.begin());
        return gtensor::cursor_walker<config_type,iterator_type,order>{descriptor.adapted_strides(),descriptor.reset_strides(),t.begin()};
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}

//create indexer
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto create_walker_indexer(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    using walker_type = decltype(create_walker(t,descriptor));
    return gtensor::walker_indexer<walker_type, TraverseOrder>{descriptor.strides_div(TraverseOrder{}), create_walker(t,descriptor)};
}
template<typename Core, typename Descriptor>
inline auto create_indexer(Core& t, const Descriptor& descriptor){
    if constexpr (has_callable_create_indexer<Core>::value){
        return t.create_indexer();
    }else if constexpr (has_callable_create_walker<Core>::value){
        return create_walker_indexer<typename Core::order>(t,descriptor);
    }else if constexpr (has_callable_random_access_iterator<Core>::value){
        using iterator_type = decltype(t.begin());
        return gtensor::iterator_indexer<iterator_type>{t.begin()};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        return gtensor::basic_indexer<Core&>{t};
    }else if constexpr (has_callable_iterator<Core>::value){
        using iterator_type = decltype(t.begin());
        return gtensor::iterator_indexer<iterator_type>{t.begin()};
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto create_indexer(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    if constexpr (std::is_same_v<typename Core::order,TraverseOrder>){
        return create_indexer(t,descriptor);
    }else{
        return create_walker_indexer<TraverseOrder>(t,descriptor);
    }
}

//create trivial walker
template<typename Core, typename Descriptor>
inline auto create_trivial_walker(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    using order = typename Core::order;
    if constexpr (has_callable_create_trivial_indexer<Core>::value){
        using indexer_type = decltype(t.create_trivial_indexer());
        return gtensor::indexer_walker<config_type,indexer_type,order>{descriptor.adapted_strides(),descriptor.reset_strides(),index_type{0},t.create_trivial_indexer()};
    }else{
        return create_walker(t,descriptor);
    }
}

//create trivial indexer
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto create_trivial_walker_indexer(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    using walker_type = decltype(create_trivial_walker(t,descriptor));
    return gtensor::walker_indexer<walker_type, TraverseOrder>{descriptor.strides_div(TraverseOrder{}), create_trivial_walker(t,descriptor)};
}
template<typename Core, typename Descriptor>
inline auto create_trivial_indexer(Core& t, const Descriptor& descriptor){
    if constexpr (has_callable_create_trivial_indexer<Core>::value){
        return t.create_trivial_indexer();
    }else{
        return create_indexer(t,descriptor);
    }
}
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto create_trivial_indexer(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    if constexpr (std::is_same_v<typename Core::order,TraverseOrder>){
        return create_trivial_indexer(t,descriptor);
    }else{
        return create_trivial_walker_indexer<TraverseOrder>(t,descriptor);
    }
}

//create iterator
template<typename TraverseOrder, typename Core, typename Descriptor, typename IdxT>
inline auto create_walker_iterator(Core& t, const Descriptor& descriptor, const IdxT& pos){
    ASSERT_ORDER(TraverseOrder);
    using config_type = typename Core::config_type;
    using walker_type = decltype(create_walker(t,descriptor));
    using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,TraverseOrder>;
    return gtensor::walker_iterator<config_type,traverser_type>{
        create_walker(t,descriptor),
        descriptor.shape(),
        descriptor.strides_div(TraverseOrder{}),
        pos
    };
}
template<typename Core, typename Descriptor>
inline auto begin(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr (has_callable_iterator<Core>::value){
        return t.begin();
    }else if constexpr (has_callable_create_indexer<Core>::value){
        return gtensor::indexer_iterator<config_type, decltype(t.create_indexer())>{t.create_indexer(), index_type{0}};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return gtensor::indexer_iterator<config_type, indexer_type>{indexer_type{t}, index_type{0}};
    }else if constexpr(has_callable_create_walker<Core>::value){
        return create_walker_iterator<typename Core::order>(t,descriptor,index_type{0});
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}
template<typename Core, typename Descriptor>
inline auto end(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    if constexpr (has_callable_iterator<Core>::value){
        return t.end();
    }else if constexpr (has_callable_create_indexer<Core>::value){
        return gtensor::indexer_iterator<config_type, decltype(t.create_indexer())>{t.create_indexer(), descriptor.size()};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return gtensor::indexer_iterator<config_type, indexer_type>{indexer_type{t}, descriptor.size()};
    }else if constexpr(has_callable_create_walker<Core>::value){
        return create_walker_iterator<typename Core::order>(t,descriptor,descriptor.size());
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto begin(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr (std::is_same_v<typename Core::order,TraverseOrder>){
        return begin(t,descriptor);
    }else{
        return create_walker_iterator<TraverseOrder>(t,descriptor,index_type{0});
    }
}
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto end(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    if constexpr (std::is_same_v<typename Core::order,TraverseOrder>){
        return end(t,descriptor);
    }else{
        return create_walker_iterator<TraverseOrder>(t,descriptor,descriptor.size());
    }
}

//create trivial iterator
template<typename TraverseOrder, typename Core, typename Descriptor, typename IdxT>
inline auto create_trivial_walker_iterator(Core& t, const Descriptor& descriptor, const IdxT& pos){
    ASSERT_ORDER(TraverseOrder);
    using config_type = typename Core::config_type;
    using walker_type = decltype(create_trivial_walker(t,descriptor));
    using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,TraverseOrder>;
    return gtensor::walker_iterator<config_type,traverser_type>{
        create_trivial_walker(t,descriptor),
        descriptor.shape(),
        descriptor.strides_div(TraverseOrder{}),
        pos
    };
}
template<typename Core, typename Descriptor>
inline auto begin_trivial(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr (has_callable_iterator_trivial<Core>::value){
        return t.begin_trivial();
    }else if constexpr (has_callable_create_trivial_indexer<Core>::value){
        return gtensor::indexer_iterator<config_type, decltype(t.create_trivial_indexer())>{t.create_trivial_indexer(), index_type{0}};
    }else{
        return begin(t,descriptor);
    }
}
template<typename Core, typename Descriptor>
inline auto end_trivial(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    if constexpr (has_callable_iterator_trivial<Core>::value){
        return t.end_trivial();
    }else if constexpr (has_callable_create_trivial_indexer<Core>::value){
        return gtensor::indexer_iterator<config_type, decltype(t.create_trivial_indexer())>{t.create_trivial_indexer(), descriptor.size()};
    }else{
        return end(t,descriptor);
    }
}
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto begin_trivial(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr (std::is_same_v<typename Core::order,TraverseOrder>){
        return begin_trivial(t,descriptor);
    }else{
        return create_trivial_walker_iterator<TraverseOrder>(t,descriptor,index_type{0});
    }
}
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto end_trivial(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    if constexpr (std::is_same_v<typename Core::order,TraverseOrder>){
        return end_trivial(t,descriptor);
    }else{
        return create_trivial_walker_iterator<TraverseOrder>(t,descriptor,descriptor.size());
    }
}

//create reverse iterator
template<typename TraverseOrder, typename Core, typename Descriptor, typename IdxT>
inline auto create_reverse_walker_iterator(Core& t, const Descriptor& descriptor, const IdxT& pos){
    ASSERT_ORDER(TraverseOrder);
    using config_type = typename Core::config_type;
    using walker_type = decltype(create_walker(t,descriptor));
    using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,TraverseOrder>;
    return gtensor::reverse_walker_iterator<config_type,traverser_type>{
        create_walker(t,descriptor),
        descriptor.shape(),
        descriptor.strides_div(TraverseOrder{}),
        pos
    };
}
template<typename It>
inline std::reverse_iterator<It> create_reverse_iterator(It it){
    return std::reverse_iterator<It>{std::move(it)};
}
template<typename...Ts>
inline auto create_reverse_iterator(indexer_iterator<Ts...> it){
    return gtensor::reverse_iterator_generic<indexer_iterator<Ts...>>{std::move(it)};
}
template<typename...Ts>
inline auto create_reverse_iterator(walker_iterator<Ts...> it){
    return gtensor::reverse_iterator_generic<walker_iterator<Ts...>>{std::move(it)};
}
template<typename Core, typename Descriptor>
inline auto rbegin(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    if constexpr (has_callable_reverse_iterator<Core>::value){
        return t.rbegin();
    }else if constexpr (has_callable_iterator<Core>::value){
        return create_reverse_iterator(t.end());
    }else if constexpr (has_callable_create_indexer<Core>::value){
        return gtensor::reverse_indexer_iterator<config_type, decltype(t.create_indexer())>{t.create_indexer(), descriptor.size()};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return gtensor::reverse_indexer_iterator<config_type, indexer_type>{indexer_type{t}, descriptor.size()};
    }else if constexpr(has_callable_create_walker<Core>::value){
        return create_reverse_walker_iterator<typename Core::order>(t, descriptor, descriptor.size());
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}
template<typename Core, typename Descriptor>
inline auto rend(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr (has_callable_reverse_iterator<Core>::value){
        return t.rend();
    }else if constexpr (has_callable_iterator<Core>::value){
        return create_reverse_iterator(t.begin());
    }else if constexpr (has_callable_create_indexer<Core>::value){
        return gtensor::reverse_indexer_iterator<config_type, decltype(t.create_indexer())>{t.create_indexer(), index_type{0}};
    }else if constexpr (has_callable_subscript_operator<Core>::value){
        using indexer_type = gtensor::basic_indexer<Core&>;
        return gtensor::reverse_indexer_iterator<config_type, indexer_type>{indexer_type{t}, index_type{0}};
    }else if constexpr(has_callable_create_walker<Core>::value){
        return create_reverse_walker_iterator<typename Core::order>(t, descriptor, index_type{0});
    }else{
        static_assert(detail::always_false<Core>,"can't make data accessor");
    }
}
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto rbegin(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    if constexpr (std::is_same_v<typename Core::order,TraverseOrder>){
        return rbegin(t,descriptor);
    }else{
        return create_reverse_walker_iterator<TraverseOrder>(t, descriptor, descriptor.size());
    }
}
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto rend(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr (std::is_same_v<typename Core::order,TraverseOrder>){
        return rend(t,descriptor);
    }else{
        return create_reverse_walker_iterator<TraverseOrder>(t, descriptor, index_type{0});
    }
}

//create trivial reverse iterator
template<typename TraverseOrder, typename Core, typename Descriptor, typename IdxT>
inline auto create_reverse_trivial_walker_iterator(Core& t, const Descriptor& descriptor, const IdxT& pos){
    ASSERT_ORDER(TraverseOrder);
    using config_type = typename Core::config_type;
    using walker_type = decltype(create_trivial_walker(t,descriptor));
    using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,TraverseOrder>;
    return gtensor::reverse_walker_iterator<config_type,traverser_type>{
        create_trivial_walker(t,descriptor),
        descriptor.shape(),
        descriptor.strides_div(TraverseOrder{}),
        pos
    };
}
template<typename Core, typename Descriptor>
inline auto rbegin_trivial(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    if constexpr (has_callable_reverse_iterator_trivial<Core>::value){
        return t.rbegin_trivial();
    }else if constexpr (has_callable_create_trivial_indexer<Core>::value){
        return gtensor::reverse_indexer_iterator<config_type, decltype(t.create_trivial_indexer())>{t.create_trivial_indexer(), descriptor.size()};
    }else{
        return rbegin(t,descriptor);
    }
}
template<typename Core, typename Descriptor>
inline auto rend_trivial(Core& t, const Descriptor& descriptor){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr (has_callable_reverse_iterator_trivial<Core>::value){
        return t.rend_trivial();
    }else if constexpr (has_callable_create_trivial_indexer<Core>::value){
        return gtensor::reverse_indexer_iterator<config_type, decltype(t.create_trivial_indexer())>{t.create_trivial_indexer(), index_type{0}};
    }else{
        return rend(t,descriptor);
    }
}
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto rbegin_trivial(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    if constexpr (std::is_same_v<typename Core::order,TraverseOrder>){
        return rbegin_trivial(t,descriptor);
    }else{
        return create_reverse_trivial_walker_iterator<TraverseOrder>(t,descriptor,descriptor.size());
    }
}
template<typename TraverseOrder, typename Core, typename Descriptor>
inline auto rend_trivial(Core& t, const Descriptor& descriptor){
    ASSERT_ORDER(TraverseOrder);
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    if constexpr (std::is_same_v<typename Core::order,TraverseOrder>){
        return rend_trivial(t,descriptor);
    }else{
        return create_reverse_trivial_walker_iterator<TraverseOrder>(t,descriptor,index_type{0});
    }
}

//create broadcast iterator
template<typename TraverseOrder, typename Core, typename Descriptor, typename ShT, typename IdxT>
inline auto create_broadcast_iterator(Core& t, const Descriptor& descriptor, ShT&& shape, const IdxT& pos){
    ASSERT_ORDER(TraverseOrder);
    using config_type = typename Core::config_type;
    using dim_type = typename config_type::dim_type;
    dim_type max_dim = std::max(descriptor.dim(), detail::make_dim(shape));
    auto strides_div = make_strides_div<config_type>(shape, TraverseOrder{});
    using walker_type = decltype(create_walker(t,descriptor,max_dim));
    using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,TraverseOrder>;
    return broadcast_iterator<config_type,traverser_type>{
        create_walker(t,descriptor,max_dim),
        std::forward<ShT>(shape),
        std::move(strides_div),
        pos
    };
}
template<typename TraverseOrder, typename Core, typename Descriptor, typename ShT>
inline auto begin_broadcast(Core& t, const Descriptor& descriptor, ShT&& shape){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    return create_broadcast_iterator<TraverseOrder>(t, descriptor, std::forward<ShT>(shape),index_type{0});
}
template<typename TraverseOrder, typename Core, typename Descriptor, typename ShT>
inline auto end_broadcast(Core& t, const Descriptor& descriptor, ShT&& shape){
    auto size = make_size(shape);
    return create_broadcast_iterator<TraverseOrder>(t, descriptor, std::forward<ShT>(shape), size);
}

//create reverse broadcast iterator
//non const
template<typename TraverseOrder, typename Core, typename Descriptor, typename ShT, typename IdxT>
inline auto create_reverse_broadcast_iterator(Core& t, const Descriptor& descriptor, ShT&& shape, const IdxT& pos){
    ASSERT_ORDER(TraverseOrder);
    using config_type = typename Core::config_type;
    using dim_type = typename config_type::dim_type;
    dim_type max_dim = std::max(descriptor.dim(), detail::make_dim(shape));
    auto strides_div = make_strides_div<config_type>(shape, TraverseOrder{});
    using walker_type = decltype(create_walker(t,descriptor,max_dim));
    using traverser_type = gtensor::walker_random_access_traverser<gtensor::walker_bidirectional_traverser<gtensor::walker_forward_traverser<config_type,walker_type>>,TraverseOrder>;
    return reverse_broadcast_iterator<config_type,traverser_type>{
        create_walker(t,descriptor,max_dim),
        std::forward<ShT>(shape),
        std::move(strides_div),
        pos
    };
}
template<typename TraverseOrder, typename Core, typename Descriptor, typename ShT>
inline auto rbegin_broadcast(Core& t, const Descriptor& descriptor, ShT&& shape){
    auto size = make_size(shape);
    return create_reverse_broadcast_iterator<TraverseOrder>(t, descriptor, std::forward<ShT>(shape), size);
}
template<typename TraverseOrder, typename Core, typename Descriptor, typename ShT>
inline auto rend_broadcast(Core& t, const Descriptor& descriptor, ShT&& shape){
    using config_type = typename Core::config_type;
    using index_type = typename config_type::index_type;
    return create_reverse_broadcast_iterator<TraverseOrder>(t, descriptor, std::forward<ShT>(shape),index_type{0});
}

//is_trivial
template<typename Core>
inline bool is_trivial(Core& t){
    if constexpr (has_callable_is_trivial<Core>::value){
        return t.is_trivial();
    }else{
        return false;
    }
}

//element
template<typename Core, std::size_t...I, typename...IdxT>
inline auto element_index(Core& t, std::index_sequence<I...>, const IdxT&...idx){
    if constexpr (sizeof...(IdxT)==0){
        return 0;
    }else{
        auto st_it = t.descriptor().strides().begin();
        return ((idx**(st_it+I))+...);
    }
}
template<typename Core, std::size_t...I, typename...IdxT>
inline decltype(auto) element_helper(Core& t, std::index_sequence<I...> seq, const IdxT&...idx){
    if constexpr (has_callable_create_walker<Core>::value){
        auto w = t.create_walker();
        (w.walk(I,idx),...);
        return *w;
    }else{
        const auto i = element_index(t,seq,idx...);
        if constexpr (has_callable_random_access_iterator<Core>::value){
            return *(t.begin()+i);
        }else if constexpr (has_callable_subscript_operator<Core>::value){
            return t[i];
        }else if constexpr (has_callable_create_indexer<Core>::value){
            return t.create_indexer()[i];
        }else if constexpr (has_callable_iterator<Core>::value){
            return *std::next(t.begin(),i);
        }else{
            static_assert(detail::always_false<Core>,"can't make data accessor");
        }
    }
}
template<typename Core, typename...IdxT>
inline decltype(auto) element(Core& t, const IdxT&...idx){
    if constexpr (has_callable_element<Core>::value){
        return t.element(idx...);
    }else{
        return element_helper(t,std::make_index_sequence<sizeof...(IdxT)>{},idx...);
    }
}

}   //end of namespace detail

//Core must provide interface to access data and meta-data:
//descriptor() method for meta-data
//create_indexer() or create_walker() or both for data
//if Core provide iterators they are used, if not iterators are made using selected data accessor i.e. indexer or walker
template<typename Core>
class tensor_implementation
{
    using core_type = Core;
public:
    using order = typename core_type::order;
    using config_type = typename core_type::config_type;
    using value_type = typename core_type::value_type;
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    template<typename...> struct forward_args : std::true_type{};
    template<typename U> struct forward_args<U> : std::bool_constant<!std::is_same_v<U,tensor_implementation>>{};

    template<typename...Args, std::enable_if_t<forward_args<Args...>::value,int> =0>
    explicit tensor_implementation(Args&&...args):
        core_(std::forward<Args>(args)...)
    {}

    tensor_implementation(const tensor_implementation&) = delete;
    tensor_implementation& operator=(const tensor_implementation&) = delete;
    tensor_implementation(tensor_implementation&&) = delete;
    tensor_implementation& operator=(tensor_implementation&&) = delete;

    auto clone()const{
        return std::make_shared<tensor_implementation>(core_);
    }

    //meta-data interface
    const auto& descriptor()const{
        return core_.descriptor();
    }
    index_type size()const{
        return descriptor().size();
    }
    bool empty()const{
        return size() == index_type{0};
    }
    dim_type dim()const{
        return descriptor().dim();
    }
    const shape_type& shape()const{
        return descriptor().shape();
    }
    const shape_type& strides()const{
        return descriptor().strides();
    }

    //data interface
    template<typename Order>
    auto begin(){
        return detail::begin<Order>(core_,descriptor());
    }
    template<typename Order>
    auto end(){
        return detail::end<Order>(core_,descriptor());
    }
    template<typename Order>
    auto rbegin(){
        return detail::rbegin<Order>(core_,descriptor());
    }
    template<typename Order>
    auto rend(){
        return detail::rend<Order>(core_,descriptor());
    }
    template<typename Order, typename Container>
    auto begin(Container&& shape){
        return detail::begin_broadcast<Order>(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Order, typename Container>
    auto end(Container&& shape){
        return detail::end_broadcast<Order>(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Order, typename Container>
    auto rbegin(Container&& shape){
        return detail::rbegin_broadcast<Order>(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Order, typename Container>
    auto rend(Container&& shape){
        return detail::rend_broadcast<Order>(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Order>
    auto create_indexer(){
        return detail::create_indexer<Order>(core_,descriptor());
    }
    auto create_walker(dim_type max_dim){
        return detail::create_walker(core_,descriptor(),max_dim);
    }
    auto create_walker(){
        return detail::create_walker(core_,descriptor());
    }
    auto data(){
        return core_.data();
    }
    template<typename...IdxT>
    decltype(auto) element(const IdxT&...idx){
        return detail::element(core_,idx...);
    }

    //const data interface
    template<typename Order>
    auto begin()const{
        return detail::begin<Order>(core_,descriptor());
    }
    template<typename Order>
    auto end()const{
        return detail::end<Order>(core_,descriptor());
    }
    template<typename Order>
    auto rbegin()const{
        return detail::rbegin<Order>(core_,descriptor());
    }
    template<typename Order>
    auto rend()const{
        return detail::rend<Order>(core_,descriptor());
    }
    template<typename Order, typename Container>
    auto begin(Container&& shape)const{
        return detail::begin_broadcast<Order>(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Order, typename Container>
    auto end(Container&& shape)const{
        return detail::end_broadcast<Order>(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Order, typename Container>
    auto rbegin(Container&& shape)const{
        return detail::rbegin_broadcast<Order>(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Order, typename Container>
    auto rend(Container&& shape)const{
        return detail::rend_broadcast<Order>(core_,descriptor(),detail::make_shape_of_type<shape_type>(std::forward<Container>(shape)));
    }
    template<typename Order>
    auto create_indexer()const{
        return detail::create_indexer<Order>(core_,descriptor());
    }
    auto create_walker(dim_type max_dim)const{
        return detail::create_walker(core_,descriptor(),max_dim);
    }
    auto create_walker()const{
        return detail::create_walker(core_,descriptor());
    }
    auto data()const{
        return core_.data();
    }
    template<typename...IdxT>
    decltype(auto) element(const IdxT&...idx)const{
        return detail::element(core_,idx...);
    }

    //trivial data interface
    template<typename Order>
    auto begin_trivial(){
        return detail::begin_trivial<Order>(core_,descriptor());
    }
    template<typename Order>
    auto end_trivial(){
        return detail::end_trivial<Order>(core_,descriptor());
    }
    template<typename Order>
    auto rbegin_trivial(){
        return detail::rbegin_trivial<Order>(core_,descriptor());
    }
    template<typename Order>
    auto rend_trivial(){
        return detail::rend_trivial<Order>(core_,descriptor());
    }
    template<typename Order>
    auto create_trivial_indexer(){
        return detail::create_trivial_indexer<Order>(core_,descriptor());
    }
    auto create_trivial_walker(){
        return detail::create_trivial_walker(core_,descriptor());
    }

    //trivial const data interface
    template<typename Order>
    auto begin_trivial()const{
        return detail::begin_trivial<Order>(core_,descriptor());
    }
    template<typename Order>
    auto end_trivial()const{
        return detail::end_trivial<Order>(core_,descriptor());
    }
    template<typename Order>
    auto rbegin_trivial()const{
        return detail::rbegin_trivial<Order>(core_,descriptor());
    }
    template<typename Order>
    auto rend_trivial()const{
        return detail::rend_trivial<Order>(core_,descriptor());
    }
    template<typename Order>
    auto create_trivial_indexer()const{
        return detail::create_trivial_indexer<Order>(core_,descriptor());
    }
    auto create_trivial_walker()const{
        return detail::create_trivial_walker(core_,descriptor());
    }

    //is_trivial
    bool is_trivial()const{
        return detail::is_trivial(core_);
    }

private:
    core_type core_;
};

}   //end of namespace gtensor
#endif