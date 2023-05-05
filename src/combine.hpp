#ifndef COMBINE_HPP_
#define COMBINE_HPP_

#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include "common.hpp"
#include "tensor_init_list.hpp"
#include "type_selector.hpp"
#include "tensor.hpp"

namespace gtensor{

class combine_exception : public std::runtime_error{
public:
    explicit combine_exception(const char* what):
        runtime_error(what)
    {}
};

namespace detail{

template<typename T, typename = void> constexpr inline bool is_tensor_container_v = false;
template<typename T> constexpr inline bool is_tensor_container_v<T, std::void_t<std::enable_if_t<is_container_v<T>>>> = !is_tensor_v<T> && is_tensor_v<typename T::value_type>;

template<typename> constexpr inline std::size_t nested_tuple_depth_v = 0;
template<typename T, typename...Ts> constexpr inline std::size_t nested_tuple_depth_v<std::tuple<T,Ts...>> = nested_tuple_depth_v<T>+1;

template<typename T> constexpr inline bool is_tensor_nested_tuple_helper_v = is_tensor_v<T>;
template<typename T> constexpr inline bool is_tensor_nested_tuple_helper_v<std::tuple<T>> = is_tensor_nested_tuple_helper_v<T>;
template<typename T, typename...Ts> constexpr inline bool is_tensor_nested_tuple_helper_v<std::tuple<T, Ts...>> =
    ((nested_tuple_depth_v<T> == nested_tuple_depth_v<Ts>)&&...) && is_tensor_nested_tuple_helper_v<T> && (is_tensor_nested_tuple_helper_v<Ts>&&...);

template<typename T> constexpr inline bool is_tensor_nested_tuple_v = false;
template<typename...Ts> constexpr inline bool is_tensor_nested_tuple_v<std::tuple<Ts...>> = is_tensor_nested_tuple_helper_v<std::tuple<Ts...>>;

template<typename T>
struct tensor_nested_tuple_config_type
{
    using type = typename T::config_type;
};
template<typename T, typename...Ts>
struct tensor_nested_tuple_config_type<std::tuple<T, Ts...>>
{
    using type = typename tensor_nested_tuple_config_type<T>::type;
};
template<typename T> using tensor_nested_tuple_config_type_t = typename tensor_nested_tuple_config_type<T>::type;


template<typename SizeT, typename ShT, typename...ShTs>
void check_stack_variadic_args(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using dim_type = SizeT;
    dim_type dim = shape.size();
    if (direction > dim){
        throw combine_exception{"bad stack direction"};
    }
    if constexpr (sizeof...(ShTs) > 0){
        if (!((shape==shapes)&&...)){
            throw combine_exception{"tensors to stack must have equal shapes"};
        }
    }
}

template<typename SizeT, typename Container>
void check_stack_container_args(const SizeT& direction, const Container& shapes){
    using dim_type = SizeT;
    if (std::empty(shapes)){
        throw combine_exception("stack empty container");
    }
    auto it = shapes.begin();
    const auto& first_shape = *it;
    dim_type first_dim = first_shape.size();
    if (direction > first_dim){
        throw combine_exception{"bad stack direction"};
    }
    for(++it; it!=shapes.end(); ++it){
        if (first_shape != *it){
            throw combine_exception{"tensors to stack must have equal shapes"};
        }
    }
}

template<typename SizeT, typename ShT, typename...ShTs>
void check_concatenate_variadic_args(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using dim_type = SizeT;
    dim_type dim = shape.size();
    if (direction >= dim){
        throw combine_exception{"bad concatenate direction"};
    }
    if constexpr (sizeof...(ShTs) > 0){
        if (!((dim==static_cast<dim_type>(shapes.size()))&&...)){
            throw combine_exception{"tensors to concatenate must have equal dimentions number"};
        }
        for (dim_type d{0}; d!=dim; ++d){
            if (!((shape[d]==shapes[d])&&...)){
                if (d!=direction){
                    throw combine_exception{"tensors to concatenate must have equal shapes"};
                }
            }
        }
    }
}

template<typename SizeT, typename ShT, typename...ShTs>
void check_concatenate_variadic_args(const SizeT& direction, const std::tuple<ShT, ShTs...>& shapes){
    std::apply([&direction](const auto&...shapes_){check_concatenate_variadic_args(direction,shapes_...);},shapes);
}

template<typename SizeT, typename Container>
void check_concatenate_container_args(const SizeT& direction, const Container& shapes){
    using dim_type = SizeT;
    if (shapes.empty()){
        throw combine_exception{"nothing to concatenate"};
    }
    auto it = shapes.begin();
    const auto& first_shape = *it;
    const dim_type& first_dim = first_shape.size();
    if (direction >= first_dim){
        throw combine_exception{"bad concatenate direction"};
    }
    for(++it; it!=shapes.end(); ++it){
        const auto& shape = *it;
        if (first_dim!=shape.size()){
            throw combine_exception("tensors to concatenate must have equal dimensions number");
        }
        for (dim_type d{0}; d!=first_dim; ++d){
            if (first_shape[d]!=shape[d]){
                if (d!=direction){
                    throw combine_exception{"tensors to concatenate must have equal shapes"};
                }
            }
        }
    }
}

template<typename...Ts>
void check_split_by_points_args(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::dim_type& direction){
    if (direction >= t.dim()){
        throw combine_exception("invalid split direction");
    }
}

template<typename...Ts>
void check_split_by_equal_parts_args(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::dim_type& direction, const typename basic_tensor<Ts...>::index_type& parts_number){
    using index_type = typename basic_tensor<Ts...>::index_type;
    if (direction >= t.dim()){
        throw combine_exception("invalid split direction");
    }
    if (parts_number == index_type{0} || t.shape()[direction] % parts_number != index_type{0}){
        throw combine_exception("can't split in equal parts");
    }
}

template<typename...Ts>
void check_vsplit_args(const basic_tensor<Ts...>& t){
    using dim_type = typename basic_tensor<Ts...>::dim_type;
    if (t.dim() < dim_type{2}){
        throw combine_exception("vsplit works only for 2 or more dimensions");
    }
}

template<typename Container>
auto make_shapes_container(const Container& ts){
    using tensor_type = typename Container::value_type;
    using config_type = typename tensor_type::config_type;
    using shape_type = typename tensor_type::shape_type;
    typename config_type::template container<shape_type> shapes{};
    shapes.reserve(ts.size());
    std::for_each(ts.begin(), ts.end(), [&shapes](const auto& t)mutable{shapes.push_back(t.shape());});
    return shapes;
}

template<typename Container>
auto make_iterators_container(const Container& ts){
    using tensor_type = typename Container::value_type;
    using config_type = typename tensor_type::config_type;
    using iterator_type = decltype(std::declval<const tensor_type>().begin());
    typename config_type::template container<iterator_type> iterators{};
    iterators.reserve(ts.size());
    std::for_each(ts.begin(), ts.end(), [&iterators](const auto& t)mutable{iterators.push_back(t.begin());});
    return iterators;
}

template<typename SizeT, typename ShT>
auto make_stack_shape(const SizeT& direction, const ShT& shape, const typename ShT::value_type& tensors_number){
    using dim_type = SizeT;
    using shape_type = ShT;
    dim_type dim = shape.size();
    shape_type res(dim+dim_type{1});
    std::copy(shape.begin(), shape.begin()+direction, res.begin());
    std::copy(shape.begin()+direction, shape.end(), res.begin()+direction+dim_type{1});
    res[direction] = tensors_number;
    return res;
}

template<typename SizeT, typename ShT, typename...ShTs>
auto make_concatenate_variadic_shape(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    index_type direction_size{shape[direction]};
    ((direction_size+=shapes[direction]),...);
    shape_type res{shape};
    res[direction]=direction_size;
    return res;
}

template<typename SizeT, typename ShT, typename...ShTs>
auto make_concatenate_variadic_shape(const SizeT& direction, const std::tuple<ShT, ShTs...>& shapes){
    return std::apply([&direction](const auto&...shapes_){return make_concatenate_variadic_shape(direction,shapes_...);},shapes);
}

template<typename SizeT, typename Container>
auto make_concatenate_container_shape(const SizeT& direction, const Container& shapes){
    using shape_type = typename Container::value_type;
    using index_type = typename shape_type::value_type;
    auto it = shapes.begin();
    const auto& first_shape = *it;
    shape_type res{first_shape};
    index_type direction_size{first_shape[direction]};
    for(++it; it!=shapes.end(); ++it){
        direction_size+=(*it)[direction];
    }
    res[direction] = direction_size;
    return res;
}

template<typename SizeT, typename ShT>
auto make_stack_chunk_size(const SizeT& direction, const ShT& shape){
    using index_type = typename ShT::value_type;
    return std::accumulate(shape.begin()+direction, shape.end(), index_type{1}, std::multiplies{});
}

template<typename SizeT, typename ShT, typename ResultIt, typename...It>
auto fill_stack(const SizeT& direction, const ShT& shape, const typename ShT::value_type& size, ResultIt res_it, It...it){
    using index_type = typename ShT::value_type;
    const index_type chunk_size = make_stack_chunk_size(direction, shape);
    auto filler = [chunk_size, res_it](auto& it) mutable {
        for (index_type i{0}; i!=chunk_size; ++i, ++res_it, ++it){
            *res_it = *it;
        }
    };
    const index_type iterations_number = size/chunk_size;
    for (index_type i{0}; i!=iterations_number; ++i){
        (filler(it),...);
    }
}

template<typename SizeT, typename ShT, typename ResultIt, typename ItContainer>
auto fill_stack_container(const SizeT& direction, const ShT& shape, ResultIt res_it, ItContainer& iterators){
    using index_type = typename ShT::value_type;
    const index_type chunk_size = make_stack_chunk_size(direction, shape);
    const index_type iterations_number = std::accumulate(shape.begin(), shape.begin()+direction, index_type{1}, std::multiplies{});
    if (chunk_size == index_type{1}){
        for (index_type i{0}; i!=iterations_number; ++i){
            for (auto it=iterators.begin(); it!=iterators.end(); ++it){
                auto& iterator = *it;
                *res_it = *iterator;
                ++iterator;
                ++res_it;
            }
        }
    }else{
        for (index_type i{0}; i!=iterations_number; ++i){
            for (auto it=iterators.begin(); it!=iterators.end(); ++it){
                auto& iterator = *it;
                for (index_type j{0}; j!=chunk_size; ++j,++iterator,++res_it){
                    *res_it = *iterator;
                }
            }
        }
    }
}

template<typename SizeT, typename ShT, typename...ShTs>
auto make_concatenate_chunk_size(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using dim_type = SizeT;
    using index_type = typename ShT::value_type;
    index_type chunk_size_ = std::accumulate(shape.begin()+direction+dim_type{1}, shape.end(), index_type{1}, std::multiplies{});
    return std::make_tuple(shape[direction]*chunk_size_, shapes[direction]*chunk_size_...);
}

template<typename SizeT, typename ShT, typename...ShTs>
auto make_concatenate_chunk_size(const SizeT& direction, const std::tuple<ShT, ShTs...>& shapes){
    return std::apply([&direction](const auto&...shapes_){return make_concatenate_chunk_size(direction,shapes_...);},shapes);
}

template<typename SizeT, typename ShT, typename...ShTs, typename ResultIt, std::size_t...I, typename...It>
auto fill_concatenate(const SizeT& direction, const std::tuple<ShT, ShTs...>& shapes, ResultIt res_it, std::index_sequence<I...>, It...it){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;

    const auto chunk_size = make_concatenate_chunk_size(direction, shapes);
    const auto first_shape = std::get<0>(shapes);
    auto filler = [res_it](const auto& chunk_size_, auto& it)mutable{
        for (index_type i{0}; i!=chunk_size_; ++i,++it,++res_it){
            *res_it = *it;
        }
    };
    index_type iterations_number = std::accumulate(first_shape.begin(), first_shape.begin()+direction, index_type{1}, std::multiplies{});
    for (index_type i{0}; i!=iterations_number; ++i){
        (filler(std::get<I>(chunk_size),it),...);
    }
}

template<typename SizeT, typename ShapeContainer, typename ResIt, typename TensorContainer>
auto fill_concatenate_container(const SizeT& direction, const ShapeContainer& shapes, ResIt res_it, const TensorContainer& ts){
    using tensor_type = typename TensorContainer::value_type;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using config_type = typename tensor_type::config_type;
    //0chunk_size,1iterator
    using ts_internals_type = std::tuple<index_type, decltype((*ts.begin()).begin())>;

    typename config_type::template container<ts_internals_type> ts_internals{};
    ts_internals.reserve(ts.size());
    auto shapes_it = shapes.begin();
    const auto& first_shape = *shapes_it;
    //the same part for all shapes
    const index_type chunk_size_ = std::accumulate(first_shape.begin()+direction+dim_type{1}, first_shape.end(), index_type{1}, std::multiplies{});
    for (auto ts_it = ts.begin(); ts_it!=ts.end(); ++ts_it,++shapes_it){
        index_type chunk_size = chunk_size_*(*shapes_it)[direction];
        ts_internals.emplace_back(chunk_size, (*ts_it).begin());
    }
    index_type iterations_number = std::accumulate(first_shape.begin(), first_shape.begin()+direction, index_type{1}, std::multiplies{});
    for (index_type j = 0; j!=iterations_number; ++j){
        for (auto ts_internals_it = ts_internals.begin(); ts_internals_it!=ts_internals.end(); ++ts_internals_it){
            auto chunk_size = std::get<0>(*ts_internals_it);
            auto& it = std::get<1>(*ts_internals_it);
            for (index_type i{0}; i!=chunk_size; ++i, ++it, ++res_it){
                *res_it = *it;
            }
        }
    }
}

//add leading ones to shape to make dim new_dim, if dim>= new_dim do nothing
//returns new shape
template<typename ShT, typename SizeT>
auto widen_shape(const ShT& shape, const SizeT& new_dim){
    using dim_type = SizeT;
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    dim_type dim = shape.size();
    if (dim < new_dim){
        shape_type res(new_dim, index_type{1});
        std::copy(shape.rbegin(),shape.rend(),res.rbegin());
        return res;
    }else{
        return shape;
    }
}
template<typename...Ts>
auto widen_tensor(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::dim_type& new_dim){
    return t.reshape(widen_shape(t.shape(),new_dim));
}

}   //end of namespace detail

class combiner
{
//join tensors along new direction, tensors must have the same shape
//variadic
template<typename SizeT, typename...Us, typename...Ts>
static auto stack_variadic(const SizeT& direction, const basic_tensor<Us...>& t, const Ts&...ts){
    using tensor_type = basic_tensor<Us...>;
    using config_type = typename tensor_type::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = std::common_type_t<typename tensor_type::value_type, typename Ts::value_type...>;

    const auto& shape = t.shape();
    detail::check_stack_variadic_args(direction, shape, ts.shape()...);
    constexpr auto tensors_number = sizeof...(Ts) + 1;
    auto res_shape = detail::make_stack_shape(direction, shape, index_type{tensors_number});
    if constexpr (tensors_number == 1){
        return tensor<res_value_type, config_type>(std::move(res_shape), t.begin(), t.end());
    }else{
        auto res = tensor<res_value_type, config_type>(std::move(res_shape), res_value_type{});
        if (!res.empty()){
            detail::fill_stack(direction, shape, t.size(), res.begin(), t.begin(), ts.begin()...);
        }
        return res;
    }
}
template<typename SizeT, typename Container>
static auto stack_container(const SizeT& direction, const Container& ts){
    using tensor_type = typename Container::value_type;
    using config_type = typename tensor_type::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = typename tensor_type::value_type;

    const auto shapes = detail::make_shapes_container(ts);
    detail::check_stack_container_args(direction, shapes);
    index_type tensors_number = static_cast<index_type>(ts.size());
    const auto& shape = *shapes.begin();
    auto res_shape = detail::make_stack_shape(direction, shape, tensors_number);
    if (tensors_number == index_type{1}){
        const auto& t = *ts.begin();
        return tensor<res_value_type, config_type>(std::move(res_shape), t.begin(), t.end());
    }else{
        auto res = tensor<res_value_type, config_type>(std::move(res_shape), res_value_type{});
        if (!res.empty()){
            auto iterators = detail::make_iterators_container(ts);
            detail::fill_stack_container(direction, shape, res.begin(), iterators);
        }
        return res;
    }
}

//join tensors along existing direction, tensors must have the same shape except concatenate direction
//variadic
template<typename SizeT, typename...Us, typename...Ts>
static auto concatenate_variadic(const SizeT& direction, const basic_tensor<Us...>& t, const Ts&...ts){
    using tensor_type = basic_tensor<Us...>;
    using config_type = typename tensor_type::config_type;
    using res_value_type = std::common_type_t<typename tensor_type::value_type, typename Ts::value_type...>;

    auto shapes = std::make_tuple(t.shape(), ts.shape()...);
    detail::check_concatenate_variadic_args(direction, shapes);
    auto res_shape = detail::make_concatenate_variadic_shape(direction, shapes);
    constexpr auto tensors_number = sizeof...(Ts) + 1;
    if constexpr (tensors_number == 1){
        return tensor<res_value_type, config_type>(std::move(res_shape),t.begin(),t.end());
    }else{
        auto res = tensor<res_value_type, config_type>(std::move(res_shape), res_value_type{});
        if (!res.empty()){
            detail::fill_concatenate(direction, shapes, res.begin(), std::make_index_sequence<tensors_number>{}, t.begin(), ts.begin()...);
        }
        return res;
    }
}
//container
template<typename SizeT, typename Container>
static auto concatenate_container(const SizeT& direction, const Container& ts){
    using tensor_type = typename Container::value_type;
    using config_type = typename tensor_type::config_type;
    using res_value_type = typename tensor_type::value_type;

    const auto shapes = detail::make_shapes_container(ts);
    detail::check_concatenate_container_args(direction, shapes);
    auto res = tensor<res_value_type, config_type>(detail::make_concatenate_container_shape(direction, shapes), res_value_type{});
    if (!res.empty()){
        detail::fill_concatenate_container(direction,shapes,res.begin(),ts);
    }
    return res;
}
//vstack - concatenate along 0 direction, reshapes 1-d tensors by adding leading 1 (n) -> (1,n)
template<typename...Us, typename...Ts>
static auto vstack_variadic(const basic_tensor<Us...>& t, const Ts&...ts){
    using tensor_type = basic_tensor<Us...>;
    using dim_type = typename tensor_type::dim_type;
    const dim_type direction{0};
    const dim_type min_dim{2};
    if (t.dim()==dim_type{1} || ((ts.dim()==dim_type{1})||...)){
        return concatenate(direction, detail::widen_tensor(t,min_dim), detail::widen_tensor(ts,min_dim)...);
    }else{
        return concatenate(direction,t,ts...);
    }
}
template<typename Container>
static auto vstack_container(const Container& ts){
    using tensor_type = typename Container::value_type;
    using dim_type = typename tensor_type::dim_type;
    using config_type = typename tensor_type::config_type;
    bool need_reshape{false};
    const dim_type direction{0};
    const dim_type min_dim{2};
    for (auto it = ts.begin(); it!=ts.end(); ++it){
        if ((*it).dim() == dim_type{1}){
            need_reshape = true;
            break;
        }
    }
    if (need_reshape){
        using view_type = decltype(detail::widen_tensor(*ts.begin(),min_dim));
        typename config_type::template container<view_type> ts_{};
        ts_.reserve(ts.size());
        for(auto it = ts.begin(); it!=ts.end(); ++it){
            ts_.push_back(detail::widen_tensor(*it,min_dim));
        }
        return concatenate_container(direction, ts_);
    }else{
        return concatenate_container(direction, ts);
    }
}
//hstack - concatenate along 1 direction, 1-d tensors along 0 direction
template<typename...Us, typename...Ts>
static auto hstack_variadic(const basic_tensor<Us...>& t, const Ts&...ts){
    using tensor_type = basic_tensor<Us...>;
    using dim_type = typename tensor_type::dim_type;
    const dim_type direction = t.dim()==dim_type{1} ? dim_type{0}:dim_type{1};
    return concatenate_variadic(direction, t, ts...);
}
template<typename Container>
static auto hstack_container(const Container& ts){
    using tensor_type = typename Container::value_type;
    using dim_type = typename tensor_type::dim_type;
    const dim_type direction = (*ts.begin()).dim()==dim_type{1} ? dim_type{0}:dim_type{1};
    return concatenate_container(direction, ts);
}

//assemble tensor from blocks
//blocks in nested tuples
template<typename...Us, typename...Ts>
static auto block_tuple_helper(std::size_t depth, const basic_tensor<Us...>& t, const Ts&...ts){
    using tensor_type = basic_tensor<Us...>;
    using dim_type = typename tensor_type::dim_type;
    const dim_type depth_ = static_cast<dim_type>(depth);
    const dim_type max_dim = std::max({t.dim(),ts.dim()...});
    const dim_type res_dim = std::max(depth_,max_dim);
    const dim_type direction = res_dim - depth_;
    if (t.dim()!=res_dim || ((ts.dim()!=res_dim)||...)){
        return concatenate_variadic(direction, detail::widen_tensor(t,res_dim), detail::widen_tensor(ts,res_dim)...);
    }else{
        return concatenate_variadic(direction,t,ts...);
    }
}
template<typename T, typename...Ts, std::enable_if_t<detail::is_tensor_v<T> ,int> =0>
static auto block_tuple(const std::tuple<T,Ts...>& blocks){
    //depth is 1
    auto apply_blocks = [](const auto&...ts){
        return block_tuple_helper(1, ts...);
    };
    return std::apply(apply_blocks, blocks);
}
template<typename...Us, typename...Ts>
static auto block_tuple(const std::tuple<std::tuple<Us...>, Ts...>& blocks){
    std::size_t depth = detail::nested_tuple_depth_v<std::tuple<std::tuple<Us...>, Ts...>>;
    auto apply_blocks = [&depth](const auto&...blocks_){
        return block_tuple_helper(depth, block_tuple(blocks_)...);
    };
    return std::apply(apply_blocks, blocks);
}
//blocks in nested initializer list
template<typename Container>
static auto block_init_list_helper(std::size_t depth, const Container& ts){
    using tensor_type = typename Container::value_type;
    using config_type = typename tensor_type::config_type;
    using dim_type = typename tensor_type::dim_type;

    const dim_type depth_ = static_cast<dim_type>(depth);
    dim_type max_dim{0};
    for(auto it=ts.begin(); it!=ts.end(); ++it){
        max_dim = std::max(max_dim,(*it).dim());
    }
    const dim_type res_dim = std::max(depth_,max_dim);
    const dim_type direction = res_dim - depth_;
    bool need_reshape{false};
    for (auto it = ts.begin(); it!=ts.end(); ++it){
        if ((*it).dim() != res_dim){
            need_reshape = true;
            break;
        }
    }
    if (need_reshape){
        using view_type = decltype(detail::widen_tensor(*ts.begin(),res_dim));
        typename config_type::template container<view_type> ts_{};
        ts_.reserve(ts.size());
        for(auto it = ts.begin(); it!=ts.end(); ++it){
            ts_.push_back(detail::widen_tensor(*it,res_dim));
        }
        return concatenate_container(direction, ts_);
    }else{
        return concatenate_container(direction, ts);
    }
}
template<typename T, std::enable_if_t<detail::is_tensor_v<T> ,int> =0>
static auto block_init_list(std::initializer_list<T> blocks){
    return block_init_list_helper(1, blocks);
}
template<typename Nested>
static auto block_init_list(std::initializer_list<std::initializer_list<Nested>> blocks){
    using tensor_type = typename detail::nested_initialiser_list_value_type<Nested>::type;
    using config_type = typename tensor_type::config_type;
    using block_type = decltype(block_init_list(*blocks.begin()));

    std::size_t depth = detail::nested_initialiser_list_depth<decltype(blocks)>::value;
    typename config_type::template container<block_type> blocks_{};
    blocks_.reserve(blocks.size());
    for (auto it = blocks.begin(); it!=blocks.end(); ++it){
        blocks_.push_back(block_init_list(*it));
    }
    return block_init_list_helper(depth, blocks_);
}

//Split tensor and return container of slice views
//split points determined using split_points parameter that is container of split points
template<typename...Ts, typename IdxContainer>
static auto split_by_points(const basic_tensor<Ts...>& t, const IdxContainer& split_points, const typename basic_tensor<Ts...>::dim_type& direction){
    using tensor_type = basic_tensor<Ts...>;
    using config_type = typename tensor_type::config_type;
    using index_type = typename tensor_type::index_type;
    using slice_type = typename slice_traits<config_type>::slice_type;
    using view_type = decltype(t(slice_type{}));
    using res_type = typename config_type::template container<view_type>;
    using res_size_type = typename res_type::size_type;
    using slices_type = typename config_type::template container<slice_type>;
    using slices_size_type = typename slices_type::size_type;

    detail::check_split_by_points_args(t, direction);
    if (std::empty(split_points)){
        return res_type{t(slice_type{})};
    }else{
        const res_size_type parts_number = static_cast<res_size_type>(split_points.size()) + res_size_type{1};
        res_type res{};
        res.reserve(parts_number);
        auto split_points_it = std::begin(split_points);
        index_type point{0};
        slices_type slices(static_cast<slices_size_type>(t.dim()));
        auto slices_direction_it = std::next(slices.begin(),static_cast<slices_size_type>(direction));
        do{
            index_type next_point = *split_points_it;
            *slices_direction_it = slice_type{point, next_point};
            res.push_back(t(slices));
            point = next_point;
            ++split_points_it;
        }while(split_points_it != std::end(split_points));
        *slices_direction_it = slice_type{point};
        res.push_back(t(slices));
        return res;
    }
}
//split points determined by dividing tensor by equal parts
template<typename...Ts>
static auto split_equal_parts(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::index_type& parts_number, const typename basic_tensor<Ts...>::dim_type& direction){
    using tensor_type = basic_tensor<Ts...>;
    using config_type = typename tensor_type::config_type;
    using index_type = typename tensor_type::index_type;
    using slice_type = typename slice_traits<config_type>::slice_type;
    using nop_type = typename slice_traits<config_type>::nop_type;
    using view_type = decltype(t(slice_type{}));
    using res_type = typename config_type::template container<view_type>;
    using res_size_type = typename res_type::size_type;
    using slices_type = typename config_type::template container<slice_type>;
    using slices_size_type = typename slices_type::size_type;

    detail::check_split_by_equal_parts_args(t,direction,parts_number);
    const index_type direction_size = t.shape()[direction];
    const index_type part_size = direction_size/parts_number;
    index_type stop{part_size};
    res_type res{};
    res.reserve(static_cast<res_size_type>(parts_number));
    slices_type slices(static_cast<slices_size_type>(t.dim()));
    auto slices_direction_it = std::next(slices.begin(),static_cast<slices_size_type>(direction));
    *slices_direction_it = slice_type{nop_type{},stop};
    res.push_back(t(slices));
    for (index_type i{1}; i!=parts_number; ++i){
        *slices_direction_it = slice_type{stop,stop+part_size};
        res.push_back(t(slices));
        stop+=part_size;
    }
    return res;
}
//split in direction 0
template<typename...Ts, typename IdxContainer>
static auto vsplit_by_points(const basic_tensor<Ts...>& t, const IdxContainer& split_points){
    using dim_type = typename basic_tensor<Ts...>::dim_type;
    detail::check_vsplit_args(t);
    const dim_type direction{0};
    return split_by_points(t,split_points,direction);
}
template<typename...Ts>
static auto vsplit_equal_parts(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::index_type& parts_number){
    using dim_type = typename basic_tensor<Ts...>::dim_type;
    detail::check_vsplit_args(t);
    const dim_type direction{0};
    return split_equal_parts(t,parts_number,direction);
}
//split in direction 1, for 1-d split in direction 0
template<typename...Ts, typename IdxContainer>
static auto hsplit_by_points(const basic_tensor<Ts...>& t, const IdxContainer& split_points){
    using dim_type = typename basic_tensor<Ts...>::dim_type;
    const dim_type direction = t.dim() == dim_type{1} ? dim_type{0} : dim_type{1};
    return split_by_points(t,split_points,direction);
}
template<typename...Ts>
static auto hsplit_equal_parts(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::index_type& parts_number){
    using dim_type = typename basic_tensor<Ts...>::dim_type;
    const dim_type direction = t.dim() == dim_type{1} ? dim_type{0} : dim_type{1};
    return split_equal_parts(t,parts_number,direction);
}

public:
//combiner interface
template<typename...Us, typename...Ts>
static auto stack(const typename basic_tensor<Us...>::dim_type& direction, const basic_tensor<Us...>& t, const Ts&...ts){
    return stack_variadic(direction, t, ts...);
}
template<typename Container, std::enable_if_t<detail::is_tensor_container_v<Container>,int> =0>
static auto stack(const typename Container::value_type::dim_type& direction, const Container& ts){
    return stack_container(direction, ts);
}
template<typename...Us, typename...Ts>
static auto concatenate(const typename basic_tensor<Us...>::dim_type& direction, const basic_tensor<Us...>& t, const Ts&...ts){
    return concatenate_variadic(direction, t, ts...);
}
template<typename Container, std::enable_if_t<detail::is_tensor_container_v<Container>,int> =0>
static auto concatenate(const typename Container::value_type::dim_type& direction, const Container& ts){
    return concatenate_container(direction, ts);
}
template<typename...Us, typename...Ts>
static auto vstack(const basic_tensor<Us...>& t, const Ts&...ts){
    return vstack_variadic(t, ts...);
}
template<typename Container, std::enable_if_t<detail::is_tensor_container_v<Container>,int> =0>
static auto vstack(const Container& ts){
    return vstack_container(ts);
}
template<typename...Us, typename...Ts>
static auto hstack(const basic_tensor<Us...>& t, const Ts&...ts){
    return hstack_variadic(t, ts...);
}
template<typename Container, std::enable_if_t<detail::is_tensor_container_v<Container>,int> =0>
static auto hstack(const Container& ts){
    return hstack_container(ts);
}
template<typename...Ts>
static auto block(const std::tuple<Ts...>& blocks){
    return block_tuple(blocks);
}
template<typename T>
static auto block(std::initializer_list<T> blocks){
    return block_init_list(blocks);
}
template<typename T>
static auto block(std::initializer_list<std::initializer_list<T>> blocks){
    return block_init_list(blocks);
}
template<typename T>
static auto block(std::initializer_list<std::initializer_list<std::initializer_list<T>>> blocks){
    return block_init_list(blocks);
}
template<typename T>
static auto block(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> blocks){
    return block_init_list(blocks);
}
template<typename...Ts, typename IdxContainer, std::enable_if_t<detail::is_container_of_type_v<IdxContainer,typename basic_tensor<Ts...>::index_type>,int> =0>
static auto split(const basic_tensor<Ts...>& t, const IdxContainer& split_points, const typename basic_tensor<Ts...>::dim_type& direction){
    return split_by_points(t, split_points, direction);
}
template<typename...Ts>
static auto split(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::index_type> split_points, const typename basic_tensor<Ts...>::dim_type& direction){
    return split_by_points(t, split_points, direction);
}
template<typename...Ts>
static auto split(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::index_type& parts_number, const typename basic_tensor<Ts...>::dim_type& direction){
    return split_equal_parts(t, parts_number, direction);
}
template<typename...Ts, typename IdxContainer, std::enable_if_t<detail::is_container_of_type_v<IdxContainer,typename basic_tensor<Ts...>::index_type>,int> =0>
static auto vsplit(const basic_tensor<Ts...>& t, const IdxContainer& split_points){
    return vsplit_by_points(t, split_points);
}
template<typename...Ts>
static auto vsplit(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::index_type& parts_number){
    return vsplit_equal_parts(t, parts_number);
}
template<typename...Ts, typename IdxContainer, std::enable_if_t<detail::is_container_of_type_v<IdxContainer,typename basic_tensor<Ts...>::index_type>,int> =0>
static auto hsplit(const basic_tensor<Ts...>& t, const IdxContainer& split_points){
    return hsplit_by_points(t, split_points);
}
template<typename...Ts>
static auto hsplit(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::index_type& parts_number){
    return hsplit_equal_parts(t, parts_number);
}

};  //end of class combiner

//combine module interface
template<typename...Us, typename...Ts>
auto stack(const typename basic_tensor<Us...>::dim_type& direction, const basic_tensor<Us...>& t, const Ts&...ts){
    static_assert((detail::is_tensor_v<Ts>&&...));
    using config_type = typename basic_tensor<Us...>::config_type;
    return combiner_selector_t<config_type>::stack(direction, t, ts...);
}
template<typename Container, std::enable_if_t<detail::is_tensor_container_v<Container>,int> =0>
auto stack(const typename Container::value_type::dim_type& direction, const Container& ts){
    static_assert(detail::is_tensor_container_v<Container>);
    using config_type = typename Container::value_type::config_type;
    return combiner_selector_t<config_type>::stack(direction, ts);
}
template<typename...Us, typename...Ts>
auto concatenate(const typename basic_tensor<Us...>::dim_type& direction, const basic_tensor<Us...>& t, const Ts&...ts){
    static_assert((detail::is_tensor_v<Ts>&&...));
    using config_type = typename basic_tensor<Us...>::config_type;
    return combiner_selector_t<config_type>::concatenate(direction, t, ts...);
}
template<typename Container, std::enable_if_t<detail::is_tensor_container_v<Container>,int> =0>
auto concatenate(const typename Container::value_type::dim_type& direction, const Container& ts){
    static_assert(detail::is_tensor_container_v<Container>);
    using config_type = typename Container::value_type::config_type;
    return combiner_selector_t<config_type>::concatenate(direction, ts);
}
template<typename...Us, typename...Ts>
auto vstack(const basic_tensor<Us...>& t, const Ts&...ts){
    static_assert((detail::is_tensor_v<Ts>&&...));
    using tensor_type = basic_tensor<Us...>;
    using config_type = typename tensor_type::config_type;
    return combiner_selector_t<config_type>::vstack(t, ts...);
}
template<typename Container, std::enable_if_t<detail::is_tensor_container_v<Container>,int> =0>
auto vstack(const Container& ts){
    using tensor_type = typename Container::value_type;
    using config_type = typename tensor_type::config_type;
    return combiner_selector_t<config_type>::vstack(ts);
}
template<typename...Us, typename...Ts>
auto hstack(const basic_tensor<Us...>& t, const Ts&...ts){
    static_assert((detail::is_tensor_v<Ts>&&...));
    using tensor_type = basic_tensor<Us...>;
    using config_type = typename tensor_type::config_type;
    return combiner_selector_t<config_type>::hstack(t, ts...);
}
template<typename Container, std::enable_if_t<detail::is_tensor_container_v<Container>,int> =0>
auto hstack(const Container& ts){
    using tensor_type = typename Container::value_type;
    using config_type = typename tensor_type::config_type;
    return combiner_selector_t<config_type>::hstack(ts);
}
template<typename...Ts>
auto block(const std::tuple<Ts...>& blocks){
    static_assert(detail::is_tensor_nested_tuple_v<std::tuple<Ts...>>);
    using config_type = detail::tensor_nested_tuple_config_type_t<std::tuple<Ts...>>;
    return combiner_selector_t<config_type>::block(blocks);
}
template<typename T>
auto block(std::initializer_list<T> blocks){
    static_assert(detail::is_tensor_v<T>);
    using config_type = typename T::config_type;
    return combiner_selector_t<config_type>::block(blocks);
}
template<typename T>
auto block(std::initializer_list<std::initializer_list<T>> blocks){
    static_assert(detail::is_tensor_v<T>);
    using config_type = typename T::config_type;
    return combiner_selector_t<config_type>::block(blocks);
}
template<typename T>
auto block(std::initializer_list<std::initializer_list<std::initializer_list<T>>> blocks){
    static_assert(detail::is_tensor_v<T>);
    using config_type = typename T::config_type;
    return combiner_selector_t<config_type>::block(blocks);
}
template<typename T>
auto block(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> blocks){
    static_assert(detail::is_tensor_v<T>);
    using config_type = typename T::config_type;
    return combiner_selector_t<config_type>::block(blocks);
}
template<typename...Ts, typename IdxContainer, std::enable_if_t<detail::is_container_of_type_v<IdxContainer,typename basic_tensor<Ts...>::index_type>,int> =0>
auto split(const basic_tensor<Ts...>& t, const IdxContainer& split_points, const typename basic_tensor<Ts...>::dim_type& direction){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return combiner_selector_t<config_type>::split(t, split_points, direction);
}
template<typename...Ts>
auto split(const basic_tensor<Ts...>& t, std::initializer_list<typename basic_tensor<Ts...>::index_type> split_points, const typename basic_tensor<Ts...>::dim_type& direction){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return combiner_selector_t<config_type>::split(t, split_points, direction);
}
template<typename...Ts>
auto split(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::index_type& parts_number, const typename basic_tensor<Ts...>::dim_type& direction){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return combiner_selector_t<config_type>::split(t, parts_number, direction);
}
template<typename...Ts, typename IdxContainer, std::enable_if_t<detail::is_container_of_type_v<IdxContainer,typename basic_tensor<Ts...>::index_type>,int> =0>
auto vsplit(const basic_tensor<Ts...>& t, const IdxContainer& split_points){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return combiner_selector_t<config_type>::vsplit(t, split_points);
}
template<typename...Ts>
auto vsplit(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::index_type& parts_number){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return combiner_selector_t<config_type>::vsplit(t, parts_number);
}
template<typename...Ts, typename IdxContainer, std::enable_if_t<detail::is_container_of_type_v<IdxContainer,typename basic_tensor<Ts...>::index_type>,int> =0>
auto hsplit(const basic_tensor<Ts...>& t, const IdxContainer& split_points){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return combiner_selector_t<config_type>::hsplit(t, split_points);
}
template<typename...Ts>
auto hsplit(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::index_type& parts_number){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return combiner_selector_t<config_type>::hsplit(t, parts_number);
}

}   //end of namespace gtensor

#endif