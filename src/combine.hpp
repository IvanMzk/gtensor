#ifndef COMBINE_HPP_
#define COMBINE_HPP_

#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include "forward_decl.hpp"
#include "common.hpp"
#include "tensor_init_list.hpp"
#include "type_selector.hpp"
#include "tensor_factory.hpp"
#include "slice.hpp"

namespace gtensor{

class combine_exception : public std::runtime_error{
public:
    explicit combine_exception(const char* what):
        runtime_error(what)
    {}
};

namespace detail{

template<typename T, typename IdxT, typename = void> constexpr inline bool is_index_container_v = false;
template<typename T, typename IdxT> constexpr inline bool is_index_container_v<T, IdxT, std::void_t<std::enable_if_t<is_container_v<T>>>> = std::is_convertible_v<typename T::value_type,IdxT>;

template<typename T, typename = void> constexpr inline bool is_tensor_container_v = false;
template<typename T> constexpr inline bool is_tensor_container_v<T, std::void_t<std::enable_if_t<is_container_v<T>>>> = is_tensor_v<typename T::value_type>;


template<typename SizeT, typename ShT, typename...ShTs>
void check_stack_args(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using size_type = SizeT;
    size_type dim = shape.size();
    if (dim > size_type{0} && direction > dim){
        throw combine_exception{"bad stack direction"};
    }
    if constexpr (sizeof...(ShTs) > 0){
        if (!((shape==shapes)&&...)){
            throw combine_exception{"tensors to stack must have equal shapes"};
        }
    }
}

template<typename SizeT, typename ShT, typename...ShTs>
void check_concatenate_args(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using size_type = SizeT;
    size_type dim = shape.size();
    if (dim > size_type{0} && direction >= dim){
        throw combine_exception{"bad concatenate direction"};
    }
    if constexpr (sizeof...(ShTs) > 0){
        if (!((dim==static_cast<size_type>(shapes.size()))&&...)){
            throw combine_exception{"tensors to concatenate must have equal shapes"};
        }
        for (size_type d{0}; d!=dim; ++d){
            if (!((shape[d]==shapes[d])&&...)){
                if (d!=direction){
                    throw combine_exception{"tensors to concatenate must have equal shapes"};
                }
            }
        }
    }
}

template<typename SizeT, typename Container>
void check_concatenate_container_args(const SizeT& direction, const Container& ts){
    auto it = ts.begin();
    auto first_dim = (*it).dim();
    std::for_each(++it, ts.end(),
        [&first_dim](const auto& t){
            if (first_dim!=t.dim()){
                throw combine_exception("tensors to concatenate must have equal shapes");
            }
        }
    );
    if (first_dim > SizeT{0} && direction > first_dim){
        throw combine_exception{"bad concatenate direction"};
    }
}

template<typename SizeT, typename ShT>
auto make_stack_shape(const SizeT& direction, const ShT& shape, const typename ShT::value_type& tensors_number){
    using size_type = SizeT;
    using shape_type = ShT;
    size_type pdim = shape.size();
    if (pdim == size_type{0}){
        return shape_type{};
    }else{
        shape_type res(pdim+size_type{1});
        std::copy(shape.begin(), shape.begin()+direction, res.begin());
        std::copy(shape.begin()+direction, shape.end(), res.begin()+direction+size_type{1});
        res[direction] = tensors_number;
        return res;
    }
}

template<typename SizeT, typename ShT, typename...ShTs>
auto make_concatenate_shape(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    using size_type = SizeT;
    size_type dim = shape.size();
    if (dim == size_type{0}){
        return shape_type{};
    }else{
        index_type direction_size{shape[direction]};
        ((direction_size+=shapes[direction]),...);
        shape_type res{shape};
        res[direction]=direction_size;
        return res;
    }
}

template<typename SizeT, typename ShT>
auto make_stack_block_size(const SizeT& direction, const ShT& shape){
    using size_type = SizeT;
    using index_type = typename ShT::value_type;
    size_type pdim = shape.size();
    if (pdim == size_type{0}){
        return index_type{0};
    }else{
        return std::accumulate(shape.begin()+direction, shape.end(), index_type{1}, std::multiplies{});
    }
}

template<typename SizeT, typename ShT, typename ResultIt, typename...It>
auto fill_stack(const SizeT& direction, const ShT& shape, const typename ShT::value_type& size, ResultIt res_it, It...it){
    using index_type = typename ShT::value_type;
    index_type block_size = make_stack_block_size(direction, shape);
    index_type i{0};
    auto filler = [i, block_size, res_it](auto& it) mutable {
        auto i_ = i;
        i += block_size;
        for (;i_!=i; ++i_, ++res_it, ++it){
            *res_it = *it;
        }
    };
    index_type iterations_number = size/block_size;
    for (;i!=iterations_number; ++i){
        (filler(it),...);
    }
}

template<typename SizeT, typename ShT, typename...ShTs>
auto make_concatenate_chunk_size(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using size_type = SizeT;
    using index_type = typename ShT::value_type;
    index_type chunk_size_ = std::accumulate(shape.begin()+direction+size_type{1}, shape.end(), index_type{1}, std::multiplies{});
    return std::make_tuple(shape[direction]*chunk_size_, shapes[direction]*chunk_size_...);
}

template<typename SizeT, typename IdxT, typename ResultIt, typename ImplT, typename...ImplTs>
class concatenate_filler
{
    using index_type = IdxT;
    using chunk_size_type = decltype(make_concatenate_chunk_size(std::declval<SizeT>(), std::declval<ImplT>().shape(), std::declval<ImplTs>().shape()...));
    chunk_size_type chunk_sizes_;
    ResultIt res_it_;
public:
    concatenate_filler(const SizeT& direction__, ResultIt res_it__, const ImplT& t__, const ImplTs&...ts__):
        chunk_sizes_{make_concatenate_chunk_size(direction__, t__.shape(), ts__.shape()...)},
        res_it_{res_it__}
    {}
    template<std::size_t I, typename It>
    void fill(It& it){
        index_type chunk_size = std::get<I>(chunk_sizes_);
        for (index_type i{0}; i!=chunk_size; ++i, ++it, ++res_it_){
            *res_it_ = *it;
        }
    }
};

template<typename SizeT, typename ResultIt, std::size_t...I, typename ImplT, typename...ImplTs>
auto fill_concatenate(const SizeT& direction,  ResultIt res_it, std::index_sequence<I...>, const ImplT& t, const ImplTs&...ts){
    using shape_type = std::decay_t<decltype(t.shape())>;
    using index_type = typename shape_type::value_type;
    using filler_type = concatenate_filler<SizeT, index_type, ResultIt, ImplT, ImplTs...>;
    auto iters = std::make_tuple(t.engine().begin(), ts.engine().begin()...);
    filler_type filler{direction, res_it, t, ts...};
    index_type iterations_number = std::accumulate(t.shape().begin(), t.shape().begin()+direction, index_type{1}, std::multiplies{});
    for (index_type i{0}; i!=iterations_number; ++i){
        ((filler.template fill<I>(std::get<I>(iters))),...);
    }
}

//returns dim of result of block
template<typename T>
auto max_block_dim(const T& t){
    return t.dim();
}
template<typename Nested>
auto max_block_dim(std::initializer_list<Nested> blocks){
    using tensor_type = typename nested_initialiser_list_value_type<std::initializer_list<Nested>>::type;
    using size_type = typename tensor_type::size_type;
    size_type dim{0};
    for(auto it = blocks.begin(); it!=blocks.end(); ++it){
        dim = std::max(max_block_dim(*it), dim);
    }
    return dim;
}

//add leading ones to shape to make dim new_dim, if dim>= new_dim do nothing
//returns result shape dim
template<typename ShT, typename SizeT>
auto widen_shape(ShT& shape, const SizeT& new_dim){
    using size_type = SizeT;
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    size_type dim = shape.size();
    if (dim < new_dim){
        auto ones_number = new_dim - dim;
        for (size_type i{0}; i!=ones_number; ++i){
            shape.push_back(index_type{1});
        }
        std::rotate(shape.begin(), shape.begin()+dim, shape.end());
    }
    return shape.size();
}

//concatenates shapes and accumulates result in res_shape
//depth means direction to concatenate, 1 is the last direction with lowest stride, 2 is next to last direction..., so direction is inverted
//res_shape initialized with first shape for current depth
template<typename ShT, typename SizeT>
auto make_block_shape_helper(ShT& res_shape, const ShT& shape, const SizeT& depth){
    using size_type = SizeT;
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    size_type res_dim = res_shape.size();
    const size_type dim = shape.size();
    const size_type max_dim = std::max(dim,depth);

    //add leading ones
    res_dim = widen_shape(res_shape, max_dim);

    const size_type concatenate_direction = res_dim - depth;
    const size_type offset = res_dim - dim;

    //res_dim >= max_dim
    for (size_type d{res_dim}; d!=size_type{0};){
        --d;
        index_type& res_element = res_shape[d];
        index_type shape_element = d >= offset ? shape[d-offset] : index_type{1};
        if (d == concatenate_direction){
                res_element+=shape_element;
        }else{
            if (res_element!=shape_element){
                throw combine_exception("tensors to concatenate must have equal shapes");
            }
        }
    }
}
template<typename T, typename SizeT>
auto make_block_shape(const T& t, const SizeT&, const SizeT&){
    return t.shape();
}
template<typename Nested, typename SizeT>
auto make_block_shape(std::initializer_list<Nested> blocks, const SizeT& res_dim, const SizeT& depth = nested_initialiser_list_depth<std::initializer_list<Nested>>::value){
    using tensor_type = typename nested_initialiser_list_value_type<std::initializer_list<Nested>>::type;
    using shape_type = typename tensor_type::shape_type;

    auto it = blocks.begin();
    shape_type res{make_block_shape(*it, res_dim, depth-1)};
    res.reserve(res_dim);
    widen_shape(res, depth);
    ++it;
    for (;it!=blocks.end(); ++it){
        make_block_shape_helper(res, make_block_shape(*it, res_dim, depth-1), depth);
    }
    return res;
}

}   //end of namespace detail

class combiner{
//join tensors along new direction, tensors must have the same shape
template<typename SizeT, typename ImplT, typename...ImplTs>
static auto stack_(const SizeT& direction, const ImplT& t, const ImplTs&...ts){
    using config_type = typename ImplT::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = std::common_type_t<typename ImplT::value_type, typename ImplTs::value_type...>;
    const auto& shape = t.shape();
    detail::check_stack_args(direction, shape, ts.shape()...);
    index_type tensors_number = sizeof...(ImplTs) + 1;
    auto res_shape = detail::make_stack_shape(direction,shape,tensors_number);
    if constexpr (sizeof...(ImplTs) == 0){
        return storage_tensor_factory<config_type, res_value_type>::make(std::move(res_shape), t.engine().begin(), t.engine().end());
    }else{
        auto res = storage_tensor_factory<config_type, res_value_type>::make(std::move(res_shape), res_value_type{});
        if (res.size() > index_type{0}){
            detail::fill_stack(direction, shape, t.size(), res.begin(), t.engine().begin(), ts.engine().begin()...);
        }
        return res;
    }
}
template<typename SizeT, typename...Us, typename...Ts>
static auto stack_(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return stack_(direction, t.impl_ref(), ts.impl_ref()...);
}

//join tensors along existing direction, tensors must have the same shape except concatenate direction
template<typename SizeT, typename ImplT, typename...ImplTs>
static auto concatenate_variadic(const SizeT& direction, const ImplT& t, const ImplTs&...ts){
    using config_type = typename ImplT::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = std::common_type_t<typename ImplT::value_type, typename ImplTs::value_type...>;
    detail::check_concatenate_args(direction, t.shape(), ts.shape()...);
    auto res_shape = detail::make_concatenate_shape(direction, t.shape(), ts.shape()...);
    if constexpr (sizeof...(ImplTs) == 0){
        return storage_tensor_factory<config_type, res_value_type>::make(std::move(res_shape),t.engine().begin(),t.engine().end());
    }else{
        auto res = storage_tensor_factory<config_type, res_value_type>::make(std::move(res_shape), res_value_type{});
        if (res.size() > index_type{0}){
            detail::fill_concatenate(direction, res.begin(), std::make_index_sequence<sizeof...(ImplTs) + 1>{}, t, ts...);
        }
        return res;
    }
}
template<typename SizeT, typename...Us, typename...Ts>
static auto concatenate_variadic(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return concatenate_variadic(direction, t.impl_ref(), ts.impl_ref()...);
}
template<typename SizeT, typename Container>
static auto concatenate_container(const SizeT& direction, const SizeT& res_dim, const Container& ts){
    static_assert(detail::is_tensor_container_v<Container>);
    using tensor_type = typename Container::value_type;
    using size_type = typename tensor_type::size_type;
    using index_type = typename tensor_type::index_type;
    using shape_type = typename tensor_type::shape_type;
    using config_type = typename tensor_type::config_type;
    using res_value_type = typename tensor_type::value_type;

    const auto& first_shape = (*ts.begin()).shape();
    shape_type res_shape(res_dim, index_type{1});
    if (first_shape.size() == size_type{0}){
        *res_shape.rbegin() = index_type{0};
    }else{
        std::copy(first_shape.rbegin(), first_shape.rend(), res_shape.rbegin());    //start copying from lowest direction, ones is leading
    }
    index_type first_chunk_size = std::accumulate(res_shape.begin()+direction, res_shape.end(), index_type{1}, std::multiplies{});

    using blocks_internals_type = std::tuple<index_type, decltype((*ts.begin()).begin())>;
    typename config_type::template container<blocks_internals_type> blocks_internals{};
    blocks_internals.reserve(ts.size());
    blocks_internals.emplace_back(first_chunk_size, (*ts.begin()).begin());
    index_type chunk_size_sum = first_chunk_size;

    for (auto it = ts.begin()+1; it!=ts.end(); ++it){
        const auto& shape = (*it).shape();
        const size_type dim = shape.size();
        index_type chunk_size{1};
        auto shape_element_ = [&shape, &dim, &res_dim](const auto& d_)
        {
            if (dim == size_type{0}){
                return d_ == res_dim-1 ? index_type{0} : index_type{1};
            }else{
                const size_type offset = res_dim - dim;
                return d_ >= offset ? shape[d_-offset] : index_type{1};
            }
        };
        for (size_type d{res_dim}; d!=size_type{0};){
            --d;
            index_type& res_element = res_shape[d];
            index_type shape_element = shape_element_(d);
            if (d >= direction){
                chunk_size *= shape_element;
            }
            if (d == direction){
                    res_element+=shape_element;
            }else{
                if (res_element!=shape_element){
                    throw combine_exception("tensors to concatenate must have equal shapes");
                }
            }
        }
        chunk_size_sum += chunk_size;
        blocks_internals.emplace_back(chunk_size, (*it).begin());
    }
    auto res = storage_tensor_factory<config_type, res_value_type>::make(std::move(res_shape), res_value_type{});
    index_type res_size = res.size();
    index_type iterations_number = res_size / chunk_size_sum;
    auto res_it = res.begin();
    for (index_type j = 0; j!=iterations_number; ++j){
        for (auto blocks_internals_it = blocks_internals.begin(); blocks_internals_it!=blocks_internals.end(); ++blocks_internals_it){
            auto chunk_size = std::get<0>(*blocks_internals_it);
            auto& it = std::get<1>(*blocks_internals_it);
            for (index_type i{0}; i!=chunk_size; ++i, ++it, ++res_it){
                *res_it = *it;
            }
        }
    }
    return res;
}
template<typename SizeT, typename Container>
static auto concatenate_container(const SizeT& direction, const Container& ts){
    static_assert(detail::is_tensor_container_v<Container>);
    using tensor_type = typename Container::value_type;
    using config_type = typename tensor_type::config_type;
    using size_type = typename tensor_type::size_type;
    using res_value_type = typename tensor_type::value_type;
    detail::check_concatenate_container_args(direction, ts);
    size_type first_dim = (*ts.begin()).dim();
    if (first_dim == size_type{0}){
        return storage_tensor_factory<config_type, res_value_type>::make();
    }else{
        return combiner::concatenate_container(direction, first_dim, ts);
    }
}

//Assemble tensor from nested lists of blocks
template<typename Container>
static auto concatenate_blocks(std::size_t depth, const Container& blocks){
    static_assert(detail::is_tensor_container_v<Container>);
    using tensor_type = typename Container::value_type;
    using size_type = typename tensor_type::size_type;
    using config_type = typename tensor_type::config_type;
    using res_value_type = typename tensor_type::value_type;

    size_type max_dim{0};
    for (auto it = blocks.begin(); it!=blocks.end(); ++it){
        max_dim = std::max((*it).dim(), max_dim);
    }
    if (max_dim == size_type{0}){
        return  storage_tensor_factory<config_type, res_value_type>::make();
    }else{
        const auto depth_ = static_cast<size_type>(depth);
        size_type res_dim = std::max(max_dim, depth_);
        const size_type direction = res_dim - depth_;
        return concatenate_container(direction, res_dim, blocks);
    }
}
template<typename...Ts>
static auto block_(std::initializer_list<tensor<Ts...>> blocks, std::size_t depth = detail::nested_initialiser_list_depth<decltype(blocks)>::value){
    return concatenate_blocks(depth, blocks);
}
template<typename Nested>
static auto block_(std::initializer_list<std::initializer_list<Nested>> blocks, std::size_t depth = detail::nested_initialiser_list_depth<decltype(blocks)>::value){
    using tensor_type = typename detail::nested_initialiser_list_value_type<Nested>::type;
    static_assert(detail::is_tensor_v<tensor_type>);
    using config_type = typename tensor_type::config_type;
    using block_type = decltype(block_(*blocks.begin(), depth-1));

    typename config_type::template container<block_type> blocks_{};
    blocks_.reserve(blocks.size());
    for (auto it = blocks.begin(); it!=blocks.end(); ++it){
        blocks_.push_back(block_(*it, depth-1));
    }
    return concatenate_blocks(depth, blocks_);
}

//Split tensor and return container of slice views
template<typename...Ts, typename IdxContainer>
static auto split_by_points(const tensor<Ts...>& t, const IdxContainer& split_points, const typename tensor<Ts...>::size_type& direction){
    using tensor_type = tensor<Ts...>;
    using config_type = typename tensor_type::config_type;
    using size_type = typename tensor_type::size_type;
    using index_type = typename tensor_type::index_type;
    using nop_type = typename slice_traits<config_type>::nop_type;
    using slice_type = typename slice_traits<config_type>::slice_type;
    using view_type = decltype(t(slice_type{},size_type{0}));
    using res_type = typename config_type::template container<view_type>;
    using res_size_type = typename res_type::size_type;
    static_assert(detail::is_index_container_v<IdxContainer, index_type>);

    const res_size_type parts_number = static_cast<res_size_type>(split_points.size()) + res_size_type{1};
    if (direction >= t.dim()){
        throw combine_exception("invalid split direction");
    }

    if (parts_number == res_size_type{1}){
        return res_type(parts_number, t({},size_type{0}));
    }else{
        res_type res{};
        res.reserve(parts_number);
        auto split_points_it = std::begin(split_points);
        index_type point{0};
        do{
            index_type next_point = *split_points_it;
            res.push_back(t(slice_type{point, next_point},direction));
            point = next_point;
            ++split_points_it;
        }while(split_points_it != std::end(split_points));
        res.push_back(t(slice_type{point, nop_type{}},direction));
        return res;
    }
}
template<typename...Ts>
static auto split_equal_parts(const tensor<Ts...>& t, const typename tensor<Ts...>::index_type& parts_number, const typename tensor<Ts...>::size_type& direction){
    using tensor_type = tensor<Ts...>;
    using config_type = typename tensor_type::config_type;
    using size_type = typename tensor_type::size_type;
    using index_type = typename tensor_type::index_type;
    using slice_type = typename slice_traits<config_type>::slice_type;
    using view_type = decltype(t(slice_type{},size_type{0}));
    using res_type = typename config_type::template container<view_type>;
    using res_size_type = typename res_type::size_type;

    const res_size_type parts_number_ = static_cast<res_size_type>(parts_number);
    if (direction >= t.dim()){
        throw combine_exception("invalid split direction");
    }
    const index_type direction_size = t.descriptor().shape()[direction];
    if (parts_number == index_type{0} || direction_size % parts_number != index_type{0}){
        throw combine_exception("can't split in equal parts");
    }

    if (parts_number == index_type{1}){
        return res_type(parts_number_, t({},size_type{0}));
    }else{
        res_type res{};
        res.reserve(parts_number_);
        index_type point{0};
        const index_type part_size = direction_size/parts_number;
        do{
            index_type next_point = point+part_size;
            res.push_back(t(slice_type{point,next_point},direction));
            point = next_point;
        }while(point!=direction_size);
        return res;
    }
}

public:
//combiner interface
template<typename SizeT, typename...Ts, typename...Tensors>
static auto stack(const SizeT& direction, const tensor<Ts...>& t, const Tensors&...ts){
    return stack_(direction, t, ts...);
}
template<typename SizeT, typename...Ts, typename...Tensors>
static auto concatenate(const SizeT& direction, const tensor<Ts...>& t, const Tensors&...ts){
    return concatenate_variadic(direction, t, ts...);
}
template<typename SizeT, typename Container>
static auto concatenate(const SizeT& direction, const Container& ts){
    static_assert(detail::is_tensor_container_v<Container>);
    return concatenate_container(direction, ts);
}
template<typename...Ts>
static auto block(std::initializer_list<tensor<Ts...>> blocks){
    return block_(blocks);
}
template<typename...Ts>
static auto block(std::initializer_list<std::initializer_list<tensor<Ts...>>> blocks){
    return block_(blocks);
}
template<typename...Ts>
static auto block(std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>> blocks){
    return block_(blocks);
}
template<typename...Ts>
static auto block(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>>> blocks){
    return block_(blocks);
}
template<typename...Ts, typename IdxContainer, typename SizeT, std::enable_if_t<detail::is_index_container_v<IdxContainer, typename tensor<Ts...>::index_type>,int> =0>
static auto split(const tensor<Ts...>& t, const IdxContainer& split_points, const SizeT& direction){
    return split_by_points(t, split_points, direction);
}
template<typename...Ts, typename SizeT>
static auto split(const tensor<Ts...>& t, std::initializer_list<typename tensor<Ts...>::index_type> split_points, const SizeT& direction){
    return split_by_points(t, split_points, direction);
}
template<typename...Ts>
static auto split(const tensor<Ts...>& t, const typename tensor<Ts...>::index_type& parts_number, const typename tensor<Ts...>::size_type& direction){
    return split_equal_parts(t, parts_number, direction);
}

};  //end of class combiner

//combine module free functions
template<typename SizeT, typename...Ts, typename...Tensors>
auto stack(const SizeT& direction, const tensor<Ts...>& t, const Tensors&...ts){
    using config_type = typename tensor<Ts...>::config_type;
    return combiner_selector<config_type>::type::stack(direction, t, ts...);
}
template<typename SizeT, typename...Ts, typename...Tensors>
auto concatenate(const SizeT& direction, const tensor<Ts...>& t, const Tensors&...ts){
    using config_type = typename tensor<Ts...>::config_type;
    return combiner_selector<config_type>::type::concatenate(direction, t, ts...);
}
template<typename SizeT, typename Container>
auto concatenate(const SizeT& direction, const Container& ts){
    static_assert(detail::is_tensor_container_v<Container>);
    using config_type = typename Container::value_type::config_type;
    return combiner_selector<config_type>::type::concatenate(direction, ts);
}
template<typename...Ts>
auto block(std::initializer_list<tensor<Ts...>> blocks){
    using config_type = typename tensor<Ts...>::config_type;
    return combiner_selector<config_type>::type::block(blocks);
}
template<typename...Ts>
auto block(std::initializer_list<std::initializer_list<tensor<Ts...>>> blocks){
    using config_type = typename tensor<Ts...>::config_type;
    return combiner_selector<config_type>::type::block(blocks);
}
template<typename...Ts>
auto block(std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>> blocks){
    using config_type = typename tensor<Ts...>::config_type;
    return combiner_selector<config_type>::type::block(blocks);
}
template<typename...Ts>
auto block(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>>> blocks){
    using config_type = typename tensor<Ts...>::config_type;
    return combiner_selector<config_type>::type::block(blocks);
}
template<typename...Ts, typename IdxContainer, typename SizeT, std::enable_if_t<detail::is_index_container_v<IdxContainer, typename tensor<Ts...>::index_type>,int> =0>
auto split(const tensor<Ts...>& t, const IdxContainer& split_points, const SizeT& direction){
    using config_type = typename tensor<Ts...>::config_type;
    return combiner_selector<config_type>::type::split(t, split_points, direction);
}
template<typename...Ts, typename SizeT>
auto split(const tensor<Ts...>& t, std::initializer_list<typename tensor<Ts...>::index_type> split_points, const SizeT& direction){
    using config_type = typename tensor<Ts...>::config_type;
    return combiner_selector<config_type>::type::split(t, split_points, direction);
}
template<typename...Ts>
auto split(const tensor<Ts...>& t, const typename tensor<Ts...>::index_type& parts_number, const typename tensor<Ts...>::size_type& direction){
    using config_type = typename tensor<Ts...>::config_type;
    return combiner_selector<config_type>::type::split(t, parts_number, direction);
}

}   //end of namespace gtensor

#endif