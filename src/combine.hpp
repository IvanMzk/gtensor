#ifndef COMBINE_HPP_
#define COMBINE_HPP_

#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include "forward_decl.hpp"
#include "tensor_init_list.hpp"
#include "tensor_factory.hpp"

namespace gtensor{

class combine_exception : public std::runtime_error{
public:
    explicit combine_exception(const char* what):
        runtime_error(what)
    {}
};

namespace detail{

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
    using tensor_type = typename nested_initialiser_list_value_type<std::initializer_list<Nested>>::value_type;
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
    using tensor_type = typename nested_initialiser_list_value_type<std::initializer_list<Nested>>::value_type;
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
    using tensor_type = typename Container::value_type;
    using size_type = typename tensor_type::size_type;
    using index_type = typename tensor_type::index_type;
    using shape_type = typename tensor_type::shape_type;
    using config_type = typename tensor_type::config_type;
    using res_value_type = typename tensor_type::value_type;

    const auto& first_shape = (*ts.begin()).shape();
    shape_type res_shape(res_dim, index_type{1});
    std::copy(first_shape.rbegin(), first_shape.rend(), res_shape.rbegin());    //start copying from lowest direction, ones is leading
    index_type first_chunk_size = std::accumulate(res_shape.begin()+direction, res_shape.end(), index_type{1}, std::multiplies{});

    using blocks_internals_type = std::tuple<index_type, decltype((*ts.begin()).begin())>;
    std::vector<blocks_internals_type> blocks_internals{};
    blocks_internals.reserve(ts.size());
    blocks_internals.emplace_back(first_chunk_size, (*ts.begin()).begin());
    index_type chunk_size_sum = first_chunk_size;

    for (auto it = ts.begin()+1; it!=ts.end(); ++it){
        const auto& shape = (*it).shape();
        const size_type offset = res_dim - shape.size();
        index_type chunk_size{1};
        for (size_type d{res_dim}; d!=size_type{0};){
            --d;
            index_type& res_element = res_shape[d];
            index_type shape_element = d >= offset ? shape[d-offset] : index_type{1};
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
    using tensor_type = typename Container::value_type;
    using size_type = typename tensor_type::size_type;

    size_type res_dim{depth};
    for (auto it = blocks.begin(); it!=blocks.end(); ++it){
        res_dim = std::max((*it).dim(), res_dim);
    }
    const size_type direction = res_dim - static_cast<size_type>(depth);
    return concatenate_container(direction, res_dim, blocks);
}
template<typename T>
static auto block_(std::initializer_list<T> blocks, std::size_t depth = detail::nested_initialiser_list_depth<decltype(blocks)>::value){
    return concatenate_blocks(depth, blocks);
}
template<typename Nested>
static auto block_(std::initializer_list<std::initializer_list<Nested>> blocks, std::size_t depth = detail::nested_initialiser_list_depth<decltype(blocks)>::value){
    using block_type = decltype(block_(*blocks.begin(), depth-1));

    std::vector<block_type> blocks_{};
    blocks_.reserve(blocks.size());
    for (auto it = blocks.begin(); it!=blocks.end(); ++it){
        blocks_.push_back(block_(*it, depth-1));
    }
    return concatenate_blocks(depth, blocks_);
}

//friend interface
template<typename SizeT, typename...Us, typename...Ts>
friend auto stack(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts);
template<typename SizeT, typename...Us, typename...Ts>
friend auto concatenate(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts);
template<typename SizeT, typename Container>
friend auto concatenate(const SizeT& direction, const Container& ts);
template<typename...Ts>
friend auto block(std::initializer_list<tensor<Ts...>> blocks);
template<typename...Ts>
friend auto block(std::initializer_list<std::initializer_list<tensor<Ts...>>> blocks);
template<typename...Ts>
friend auto block(std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>> blocks);
template<typename...Ts>
friend auto block(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>>> blocks);

};  //end of class combiner

template<typename SizeT, typename...Us, typename...Ts>
auto stack(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return combiner::stack_(direction, t, ts...);
}
template<typename SizeT, typename...Us, typename...Ts>
auto concatenate(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return combiner::concatenate_variadic(direction, t, ts...);
}
template<typename SizeT, typename Container>
auto concatenate(const SizeT& direction, const Container& ts){
    return combiner::concatenate_container(direction, ts);
}
template<typename...Ts>
auto block(std::initializer_list<tensor<Ts...>> blocks){
    return combiner::block_(blocks);
}
template<typename...Ts>
auto block(std::initializer_list<std::initializer_list<tensor<Ts...>>> blocks){
    return combiner::block_(blocks);
}
template<typename...Ts>
auto block(std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>> blocks){
    return combiner::block_(blocks);
}
template<typename...Ts>
auto block(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>>> blocks){
    return combiner::block_(blocks);
}

}   //end of namespace gtensor

#endif