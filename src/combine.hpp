#ifndef COMBINE_HPP_
#define COMBINE_HPP_

#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include "forward_decl.hpp"
#include "tensor_init_list.hpp"
//#include

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
auto make_concatenate_block_size(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using size_type = SizeT;
    using index_type = typename ShT::value_type;
    index_type block_size_ = std::accumulate(shape.begin()+direction+size_type{1}, shape.end(), index_type{1}, std::multiplies{});
    return std::make_tuple(shape[direction]*block_size_, shapes[direction]*block_size_...);
}

template<typename SizeT, typename IdxT, typename ResultIt, typename ImplT, typename...ImplTs>
class concatenate_filler
{
    using index_type = IdxT;
    using block_size_type = decltype(make_concatenate_block_size(std::declval<SizeT>(), std::declval<ImplT>().shape(), std::declval<ImplTs>().shape()...));
    block_size_type block_sizes_;
    ResultIt res_it_;
public:
    concatenate_filler(const SizeT& direction__, ResultIt res_it__, const ImplT& t__, const ImplTs&...ts__):
        block_sizes_{make_concatenate_block_size(direction__, t__.shape(), ts__.shape()...)},
        res_it_{res_it__}
    {}
    template<std::size_t I, typename It>
    void fill(It& it){
        index_type block_size = std::get<I>(block_sizes_);
        for (index_type i{0}; i!=block_size; ++i, ++it, ++res_it_){
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

template<typename ResultIt, typename T>
auto fill_block_helper(ResultIt& res_it, std::initializer_list<T> blocks){
    using tensor_type = T;
    using size_type = typename tensor_type::size_type;
    using index_type = typename tensor_type::index_type;

    if (blocks.size() == 1){
        auto it = (*blocks.begin()).begin();
        auto it_end = (*blocks.begin()).end();
        for (;it!=it_end; ++it, ++res_it){
            *res_it = *it;
        }
    }else{
        //0block_size,1block_iterator
        using blocks_internals_type = std::tuple<index_type, decltype((*blocks.begin()).begin())>;
        std::vector<blocks_internals_type> blocks_internals{};
        blocks_internals.reserve(blocks.size());
        index_type blocks_size{0};
        index_type blocks_block_size{0};
        for (auto blocks_it = blocks.begin(); blocks_it!=blocks.end(); ++blocks_it){
            size_type dim = (*blocks_it).dim();
            blocks_size+=(*blocks_it).size();
            index_type block_size = (*blocks_it).shape()[dim-1];
            blocks_block_size+=block_size;
            blocks_internals.emplace_back(block_size, (*blocks_it).begin());
        }
        index_type iters_number = blocks_size / blocks_block_size;

        for (index_type j = 0; j!=iters_number; ++j){
            for (auto blocks_internals_it = blocks_internals.begin(); blocks_internals_it!=blocks_internals.end(); ++blocks_internals_it){
                auto block_size = std::get<0>(*blocks_internals_it);
                auto& it = std::get<1>(*blocks_internals_it);
                for (index_type i{0}; i!=block_size; ++i, ++it, ++res_it){
                    *res_it = *it;
                }
            }
        }
    }
}

template<typename ResultIt, typename Nested>
auto fill_block_helper(ResultIt& res_it, std::initializer_list<std::initializer_list<Nested>> blocks){
    for (auto it = blocks.begin(); it!=blocks.end(); ++it){
        fill_block_helper(res_it, *it);
    }
}
template<typename ResultIt, typename Nested>
auto fill_block(ResultIt res_it, std::initializer_list<Nested> blocks){
    fill_block_helper(res_it, blocks);
}

}   //end of namespace detail

class combiner{
//join tensors along new direction, tensors must have the same shape
template<typename SizeT, typename ImplT, typename...ImplTs>
static auto stack_(const SizeT& direction, const ImplT& t, const ImplTs&...ts){
    using config_type = typename ImplT::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = std::common_type_t<typename ImplT::value_type, typename ImplTs::value_type...>;
    using res_impl_type = storage_tensor<typename detail::storage_engine_traits<typename config_type::host_engine,config_type,typename config_type::template storage<res_value_type>>::type>;
    const auto& shape = t.shape();
    detail::check_stack_args(direction, shape, ts.shape()...);
    index_type tensors_number = sizeof...(ImplTs) + 1;
    auto res_shape = detail::make_stack_shape(direction,shape,tensors_number);
    if constexpr (sizeof...(ImplTs) == 0){
        return tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape), t.engine().begin(), t.engine().end());
    }else{
        auto res = tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape), res_value_type{});
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
static auto concatenate_(const SizeT& direction, const ImplT& t, const ImplTs&...ts){
    using config_type = typename ImplT::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = std::common_type_t<typename ImplT::value_type, typename ImplTs::value_type...>;
    using res_impl_type = storage_tensor<typename detail::storage_engine_traits<typename config_type::host_engine,config_type,typename config_type::template storage<res_value_type>>::type>;
    detail::check_concatenate_args(direction, t.shape(), ts.shape()...);
    auto res_shape = detail::make_concatenate_shape(direction, t.shape(), ts.shape()...);
    if constexpr (sizeof...(ImplTs) == 0){
        return tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape),t.engine().begin(),t.engine().end());
    }else{
        auto res = tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape), res_value_type{});
        if (res.size() > index_type{0}){
            detail::fill_concatenate(direction, res.begin(), std::make_index_sequence<sizeof...(ImplTs) + 1>{}, t, ts...);
        }
        return res;
    }
}
template<typename SizeT, typename...Us, typename...Ts>
static auto concatenate_(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return concatenate_(direction, t.impl_ref(), ts.impl_ref()...);
}

//make tensor from blocks
template<typename Nested>
static auto block_(std::initializer_list<Nested> blocks){
    using tensor_type = typename detail::nested_initialiser_list_value_type<std::initializer_list<Nested>>::value_type;
    using config_type = typename tensor_type::config_type;
    using size_type = typename tensor_type::size_type;
    using res_value_type = typename tensor_type::value_type;
    using res_impl_type = storage_tensor<typename detail::storage_engine_traits<typename config_type::host_engine,config_type,typename config_type::template storage<res_value_type>>::type>;
    size_type depth = detail::nested_initialiser_list_depth<decltype(blocks)>::value;
    size_type res_dim = std::max(depth, detail::max_block_dim(blocks));
    auto res_shape = detail::make_block_shape(blocks, res_dim, depth);
    auto res = tensor<res_value_type, config_type, res_impl_type>::make_tensor(std::move(res_shape), res_value_type{});
    detail::fill_block(res.begin(), blocks);
    return res;
}


template<typename SizeT, typename...Us, typename...Ts>
friend auto stack(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts);
template<typename SizeT, typename...Us, typename...Ts>
friend auto concatenate(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts);
template<typename...Ts>
friend auto block(std::initializer_list<tensor<Ts...>> blocks);
template<typename...Ts>
friend auto block(std::initializer_list<std::initializer_list<tensor<Ts...>>> blocks);
template<typename...Ts>
friend auto block(std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>> blocks);

};  //end of class combiner

template<typename SizeT, typename...Us, typename...Ts>
auto stack(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return combiner::stack_(direction, t, ts...);
}
template<typename SizeT, typename...Us, typename...Ts>
auto concatenate(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return combiner::concatenate_(direction, t, ts...);
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

}   //end of namespace gtensor

#endif