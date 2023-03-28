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

template<typename T, typename = void> constexpr inline bool is_tensor_container_v = false;
template<typename T> constexpr inline bool is_tensor_container_v<T, std::void_t<std::enable_if_t<is_container_v<T>>>> = is_tensor_v<typename T::value_type>;

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
    using size_type = SizeT;
    size_type dim = shape.size();
    if (direction > dim){
        throw combine_exception{"bad stack direction"};
    }
    if constexpr (sizeof...(ShTs) > 0){
        if (!((shape==shapes)&&...)){
            throw combine_exception{"tensors to stack must have equal shapes"};
        }
    }
}

template<typename SizeT, typename ShT, typename...ShTs>
void check_concatenate_variadic_args(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
    using size_type = SizeT;
    size_type dim = shape.size();
    if (direction >= dim){
        throw combine_exception{"bad concatenate direction"};
    }
    if constexpr (sizeof...(ShTs) > 0){
        if (!((dim==static_cast<size_type>(shapes.size()))&&...)){
            throw combine_exception{"tensors to concatenate must have equal dimentions number"};
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

template<typename SizeT, typename ShT, typename...ShTs>
void check_concatenate_variadic_args(const SizeT& direction, const std::tuple<ShT, ShTs...>& shapes){
    std::apply([&direction](const auto&...shapes_){check_concatenate_variadic_args(direction,shapes_...);},shapes);
}

template<typename SizeT, typename Container>
void check_concatenate_container_args(const SizeT& direction, const Container& shapes){
    using size_type = SizeT;
    if (shapes.empty()){
        throw combine_exception{"nothing to concatenate"};
    }
    auto it = shapes.begin();
    const auto& first_shape = *it;
    const size_type& first_dim = first_shape.size();
    if (direction >= first_dim){
        throw combine_exception{"bad concatenate direction"};
    }
    for(++it; it!=shapes.end(); ++it){
        const auto& shape = *it;
        if (first_dim!=shape.size()){
            throw combine_exception("tensors to concatenate must have equal dimensions number");
        }
        for (size_type d{0}; d!=first_dim; ++d){
            if (first_shape[d]!=shape[d]){
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
    size_type dim = shape.size();
    shape_type res(dim+size_type{1});
    std::copy(shape.begin(), shape.begin()+direction, res.begin());
    std::copy(shape.begin()+direction, shape.end(), res.begin()+direction+size_type{1});
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
    index_type chunk_size = make_stack_chunk_size(direction, shape);
    auto filler = [chunk_size, res_it](auto& it) mutable {
        for (index_type i{0}; i!=chunk_size; ++i, ++res_it, ++it){
            *res_it = *it;
        }
    };
    index_type iterations_number = size/chunk_size;
    for (index_type i{0}; i!=iterations_number; ++i){
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
    using size_type = typename tensor_type::size_type;
    using index_type = typename tensor_type::index_type;
    using config_type = typename tensor_type::config_type;
    //0chunk_size,1iterator
    using ts_internals_type = std::tuple<index_type, decltype((*ts.begin()).begin())>;

    typename config_type::template container<ts_internals_type> ts_internals{};
    ts_internals.reserve(ts.size());
    auto shapes_it = shapes.begin();
    const auto& first_shape = *shapes_it;
    //the same part for all shapes
    const index_type chunk_size_ = std::accumulate(first_shape.begin()+direction+size_type{1}, first_shape.end(), index_type{1}, std::multiplies{});
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
    using size_type = SizeT;
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    size_type dim = shape.size();
    if (dim < new_dim){
        shape_type res(new_dim, index_type{1});
        std::copy(shape.rbegin(),shape.rend(),res.rbegin());
        return res;
    }else{
        return shape;
    }
}


// template<typename SizeT, typename ShT, typename...ShTs>
// void check_vstack_variadic_args(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
//     using size_type = SizeT;
//     using index_type = typename ShT::value_type;
//     if (direction != size_type{0}){
//         throw combine_exception{"bad vstack direction"};
//     }
//     auto dim = [](const auto& shape_){
//         size_type dim_ = shape_.size();
//         return dim_==size_type{1}?dim_+size_type{1}:dim_;
//     };
//     auto shape_element = [](const auto& shape_, const auto& d_){
//         size_type dim_ = shape_.size();
//         return dim_==size_type{1} ? d_==size_type{0}?index_type{1}:shape_[size_type{0}] : shape_[d_];
//     };
//     size_type dim_ = dim(shape);
//     if constexpr (sizeof...(ShTs) > 0){
//         if (!((dim_==dim(shapes))&&...)){
//             throw combine_exception{"tensors to concatenate must have equal shapes"};
//         }
//         for (size_type d{0}; d!=dim_; ++d){
//             if (!((shape_element(shape,d)==shape_element(shapes,d))&&...)){
//                 if (d!=direction){
//                     throw combine_exception{"tensors to concatenate must have equal shapes"};
//                 }
//             }
//         }
//     }
// }

// // template<typename SizeT, typename Container>
// // void check_vstack_container_args(const SizeT& direction, const Container& ts){
// //     using tensor_type = typename Container::value_type;
// //     using size_type = typename tensor_type::size_type;
// //     using index_type = typename tensor_type::index_type;
// //     if (direction != size_type{0}){
// //         throw combine_exception{"bad vstack direction"};
// //     }
// //     auto dim = [](const auto& dim_){
// //         return dim_==size_type{1}?dim_+size_type{1}:dim_;
// //     };
// //     auto shape_element = [](const auto& shape_, const auto& d_){
// //         size_type dim_ = shape_.size();
// //         return dim_==size_type{1} ? d_==size_type{0}?index_type{1}:shape_[size_type{0}] : shape_[d_];
// //     };
// //     auto it = ts.begin();
// //     const auto& first = *it;
// //     const size_type first_dim = dim(first.dim());
// //     const auto& first_shape = first.shape();
// //     for (++it;it!=ts.end();++it){
// //         const auto& tensor_ = *it;
// //         const auto& shape_ = tensor_.shape();
// //         if (first_dim!=dim(tensor_.dim())){
// //             throw combine_exception("tensors to concatenate must have equal shapes");
// //         }
// //         for (size_type d{0}; d!=first_dim; ++d){
// //             if (!(shape_element(first_shape,d)==shape_element(shape_,d))){
// //                 if (d!=direction){
// //                     throw combine_exception{"tensors to concatenate must have equal shapes"};
// //                 }
// //             }
// //         }
// //     }
// // }

// template<typename SizeT, typename ShapeElement, typename Container>
// void check_concatenate_container_args_helper(const SizeT& direction, const ShapeElement& shape_element , const Container& ts){
//     using tensor_type = typename Container::value_type;
//     using size_type = typename tensor_type::size_type;
//     using index_type = typename tensor_type::index_type;

//     auto it = ts.begin();
//     const auto& first = *it;
//     const auto& first_shape = first.shape();
//     for (++it;it!=ts.end();++it){
//         const auto& tensor_ = *it;
//         const auto& shape_ = tensor_.shape();
//         for (size_type d{0}; d!=first_dim; ++d){
//             if (!(shape_element(first_shape,d)==shape_element(shape_,d))){
//                 if (d!=direction){
//                     throw combine_exception{"tensors to concatenate must have equal shapes"};
//                 }
//             }
//         }
//     }
// }

// template<typename SizeT, typename Container>
// void check_broadcast_concatenate_container_args(const SizeT& direction, const SizeT& res_dim, const Container& ts){
//     using tensor_type = typename Container::value_type;
//     using size_type = typename tensor_type::size_type;
//     using index_type = typename tensor_type::index_type;

//     auto dim = [&res_dim](const auto& dim_){
//         return dim_!=res_dim?res_dim:dim_;
//     };

//     auto shape_element = [&res_dim](const auto& shape_, const auto& d_){
//         size_type dim_ = shape_.size();
//         const size_type offset_ = res_dim - dim_;
//         return d_ >= offset_ ? shape[d_-offset_] : index_type{1};
//     };
//     auto it = ts.begin();
//     const auto& first = *it;
//     const size_type first_dim = dim(first.dim());
//     const auto& first_shape = first.shape();
//     for (++it;it!=ts.end();++it){
//         const auto& tensor_ = *it;
//         const auto& shape_ = tensor_.shape();
//         if (first_dim!=dim(tensor_.dim())){
//             throw combine_exception("tensors to concatenate must have equal shapes");
//         }
//         for (size_type d{0}; d!=first_dim; ++d){
//             if (!(shape_element(first_shape,d)==shape_element(shape_,d))){
//                 if (d!=direction){
//                     throw combine_exception{"tensors to concatenate must have equal shapes"};
//                 }
//             }
//         }
//     }
// }

// //
// // template<typename...Ts>
// // auto widen_tensor(const gtensor::tensor<Ts...>& t, typename gtensor::tensor<Ts...>::size_type& new_dim){

// // }

// // auto reshaper = [](const auto& t_){
// //             if (t_.dim() == size_type{1}){
// //                 return t_.reshape(index_type{1},t_.shape()[size_type{0}]);
// //             }
// //             return t_.reshape();
// //         };

// template<typename SizeT, typename ShT, typename...ShTs>
// auto make_vstack_shape(const SizeT& direction, const ShT& shape, const ShTs&...shapes){
//     using shape_type = ShT;
//     using index_type = typename shape_type::value_type;
//     using size_type = SizeT;
//     if (direction != size_type{0}){
//         throw combine_exception{"bad vstack direction"};
//     }
//     auto dim = [](const auto& shape_){
//         const size_type dim_ = shape_.size();
//         return dim_==size_type{1}?dim_+size_type{1}:dim_;
//     };
//     auto shape_element = [](const auto& shape_, const auto& d_){
//         const size_type dim_ = shape_.size();
//         return dim_==size_type{1} ? d_==size_type{0}?index_type{1}:shape_[size_type{0}] : shape_[d_];
//     };
//     const size_type dim_ = dim(shape);
//     //const size_type dim_ = shape.size();
//     if (dim_ == size_type{0}){
//         return shape_type{};
//     }else{
//         index_type direction_size{shape_element(shape, direction)};
//         ((direction_size+=shape_element(shapes, direction)),...);
//         shape_type res(dim_, index_type{1});
//         std::copy(shape.rbegin(),shape.rend(),res.rbegin());
//         res[direction]=direction_size;
//         return res;
//     }
// }



// //returns dim of result of block
// template<typename T>
// auto max_block_dim(const T& t){
//     return t.dim();
// }
// template<typename Nested>
// auto max_block_dim(std::initializer_list<Nested> blocks){
//     using tensor_type = typename nested_initialiser_list_value_type<std::initializer_list<Nested>>::type;
//     using size_type = typename tensor_type::size_type;
//     size_type dim{0};
//     for(auto it = blocks.begin(); it!=blocks.end(); ++it){
//         dim = std::max(max_block_dim(*it), dim);
//     }
//     return dim;
// }

// //concatenates shapes and accumulates result in res_shape
// //depth means direction to concatenate, 1 is the last direction with lowest stride, 2 is next to last direction..., so direction is inverted
// //res_shape initialized with first shape for current depth
// template<typename ShT, typename SizeT>
// auto make_block_shape_helper(ShT& res_shape, const ShT& shape, const SizeT& depth){
//     using size_type = SizeT;
//     using shape_type = ShT;
//     using index_type = typename shape_type::value_type;
//     size_type res_dim = res_shape.size();
//     const size_type dim = shape.size();
//     const size_type max_dim = std::max(dim,depth);

//     //add leading ones
//     res_dim = widen_shape(res_shape, max_dim);

//     const size_type concatenate_direction = res_dim - depth;
//     const size_type offset = res_dim - dim;

//     //res_dim >= max_dim
//     for (size_type d{res_dim}; d!=size_type{0};){
//         --d;
//         index_type& res_element = res_shape[d];
//         index_type shape_element = d >= offset ? shape[d-offset] : index_type{1};
//         if (d == concatenate_direction){
//                 res_element+=shape_element;
//         }else{
//             if (res_element!=shape_element){
//                 throw combine_exception("tensors to concatenate must have equal shapes");
//             }
//         }
//     }
// }
// template<typename T, typename SizeT>
// auto make_block_shape(const T& t, const SizeT&, const SizeT&){
//     return t.shape();
// }
// template<typename Nested, typename SizeT>
// auto make_block_shape(std::initializer_list<Nested> blocks, const SizeT& res_dim, const SizeT& depth = nested_initialiser_list_depth<std::initializer_list<Nested>>::value){
//     using tensor_type = typename nested_initialiser_list_value_type<std::initializer_list<Nested>>::type;
//     using shape_type = typename tensor_type::shape_type;

//     auto it = blocks.begin();
//     shape_type res{make_block_shape(*it, res_dim, depth-1)};
//     res.reserve(res_dim);
//     widen_shape(res, depth);
//     ++it;
//     for (;it!=blocks.end(); ++it){
//         make_block_shape_helper(res, make_block_shape(*it, res_dim, depth-1), depth);
//     }
//     return res;
// }

}   //end of namespace detail

class combiner{
//join tensors along new direction, tensors must have the same shape
template<typename SizeT, typename...Us, typename...Ts>
static auto stack_variadic(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    static_assert((detail::is_tensor_v<Ts>&&...));
    using tensor_type = tensor<Us...>;
    using config_type = typename tensor_type::config_type;
    using index_type = typename config_type::index_type;
    using res_value_type = std::common_type_t<typename tensor_type::value_type, typename Ts::value_type...>;

    const auto& shape = t.shape();
    detail::check_stack_variadic_args(direction, shape, ts.shape()...);
    index_type tensors_number = sizeof...(Ts) + 1;
    auto res_shape = detail::make_stack_shape(direction,shape,tensors_number);
    if constexpr (sizeof...(Ts) == 0){
        return storage_tensor_factory<config_type, res_value_type>::make(std::move(res_shape), t.begin(), t.end());
    }else{
        auto res = storage_tensor_factory<config_type, res_value_type>::make(std::move(res_shape), res_value_type{});
        if (!res.empty()){
            detail::fill_stack(direction, shape, t.size(), res.begin(), t.begin(), ts.begin()...);
        }
        return res;
    }
}

//join tensors along existing direction, tensors must have the same shape except concatenate direction
template<typename SizeT, typename...Us, typename...Ts>
static auto concatenate_variadic(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    using tensor_type = tensor<Us...>;
    using config_type = typename tensor_type::config_type;
    using res_value_type = std::common_type_t<typename tensor_type::value_type, typename Ts::value_type...>;

    auto shapes = std::make_tuple(t.shape(), ts.shape()...);
    detail::check_concatenate_variadic_args(direction, shapes);
    auto res_shape = detail::make_concatenate_variadic_shape(direction, shapes);
    if constexpr (sizeof...(Ts) == 0){
        return storage_tensor_factory<config_type, res_value_type>::make(std::move(res_shape),t.begin(),t.end());
    }else{
        auto res = storage_tensor_factory<config_type, res_value_type>::make(std::move(res_shape), res_value_type{});
        if (!res.empty()){
            detail::fill_concatenate(direction, shapes, res.begin(), std::make_index_sequence<sizeof...(Ts) + 1>{}, t.begin(), ts.begin()...);
        }
        return res;
    }
}
template<typename SizeT, typename Container>
static auto concatenate_container(const SizeT& direction, const Container& ts){
    using tensor_type = typename Container::value_type;
    using config_type = typename tensor_type::config_type;
    using shape_type = typename tensor_type::shape_type;
    using res_value_type = typename tensor_type::value_type;

    typename config_type::template container<shape_type> shapes{};
    shapes.reserve(ts.size());
    std::for_each(ts.begin(), ts.end(), [&shapes](const auto& t)mutable{shapes.push_back(t.shape());});
    detail::check_concatenate_container_args(direction, shapes);
    auto res = storage_tensor_factory<config_type, res_value_type>::make(detail::make_concatenate_container_shape(direction, shapes), res_value_type{});
    if (!res.empty()){
        detail::fill_concatenate_container(direction,shapes,res.begin(),ts);
    }
    return res;
}

//Assemble tensor from nested tuples of blocks
template<typename...Us, typename...Ts>
static auto block_variadic(std::size_t depth, const tensor<Us...>& t, const Ts&...ts){
    using tensor_type = tensor<Us...>;
    using size_type = typename tensor_type::size_type;
    const size_type depth_ = static_cast<size_type>(depth);
    const size_type max_dim = std::max({t.dim(),ts.dim()...});
    const size_type res_dim = std::max(depth_,max_dim);
    const size_type direction = res_dim - depth_;
    return concatenate_variadic(direction, t.reshape(detail::widen_shape(t.shape(),res_dim)), ts.reshape(detail::widen_shape(ts.shape(),res_dim))...);
}
template<typename...Us, typename...Ts>
static auto block_tuple(const std::tuple<tensor<Us...>, Ts...>& blocks){
    //depth is 1
    auto apply_blocks = [](const auto&...ts){
        return block_variadic(1, ts...);
    };
    return std::apply(apply_blocks, blocks);
}
template<typename...Us, typename...Ts>
static auto block_tuple(const std::tuple<std::tuple<Us...>, Ts...>& blocks){
    std::size_t depth = detail::nested_tuple_depth_v<std::tuple<std::tuple<Us...>, Ts...>>;
    auto apply_blocks = [&depth](const auto&...blocks_){
        return block_variadic(depth, block_tuple(blocks_)...);
    };
    return std::apply(apply_blocks, blocks);
}

// template<typename SizeT, typename Container>
// static auto concatenate_container(const SizeT& direction, const SizeT& res_dim, const Container& ts){
//     static_assert(detail::is_tensor_container_v<Container>);
//     using tensor_type = typename Container::value_type;
//     using size_type = typename tensor_type::size_type;
//     using index_type = typename tensor_type::index_type;
//     using shape_type = typename tensor_type::shape_type;
//     using config_type = typename tensor_type::config_type;
//     using res_value_type = typename tensor_type::value_type;

//     const auto& first_shape = (*ts.begin()).shape();
//     shape_type res_shape(res_dim, index_type{1});
//     if (first_shape.size() == size_type{0}){
//         *res_shape.rbegin() = index_type{0};
//     }else{
//         std::copy(first_shape.rbegin(), first_shape.rend(), res_shape.rbegin());    //start copying from lowest direction, ones is leading
//     }
//     index_type first_chunk_size = std::accumulate(res_shape.begin()+direction, res_shape.end(), index_type{1}, std::multiplies{});

//     using blocks_internals_type = std::tuple<index_type, decltype((*ts.begin()).begin())>;
//     typename config_type::template container<blocks_internals_type> blocks_internals{};
//     blocks_internals.reserve(ts.size());
//     blocks_internals.emplace_back(first_chunk_size, (*ts.begin()).begin());
//     index_type chunk_size_sum = first_chunk_size;

//     for (auto it = ts.begin()+1; it!=ts.end(); ++it){
//         const auto& shape = (*it).shape();
//         const size_type dim = shape.size();
//         index_type chunk_size{1};
//         auto shape_element_ = [&shape, &dim, &res_dim](const auto& d_)
//         {
//             if (dim == size_type{0}){
//                 return d_ == res_dim-1 ? index_type{0} : index_type{1};
//             }else{
//                 const size_type offset = res_dim - dim;
//                 return d_ >= offset ? shape[d_-offset] : index_type{1};
//             }
//         };
//         for (size_type d{res_dim}; d!=size_type{0};){
//             --d;
//             index_type& res_element = res_shape[d];
//             index_type shape_element = shape_element_(d);
//             if (d >= direction){
//                 chunk_size *= shape_element;
//             }
//             if (d == direction){
//                     res_element+=shape_element;
//             }else{
//                 if (res_element!=shape_element){
//                     throw combine_exception("tensors to concatenate must have equal shapes");
//                 }
//             }
//         }
//         chunk_size_sum += chunk_size;
//         blocks_internals.emplace_back(chunk_size, (*it).begin());
//     }
//     auto res = storage_tensor_factory<config_type, res_value_type>::make(std::move(res_shape), res_value_type{});
//     index_type res_size = res.size();
//     index_type iterations_number = res_size / chunk_size_sum;
//     auto res_it = res.begin();
//     for (index_type j = 0; j!=iterations_number; ++j){
//         for (auto blocks_internals_it = blocks_internals.begin(); blocks_internals_it!=blocks_internals.end(); ++blocks_internals_it){
//             auto chunk_size = std::get<0>(*blocks_internals_it);
//             auto& it = std::get<1>(*blocks_internals_it);
//             for (index_type i{0}; i!=chunk_size; ++i, ++it, ++res_it){
//                 *res_it = *it;
//             }
//         }
//     }
//     return res;
// }
// template<typename SizeT, typename Container>
// static auto concatenate_container(const SizeT& direction, const Container& ts){
//     static_assert(detail::is_tensor_container_v<Container>);
//     using tensor_type = typename Container::value_type;
//     using config_type = typename tensor_type::config_type;
//     using size_type = typename tensor_type::size_type;
//     using res_value_type = typename tensor_type::value_type;
//     detail::check_concatenate_container_args(direction, ts);
//     size_type first_dim = (*ts.begin()).dim();
//     if (first_dim == size_type{0}){
//         return storage_tensor_factory<config_type, res_value_type>::make();
//     }else{
//         return combiner::concatenate_container(direction, first_dim, ts);
//     }
// }
// template<typename...Ts, typename...Tensors>
// static auto vstack_variadic(const tensor<Ts...>& t, const Tensors&...ts){
//     using tensor_type = tensor<Ts...>;
//     using size_type = typename tensor_type::size_type;
//     using index_type = typename tensor_type::index_type;
//     if (t.dim()==size_type{1} || ((ts.dim()==size_type{1})||...)){
//         auto reshaper = [](const auto& t_){
//             if (t_.dim() == size_type{1}){
//                 return t_.reshape(index_type{1},t_.shape()[size_type{0}]);
//             }
//             return t_.reshape();
//         };
//         return concatenate(size_type{0},reshaper(t), reshaper(ts)...);
//     }else{
//         return concatenate(size_type{0},t,ts...);
//     }
// }
// // static auto vstack_variadic(const tensor<Ts...>& t, const Tensors&...ts){
// //     using tensor_type = tensor<Ts...>;
// //     using size_type = typename tensor_type::size_type;
// //     using index_type = typename tensor_type::index_type;
// //     size_type min_dim{t.dim()};
// //     ((min_dim = std::min(min_dim, ts.dim())),...);
// //     if (min_dim != size_type{1}){
// //         return concatenate(size_type{0},t,ts...);
// //     }else{
// //         auto reshaper = [](const auto& t_){
// //             if (t_.dim() == size_type{1}){
// //                 return t_.reshape(index_type{1},t_.shape()[size_type{0}]);
// //             }
// //             return t_.reshape();
// //         };
// //         return concatenate(size_type{0},reshaper(t), reshaper(ts)...);
// //     }
// // }
// template<typename Container>
// static auto vstack_container(const Container& ts){
//     static_assert(detail::is_tensor_container_v<Container>);
//     using tensor_type = typename Container::value_type;
//     using size_type = typename tensor_type::size_type;
//     using config_type = typename tensor_type::config_type;
//     using res_value_type = typename tensor_type::value_type;



// }

// // template<typename Container>
// // static auto vstack_container(const Container& ts){
// //     static_assert(detail::is_tensor_container_v<Container>);
// //     using tensor_type = typename Container::value_type;
// //     using size_type = typename tensor_type::size_type;
// //     using config_type = typename tensor_type::config_type;
// //     using res_value_type = typename tensor_type::value_type;

// //     detail::check_vstack_container_args(size_type{0}, ts);
// //     size_type res_dim{0};
// //     for (auto it = ts.begin(); it!=ts.end(); ++it){
// //         res_dim = std::max((*it).dim(), res_dim);
// //     }
// //     if (res_dim == size_type{0}){
// //         return  storage_tensor_factory<config_type, res_value_type>::make();
// //     }else if (res_dim == size_type{1}){
// //         ++res_dim;
// //     }
// //     return concatenate_container(size_type{0}, res_dim, ts);
// // }


// //Assemble tensor from nested lists of blocks
// template<typename Container>
// static auto concatenate_blocks(std::size_t depth, const Container& blocks){
//     static_assert(detail::is_tensor_container_v<Container>);
//     using tensor_type = typename Container::value_type;
//     using size_type = typename tensor_type::size_type;
//     using config_type = typename tensor_type::config_type;
//     using res_value_type = typename tensor_type::value_type;

//     size_type max_dim{0};
//     for (auto it = blocks.begin(); it!=blocks.end(); ++it){
//         max_dim = std::max((*it).dim(), max_dim);
//     }
//     if (max_dim == size_type{0}){
//         return  storage_tensor_factory<config_type, res_value_type>::make();
//     }else{
//         const auto depth_ = static_cast<size_type>(depth);
//         size_type res_dim = std::max(max_dim, depth_);
//         const size_type direction = res_dim - depth_;
//         return concatenate_container(direction, res_dim, blocks);
//     }
// }
// template<typename...Ts>
// static auto block_(std::initializer_list<tensor<Ts...>> blocks, std::size_t depth = detail::nested_initialiser_list_depth<decltype(blocks)>::value){
//     return concatenate_blocks(depth, blocks);
// }
// template<typename Nested>
// static auto block_(std::initializer_list<std::initializer_list<Nested>> blocks, std::size_t depth = detail::nested_initialiser_list_depth<decltype(blocks)>::value){
//     using tensor_type = typename detail::nested_initialiser_list_value_type<Nested>::type;
//     static_assert(detail::is_tensor_v<tensor_type>);
//     using config_type = typename tensor_type::config_type;
//     using block_type = decltype(block_(*blocks.begin(), depth-1));

//     typename config_type::template container<block_type> blocks_{};
//     blocks_.reserve(blocks.size());
//     for (auto it = blocks.begin(); it!=blocks.end(); ++it){
//         blocks_.push_back(block_(*it, depth-1));
//     }
//     return concatenate_blocks(depth, blocks_);
// }

// //Split tensor and return container of slice views
// template<typename...Ts, typename IdxContainer>
// static auto split_by_points(const tensor<Ts...>& t, const IdxContainer& split_points, const typename tensor<Ts...>::size_type& direction){
//     using tensor_type = tensor<Ts...>;
//     using config_type = typename tensor_type::config_type;
//     using size_type = typename tensor_type::size_type;
//     using index_type = typename tensor_type::index_type;
//     using nop_type = typename slice_traits<config_type>::nop_type;
//     using slice_type = typename slice_traits<config_type>::slice_type;
//     using view_type = decltype(t(slice_type{},size_type{0}));
//     using res_type = typename config_type::template container<view_type>;
//     using res_size_type = typename res_type::size_type;
//     static_assert(detail::is_index_container_v<IdxContainer, index_type>);

//     const res_size_type parts_number = static_cast<res_size_type>(split_points.size()) + res_size_type{1};
//     if (direction >= t.dim()){
//         throw combine_exception("invalid split direction");
//     }

//     if (parts_number == res_size_type{1}){
//         return res_type(parts_number, t({},size_type{0}));
//     }else{
//         res_type res{};
//         res.reserve(parts_number);
//         auto split_points_it = std::begin(split_points);
//         index_type point{0};
//         do{
//             index_type next_point = *split_points_it;
//             res.push_back(t(slice_type{point, next_point},direction));
//             point = next_point;
//             ++split_points_it;
//         }while(split_points_it != std::end(split_points));
//         res.push_back(t(slice_type{point, nop_type{}},direction));
//         return res;
//     }
// }
// template<typename...Ts>
// static auto split_equal_parts(const tensor<Ts...>& t, const typename tensor<Ts...>::index_type& parts_number, const typename tensor<Ts...>::size_type& direction){
//     using tensor_type = tensor<Ts...>;
//     using config_type = typename tensor_type::config_type;
//     using size_type = typename tensor_type::size_type;
//     using index_type = typename tensor_type::index_type;
//     using slice_type = typename slice_traits<config_type>::slice_type;
//     using view_type = decltype(t(slice_type{},size_type{0}));
//     using res_type = typename config_type::template container<view_type>;
//     using res_size_type = typename res_type::size_type;

//     const res_size_type parts_number_ = static_cast<res_size_type>(parts_number);
//     if (direction >= t.dim()){
//         throw combine_exception("invalid split direction");
//     }
//     const index_type direction_size = t.descriptor().shape()[direction];
//     if (parts_number == index_type{0} || direction_size % parts_number != index_type{0}){
//         throw combine_exception("can't split in equal parts");
//     }

//     if (parts_number == index_type{1}){
//         return res_type(parts_number_, t({},size_type{0}));
//     }else{
//         res_type res{};
//         res.reserve(parts_number_);
//         index_type point{0};
//         const index_type part_size = direction_size/parts_number;
//         do{
//             index_type next_point = point+part_size;
//             res.push_back(t(slice_type{point,next_point},direction));
//             point = next_point;
//         }while(point!=direction_size);
//         return res;
//     }
// }

public:
//combiner interface
template<typename SizeT, typename...Us, typename...Ts>
static auto stack(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return stack_variadic(direction, t, ts...);
}
template<typename SizeT, typename...Us, typename...Ts>
static auto concatenate(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    return concatenate_variadic(direction, t, ts...);
}
template<typename SizeT, typename Container>
static auto concatenate(const SizeT& direction, const Container& ts){
    return concatenate_container(direction, ts);
}
template<typename...Ts>
static auto block(const std::tuple<Ts...>& blocks){
    return block_tuple(blocks);
}
// template<typename...Ts>
// static auto block(std::initializer_list<tensor<Ts...>> blocks){
//     return block_(blocks);
// }
// template<typename...Ts>
// static auto block(std::initializer_list<std::initializer_list<tensor<Ts...>>> blocks){
//     return block_(blocks);
// }
// template<typename...Ts>
// static auto block(std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>> blocks){
//     return block_(blocks);
// }
// template<typename...Ts>
// static auto block(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>>> blocks){
//     return block_(blocks);
// }
// template<typename...Ts, typename IdxContainer, typename SizeT, std::enable_if_t<detail::is_index_container_v<IdxContainer, typename tensor<Ts...>::index_type>,int> =0>
// static auto split(const tensor<Ts...>& t, const IdxContainer& split_points, const SizeT& direction){
//     return split_by_points(t, split_points, direction);
// }
// template<typename...Ts, typename SizeT>
// static auto split(const tensor<Ts...>& t, std::initializer_list<typename tensor<Ts...>::index_type> split_points, const SizeT& direction){
//     return split_by_points(t, split_points, direction);
// }
// template<typename...Ts>
// static auto split(const tensor<Ts...>& t, const typename tensor<Ts...>::index_type& parts_number, const typename tensor<Ts...>::size_type& direction){
//     return split_equal_parts(t, parts_number, direction);
// }
// template<typename Container>
// static auto vstack(const Container& ts){
//     static_assert(detail::is_tensor_container_v<Container>);
//     return vstack_container(ts);
// }
// template<typename...Ts, typename...Tensors>
// static auto vstack(const tensor<Ts...>& t, const Tensors&...ts){
//     return vstack_variadic(t, ts...);
// }

};  //end of class combiner

//combine module free functions
//call to combiner interface
template<typename SizeT, typename...Us, typename...Ts>
auto stack(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    static_assert((detail::is_tensor_v<Ts>&&...));
    using config_type = typename tensor<Us...>::config_type;
    return combiner_selector<config_type>::type::stack(direction, t, ts...);
}
template<typename SizeT, typename...Us, typename...Ts>
auto concatenate(const SizeT& direction, const tensor<Us...>& t, const Ts&...ts){
    static_assert((detail::is_tensor_v<Ts>&&...));
    using config_type = typename tensor<Us...>::config_type;
    return combiner_selector<config_type>::type::concatenate(direction, t, ts...);
}
template<typename SizeT, typename Container>
auto concatenate(const SizeT& direction, const Container& ts){
    static_assert(detail::is_tensor_container_v<Container>);
    using config_type = typename Container::value_type::config_type;
    return combiner_selector<config_type>::type::concatenate(direction, ts);
}
template<typename...Ts>
static auto block(const std::tuple<Ts...>& blocks){
    static_assert(detail::is_tensor_nested_tuple_v<std::tuple<Ts...>>);
    using config_type = detail::tensor_nested_tuple_config_type_t<std::tuple<Ts...>>;
    return combiner_selector<config_type>::type::block(blocks);
}
// template<typename...Ts>
// auto block(std::initializer_list<tensor<Ts...>> blocks){
//     using config_type = typename tensor<Ts...>::config_type;
//     return combiner_selector<config_type>::type::block(blocks);
// }
// template<typename...Ts>
// auto block(std::initializer_list<std::initializer_list<tensor<Ts...>>> blocks){
//     using config_type = typename tensor<Ts...>::config_type;
//     return combiner_selector<config_type>::type::block(blocks);
// }
// template<typename...Ts>
// auto block(std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>> blocks){
//     using config_type = typename tensor<Ts...>::config_type;
//     return combiner_selector<config_type>::type::block(blocks);
// }
// template<typename...Ts>
// auto block(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<tensor<Ts...>>>>> blocks){
//     using config_type = typename tensor<Ts...>::config_type;
//     return combiner_selector<config_type>::type::block(blocks);
// }
// template<typename...Ts, typename IdxContainer, typename SizeT, std::enable_if_t<detail::is_index_container_v<IdxContainer, typename tensor<Ts...>::index_type>,int> =0>
// auto split(const tensor<Ts...>& t, const IdxContainer& split_points, const SizeT& direction){
//     using config_type = typename tensor<Ts...>::config_type;
//     return combiner_selector<config_type>::type::split(t, split_points, direction);
// }
// template<typename...Ts, typename SizeT>
// auto split(const tensor<Ts...>& t, std::initializer_list<typename tensor<Ts...>::index_type> split_points, const SizeT& direction){
//     using config_type = typename tensor<Ts...>::config_type;
//     return combiner_selector<config_type>::type::split(t, split_points, direction);
// }
// template<typename...Ts>
// auto split(const tensor<Ts...>& t, const typename tensor<Ts...>::index_type& parts_number, const typename tensor<Ts...>::size_type& direction){
//     using config_type = typename tensor<Ts...>::config_type;
//     return combiner_selector<config_type>::type::split(t, parts_number, direction);
// }
// template<typename Container>
// auto vstack(const Container& ts){
//     static_assert(detail::is_tensor_container_v<Container>);
//     using tensor_type = typename Container::value_type;
//     using config_type = typename tensor_type::config_type;
//     return combiner_selector<config_type>::type::vstack(ts);
// }
// template<typename...Ts, typename...Tensors>
// auto vstack(const tensor<Ts...>& t, const Tensors&...ts){
//     using tensor_type = tensor<Ts...>;
//     using config_type = typename tensor_type::config_type;
//     return combiner_selector<config_type>::type::vstack(t, ts...);
// }

// //call to free functions
// template<typename...Ts, typename IdxContainer_PartsNumber>
// auto hsplit(const tensor<Ts...>& t, const IdxContainer_PartsNumber& split_arg){
//     using size_type = typename tensor<Ts...>::size_type;
//     size_type direction = t.dim() == size_type{1} ? size_type{0} : size_type{1};
//     return split(t, split_arg, direction);
// }
// template<typename...Ts, typename IdxContainer_PartsNumber>
// auto vsplit(const tensor<Ts...>& t, const IdxContainer_PartsNumber& split_arg){
//     using size_type = typename tensor<Ts...>::size_type;
//     return split(t, split_arg, size_type{0});
// }


// //container is homogeneous, types of tensors must be the same
// //add variadic, init list overloads
// // template<typename Container>
// // auto hstack(const Container& ts){
// //     static_assert(detail::is_tensor_container_v<Container>);
// //     using size_type = typename Container::value_type::size_type;
// //     size_type direction = t.dim() == size_type{1} ? size_type{0} : size_type{1};
// //     return concatenate(direction, ts);
// // }

// //add vstack to combiner interface and add implementation not to repack if there are 1-d tensors in ts???
// // template<typename Container>
// // auto vstack(const Container& ts){
// //     static_assert(detail::is_tensor_container_v<Container>);
// //     return
// //     // using tensor_type = typename Container::value_type;
// //     // using config_type = typename tensor_type::config_type;
// //     // using size_type = typename tensor_type::size_type;
// //     //typename config_type::template container<>
// //     //if at least one 1-d tensor in ts, than need reshape all tensors, type to be the same and utilize concatenate(dir, new_container), or need separate implementation
// //     //or parameterize existing impl
// //     //if dim == 1 reshape to (1,shape[0])
// //     //if dim > 1 trivial reshape()
// //     //


// //     //return concatenate(size_type{0}, ts);
// // }


}   //end of namespace gtensor

#endif