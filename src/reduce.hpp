#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include "type_selector.hpp"
#include "tensor.hpp"

namespace gtensor{

class reduce_exception : public std::runtime_error{
public:
    explicit reduce_exception(const char* what):
        runtime_error(what)
    {}
};

namespace detail{

template<typename ShT, typename SizeT>
auto check_reduce_args(const ShT& pshape, const SizeT& direction){
    using dim_type = SizeT;
    const dim_type pdim = pshape.size();
    if (direction >= pdim){
        throw reduce_exception("bad reduce direction");
    }
}
template<typename ShT, typename SizeT>
auto make_reduce_shape(const ShT& shape, const SizeT& direction){
    using dim_type = SizeT;
    using shape_type = ShT;
    dim_type dim = shape.size();
    shape_type res(--dim);
    auto shape_stop = shape.begin()+direction;
    std::copy(shape.begin(), shape_stop, res.begin());
    std::copy(++shape_stop, shape.end(), res.begin()+direction);
    return res;
}

template<typename Config, typename Walker>
class walker_reduce_forward_traverser
{
    using config_type = Config;
    using walker_type = Walker;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using dim_type = typename config_type::dim_type;

    const shape_type* shape_;
    walker_type walker_;
    const dim_type reduce_direction_;
    const dim_type dim_;
    shape_type index_;
    const index_type reduce_direction_size_{(*shape_)[reduce_direction_]};
    index_type reduce_counter_{reduce_direction_size_};

public:
    template<typename Walker_>
    walker_reduce_forward_traverser(const shape_type& shape__, Walker_&& walker__, const dim_type direction__):
        shape_{&shape__},
        walker_{std::forward<Walker_>(walker__)},
        reduce_direction_{direction__},
        dim_{shape_->size()},
        index_(dim_, index_type{0})
    {}

    const auto& walker()const{return walker_;}

    bool next(){
        walker_.reset_back(reduce_direction_);
        reduce_counter_ = reduce_direction_size_;
        auto direction = dim_;
        auto index_it = index_.end();
        while(direction!=dim_type{0}){
            --direction;
            --index_it;
            if (direction == reduce_direction_){
                continue;
            }else{
                if (*index_it == (*shape_)[direction]-index_type{1}){   //direction at their max
                    *index_it = index_type{0};
                    walker_.reset_back(direction);
                }else{  //can next on direction
                    ++(*index_it);
                    walker_.step(direction);
                    return true;
                }
            }
        }
        return false;
    }

    bool next_reduce(){
        if (--reduce_counter_ == index_type{0}){
            return false;
        }else{
            walker_.step(reduce_direction_);
            return true;
        }
    }
};

}   //end of namespace detail

class reducer
{
    template<typename BinaryOp, typename...Ts>
    static auto reduce_(const basic_tensor<Ts...>& parent, const typename basic_tensor<Ts...>::dim_type& direction, BinaryOp op){
        using parent_type = basic_tensor<Ts...>;
        using value_type = typename parent_type::value_type;
        using config_type = typename parent_type::config_type;
        using dim_type = typename config_type::dim_type;
        using index_type = typename config_type::index_type;
        using res_value_type = std::decay_t<decltype(op(std::declval<value_type>(),std::declval<value_type>()))>;
        const auto& pshape = parent.shape();
        detail::check_reduce_args(pshape, direction);
        auto res = tensor<res_value_type,config_type>{detail::make_reduce_shape(pshape, direction)};
        index_type reduce_direction_size = pshape[direction];
        if (!res.empty()){
            if (reduce_direction_size == index_type{0}){    //fill with default
                if constexpr (std::is_default_constructible_v<value_type>){
                    detail::fill(res.begin(), res.end(), value_type{});
                }else{
                    throw reduce_exception("reduce can't fill result, value_type is not default constructible");
                }
            }else{
                auto pdim = parent.dim();
                if (pdim == dim_type{1}){
                    auto pit = parent.begin();
                    auto init = *pit;
                    *res.begin() = std::accumulate(++pit, parent.end(), init, op);
                }else{
                    auto it = detail::walker_reduce_forward_traverser<config_type, decltype(parent.create_walker())>{pshape, parent.create_walker(), direction};
                    auto res_it = res.begin();
                    do{
                        auto init = *it.walker();
                        while(it.next_reduce()){
                            init = op(init, *it.walker());
                        }
                        *res_it = init;
                        ++res_it;
                    }while(it.next());
                }
            }
        }
        return res;
    }
public:
    //interface
    template<typename BinaryOp, typename...Ts>
    static auto reduce(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::dim_type& direction, BinaryOp op){
        return reduce_(t,direction,op);
    }
};

template<typename BinaryOp, typename...Ts>
auto reduce(const basic_tensor<Ts...>& t, const typename basic_tensor<Ts...>::dim_type& direction, BinaryOp op){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return reducer_selector_t<config_type>::reduce(t, direction, op);
}

}   //end of namespace gtensor
#endif