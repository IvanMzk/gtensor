#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include "type_selector.hpp"
#include "tensor_factory.hpp"
#include "broadcast.hpp"

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
    using index_type = typename ShT::value_type;
    const dim_type pdim = pshape.size();
    if (direction >= pdim){
        throw reduce_exception("bad reduce direction");
    }
    if (pshape[direction] == index_type{0}){
        throw reduce_exception("reduce on zero size direction");
    }
}
template<typename ShT, typename SizeT>
auto make_reduce_shape(const ShT& shape, const SizeT& direction){
    using dim_type = SizeT;
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    dim_type dim = shape.size();
    if (dim == dim_type{1}){
        return shape_type{index_type{1}};
    }else{
        shape_type res(--dim);
        auto shape_stop = shape.begin()+direction;
        std::copy(shape.begin(), shape_stop, res.begin());
        std::copy(++shape_stop, shape.end(), res.begin()+direction);
        return res;
    }
}

template<typename CfgT, typename Walker>
class walker_reducer_adapter : public walker_forward_adapter<CfgT, Walker>
{
protected:
    using walker_forward_adapter_base = walker_forward_adapter<CfgT, Walker>;
    using typename walker_forward_adapter_base::config_type;
    using typename walker_forward_adapter_base::walker_type;
    using typename walker_forward_adapter_base::shape_type;
    using typename walker_forward_adapter_base::index_type;
    using typename walker_forward_adapter_base::dim_type;
    using walker_forward_adapter_base::walker_;
    using walker_forward_adapter_base::dim_;
    using walker_forward_adapter_base::index_;
    using walker_forward_adapter_base::shape_;

    index_type reduce_direction_size_;
    dim_type reduce_direction_;
    index_type reduce_counter_{reduce_direction_size_};

public:
    template<typename Walker_>
    walker_reducer_adapter(const shape_type& shape__, Walker_&& walker__, const dim_type direction__):
        walker_forward_adapter_base{shape__, std::forward<Walker_>(walker__)},
        reduce_direction_size_{shape__[direction__]},
        reduce_direction_{dim_ - dim_type{1} - direction__}
    {}

    bool next(){
        walker_.reset(reduce_direction_);
        reduce_counter_ = reduce_direction_size_;
        dim_type direction{0}; //start from direction with min stride
        auto index_it = index_.end();
        for(;direction!=dim_;++direction){
            if (direction == reduce_direction_){
                continue;
            }else{
                if (*--index_it == shape_.element(direction)-index_type{1}){   //direction at their max
                    *index_it = index_type{0};
                    walker_.reset(direction);
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

namespace reduce_operations{

struct max
{
    template<typename T>
    auto operator()(const T& u, const T& v){
        return std::max(u,v);
    }
};
struct min
{
    template<typename T>
    auto operator()(const T& u, const T& v){
        return std::min(u,v);
    }
};

}   //end of namespace reduce_operations

class reducer
{
    template<typename ImplT, typename BinaryOp>
    static auto reduce_(const ImplT& parent, const typename ImplT::dim_type& direction, BinaryOp op){
        using value_type = typename ImplT::value_type;
        using config_type = typename ImplT::config_type;
        using dim_type = typename config_type::dim_type;
        using res_value_type = std::decay_t<decltype(op(std::declval<value_type>(),std::declval<value_type>()))>;

        auto res = storage_tensor_factory<config_type,res_value_type>::make(detail::make_reduce_shape(parent.shape(), direction), res_value_type{});
        if (!res.empty()){
            auto pdim = parent.dim();
            if (pdim == dim_type{1}){
                auto pit = parent.engine().begin();
                auto init = *pit;
                *res.begin() = std::accumulate(++pit, parent.engine().end(), init, op);
            }else{
                auto it = detail::walker_reducer_adapter<config_type, decltype(parent.engine().create_walker())>{parent.shape(), parent.engine().create_walker(), direction};
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
        return res;
    }

    template<typename BinaryOp, typename...Ts>
    static auto reduce_(const tensor<Ts...>& t, const typename tensor<Ts...>::dim_type& direction, BinaryOp op){
        return reduce_(t.impl_ref(), direction, op);
    }
public:
    //interface
    template<typename BinaryOp, typename...Ts>
    static auto reduce(const tensor<Ts...>& t, const typename tensor<Ts...>::dim_type& direction, BinaryOp op){
        return reduce_(t,direction,op);
    }
};

template<typename BinaryOp, typename...Ts>
auto reduce(const tensor<Ts...>& t, const typename tensor<Ts...>::dim_type& direction, BinaryOp op){
    using config_type = typename tensor<Ts...>::config_type;
    return reducer_selector<config_type>::type::reduce(t, direction, op);
}



}   //end of namespace gtensor

#endif