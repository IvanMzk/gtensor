#ifndef BUILDER_HPP_
#define BUILDER_HPP_

#include "common.hpp"
#include "module_selector.hpp"
#include "tensor.hpp"

namespace gtensor{

namespace detail{

template<typename IdxT>
void check_eye_args(const IdxT& n,const IdxT& m, const IdxT& k){

}

}   //end of namespace detail

//bulder module implementation

struct builder
{

    // template<typename T, typename ShT, typename Order = config::c_order, typename Config = config::default_config>
    // static auto empty(ShT&& shape, Order order = config::c_order{}, Config config = config::default_config{}){
    //     return tensor<T,Order>(std::forward<ShT>(shape));
    // }

    // template<typename T, typename Config = config::default_config, typename Order = config::c_order, typename ShT>
    // static auto empty(ShT&& shape, Order order = Order{}){
    //     return tensor<T,Order,config::extend_config_t<Config,T>>(std::forward<ShT>(shape));
    // }
    // template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename U>
    // static auto empty(std::initializer_list<U> shape){
    //     using tensor_type = tensor<T,Order,config::extend_config_t<Config,T>>;
    //     using shape_type = typename tensor_type::shape_type;
    //     return tensor<T,Order,config::extend_config_t<Config,T>>(shape_type(shape));
    // }

    template<typename T, typename Order, typename Config, typename ShT>
    static auto empty(ShT&& shape){
        ASSERT_ORDER(Order);
        using tensor_type = tensor<T,Order,config::extend_config_t<Config,T>>;
        using shape_type = typename tensor_type::shape_type;
        return tensor_type(detail::make_shape_of_type<shape_type>(std::forward<ShT>(shape)));
    }

    template<typename T, typename Order, typename Config, typename ShT, typename U>
    static auto full(ShT&& shape, const U& v){
        ASSERT_ORDER(Order);
        using tensor_type = tensor<T,Order,config::extend_config_t<Config,T>>;
        using shape_type = typename tensor_type::shape_type;
        return tensor_type(detail::make_shape_of_type<shape_type>(std::forward<ShT>(shape)),v);
    }

    template<typename T, typename Order, typename Config, typename ShT>
    static auto zeros(ShT&& shape){
        return full<T,Order,Config>(std::forward<ShT>(shape),T{0});
    }

    template<typename T, typename Order, typename Config, typename ShT>
    static auto ones(ShT&& shape){
        return full<T,Order,Config>(std::forward<ShT>(shape),T{1});
    }

    template<typename T, typename Order, typename Config, typename IdxT>
    static auto eye(const IdxT& n_, const IdxT& m_, const IdxT& k_){
        ASSERT_ORDER(Order);
        using tensor_type = tensor<T,Order,config::extend_config_t<Config,T>>;
        using index_type = typename tensor_type::index_type;
        auto res = zeros<T,Order,Config>(std::initializer_list<IdxT>{n_,m_});
        if (!res.empty()){
            const auto n = static_cast<index_type>(n_);
            const auto m = static_cast<index_type>(m_);
            const auto k = static_cast<index_type>(k_);
            if (k>-n && k<m){
                const auto d = k>=0 ? std::min(n,m-k) : std::min(n+k,m);
                auto it = res.template traverse_order_adapter<Order>().begin();
                index_type step{0};
                if constexpr (std::is_same_v<Order,config::c_order>){
                    step = m+1;
                    if (k>=0){
                        it+=k;
                    }else{
                        it+=(-k)*m;
                    }
                }else{  //f_order
                    step = n+1;
                    if (k>=0){
                        it+=(k)*n;
                    }else{
                        it+=-k;
                    }
                }
                index_type i{1};
                while(true){
                    *it = T{1};
                    if (i==d){
                        break;
                    }else{
                        ++i;
                        it+=step;
                    }
                }
            }
        }
        return res;
    }

    template<typename T, typename Order, typename Config, typename IdxT>
    static auto identity(const IdxT& n){
        ASSERT_ORDER(Order);
        return eye<T,Order,Config>(n,n,IdxT{0});
    }

    template<typename...Ts>
    static auto empty_like(const basic_tensor<Ts...>& t){
        using tensor_type = basic_tensor<Ts...>;
        using value_type = typename tensor_type::value_type;
        using order = typename tensor_type::order;
        using config_type = typename tensor_type::config_type;
        return empty<value_type,order,config_type>(t.shape());
    }

    template<typename U, typename...Ts>
    static auto full_like(const basic_tensor<Ts...>& t, const U& v){
        using tensor_type = basic_tensor<Ts...>;
        using value_type = typename tensor_type::value_type;
        using order = typename tensor_type::order;
        using config_type = typename tensor_type::config_type;
        return full<value_type,order,config_type>(t.shape(),v);
    }

    template<typename...Ts>
    static auto zeros_like(const basic_tensor<Ts...>& t){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return full_like(t,value_type{0});
    }

    template<typename...Ts>
    static auto ones_like(const basic_tensor<Ts...>& t){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return full_like(t,value_type{1});
    }

};  //end of struct builder

//builder module frontend

//make tensor of given shape
//tensor elements initialization depends on underlaying storage implementation: does storage initialize elements when constructed like this: storage_type(n), n is storage size
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename ShT>
static auto empty(ShT&& shape){
    return builder_selector_t<Config>::template empty<T,Order,Config>(std::forward<ShT>(shape));
}
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename U>
static auto empty(std::initializer_list<U> shape){
    return builder_selector_t<Config>::template empty<T,Order,Config>(shape);
}

//make tensor of given shape, initialized with v
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename ShT, typename V>
auto full(ShT&& shape, const V& v){
    return builder_selector_t<Config>::template full<T,Order,Config>(std::forward<ShT>(shape), v);
}
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename U, typename V>
auto full(std::initializer_list<U> shape, const V& v){
    return builder_selector_t<Config>::template full<T,Order,Config>(shape, v);
}

//make tensor of given shape, initialized with zeros
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename ShT>
auto zeros(ShT&& shape){
    return builder_selector_t<Config>::template zeros<T,Order,Config>(std::forward<ShT>(shape));
}
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename U>
auto zeros(std::initializer_list<U> shape){
    return builder_selector_t<Config>::template zeros<T,Order,Config>(shape);
}

//make tensor of given shape, initialized with ones
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename ShT>
auto ones(ShT&& shape){
    return builder_selector_t<Config>::template ones<T,Order,Config>(std::forward<ShT>(shape));
}
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename U>
auto ones(std::initializer_list<U> shape){
    return builder_selector_t<Config>::template ones<T,Order,Config>(shape);
}

//make tensor of shape (n,n) with ones on main diagonal
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename IdxT>
auto identity(const IdxT& n){
    return builder_selector_t<Config>::template identity<T,Order,Config>(n);
}

//make tensor of shape (n,m) with ones on kth diagonal
//k=0 refers to the main diagonal, k>0 refers to an upper diagonal, k<0 to lower diagonal
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename IdxT = int>
auto eye(const IdxT& n, const IdxT& m, const IdxT& k=0){
    return builder_selector_t<Config>::template eye<T,Order,Config>(n,m,k);
}

//make tensor of the same shape,layout,value_type,config_type as t
//elements initialization depends on underlaying storage implementation
template<typename...Ts>
auto empty_like(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::empty_like(t);
}

//make tensor of the same shape,layout,value_type,config_type as t, initialized with v
template<typename U, typename...Ts>
auto full_like(const basic_tensor<Ts...>& t, const U& v){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::full_like(t,v);
}

//make tensor of the same shape,layout,value_type,config_type as t, initialized with zeros
template<typename...Ts>
auto zeros_like(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::zeros_like(t);
}

//make tensor of the same shape,layout,value_type,config_type as t, initialized with ones
template<typename...Ts>
auto ones_like(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::ones_like(t);
}

}   //end of namespace gtensor
#endif