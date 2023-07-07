#ifndef BUILDER_HPP_
#define BUILDER_HPP_

#include "common.hpp"
#include "module_selector.hpp"
#include "math.hpp"
#include "tensor.hpp"
#include "tensor_operators.hpp"
#include "reduce.hpp"

namespace gtensor{

//bulder module implementation

struct builder
{
    //build from shape and value
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

    //build from numerical ranges
    template<typename T, typename Order, typename Config, typename U>
    static auto arange(const U& start, const U& stop, const U& step){
        ASSERT_ORDER(Order);
        static_assert(math::numeric_traits<U>::is_integral() || math::numeric_traits<U>::is_floating_point(),"arange arguments must be of numeric type");
        using tensor_type = tensor<T,Order,config::extend_config_t<Config,T>>;
        using value_type = typename tensor_type::value_type;
        using index_type = typename tensor_type::index_type;
        using shape_type = typename tensor_type::shape_type;
        using integral_type = math::make_integral_t<U>;
        using fp_type = math::make_floating_point_t<U>;
        const auto n = static_cast<index_type>(static_cast<integral_type>(math::ceil((stop-start)/static_cast<fp_type>(step))));
        tensor_type res(shape_type{n});
        const auto& step_ = static_cast<const value_type&>(step);
        auto start_ = static_cast<value_type>(start);
        for (auto it=res.begin(),last=res.end(); it!=last; ++it,start_+=step_){
            *it = start_;
        }
        return res;
    }

    template<typename T, typename Order, typename Config, typename Start, typename Stop, typename U, typename DimT>
    static auto linspace(const Start& start, const Stop& stop, const U& num, bool end_point, const DimT& axis){
        auto generator = [end_point](auto first, auto last, const auto& start, const auto& stop, const auto& num){
            using num_type = std::remove_cv_t<std::remove_reference_t<decltype(num)>>;
            using fp_type = math::make_floating_point_t<num_type>;
            const auto intervals_n = end_point ?  static_cast<fp_type>(num-1) : static_cast<fp_type>(num);
            const auto step = (stop-start)/intervals_n;
            auto start_ = static_cast<fp_type>(start);
            for(;first!=last; ++first,start_+=step){
                *first = start_;
            }
        };
        return make_space<T,Order,Config>(start,stop,num,axis,generator);
    }

    template<typename T, typename Order, typename Config, typename Start, typename Stop, typename U, typename Base, typename DimT>
    static auto logspace(const Start& start, const Stop& stop, const U& num, bool end_point, const Base& base, const DimT& axis){
        auto generator = [end_point,base](auto first, auto last, const auto& start, const auto& stop, const auto& num){
            using num_type = std::remove_cv_t<std::remove_reference_t<decltype(num)>>;
            using fp_type = math::make_floating_point_t<num_type>;
            const auto intervals_n = end_point ?  static_cast<fp_type>(num-1) : static_cast<fp_type>(num);
            const auto step = (stop-start)/intervals_n;
            auto start_ = static_cast<fp_type>(start);
            auto base_ = static_cast<fp_type>(base);
            for(;first!=last; ++first,start_+=step){
                *first = math::pow(base_,start_);
            }
        };
        return make_space<T,Order,Config>(start,stop,num,axis,generator);
    }

    template<typename T, typename Order, typename Config, typename Start, typename Stop, typename U, typename DimT>
    static auto geomspace(const Start& start, const Stop& stop, const U& num, bool end_point, const DimT& axis){
        auto generator = [end_point](auto first, auto last, const auto& start, const auto& stop, const auto& num){
            using num_type = std::remove_cv_t<std::remove_reference_t<decltype(num)>>;
            using fp_type = math::make_floating_point_t<num_type>;
            const auto intervals_n = end_point ?  static_cast<fp_type>(num-1) : static_cast<fp_type>(num);
            const auto step = math::pow(static_cast<fp_type>(stop)/static_cast<fp_type>(start),1/intervals_n);
            auto start_ = static_cast<fp_type>(start);
            for(;first!=last; ++first,start_*=step){
                *first = start_;
            }
        };
        return make_space<T,Order,Config>(start,stop,num,axis,generator);
    }



private:

    template<typename ShT, typename IdxT, typename DimT>
    static auto make_space_shape(const ShT& shape, const IdxT& num, const DimT& axis){
        const auto dim = detail::make_dim(shape);
        ShT res(dim+1);
        std::copy(shape.begin(), shape.begin()+axis, res.begin());
        std::copy(shape.begin()+axis, shape.end(), res.begin()+(axis+1));
        res[axis] = num;
        return res;
    }

    template<typename T, typename Order, typename Config, typename Start, typename Stop, typename U, typename DimT, typename Generator>
    static auto make_space(const Start& start, const Stop& stop, const U& num, const DimT& axis_, Generator generator){
        static constexpr bool is_start_numeric = math::numeric_traits<Start>::is_integral() || math::numeric_traits<Start>::is_floating_point();
        static constexpr bool is_stop_numeric = math::numeric_traits<Stop>::is_integral() || math::numeric_traits<Stop>::is_floating_point();
        static_assert(math::numeric_traits<U>::is_integral(),"num must be of integral type");
        static_assert(is_start_numeric || detail::is_tensor_v<Start>,"Start must be of numeric or tensor type");
        static_assert(is_stop_numeric || detail::is_tensor_v<Stop>,"Stop must be of numeric or tensor type");
        using tensor_type = tensor<T,Order,config::extend_config_t<Config,T>>;
        using config_type = typename tensor_type::config_type;
        using dim_type = typename tensor_type::dim_type;
        using index_type = typename tensor_type::index_type;
        using shape_type = typename tensor_type::shape_type;
        const auto n = static_cast<index_type>(num);
        if constexpr (is_start_numeric && is_stop_numeric){
            tensor_type res(shape_type{n});
            generator(res.begin(),res.end(),start,stop,num);
            return res;
        }else{  //start or stop or both are tensors
            auto intervals = n_operator(
                [](const auto& start_element, const auto& stop_element){
                    return std::make_pair(start_element,stop_element);
                },
                start,
                stop
            );
            const auto res_dim = intervals.dim()+1;
            const auto axis = detail::make_axis(res_dim, axis_);
            tensor_type res(make_space_shape(intervals.shape(),n,axis));
            if (!res.empty()){
                using predicate_type = detail::reduce_traverse_predicate<config_type, dim_type>;
                using res_traverser_type = walker_forward_traverser<config_type, decltype(res.create_walker()), predicate_type>;
                predicate_type predicate{axis, true};
                const auto& res_shape = res.shape();
                res_traverser_type res_traverser{res_shape, res.create_walker(), predicate};
                auto a = intervals.template traverse_order_adapter<Order>();
                for (auto it=a.begin(),last=a.end(); it!=last; ++it,res_traverser.template next<Order>()){
                    const auto& interval = *it;
                    generator(
                        detail::make_axis_iterator(res_traverser.walker(),axis,index_type{0}),
                        detail::make_axis_iterator(res_traverser.walker(),axis,n),
                        interval.first,
                        interval.second,
                        num
                    );
                }
            }
            return res;
        }
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

//make 1d tensor of evenly spaced values whithin a given interval
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename U>
auto arange(const U& start, const U& stop, const U& step){
    return builder_selector_t<Config>::template arange<T,Order,Config>(start,stop,step);
}
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename U>
auto arange(const U& stop){
    return builder_selector_t<Config>::template arange<T,Order,Config>(U{0},stop,U{1});
}

//make tensor of num evenly spaced samples, calculated over the interval start, stop
template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename U, typename DimT>
auto linspace(const Start& start, const Stop& stop, const U& num, bool end_point, const DimT& axis){
    return builder_selector_t<Config>::template linspace<T,Order,Config>(start,stop,num,end_point,axis);
}

template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename U, typename Base, typename DimT>
auto logspace(const Start& start, const Stop& stop, const U& num, bool end_point, const Base& base, const DimT& axis){
    return builder_selector_t<Config>::template logspace<T,Order,Config>(start,stop,num,end_point,base,axis);
}

template<typename T, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename U, typename DimT>
auto geomspace(const Start& start, const Stop& stop, const U& num, bool end_point, const DimT& axis){
    return builder_selector_t<Config>::template geomspace<T,Order,Config>(start,stop,num,end_point,axis);
}


}   //end of namespace gtensor
#endif