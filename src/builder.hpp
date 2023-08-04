#ifndef BUILDER_HPP_
#define BUILDER_HPP_

#include "module_selector.hpp"
#include "common.hpp"
#include "math.hpp"
#include "indexing.hpp"

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
        auto res = zeros<T,Order,Config>(std::initializer_list<IdxT>{n_,m_});
        if (!res.empty()){
            traverse_diagonal<Order>(
                res.traverse_order_adapter(Order{}).begin(),
                [](auto& e){e=T{1};},
                n_,
                m_,
                k_
            );
        }
        return res;
    }

    template<typename T, typename Order, typename Config, typename IdxT>
    static auto identity(const IdxT& n){
        ASSERT_ORDER(Order);
        return eye<T,Order,Config>(n,n,IdxT{0});
    }

    template<typename ShT = detail::no_value, typename...Ts>
    static auto empty_like(const basic_tensor<Ts...>& t, ShT&& shape=ShT{}){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using config_type = typename tensor_type::config_type;
        using value_type = typename tensor_type::value_type;
        using shape_type = typename tensor_type::shape_type;
        using index_type = typename tensor_type::index_type;
        using ShT_ = std::remove_cv_t<std::remove_reference_t<ShT>>;
        static constexpr bool is_shape = detail::is_container_of_type_v<ShT_,index_type>;
        static_assert(is_shape || std::is_same_v<ShT_,detail::no_value>,"shape must be no_value or container");
        if constexpr (is_shape){
            return empty<value_type,order,config_type>(detail::make_shape_of_type<shape_type>(std::forward<ShT>(shape)));
        }else{
            return empty<value_type,order,config_type>(t.shape());
        }
    }

    template<typename U, typename ShT=detail::no_value, typename...Ts>
    static auto full_like(const basic_tensor<Ts...>& t, const U& v, ShT&& shape=ShT{}){
        using tensor_type = basic_tensor<Ts...>;
        using value_type = typename tensor_type::value_type;
        using order = typename tensor_type::order;
        using config_type = typename tensor_type::config_type;
        using shape_type = typename tensor_type::shape_type;
        using index_type = typename tensor_type::index_type;
        using ShT_ = std::remove_cv_t<std::remove_reference_t<ShT>>;
        static constexpr bool is_shape = detail::is_container_of_type_v<ShT_,index_type>;
        static_assert(is_shape || std::is_same_v<ShT_,detail::no_value>,"shape must be no_value or container");
        if constexpr (is_shape){
            return full<value_type,order,config_type>(detail::make_shape_of_type<shape_type>(std::forward<ShT>(shape)),v);
        }else{
            return full<value_type,order,config_type>(t.shape(),v);
        }
    }

    template<typename ShT=detail::no_value, typename...Ts>
    static auto zeros_like(const basic_tensor<Ts...>& t, ShT&& shape=ShT{}){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return full_like(t,value_type{0},shape);
    }

    template<typename ShT=detail::no_value, typename...Ts>
    static auto ones_like(const basic_tensor<Ts...>& t, ShT&& shape=ShT{}){
        using value_type = typename basic_tensor<Ts...>::value_type;
        return full_like(t,value_type{1},shape);
    }

    //build from numerical ranges
    //T may be numeric type or no_value, if no_value result tensor value_type is made from Start,Stop,Step types
    template<typename T, typename Order, typename Config, typename Start, typename Stop, typename Step>
    static auto arange(const Start& start, const Stop& stop, const Step& step){
        ASSERT_ORDER(Order);
        using common_value_type = std::common_type_t<Start,Stop,Step>;
        static_assert(math::numeric_traits<common_value_type>::is_integral() || math::numeric_traits<common_value_type>::is_floating_point(),"arange arguments must be of numeric type");
        using value_type = std::conditional_t<std::is_same_v<T,detail::no_value>,common_value_type,T>;
        using tensor_type = tensor<value_type,Order,config::extend_config_t<Config,T>>;
        using index_type = typename tensor_type::index_type;
        using shape_type = typename tensor_type::shape_type;
        using integral_type = math::make_integral_t<common_value_type>;
        using fp_type = math::make_floating_point_t<common_value_type>;
        auto n = static_cast<index_type>(static_cast<integral_type>(math::ceil((stop-start)/static_cast<fp_type>(step))));
        n = n > 0 ? n : index_type{0};
        tensor_type res(shape_type{n});
        const auto& step_ = static_cast<const value_type&>(step);
        auto start_ = static_cast<value_type>(start);
        for (auto it=res.begin(),last=res.end(); it!=last; ++it,start_+=step_){
            *it = start_;
        }
        return res;
    }

    template<typename T, typename Order, typename Config, typename Start, typename Stop, typename Num, typename DimT>
    static auto linspace(const Start& start, const Stop& stop, const Num& num, bool end_point, const DimT& axis){
        auto generator = [end_point](auto first, auto last, const auto& start, const auto& stop, const auto& num){
            using fp_type = typename std::iterator_traits<decltype(first)>::value_type;
            const auto intervals_n = end_point ?  static_cast<fp_type>(num-1) : static_cast<fp_type>(num);
            const auto step = (stop-start)/intervals_n;
            auto start_ = static_cast<fp_type>(start);
            for(;first!=last; ++first,start_+=step){
                *first = start_;
            }
        };
        return make_space<T,Order,Config>(start,stop,num,axis,generator);
    }

    template<typename T, typename Order, typename Config, typename Start, typename Stop, typename Num, typename Base, typename DimT>
    static auto logspace(const Start& start, const Stop& stop, const Num& num, bool end_point, const Base& base, const DimT& axis){
        auto generator = [end_point,base](auto first, auto last, const auto& start, const auto& stop, const auto& num){
            using fp_type = typename std::iterator_traits<decltype(first)>::value_type;
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

    template<typename T, typename Order, typename Config, typename Start, typename Stop, typename Num, typename DimT>
    static auto geomspace(const Start& start, const Stop& stop, const Num& num, bool end_point, const DimT& axis){
        auto generator = [end_point](auto first, auto last, const auto& start, const auto& stop, const auto& num){
            using fp_type = typename std::iterator_traits<decltype(first)>::value_type;
            check_geomspace_args(start,stop);
            const auto intervals_n = end_point ?  static_cast<fp_type>(num-1) : static_cast<fp_type>(num);
            const auto step = math::pow(static_cast<fp_type>(stop)/static_cast<fp_type>(start),1/intervals_n);
            auto start_ = static_cast<fp_type>(start);
            for(;first!=last; ++first,start_*=step){
                *first = start_;
            }
        };
        return make_space<T,Order,Config>(start,stop,num,axis,generator);
    }

    template<typename IdxT, typename...Ts>
    static auto diag(const basic_tensor<Ts...>& t, const IdxT& k_){
        using tensor_type = basic_tensor<Ts...>;
        using order = typename tensor_type::order;
        using value_type = typename tensor_type::value_type;
        using config_type = typename tensor_type::config_type;
        using index_type = typename tensor_type::index_type;
        using shape_type = typename tensor_type::shape_type;
        static_assert(math::numeric_traits<IdxT>::is_integral(),"k must be of integral type");
        const auto& k = static_cast<const index_type&>(k_);
        if (t.dim() == 1){
            const auto d = t.size();
            const auto n = k>=0 ? k+d : -k+d;
            auto res = zeros<value_type,order,config_type>(shape_type{n,n});
            traverse_diagonal<order>(
                res.traverse_order_adapter(order{}).begin(),
                [it=t.traverse_order_adapter(order{}).begin()](auto& e)mutable{
                    e = *it;
                    ++it;
                },
                n,
                n,
                k
            );
            return res;
        }else if (t.dim() == 2){    //return copy of kth diagonal
            const auto& shape = t.shape();
            const auto& n = shape[0];
            const auto& m = shape[1];
            const auto d = make_diagonal_size(n,m,k);
            auto res = empty<value_type,order,config_type>(shape_type{d});
            traverse_diagonal<order>(
                t.traverse_order_adapter(order{}).begin(),
                [it=res.traverse_order_adapter(order{}).begin()](auto& e)mutable{
                    *it = e;
                    ++it;
                },
                n,
                m,
                k
            );
            return res;
        }else{
            throw value_error("input must be 1d or 2d");
        }
    }

private:

    template<typename U>
    static void check_make_space_args(const U& num){
        if (num < U{0}){
            throw value_error("number of samples, must be non-negative");
        }
    }
    template<typename Start, typename Stop>
    static void check_geomspace_args(const Start& start, const Stop& stop){
        if (start == Start{0} || stop == Stop{0}){
            throw value_error("geometric sequence cannot include zero");
        }
    }

    template<typename IdxT>
    static auto make_diagonal_size(const IdxT& n, const IdxT& m, const IdxT& k){
        return n!=0 && m!=0 && k>-n && k<m ? (k>=0 ? std::min(n,m-k) : std::min(n+k,m)) : IdxT{0};
    }

    template<typename Order, typename It, typename UnaryOp, typename IdxT>
    static void traverse_diagonal(It first, UnaryOp op, const IdxT& n_, const IdxT& m_, const IdxT& k_){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        const auto n = static_cast<difference_type>(n_);
        const auto m = static_cast<difference_type>(m_);
        const auto k = static_cast<difference_type>(k_);
        const auto d = make_diagonal_size(n,m,k);
        if (d!=0){
            difference_type step{0};
            if constexpr (std::is_same_v<Order,config::c_order>){
                step = m+1;
                if (k>=0){
                    first+=k;
                }else{
                    first+=(-k)*m;
                }
            }else{  //f_order
                step = n+1;
                if (k>=0){
                    first+=(k)*n;
                }else{
                    first+=-k;
                }
            }
            difference_type i{1};
            while(true){
                op(*first);
                if (i==d){
                    break;
                }else{
                    ++i;
                    first+=step;
                }
            }
        }
    }

    template<typename ShT, typename IdxT, typename DimT>
    static auto make_space_shape(const ShT& shape, const IdxT& num, const DimT& axis){
        const auto dim = detail::make_dim(shape);
        ShT res(dim+1);
        std::copy(shape.begin(), shape.begin()+axis, res.begin());
        std::copy(shape.begin()+axis, shape.end(), res.begin()+(axis+1));
        res[axis] = num;
        return res;
    }

    template<typename T, typename Order, typename Config, typename Start, typename Stop, typename Num, typename DimT, typename Generator>
    static auto make_space(const Start& start, const Stop& stop, const Num& num, const DimT& axis_, Generator generator){
        static constexpr bool is_start_numeric = math::numeric_traits<Start>::is_integral() || math::numeric_traits<Start>::is_floating_point();
        static constexpr bool is_stop_numeric = math::numeric_traits<Stop>::is_integral() || math::numeric_traits<Stop>::is_floating_point();
        static_assert(math::numeric_traits<Num>::is_integral(),"num must be of integral type");
        static_assert(is_start_numeric || detail::is_tensor_v<Start>,"Start must be of numeric or tensor type");
        static_assert(is_stop_numeric || detail::is_tensor_v<Stop>,"Stop must be of numeric or tensor type");
        using common_value_type = detail::tensor_common_value_type_t<Start,Stop,Num>;
        using res_value_type = std::conditional_t<std::is_same_v<T,detail::no_value>,math::make_floating_point_t<common_value_type>,math::make_floating_point_t<T>>;
        using tensor_type = tensor<res_value_type,Order,config::extend_config_t<Config,T>>;
        using index_type = typename tensor_type::index_type;
        using shape_type = typename tensor_type::shape_type;
        check_make_space_args(num);
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
                auto predicate = detail::make_traverse_predicate(axis,std::true_type{});    //inverse, traverse all but axis
                const auto& res_shape = res.shape();
                auto res_traverser = detail::make_forward_traverser(res_shape,res.create_walker(),predicate);
                auto a = intervals.traverse_order_adapter(Order{});
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

//make tensor of the same layout,value_type,config_type as t
//elements initialization depends on underlaying storage implementation
//if no shape specified result has input tensor's shape
template<typename ShT, typename...Ts>
auto empty_like(const basic_tensor<Ts...>& t, ShT&& shape){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::empty_like(t,std::forward<ShT>(shape));
}
template<typename...Ts>
auto empty_like(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::empty_like(t);
}

//make tensor of the same layout,value_type,config_type as t, initialized with v
//if no shape specified result has input tensor's shape
template<typename U, typename ShT, typename...Ts>
auto full_like(const basic_tensor<Ts...>& t, const U& v, ShT&& shape){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::full_like(t,v,std::forward<ShT>(shape));
}
template<typename U, typename...Ts>
auto full_like(const basic_tensor<Ts...>& t, const U& v){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::full_like(t,v);
}

//make tensor of the same layout,value_type,config_type as t, initialized with zeros
//if no shape specified result has input tensor's shape
template<typename ShT, typename...Ts>
auto zeros_like(const basic_tensor<Ts...>& t, ShT&& shape){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::zeros_like(t,std::forward<ShT>(shape));
}
template<typename...Ts>
auto zeros_like(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::zeros_like(t);
}

//make tensor of the same shape,layout,value_type,config_type as t, initialized with ones
//if no shape specified result has input tensor's shape
template<typename ShT, typename...Ts>
auto ones_like(const basic_tensor<Ts...>& t, ShT&& shape){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::ones_like(t,std::forward<ShT>(shape));
}
template<typename...Ts>
auto ones_like(const basic_tensor<Ts...>& t){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::ones_like(t);
}

//make 1d tensor of evenly spaced values whithin a given interval
//result's value_type, layout and config may be specified by explicit specialization of T,Order,Config template's parameters
//T is not specialized explicitly result value_type is infered from Start,Stop,Step types
template<typename T=detail::no_value, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename Step=int>
auto arange(const Start& start, const Stop& stop, const Step& step=Step{1}){
    return builder_selector_t<Config>::template arange<T,Order,Config>(start,stop,step);
}
template<typename T=detail::no_value, typename Order = config::c_order, typename Config = config::default_config, typename Stop>
auto arange(const Stop& stop){
    return builder_selector_t<Config>::template arange<T,Order,Config>(Stop{0},stop,Stop{1});
}

//make tensor of num evenly spaced samples, calculated over the interval start, stop
//start, stop may be scalar or tensor, if either is tensor samples will be along axis
//result's value_type, layout and config may be specified by explicit specialization of T,Order,Config template's parameters
//T is not specialized explicitly result value_type is infered from Start,Stop,Num types
template<typename T=detail::no_value, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename Num=int, typename DimT=int>
auto linspace(const Start& start, const Stop& stop, const Num& num=50, bool end_point=true, const DimT& axis=0){
    return builder_selector_t<Config>::template linspace<T,Order,Config>(start,stop,num,end_point,axis);
}

//make tensor of numbers spaced evenly on a log scale
template<typename T=detail::no_value, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename Num=int, typename Base=double, typename DimT=int>
auto logspace(const Start& start, const Stop& stop, const Num& num=50, bool end_point=true, const Base& base=10.0, const DimT& axis=0){
    return builder_selector_t<Config>::template logspace<T,Order,Config>(start,stop,num,end_point,base,axis);
}

//make tensor of numbers spaced evenly on a log scale with endpoints specified directly
template<typename T=detail::no_value, typename Order = config::c_order, typename Config = config::default_config, typename Start, typename Stop, typename Num=int, typename DimT=int>
auto geomspace(const Start& start, const Stop& stop, const Num& num=50, bool end_point=true, const DimT& axis=0){
    return builder_selector_t<Config>::template geomspace<T,Order,Config>(start,stop,num,end_point,axis);
}

//if t is 2d tensor return 1d tensor that is t's kth diagonal
//if t is 1d tensor return 2d square tensor with t on its kth diagonal
template<typename IdxT=int, typename...Ts>
auto diag(const basic_tensor<Ts...>& t, const IdxT& k=0){
    using config_type = typename basic_tensor<Ts...>::config_type;
    return builder_selector_t<config_type>::diag(t,k);
}

}   //end of namespace gtensor
#endif