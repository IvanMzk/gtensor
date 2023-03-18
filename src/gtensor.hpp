#ifndef GTENSOR_HPP_
#define GTENSOR_HPP_

#include <memory>
#include "tensor_factory.hpp"
#include "tensor.hpp"
#include "tensor_operators.hpp"
#include "slice.hpp"
#include "view_factory.hpp"
#include "expression_template_engine.hpp"
#include "reduce.hpp"


namespace gtensor{

namespace detail{

template<typename> struct true_type : std::true_type{};
template<typename T, template<typename...> typename C = true_type> struct is_tensor{
private:
    template<typename...Ts> static C<typename tensor<Ts...>::value_type> selector(const tensor<Ts...>&);
    static std::false_type selector(...);
public:
    using type = typename decltype(selector(std::declval<T>()))::type;
    static constexpr bool value = type::value;
};
template<typename ValT, typename IdxT> struct is_index : std::is_convertible<ValT,IdxT>::type{};
template<typename IdxT> struct is_index<bool,IdxT> : std::false_type{};
template<typename ValT> struct is_bool : std::false_type{};
template<> struct is_bool<bool> : std::true_type{};

template<typename T, typename IdxT> struct is_index_tensor
{
    template<typename V> using checker = is_index<V,IdxT>;
    using type = typename is_tensor<T, checker>::type;
    static constexpr bool value = type::value;
};
template<typename T> struct is_bool_tensor
{
    using type = typename is_tensor<T, is_bool>::type;
    static constexpr bool value = type::value;
};

}   //end of namespace detail

/*
* tensor is abstraction of tensor implementation
* tensors can be combined using operators or view of tensor can be made, that makes new tensor with apropriate implementation
*/
template<
    typename ValT,
    typename CfgT = config::default_config,
    typename ImplT = typename storage_tensor_implementation_selector<CfgT,ValT>::base_type
>
class tensor{
    using impl_type = ImplT;
    using slice_type = typename slice_traits<CfgT>::slice_type;
    using slices_init_type = typename slice_traits<CfgT>::slices_init_type;
    using slices_collection_type = typename slice_traits<CfgT>::slices_collection_type;

    //initialize storage implementation by forwarding arguments, this constructor should be used by all public constructors
    class forward_tag{};
    template<typename...Args>
    tensor(forward_tag, Args&&...args):
        impl_{std::make_shared<typename storage_tensor_implementation_selector<CfgT,ValT>::type>(std::forward<Args>(args)...)}
    {}
    //constructor is used by make_tensor static method to create tensor with explicitly specified implementation
    tensor(std::shared_ptr<impl_type>&& impl__):
        impl_{std::move(impl__)}
    {}

    friend struct tensor_operator_dispatcher;
    template<typename,typename> friend class view_factory;
    friend class reducer<ValT,CfgT>;
    friend class combiner;
    std::shared_ptr<impl_type> impl_;
protected:
    auto impl()const{return impl_;}
    auto& impl_ref()const{return *impl_;}
    const auto& engine()const{return static_cast<const impl_type*>(impl_.get())->engine();}
    auto& engine(){return impl_->engine();}
    const auto& descriptor()const{return static_cast<const impl_type*>(impl_.get())->descriptor();}
    auto& descriptor(){return impl_->descriptor();}

public:
    using config_type = CfgT;
    using index_type = typename config_type::index_type;
    using size_type = typename config_type::size_type;
    using shape_type = typename config_type::shape_type;
    using value_type = ValT;
    //constructs tensor using default implementation constructor
    tensor():
        tensor(forward_tag{})
    {}
    //copy operartions has reference semantic, to copy by value should use copy method
    tensor(const tensor&) = default;
    tensor& operator=(const tensor&) = default;
    tensor(tensor&&) = default;
    tensor& operator=(tensor&&) = default;

    //storage tensor constructors
    //nested init_list constructors
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<U> init_data):tensor(forward_tag{}, init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<U>> init_data):tensor(forward_tag{}, init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<U>>> init_data):tensor(forward_tag{}, init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> init_data):tensor(forward_tag{}, init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>> init_data):tensor(forward_tag{}, init_data){}
    //init list shape and value
    template<typename U>
    tensor(std::initializer_list<U> shape__, const value_type& value__):
        tensor(forward_tag{}, shape__, value__)
    {}
    //init list shape and range
    template<typename U, typename It>
    tensor(std::initializer_list<U> shape__, It begin__, It end__):
        tensor(forward_tag{}, shape__, begin__, end__)
    {}
    //arbitrary container shape and value
    template<typename U>
    tensor(U&& shape__, const value_type& value__):
        tensor(forward_tag{}, std::forward<U>(shape__), value__)
    {}
    //arbitrary container shape and range
    template<typename U, typename It>
    tensor(U&& shape__, It begin__, It end__):
        tensor(forward_tag{}, std::forward<U>(shape__), begin__, end__)
    {}
    //makes tensor with explicitly specified implementation type
    //Implementation must be convertible to tensor::impl_type
    template<typename Implementation, typename...Args>
    static tensor make_tensor(Args&&...args){
        return tensor(std::make_shared<Implementation>(std::forward<Args>(args)...));
    }
    //makes new storage tensor by copying shape and elements from this tensor
    auto copy()const{
        return storage_tensor_factory<CfgT,ValT>::make(descriptor().shape(),begin(),end());
    }
    auto begin(){return engine().begin();}
    auto end(){return engine().end();}
    auto begin()const{return engine().begin();}
    auto end()const{return engine().end();}
    auto rbegin(){return engine().rbegin();}
    auto rend(){return engine().rend();}
    auto rbegin()const{return engine().rbegin();}
    auto rend()const{return engine().rend();}
    auto begin_broadcast(const shape_type& shape){return engine().begin_broadcast(shape);}
    auto end_broadcast(const shape_type& shape){return engine().end_broadcast(shape);}
    auto begin_broadcast(const shape_type& shape)const{return engine().begin_broadcast(shape);}
    auto end_broadcast(const shape_type& shape)const{return engine().end_broadcast(shape);}
    auto rbegin_broadcast(const shape_type& shape){return engine().rbegin_broadcast(shape);}
    auto rend_broadcast(const shape_type& shape){return engine().rend_broadcast(shape);}
    auto rbegin_broadcast(const shape_type& shape)const{return engine().rbegin_broadcast(shape);}
    auto rend_broadcast(const shape_type& shape)const{return engine().rend_broadcast(shape);}

    auto size()const{return descriptor().size();}
    auto dim()const{return descriptor().dim();}
    auto shape()const{return descriptor().shape();}
    //compare content of this tensor and other
    template<typename RImpl>
    auto equals(const tensor<value_type,CfgT,RImpl>& other)const{return gtensor::equals(*this, other);}
    //broadcast value assigmnent, lhs may be any rvalue tensor but it makes sence only for viewing tensor to modify underlying storage tensor
    //lhs that is rvalue evaluating tensor or view of evaluating tensor will not compile
    template<typename RVal, typename RImpl>
    tensor& operator=(const tensor<RVal,CfgT,RImpl>& rhs) &&{
        operator_assign(*this, rhs);
        return *this;
    }

    //view construction operators and methods
    //slice view
    auto operator()(slices_init_type subs)const{
        detail::check_slices_number(subs);
        slices_collection_type filled_subs = detail::fill_slices<slice_type>(descriptor().shape(),subs);
        detail::check_slices(descriptor().shape(), filled_subs);
        return view_factory<ValT,CfgT>::create_view_slice(impl(), filled_subs);
    }
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,slice_type>...>,int> = 0 >
    auto operator()(const Subs&...subs)const{
        detail::check_slices_number(subs...);
        slices_collection_type filled_subs = detail::fill_slices<slice_type>(descriptor().shape(),subs...);
        detail::check_slices(descriptor().shape(), filled_subs);
        return view_factory<ValT,CfgT>::create_view_slice(impl(), filled_subs);
    }
    //transpose view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,size_type>...>,int> = 0 >
    auto transpose(const Subs&...subs)const{
        detail::check_transpose_subs(dim(),subs...);
        return view_factory<ValT,CfgT>::create_view_transpose(impl(), static_cast<size_type>(subs)...);
    }
    //subdimension view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,index_type>...>,int> = 0 >
    auto operator()(const Subs&...subs)const{
        shape_type subs_{subs...};
        detail::check_subdim_subs(descriptor().shape(), subs_);
        return view_factory<ValT,CfgT>::create_view_subdim(impl(), std::move(subs_));
    }
    auto operator()()const{
        return view_factory<ValT,CfgT>::create_view_subdim(impl(), shape_type{});
    }
    //reshape view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,index_type>...>,int> = 0 >
    auto reshape(const Subs&...subs)const{
        detail::check_reshape_subs(size(), static_cast<index_type>(subs)...);
        return view_factory<ValT,CfgT>::create_view_reshape(impl(), shape_type{subs...});
    }
    //mapping view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<detail::is_index_tensor<Subs,index_type>...>,int> = 0 >
    auto operator()(const Subs&...subs)const{
        return view_factory<ValT,CfgT>::create_index_mapping_view(impl(), subs...);
    }
    template<typename Subs, std::enable_if_t<detail::is_bool_tensor<Subs>::value ,int> = 0 >
    auto operator()(const Subs& subs)const{
        return view_factory<ValT,CfgT>::create_bool_mapping_view(impl(), subs);
    }

    //reduce
    template<typename BinaryOp>
    auto reduce(const size_type& direction, BinaryOp op)const{
        detail::check_reduce_direction(direction, dim());
        return gtensor::reduce(*this, direction, op);
    }
};

template<typename...Ts>
auto str(const tensor<Ts...>& t){
    std::stringstream ss{};
    ss<<"{"<<detail::shape_to_str(t.shape())<<[&]{for(const auto& i:t){ss<<i<<" ";}; return "}";}();
    return ss.str();
}

template<typename...Ts>
std::ostream& operator<<(std::ostream& os, const tensor<Ts...>& lhs){return os<<str(lhs);}

}   //end of namespace gtensor

#endif