#ifndef GTENSOR_HPP_
#define GTENSOR_HPP_

#include <memory>
#include "common.hpp"
#include "tensor_factory.hpp"
#include "tensor.hpp"
#include "tensor_operators.hpp"
#include "slice.hpp"
#include "view_factory.hpp"
#include "expression_template_engine.hpp"
#include "reduce.hpp"


namespace gtensor{

namespace detail{

template<typename T, typename IdxT, typename=void> constexpr inline bool is_index_tensor_v = false;
template<typename T, typename IdxT> constexpr inline bool is_index_tensor_v<T,IdxT,std::void_t<std::enable_if_t<is_tensor_v<T>>>> = std::is_convertible_v<typename T::value_type, IdxT>;

template<typename T, typename=void> constexpr inline bool is_bool_tensor_v = false;
template<typename T> constexpr inline bool is_bool_tensor_v<T,std::void_t<std::enable_if_t<is_tensor_v<T>>>> = std::is_same_v<typename T::value_type, bool>;

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
    using slice_init_type = typename slice_traits<CfgT>::slice_init_type;
    using slices_init_type = typename slice_traits<CfgT>::slices_init_type;
    using slices_container_type = typename slice_traits<CfgT>::slices_container_type;

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
    friend class reducer;
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

    //copy operartions has reference semantic, to copy by value should use copy method
    tensor(const tensor&) = default;
    tensor& operator=(const tensor&) = default;
    tensor(tensor&&) = default;
    tensor& operator=(tensor&&) = default;

    //all public constructors create tensor with storage implementation
    //nested init_list constructors
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<U> init_data):tensor(forward_tag{}, init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<U>> init_data):tensor(forward_tag{}, init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<U>>> init_data):tensor(forward_tag{}, init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> init_data):tensor(forward_tag{}, init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>> init_data):tensor(forward_tag{}, init_data){}
    //default constructor makes empty 1-d tensor
    tensor():
        tensor(std::initializer_list<value_type>{})
    {}
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
    auto empty()const{return size() == index_type{0};}
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
    auto operator()(const slice_type& subs, const size_type& direction)const{
        detail::check_slice_direction(descriptor().shape(), direction);
        index_type direction_size = descriptor().shape()[direction];
        slice_type filled_subs = detail::fill_slice(subs, direction_size);
        detail::check_slice(filled_subs, direction_size);
        return view_factory<ValT,CfgT>::create_view_slice(impl(), filled_subs, direction);
    }
    auto operator()(slice_init_type subs, const size_type& direction)const{
        return operator()(slice_type{subs}, direction);
    }
    auto operator()(slices_init_type subs)const{
        detail::check_slices_number(descriptor().shape(), subs);
        slices_container_type filled_subs = detail::fill_slices<slices_container_type>(descriptor().shape(),subs);
        detail::check_slices(descriptor().shape(), filled_subs);
        return view_factory<ValT,CfgT>::create_view_slice(impl(), filled_subs);
    }
    template<typename...Subs, std::enable_if_t<(std::is_convertible_v<Subs,slice_type>&&...),int> = 0 >
    auto operator()(const Subs&...subs)const{
        detail::check_slices_number(descriptor().shape(), subs...);
        slices_container_type filled_subs = detail::fill_slices<slices_container_type>(descriptor().shape(),subs...);
        detail::check_slices(descriptor().shape(), filled_subs);
        return view_factory<ValT,CfgT>::create_view_slice(impl(), filled_subs);
    }
    template<typename Slices, std::enable_if_t<detail::is_slices_container<Slices> ,int> =0>
    auto operator()(const Slices& subs){
        detail::check_slices_number(descriptor().shape(), subs);
        slices_container_type filled_subs = detail::fill_slices<slices_container_type>(descriptor().shape(),subs);
        detail::check_slices(descriptor().shape(), filled_subs);
        return view_factory<ValT,CfgT>::create_view_slice(impl(), filled_subs);
    }
    //transpose view
    template<typename...Subs, std::enable_if_t<(std::is_convertible_v<Subs,size_type>&&...),int> = 0 >
    auto transpose(const Subs&...subs)const{
        detail::check_transpose_subs(dim(),subs...);
        return view_factory<ValT,CfgT>::create_view_transpose(impl(), static_cast<size_type>(subs)...);
    }
    //subdimension view
    template<typename...Subs, std::enable_if_t<(std::is_convertible_v<Subs,index_type>&&...),int> = 0>
    auto subdim(const Subs&...subs)const{
        return create_view_subdim(*this, typename config_type::template container<index_type>{subs...});
    }
    template<typename Container, std::enable_if_t<detail::is_container_of_type_v<Container,index_type>,int> = 0>
    auto subdim(const Container& subs)const{
        return create_view_subdim(*this, subs);
    }
    //reshape view
    template<typename...Subs, std::enable_if_t<(std::is_convertible_v<Subs,index_type>&&...),int> = 0 >
    auto reshape(const Subs&...subs)const{
        detail::check_reshape_subs_variadic(size(), static_cast<index_type>(subs)...);
        return view_factory<ValT,CfgT>::create_view_reshape(impl(), shape_type{subs...});
    }
    template<typename Container, std::enable_if_t<detail::is_container_of_type_v<Container,index_type>,int> = 0 >
    auto reshape(Container&& subs)const{
        detail::check_reshape_subs_container(size(), subs);
        return view_factory<ValT,CfgT>::create_view_reshape(impl(), detail::make_shape_of_type<shape_type>(std::forward<Container>(subs)));
    }
    auto reshape(std::initializer_list<index_type> subs)const{
        detail::check_reshape_subs_container(size(), subs);
        return view_factory<ValT,CfgT>::create_view_reshape(impl(), detail::make_shape_of_type<shape_type>(subs));
    }
    //mapping view
    template<typename...Subs, std::enable_if_t<(detail::is_index_tensor_v<Subs,index_type>&&...),int> = 0 >
    auto operator()(const Subs&...subs)const{
        return view_factory<ValT,CfgT>::create_index_mapping_view(impl(), subs...);
    }
    template<typename Subs, std::enable_if_t<detail::is_bool_tensor_v<Subs> ,int> = 0 >
    auto operator()(const Subs& subs)const{
        return view_factory<ValT,CfgT>::create_bool_mapping_view(impl(), subs);
    }

    //reduce
    template<typename BinaryOp>
    auto reduce(const size_type& direction, BinaryOp op)const{
        detail::check_reduce_args(descriptor().shape(), direction);
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