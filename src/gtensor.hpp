#ifndef GTENSOR_HPP_
#define GTENSOR_HPP_

#include <memory>
#include "tensor.hpp"
#include "tensor_operators.hpp"
#include "slice.hpp"
#include "view_factory.hpp"
#include "expression_template_engine//expression_template_engine.hpp"


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
    typename ImplT = storage_tensor<typename detail::storage_engine_traits<typename CfgT::host_engine,CfgT,typename CfgT::template storage<ValT>>::type>
>
class tensor{
    using impl_type = ImplT;
    using slice_type = typename CfgT::slice_type;
    using slices_init_type = typename CfgT::slices_init_type;
    using slices_collection_type = typename CfgT::slices_collection_type;
    class forward_tag{};

    friend class tensor_operator_dispatcher;
    template<typename,typename> friend class view_factory;

    //initialize implementation by forwarding arguments, this constructor should be used by all public constructors
    template<typename...Args>
    tensor(forward_tag, Args&&...args):
        impl_{std::make_shared<impl_type>(std::forward<Args>(args)...)}
    {}

    std::shared_ptr<impl_type> impl_;
protected:
    auto impl()const{return impl_;}
    auto& engine()const{return static_cast<const impl_type*>(impl_.get())->engine();}
    auto& engine(){return impl_->engine();}

public:
    using config_type = CfgT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
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
    tensor(typename detail::nested_initializer_list_type<value_type,1>::type init_data):tensor(forward_tag{}, init_data){}
    tensor(typename detail::nested_initializer_list_type<value_type,2>::type init_data):tensor(forward_tag{}, init_data){}
    tensor(typename detail::nested_initializer_list_type<value_type,3>::type init_data):tensor(forward_tag{}, init_data){}
    tensor(typename detail::nested_initializer_list_type<value_type,4>::type init_data):tensor(forward_tag{}, init_data){}
    tensor(typename detail::nested_initializer_list_type<value_type,5>::type init_data):tensor(forward_tag{}, init_data){}
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
    template<typename...Args>
    static tensor make_tensor(Args&&...args){
        return tensor(forward_tag{}, std::forward<Args>(args)...);
    }
    //makes new storage tensor by copying shape and elements from this tensor
    auto copy()const{
        using storage_impl_type = storage_tensor<typename detail::storage_engine_traits<typename CfgT::host_engine,CfgT,typename CfgT::template storage<ValT>>::type>;
        return tensor<value_type,CfgT,storage_impl_type>::make_tensor(shape(),begin(),end());
    }
    auto begin(){return engine().begin();}
    auto end(){return engine().end();}
    auto begin()const{return engine().begin();}
    auto end()const{return engine().end();}
    auto rbegin(){return engine().rbegin();}
    auto rend(){return engine().rend();}
    auto rbegin()const{return engine().rbegin();}
    auto rend()const{return engine().rend();}
    auto size()const{return impl()->size();}
    auto dim()const{return impl()->dim();}
    auto shape()const{return impl()->shape();}
    auto to_str()const{return impl()->to_str();}
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
    friend std::ostream& operator<<(std::ostream& os, const tensor& lhs){return os<<lhs.impl_->to_str();}

    //view construction operators and methods
    //slice view
    auto operator()(slices_init_type subs)const{
        detail::check_slices_number(subs);
        slices_collection_type filled_subs = detail::fill_slices<slice_type>(shape(),subs);
        detail::check_slices(shape(), filled_subs);
        return view_factory<ValT,CfgT>::create_view_slice(impl(), filled_subs);
    }
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,slice_type>...>,int> = 0 >
    auto operator()(const Subs&...subs)const{
        detail::check_slices_number(subs...);
        slices_collection_type filled_subs = detail::fill_slices<slice_type>(shape(),subs...);
        detail::check_slices(shape(), filled_subs);
        return view_factory<ValT,CfgT>::create_view_slice(impl(), filled_subs);
    }
    //transpose view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,index_type>...>,int> = 0 >
    auto transpose(const Subs&...subs)const{
        detail::check_transpose_subs(dim(),subs...);
        return view_factory<ValT,CfgT>::create_view_transpose(impl(), shape_type{subs...});
    }
    //subdimension view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,index_type>...>,int> = 0 >
    auto operator()(const Subs&...subs)const{
        detail::check_subdim_subs(shape(), subs...);
        return view_factory<ValT,CfgT>::create_view_subdim(impl(), shape_type{subs...});
    }
    auto operator()()const{
        detail::check_subdim_subs(shape());
        return view_factory<ValT,CfgT>::create_view_subdim(impl(), shape_type{});
    }
    //reshape view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<std::is_convertible<Subs,index_type>...>,int> = 0 >
    auto reshape(const Subs&...subs)const{
        detail::check_reshape_subs(size(), subs...);
        return view_factory<ValT,CfgT>::create_view_reshape(impl(), shape_type{subs...});
    }
    //mapping view
    template<typename...Subs, std::enable_if_t<std::conjunction_v<detail::is_index_tensor<Subs,index_type>...>,int> = 0 >
    auto operator()(const Subs&...subs)const{
        return view_factory<ValT,CfgT>::create_mapping_view_index_tensor(impl(), subs...);
    }
    template<typename Sub, std::enable_if_t<detail::is_bool_tensor<Sub>::value ,int> = 0 >
    auto operator()(const Sub& sub)const{
        return view_factory<ValT,CfgT>::create_mapping_view_bool_tensor(impl(), sub);
    }
};

}   //end of namespace gtensor

#endif