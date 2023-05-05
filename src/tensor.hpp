#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <sstream>
#include "type_selector.hpp"
#include "tensor_factory.hpp"
#include "tensor_operators.hpp"
#include "slice.hpp"
#include "view_factory.hpp"
#include "reduce.hpp"

namespace gtensor{
template<typename T, typename Config> class tensor;

template<typename Impl>
class basic_tensor
{
    using impl_type = Impl;
    std::shared_ptr<impl_type> impl_;
public:
    using config_type = typename impl_type::config_type;
    using value_type = typename impl_type::value_type;
    using dim_type = typename impl_type::dim_type;
    using index_type = typename impl_type::index_type;
    using shape_type = typename impl_type::shape_type;
    using size_type = index_type;
    using difference_type = index_type;
    using slice_type = slice<index_type>;
    using slice_item_type = typename slice_type::slice_item_type;

    basic_tensor(const basic_tensor&) = default;
    basic_tensor& operator=(const basic_tensor&) = default;
    basic_tensor(basic_tensor&&) = default;
    basic_tensor& operator=(basic_tensor&&) = default;

    basic_tensor(std::shared_ptr<impl_type>&& impl__):
        impl_{std::move(impl__)}
    {}
    //makes tensor by copying shape and elements from this
    auto copy()const{
        return tensor<value_type,config_type>{shape(),begin(),end()};
    }
    //compare content of this tensor and other
    // template<typename OtherImpl>
    // auto equals(const basic_tensor<OtherImpl>& other)const{
    //     return gtensor::equals(*this, other);
    // }
    //meta-data interface
    const auto& descriptor()const{return impl_->descriptor();}
    index_type size()const{return impl_->size();}
    dim_type dim()const{return impl_->dim();}
    const shape_type& shape()const{return impl_->shape();}
    const shape_type& strides()const{return impl_->strides();}
    bool empty()const{return impl_->empty();}
    //data interface
    auto begin(){return impl_->begin();}
    auto end(){return impl_->end();}
    auto rbegin(){return impl_->rbegin();}
    auto rend(){return impl_->rend();}
    auto create_indexer(){return impl_->create_indexer();}
    auto create_walker(dim_type max_dim){return impl_->create_walker(max_dim);}
    auto create_walker(){return impl_->create_walker();}
    //const data interface
    auto begin()const{return impl_->begin();}
    auto end()const{return impl_->end();}
    auto rbegin()const{return impl_->rbegin();}
    auto rend()const{return impl_->rend();}
    auto create_indexer()const{return impl_->create_indexer();}
    auto create_walker(dim_type max_dim)const{return impl_->create_walker(max_dim);}
    auto create_walker()const{return impl_->create_walker();}
    //reduce
    template<typename BinaryOp>
    auto reduce(const dim_type& direction, BinaryOp op)const{
        return reduce(*this, direction, op);
    }
    //view construction operators and methods
    //slice view
    auto operator()(std::initializer_list<std::initializer_list<slice_item_type>> subs)const{
        return create_view(std::integral_constant<view_kind_type,view_kind_type::slice_view>{}, *this, subs);
    }
    template<typename...Subs, std::enable_if_t<((std::is_convertible_v<Subs,slice_type>||std::is_convertible_v<Subs,index_type>)&&...),int> = 0 >
    auto operator()(const Subs&...subs)const{
        return create_view(std::integral_constant<view_kind_type,view_kind_type::slice_view>{}, *this, subs...);
    }
    template<typename Container, std::enable_if_t<detail::is_container_of_type_v<Container,slice_type>,int> = 0>
    auto operator()(const Container& subs)const{
        return create_view(std::integral_constant<view_kind_type,view_kind_type::slice_view>{}, *this, subs);
    }
    //transpose view
    template<typename...Subs, std::enable_if_t<(std::is_convertible_v<Subs,dim_type>&&...),int> = 0 >
    auto transpose(const Subs&...subs)const{
        return create_view(std::integral_constant<view_kind_type,view_kind_type::transpose_view>{}, *this, subs...);
    }
    template<typename Container, std::enable_if_t<detail::is_container_of_type_v<Container,dim_type>,int> = 0>
    auto transpose(const Container& subs)const{
        return create_view(std::integral_constant<view_kind_type,view_kind_type::transpose_view>{}, *this, subs);
    }
    //reshape view
    template<typename...Subs, std::enable_if_t<(std::is_convertible_v<Subs,index_type>&&...),int> = 0 >
    auto reshape(const Subs&...subs)const{
        return create_view(std::integral_constant<view_kind_type,view_kind_type::reshape_view>{}, *this, subs...);
    }
    template<typename Container, std::enable_if_t<detail::is_container_of_type_v<Container,index_type>,int> = 0 >
    auto reshape(const Container& subs)const{
        return create_view(std::integral_constant<view_kind_type,view_kind_type::reshape_view>{}, *this, subs);
    }
    auto reshape(std::initializer_list<index_type> subs)const{
        return create_view(std::integral_constant<view_kind_type,view_kind_type::reshape_view>{}, *this, subs);
    }
    //mapping view
    template<typename...Subs, std::enable_if_t<(detail::is_tensor_of_type_v<Subs,index_type>&&...),int> = 0 >
    auto operator()(const Subs&...subs)const{
        return create_view(std::integral_constant<view_kind_type,view_kind_type::index_mapping_view>{}, *this, subs...);
    }
    template<typename Subs, std::enable_if_t<detail::is_bool_tensor_v<Subs> ,int> = 0 >
    auto operator()(const Subs& subs)const{
        return create_view(std::integral_constant<view_kind_type,view_kind_type::bool_mapping_view>{}, *this, subs);
    }
private:
    enum class view_kind_type{slice_view, transpose_view, reshape_view, index_mapping_view, bool_mapping_view};
    template<typename Impl_>
    auto create_view(std::shared_ptr<Impl_>&& impl__)const{
        return basic_tensor<Impl_>{std::move(impl__)};
    }
    template<typename ViewKind, typename...Args>
    auto create_view(ViewKind, Args&&...args)const{
        using view_factory_type = view_factory_selector_t<config_type>;
        if constexpr (ViewKind::value == view_kind_type::slice_view){
            return create_view(view_factory_type::create_slice_view(std::forward<Args>(args)...));
        }else if constexpr (ViewKind::value == view_kind_type::transpose_view){
            return create_view(view_factory_type::create_transpose_view(std::forward<Args>(args)...));
        }else if constexpr (ViewKind::value == view_kind_type::reshape_view){
            return create_view(view_factory_type::create_reshape_view(std::forward<Args>(args)...));
        }else if constexpr (ViewKind::value == view_kind_type::index_mapping_view){
            return create_view(view_factory_type::create_index_mapping_view(std::forward<Args>(args)...));
        }else if constexpr (ViewKind::value == view_kind_type::bool_mapping_view){
            return create_view(view_factory_type::create_bool_mapping_view(std::forward<Args>(args)...));
        }else{
            return;
        }
    }

};

//tensor is basic_tensor with storage implementation and constructors
template<typename T, typename Config = config::default_config>
class tensor : public basic_tensor<typename tensor_factory_selector_t<Config,T>::result_type>
{
    using tensor_factory_type = tensor_factory_selector_t<Config,T>;
    using basic_tensor_base = basic_tensor<typename tensor_factory_type::result_type>;

    class forward_tag{
        struct private_tag{};
        forward_tag(private_tag){}  //make not default constructible
    public:
        static auto tag(){return forward_tag{private_tag{}};}
    };
    //this constructor should be used by all public constructors
    template<typename...Args>
    tensor(forward_tag, Args&&...args):
        basic_tensor_base(tensor_factory_type::create(std::forward<Args>(args)...))
    {}
public:
    using config_type = typename basic_tensor_base::config_type;
    using value_type = typename basic_tensor_base::value_type;
    using dim_type = typename basic_tensor_base::dim_type;
    using index_type = typename basic_tensor_base::index_type;
    using shape_type = typename basic_tensor_base::shape_type;
    using size_type = typename basic_tensor_base::size_type;
    using difference_type = typename basic_tensor_base::difference_type;

    tensor(const tensor&) = default;
    tensor& operator=(const tensor&) = default;
    tensor(tensor&&) = default;
    tensor& operator=(tensor&&) = default;

    //nested init_list constructors
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<U> init_data):tensor(forward_tag::tag(), init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<U>> init_data):tensor(forward_tag::tag(), init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<U>>> init_data):tensor(forward_tag::tag(), init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> init_data):tensor(forward_tag::tag(), init_data){}
    template<typename U, std::enable_if_t<std::is_convertible_v<U,value_type>,int> =0> tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>>> init_data):tensor(forward_tag::tag(), init_data){}
    //default constructor makes empty 1-d tensor
    tensor():
        tensor(forward_tag::tag(), std::initializer_list<value_type>{})
    {}
    //0-dim tensor constructor (aka tensor-scalar)
    explicit tensor(const value_type& value__):
        tensor(forward_tag::tag(), shape_type{}, value__)
    {}
    //init list shape and value
    tensor(std::initializer_list<index_type> shape__, const value_type& value__):
        tensor(forward_tag::tag(), shape__, value__)
    {}
    //init list shape and range
    template<typename It>
    tensor(std::initializer_list<index_type> shape__, It begin__, It end__):
        tensor(forward_tag::tag(), shape__, begin__, end__)
    {}

    template<typename U> struct disable_forward_arg : std::disjunction<
        std::is_convertible<U,value_type>,
        std::is_convertible<U,tensor>
    >{};
    //container shape, disambiguate with 0-dim constructor, copy,move constructor
    template<typename Container, std::enable_if_t<!disable_forward_arg<Container>::value,int> =0>
    explicit tensor(Container&& shape__):
        tensor(forward_tag::tag(), std::forward<Container>(shape__))
    {}
    //container shape and value
    template<typename Container>
    tensor(Container&& shape__, const value_type& value__):
        tensor(forward_tag::tag(), std::forward<Container>(shape__), value__)
    {}
    //container shape and range
    template<typename Container, typename It>
    tensor(Container&& shape__, It begin__, It end__):
        tensor(forward_tag::tag(), std::forward<Container>(shape__), begin__, end__)
    {}
};

template<typename...Ts>
auto str(const basic_tensor<Ts...>& t){
    std::stringstream ss{};
    ss<<"{"<<detail::shape_to_str(t.shape())<<[&]{for(const auto& i:t){ss<<i<<" ";}; return "}";}();
    return ss.str();
}

}   //end of namespace gtensor
#endif