#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include "config.hpp"
#include "tensor_init_list.hpp"
#include "descriptor.hpp"
#include "broadcast.hpp"

namespace gtensor{

//base type for all tensor implementations
//Impl - tensor implementation type
//Descriptor - descriptor type used by tensor implementation
//Impl must provide interface to make data accessor:
//create_indexer() or create_walker() or both, if both is proveded type alias tag Impl::data_accessor_tag is used to select accessor
template<typename Impl, typename Descriptor>
class tensor_crtp_base
{
    using impl_type = Impl;
    static_assert(std::is_convertible_v<impl_type*,const tensor_crtp_base*>);

    const impl_type& impl()const{return static_cast<const impl_type&>(*this);}
    impl_type& impl(){return static_cast<impl_type&>(*this);}
protected:
    using descriptor_type = Descriptor;
    ~tensor_crtp_base(){}
    template<typename...Args>
    tensor_crtp_base(Args&&...args):
        descriptor_{std::forward<Args>(args)...}
    {}
public:
    using value_type = typename impl_type::value_type;
    using config_type = typename impl_type::config_type;
    using dim_type = typename impl_type::dim_type;
    using index_type = typename impl_type::index_type;
    using shape_type = typename impl_type::shape_type;

    //implementation accessor
    //result_type
    //create_walker, check what interface is provided by implementation, need some helpers

    tensor_crtp_base(const tensor_crtp_base&) = delete;
    tensor_crtp_base& operator=(const tensor_crtp_base&) = delete;
    tensor_crtp_base(tensor_crtp_base&&) = delete;
    tensor_crtp_base& operator=(tensor_crtp_base&&) = delete;

    const descriptor_type& descriptor()const{return descriptor_;}
    descriptor_type& descriptor(){return descriptor_;}
    index_type size()const{return descriptor_.size();}
    dim_type dim()const{return descriptor_.dim();}
    const shape_type& shape()const{return descriptor_.shape();}
    const shape_type& strides()const{return descriptor_.strides();}

private:
    descriptor_type descriptor_;
};

//tensor implementation that owns storage of data elements of type ValT
//storage type is specialization of CfgT::storage template
template<typename CfgT, typename ValT>
class storage_tensor : public tensor_crtp_base<storage_tensor<CfgT,ValT>,basic_descriptor<CfgT>>
{
    using tensor_crtp_base_type = tensor_crtp_base<storage_tensor<CfgT,ValT>,basic_descriptor<CfgT>>;
public:
    using value_type = ValT;
    using config_type = config::extend_config_t<CfgT, ValT>;
    using storage_type = typename config_type::storage_type
    using dim_type = typename config_type::dim_type;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    //make protected interface for crtp_base accessor
    //add operator[]
    //result_type
    //consider that storage_type has only operator[] and no iterators, need some helpers to init

    template<typename Nested>
    storage_tensor(std::initializer_list<Nested> init_data):
        tensor_crtp_base_type{detail::list_parse<dim_type,shape_type>(init_data)},
        elements_(descriptor_.size())
    {
        detail::fill_from_list(init_data, elements_.begin());
    }
    template<typename ShT>
    storage_tensor(ShT&& shape, const value_type& v):
        tensor_crtp_base_type{std::forward<ShT>(shape)},
        elements_(descriptor_.size(),v)
    {}
    template<typename ShT, typename ItT>
    storage_tensor(ShT&& shape, ItT begin, ItT end):
        tensor_crtp_base_type{std::forward<ShT>(shape)},
        elements_(descriptor_.size())
    {
        index_type n = std::distance(begin,end);
        if (descriptor_.size() < n){
            for(auto elements_it = elements_.begin(), elements_end = elements_.end(); elements_it!=elements_end; ++elements_it,++begin){
                *elements_it = *begin;
            }
        }else{
            std::copy(begin, end, elements_.begin());
        }
    }


    //inplace
    template<typename ShT>
    void resize(ShT&& shape){
        using typename tensor_crtp_base_type::descriptor_type;
        tensor_crtp_base_type::descriptor() = descriptor_type{std::forward<ShT>(shape)};
        elements_.resize(descriptor_.size());
        elements_.shrink_to_fit();
    }
private:
    storage_type elements_;
};

// template<typename EngineT>
// class storage_tensor : public basic_tensor<basic_descriptor<typename EngineT::config_type>, EngineT>
// {
//     using basic_tensor_base = basic_tensor<basic_descriptor<typename EngineT::config_type>, EngineT>;
// public:
//     using typename basic_tensor_base::config_type;
//     using typename basic_tensor_base::value_type;
//     using typename basic_tensor_base::dim_type;
//     using typename basic_tensor_base::index_type;
//     using typename basic_tensor_base::shape_type;
// private:
//     using typename basic_tensor_base::engine_type;
//     using typename basic_tensor_base::descriptor_type;

//     template<typename ShT, typename Nested>
//     storage_tensor(ShT&& shape, std::initializer_list<Nested> init_data):
//         storage_tensor{detail::make_size(shape), std::forward<ShT>(shape), init_data}
//     {}
//     template<typename ShT, typename...InitT>
//     storage_tensor(const index_type& size, ShT&& shape, InitT...init_data):
//         basic_tensor_base{engine_type{this, size, init_data...}, descriptor_type{std::forward<ShT>(shape)}}
//     {}
// public:
//     template<typename Nested>
//     storage_tensor(std::initializer_list<Nested> init_data):
//         storage_tensor{detail::list_parse<dim_type,shape_type>(init_data), init_data}
//     {}
//     template<typename ShT>
//     storage_tensor(ShT&& shape, const value_type& v):
//         storage_tensor{detail::make_size<index_type>(shape), std::forward<ShT>(shape), v}
//     {}
//     template<typename ShT, typename ItT>
//     storage_tensor(ShT&& shape, ItT begin, ItT end):
//         storage_tensor{detail::make_size<index_type>(shape), std::forward<ShT>(shape), begin, end}
//     {}

//     //inplace
//     template<typename ShT>
//     void resize(ShT&& shape){
//         basic_tensor_base::descriptor() = descriptor_type{std::forward<ShT>(shape)};
//         basic_tensor_base::engine().resize(basic_tensor_base::descriptor().size());
//     }
// };

template<typename EngineT>
class evaluating_tensor : public basic_tensor<basic_descriptor<typename EngineT::config_type>, EngineT>
{
    using basic_tensor_base = basic_tensor<basic_descriptor<typename EngineT::config_type>, EngineT>;
public:
    using typename basic_tensor_base::config_type;
    using typename basic_tensor_base::value_type;
    using typename basic_tensor_base::dim_type;
    using typename basic_tensor_base::index_type;
    using typename basic_tensor_base::shape_type;
private:
    using typename basic_tensor_base::engine_type;
    using typename basic_tensor_base::descriptor_type;

    template<typename F, typename...Operands>
    evaluating_tensor(shape_type&& shape, F&& f, Operands&&...operands):
        basic_tensor_base{engine_type{this, std::forward<F>(f),std::forward<Operands>(operands)...}, descriptor_type{std::move(shape)}}
    {}
public:
    template<typename F, typename...Operands>
    evaluating_tensor(F&& f, Operands&&...operands):
        evaluating_tensor{detail::broadcast_shape<shape_type>(operands->shape()...),std::forward<F>(f),std::forward<Operands>(operands)...}
    {}
};

template<typename DescT, typename EngineT>
class viewing_tensor : public basic_tensor<DescT, EngineT>
{
    using basic_tensor_base = basic_tensor<DescT, EngineT>;
    using typename basic_tensor_base::engine_type;
    using typename basic_tensor_base::descriptor_type;
public:
    using typename basic_tensor_base::config_type;
    using typename basic_tensor_base::value_type;
    using typename basic_tensor_base::dim_type;
    using typename basic_tensor_base::index_type;
    using typename basic_tensor_base::shape_type;
public:
    template<typename U>
    viewing_tensor(descriptor_type&& descriptor, U&& parent):
        basic_tensor_base{engine_type{this,std::forward<U>(parent)},std::move(descriptor)}
    {}
};

}   //end of namespace gtensor

#endif