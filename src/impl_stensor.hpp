#ifndef IMPL_STENSOR_HPP_
#define IMPL_STENSOR_HPP_

#include "shareable_storage.hpp"
#include "impl_tensor_base.hpp"
#include "stensor_descriptor.hpp"
#include "tensor_init_list.hpp"
#include "view_factory.hpp"
#include "walker_factory.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail


/*
* implementation of tensor with storage
*/
template<typename ValT, template<typename> typename Cfg>
class stensor_impl : public tensor_impl_base<ValT,Cfg>{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::storage_type;
    using descriptor_type = stensor_descriptor<value_type, Cfg>;
    using slices_collection_type = typename config_type::slices_collection_type;
    using view_factory_type = view_factory<value_type,Cfg>;
    using walker_factory_type = walker_factory<value_type,Cfg>;

    descriptor_type descriptor;
    storage_type elements;
    view_factory_type view_maker;
    walker_factory_type walker_maker;

    template<typename Nested>
    stensor_impl(std::initializer_list<Nested> init_data, int):
        descriptor{detail::list_parse<index_type,shape_type>(init_data)},
        elements(descriptor.size()),
        view_maker{*this,descriptor,elements},
        walker_maker{*this,descriptor,elements}
    {detail::fill_from_list(init_data, elements.begin());}

public:
    stensor_impl& operator=(const stensor_impl& other) = delete;
    stensor_impl& operator=(stensor_impl&& other) = delete;
    stensor_impl() = default;
    stensor_impl(const stensor_impl& other):
        descriptor{other.descriptor},
        elements{other.elements},
        view_maker{*this,descriptor,elements},
        walker_maker{*this,descriptor,elements}
    {}
    stensor_impl(stensor_impl&& other):
        descriptor{std::move(other.descriptor)},
        elements{std::move(other.elements)},
        view_maker{*this,descriptor,elements},
        walker_maker{*this,descriptor,elements}
    {}

    stensor_impl(typename detail::nested_initializer_list_type<value_type,1>::type init_data):stensor_impl(init_data,0){}
    stensor_impl(typename detail::nested_initializer_list_type<value_type,2>::type init_data):stensor_impl(init_data,0){}
    stensor_impl(typename detail::nested_initializer_list_type<value_type,3>::type init_data):stensor_impl(init_data,0){}
    stensor_impl(typename detail::nested_initializer_list_type<value_type,4>::type init_data):stensor_impl(init_data,0){}
    stensor_impl(typename detail::nested_initializer_list_type<value_type,5>::type init_data):stensor_impl(init_data,0){}

    index_type size()const override{return descriptor.size();}
    index_type dim()const override{return descriptor.dim();}
    const shape_type& shape()const override{return descriptor.shape();}
    const shape_type& strides()const override{return descriptor.strides();}

    walker<ValT,Cfg> create_walker()const override{return walker_maker.create_walker();}
    std::shared_ptr<impl_base_type> create_view_slice(const slices_collection_type& subs)const override{return view_maker.create_view_slice(subs);}
    std::shared_ptr<impl_base_type> create_view_transpose(const shape_type& subs)const override{return view_maker.create_view_transpose(subs);}
    std::shared_ptr<impl_base_type> create_view_subdim(const shape_type& subs)const override{return view_maker.create_view_subdim(subs);}
    std::shared_ptr<impl_base_type> create_view_reshape(const shape_type& subs)const override{return view_maker.create_view_reshape(subs);}

    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"{"<<[&ss,this](){ss<<descriptor.to_str(); for(const auto& i:elements){ss<<i<<",";} return "}";}();
        return ss.str();
    }
};

}   //end of namespace gtensor

#endif