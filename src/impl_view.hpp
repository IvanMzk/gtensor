#ifndef IMPL_VIEW_HPP_
#define IMPL_VIEW_HPP_

#include "shareable_storage.hpp"
#include "impl_tensor_base.hpp"
#include "walker_factory.hpp"

namespace gtensor{

template<typename ValT, template<typename> typename Cfg, typename DescT, typename StorT>
class view_impl : public tensor_impl_base<ValT, Cfg> {
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using view_storage_type = StorT;
    using descriptor_type = DescT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using storage_type = typename config_type::storage_type;
    using slices_collection_type = typename config_type::slices_collection_type;
    using walker_factory_type = walker_factory<ValT,Cfg>;

    descriptor_type descriptor;
    view_storage_type elements;
    storage_type cache{};
    walker_factory_type walker_maker;

public:
    template<typename DtT, typename StT>
    view_impl(DtT&& descriptor_, StT&& elements_):
        descriptor{std::forward<DtT>(descriptor_)},
        elements{std::forward<StT>(elements_)},
        walker_maker{*this, descriptor, elements, cache}
    {}
    view_impl(const view_impl& other):
        descriptor{other.descriptor},
        elements{other.elements},
        cache{other.cache},
        walker_maker{*this, descriptor, elements, cache}
    {}
    view_impl(view_impl&& other):
        descriptor{std::move(other.descriptor)},
        elements{std::move(other.elements)},
        cache{std::move(other.cache)},
        walker_maker{*this, descriptor, elements, cache}
    {}

    index_type size()const override{return descriptor.size();}
    index_type dim()const override{return descriptor.dim();}
    const shape_type& shape()const override{return descriptor.shape();}
    const shape_type& strides()const override{return descriptor.strides();}
    bool is_cached()const{return cache.size();}
    value_type trivial_at(const index_type& idx)const override{return value_type(0);}
    walker<ValT,Cfg> create_walker()const override{return walker_maker.create_walker();}

    std::shared_ptr<impl_base_type> create_view_slice(const slices_collection_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_view_transpose(const shape_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_view_subdim(const shape_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_view_reshape(const shape_type&)const override{return nullptr;}

    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"{"<<[&ss,this](){ss<<descriptor.to_str(); for(const auto& i : elements){ss<<i<<",";} return "}";}();
        return ss.str();
    }


};

}   //end of namespace gtensor


#endif