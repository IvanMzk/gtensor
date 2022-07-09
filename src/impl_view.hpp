#ifndef IMPL_VIEW_HPP_
#define IMPL_VIEW_HPP_

#include "shareable_storage.hpp"
#include "impl_tensor_base.hpp"

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
    using slices_init_type = typename config_type::slices_init_type;
    using slices_collection_type = typename config_type::slices_collection_type;

    descriptor_type descriptor;
    view_storage_type elements;
    storage_type cache{};

    std::unique_ptr<walker_impl_base<ValT, Cfg>> create_walker()const override{
        return nullptr;
    }

public:
    template<typename DtT, typename StT>
    view_impl(DtT&& descriptor_, StT&& elements_):
        descriptor{std::forward<DtT>(descriptor_)},
        elements{std::forward<StT>(elements_)}
    {}    

    index_type size()const override{return descriptor.size();}
    index_type dim()const override{return descriptor.dim();}
    const shape_type& shape()const override{return descriptor.shape();}

    std::shared_ptr<impl_base_type> create_view_slice(slices_init_type)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_view_slice(const slices_collection_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_view_transpose(const shape_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_view_subdim(const shape_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_view_reshape(const shape_type&)const override{return nullptr;}

    std::string to_str()const override{
        std::stringstream ss{};
        ss<<"{"<<[&ss,this](){ss<<descriptor.to_str(); return "}";}();
        return ss.str();
    }


};

}   //end of namespace gtensor


#endif