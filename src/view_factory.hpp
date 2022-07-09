#ifndef VIEW_FACTORY_HPP_
#define VIEW_FACTORY_HPP_

#include "view_slice_descriptor.hpp"
#include "view_subdim_descriptor.hpp"
#include "impl_view.hpp"


namespace gtensor{

template<typename ValT, template<typename> typename Cfg>
class view_factory{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using shape_type = typename config_type::shape_type;
    using slices_collection_type = typename config_type::slices_collection_type;
public:
    virtual std::shared_ptr<impl_base_type> create_slice_view(const slices_collection_type&)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_slice_view(const slices_collection_type&)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_transpose_view(const shape_type&)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_subdim_view(const shape_type&)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_reshape_view(const shape_type&)const{return nullptr;}
    
    template<typename...T, typename DescT, typename StorT>
    inline static std::unique_ptr<view_factory> create_factory(const stensor_impl<T...>& parent, const DescT& descriptor, const StorT& elements);
    template<typename...T, typename DescT, typename StorT, typename CacheT>
    inline static std::unique_ptr<view_factory> create_factory(const expression_impl<T...>& parent, const DescT& descriptor, const StorT& elements, const CacheT& cache);
    template<typename...T, typename DescT, typename StorT, typename CacheT>
    inline static std::unique_ptr<view_factory> create_factory(const view_impl<T...>& parent, const DescT& descriptor, const StorT& elements, const CacheT& cache);
};

template<typename ValT, template<typename> typename Cfg, typename ParentT, typename DescT, typename StorT, typename CacheT>
class view_of_stensor_factory : public view_factory<ValT, Cfg>{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;

    const ParentT* parent;
    const DescT* descriptor;
    const StorT* elements;

    view_of_stensor_factory(const ParentT& parent_, const DescT& descriptor_, const StorT& elements_):
        parent{parent_},
        descriptor{descriptor_},
        elements{elements_}
    {}

    virtual std::shared_ptr<impl_base_type> create_slice_view(slices_init_type)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_slice_view(const slices_collection_type&)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_transpose_view(const shape_type&)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_subdim_view(const shape_type&)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_reshape_view(const shape_type&)const{return nullptr;}
};
template<typename ValT, template<typename> typename Cfg, typename ParentT, typename DescT, typename StorT, typename CacheT>
class view_of_stensor_factory_impl : public view_factory<ValT, Cfg>{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;

    const ParentT* parent;
    const DescT* descriptor;
    const StorT* elements;
    const CacheT* cache;

    view_factory_impl(const ParentT& parent_, const DescT& descriptor_, const StorT& elements_, const CacheT& cache_):
        parent{parent_},
        descriptor{descriptor_},
        elements{elements_},
        cache{cache_}
    {}

    std::shared_ptr<impl_base_type> create_view(slices_init_type)const override{
        return nullptr;
    }
    std::shared_ptr<impl_base_type> create_view(const slices_collection_type&)const override{
        return nullptr;
    }
};


template<typename ValT, template<typename> typename Cfg>
template<typename...T, typename DescT, typename StorT>
static std::unique_ptr<view_factory<ValT,Cfg>> view_factory<ValT,Cfg>::create_factory(const stensor_impl<T...>& parent, const DescT& descriptor, const StorT& elements){

}
template<typename ValT, template<typename> typename Cfg>
template<typename...T, typename DescT, typename StorT, typename CacheT>
static std::unique_ptr<view_factory<ValT,Cfg>> view_factory<ValT,Cfg>::create_factory(const expression_impl<T...>& parent, const DescT& descriptor, const StorT& elements, const CacheT& cache){

}
template<typename ValT, template<typename> typename Cfg>
template<typename...T, typename DescT, typename StorT, typename CacheT>
static std::unique_ptr<view_factory<ValT,Cfg>> view_factory<ValT,Cfg>::create_factory(const view_impl<T...>& parent, const DescT& descriptor, const StorT& elements, const CacheT& cache){

}


}   //end of namespace gtensor



#endif