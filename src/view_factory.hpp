#ifndef VIEW_FACTORY_HPP_
#define VIEW_FACTORY_HPP_

#include "view_slice_descriptor.hpp"
#include "view_subdim_descriptor.hpp"
#include "impl_view.hpp"


namespace gtensor{

namespace detail{

template<typename T>
inline auto make_view_shape_element(const T& sub){
    using difference_type = typename T::difference_type;
    difference_type step_ = sub.step > difference_type(0) ? sub.step : -sub.step;
    return sub.start == sub.stop ?
        typename difference_type(0) :
        sub.start < sub.stop ?
            (sub.stop - sub.start-difference_type(1))/step_ + difference_type(1) :
            (sub.start - sub.stop-difference_type(1))/step_ + difference_type(1);
}
/*make view slice shape*/
template<typename ShT, typename SubsT>
inline ShT make_view_slice_shape(const ShT& pshape, const SubsT& subs){
    ShT res{};
    res.reserve(pshape.size());    
    std::for_each(subs.begin(), subs.end(), [&res](const auto& sub){res.push_back(make_view_shape_element(sub));});
    std::for_each(pshape.data()+subs.size(), pshape.data()+pshape.size(), [&res](const auto& elem){res.push_back(elem);});
    return res;
}
/*make view slice offset*/
template<typename ShT, typename SubsT>
inline typename ShT::value_type make_view_slice_offset(const ShT& pstrides, const SubsT& subs){
    using index_type = typename ShT::value_type;
    index_type res{0};    
    std::for_each(subs.begin(), subs.end(), [&res, pstrides_it = pstrides.begin()](const auto& sub)mutable{res+=sub.start*(*pstrides_it); ++pstrides_it;});
    return res;
}
/*make view slice cstrides*/
template<typename ShT, typename SubsT>
inline ShT make_view_slice_cstrides(const ShT& pstrides, const SubsT& subs){
    using index_type = typename ShT::value_type;
    ShT res{};
    res.reserve(pstrides.size());
    auto pstrides_it = pstrides.begin();
    std::for_each(subs.begin(), subs.end(), [&res, &pstrides_it](const auto& sub){res.push_back(sub.step*(*pstrides_it)); ++pstrides_it;});
    std::for_each(pstrides_it, pstrides.end(), [&res](const auto& elem){res.push_back(elem);});    
    return res;
}

/*make transposed 
* indeces is positions of source shape elements in transposed shape
* if indeces is empty transposed shape is reverse of source
*/
template<typename ShT>
ShT transpose(const ShT& src, const ShT& indeces){
    ShT res{};
    res.reserve(src.size());
    if (indeces.empty()){
        res.assign(src.rbegin(), src.rend());
    }else{
        std::for_each(indeces.begin(), indeces.end(), [&src, &res](const auto& pos){res.push_back(src[pos]);});
    }
    return res;
}

/*make view subdim shape*/
template<typename ShT>
inline ShT make_view_subdim_shape(const ShT& pshape, const ShT& subs){
    if (subs.empty()){
        return pshape;
    }else{
        return ShT(pshape.data()+subs.size(), pshape.data()+pshape.size());
    }
}
/*make view subdim offset*/
template<typename ShT>
inline typename ShT::value_type make_view_subdim_offset(const ShT& pstrides, const ShT& subs){
    using index_type = typename ShT::value_type;
    return std::inner_product(subs.begin(),subs.end(),pstrides.begin(),index_type(0));
}

}   //end of namespace detail


template<typename ValT, template<typename> typename Cfg>
class view_factory{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using shape_type = typename config_type::shape_type;
    using slices_collection_type = typename config_type::slices_collection_type;
public:
    virtual std::shared_ptr<impl_base_type> create_view_slice(const slices_collection_type&, bool move = false)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_view_transpose(const shape_type&, bool move = false)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_view_subdim(const shape_type&, bool move = false)const{return nullptr;}
    virtual std::shared_ptr<impl_base_type> create_view_reshape(const shape_type&, bool move = false)const{return nullptr;}
    
    template<typename...T, typename DescT, typename StorT>
    inline static std::unique_ptr<view_factory> create_factory(const stensor_impl<T...>& parent, const DescT& descriptor, const StorT& elements);
    template<typename...T, typename DescT, typename StorT, typename CacheT>
    inline static std::unique_ptr<view_factory> create_factory(const expression_impl<T...>& parent, const DescT& descriptor, const StorT& elements, const CacheT& cache);
    template<typename...T, typename DescT, typename StorT, typename CacheT>
    inline static std::unique_ptr<view_factory> create_factory(const view_impl<T...>& parent, const DescT& descriptor, const StorT& elements, const CacheT& cache);
};

template<typename ValT, template<typename> typename Cfg, typename DescT, typename StorT>
class view_simple_descriptor_factory : public view_factory<ValT, Cfg>{
    using config_type = Cfg<ValT>;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using view_slice_descriptor_type = view_slice_descriptor<ValT,Cfg>;
    using view_subdim_descriptor_type = view_subdim_descriptor<ValT,Cfg>;
    using view_slice_type = gtensor::view_impl<ValT,Cfg,view_slice_descriptor_type, StorT>;
    using view_subdim_type = gtensor::view_impl<ValT,Cfg,view_subdim_descriptor_type, StorT>;
    
    const DescT* descriptor;
    const StorT* elements;

    view_simple_descriptor_factory(const DescT& descriptor_, const StorT& elements_):
        descriptor{&descriptor_},
        elements{&elements_}
    {}

    view_slice_descriptor_type create_view_slice_descriptor(const slices_collection_type& subs, bool)const{
        return view_slice_descriptor_type{
            detail::make_view_slice_shape(descriptor->shape(),subs), 
            detail::make_view_slice_cstrides(descriptor->strides(),subs),
            detail::make_view_slice_offset(descriptor->strides(),subs)
        };
    }    
    view_slice_descriptor_type create_view_transpose_descriptor(const shape_type& subs, bool)const{
        return view_slice_descriptor_type{detail::transpose(descriptor->shape(),subs), detail::transpose(descriptor->strides(),subs),index_type(0)};
    }
    view_subdim_descriptor_type create_view_subdim_descriptor(const shape_type& subs, bool)const{
        return view_subdim_descriptor_type{detail::make_view_subdim_shape(descriptor->shape(),subs), detail::make_view_subdim_offset(descriptor->strides(),subs)};
    }
    view_subdim_descriptor_type create_view_reshape_descriptor(const shape_type& subs, bool)const{
        return view_subdim_descriptor_type{subs,index_type(0)};
    }

    std::shared_ptr<impl_base_type> create_view_slice(const slices_collection_type& subs, bool move = false)const override{
        return std::static_pointer_cast<impl_base_type>(move ? 
            std::make_shared<view_slice_type>(create_view_slice_descriptor(subs,move),std::move(const_cast<StorT>(*elements))):
            std::make_shared<view_slice_type>(create_view_slice_descriptor(subs,move),*elements)
        );        
    }
    std::shared_ptr<impl_base_type> create_view_transpose(const shape_type& subs, bool move = false)const override{
        return std::static_pointer_cast<impl_base_type>(move ?
            std::make_shared<view_slice_type>(create_view_transpose_descriptor(subs,move),std::move(const_cast<StorT>(*elements))):
            std::make_shared<view_slice_type>(create_view_transpose_descriptor(subs,move),*elements):
        );        
    }
    std::shared_ptr<impl_base_type> create_view_subdim(const shape_type& subs, bool move = false)const override{
        return std::static_pointer_cast<impl_base_type>(move ?
            std::make_shared<view_subdim_type>(create_view_subdim_descriptor(subs,move),std::move(const_cast<StorT>(*elements))):
            std::make_shared<view_subdim_type>(create_view_subdim_descriptor(subs,move),*elements)
        );        
    }
    std::shared_ptr<impl_base_type> create_view_reshape(const shape_type& subs, bool move = false)const override{
        return std::static_pointer_cast<impl_base_type>(move ?
            std::make_shared<view_subdim_type>(create_view_reshape_descriptor(subs,move),std::move(const_cast<StorT>(*elements))):
            std::make_shared<view_subdim_type>(create_view_reshape_descriptor(subs,move),*elements)
        );                
    }
};

template<typename ValT, template<typename> typename Cfg, typename DescT, typename StorT>
class view_complex_descriptor_factory : public view_factory<ValT, Cfg>{
    using config_type = Cfg<ValT>;
    using index_type = typename config_type::index_type;
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using view_slice_descriptor_type = view_slice_descriptor<ValT,Cfg,DescT>;
    using view_subdim_descriptor_type = view_subdim_descriptor<ValT,Cfg,DescT>;
    using view_slice_type = gtensor::view_impl<ValT,Cfg,view_slice_descriptor_type, StorT>;
    using view_subdim_type = gtensor::view_impl<ValT,Cfg,view_subdim_descriptor_type, StorT>;

    const DescT* descriptor;
    const StorT* elements;

    view_complex_descriptor_factory(const DescT& descriptor_, const StorT& elements_):
        descriptor{&descriptor_},
        elements{&elements_},
    {}

    view_slice_descriptor_type create_view_slice_descriptor(const slices_collection_type& subs, bool move)const{
        return move ? 
            view_slice_descriptor_type{
                detail::make_view_slice_shape(descriptor->shape(),subs), 
                detail::make_view_slice_cstrides(descriptor->strides(),subs),
                detail::make_view_slice_offset(descriptor->strides(),subs),
                std::move(const_cast<DescT>(*descriptor))
            }:
            view_slice_descriptor_type{
                detail::make_view_slice_shape(descriptor->shape(),subs), 
                detail::make_view_slice_cstrides(descriptor->strides(),subs),
                detail::make_view_slice_offset(descriptor->strides(),subs),
                *descriptor
            };
    }    
    view_slice_descriptor_type create_view_transpose_descriptor(const shape_type& subs, bool move)const{
        return move ? 
            view_slice_descriptor_type{
                detail::transpose(descriptor->shape(),subs), 
                detail::transpose(descriptor->strides(),subs),
                index_type(0),
                std::move(const_cast<DescT>(*descriptor))
            }:
            view_slice_descriptor_type{
                detail::transpose(descriptor->shape(),subs), 
                detail::transpose(descriptor->strides(),subs),
                index_type(0),
                *descriptor
            };
    }
    view_subdim_descriptor_type create_view_subdim_descriptor(const shape_type& subs, bool move)const{
        return move ?
            view_subdim_descriptor_type{
                detail::make_view_subdim_shape(descriptor->shape(),subs), 
                detail::make_view_subdim_offset(descriptor->strides(),subs),
                std::move(const_cast<DescT>(*descriptor))
            }:
            view_subdim_descriptor_type{
                detail::make_view_subdim_shape(descriptor->shape(),subs), 
                detail::make_view_subdim_offset(descriptor->strides(),subs),
                *descriptor
            };
    }
    view_subdim_descriptor_type create_view_reshape_descriptor(const shape_type& subs, bool move)const{
        return move ? 
            view_subdim_descriptor_type{subs,index_type(0),std::move(const_cast<DescT>(*descriptor))}:
            view_subdim_descriptor_type{subs,index_type(0),*descriptor};
    }

    std::shared_ptr<impl_base_type> create_view_slice(const slices_collection_type& subs, bool move = false)const override{
        return std::static_pointer_cast<impl_base_type>(move ? 
            std::make_shared<view_slice_type>(create_view_slice_descriptor(subs,move),std::move(const_cast<StorT>(*elements))):
            std::make_shared<view_slice_type>(create_view_slice_descriptor(subs,move),*elements)
        );        
    }
    std::shared_ptr<impl_base_type> create_view_transpose(const shape_type& subs, bool move = false)const override{
        return std::static_pointer_cast<impl_base_type>(move ?
            std::make_shared<view_slice_type>(create_view_transpose_descriptor(subs,move),std::move(const_cast<StorT>(*elements))):
            std::make_shared<view_slice_type>(create_view_transpose_descriptor(subs,move),*elements):
        );        
    }
    std::shared_ptr<impl_base_type> create_view_subdim(const shape_type& subs, bool move = false)const override{
        return std::static_pointer_cast<impl_base_type>(move ?
            std::make_shared<view_subdim_type>(create_view_subdim_descriptor(subs,move),std::move(const_cast<StorT>(*elements))):
            std::make_shared<view_subdim_type>(create_view_subdim_descriptor(subs,move),*elements)
        );        
    }
    std::shared_ptr<impl_base_type> create_view_reshape(const shape_type& subs, bool move = false)const override{
        return std::static_pointer_cast<impl_base_type>(move ?
            std::make_shared<view_subdim_type>(create_view_reshape_descriptor(subs,move),std::move(const_cast<StorT>(*elements))):
            std::make_shared<view_subdim_type>(create_view_reshape_descriptor(subs,move),*elements)
        );                
    }
};

template<typename ValT, template<typename> typename Cfg, typename DescT, typename StorT, typename CacheT>
class view_of_expression_factory : public view_factory<ValT, Cfg>{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using view_cached_factory_type = view_simple_descriptor_factory<ValT,Cfg,DescT,CacheT>;
    using view_not_cached_factory_type = view_simple_descriptor_factory<ValT,Cfg,DescT,StorT>;

    std::unique_ptr<view_factory<ValT,Cfg>> factory;

    view_of_expression_factory(const DescT& descriptor_, const StorT& elements_, const CacheT& cache_, bool is_cached_):
        factory{is_cached_ ? new view_cached_factory_type{descriptor_,cache_} : new view_not_cached_factory_type{descriptor_,elements_} };
    {}

    std::shared_ptr<impl_base_type> create_view_slice(const slices_collection_type& subs, bool move = false)const override{return factory->create_view_slice(subs,move);}
    std::shared_ptr<impl_base_type> create_view_transpose(const shape_type& subs, bool move = false)const override{return factory->create_view_transpose(subs,move);}
    std::shared_ptr<impl_base_type> create_view_subdim(const shape_type& subs, bool move = false)const override{return factory->create_view_subdim(subs,move);}
    std::shared_ptr<impl_base_type> create_view_reshape(const shape_type& subs, bool move = false)const override{return factory->create_view_reshape(subs,move);}
};

template<typename ValT, template<typename> typename Cfg, typename DescT, typename StorT, typename CacheT>
class view_of_view_factory : public view_factory<ValT, Cfg>{
    using impl_base_type = tensor_impl_base<ValT,Cfg>;
    using view_cached_factory_type = view_simple_descriptor_factory<ValT,Cfg,DescT,CacheT>;
    using view_not_cached_factory_type = view_complex_descriptor_factory<ValT,Cfg,DescT,StorT>;

    std::unique_ptr<view_factory<ValT,Cfg>> factory;

    view_of_view_factory(const DescT& descriptor_, const StorT& elements_, const CacheT& cache_, bool is_cached_):
        factory{is_cached_ ? new view_cached_factory_type{descriptor_,cache_} : new view_not_cached_factory_type{descriptor_,elements_} };
    {}

    std::shared_ptr<impl_base_type> create_view_slice(const slices_collection_type& subs, bool move = false)const override{return factory->create_view_slice(subs,move);}
    std::shared_ptr<impl_base_type> create_view_transpose(const shape_type& subs, bool move = false)const override{return factory->create_view_transpose(subs,move);}
    std::shared_ptr<impl_base_type> create_view_subdim(const shape_type& subs, bool move = false)const override{return factory->create_view_subdim(subs,move);}
    std::shared_ptr<impl_base_type> create_view_reshape(const shape_type& subs, bool move = false)const override{return factory->create_view_reshape(subs,move);}
};


template<typename ValT, template<typename> typename Cfg>
template<typename...T, typename DescT, typename StorT>
static std::unique_ptr<view_factory<ValT,Cfg>> view_factory<ValT,Cfg>::create_factory(const stensor_impl<T...>&, const DescT& descriptor, const StorT& elements){
    return std::unique_ptr<view_factory>{new view_simple_descriptor_factory<ValT, Cfg, DescT, StorT>{descriptor,elements}};
}
template<typename ValT, template<typename> typename Cfg>
template<typename...T, typename DescT, typename StorT, typename CacheT>
static std::unique_ptr<view_factory<ValT,Cfg>> view_factory<ValT,Cfg>::create_factory(const expression_impl<T...>& parent, const DescT& descriptor, const StorT& elements, const CacheT& cache){
    return std::unique_ptr<view_factory>{new view_of_expression_factory<ValT, Cfg, DescT, CacheT>{descriptor,elements, cache, parent.is_cached()}};
}
template<typename ValT, template<typename> typename Cfg>
template<typename...T, typename DescT, typename StorT, typename CacheT>
static std::unique_ptr<view_factory<ValT,Cfg>> view_factory<ValT,Cfg>::create_factory(const view_impl<T...>& parent, const DescT& descriptor, const StorT& elements, const CacheT& cache){
    return std::unique_ptr<view_factory>{new view_of_view_factory<ValT, Cfg, DescT, CacheT>{descriptor,elements, cache, parent.is_cached()}};
}


}   //end of namespace gtensor



#endif