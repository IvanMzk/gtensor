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

template<typename ValT, template<typename> typename Cfg, typename ParentT, typename DescT, typename StorT>
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

    std::shared_ptr<impl_base_type> create_slice_view(const slices_collection_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_transpose_view(const shape_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_subdim_view(const shape_type&)const override{return nullptr;}
    std::shared_ptr<impl_base_type> create_reshape_view(const shape_type&)const override{return nullptr;}
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