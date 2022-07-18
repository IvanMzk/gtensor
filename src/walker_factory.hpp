#ifndef WALKER_FACTORY_HPP_
#define WALKER_FACTORY_HPP_

#include <memory>
#include "forward_decl.hpp"
#include "impl_swalker.hpp"
#include "impl_ewalker.hpp"
#include "impl_vwalker.hpp"
#include "impl_ewalker_trivial.hpp"


namespace gtensor{

namespace detail{


}   //end of namespace detail


template<typename ValT, template<typename> typename Cfg>
class walker_factory_base{
public:
    virtual ~walker_factory_base(){}
    virtual walker<ValT, Cfg> create_walker()const = 0;
};

template<typename ValT, template<typename> typename Cfg>
class storage_walker_factory : public walker_factory_base<ValT,Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using shape_type = typename config_type::shape_type;

    const shape_type* shape;
    const shape_type* strides;
    const value_type* data;
public:
    storage_walker_factory(const shape_type& shape_, const shape_type& strides_, const value_type* data_):
        shape{&shape_},
        strides{&strides_},
        data{data_}
    {}
    walker<ValT, Cfg> create_walker()const override{
        return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new storage_walker_impl<ValT,Cfg>{*shape, *strides, data}};
    }
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class evaluating_walker_factory : public walker_factory_base<ValT,Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using shape_type = typename config_type::shape_type;
    using evaluating_walker_type = evaluating_walker_impl<ValT,Cfg,F,decltype(std::declval<Ops>()->create_walker())...>;

    const shape_type* shape;
    const F* f;
    const std::tuple<Ops...>* operands;
    
    template<std::size_t...I>
    walker<ValT,Cfg> create_walker_helper(std::index_sequence<I...>)const{
        return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new evaluating_walker_type{*shape,*f,std::get<I>(*operands)->create_walker()...}};
    }
public:
    evaluating_walker_factory(const shape_type& shape_, const F& f_, const std::tuple<Ops...>& operands_):
        shape{&shape_},
        f{&f_},
        operands{&operands_}
    {}
    walker<ValT, Cfg> create_walker()const override{
        return create_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
    }
};

template<typename ValT, template<typename> typename Cfg>
class trivial_ewalker_factory : public walker_factory_base<ValT,Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using shape_type = typename config_type::shape_type;
    using trivial_ewalker_type = ewalker_trivial_impl<ValT,Cfg>;

    const shape_type* shape;
    const shape_type* strides;
    const tensor_impl_base<ValT,Cfg>* parent;
    
public:
    trivial_ewalker_factory(const shape_type& shape_, const shape_type& strides_, const tensor_impl_base<ValT,Cfg>& parent_):
        shape{&shape_},
        strides{&strides_},
        parent{&parent_}
    {}
    walker<ValT, Cfg> create_walker()const override{
        return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new trivial_ewalker_type{*shape,*strides,*parent}};
    }
};

template<typename ValT, template<typename> typename Cfg, typename ParentT, typename DescT, typename F, typename...Ops>
class walker_of_expression_factory : public walker_factory_base<ValT,Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    
    const ParentT* parent;
    storage_walker_factory<ValT,Cfg> storage_walker_maker;
    evaluating_walker_factory<ValT,Cfg,F,Ops...> evaluating_walker_maker;
    trivial_ewalker_factory<ValT,Cfg> trivial_ewalker_maker;

    walker<ValT, Cfg> create_walker()const override{
        if (parent->is_cached()){
            return storage_walker_maker.create_walker();
        }else if(parent->is_trivial()){
            return trivial_ewalker_maker.create_walker();
        }else{
            return evaluating_walker_maker.create_walker();
        }
        //return parent->is_cached() ? storage_walker_maker.create_walker() : evaluating_walker_maker.create_walker();
    }    
public:
    template<typename CacheT>
    walker_of_expression_factory(const ParentT& parent_, const DescT& descriptor_, const F& f_, const CacheT& cache_, const std::tuple<Ops...>& operands_):
        parent{&parent_},
        storage_walker_maker{descriptor_.shape(), descriptor_.strides(), cache_.data()},
        evaluating_walker_maker{descriptor_.shape(), f_, operands_},
        trivial_ewalker_maker{descriptor_.shape(), descriptor_.strides(), parent_}
    {}
};

template<typename ValT, template<typename> typename Cfg, typename ParentT, typename DescT, typename StorT>
class walker_of_view_factory : public walker_factory_base<ValT,Cfg>{    
    using vwalker_type = vwalker_impl<ValT,Cfg,DescT,StorT>;
    
    const ParentT* parent;
    const DescT* descriptor;
    const StorT* elements;
    storage_walker_factory<ValT,Cfg> storage_walker_maker;

    walker<ValT, Cfg> create_vwalker_helper()const{        
        return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new vwalker_type{*descriptor,*elements}};
    }
    walker<ValT, Cfg> create_walker()const override{        
        return parent->is_cached() ? storage_walker_maker.create_walker() : create_vwalker_helper();
    }    
public:
    template<typename CacheT>
    walker_of_view_factory(const ParentT& parent_, const DescT& descriptor_, const StorT& elements_, const CacheT& cache_):
        parent{&parent_},
        descriptor{&descriptor_},
        elements{&elements_},
        storage_walker_maker{descriptor_.shape(), descriptor_.strides(), cache_.data()}
    {}
};

template<typename ValT, template<typename> typename Cfg>
class walker_factory : public walker_factory_base<ValT,Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    std::unique_ptr<walker_factory_base<ValT,Cfg>> factory;
    
    template<typename...T, typename DescT, typename StorT>
    void reset_factory(const stensor_impl<T...>& parent, DescT& descriptor, StorT& elements){
        factory.reset(new storage_walker_factory<ValT, Cfg>{descriptor.shape(),descriptor.strides(),elements.data()});
    }
    template<typename...T, typename DescT, typename F, typename CacheT, typename...Ops>
    void reset_factory(const expression_impl<T...>& parent_, const DescT& descriptor_, const F& f_, const CacheT& cache_, const std::tuple<Ops...>& operands_){
        using parent_type = expression_impl<T...>;
        factory.reset(new walker_of_expression_factory<ValT,Cfg,parent_type,DescT,F,Ops...>{parent_,descriptor_,f_,cache_,operands_});
    }
    template<typename...T, typename DescT, typename StorT, typename CacheT>
    void reset_factory(const view_impl<T...>& parent_, const DescT& descriptor_, const StorT& elements_, const CacheT& cache_){
        using parent_type = view_impl<T...>;
        factory.reset(new walker_of_view_factory<ValT,Cfg,parent_type,DescT,StorT>{parent_,descriptor_,elements_,cache_});
    }
public:    
    walker<ValT, Cfg> create_walker()const override{return factory->create_walker();}

    walker_factory() = default;
    walker_factory(const walker_factory&) = delete;
    walker_factory& operator=(const walker_factory&) = delete;
    walker_factory(walker_factory&&) = delete;
    walker_factory& operator=(walker_factory&&) = delete;

    template<typename...T, typename DescT, typename StorT>
    walker_factory(const stensor_impl<T...>& parent, DescT& descriptor, StorT& elements){
        reset_factory(parent, descriptor, elements);
    }
    template<typename...T, typename DescT, typename F, typename CacheT, typename...Ops>
    walker_factory(const expression_impl<T...>& parent_, const DescT& descriptor_, const F& f_, const CacheT& cache_, const std::tuple<Ops...>& operands_){
        reset_factory(parent_,descriptor_,f_,cache_,operands_);
    }
    template<typename...T, typename DescT, typename StorT, typename CacheT>
    walker_factory(const view_impl<T...>& parent_, const DescT& descriptor_, const StorT& elements_, const CacheT& cache_){
        reset_factory(parent_,descriptor_,elements_,cache_);
    }
};




}   //end of namespace gtensor


#endif