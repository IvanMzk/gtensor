#ifndef WALKER_FACTORY_HPP_
#define WALKER_FACTORY_HPP_

#include <memory>
#include "forward_decl.hpp"
#include "impl_swalker.hpp"
#include "impl_ewalker.hpp"


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
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    const shape_type* shape;
    const shape_type* strides;
    const value_type* data;
public:
    storage_walker_factory(const shape_type& shape_, const shape_type& strides_, const value_type* data_):
        shape{&shape_},
        strides{&strides_},
        data{data_},
    {}
    walker<ValT, Cfg> create_walker()const override{
        return new storage_walker_impl<ValT,Cfg>{*shape, *strides, data};
    }
};

template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
class evaluating_walker_factory : public walker_factory_base<ValT,Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using impl_base_type = walker_impl_base<ValT, Cfg>;
    using expression_walker_type = expression_walker_impl<ValT,Cfg,F,decltype(std::declval<Ops>()->create_walker())...>;

    const shape_type* shape;
    const F* f;
    const std::tuple<Ops...>* operands;
    
    template<std::size_t...I>
    walker<ValT,Cfg> create_walker_helper(std::index_sequence<I...>)const{
        return new evaluating_walker_impl{*shape,*f,std::get<I>(*operands)->create_walker()...};
    }
public:
    expression_walker_factory(const shape_type& shape_, const F& f_, const std::tuple<Ops...>& operands_):
        shape{&shape_},
        f{&f_},
        operands{&operands_}
    {}
    walker<ValT, Cfg> create_walker()const override{
        return create_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
    }
};

template<typename ValT, template<typename> typename Cfg, typename DescT, typename F, typename CacheT, typename...Ops>
class walker_of_expression_factory : public walker_factory_base<ValT,Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using parent_base_type = tensor_impl_base<ValT,CfgT>;    
    
    const parent_base_type* parent;
    storage_walker_factory<ValT,Cfg> storage_walker_maker;
    evaluating_walker_factory<ValT,Cfg,F,Ops...> evaluating_walker_maker;

    walker<ValT, Cfg> create_walker()const override{
        return parent->is_cached() ? storage_walker_maker.create_walker() : evaluating_walker_maker.create_walker();
    }    
public:
    walker_of_expression_factory(const parent_base_type& parent_, const DescT& descriptor_, const F& f_, const CacheT& cache_, const std::tuple<Ops...>& operands_):
        parent{&parent_},
        storage_walker_maker{descriptor_.shape(), descriptor_.strides(), cache_.data()},
        evaluating_walker_maker{descriptor_.shape(), f_, operands_}.create_walker()
    {}
};



template<typename ValT, template<typename> typename Cfg>
class walker_factory{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;

    std::unique_ptr<walker_factory_base<ValT,Cfg>> factory;
    
    template<typename...T, typename DescT, typename StorT>
    void reset_factory(const stensor_impl<T...>& parent, DescT& descriptor, StorT& elements){
        factory.reset(new storage_walker_factory<ValT, Cfg>{descriptor.shape(),descriptor.strides(),elements.data()});
    }
public:    
    walker<ValT, Cfg> create_walker()const override{return factory->create_walker();}

    template<typename...T, typename DescT, typename StorT>
    walker_factory(const stensor_impl<T...>& parent, DescT& descriptor, StorT& elements){
        reset_factory(parent, descriptor, elements);
    }
};




}   //end of namespace gtensor


#endif