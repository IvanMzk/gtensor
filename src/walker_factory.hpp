#ifndef WALKER_FACTORY_HPP_
#define WALKER_FACTORY_HPP_

#include <memory>
#include "forward_decl.hpp"
#include "impl_swalker.hpp"
#include "impl_ewalker.hpp"
#include "impl_vwalker.hpp"
#include "impl_ewalker_trivial.hpp"
#include "dispatcher.hpp"


namespace gtensor{

namespace detail{

 template<typename ValT, template<typename> typename Cfg>
 bool is_storage(const tensor_impl_base<ValT,Cfg>& t){
    return t.tensor_kind() == detail::tensor_kinds::storage_tensor || 
        t.tensor_kind() == detail::tensor_kinds::expression && t.is_storage() ||
        t.tensor_kind() == detail::tensor_kinds::view && t.as_view()->is_cached();
}


}   //end of namespace detail


template<typename ValT, template<typename> typename Cfg>
class walker_factory_base{
public:
    virtual ~walker_factory_base(){}
    virtual walker<ValT, Cfg> create_walker()const = 0;
};

template<typename ValT, template<typename> typename Cfg>
class storage_walker_factory
{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using shape_type = typename config_type::shape_type;
public: 
    static storage_walker_impl<ValT, Cfg> create_walker(const shape_type& shape, const shape_type& strides, const value_type* data){
        return storage_walker_impl<ValT,Cfg>{shape, strides, data};
    }
};

template<typename ValT, template<typename> typename Cfg>
class trivial_walker_factory
{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using shape_type = typename config_type::shape_type;
public: 
    static ewalker_trivial_impl<ValT, Cfg> create_walker(const shape_type& shape, const shape_type& strides, const tensor_impl_base<ValT,Cfg>& parent){
        return ewalker_trivial_impl<ValT,Cfg>{shape, strides, parent};
    }
};

template<typename ValT, template<typename> typename Cfg>
class evaluating_walker_factory
{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using shape_type = typename config_type::shape_type;
    
    template<typename F>
    struct walker_maker{
        const shape_type& shape;
        walker_maker(const shape_type& shape_):
            shape{shape_}
        {}
        template<typename...Args>
        walker<ValT,Cfg> operator()(const Args&...args)const{
            using evaluating_walker_type = evaluating_walker_impl<ValT,Cfg,F,decltype(std::declval<Args>().create_walker())...>;
            return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new evaluating_walker_type{shape,args.create_walker()...}};
        }
    };
    template<typename MakerT, typename...Ops, std::size_t...I>
    static walker<ValT, Cfg> create_walker_helper(const MakerT& maker, const std::tuple<Ops...>& operands, std::index_sequence<I...>){
        using dispatcher_type = detail::dispatcher<ValT,Cfg>;
        return dispatcher_type::call(maker, *std::get<I>(operands)...);
    }
    
    template<typename StT, typename F>
    struct storage_maker{
        using strides_type = StT;
        const shape_type& shape;
        const strides_type& strides;
        storage_maker(const shape_type& shape_, const strides_type& strides_):
            shape{shape_},
            strides{strides_}
        {}
        template<typename...Args>
        evaluating_storage<ValT,Cfg> operator()(const Args&...args)const{
            using evaluating_storage_type = evaluating_storage_impl<ValT,Cfg,F,decltype(std::declval<Args>().create_walker())...>;
            return std::unique_ptr<evaluating_storage_impl_base<ValT,Cfg>>{new evaluating_storage_type{shape,strides,args.create_walker()...}};
        }
    };
    template<typename MakerT, typename...Ops, std::size_t...I>
    static evaluating_storage<ValT,Cfg> create_storage_helper(const MakerT& maker, const std::tuple<Ops...>& operands, std::index_sequence<I...>){
        using dispatcher_type = detail::dispatcher<ValT,Cfg>;
        return dispatcher_type::call(maker, *std::get<I>(operands)...);
    }
public: 
    template<typename F, typename...Ops>
    static walker<ValT, Cfg> create_walker(const shape_type& shape, const F&, const std::tuple<Ops...>& operands){
        using maker_type = walker_maker<F>;
        return create_walker_helper(maker_type{shape}, operands, std::make_index_sequence<sizeof...(Ops)>{});
    }
    template<typename StT, typename F, typename...Ops>
    static evaluating_storage<ValT,Cfg> create_storage(const shape_type& shape, const StT& strides, const F&, const std::tuple<Ops...>& operands){
        using maker_type = storage_maker<StT, F>;
        return create_storage_helper(maker_type{shape, strides}, operands, std::make_index_sequence<sizeof...(Ops)>{});
    }
};




template<typename ValT, template<typename> typename Cfg>
class polymorphic_walker_factory
{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using shape_type = typename config_type::shape_type;    
    using index_type = typename config_type::index_type;    

    static walker<ValT,Cfg> create_walker_helper(const tensor_impl_base<ValT,Cfg>&, const shape_type& shape, const shape_type& strides, const value_type* data){
        return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new storage_walker_impl<ValT,Cfg>{shape,strides,data}};
    }    
    template<typename F, typename...Ops>
    static walker<ValT,Cfg> create_walker_helper(const tensor_impl_base<ValT,Cfg>& expression,const F& f, const std::tuple<Ops...>& operands, const value_type* cache){        
        if (expression.is_storage()){
            return create_walker_helper(expression, expression.descriptor().shape(), expression.descriptor().strides(), cache);
        }else if(expression.is_trivial()){
            return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new ewalker_trivial_impl<ValT,Cfg>{expression.descriptor().shape(),expression.descriptor().strides(),expression}};
        }else{
            return create_evaluating_walker_helper(expression.descriptor().shape(), f, operands, std::make_index_sequence<sizeof...(Ops)>{});
        }
    }
    static walker<ValT,Cfg> create_walker_helper(
                                                const tensor_impl_base<ValT,Cfg>& view, 
                                                const tensor_impl_base<ValT,Cfg>& view_parent, 
                                                const tensor_impl_base<ValT,Cfg>& view_root, 
                                                const value_type* cache)
    {
        if (detail::is_storage(view)){
            return create_walker_helper(view, view.descriptor().shape(), view.descriptor().strides(), cache);
        }else if(detail::is_storage(view_parent)){
            return create_walker_helper(view, view.descriptor().shape(), view.descriptor().cstrides(), view_parent.as_storage_tensor()->data()+view.descriptor().offset());
        }else if(view_root.tensor_kind() == detail::tensor_kinds::expression){
            return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new view_expression_walker_impl<ValT,Cfg>{
                view.descriptor().shape(),
                view.descriptor().cstrides(),
                view.descriptor().offset(), 
                view_parent.as_index_converter(), 
                view_root.as_expression()->create_storage()
                }
            };
        }else{
            return nullptr;
        }
    }

    


    template<typename F, typename...Ops, std::size_t...I>
    static walker<ValT,Cfg> create_evaluating_walker_helper(const shape_type& shape, const F&, const std::tuple<Ops...>& operands, std::index_sequence<I...>){
        using evaluating_walker_type = evaluating_walker_impl<ValT,Cfg,F,decltype(std::declval<Ops>()->as_walker_maker()->create_walker())...>;
        return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new evaluating_walker_type{shape,std::get<I>(operands)->as_walker_maker()->create_walker()...}};
    }


public:
    template<typename...Args>
    static walker<ValT, Cfg> create_walker(const tensor_impl_base<ValT,Cfg>& t, Args&&...args){
        return create_walker_helper(t, std::forward<Args>(args)...);
    }
};







// template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
// class evaluating_walker_factory : public walker_factory_base<ValT,Cfg>{
//     using config_type = Cfg<ValT>;        
//     using value_type = ValT;
//     using shape_type = typename config_type::shape_type;
//     using evaluating_walker_type = evaluating_walker_impl<ValT,Cfg,F,decltype(std::declval<Ops>()->create_walker())...>;

//     const shape_type* shape;
//     const F* f;
//     const std::tuple<Ops...>* operands;
    
//     template<std::size_t...I>
//     walker<ValT,Cfg> create_walker_helper(std::index_sequence<I...>)const{
//         return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new evaluating_walker_type{*shape,*f,std::get<I>(*operands)->create_walker()...}};
//     }
// public:
//     evaluating_walker_factory(const shape_type& shape_, const F& f_, const std::tuple<Ops...>& operands_):
//         shape{&shape_},
//         f{&f_},
//         operands{&operands_}
//     {}
//     walker<ValT, Cfg> create_walker()const override{
//         return create_walker_helper(std::make_index_sequence<sizeof...(Ops)>{});
//     }
// };

// template<typename ValT, template<typename> typename Cfg>
// class trivial_ewalker_factory : public walker_factory_base<ValT,Cfg>{
//     using config_type = Cfg<ValT>;        
//     using value_type = ValT;
//     using shape_type = typename config_type::shape_type;
//     using trivial_ewalker_type = ewalker_trivial_impl<ValT,Cfg>;

//     const shape_type* shape;
//     const shape_type* strides;
//     const tensor_impl_base<ValT,Cfg>* parent;
    
// public:
//     trivial_ewalker_factory(const shape_type& shape_, const shape_type& strides_, const tensor_impl_base<ValT,Cfg>& parent_):
//         shape{&shape_},
//         strides{&strides_},
//         parent{&parent_}
//     {}
//     walker<ValT, Cfg> create_walker()const override{
//         return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new trivial_ewalker_type{*shape,*strides,*parent}};
//     }
// };

// template<typename ValT, template<typename> typename Cfg, typename ParentT, typename DescT, typename F, typename...Ops>
// class walker_of_expression_factory : public walker_factory_base<ValT,Cfg>{
//     using config_type = Cfg<ValT>;        
//     using value_type = ValT;
//     using index_type = typename config_type::index_type;
//     using shape_type = typename config_type::shape_type;
    
//     const ParentT* parent;
//     storage_walker_factory<ValT,Cfg> storage_walker_maker;
//     evaluating_walker_factory<ValT,Cfg,F,Ops...> evaluating_walker_maker;
//     trivial_ewalker_factory<ValT,Cfg> trivial_ewalker_maker;

//     walker<ValT, Cfg> create_walker()const override{
//         if (parent->is_cached()){
//             return storage_walker_maker.create_walker();
//         }else if(parent->is_trivial()){
//             return trivial_ewalker_maker.create_walker();
//         }else{
//             return evaluating_walker_maker.create_walker();
//         }
//         //return parent->is_cached() ? storage_walker_maker.create_walker() : evaluating_walker_maker.create_walker();
//     }    
// public:
//     template<typename CacheT>
//     walker_of_expression_factory(const ParentT& parent_, const DescT& descriptor_, const F& f_, const CacheT& cache_, const std::tuple<Ops...>& operands_):
//         parent{&parent_},
//         storage_walker_maker{descriptor_.shape(), descriptor_.strides(), cache_.data()},
//         evaluating_walker_maker{descriptor_.shape(), f_, operands_},
//         trivial_ewalker_maker{descriptor_.shape(), descriptor_.strides(), parent_}
//     {}
// };

// template<typename ValT, template<typename> typename Cfg, typename ParentT, typename DescT, typename StorT>
// class walker_of_view_factory : public walker_factory_base<ValT,Cfg>{    
//     using vwalker_type = vwalker_impl<ValT,Cfg,DescT,StorT>;
    
//     const ParentT* parent;
//     const DescT* descriptor;
//     const StorT* elements;
//     storage_walker_factory<ValT,Cfg> storage_walker_maker;

//     walker<ValT, Cfg> create_vwalker_helper()const{        
//         return std::unique_ptr<walker_impl_base<ValT,Cfg>>{new vwalker_type{*descriptor,*elements}};
//     }
//     walker<ValT, Cfg> create_walker()const override{        
//         return parent->is_cached() ? storage_walker_maker.create_walker() : create_vwalker_helper();
//     }    
// public:
//     template<typename CacheT>
//     walker_of_view_factory(const ParentT& parent_, const DescT& descriptor_, const StorT& elements_, const CacheT& cache_):
//         parent{&parent_},
//         descriptor{&descriptor_},
//         elements{&elements_},
//         storage_walker_maker{descriptor_.shape(), descriptor_.strides(), cache_.data()}
//     {}
// };

// template<typename ValT, template<typename> typename Cfg>
// class walker_factory : public walker_factory_base<ValT,Cfg>{
//     using config_type = Cfg<ValT>;        
//     using value_type = ValT;
//     using index_type = typename config_type::index_type;
//     using shape_type = typename config_type::shape_type;

//     std::unique_ptr<walker_factory_base<ValT,Cfg>> factory;
    
//     template<typename...T, typename DescT, typename StorT>
//     void reset_factory(const stensor_impl<T...>& parent, DescT& descriptor, StorT& elements){
//         factory.reset(new storage_walker_factory<ValT, Cfg>{descriptor.shape(),descriptor.strides(),elements.data()});
//     }
//     template<typename...T, typename DescT, typename F, typename CacheT, typename...Ops>
//     void reset_factory(const expression_impl<T...>& parent_, const DescT& descriptor_, const F& f_, const CacheT& cache_, const std::tuple<Ops...>& operands_){
//         using parent_type = expression_impl<T...>;
//         factory.reset(new walker_of_expression_factory<ValT,Cfg,parent_type,DescT,F,Ops...>{parent_,descriptor_,f_,cache_,operands_});
//     }
//     template<typename...T, typename DescT, typename StorT, typename CacheT>
//     void reset_factory(const view_impl<T...>& parent_, const DescT& descriptor_, const StorT& elements_, const CacheT& cache_){
//         using parent_type = view_impl<T...>;
//         factory.reset(new walker_of_view_factory<ValT,Cfg,parent_type,DescT,StorT>{parent_,descriptor_,elements_,cache_});
//     }
// public:    
//     walker<ValT, Cfg> create_walker()const override{return factory->create_walker();}

//     walker_factory() = default;
//     walker_factory(const walker_factory&) = delete;
//     walker_factory& operator=(const walker_factory&) = delete;
//     walker_factory(walker_factory&&) = delete;
//     walker_factory& operator=(walker_factory&&) = delete;

//     template<typename...T, typename DescT, typename StorT>
//     walker_factory(const stensor_impl<T...>& parent, DescT& descriptor, StorT& elements){
//         reset_factory(parent, descriptor, elements);
//     }
//     template<typename...T, typename DescT, typename F, typename CacheT, typename...Ops>
//     walker_factory(const expression_impl<T...>& parent_, const DescT& descriptor_, const F& f_, const CacheT& cache_, const std::tuple<Ops...>& operands_){
//         reset_factory(parent_,descriptor_,f_,cache_,operands_);
//     }
//     template<typename...T, typename DescT, typename StorT, typename CacheT>
//     walker_factory(const view_impl<T...>& parent_, const DescT& descriptor_, const StorT& elements_, const CacheT& cache_){
//         reset_factory(parent_,descriptor_,elements_,cache_);
//     }
// };




}   //end of namespace gtensor


#endif