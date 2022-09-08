#ifndef WALKER_FACTORY_HPP_
#define WALKER_FACTORY_HPP_

#include <memory>
#include "forward_decl.hpp"
#include "storage_walker.hpp"
#include "evaluating_walker.hpp"
#include "viewing_walker.hpp"
#include "trivial_walker.hpp"
#include "dispatcher.hpp"


namespace gtensor{

namespace detail{

//  template<typename ValT, typename CfgT>
//  bool is_storage(const tensor_base<ValT,CfgT>& t){
//     return t.tensor_kind() == detail::tensor_kinds::storage_tensor || 
//         t.tensor_kind() == detail::tensor_kinds::expression && t.is_storage() ||
//         t.tensor_kind() == detail::tensor_kinds::view && t.is_cached();
// }

// template<typename CfgT, std::enable_if_t<detail::is_mode_div_native<CfgT> ,int> =0 >
// inline const auto& strides_div(const descriptor_with_libdivide<CfgT>& desc){
//     return desc.strides();
// }
// template<typename CfgT, std::enable_if_t<detail::is_mode_div_libdivide<CfgT> ,int> =0 >
// inline const auto& strides_div(const descriptor_with_libdivide<CfgT>& desc){
//     return desc.strides_libdivide();
// }

}   //end of namespace detail



template<typename ValT, typename CfgT>
class storage_walker_factory
{    
    using shape_type = typename CfgT::shape_type;
public: 
    static auto create_walker(const shape_type& shape, const shape_type& strides, const ValT* data){
        return storage_walker<ValT,CfgT>{shape, strides, data};
    }
};

// template<typename ValT, typename CfgT>
// class trivial_walker_factory
// {    
//     using value_type = ValT;
//     using shape_type = typename CfgT::shape_type;
// public: 
//     static evaluating_trivial_walker<ValT, CfgT> create_walker(const shape_type& shape, const shape_type& strides, const tensor_base<ValT,CfgT>& parent){
//         return evaluating_trivial_walker<ValT,CfgT>{shape, strides, parent};
//     }
// };




// template<typename ValT, typename CfgT>
// class expression_template_evaluation_base{
// public:
//     virtual ~expression_template_evaluation_base(){}
//     virtual walker<ValT, CfgT> create_evaluator()const = 0;
//     virtual bool is_trivial_evaluation()const = 0;
// };


//elementwise_evaluation class is responsible for handling arithmetic operations +,-,*,/,<,>, ...
//i.e. such operations that can be done in elemenwise fashion, evaluation is broadcasted if possible
//depending on config it also may cache operands to make broadcast evaluation more efficient
//evaluation can be done by pure elementwise calculations (trivial broadcasting) if all nodes in evaluation tree support such an evaluation
//for elementwise_evaluation it depends only on shapes of nodes in tree, but for other routines in may differ
// template<typename ValT, typename CfgT, typename F, typename...Ops>
// class expression_template_elementwise_evaluation
// {    
//     using value_type = ValT;
//     using shape_type = typename CfgT::shape_type;    
    
//     const tensor_base<ValT,CfgT>* root;
//     F f;
//     std::tuple<std::shared_ptr<tensor_base<typename Ops::value_type, CfgT> >...> operands;

    
//     auto walker_maker()const{
//         return [this](const auto&...args){
//             using evaluating_walker_type = evaluating_walker<ValT,CfgT,F,decltype(std::declval<decltype(args)>().create_walker())...>;
//             return walker<ValT,CfgT>{std::make_unique<evaluating_walker_polymorphic<ValT,CfgT,evaluating_walker_type>>(evaluating_walker_type{root->shape(),args.create_walker()...})};
//             };
//     }
    
//     auto indexer_maker()const{
//         return [this](const auto&...args){
//                 using evaluating_walker_type = evaluating_walker<ValT,CfgT,F,decltype(std::declval<decltype(args)>().create_walker())...>;
//                 using evaluating_indexer_type = evaluating_indexer<ValT,CfgT,evaluating_walker_type>;
//                 return indexer<ValT,CfgT>{std::make_unique<evaluating_indexer_type>(detail::strides_div(*root->descriptor().as_descriptor_with_libdivide()) , evaluating_walker_type{root->shape(),args.create_walker()...})};
//             };
//     }
    
// public:

//     template<typename...Args>
//     expression_template_elementwise_evaluation(const tensor_base<ValT,CfgT>* root_, const F& f_, const Args&...args):
//         root{root_},
//         f{f_},
//         operands{args...}
//     {}

//     walker<ValT,CfgT> create_walker()const{
//         return std::apply([this](const auto&...args){return detail::dispatcher<ValT,CfgT>::call(walker_maker(), *args...);}, operands);
//     }        
//     indexer<ValT,CfgT> create_indexer()const{
//         return std::apply([this](const auto&...args){return detail::dispatcher<ValT,CfgT>::call(indexer_maker(), *args...);}, operands);
//     }
// };

// template<typename ValT, typename CfgT>
// class evaluating_walker_factory
// {    
//     using value_type = ValT;
//     using shape_type = typename CfgT::shape_type;
    
//     template<typename F>
//     struct walker_maker{
//         const shape_type& shape;
//         walker_maker(const shape_type& shape_):
//             shape{shape_}
//         {}
//         template<typename...Args>
//         walker<ValT,CfgT> operator()(const Args&...args)const{
//             using evaluating_walker_type = evaluating_walker<ValT,CfgT,F,decltype(std::declval<Args>().create_walker())...>;
//             return std::make_unique<evaluating_walker_polymorphic<ValT,CfgT,evaluating_walker_type>>(evaluating_walker_type{shape,args.create_walker()...});            
//         }
//     };
    
//     template<typename StT, typename F>
//     struct indexer_maker{
//         using strides_type = StT;
//         const shape_type& shape;
//         const strides_type& strides;
//         indexer_maker(const shape_type& shape_, const strides_type& strides_):
//             shape{shape_},
//             strides{strides_}
//         {}
//         template<typename...Args>
//         indexer<ValT,CfgT> operator()(const Args&...args)const{
//             using evaluating_walker_type = evaluating_walker<ValT,CfgT,F,decltype(std::declval<Args>().create_walker())...>;
//             using evaluating_indexer_type = evaluating_indexer<ValT,CfgT,evaluating_walker_type>;
//             return std::make_unique<evaluating_indexer_type>(strides, evaluating_walker_type{shape,args.create_walker()...});            
//         }
//     };    

//     template<typename StT, typename F>
//     static auto maker_maker(const shape_type& shape, const StT& strides, F){return indexer_maker<StT,F>{shape,strides};}
//     template<typename F>
//     static auto maker_maker(const shape_type& shape, F){return walker_maker<F>{shape};}

//     template<typename MakerT, typename...Ops, std::size_t...I>
//     static auto create_helper(const MakerT& maker, const std::tuple<Ops...>& operands, std::index_sequence<I...>){        
//         return detail::dispatcher<ValT,CfgT>::call(maker, *std::get<I>(operands)...);
//     }

// public: 
//     template<typename DescT, typename F, typename...Ops>
//     static auto create_walker(const DescT& descriptor, const F&, const std::tuple<Ops...>& operands){
//         return create_helper(maker_maker(descriptor.shape(), F{}), operands, std::make_index_sequence<sizeof...(Ops)>{});        
//     }
//     template<typename DescT, typename F, typename...Ops>
//     static indexer<ValT,CfgT> create_indexer(const DescT& descriptor, const F&, const std::tuple<Ops...>& operands){
//         return create_helper(maker_maker(descriptor.shape(), detail::strides_div(descriptor), F{}), operands, std::make_index_sequence<sizeof...(Ops)>{});
//     }
// };




// template<typename ValT, template<typename> typename Cfg>
// class polymorphic_walker_factory
// {
//     using config_type = Cfg<ValT>;        
//     using value_type = ValT;
//     using shape_type = typename config_type::shape_type;    
//     using index_type = typename config_type::index_type;    

//     static walker<ValT,Cfg> create_walker_helper(const tensor_base<ValT,Cfg>&, const shape_type& shape, const shape_type& strides, const value_type* data){
//         return std::unique_ptr<walker_base<ValT,Cfg>>{new storage_walker_polymorphic<ValT,Cfg>{shape,strides,data}};
//     }    
//     template<typename F, typename...Ops>
//     static walker<ValT,Cfg> create_walker_helper(const tensor_base<ValT,Cfg>& expression,const F& f, const std::tuple<Ops...>& operands, const value_type* cache){        
//         if (expression.is_storage()){
//             return create_walker_helper(expression, expression.descriptor().shape(), expression.descriptor().strides(), cache);
//         }else if(expression.is_trivial()){
//             return std::unique_ptr<walker_base<ValT,Cfg>>{new evaluating_trivial_walker<ValT,Cfg>{expression.descriptor().shape(),expression.descriptor().strides(),expression}};
//         }else{
//             return create_evaluating_walker_helper(expression.descriptor().shape(), f, operands, std::make_index_sequence<sizeof...(Ops)>{});
//         }
//     }
//     static walker<ValT,Cfg> create_walker_helper(
//                                                 const tensor_base<ValT,Cfg>& view, 
//                                                 const tensor_base<ValT,Cfg>& view_parent, 
//                                                 const tensor_base<ValT,Cfg>& view_root, 
//                                                 const value_type* cache)
//     {
//         if (detail::is_storage(view)){
//             return create_walker_helper(view, view.descriptor().shape(), view.descriptor().strides(), cache);
//         }else if(detail::is_storage(view_parent)){
//             return create_walker_helper(view, view.descriptor().shape(), view.descriptor().cstrides(), view_parent.as_storing()->data()+view.descriptor().offset());
//         }else if(view_root->tensor_kind() == detail::tensor_kinds::expression){
//             return std::unique_ptr<walker_base<ValT,Cfg>>{new viewing_evaluating_walker<ValT,Cfg>{
//                 view.descriptor().shape(),
//                 view.descriptor().cstrides(),
//                 view.descriptor().offset(), 
//                 view_parent.as_converting(), 
//                 view_root.as_evaluating()->create_storage()
//                 }
//             };
//         }else{
//             return nullptr;
//         }
//     }

    


//     template<typename F, typename...Ops, std::size_t...I>
//     static walker<ValT,Cfg> create_evaluating_walker_helper(const shape_type& shape, const F&, const std::tuple<Ops...>& operands, std::index_sequence<I...>){
//         using evaluating_walker_type = evaluating_walker_polymorphic<ValT,Cfg,F,decltype(std::declval<Ops>()->as_walker_maker()->create_walker())...>;
//         return std::unique_ptr<walker_base<ValT,Cfg>>{new evaluating_walker_type{shape,std::get<I>(operands)->as_walker_maker()->create_walker()...}};
//     }


// public:
//     template<typename...Args>
//     static walker<ValT, Cfg> create_walker(const tensor_base<ValT,Cfg>& t, Args&&...args){
//         return create_walker_helper(t, std::forward<Args>(args)...);
//     }
// };







// template<typename ValT, template<typename> typename Cfg, typename F, typename...Ops>
// class evaluating_walker_factory : public walker_factory_base<ValT,Cfg>{
//     using config_type = Cfg<ValT>;        
//     using value_type = ValT;
//     using shape_type = typename config_type::shape_type;
//     using evaluating_walker_type = evaluating_walker_polymorphic<ValT,Cfg,F,decltype(std::declval<Ops>()->create_walker())...>;

//     const shape_type* shape;
//     const F* f;
//     const std::tuple<Ops...>* operands;
    
//     template<std::size_t...I>
//     walker<ValT,Cfg> create_walker_helper(std::index_sequence<I...>)const{
//         return std::unique_ptr<walker_base<ValT,Cfg>>{new evaluating_walker_type{*shape,*f,std::get<I>(*operands)->create_walker()...}};
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
//     using trivial_ewalker_type = evaluating_trivial_walker<ValT,Cfg>;

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
//         return std::unique_ptr<walker_base<ValT,Cfg>>{new trivial_ewalker_type{*shape,*strides,*parent}};
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
//         return std::unique_ptr<walker_base<ValT,Cfg>>{new vwalker_type{*descriptor,*elements}};
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