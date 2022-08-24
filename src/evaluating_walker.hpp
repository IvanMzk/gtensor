#ifndef EVALUATING_WALKER_HPP_
#define EVALUATING_WALKER_HPP_

#include "walker_base.hpp"
#include "libdivide_helper.hpp"

namespace gtensor{

template<typename ValT, typename CfgT, typename F, typename...Wks>
class evaluating_walker
{    
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    index_type dim_;
    detail::shape_inverter<index_type,shape_type> shape;
    std::tuple<Wks...> walkers;
    F f{};
        
    template<std::size_t...I>
    void step_helper(const index_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).step(direction),...);}    
    template<std::size_t...I>
    void step_back_helper(const index_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).step_back(direction),...);}    
    template<std::size_t...I>
    void reset_helper(const index_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).reset(direction),...);}
    template<std::size_t...I>
    void reset_helper(std::index_sequence<I...>){(std::get<I>(walkers).reset(),...);}    
    template<std::size_t...I>
    value_type deref_helper(std::index_sequence<I...>) const {return f(*std::get<I>(walkers)...);}
    template<std::size_t...I>
    void walk_helper(const index_type& direction, const index_type& steps, std::index_sequence<I...>){(std::get<I>(walkers).walk(direction,steps),...);}

public:

    evaluating_walker(const shape_type& shape_, Wks&&...walkers_):
        dim_{static_cast<index_type>(shape_.size())},
        shape{shape_},
        walkers{std::move(walkers_)...}
    {}

    index_type dim()const{return dim_;}

    //walk method without check to utilize in evaluating_indexer
    void walk_without_check(const index_type& direction, const index_type& steps){
        walk_helper(direction,steps,std::make_index_sequence<sizeof...(Wks)>{});        
    }    
    void walk(const index_type& direction, const index_type& steps){
        if (detail::can_walk(direction,dim_,shape.element(direction))){
            walk_helper(direction,steps,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void step(const index_type& direction){
        if (detail::can_walk(direction,dim_,shape.element(direction))){
            step_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void step_back(const index_type& direction){
        if (detail::can_walk(direction,dim_,shape.element(direction))){
            step_back_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void reset(const index_type& direction){
        if (detail::can_walk(direction,dim_,shape.element(direction))){
            reset_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void reset(){reset_helper(std::make_index_sequence<sizeof...(Wks)>{});}
    value_type operator*() const {return deref_helper(std::make_index_sequence<sizeof...(Wks)>{});}
};

template<typename ValT, typename CfgT, typename ImplT>
class evaluating_walker_polymorphic : public walker_base<ValT, CfgT>    
{
    using impl_type = ImplT;
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    impl_type impl_;

    std::unique_ptr<walker_base<ValT,CfgT>> clone()const override{return std::make_unique<evaluating_walker_polymorphic>(*this);}

public:
    evaluating_walker_polymorphic(impl_type&& impl__):
        impl_{std::move(impl__)}
    {}
    
    void walk(const index_type& direction, const index_type& steps)override{return impl_.walk(direction,steps);}
    void step(const index_type& direction)override{return impl_.step(direction);}
    void step_back(const index_type& direction)override{return impl_.step_back(direction);}
    void reset(const index_type& direction)override{return impl_.reset(direction);}
    void reset()override{return impl_.reset();}
    value_type operator*() const override{return impl_.operator*();}
};

template<typename ValT, typename CfgT, typename WlkT>
class evaluating_indexer : public indexer_base<ValT, CfgT>    
{    
    using walker_type = WlkT;    
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;
    using strides_type = typename detail::libdiv_strides_traits<CfgT>::type;
    
    walker_type walker_;
    const strides_type* strides;
    value_type data_cache{evaluate_at(0)};
    index_type index_cache{0};

    std::unique_ptr<indexer_base<ValT,CfgT>> clone(int)const override{return std::make_unique<evaluating_indexer>(*this);}

    void walk(const index_type& direction, const index_type& steps){walker_.walk_without_check(direction,steps);}
    
    value_type operator[](index_type idx)override{
        if (index_cache == idx){
            return data_cache;
        }else{
            return evaluate_at(idx);
        }
    }
    
    value_type evaluate_at(index_type idx){
        index_cache = idx;
        walker_.reset();
        auto sit_begin{(*strides).begin()};
        auto sit_end{(*strides).end()};
        for(index_type d{walker_.dim()-1};sit_begin!=sit_end; ++sit_begin,--d){
            auto q = detail::divide(idx,*sit_begin);
            if (q!=0){
                walk(d,q);
            }
        }
        data_cache = walker_.operator*();
        return data_cache;
    }

public:
    evaluating_indexer(const strides_type& strides_, walker_type&& walker__):
        walker_{std::move(walker__)},
        strides{&strides_}
    {}

};


}   //end of namespace gtensor


#endif