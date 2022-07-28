#ifndef IMPL_EWALKER_HPP_
#define IMPL_EWALKER_HPP_

#include "impl_walker_base.hpp"

namespace gtensor{

template<typename ValT, template<typename> typename Cfg, typename F, typename...Wks>
class evaluating_walker_impl : public walker_impl_base<ValT, Cfg>{
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    
    const shape_type* shape;
    std::tuple<Wks...> walkers;    
    index_type dim{static_cast<index_type>(shape->size())};
    F f{};


    auto shape_element(const index_type& direction)const{return (*shape)[direction];}
    bool can_walk(const index_type& direction)const{return direction < dim && shape_element(direction) != index_type(1);}
    template<std::size_t...I>
    void walk_helper(const index_type& direction, const index_type& steps, std::index_sequence<I...>){(std::get<I>(walkers).walk(direction,steps),...);}
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
    std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const override{return std::make_unique<evaluating_walker_impl<ValT,Cfg,F,Wks...>>(*this);}
public:
    evaluating_walker_impl(const shape_type& shape_, Wks&&...walkers_):
        shape{&shape_},
        walkers{std::move(walkers_)...}
    {}
    
    void walk(const index_type& direction, const index_type& steps)override{
        if (can_walk(direction)){
            walk_helper(direction,steps,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void step(const index_type& direction)override{
        if (can_walk(direction)){
            step_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void step_back(const index_type& direction)override{
        if (can_walk(direction)){
            step_back_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void reset(const index_type& direction)override{
        if (can_walk(direction)){
            reset_helper(direction,std::make_index_sequence<sizeof...(Wks)>{});
        }
    }
    void reset()override{reset_helper(std::make_index_sequence<sizeof...(Wks)>{});}
    value_type operator*() const override{return deref_helper(std::make_index_sequence<sizeof...(Wks)>{});}
};


}   //end of namespace gtensor


#endif