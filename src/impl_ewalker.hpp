#ifndef IMPL_EWALKER_HPP_
#define IMPL_EWALKER_HPP_

#include "impl_walker_base.hpp"
#include "libdivide_helper.hpp"

namespace gtensor{

template<typename ValT, template<typename> typename Cfg, typename F, typename...Wks>
class evaluating_walker_impl : 
    public walker_impl_base<ValT, Cfg>,
    walker_shape<ValT,Cfg>
{
    using base_walker_shape = walker_shape<ValT,Cfg>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
        
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
    std::unique_ptr<walker_impl_base<ValT,Cfg>> clone()const override{return std::make_unique<evaluating_walker_impl<ValT,Cfg,F,Wks...>>(*this);}    
protected:
    template<std::size_t...I>
    void walk_helper(const index_type& direction, const index_type& steps, std::index_sequence<I...>){(std::get<I>(walkers).walk(direction,steps),...);}
    using base_walker_shape::dim;
public:
    evaluating_walker_impl(const shape_type& shape_, Wks&&...walkers_):
        base_walker_shape{static_cast<index_type>(shape_.size()), shape_},
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

template<typename ValT, template<typename> typename Cfg, typename F, typename...Wks>
class evaluating_storage_impl : 
    public evaluating_storage_impl_base<ValT, Cfg>,
    private evaluating_walker_impl<ValT, Cfg, F, Wks...>
{
    using base_type = evaluating_walker_impl<ValT, Cfg, F, Wks...>;
    using config_type = Cfg<ValT>;        
    using value_type = ValT;
    using index_type = typename config_type::index_type;
    using shape_type = typename config_type::shape_type;
    using strides_type = typename detail::libdiv_strides_traits<config_type>::type;
    
    const strides_type* strides;
    value_type data_cache{evaluate_at(0)};
    index_type index_cache{0};
    std::unique_ptr<evaluating_storage_impl_base<ValT,Cfg>> clone(int)const override{return std::make_unique<evaluating_storage_impl<ValT,Cfg,F,Wks...>>(*this);}
    value_type operator[](index_type idx)override{
        if (index_cache == idx){
            return data_cache;
        }else{
            return evaluate_at(idx);
        }
    }
    value_type evaluate_at(index_type idx){
        index_cache = idx;
        base_type::reset();
        auto sit_begin{(*strides).begin()};
        auto sit_end{(*strides).end()};
        for(index_type d{base_type::dim()-1};sit_begin!=sit_end; ++sit_begin,--d){
            auto q = detail::divide(idx,*sit_begin);
            if (q!=0){
                base_type::walk_helper(d,q,std::make_index_sequence<sizeof...(Wks)>{});                
            }
        }
        data_cache = base_type::operator*();
        return data_cache;
    }

public:
    evaluating_storage_impl(const shape_type& shape_, const strides_type& strides_, Wks&&...walkers_):
        base_type{shape_, std::move(walkers_)...},
        strides{&strides_}
    {}

};


}   //end of namespace gtensor


#endif