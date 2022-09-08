#ifndef TRIVIAL_WALKER_HPP_
#define TRIVIAL_WALKER_HPP_

#include "walker_base.hpp"

namespace gtensor{

namespace detail{
}   //end of namespace detail

template<typename ValT, typename CfgT>
class storage_trivial_walker
{       
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    const value_type* offset_;

public:    
    storage_trivial_walker(const value_type* offset__):
        offset_{offset__}
    {}    
    value_type operator[](const index_type& idx)const{return *(offset_+idx);}
};

template<typename ValT, typename CfgT, typename F, typename...Wks>
class evaluating_trivial_walker
{    
    using value_type = ValT;
    using index_type = typename CfgT::index_type;    

    std::tuple<Wks...> walkers;
    F f{};
            
    // template<typename...U>
    // auto& as_trivial(const walker<U...>& w)const{return w.as_trivial();}
    // template<typename...U>
    // auto& as_trivial(const storage_walker<U...>& w)const{return w;}
    // template<typename...U>
    // auto& as_trivial(const evaluating_trivial_walker<U...>& w)const{return w;}
    // template<typename...U>
    // auto& as_trivial(const evaluating_trivial_root_walker<U...>& w)const{return w;}
    template<typename U>
    auto& as_trivial(const U& w)const{return w;}
    
public:    
    evaluating_trivial_walker(Wks&&...walkers_):
        walkers{std::move(walkers_)...}
    {}                
    value_type operator[](const index_type& idx)const {
        return std::apply([&](const auto&...args){return f(as_trivial(args[idx])...);}, walkers);        
    }
};

template<typename ValT, typename CfgT, typename F, typename...Wks>
class evaluating_trivial_root_walker : 
    private basic_walker<ValT, CfgT, typename CfgT::index_type>,
    private evaluating_trivial_walker<ValT,CfgT,F,Wks...>
{    
    using value_type = ValT;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

public:    
    evaluating_trivial_root_walker(const shape_type& shape_, const shape_type& strides_,  Wks&&...walkers_):
        basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, index_type{0}},
        evaluating_trivial_walker{std::move(walkers_)...}
    {}
        
    void walk(const index_type& direction, const index_type& steps){basic_walker::walk(direction,steps);}
    void step(const index_type& direction){basic_walker::step(direction);}
    void step_back(const index_type& direction){basic_walker::step_back(direction);}
    void reset(const index_type& direction){basic_walker::reset(direction);}
    void reset(){basic_walker::reset();}
    value_type operator[](const index_type& idx)const {return evaluating_trivial_walker::operator[](idx);}
    value_type operator*() const {return operator[](cursor());}
};



// template<typename ValT, typename CfgT, typename ImplT>
// class evaluating_walker_polymorphic : public walker_base<ValT, CfgT>    
// {
//     using impl_type = ImplT;
//     using value_type = ValT;
//     using index_type = typename CfgT::index_type;
//     using shape_type = typename CfgT::shape_type;

//     impl_type impl_;

//     std::unique_ptr<walker_base<ValT,CfgT>> clone()const override{return std::make_unique<evaluating_walker_polymorphic>(*this);}

// public:
//     evaluating_walker_polymorphic(impl_type&& impl__):
//         impl_{std::move(impl__)}
//     {}
    
//     void walk(const index_type& direction, const index_type& steps)override{return impl_.walk(direction,steps);}
//     void step(const index_type& direction)override{return impl_.step(direction);}
//     void step_back(const index_type& direction)override{return impl_.step_back(direction);}
//     void reset(const index_type& direction)override{return impl_.reset(direction);}
//     void reset()override{return impl_.reset();}
//     value_type operator*() const override{return impl_.operator*();}
// };



// template<typename ValT, typename CfgT, typename F, typename...Wks>
// class evaluating_trivial_walker_polymorphic : public walker_trivial_base<ValT, CfgT>    
// {    
//     using value_type = ValT;
//     using index_type = typename CfgT::index_type;
//     using shape_type = typename CfgT::shape_type;

//     std::tuple<Wks...> walkers;
//     F f{};
        
//     std::unique_ptr<walker_base<ValT,CfgT>> clone()const override{return std::make_unique<evaluating_trivial_walker>(*this);}
//     template<typename...U>
//     auto& as_trivial(const walker<U...>& w){return w.as_trivial();}
//     template<typename...U>
//     auto& as_trivial(const storage_walker<U...>& w){return w;}

//     template<std::size_t...I>
//     value_type subscription_helper(const index_type& idx, std::index_sequence<I...>) const {return f(as_trivial(std::get<I>(walkers))[idx]...);}

// public:    
//     evaluating_trivial_walker(const shape_type& shape_, const shape_type& strides_,  Wks&&...walkers_):
//         basic_walker{static_cast<index_type>(shape_.size()), shape_, strides_, index_type{0}},
//         walkers{std::move(walkers_)}
//     {}
        
//     void walk(const index_type& direction, const index_type& steps)override{basic_walker::walk(direction,steps);}
//     void step(const index_type& direction)override{basic_walker::step(direction);}
//     void step_back(const index_type& direction)override{basic_walker::step_back(direction);}
//     void reset(const index_type& direction)override{basic_walker::reset(direction);}
//     void reset()override{basic_walker::reset();}
//     value_type operator[](const index_type& idx)const override{return subscription_helper(idx, std::make_index_sequence<sizeof...(Wks)>{});}
//     value_type operator*() const override{return operator[](cursor());}
// };




}   //end of namespace gtensor

#endif