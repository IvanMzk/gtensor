#ifndef EVALUATING_WALKER_HPP_
#define EVALUATING_WALKER_HPP_

#include "libdivide_helper.hpp"
#include "broadcast.hpp"

namespace gtensor{

template<typename CfgT, typename F, typename...Walkers>
class evaluating_walker
{
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    index_type dim_;
    detail::shape_inverter<index_type,shape_type> shape;
    std::tuple<Walkers...> walkers;
    F f{};

    template<std::size_t...I>
    void step_helper(const index_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).step(direction),...);}
    template<std::size_t...I>
    void step_back_helper(const index_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).step_back(direction),...);}
    template<std::size_t...I>
    void reset_helper(const index_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).reset(direction),...);}
    template<std::size_t...I>
    void reset_back_helper(const index_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).reset_back(direction),...);}
    template<std::size_t...I>
    void reset_helper(std::index_sequence<I...>){(std::get<I>(walkers).reset(),...);}
    template<std::size_t...I>
    auto deref_helper(std::index_sequence<I...>) const {return f(*std::get<I>(walkers)...);}
    template<std::size_t...I>
    void walk_helper(const index_type& direction, const index_type& steps, std::index_sequence<I...>){(std::get<I>(walkers).walk(direction,steps),...);}

public:

    evaluating_walker(const shape_type& shape_, Walkers&&...walkers_):
        dim_{static_cast<index_type>(shape_.size())},
        shape{shape_},
        walkers{std::move(walkers_)...}
    {}

    index_type dim()const{return dim_;}

    //walk method without check to utilize in evaluating_indexer
    void walk_without_check(const index_type& direction, const index_type& steps){
        walk_helper(direction,steps,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void walk(const index_type& direction, const index_type& steps){
        walk_helper(direction,steps,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void step(const index_type& direction){
        step_helper(direction,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void step_back(const index_type& direction){
        step_back_helper(direction,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void reset(const index_type& direction){
        reset_helper(direction,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void reset_back(const index_type& direction){
        reset_back_helper(direction,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void reset(){reset_helper(std::make_index_sequence<sizeof...(Walkers)>{});}
    auto operator*() const {return deref_helper(std::make_index_sequence<sizeof...(Walkers)>{});}
};

template<typename CfgT, typename Walker>
class evaluating_indexer
{
    using walker_type = Walker;
    using value_type = decltype(std::declval<walker_type>().operator*());
    using index_type = typename CfgT::index_type;
    using strides_div_type = typename detail::strides_div_traits<CfgT>::type;

    const strides_div_type* strides;
    mutable walker_type walker_;
    mutable value_type data_cache{evaluate_at(0)};
    mutable index_type index_cache{0};

    void walk(const index_type& direction, const index_type& steps)const{
        walker_.walk_without_check(direction,steps);
    }
    auto evaluate_at(index_type idx)const{
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
    evaluating_indexer(const strides_div_type& strides_, walker_type&& walker__):
        strides{&strides_},
        walker_{std::move(walker__)}
    {}
    auto operator[](index_type idx)const{
        if (index_cache == idx){
            return data_cache;
        }else{
            return evaluate_at(idx);
        }
    }
};

template<typename CfgT, typename F, typename...Indexers>
class evaluating_trivial_indexer
{
    using index_type = typename CfgT::index_type;

    std::tuple<Indexers...> indexers;
    F f{};
public:
    evaluating_trivial_indexer(Indexers&&...indexers_):
        indexers{std::move(indexers_)...}
    {}
    auto operator[](const index_type& idx)const {
        return std::apply([&](const auto&...args){return f(args[idx]...);}, indexers);
    }
};

}   //end of namespace gtensor


#endif