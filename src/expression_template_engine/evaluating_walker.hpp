#ifndef EVALUATING_WALKER_HPP_
#define EVALUATING_WALKER_HPP_

#include "libdivide_helper.hpp"
#include "broadcast.hpp"

namespace gtensor{

template<typename CfgT, typename F, typename...Walkers>
class evaluating_walker
{
    using size_type = typename CfgT::size_type;
    using index_type = typename CfgT::index_type;
    using shape_type = typename CfgT::shape_type;

    size_type dim_;
    detail::shape_inverter<CfgT> shape;
    F f;
    std::tuple<Walkers...> walkers;

    template<std::size_t...I>
    void step_helper(const size_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).step(direction),...);}
    template<std::size_t...I>
    void step_back_helper(const size_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).step_back(direction),...);}
    template<std::size_t...I>
    void reset_helper(const size_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).reset(direction),...);}
    template<std::size_t...I>
    void reset_back_helper(const size_type& direction, std::index_sequence<I...>){(std::get<I>(walkers).reset_back(direction),...);}
    template<std::size_t...I>
    void reset_helper(std::index_sequence<I...>){(std::get<I>(walkers).reset(),...);}
    template<std::size_t...I>
    auto deref_helper(std::index_sequence<I...>) const {return f(*std::get<I>(walkers)...);}
    template<std::size_t...I>
    void walk_helper(const size_type& direction, const index_type& steps, std::index_sequence<I...>){(std::get<I>(walkers).walk(direction,steps),...);}

public:

    evaluating_walker(const shape_type& shape_, const F& f_, Walkers&&...walkers_):
        dim_{static_cast<size_type>(shape_.size())},
        shape{shape_},
        f{f_},
        walkers{std::move(walkers_)...}
    {}

    size_type dim()const{return dim_;}

    //walk method without check to utilize in evaluating_indexer
    void walk_without_check(const size_type& direction, const index_type& steps){
        walk_helper(direction,steps,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void walk(const size_type& direction, const index_type& steps){
        walk_helper(direction,steps,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void step(const size_type& direction){
        step_helper(direction,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void step_back(const size_type& direction){
        step_back_helper(direction,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void reset(const size_type& direction){
        reset_helper(direction,std::make_index_sequence<sizeof...(Walkers)>{});
    }
    void reset_back(const size_type& direction){
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
    using size_type = typename CfgT::size_type;
    using strides_div_type = typename detail::strides_div_traits<CfgT>::type;

    const strides_div_type* strides;
    mutable walker_type walker_;
    mutable value_type data_cache{evaluate_at(0)};
    mutable index_type index_cache{0};

    void walk(const size_type& direction, const index_type& steps)const{
        walker_.walk_without_check(direction,steps);
    }
    auto evaluate_at(index_type idx)const{
        index_cache = idx;
        walker_.reset();
        auto sit_begin{(*strides).begin()};
        auto sit_end{(*strides).end()};
        for(size_type direction{walker_.dim()-size_type(1)}; sit_begin!=sit_end; ++sit_begin,--direction){
            index_type steps = detail::divide(idx,*sit_begin);
            if (steps!=index_type{0}){
                walk(direction,steps);
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

    F f;
    std::tuple<Indexers...> indexers;
public:
    evaluating_trivial_indexer(const F& f_, Indexers&&...indexers_):
        f{f_},
        indexers{std::move(indexers_)...}
    {}
    auto operator[](const index_type& idx)const {
        return std::apply([&](const auto&...args){return f(args[idx]...);}, indexers);
    }
};

}   //end of namespace gtensor


#endif