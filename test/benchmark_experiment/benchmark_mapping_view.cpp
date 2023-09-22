#include <algorithm>
#include <numeric>
#include <random>
#include "../benchmark_helpers.hpp"
#include "../helpers_for_testing.hpp"
#include "../test_config.hpp"
#include "tensor.hpp"


namespace benchmark_mapping_view{

template<typename T>
struct test_tensor : public T{
    test_tensor(const T& base):
        T{base}
    {}
    using T::descriptor;
    using T::engine;
    using T::impl;
};

template<template<typename> typename TestT = test_tensor, typename T>
auto make_test_tensor(T&& t){return TestT<std::decay_t<T>>{t};}

template<typename ShT>
auto check_bool_mapping_view_subs(const ShT& pshape, const ShT& subs_shape){
    using dim_type = typename ShT::difference_type;
    dim_type pdim = pshape.size();
    dim_type subs_dim = subs_shape.size();
    if (subs_dim > pdim){
        throw gtensor::subscript_exception("invalid bool tensor subscript");
    }
    for (auto subs_shape_it = subs_shape.begin(), pshape_it = pshape.begin(); subs_shape_it!=subs_shape.end(); ++subs_shape_it, ++pshape_it){
        if (*subs_shape_it > *pshape_it){
            throw gtensor::subscript_exception("invalid bool tensor subscript");
        }
    }
}

template<typename ShT, typename SizeT>
inline auto mapping_view_block_size(const ShT& pshape, const SizeT& subs_dim_or_subs_number){
    using index_type = typename ShT::value_type;
    return std::accumulate(pshape.begin()+subs_dim_or_subs_number,pshape.end(),index_type(1),std::multiplies<index_type>{});
}

namespace subs_iterate_once{

template<typename ShT, typename SizeT>
inline ShT make_bool_mapping_view_shape(const ShT& pshape, const typename ShT::value_type& subs_trues_number, const SizeT& subs_dim){
    using shape_type = ShT;
    using dim_type = SizeT;
    dim_type pdim = pshape.size();
    auto res = shape_type(pdim - subs_dim + dim_type{1});
    auto res_it = res.begin();
    *res_it = subs_trues_number;
    ++res_it;
    std::copy(pshape.begin()+subs_dim, pshape.end(), res_it);
    return res;
}
template<typename ShT, typename ParentIndexer, typename ResIt, typename Subs>
auto fill_bool_mapping_view(const ShT& pshape, const ShT& pstrides, ParentIndexer pindexer, ResIt res_it, const Subs& subs){
    using config_type = typename Subs::config_type;
    using index_type = typename ShT::value_type;
    using dim_type = typename ShT::difference_type;

    dim_type subs_dim = subs.dim();
    index_type block_size = mapping_view_block_size(pshape, subs_dim);
    index_type trues_number{0};
    gtensor::walker_forward_adapter<config_type, decltype(subs.engine().create_walker())> subs_it{subs.descriptor().shape(), subs.engine().create_walker()};

    if (block_size == index_type{1}){
        do{
            if(*subs_it.walker()){
                ++trues_number;
                index_type pindex = std::inner_product(subs_it.index().begin(), subs_it.index().end(), pstrides.begin(), index_type{0});
                *res_it = pindexer[pindex];
                ++res_it;
            }
        }while(subs_it.next());
    }else{
        do{
            if(*subs_it.walker()){
                ++trues_number;
                auto block_first = std::inner_product(subs_it.index().begin(), subs_it.index().end(), pstrides.begin(), index_type{0});
                for(index_type i{0}; i!=block_size; ++i){
                    *res_it = pindexer[block_first+i];
                    ++res_it;
                }
            }
        }while(subs_it.next());
    }
    return trues_number;
}
template<typename Parent, typename Subs>
auto make_bool_mapping_view(const Parent& parent, const Subs& subs){
    using config_type = typename Parent::config_type;
    using value_type = typename Parent::value_type;
    const auto& pshape = parent.shape();
    const auto& subs_shape = subs.shape();
    check_bool_mapping_view_subs(pshape, subs_shape);
    auto res = make_test_tensor(gtensor::storage_tensor_factory<config_type,value_type>::make(pshape, value_type{}));
        //dim_type subs_dim = subs.dim();
    auto subs_trues_number = fill_bool_mapping_view(
        pshape,
        parent.descriptor().strides(),
        parent.engine().create_indexer(),
        res.begin(),
        subs
    );
    res.impl()->resize(make_bool_mapping_view_shape(pshape, subs_trues_number, subs.dim()));
    return res;
}

}   //end of namespace subs_iterate_twice

namespace subs_iterate_twice{

template<typename ShT, typename Subs>
inline ShT make_bool_mapping_view_shape(const ShT& pshape, const Subs& subs){
    using shape_type = ShT;
    using index_type = typename shape_type::value_type;
    using dim_type = typename shape_type::size_type;
    dim_type pdim = pshape.size();
    dim_type subs_dim = subs.dim();
    index_type subs_trues_number = std::count(subs.begin(),subs.end(),true);
    auto res = shape_type(pdim - subs_dim + dim_type{1});
    auto res_it = res.begin();
    *res_it = subs_trues_number;
    ++res_it;
    std::copy(pshape.begin()+subs_dim, pshape.end(), res_it);
    return res;
}
template<typename ShT, typename ParentIndexer, typename ResIt, typename Subs>
auto fill_bool_mapping_view(const ShT& pshape, const ShT& pstrides, ParentIndexer pindexer, ResIt res_it, const Subs& subs){
    using config_type = typename Subs::config_type;
    using index_type = typename ShT::value_type;
    using dim_type = typename ShT::difference_type;

    dim_type subs_dim = subs.dim();
    index_type block_size = mapping_view_block_size(pshape, subs_dim);
    gtensor::walker_forward_adapter<config_type, decltype(subs.engine().create_walker())> subs_it{subs.descriptor().shape(), subs.engine().create_walker()};
    if (block_size == index_type{1}){
        do{
            if(*subs_it.walker()){
                index_type pindex = std::inner_product(subs_it.index().begin(), subs_it.index().end(), pstrides.begin(), index_type{0});
                *res_it = pindexer[pindex];
                ++res_it;
            }
        }while(subs_it.next());
    }else{
        do{
            if(*subs_it.walker()){
                auto block_first = std::inner_product(subs_it.index().begin(), subs_it.index().end(), pstrides.begin(), index_type{0});
                for(index_type i{0}; i!=block_size; ++i){
                    *res_it = pindexer[block_first+i];
                    ++res_it;
                }
            }
        }while(subs_it.next());
    }
}
template<typename Parent, typename Subs>
auto make_bool_mapping_view(const Parent& parent, const Subs& subs){
    using config_type = typename Parent::config_type;
    using value_type = typename Parent::value_type;
    const auto& pshape = parent.shape();
    const auto& subs_shape = subs.shape();
    check_bool_mapping_view_subs(pshape, subs_shape);
    auto res = make_test_tensor(gtensor::storage_tensor_factory<config_type,value_type>::make(make_bool_mapping_view_shape(pshape, subs), value_type{}));
    fill_bool_mapping_view(
        pshape,
        parent.descriptor().strides(),
        parent.engine().create_indexer(),
        res.begin(),
        subs
    );
    return res;
}

}   //end of subs_iterate_twice

}   //end of namespace benchmark_mapping_view


TEMPLATE_TEST_CASE("test_make_bool_mapping_view","[benchmark_mapping_view]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using value_type = int;
    using index_tensor_type = gtensor::tensor<bool, config_type>;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using benchmark_mapping_view::make_test_tensor;
    using helpers_for_testing::apply_by_element;

    //0parent,1subs,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{0},index_tensor_type{false},tensor_type{}),
        std::make_tuple(tensor_type{0},index_tensor_type{true},tensor_type{0}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},index_tensor_type{false,false,true,false,true,true},tensor_type{3,5,6}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,2,3,4,5,6,7}, tensor_type{1,2,3,4,5,6,2,3,4,5,6,7} < tensor_type{5} ,tensor_type{1,2,3,4,2,3,4}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},index_tensor_type{{false,false,true},{true,false,true}},tensor_type{3,4,6}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}},tensor_type{{1,2,3},{4,5,6}} > tensor_type{3},tensor_type{4,5,6})
    );
    SECTION("subs_iterate_once")
    {
        using benchmark_mapping_view::subs_iterate_once::make_bool_mapping_view;
        auto test = [](const auto& t){
            auto parent = make_test_tensor(std::get<0>(t));
            auto subs = make_test_tensor(std::get<1>(t));
            auto expected = std::get<2>(t);
            auto result = make_bool_mapping_view(parent,subs);
            REQUIRE(result.equals(expected));
        };
        apply_by_element(test,test_data);
    }
    SECTION("subs_iterate_twice")
    {
        using benchmark_mapping_view::subs_iterate_twice::make_bool_mapping_view;
        auto test = [](const auto& t){
            auto parent = make_test_tensor(std::get<0>(t));
            auto subs = make_test_tensor(std::get<1>(t));
            auto expected = std::get<2>(t);
            auto result = make_bool_mapping_view(parent,subs);
            REQUIRE(result.equals(expected));
        };
        apply_by_element(test,test_data);
    }
}

TEMPLATE_TEST_CASE("benchamrk_make_bool_mapping_view","[benchmark_mapping_view]",
    typename test_config::config_host_engine_selector<gtensor::config::engine_expression_template>::config_type
)
{
    using config_type = TestType;
    using value_type = int;
    using gtensor::tensor;
    using shape_type = typename config_type::shape_type;
    using index_type = typename config_type::index_type;
    using tensor_type = gtensor::tensor<value_type, config_type>;
    using benchmark_mapping_view::make_test_tensor;
    using benchmark_helpers::benchmark;

    auto generate_uniform = [](auto first, auto last, const auto& min, const auto& max){
        using value_type = typename std::iterator_traits<decltype(first)>::value_type;
        std::mt19937_64 gen{std::random_device{}()};
        if constexpr(std::is_integral_v<value_type>){
            std::uniform_int_distribution<value_type> distr{min,max};
            for (;first!=last; ++first){
                *first = distr(gen);
            }
        }else if constexpr(std::is_floating_point_v<value_type>){
            std::uniform_real_distribution<value_type> distr{min,max};
            for (;first!=last; ++first){
                *first = distr(gen);
            }
        }else{
            //exception
        }
    };


    SECTION("test_benchmark")
    {
        const shape_type pshape{3,3};
        tensor_type t{pshape, value_type{}};
        std::iota(t.begin(),t.end(),value_type{0});
        REQUIRE(t.equals(tensor_type{{0,1,2},{3,4,5},{6,7,8}}));
        value_type max = static_cast<value_type>(t.size()/index_type{2});
        auto subs = t < tensor_type{max};
        tensor_type expected{0,1,2,3};
        SECTION("test_make_bool_mapping_view_iter_once"){
            using benchmark_mapping_view::subs_iterate_once::make_bool_mapping_view;
            auto result = make_bool_mapping_view(make_test_tensor(t), make_test_tensor(subs));
            REQUIRE(result.equals(expected));
        }
        SECTION("test_make_bool_mapping_view_iter_twice"){
            using benchmark_mapping_view::subs_iterate_twice::make_bool_mapping_view;
            auto result = make_bool_mapping_view(make_test_tensor(t), make_test_tensor(subs));
            REQUIRE(result.equals(expected));
        }

        // tensor_type tt({3,3},value_type{});
        // generate_uniform(tt.begin(), tt.end(), value_type{0}, value_type{1});
        // std::cout<<std::endl<<tt;
        // tensor<int> ttt({3,3},int{});
        // generate_uniform(ttt.begin(), ttt.end(), int{-3}, int{3});
        // std::cout<<std::endl<<ttt;
    }

    SECTION("benchmark")
    {
        const shape_type pshape{5000,5000};
        tensor_type t{pshape, value_type{}};

        value_type min = -10;
        value_type max = 10;
        generate_uniform(t.begin(),t.end(),min,max);

        value_type select_range_min = -5;
        value_type select_range_max = 5;
        //auto subs = t>tensor_type{select_range_min} && t<tensor_type{select_range_max};
        auto subs = (t>tensor_type{select_range_min} && t<tensor_type{select_range_max}).copy();

        SECTION("benchmark_make_bool_mapping_view_iter_once"){
            auto benchmark_f = [](const auto& parent_, const auto& subs_){
                return  benchmark_mapping_view::subs_iterate_once::make_bool_mapping_view(make_test_tensor(parent_), make_test_tensor(subs_));
            };
            benchmark("benchmark_make_bool_mapping_view_iter_once", benchmark_f, t, subs);
        }
        // SECTION("benchmark_make_bool_mapping_view_iter_once_copy_subs"){
        //     auto benchmark_f = [](const auto& parent_, const auto& subs_){
        //         auto subs_copy_ = subs_.copy();
        //         return  benchmark_mapping_view::subs_iterate_once::make_bool_mapping_view(make_test_tensor(parent_), make_test_tensor(subs_copy_));
        //     };
        //     benchmark("benchmark_make_bool_mapping_view_iter_once_copy_subs", benchmark_f, t, subs);
        // }
        SECTION("benchmark_make_bool_mapping_view_iter_twice"){
            auto benchmark_f = [](const auto& parent_, const auto& subs_){
                return benchmark_mapping_view::subs_iterate_twice::make_bool_mapping_view(make_test_tensor(parent_), make_test_tensor(subs_));
            };
            benchmark("benchmark_make_bool_mapping_view_iter_twice", benchmark_f, t, subs);
        }
        SECTION("benchmark_make_bool_mapping_view_iter_twice_copy_subs"){
            auto benchmark_f = [](const auto& parent_, const auto& subs_){
                auto subs_copy_ = subs_.copy();
                return benchmark_mapping_view::subs_iterate_twice::make_bool_mapping_view(make_test_tensor(parent_), make_test_tensor(subs_copy_));
            };
            benchmark("benchmark_make_bool_mapping_view_iter_twice_copy_subs", benchmark_f, t, subs);
        }
    }

}
