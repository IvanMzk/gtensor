#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"
#include "sort_search.hpp"
#include "tensor.hpp"

namespace test_partition_{

using gtensor::basic_tensor;

template<typename It, typename Comparator>
bool is_nth_partition_point(It first, const It last, const It point, Comparator comparator){
    const auto& point_element = *point;
    for (;first!=point; ++first){
        const auto& e = *first;
        if (e!=point_element && !comparator(e,point_element)){
            return false;
        }
    }
    for (;first!=last; ++first){
        const auto& e = *first;
        if (e!=point_element && comparator(e,point_element)){
            return false;
        }
    }
    return true;
}

template<typename...Ts, typename DimT, typename IdxT, typename Comparator>
bool is_partitioned_nth_scalar(const basic_tensor<Ts...>& t, const DimT& axis, const IdxT& nth_, Comparator comparator){
    using order = typename basic_tensor<Ts...>::order;
    using index_type = typename basic_tensor<Ts...>::index_type;
    using config_type = typename basic_tensor<Ts...>::config_type;
    if (t.empty()){
        return true;
    }
    auto axes_iterator_maker = gtensor::detail::make_axes_iterator_maker<config_type>(t.shape(),axis,order{});
    auto traverser = axes_iterator_maker.create_forward_traverser(t.create_walker(),std::true_type{});
    const index_type nth = static_cast<const index_type&>(nth_);
    bool res{true};
    do{
        res = res && is_nth_partition_point(
            axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{}),
            axes_iterator_maker.end_complement(traverser.walker(),std::false_type{}),
            axes_iterator_maker.begin_complement(traverser.walker(),std::false_type{})+nth,
            comparator
        );
    }while(traverser.template next<order>());
    return res;
}

template<typename...Ts, typename DimT, typename Nth, typename Comparator>
bool is_partitioned(const basic_tensor<Ts...>& t, const DimT& axis, const Nth& nth, Comparator comparator){
    if constexpr (gtensor::detail::is_container_v<Nth>){
        bool res{true};
        for (auto it=nth.begin(),last=nth.end(); it!=last; ++it){
            res = res && is_partitioned_nth_scalar(t,axis,*it,comparator);
        }
        return res;
    }else{
        return is_partitioned_nth_scalar(t,axis,nth,comparator);
    }
}

}   //end of namespace test_partition_

//test helpers
TEST_CASE("test_is_nth_partition_point","[test_sort_search]")
{
    using value_type = double;
    using test_partition_::is_nth_partition_point;
    using helpers_for_testing::apply_by_element;
    //0elements,1nth_index,2comparator,3expected
    auto test_data = std::make_tuple(
        //less
        //true
        std::make_tuple(std::vector<value_type>{},0,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{2},0,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{1,2,3,4,5},0,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{1,2,3,4,5},1,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{1,2,3,4,5},2,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{1,2,3,4,5},3,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{1,2,3,4,5},4,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{0,1,7,7,1,5,7,2,3,2,6,2,3,7},0,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{0,1,1,2,2,5,2,3,3,6,7,7,7,7},3,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{0,1,2,1,2,5,2,3,3,6,7,7,7,7},4,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{0,1,2,1,2,2,3,3,5,6,7,7,7,7},5,std::less<void>{},true),
        std::make_tuple(std::vector<value_type>{0,1,2,1,2,3,3,2,5,6,7,7,7,7},8,std::less<void>{},true),
        //false
        std::make_tuple(std::vector<value_type>{1,0,7,7,1,5,7,2,3,2,6,2,3,7},0,std::less<void>{},false),
        std::make_tuple(std::vector<value_type>{0,1,3,2,2,5,2,1,3,6,7,7,7,7},3,std::less<void>{},false),
        std::make_tuple(std::vector<value_type>{0,2,2,1,2,5,1,3,3,6,7,7,7,7},4,std::less<void>{},false),
        std::make_tuple(std::vector<value_type>{0,7,2,1,2,3,3,2,5,6,7,1,7,7},8,std::less<void>{},false),
        //greater
        //true
        std::make_tuple(std::vector<value_type>{2},0,std::greater<void>{},true),
        std::make_tuple(std::vector<value_type>{5,4,3,2,1},0,std::greater<void>{},true),
        std::make_tuple(std::vector<value_type>{5,4,1,2,3},1,std::greater<void>{},true),
        std::make_tuple(std::vector<value_type>{4,5,3,1,2},2,std::greater<void>{},true),
        std::make_tuple(std::vector<value_type>{5,3,4,2,1},3,std::greater<void>{},true),
        std::make_tuple(std::vector<value_type>{5,4,3,2,1},4,std::greater<void>{},true),
        std::make_tuple(std::vector<value_type>{7,1,0,7,1,5,7,2,3,2,6,2,3,7},0,std::greater<void>{},true),
        std::make_tuple(std::vector<value_type>{7,7,7,7,1,5,0,2,3,2,6,2,3,1},3,std::greater<void>{},true),
        std::make_tuple(std::vector<value_type>{7,7,7,7,6,5,0,2,3,2,1,2,3,1},5,std::greater<void>{},true),
        std::make_tuple(std::vector<value_type>{7,7,7,7,6,5,3,3,2,0,1,2,2,1},8,std::greater<void>{},true),
        //false
        std::make_tuple(std::vector<value_type>{6,1,0,7,1,5,7,2,3,2,7,2,3,7},0,std::greater<void>{},false),
        std::make_tuple(std::vector<value_type>{7,7,2,7,1,5,0,7,3,2,6,2,3,1},3,std::greater<void>{},false),
        std::make_tuple(std::vector<value_type>{7,7,7,7,5,6,0,2,3,2,1,2,3,1},5,std::greater<void>{},false),
        std::make_tuple(std::vector<value_type>{7,7,7,7,0,5,3,3,2,6,1,2,2,1},8,std::greater<void>{},false)
    );
    auto test = [](const auto& t){
        auto elements = std::get<0>(t);
        auto nth_index = std::get<1>(t);
        auto comparator = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = is_nth_partition_point(elements.begin(),elements.end(),elements.begin()+nth_index,comparator);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_is_partitioned_nth_scalar","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using test_partition_::is_partitioned_nth_scalar;
    using helpers_for_testing::apply_by_element;
    //0ten,1axis,2nth,3comparator,4expected
    auto test_data = std::make_tuple(
        //less
        //true
        std::make_tuple(tensor_type{},0,0,std::less<void>{},true),
        std::make_tuple(tensor_type{1},0,0,std::less<void>{},true),
        std::make_tuple(tensor_type{5},0,0,std::less<void>{},true),
        std::make_tuple(tensor_type{1,2,3,4,5},0,0,std::less<void>{},true),
        std::make_tuple(tensor_type{1,2,3,4,5},0,2,std::less<void>{},true),
        std::make_tuple(tensor_type{1,2,3,4,5},0,4,std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,6,3,2,1,2,5},0,0,std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,6,3,2,1,2,5},0,1,std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,1,3,2,6,2,5},0,2,std::less<void>{},true),
        std::make_tuple(tensor_type{1,0,1,2,2,3,6,5},0,3,std::less<void>{},true),
        std::make_tuple(tensor_type{1,0,1,2,2,3,6,5},0,4,std::less<void>{},true),
        std::make_tuple(tensor_type{1,2,0,1,2,3,6,5},0,5,std::less<void>{},true),
        std::make_tuple(tensor_type{1,2,0,1,2,3,5,6},0,6,std::less<void>{},true),
        std::make_tuple(tensor_type{1,2,0,1,2,3,5,6},0,7,std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,2,1,2,5,2,3,3,6,7,7,7,7},0,4,std::less<void>{},true),
        std::make_tuple(tensor_type{{-1,1,-1,0,2},{8,2,1,6,5},{2,7,0,4,3},{4,4,2,1,4},{3,1,6,4,3}},0,0,std::less<void>{},true),
        std::make_tuple(tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,6,4},{8,7,6,4,5}},0,2,std::less<void>{},true),
        std::make_tuple(tensor_type{{-1,1,-1,1,2},{2,1,0,4,3},{3,2,1,0,3},{4,4,2,4,4},{8,7,6,6,5}},0,3,std::less<void>{},true),
        std::make_tuple(tensor_type{{-1,1,-1,1,2},{2,1,0,4,3},{3,2,1,0,3},{4,4,2,4,4},{8,7,6,6,5}},0,4,std::less<void>{},true),
        std::make_tuple(tensor_type{{-1,1,2,6,3},{0,2,1,8,5},{-1,7,0,4,2},{1,4,2,4,4},{1,3,6,4,3}},1,0,std::less<void>{},true),
        std::make_tuple(tensor_type{{-1,1,2,6,3},{0,1,2,8,5},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}},1,2,std::less<void>{},true),
        std::make_tuple(tensor_type{{1,-1,2,3,6},{0,1,2,5,8},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}},1,3,std::less<void>{},true),
        std::make_tuple(tensor_type{{1,-1,2,3,6},{0,1,2,5,8},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}},1,4,std::less<void>{},true),
        //false
        std::make_tuple(tensor_type{1,0,7,7,1,5,7,2,3,2,6,2,3,7},0,0,std::less<void>{},false),
        std::make_tuple(tensor_type{0,1,3,2,2,5,2,1,3,6,7,7,7,7},0,3,std::less<void>{},false),
        std::make_tuple(tensor_type{0,2,2,1,2,5,1,3,3,6,7,7,7,7},0,4,std::less<void>{},false),
        std::make_tuple(tensor_type{0,7,2,1,2,3,3,2,5,6,7,1,7,7},0,8,std::less<void>{},false),
        std::make_tuple(tensor_type{{3,1,-1,0,2},{8,2,1,6,5},{2,7,0,4,3},{4,4,2,1,4},{-1,1,6,4,3}},0,0,std::less<void>{},false),
        std::make_tuple(tensor_type{{-1,1,-1,0,2},{2,1,1,1,3},{3,2,0,4,3},{4,4,2,6,4},{8,7,6,4,5}},0,2,std::less<void>{},false),
        std::make_tuple(tensor_type{{-1,1,-1,1,2},{2,1,0,4,3},{3,2,1,0,5},{4,4,2,4,4},{8,7,6,6,3}},0,3,std::less<void>{},false)
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto nth = std::get<2>(t);
        auto comparator = std::get<3>(t);
        auto expected = std::get<4>(t);
        auto result = is_partitioned_nth_scalar(ten,axis,nth,comparator);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_is_partitioned","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using test_partition_::is_partitioned;
    using helpers_for_testing::apply_by_element;
    //0ten,1axis,2nth,3comparator,4expected
    auto test_data = std::make_tuple(
        //nth scalar
        std::make_tuple(tensor_type{},0,0,std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,6,3,2,1,2,5},0,0,std::less<void>{},true),
        std::make_tuple(tensor_type{1,0,1,2,2,3,6,5},0,4,std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,2,1,2,5,2,3,3,6,7,7,7,7},0,4,std::less<void>{},true),
        std::make_tuple(tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,6,4},{8,7,6,4,5}},0,2,std::less<void>{},true),
        std::make_tuple(tensor_type{{-1,1,2,6,3},{0,1,2,8,5},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}},1,2,std::less<void>{},true),
        //false
        std::make_tuple(tensor_type{1,0,7,7,1,5,7,2,3,2,6,2,3,7},0,0,std::less<void>{},false),
        std::make_tuple(tensor_type{0,1,3,2,2,5,2,1,3,6,7,7,7,7},0,3,std::less<void>{},false),
        std::make_tuple(tensor_type{0,2,2,1,2,5,1,3,3,6,7,7,7,7},0,4,std::less<void>{},false),
        std::make_tuple(tensor_type{0,7,2,1,2,3,3,2,5,6,7,1,7,7},0,8,std::less<void>{},false),
        std::make_tuple(tensor_type{{3,1,-1,0,2},{8,2,1,6,5},{2,7,0,4,3},{4,4,2,1,4},{-1,1,6,4,3}},0,0,std::less<void>{},false),
        std::make_tuple(tensor_type{{-1,1,-1,0,2},{2,1,1,1,3},{3,2,0,4,3},{4,4,2,6,4},{8,7,6,4,5}},0,2,std::less<void>{},false),
        std::make_tuple(tensor_type{{-1,1,-1,1,2},{2,1,0,4,3},{3,2,1,0,5},{4,4,2,4,4},{8,7,6,6,3}},0,3,std::less<void>{},false),
        //nth container
        std::make_tuple(tensor_type{},0,std::vector<int>{4},std::less<void>{},true),
        std::make_tuple(tensor_type{},0,std::vector<int>{2,8},std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,2,1,2,5,2,3,3,6,7,7,7,7},0,std::vector<int>{4},std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,1,2,2,2,3,3,5,6,7,7,7,7},0,std::vector<int>{2,7},std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,1,2,2,5,7,7,3,7,6,2,3,7},0,std::vector<int>{1,4},std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,1,2,2,2,3,7,7,7,6,5,3,7},0,std::vector<int>{1,4,6},std::less<void>{},true),
        std::make_tuple(tensor_type{0,1,1,2,2,3,2,3,5,6,7,7,7,7},0,std::vector<int>{1,4,8},std::less<void>{},true),
        std::make_tuple(tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,1,4,3},{4,4,2,4,4},{8,7,6,6,5}},0,std::vector<int>{1,3},std::less<void>{},true),
        std::make_tuple(tensor_type{{-1,1,2,3,6},{0,1,2,5,8},{-1,0,2,4,7},{1,2,4,4,4},{1,3,3,4,6}},1,std::vector<int>{1,3},std::less<void>{},true),
        //false
        std::make_tuple(tensor_type{0,1,5,2,2,1,7,7,3,7,6,2,3,7},0,std::vector<int>{1,4},std::less<void>{},false),
        std::make_tuple(tensor_type{0,1,1,3,2,2,2,3,5,6,7,7,7,7},0,std::vector<int>{1,4,8},std::less<void>{},false),
        std::make_tuple(tensor_type{{-1,1,-1,0,2},{2,1,0,1,3},{3,2,2,4,3},{4,4,1,4,4},{8,7,6,6,5}},0,std::vector<int>{1,3},std::less<void>{},false),
        std::make_tuple(tensor_type{{-1,1,2,3,6},{0,1,2,5,8},{0,-1,2,4,7},{1,2,4,4,4},{1,3,3,4,6}},1,std::vector<int>{1,3},std::less<void>{},false)
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto nth = std::get<2>(t);
        auto comparator = std::get<3>(t);
        auto expected = std::get<4>(t);
        auto result = is_partitioned(ten,axis,nth,comparator);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

//partition, argpartition
TEST_CASE("test_sort_search_partition_argpartition","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::partition;
    using test_partition_::is_partitioned;
    using helpers_for_testing::apply_by_element;

    //0tensor,1nth,2axis,3comparator
    auto test_data = std::make_tuple(
        //nth scalar
        //no comparator
        std::make_tuple(tensor_type{},0,0,std::less<void>{}),
        std::make_tuple(tensor_type{1},0,0,std::less<void>{}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,0,std::less<void>{}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},4,0,std::less<void>{}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,0,std::less<void>{}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},2,0,std::less<void>{}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},4,0,std::less<void>{}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},7,0,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,0,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},2,0,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},3,0,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},4,0,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},0,1,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},2,1,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},3,1,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},4,1,std::less<void>{}),
        //comparator
        std::make_tuple(tensor_type{},0,0,std::greater<void>{}),
        std::make_tuple(tensor_type{1},0,0,std::greater<void>{}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},0,0,std::greater<void>{}),
        std::make_tuple(tensor_type{1,2,3,4,5,6},4,0,std::greater<void>{}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},0,0,std::greater<void>{}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},2,0,std::greater<void>{}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},4,0,std::greater<void>{}),
        std::make_tuple(tensor_type{2,1,6,3,2,1,0,5},7,0,std::greater<void>{}),
        //nth container
        std::make_tuple(tensor_type{},std::vector<int>{0,2,4},0,std::less<void>{}),
        std::make_tuple(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{4},0,std::less<void>{}),
        std::make_tuple(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{8,2,5},0,std::less<void>{}),
        std::make_tuple(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{0,7,13},0,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},0,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},1,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{4,0,2},0,std::less<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{4,0,2},1,std::less<void>{}),
        //comparator
        std::make_tuple(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{8,2,5},0,std::greater<void>{}),
        std::make_tuple(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,3},1,std::greater<void>{})

    );
    auto test_partition = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto nth = std::get<1>(t);
            auto axis = std::get<2>(t);
            auto comparator = std::get<3>(t);
            auto result = partition(policy...,ten,nth,axis,comparator);
            REQUIRE(is_partitioned(result,axis,nth,comparator));
        };
        apply_by_element(test,test_data);
    };
    auto test_argpartition = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto ten = std::get<0>(t);
            auto nth = std::get<1>(t);
            auto axis = std::get<2>(t);
            auto comparator = std::get<3>(t);
            auto arg_result = argpartition(policy...,ten,nth,axis,comparator);
            auto result = gtensor::take_along_axis(ten,arg_result,axis);
            REQUIRE(is_partitioned(result,axis,nth,comparator));
        };
        apply_by_element(test,test_data);
    };
    SECTION("partition_default_policy")
    {
        test_partition();
    }
    SECTION("partition_exec_pol<4>")
    {
        test_partition(multithreading::exec_pol<4>{});
    }
    SECTION("partition_exec_pol<0>")
    {
        test_partition(multithreading::exec_pol<0>{});
    }
    SECTION("argpartition_default_policy")
    {
        test_argpartition();
    }
    SECTION("argpartition_exec_pol<4>")
    {
        test_argpartition(multithreading::exec_pol<4>{});
    }
    SECTION("argpartition_exec_pol<0>")
    {
        test_argpartition(multithreading::exec_pol<0>{});
    }
}

TEST_CASE("test_sort_search_partition_argpartition_overload_default_policy","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::partition;
    using gtensor::take_along_axis;
    using test_partition_::is_partitioned;

    const auto test_tensor = tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}};

    //partition
    //default comparator
    REQUIRE(is_partitioned(partition(test_tensor,3,0),0,3,std::less<void>{}));
    REQUIRE(is_partitioned(partition(test_tensor,std::vector<int>{1,3},0),0,std::vector<int>{1,3},std::less<void>{}));
    //default comparator, axis
    REQUIRE(is_partitioned(partition(test_tensor,3),1,3,std::less<void>{}));
    REQUIRE(is_partitioned(partition(test_tensor,std::vector<int>{1,3}),1,std::vector<int>{1,3},std::less<void>{}));

    //argpartition
    //default comparator
    REQUIRE(is_partitioned(take_along_axis(test_tensor,argpartition(test_tensor,3,0),0),0,3,std::less<void>{}));
    REQUIRE(is_partitioned(take_along_axis(test_tensor,argpartition(test_tensor,std::vector<int>{1,3},0),0),0,std::vector<int>{1,3},std::less<void>{}));
    //default comparator, axis
    REQUIRE(is_partitioned(take_along_axis(test_tensor,argpartition(test_tensor,3),1),1,3,std::less<void>{}));
    REQUIRE(is_partitioned(take_along_axis(test_tensor,argpartition(test_tensor,std::vector<int>{1,3}),1),1,std::vector<int>{1,3},std::less<void>{}));
}

TEMPLATE_TEST_CASE("test_sort_search_partition_overload_policy","[test_sort_search]",
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::partition;
    using test_partition_::is_partitioned;

    const auto test_tensor = tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}};

    //default comparator
    REQUIRE(is_partitioned(partition(policy{},test_tensor,3,0),0,3,std::less<void>{}));
    REQUIRE(is_partitioned(partition(policy{},test_tensor,std::vector<int>{1,3},0),0,std::vector<int>{1,3},std::less<void>{}));
    //default comparator, axis
    REQUIRE(is_partitioned(partition(policy{},test_tensor,3),1,3,std::less<void>{}));
    REQUIRE(is_partitioned(partition(policy{},test_tensor,std::vector<int>{1,3}),1,std::vector<int>{1,3},std::less<void>{}));

    //argpartition
    //default comparator
    REQUIRE(is_partitioned(take_along_axis(test_tensor,argpartition(policy{},test_tensor,3,0),0),0,3,std::less<void>{}));
    REQUIRE(is_partitioned(take_along_axis(test_tensor,argpartition(policy{},test_tensor,std::vector<int>{1,3},0),0),0,std::vector<int>{1,3},std::less<void>{}));
    //default comparator, axis
    REQUIRE(is_partitioned(take_along_axis(test_tensor,argpartition(policy{},test_tensor,3),1),1,3,std::less<void>{}));
    REQUIRE(is_partitioned(take_along_axis(test_tensor,argpartition(policy{},test_tensor,std::vector<int>{1,3}),1),1,std::vector<int>{1,3},std::less<void>{}));
}

TEST_CASE("test_sort_search_partition_argpartition_exception","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::detail::no_value;
    using gtensor::value_error;
    using gtensor::partition;

    //partition
    //nth out of bound
    REQUIRE_THROWS_AS(partition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},14,0,no_value{}), value_error);
    REQUIRE_THROWS_AS(partition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{2,3,14},0,no_value{}), value_error);
    REQUIRE_THROWS_AS(partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,5},0,std::greater<void>{}), value_error);
    REQUIRE_THROWS_AS(partition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{6,2},1,std::greater<void>{}), value_error);
    //nth empty container
    REQUIRE_THROWS_AS(partition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{},0,no_value{}), value_error);

    //argpartition
    //nth out of bound
    REQUIRE_THROWS_AS(argpartition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},14,0,no_value{}), value_error);
    REQUIRE_THROWS_AS(argpartition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{2,3,14},0,no_value{}), value_error);
    REQUIRE_THROWS_AS(argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{1,5},0,std::greater<void>{}), value_error);
    REQUIRE_THROWS_AS(argpartition(tensor_type{{2,1,-1,6,3},{8,2,1,0,5},{-1,7,0,4,2},{4,4,2,1,4},{3,1,6,4,3}},std::vector<int>{6,2},1,std::greater<void>{}), value_error);
    //nth empty container
    REQUIRE_THROWS_AS(argpartition(tensor_type{7,1,7,7,1,5,7,2,3,2,6,2,3,0},std::vector<int>{},0,no_value{}), value_error);
}

TEST_CASE("test_argpartition_result_type","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = tensor_type::dim_type;
    using index_type = tensor_type::index_type;
    using result_tensor_type = gtensor::tensor<index_type>;
    using gtensor::detail::no_value;
    using gtensor::argpartition;

    REQUIRE(std::is_same_v<decltype(argpartition(std::declval<tensor_type>(),std::declval<index_type>(),std::declval<dim_type>(),std::declval<no_value>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argpartition(std::declval<tensor_type>(),std::declval<index_type>(),std::declval<dim_type>(),std::declval<std::less<void>>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argpartition(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<dim_type>(),std::declval<no_value>())),result_tensor_type>);
    REQUIRE(std::is_same_v<decltype(argpartition(std::declval<tensor_type>(),std::declval<std::vector<int>>(),std::declval<dim_type>(),std::declval<std::less<void>>())),result_tensor_type>);
}

