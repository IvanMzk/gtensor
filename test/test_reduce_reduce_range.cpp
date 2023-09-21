#include <algorithm>
#include "catch.hpp"
#include "builder.hpp"
#include "reduce.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

namespace test_reduce_{

struct max
{
    template<typename It>
    auto operator()(It first, It last){
        if (first==last){throw gtensor::value_error{"empty range"};}
        const auto& init = *first;
        return std::accumulate(++first,last,init, [](const auto& u, const auto& v){return std::max(u,v);});
    }
};
struct min
{
    template<typename It>
    auto operator()(It first, It last){
        if (first==last){throw gtensor::value_error{"empty range"};}
        const auto& init = *first;
        return std::accumulate(++first,last,init, [](const auto& u, const auto& v){return std::min(u,v);});
    }
};
struct min_or_zero
{
    template<typename It>
    auto operator()(It first, It last){
        auto res = min{}(first,last);
        return res < 0 ? 0 : res;
    }
};

struct sum
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{0};}
        const auto& init = *first;
        return std::accumulate(++first,last,init,std::plus{});
    }
};
struct sum_of_squares
{
    template<typename It>
    auto operator()(It first, It last){
        const auto res = sum{}(first,last);
        return res*res;
    }
};
struct sum_random_access
{
    template<typename It>
    auto operator()(It first, It last){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{0};}
        const auto n = last-first;
        difference_type i{0};
        value_type res = first[i];
        for (++i;i!=n; ++i){
            res+=first[i];
        }
        return res;
    }
};
struct sum_random_access_reverse
{
    template<typename It>
    auto operator()(It first, It last){
        using difference_type = typename std::iterator_traits<It>::difference_type;
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{0};}
        const auto n = last-first;
        difference_type i{-1};
        value_type res = last[i];
        for (--i;i!=-n-1; --i){
            res+=last[i];
        }
        return res;
    }
};

struct sum_init
{
    template<typename It>
    auto operator()(It first, It last, const typename std::iterator_traits<It>::value_type& init){
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{0};}
        return std::accumulate(first,last,init,std::plus{});
    }
};
struct prod
{
    template<typename It>
    auto operator()(It first, It last){
        using value_type = typename std::iterator_traits<It>::value_type;
        if (first==last){return value_type{1};}
        value_type prod{1};
        while(last!=first){
            prod*=*--last;
        }
        return prod;
    }
};

//take central element, order matters
struct center
{
    template<typename It>
    auto operator()(It first, It last){
        const auto n = last-first;
        const auto i=n/2;
        auto center_it = first+i;
        const auto res = *center_it;
        if (n%2==0){
            return (res+*--center_it)/2;
        }
        return res;
    }
};

}   //end of namespace test_reduce_

TEMPLATE_TEST_CASE("test_reduce_range","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sum;
    using test_reduce_::sum_of_squares;
    using test_reduce_::sum_random_access;
    using test_reduce_::sum_random_access_reverse;
    using test_reduce_::prod;
    using test_reduce_::max;
    using test_reduce_::min;
    using test_reduce_::min_or_zero;
    using test_reduce_::center;
    using gtensor::reduce_range;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{
        {{{{7,4,6,5,7},{3,1,3,3,8},{3,5,6,7,6},{5,7,1,1,6}},{{6,4,0,3,8},{5,3,3,8,7},{0,1,7,2,3},{5,5,0,2,5}},{{8,7,7,4,5},{1,8,6,8,4},{2,7,1,6,2},{6,5,6,0,3}}},
        {{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}}},
        {{{{1,7,2,7,8},{1,7,1,8,1},{2,3,4,2,0},{7,5,8,5,0}},{{5,5,1,3,8},{0,8,0,0,2},{5,1,2,3,0},{6,7,3,7,4}},{{8,0,7,0,0},{2,4,1,5,8},{5,6,8,4,8},{4,1,3,2,7}}},
        {{{0,6,2,7,3},{6,4,2,6,4},{7,0,3,3,1},{2,1,3,0,4}},{{7,4,4,7,6},{3,3,6,7,4},{1,7,4,0,1},{2,3,0,6,8}},{{2,4,1,6,0},{3,5,2,6,7},{5,7,5,4,4},{7,8,0,2,2}}}},
        {{{{0,7,1,1,0},{2,7,5,3,3},{6,5,4,8,6},{4,8,0,6,4}},{{0,0,5,8,0},{8,1,6,4,7},{2,5,4,6,3},{0,4,0,2,7}},{{6,0,3,6,4},{1,5,3,8,0},{8,7,2,4,0},{8,3,2,3,6}}},
        {{{2,8,5,4,4},{0,0,3,8,5},{4,1,4,2,1},{4,1,8,1,1}},{{7,2,8,8,3},{3,4,3,3,6},{1,6,2,7,7},{0,5,4,6,1}},{{1,4,0,7,6},{8,7,6,8,2},{6,4,0,5,8},{6,4,2,4,0}}}},
        {{{{0,5,0,8,6},{5,5,3,8,1},{8,3,7,8,5},{1,4,3,4,4}},{{4,0,0,6,8},{4,8,0,1,7},{6,2,6,4,2},{4,7,5,8,1}},{{3,3,1,5,5},{2,4,6,0,5},{3,1,7,6,5},{6,2,8,1,2}}},
        {{{4,7,5,2,1},{6,5,3,1,5},{8,8,5,5,4},{3,3,4,1,5}},{{7,8,2,8,1},{6,0,2,4,5},{8,4,5,0,3},{7,2,5,0,0}},{{2,2,2,7,8},{1,0,7,5,8},{0,2,5,4,4},{1,3,5,8,4}}}}
    };  //(4,2,3,4,5)

    //0tensor,1axes,2functor,3keep_dims,4any_order,5expected
    auto test_data = std::make_tuple(
        //single axis
        //keep_dims is false
        std::make_tuple(tensor_type{}, dim_type{0}, sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}, dim_type{0}, prod{}, false, true, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{0}, sum{}, false, true, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{1}, sum{}, false, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{0}, sum{}, false, true, tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{1}, sum{}, false, true, tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, sum{}, false, true, tensor_type{{value_type{0},value_type{0},value_type{0}},{value_type{0},value_type{0},value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, prod{}, false, true, tensor_type{{value_type{1},value_type{1},value_type{1}},{value_type{1},value_type{1},value_type{1}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{0}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{0}, sum{}, false, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{1}, sum{}, false, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{0}, sum{}, false, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{1}, sum{}, false, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum{}, false, true, tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum{}, false, true, tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, prod{}, false, true, tensor_type{6,120}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{0}, max{}, false, true, tensor_type{4,6,7,9}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{1}, min{}, false, true, tensor_type{1,0}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{0}, sum{}, false, true, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum_random_access{}, false, true, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum_random_access_reverse{}, false, true, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{1}, sum{}, false, true, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-2}, sum{}, false, true, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{2}, sum{}, false, true, tensor_type{{1,5},{9,13}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-1}, sum{}, false, true, tensor_type{{1,5},{9,13}}),
        //keep_dims is true
        std::make_tuple(tensor_type{}, dim_type{0}, sum{}, true, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{0}, sum{}, true, true, tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{1}, sum{}, true, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{0}, sum{}, true, true, tensor_type{}.reshape(1,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{1}, sum{}, true, true, tensor_type{}.reshape(2,1,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, sum{}, true, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}},{{value_type{0}},{value_type{0}},{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, prod{}, true, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}},{{value_type{1}},{value_type{1}},{value_type{1}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{0}, sum{}, true, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{0}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{1}, sum{}, true, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{0}, sum{}, true, true, tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{1}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum{}, true, true, tensor_type{{5,7,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum{}, true, true, tensor_type{{6},{15}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, prod{}, true, true, tensor_type{{6},{120}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{0}, max{}, true, true, tensor_type{{4,6,7,9}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{1}, min{}, true, true, tensor_type{{1},{0}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{0}, sum{}, true, true, tensor_type{{{4,6},{8,10}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum{}, true, true, tensor_type{{{4,6},{8,10}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{1}, sum{}, true, true, tensor_type{{{2,4}},{{10,12}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-2}, sum{}, true, true, tensor_type{{{2,4}},{{10,12}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{2}, sum{}, true, true, tensor_type{{{1},{5}},{{9},{13}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-1}, sum{}, true, true, tensor_type{{{1},{5}},{{9},{13}}}),
        //axes is container
        //keep_dims is false
        //empty axes
        std::make_tuple(tensor_type{}, std::vector<dim_type>{}, sum{}, false, true, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{}, sum{}, false, true, tensor_type{}.reshape(2,3,0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum{}, false, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum_of_squares{}, false, true, tensor_type{1,4,9,16,25,36}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum{}, false, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum_of_squares{}, false, true, tensor_type{{1},{4},{9},{16},{25},{36}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum{}, false, true, tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum_of_squares{}, false, true, tensor_type{{1,4,9,16,25,36}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{}, prod{}, false, true, tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type{{1,-2,3},{-4,5,6}}, std::vector<dim_type>{}, min_or_zero{}, false, true, tensor_type{{1,0,3},{0,5,6}}),
        //not empty axes
        std::make_tuple(tensor_type{}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0}, sum{}, false, true, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1}, sum{}, false, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0,1}, sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1,0}, sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0}, sum{}, false, true, tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{1}, sum{}, false, true, tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, sum{}, false, true, tensor_type{{value_type{0},value_type{0},value_type{0}},{value_type{0},value_type{0},value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, sum{}, false, true, tensor_type{value_type{0},value_type{0},value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, prod{}, false, true, tensor_type{{value_type{1},value_type{1},value_type{1}},{value_type{1},value_type{1},value_type{1}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, prod{}, false, true, tensor_type{value_type{1},value_type{1},value_type{1}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0,1}, sum{}, false, true, tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1}, sum{}, false, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1,0}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0,1}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{1}, sum{}, false, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0,1}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, sum{}, false, true, tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1,0}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, prod{}, false, true, tensor_type{6,120}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, prod{}, false, true, tensor_type{4,10,18}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0,1}, prod{}, false, true, tensor_type(720)),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{0}, sum{}, false, true, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, sum{}, false, true, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, prod{}, false, true, tensor_type{{0,3},{24,35}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2}, sum{}, false, true, tensor_type{{1,5},{9,13}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum_random_access{}, false, true, tensor_type{6,22}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum_random_access_reverse{}, false, true, tensor_type{6,22}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2,0}, prod{}, false, true, tensor_type{0,252}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-2,-1}, sum{}, false, true, tensor_type{6,22}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-1,-3}, prod{}, false, true, tensor_type{0,252}),
        //keep_dims is true
        //empty axes
        std::make_tuple(tensor_type{}, std::vector<dim_type>{}, sum{}, true, true, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{}, sum{}, true, true, tensor_type{}.reshape(2,3,0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum{}, true, true, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum_of_squares{}, true, true, tensor_type{1,4,9,16,25,36}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum{}, true, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum_of_squares{}, true, true, tensor_type{{1},{4},{9},{16},{25},{36}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum{}, true, true, tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum_of_squares{}, true, true, tensor_type{{1,4,9,16,25,36}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{}, prod{}, true, true, tensor_type{{1,2,3},{4,5,6}}),
        std::make_tuple(tensor_type{{1,-2,3},{-4,5,6}}, std::vector<dim_type>{}, min_or_zero{}, true, true, tensor_type{{1,0,3},{0,5,6}}),
        //not empty axes
        std::make_tuple(tensor_type{}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0}, sum{}, true, true, tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1}, sum{}, true, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0,1}, sum{}, true, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1,0}, sum{}, true, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0}, sum{}, true, true, tensor_type{}.reshape(1,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{1}, sum{}, true, true, tensor_type{}.reshape(2,1,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, sum{}, true, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}},{{value_type{0}},{value_type{0}},{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, sum{}, true, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, prod{}, true, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}},{{value_type{1}},{value_type{1}},{value_type{1}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, prod{}, true, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0,1}, sum{}, true, true, tensor_type{}.reshape(1,1,0)),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1}, sum{}, true, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1,0}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0,1}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{1}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0,1}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{{5,7,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, sum{}, true, true, tensor_type{{6},{15}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1,0}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, prod{}, true, true, tensor_type{{6},{120}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, prod{}, true, true, tensor_type{{4,10,18}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0,1}, prod{}, true, true, tensor_type{{720}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{0}, sum{}, true, true, tensor_type{{{4,6},{8,10}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, sum{}, true, true, tensor_type{{{2,4}},{{10,12}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, prod{}, true, true, tensor_type{{{0,3}},{{24,35}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2}, sum{}, true, true, tensor_type{{{1},{5}},{{9},{13}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum{}, true, true, tensor_type{{{6}},{{22}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2,0}, prod{}, true, true, tensor_type{{{0},{252}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-2,-1}, sum{}, true, true, tensor_type{{{6}},{{22}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-1,-3}, prod{}, true, true, tensor_type{{{0},{252}}}),
        //any_order false, c_order traverse along axes
        std::make_tuple(tensor_type{{{7,4,8,8},{3,4,5,6},{4,0,0,0}},{{0,1,4,7},{0,1,2,7},{2,8,3,4}}}, std::vector<dim_type>{0,1}, center{}, false, false, tensor_type{2.0,0.5,2.0,3.5}),
        std::make_tuple(tensor_type{{{7,4,8,8},{3,4,5,6},{4,0,0,0}},{{0,1,4,7},{0,1,2,7},{2,8,3,4}}}, std::vector<dim_type>{1,2}, center{}, false, false, tensor_type{4.5,1.5}),
        std::make_tuple(tensor_type{{{7,4,8,8},{3,4,5,6},{4,0,0,0}},{{0,1,4,7},{0,1,2,7},{2,8,3,4}}}, std::vector<dim_type>{0,1,2}, center{}, false, false, tensor_type(0)),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten,std::vector<int>{0,1,2},sum{},false,false,tensor_type{{360,400,304,508,412},{312,384,332,460,420},{412,384,408,376,328},{396,436,336,352,376}}),
        std::make_tuple((test_ten-1)*(test_ten+1),std::vector<int>{1,2,3},sum{},false,false,tensor_type{{491,613,533,540,639},{465,590,322,550,530},{497,532,368,728,438},{537,446,490,628,513}}),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,1)+test_ten(2,0,2),std::vector<int>{0,1,2},sum{},false,false,tensor_type{{430,324,312,539,399},{266,388,331,579,325},{471,472,346,322,210},{487,429,240,312,486}}),
        std::make_tuple((test_ten+test_ten(0))*(test_ten(1,1)-test_ten(2,0,2)),std::vector<int>{1,2,3},sum{},false,false,tensor_type{{-422,194,-6,-98,176},{-388,206,51,-79,284},{-329,169,26,-155,179},{-383,129,6,-87,220}})
    );
    auto test_reduce_range = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto tensor = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto any_order = std::get<4>(t);
            auto expected = std::get<5>(t);
            auto result = reduce_range(policy...,tensor, axes, functor, keep_dims, any_order);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    };

    SECTION("default_policy")
    {
        test_reduce_range();
    }
    SECTION("exec_pol<4>")
    {
        test_reduce_range(multithreading::exec_pol<4>{});
    }
}

TEST_CASE("test_reduce_range_custom_arg","[test_reduce]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sum_init;
    using gtensor::reduce_range;
    using helpers_for_testing::apply_by_element;
    //0tensor,1axes,2functor,3keep_dims,4any_order,5init,6expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, sum_init{}, false, true, value_type{0}, tensor_type(15)),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, sum_init{}, true, true, value_type{-1}, tensor_type{14}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum_init{}, false, true, value_type{-1}, tensor_type{4,6,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum_init{}, false, true, value_type{1}, tensor_type{7,16})
    );
    auto test_reduce_range_custom_arg = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto tensor = std::get<0>(t);
            auto axes = std::get<1>(t);
            auto functor = std::get<2>(t);
            auto keep_dims = std::get<3>(t);
            auto any_order = std::get<4>(t);
            auto init = std::get<5>(t);
            auto expected = std::get<6>(t);
            auto result = reduce_range(policy...,tensor, axes, functor, keep_dims, any_order, init);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    };

    SECTION("default_policy")
    {
        test_reduce_range_custom_arg();
    }
    SECTION("exec_pol<4>")
    {
        test_reduce_range_custom_arg(multithreading::exec_pol<4>{});
    }
}

TEST_CASE("test_reduce_range_ecxeption","[test_reduce]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sum;
    using gtensor::axis_error;
    using gtensor::reduce_range;
    using helpers_for_testing::apply_by_element;


    //0tensor,1axes,2functor,3keep_dim,4any_order
    auto test_data = std::make_tuple(
        //single axis
        std::make_tuple(tensor_type(0), dim_type{0}, sum{}, false, true),
        std::make_tuple(tensor_type{}, dim_type{1}, sum{}, false, true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{1}, sum{}, false, true),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{2}, sum{}, false, true),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{2}, sum{}, false, true),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{4}, sum{}, false, true),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{3}, sum{}, false, true),
        //axes container
        std::make_tuple(tensor_type(0), std::vector<dim_type>{0}, sum{}, false, true),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{0,0}, sum{}, false, true),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{1,1}, sum{}, false, true),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{1}, sum{}, false, true),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{0,1}, sum{}, false, true),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{1,0}, sum{}, false, true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0,0}, sum{}, false, true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{1,1}, sum{}, false, true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0,1}, sum{}, false, true),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{1,0}, sum{}, false, true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{3}, sum{}, false, true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{0,1,0}, sum{}, false, true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{1,1}, sum{}, false, true),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{0,1,2,0}, sum{}, false, true)
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto keep_dim = std::get<3>(t);
        auto any_order = std::get<4>(t);
        REQUIRE_THROWS_AS(reduce_range(multithreading::exec_pol<1>{}, tensor, axes, functor, keep_dim, any_order), axis_error);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_reduce_range_flatten","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using test_reduce_::sum;
    using test_reduce_::prod;
    using test_reduce_::max;
    using test_reduce_::min;
    using test_reduce_::center;
    using gtensor::reduce_range;
    using helpers_for_testing::apply_by_element;

    const auto test_ten = tensor_type{
        {{{{7,4,6,5,7},{3,1,3,3,8},{3,5,6,7,6},{5,7,1,1,6}},{{6,4,0,3,8},{5,3,3,8,7},{0,1,7,2,3},{5,5,0,2,5}},{{8,7,7,4,5},{1,8,6,8,4},{2,7,1,6,2},{6,5,6,0,3}}},
        {{{2,2,7,5,5},{0,0,3,7,1},{8,2,5,0,1},{0,7,7,5,8}},{{1,5,6,7,0},{6,4,1,4,2},{2,1,0,1,1},{6,6,3,6,7}},{{7,6,1,3,7},{2,3,8,0,3},{3,8,6,3,7},{5,8,4,8,5}}}},
        {{{{1,7,2,7,8},{1,7,1,8,1},{2,3,4,2,0},{7,5,8,5,0}},{{5,5,1,3,8},{0,8,0,0,2},{5,1,2,3,0},{6,7,3,7,4}},{{8,0,7,0,0},{2,4,1,5,8},{5,6,8,4,8},{4,1,3,2,7}}},
        {{{0,6,2,7,3},{6,4,2,6,4},{7,0,3,3,1},{2,1,3,0,4}},{{7,4,4,7,6},{3,3,6,7,4},{1,7,4,0,1},{2,3,0,6,8}},{{2,4,1,6,0},{3,5,2,6,7},{5,7,5,4,4},{7,8,0,2,2}}}},
        {{{{0,7,1,1,0},{2,7,5,3,3},{6,5,4,8,6},{4,8,0,6,4}},{{0,0,5,8,0},{8,1,6,4,7},{2,5,4,6,3},{0,4,0,2,7}},{{6,0,3,6,4},{1,5,3,8,0},{8,7,2,4,0},{8,3,2,3,6}}},
        {{{2,8,5,4,4},{0,0,3,8,5},{4,1,4,2,1},{4,1,8,1,1}},{{7,2,8,8,3},{3,4,3,3,6},{1,6,2,7,7},{0,5,4,6,1}},{{1,4,0,7,6},{8,7,6,8,2},{6,4,0,5,8},{6,4,2,4,0}}}},
        {{{{0,5,0,8,6},{5,5,3,8,1},{8,3,7,8,5},{1,4,3,4,4}},{{4,0,0,6,8},{4,8,0,1,7},{6,2,6,4,2},{4,7,5,8,1}},{{3,3,1,5,5},{2,4,6,0,5},{3,1,7,6,5},{6,2,8,1,2}}},
        {{{4,7,5,2,1},{6,5,3,1,5},{8,8,5,5,4},{3,3,4,1,5}},{{7,8,2,8,1},{6,0,2,4,5},{8,4,5,0,3},{7,2,5,0,0}},{{2,2,2,7,8},{1,0,7,5,8},{0,2,5,4,4},{1,3,5,8,4}}}}
    };  //(4,2,3,4,5)

    //0tensor,1functor,2keep_dims,3any_order,4expected
    auto test_data = std::make_tuple(
        //keep_dims is false
        std::make_tuple(tensor_type{}, sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}, prod{}, false, true, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{}.reshape(1,0), sum{}, false, true, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), prod{}, false, true, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, sum{}, false, true, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, prod{}, false, true, tensor_type(720)),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, false, true, tensor_type(9)),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, false, true, tensor_type(0)),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, sum{}, false, true, tensor_type(28)),
        //keep_dims is true
        std::make_tuple(tensor_type{}, sum{}, true, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}, prod{}, true, true, tensor_type{value_type{1}}),
        std::make_tuple(tensor_type{}.reshape(2,1,0), sum{}, true, true, tensor_type{{{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), prod{}, true, true, tensor_type{{{value_type{1}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, sum{}, true, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, sum{}, true, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, prod{}, true, true, tensor_type{{720}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, true, true, tensor_type{{9}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, true, true, tensor_type{{0}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, sum{}, true, true, tensor_type{{{28}}}),
        //any_order
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, center{}, false, false, tensor_type(6.5)),
        //trivial view input
        std::make_tuple(test_ten+test_ten+test_ten+test_ten,sum{},false,false,tensor_type(7696)),
        std::make_tuple((test_ten-1)*(test_ten+1),sum{},false,false,tensor_type(10450)),
        //non trivial view input
        std::make_tuple(test_ten+test_ten(0)+test_ten(1,1)+test_ten(2,0,2),sum{},false,false,tensor_type(7668)),
        std::make_tuple((test_ten+test_ten(0))*(test_ten(1,1)-test_ten(2,0,2)),sum{},false,false,tensor_type(-307))
    );
    auto test_reduce_range_flatten = [&test_data](auto...policy){
        auto test = [policy...](const auto& t){
            auto tensor = std::get<0>(t);
            auto functor = std::get<1>(t);
            auto keep_dims = std::get<2>(t);
            auto any_order = std::get<3>(t);
            auto expected = std::get<4>(t);
            auto result = reduce_range(policy...,tensor, gtensor::detail::no_value{}, functor, keep_dims, any_order);
            REQUIRE(result == expected);
        };
        apply_by_element(test, test_data);
    };

    SECTION("default_policy")
    {
        test_reduce_range_flatten();
    }
    SECTION("exec_pol<4>")
    {
        test_reduce_range_flatten(multithreading::exec_pol<4>{});
    }
}

TEMPLATE_TEST_CASE("test_reduce_range_big","[test_reduce]",
    (multithreading::exec_pol<1>),
    (multithreading::exec_pol<4>),
    (multithreading::exec_pol<0>)
)
{
    using policy = TestType;
    using value_type = std::size_t;
    using tensor_type = gtensor::tensor<value_type>;
    using shape_type = tensor_type::shape_type;
    using test_reduce_::sum;
    using helpers_for_testing::generate_lehmer;
    using helpers_for_testing::apply_by_element;

    tensor_type t(shape_type{32,16,8,64,4,16}); //1<<24
    generate_lehmer(t.begin(),t.end(),[](const auto& e){return e%2;},123);

    //0ten,1axes,2range_f,3keep_dims,4any_order,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(std::cref(t),std::vector<int>{0,2,3,5},sum{},false,true,tensor_type{{130985,131241,131122,131418},{131063,130985,130771,131109},{130880,130776,131173,130602},{130953,131504,130845,130713},{130533,131118,131072,130999},{131537,131601,131137,130747},{130771,131109,131092,131087},{131009,131288,131239,131240},{131045,130487,131331,131042},{130738,130992,131102,131046},{131303,130886,131084,131374},{130716,131235,131133,130959},{130922,131557,131289,131151},{130930,130964,131054,130756},{131444,131149,131506,130919},{131779,130963,131140,130513}}),
        std::make_tuple(std::cref(t),std::vector<int>{0,1,3,5},sum{},false,true,tensor_type{{262550,262341,261387,262795},{262126,262338,263037,261910},{262039,261752,262161,261542},{261610,261842,262138,262175},{262190,262597,262274,262151},{262391,262182,262827,261800},{261970,262571,261781,261997},{261732,262232,262485,261305}}),
        std::make_tuple(std::cref(t),std::vector<int>{0,1,2,3,5},sum{},false,true,tensor_type{2096608,2097855,2098090,2095675}),
        std::make_tuple(std::cref(t),std::vector<int>{1,2,3,4,5},sum{},false,true,tensor_type{262294,262306,261408,262194,261907,262785,262000,262093,262364,261515,261966,262240,262489,262095,262097,262023,262345,261632,262444,262166,262217,262027,262144,262242,262540,261711,262199,261935,262167,262702,261858,262123}),
        std::make_tuple(std::cref(t),std::vector<int>{0,1,2,3,4,5},sum{},false,true, tensor_type(8388228)),
        std::make_tuple(std::cref(t),gtensor::detail::no_value{},sum{},false,true,tensor_type(8388228))
    );
    auto test = [](const auto& t){
        auto& ten = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto range_f = std::get<2>(t);
        auto keep_dims = std::get<3>(t);
        auto any_order = std::get<4>(t);
        auto expected = std::get<5>(t);
        auto result = gtensor::reduce_range(policy{},ten,axes,range_f,keep_dims,any_order);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

