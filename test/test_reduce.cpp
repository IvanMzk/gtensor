#include <algorithm>
#include "catch.hpp"
#include "tensor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

TEST_CASE("test_check_reduce_args","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using gtensor::reduce_exception;
    using gtensor::detail::check_reduce_args;

    //single reduce axis
    REQUIRE_NOTHROW(check_reduce_args(shape_type{0},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{10},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},dim_type{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,0},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,0},dim_type{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,0},dim_type{2}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},dim_type{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},dim_type{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},dim_type{2}));

    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},dim_type{0}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{0},dim_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{1,0},dim_type{2}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},dim_type{3}), reduce_exception);

    //container of axes
    REQUIRE_NOTHROW(check_reduce_args(shape_type{},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{0},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{0},std::vector<int>{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1},std::vector<int>{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{10},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{10},std::vector<int>{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},std::vector<int>{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},std::vector<int>{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},std::vector<int>{0,1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{1,0},std::vector<int>{1,0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{2}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{0,1}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{1,2}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{1,2,0}));
    REQUIRE_NOTHROW(check_reduce_args(shape_type{2,3,4},std::vector<int>{0,1,2}));

    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},std::vector<int>{0}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},std::vector<int>{0,0}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},std::vector<int>{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{},std::vector<int>{1,0}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{0},std::vector<int>{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{0},std::vector<int>{0,0}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{10},std::vector<int>{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{10},std::vector<int>{0,1}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},std::vector<int>{3}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},std::vector<int>{0,0}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},std::vector<int>{0,1,0}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},std::vector<int>{1,2,0,1}), reduce_exception);
    REQUIRE_THROWS_AS(check_reduce_args(shape_type{2,3,4},std::vector<int>{1,2,3}), reduce_exception);
}

TEST_CASE("test_make_reduce_shape","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_reduce_shape;
    using helpers_for_testing::apply_by_element;
    //0pshape,1axes,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //single axis
        //keep_dims is false
        std::make_tuple(shape_type{0},dim_type{0},false,shape_type{}),
        std::make_tuple(shape_type{1},dim_type{0},false,shape_type{}),
        std::make_tuple(shape_type{10},dim_type{0},false,shape_type{}),
        std::make_tuple(shape_type{2,3,0},dim_type{0},false,shape_type{3,0}),
        std::make_tuple(shape_type{2,3,0},dim_type{1},false,shape_type{2,0}),
        std::make_tuple(shape_type{2,3,0},dim_type{2},false,shape_type{2,3}),
        std::make_tuple(shape_type{2,3,4},dim_type{0},false,shape_type{3,4}),
        std::make_tuple(shape_type{2,3,4},dim_type{1},false,shape_type{2,4}),
        std::make_tuple(shape_type{2,3,4},dim_type{2},false,shape_type{2,3}),
        //keep_dims is true
        std::make_tuple(shape_type{0},dim_type{0},true,shape_type{1}),
        std::make_tuple(shape_type{1},dim_type{0},true,shape_type{1}),
        std::make_tuple(shape_type{10},dim_type{0},true,shape_type{1}),
        std::make_tuple(shape_type{2,3,0},dim_type{0},true,shape_type{1,3,0}),
        std::make_tuple(shape_type{2,3,0},dim_type{1},true,shape_type{2,1,0}),
        std::make_tuple(shape_type{2,3,0},dim_type{2},true,shape_type{2,3,1}),
        std::make_tuple(shape_type{2,3,4},dim_type{0},true,shape_type{1,3,4}),
        std::make_tuple(shape_type{2,3,4},dim_type{1},true,shape_type{2,1,4}),
        std::make_tuple(shape_type{2,3,4},dim_type{2},true,shape_type{2,3,1}),
        //container of axees
        //keep_dims is false, empty container (all axes)
        std::make_tuple(shape_type{},std::vector<int>{},false,shape_type{}),
        std::make_tuple(shape_type{0},std::vector<int>{},false,shape_type{}),
        std::make_tuple(shape_type{1},std::vector<int>{},false,shape_type{}),
        std::make_tuple(shape_type{10},std::vector<int>{},false,shape_type{}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{},false,shape_type{}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{},false,shape_type{}),
        //keep_dims is false, not empty container
        std::make_tuple(shape_type{0},std::vector<int>{0},false,shape_type{}),
        std::make_tuple(shape_type{10},std::vector<int>{0},false,shape_type{}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0},false,shape_type{3,0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{1,0},false,shape_type{0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0,1},false,shape_type{0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{2},false,shape_type{2,3}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{1,2},false,shape_type{2}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0,1,2},false,shape_type{}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{0,1,2},false,shape_type{}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{0},false,shape_type{3,4}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{2,0},false,shape_type{3}),
        //keep_dims is true, empty container (all axes)
        std::make_tuple(shape_type{},std::vector<int>{},true,shape_type{}),
        std::make_tuple(shape_type{0},std::vector<int>{},true,shape_type{1}),
        std::make_tuple(shape_type{1},std::vector<int>{},true,shape_type{1}),
        std::make_tuple(shape_type{10},std::vector<int>{},true,shape_type{1}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{},true,shape_type{1,1,1}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{},true,shape_type{1,1,1}),
        //keep_dims is true, not empty container
        std::make_tuple(shape_type{0},std::vector<int>{0},true,shape_type{1}),
        std::make_tuple(shape_type{10},std::vector<int>{0},true,shape_type{1}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0},true,shape_type{1,3,0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{1,0},true,shape_type{1,1,0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0,1},true,shape_type{1,1,0}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{2},true,shape_type{2,3,1}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{1,2},true,shape_type{2,1,1}),
        std::make_tuple(shape_type{2,3,0},std::vector<int>{0,1,2},true,shape_type{1,1,1}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{0,1,2},true,shape_type{1,1,1}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{0},true,shape_type{1,3,4}),
        std::make_tuple(shape_type{2,3,4},std::vector<int>{2,0},true,shape_type{1,3,1})
    );
    auto test = [](const auto& t){
        auto pshape = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = make_reduce_shape(pshape,axes,keep_dims);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_check_slide_args","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;
    using gtensor::reduce_exception;
    using gtensor::detail::check_slide_args;

    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{0},dim_type{0},index_type{0},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{1},shape_type{1},dim_type{0},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{5},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{10},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{1,0},dim_type{0},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{2,3,0},dim_type{0},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{2,3,0},dim_type{0},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{2,3,0},dim_type{1},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{2,3,0},dim_type{1},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{2,3,0},dim_type{1},index_type{3},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{0},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{0},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{1},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{1},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{1},index_type{3},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{2},index_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{2},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{2},index_type{3},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{2},index_type{4},index_type{1}));
    //empty tensor, window_size and window_step doesnt matter
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{0},dim_type{0},index_type{0},index_type{0}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{0},dim_type{0},index_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{0,2,3},dim_type{1},index_type{0},index_type{0}));
    REQUIRE_NOTHROW(check_slide_args(index_type{0},shape_type{0,2,3},dim_type{1},index_type{3},index_type{1}));

    //window_size greater than axis size
    REQUIRE_THROWS_AS(check_slide_args(index_type{1},shape_type{},dim_type{0},index_type{1},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{11},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{0},index_type{3},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{1},index_type{4},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{2},index_type{5},index_type{1}), reduce_exception);
    //invalid axis
    REQUIRE_THROWS_AS(check_slide_args(index_type{0},shape_type{0},dim_type{1},index_type{1},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(index_type{0},shape_type{0,2,3},dim_type{3},index_type{1},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(index_type{10},shape_type{10},dim_type{1},index_type{1},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{3},index_type{1},index_type{1}), reduce_exception);
    //zero window_step
    REQUIRE_THROWS_AS(check_slide_args(index_type{10},shape_type{10},dim_type{0},index_type{3},index_type{0}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(index_type{24},shape_type{2,3,4},dim_type{1},index_type{1},index_type{0}), reduce_exception);
}

TEST_CASE("test_make_slide_shape","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_slide_shape;
    //0psize,1pshape,2axis,3window_size,4window_step,5expected
    using test_type = std::tuple<index_type,shape_type,dim_type,index_type,index_type,shape_type>;
    auto test_data = GENERATE(
        test_type{index_type{0},shape_type{0},dim_type{0},index_type{1},index_type{1},shape_type{0}},
        test_type{index_type{0},shape_type{0},dim_type{0},index_type{2},index_type{1},shape_type{0}},
        test_type{index_type{0},shape_type{0},dim_type{0},index_type{1},index_type{2},shape_type{0}},
        test_type{index_type{0},shape_type{20,30,0},dim_type{0},index_type{5},index_type{2},shape_type{20,30,0}},
        test_type{index_type{0},shape_type{20,30,0},dim_type{1},index_type{5},index_type{2},shape_type{20,30,0}},
        test_type{index_type{0},shape_type{20,30,0},dim_type{2},index_type{5},index_type{2},shape_type{20,30,0}},
        test_type{index_type{1},shape_type{1},dim_type{0},index_type{1},index_type{1},shape_type{1}},
        test_type{index_type{1},shape_type{1},dim_type{0},index_type{1},index_type{2},shape_type{1}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{1},index_type{1},shape_type{10}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{1},index_type{2},shape_type{5}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{1},index_type{5},shape_type{2}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{2},index_type{1},shape_type{9}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{2},index_type{2},shape_type{5}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{2},index_type{5},shape_type{2}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{5},index_type{1},shape_type{6}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{5},index_type{2},shape_type{3}},
        test_type{index_type{10},shape_type{10},dim_type{0},index_type{5},index_type{5},shape_type{2}},
        test_type{index_type{6000},shape_type{5,30,40},dim_type{0},index_type{3},index_type{2},shape_type{2,30,40}},
        test_type{index_type{6000},shape_type{5,30,40},dim_type{1},index_type{5},index_type{1},shape_type{5,26,40}},
        test_type{index_type{6000},shape_type{5,30,40},dim_type{2},index_type{10},index_type{3},shape_type{5,30,11}}
    );
    auto psize = std::get<0>(test_data);
    auto pshape = std::get<1>(test_data);
    auto axis = std::get<2>(test_data);
    auto window_size = std::get<3>(test_data);
    auto window_step = std::get<4>(test_data);
    auto expected = std::get<5>(test_data);
    auto result = make_slide_shape(psize,pshape,axis,window_size,window_step);
    REQUIRE(result == expected);
}

namespace test_reduce_{

struct max
{
    template<typename It>
    auto operator()(It first, It last){
        if (first==last){throw gtensor::reduce_exception{"call max on empty range"};}
        const auto& init = *first;
        return std::accumulate(++first,last,init, [](const auto& u, const auto& v){return std::max(u,v);});
    }
};
struct min
{
    template<typename It>
    auto operator()(It first, It last){
        if (first==last){throw gtensor::reduce_exception{"call max on empty range"};}
        const auto& init = *first;
        return std::accumulate(++first,last,init, [](const auto& u, const auto& v){return std::min(u,v);});
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

struct cumsum{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        auto cumsum_ = *first;
        *dfirst = cumsum_;
        for(++dfirst,++first; dfirst!=dlast; ++dfirst,++first){
            cumsum_+=*first;
            *dfirst = cumsum_;
        }
    }
};

struct cumprod_reverse{
    template<typename It, typename DstIt>
    void operator()(It, It last, DstIt dfirst, DstIt dlast){
        auto cumprod_ = *--last;
        *--dlast = cumprod_;
        while(dlast!=dfirst){
            cumprod_*=*--last;
            *--dlast = cumprod_;
        }
    }
};

struct moving_avarage{
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It, DstIt dfirst, DstIt dlast, const IdxT& window_size, const IdxT& window_step, const typename std::iterator_traits<It>::value_type& denom){
        using index_type = IdxT;
        using value_type = typename std::iterator_traits<It>::value_type;
        value_type sum{0};
        auto it = first;
        for (index_type i{0}; i!=window_size; ++i, ++it){
            sum+=*it;
        }
        for(;dfirst!=dlast;++dfirst){
            *dfirst = sum/denom;
            for (index_type i{0}; i!=window_step; ++i,++first){
                sum-=*first;
            }
            for (index_type i{0}; i!=window_step; ++i, ++it){
                sum+=*it;
            }
        }
    }
};

struct diff_1{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        for (;dfirst!=dlast;++dfirst){
            auto prev = *first;
            *dfirst = *(++first) - prev;
        }
    }
};

struct diff_2{
    template<typename It, typename DstIt>
    void operator()(It first, It, DstIt dfirst, DstIt dlast){
        for (;dfirst!=dlast;++dfirst){
            auto v0 = *first;
            auto v1 = *(++first);
            auto v2 = *(++first);
            *dfirst = v2-v1-v1+v0;
            --first;
        }
    }
};

struct sort{
    template<typename It>
    void operator()(It first, It last){
        std::sort(first,last);
    }
};

}   //end of namespace test_reduce_

TEST_CASE("test_reduce","[test_reduce]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sum;
    using test_reduce_::sum_random_access;
    using test_reduce_::sum_random_access_reverse;
    using test_reduce_::prod;
    using test_reduce_::max;
    using test_reduce_::min;
    using gtensor::reduce;
    using helpers_for_testing::apply_by_element;
    //0tensor,1axes,2functor,3keep_dims,4expected
    auto test_data = std::make_tuple(
        //single axis
        //keep_dims is false
        std::make_tuple(tensor_type{}, dim_type{0}, sum{}, false, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}, dim_type{0}, prod{}, false, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{0}, sum{}, false, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{1}, sum{}, false, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{0}, sum{}, false, tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{1}, sum{}, false, tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, sum{}, false, tensor_type{{value_type{0},value_type{0},value_type{0}},{value_type{0},value_type{0},value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, prod{}, false, tensor_type{{value_type{1},value_type{1},value_type{1}},{value_type{1},value_type{1},value_type{1}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{0}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{0}, sum{}, false, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{1}, sum{}, false, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{0}, sum{}, false, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{1}, sum{}, false, tensor_type{21}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum{}, false, tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum{}, false, tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, prod{}, false, tensor_type{6,120}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{0}, max{}, false, tensor_type{4,6,7,9}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{1}, min{}, false, tensor_type{1,0}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{0}, sum{}, false, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum_random_access{}, false, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum_random_access_reverse{}, false, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{1}, sum{}, false, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-2}, sum{}, false, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{2}, sum{}, false, tensor_type{{1,5},{9,13}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-1}, sum{}, false, tensor_type{{1,5},{9,13}}),
        //keep_dims is true
        std::make_tuple(tensor_type{}, dim_type{0}, sum{}, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{0}, sum{}, true, tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{1}, sum{}, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{0}, sum{}, true, tensor_type{}.reshape(1,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{1}, sum{}, true, tensor_type{}.reshape(2,1,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, sum{}, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}},{{value_type{0}},{value_type{0}},{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, prod{}, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}},{{value_type{1}},{value_type{1}},{value_type{1}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{0}, sum{}, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{0}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{1}, sum{}, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{0}, sum{}, true, tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{1}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum{}, true, tensor_type{{5,7,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum{}, true, tensor_type{{6},{15}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, prod{}, true, tensor_type{{6},{120}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{0}, max{}, true, tensor_type{{4,6,7,9}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{1}, min{}, true, tensor_type{{1},{0}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{0}, sum{}, true, tensor_type{{{4,6},{8,10}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-3}, sum{}, true, tensor_type{{{4,6},{8,10}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{1}, sum{}, true, tensor_type{{{2,4}},{{10,12}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-2}, sum{}, true, tensor_type{{{2,4}},{{10,12}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{2}, sum{}, true, tensor_type{{{1},{5}},{{9},{13}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{-1}, sum{}, true, tensor_type{{{1},{5}},{{9},{13}}}),
        //axes is container
        //keep_dims is false
        std::make_tuple(tensor_type{}, std::vector<dim_type>{0}, sum{}, false, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}, std::vector<dim_type>{}, sum{}, false, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0}, sum{}, false, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1}, sum{}, false, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0,1}, sum{}, false, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1,0}, sum{}, false, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{}, sum{}, false, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0}, sum{}, false, tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{1}, sum{}, false, tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, sum{}, false, tensor_type{{value_type{0},value_type{0},value_type{0}},{value_type{0},value_type{0},value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, sum{}, false, tensor_type{value_type{0},value_type{0},value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, prod{}, false, tensor_type{{value_type{1},value_type{1},value_type{1}},{value_type{1},value_type{1},value_type{1}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, prod{}, false, tensor_type{value_type{1},value_type{1},value_type{1}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0,1}, sum{}, false, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{}, sum{}, false, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0}, sum{}, false, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1}, sum{}, false, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1,0}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0,1}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0}, sum{}, false, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{1}, sum{}, false, tensor_type{21}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0,1}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, sum{}, false, tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, sum{}, false, tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1,0}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, prod{}, false, tensor_type{6,120}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, prod{}, false, tensor_type{4,10,18}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0,1}, prod{}, false, tensor_type(720)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{}, prod{}, false, tensor_type(720)),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{0}, sum{}, false, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, sum{}, false, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, prod{}, false, tensor_type{{0,3},{24,35}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2}, sum{}, false, tensor_type{{1,5},{9,13}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum_random_access{}, false, tensor_type{6,22}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum_random_access_reverse{}, false, tensor_type{6,22}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2,0}, prod{}, false, tensor_type{0,252}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-2,-1}, sum{}, false, tensor_type{6,22}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-1,-3}, prod{}, false, tensor_type{0,252}),
        //keep_dims is true
        std::make_tuple(tensor_type{}, std::vector<dim_type>{0}, sum{}, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}, std::vector<dim_type>{}, sum{}, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0}, sum{}, true, tensor_type{}.reshape(1,0)),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1}, sum{}, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{0,1}, sum{}, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{1,0}, sum{}, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(1,0), std::vector<dim_type>{}, sum{}, true, tensor_type{{value_type{0}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0}, sum{}, true, tensor_type{}.reshape(1,3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{1}, sum{}, true, tensor_type{}.reshape(2,1,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, sum{}, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}},{{value_type{0}},{value_type{0}},{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, sum{}, true, tensor_type{{{value_type{0}},{value_type{0}},{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2}, prod{}, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}},{{value_type{1}},{value_type{1}},{value_type{1}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{2,0}, prod{}, true, tensor_type{{{value_type{1}},{value_type{1}},{value_type{1}}}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{0,1}, sum{}, true, tensor_type{}.reshape(1,1,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), std::vector<dim_type>{}, sum{}, true, tensor_type{{{value_type{0}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0}, sum{}, true, tensor_type{21}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{}, sum{}, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1}, sum{}, true, tensor_type{{1},{2},{3},{4},{5},{6}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{1,0}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{0,1}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, std::vector<dim_type>{}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0}, sum{}, true, tensor_type{{1,2,3,4,5,6}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{1}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{0,1}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, std::vector<dim_type>{}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, sum{}, true, tensor_type{{5,7,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, sum{}, true, tensor_type{{6},{15}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1,0}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{1}, prod{}, true, tensor_type{{6},{120}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0}, prod{}, true, tensor_type{{4,10,18}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{0,1}, prod{}, true, tensor_type{{720}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, std::vector<dim_type>{}, prod{}, true, tensor_type{{720}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{0}, sum{}, true, tensor_type{{{4,6},{8,10}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, sum{}, true, tensor_type{{{2,4}},{{10,12}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1}, prod{}, true, tensor_type{{{0,3}},{{24,35}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2}, sum{}, true, tensor_type{{{1},{5}},{{9},{13}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{1,2}, sum{}, true, tensor_type{{{6}},{{22}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{2,0}, prod{}, true, tensor_type{{{0},{252}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-2,-1}, sum{}, true, tensor_type{{{6}},{{22}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::vector<dim_type>{-1,-3}, prod{}, true, tensor_type{{{0},{252}}}),
        //axes in initializer_list
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::initializer_list<dim_type>{-2,-1}, sum{}, false, tensor_type{6,22}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::initializer_list<dim_type>{-1,-3}, prod{}, false, tensor_type{0,252}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::initializer_list<dim_type>{1,2}, sum{}, true, tensor_type{{{6}},{{22}}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, std::initializer_list<dim_type>{2,0}, prod{}, true, tensor_type{{{0},{252}}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto keep_dims = std::get<3>(t);
        auto expected = std::get<4>(t);
        auto result = reduce(tensor, axes, functor, keep_dims);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_reduce_custom_arg","[test_reduce]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sum_init;
    using gtensor::reduce;
    using helpers_for_testing::apply_by_element;
    //0tensor,1axes,2functor,3keep_dims,4init,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, sum_init{}, false, value_type{0}, tensor_type(15)),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, sum_init{}, true, value_type{-1}, tensor_type{14}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum_init{}, false, value_type{-1}, tensor_type{4,6,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum_init{}, false, value_type{1}, tensor_type{7,16})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto keep_dims = std::get<3>(t);
        auto init = std::get<4>(t);
        auto expected = std::get<5>(t);
        auto result = reduce(tensor, axes, functor, keep_dims, init);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_reduce_ecxeption","[test_reduce]")
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sum;
    using gtensor::reduce_exception;
    using gtensor::reduce;
    using helpers_for_testing::apply_by_element;


    //0tensor,1axes,2functor,3keep_dim
    auto test_data = std::make_tuple(
        //single axis
        std::make_tuple(tensor_type(0), dim_type{0}, sum{}, false),
        std::make_tuple(tensor_type{}, dim_type{1}, sum{}, false),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{1}, sum{}, false),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{2}, sum{}, false),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{2}, sum{}, false),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{4}, sum{}, false),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{3}, sum{}, false),
        //axes container
        std::make_tuple(tensor_type(0), std::vector<dim_type>{0}, sum{}, false),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{0,0}, sum{}, false),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{1,1}, sum{}, false),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{1}, sum{}, false),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{0,1}, sum{}, false),
        std::make_tuple(tensor_type{0}, std::vector<dim_type>{1,0}, sum{}, false),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0,0}, sum{}, false),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{1,1}, sum{}, false),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{0,1}, sum{}, false),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, std::vector<dim_type>{1,0}, sum{}, false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{3}, sum{}, false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{0,1,0}, sum{}, false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{1,1}, sum{}, false),
        std::make_tuple(tensor_type{{{1,2},{3,4}},{{5,6},{7,8}}}, std::vector<dim_type>{0,1,2,0}, sum{}, false)
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto keep_dim = std::get<3>(t);
        REQUIRE_THROWS_AS(reduce(tensor, axes, functor, keep_dim), reduce_exception);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_reduce_flatten","[test_reduce]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using test_reduce_::sum;
    using test_reduce_::prod;
    using test_reduce_::max;
    using test_reduce_::min;
    using gtensor::reduce_flatten;
    using helpers_for_testing::apply_by_element;
    //0tensor,1functor,2keep_dims,3expected
    auto test_data = std::make_tuple(
        //keep_dims is false
        std::make_tuple(tensor_type{}, sum{}, false, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}, prod{}, false, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{}.reshape(1,0), sum{}, false, tensor_type(value_type{0})),
        std::make_tuple(tensor_type{}.reshape(1,0), prod{}, false, tensor_type(value_type{1})),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, sum{}, false, tensor_type(21)),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, prod{}, false, tensor_type(720)),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, false, tensor_type(9)),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, false, tensor_type(0)),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, sum{}, false, tensor_type(28)),
        //keep_dims is true
        std::make_tuple(tensor_type{}, sum{}, true, tensor_type{value_type{0}}),
        std::make_tuple(tensor_type{}, prod{}, true, tensor_type{value_type{1}}),
        std::make_tuple(tensor_type{}.reshape(2,1,0), sum{}, true, tensor_type{{{value_type{0}}}}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), prod{}, true, tensor_type{{{value_type{1}}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, sum{}, true, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, sum{}, true, tensor_type{{21}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, prod{}, true, tensor_type{{720}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, max{}, true, tensor_type{{9}}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, min{}, true, tensor_type{{0}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, sum{}, true, tensor_type{{{28}}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto functor = std::get<1>(t);
        auto keep_dims = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = reduce_flatten(tensor, functor, keep_dims);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}


TEST_CASE("test_slide","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using test_reduce_::cumprod_reverse;
    using test_reduce_::cumsum;
    using test_reduce_::diff_1;
    using test_reduce_::diff_2;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2functor,3window_size,4window_step,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, dim_type{0}, cumsum{}, index_type{0}, index_type{1}, tensor_type{}),
        std::make_tuple(tensor_type{}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), dim_type{1}, cumsum{}, index_type{5}, index_type{1}, tensor_type{}.reshape(0,2,3)),
        std::make_tuple(tensor_type{1}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{120,120,60,20,5}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, dim_type{0}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, dim_type{0}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{1,2,3},{5,7,9},{12,15,18}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}, tensor_type{{1,3,6},{4,9,15},{7,15,24}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{28,80,162},{28,40,54},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{6,6,3},{120,30,6},{504,72,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{-2}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{28,80,162},{28,40,54},{7,8,9}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{-1}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{{6,6,3},{120,30,6},{504,72,9}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto window_size = std::get<3>(t);
        auto window_step = std::get<4>(t);
        auto expected = std::get<5>(t);
        auto result = slide(tensor, axis, functor, window_size, window_step);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEMPLATE_TEST_CASE("test_slide_flatten","[test_reduce]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using order = TestType;
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type,order>;
    using index_type = typename tensor_type::index_type;
    using test_reduce_::cumprod_reverse;
    using test_reduce_::cumsum;
    using test_reduce_::diff_1;
    using test_reduce_::diff_2;
    using gtensor::slide_flatten;
    using helpers_for_testing::apply_by_element;

    //0tensor,1functor,2window_size,3window_step,4expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{1,2,3,4,5}, cumprod_reverse{}, index_type{1}, index_type{1}, tensor_type{120,120,60,20,5}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, cumsum{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15,21,28,36,45}),
        std::make_tuple(tensor_type{{1,3,2},{5,7,4},{6,7,8}}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{{1,3,2},{5,7,4},{6,7,8}}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto functor = std::get<1>(t);
        auto window_size = std::get<2>(t);
        auto window_step = std::get<3>(t);
        auto expected = std::get<4>(t);
        auto result = slide_flatten(tensor, functor, window_size, window_step);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_slide_custom_arg","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using test_reduce_::moving_avarage;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2functor,3window_size,4window_step,5denom,6expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10}, dim_type{0}, moving_avarage{}, index_type{3}, index_type{1}, value_type{3}, tensor_type{2,3,4,5,6,7,8,9}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10}, dim_type{0}, moving_avarage{}, index_type{3}, index_type{2}, value_type{3}, tensor_type{2,4,6,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, moving_avarage{}, index_type{2}, index_type{1}, value_type{2}, tensor_type{{2,3,4},{5,6,7}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, moving_avarage{}, index_type{2}, index_type{1}, value_type{2}, tensor_type{{1,2},{4,5},{7,8}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto window_size = std::get<3>(t);
        auto window_step = std::get<4>(t);
        auto denom = std::get<5>(t);
        auto expected = std::get<6>(t);
        auto result = slide(tensor, axis, functor, window_size, window_step, window_size, window_step, denom);
        REQUIRE(result == expected);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_slide_exception","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using gtensor::reduce_exception;
    using test_reduce_::cumsum;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2functor,3window_size,4window_step
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(0), dim_type{0}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{}.reshape(0,2,3), dim_type{3}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{1}, dim_type{0}, cumsum{}, index_type{2}, index_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{1}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumsum{}, index_type{6}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{2}, cumsum{}, index_type{1}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, cumsum{}, index_type{3}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, cumsum{}, index_type{4}, index_type{1})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto window_size = std::get<3>(t);
        auto window_step = std::get<4>(t);
        REQUIRE_THROWS_AS(slide(tensor, axis, functor, window_size, window_step), reduce_exception);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_slide_flatten_exception","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using index_type = typename tensor_type::index_type;
    using gtensor::reduce_exception;
    using test_reduce_::cumsum;
    using gtensor::slide_flatten;
    using helpers_for_testing::apply_by_element;

    //0tensor,1functor,2window_size,3window_step
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type(0), cumsum{}, index_type{2}, index_type{1}),
        std::make_tuple(tensor_type{1}, cumsum{}, index_type{2}, index_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, cumsum{}, index_type{6}, index_type{1}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, cumsum{}, index_type{7}, index_type{1})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto functor = std::get<1>(t);
        auto window_size = std::get<2>(t);
        auto window_step = std::get<3>(t);
        REQUIRE_THROWS_AS(slide_flatten(tensor, functor, window_size, window_step), reduce_exception);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_transform","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sort;
    using gtensor::transform;
    using helpers_for_testing::apply_by_element;

    //0tensor,1axis,2functor,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, dim_type{0}, sort{}, tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,3,2,1,0}, dim_type{0}, sort{}, tensor_type{0,1,1,2,2,3,3}),
        std::make_tuple(tensor_type{{2,1,3},{3,0,1}}, dim_type{0}, sort{}, tensor_type{{2,0,1},{3,1,3}}),
        std::make_tuple(tensor_type{{2,1,3},{3,0,1}}, dim_type{1}, sort{}, tensor_type{{1,2,3},{0,1,3}}),
        std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{0}, sort{}, tensor_type{{{0,1,1},{3,0,1}},{{2,2,3},{3,0,1}}}),
        std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{1}, sort{}, tensor_type{{{2,0,1},{3,1,3}},{{0,0,1},{3,2,1}}}),
        std::make_tuple(tensor_type{{{2,1,3},{3,0,1}},{{0,2,1},{3,0,1}}}, dim_type{2}, sort{}, tensor_type{{{1,2,3},{0,1,3}},{{0,1,2},{0,1,3}}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto axis = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto expected = std::get<3>(t);
        transform(tensor, axis, functor);
        REQUIRE(tensor == expected);
    };
    apply_by_element(test, test_data);
}

