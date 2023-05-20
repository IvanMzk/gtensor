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

    //single reduce direction
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

    //container of directions
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
    //0pshape,1direction,2expected
    using test_type = std::tuple<shape_type,dim_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{0},dim_type{0},shape_type{}},
        test_type{shape_type{1},dim_type{0},shape_type{}},
        test_type{shape_type{10},dim_type{0},shape_type{}},
        test_type{shape_type{2,3,0},dim_type{0},shape_type{3,0}},
        test_type{shape_type{2,3,0},dim_type{1},shape_type{2,0}},
        test_type{shape_type{2,3,0},dim_type{2},shape_type{2,3}},
        test_type{shape_type{2,3,4},dim_type{0},shape_type{3,4}},
        test_type{shape_type{2,3,4},dim_type{1},shape_type{2,4}},
        test_type{shape_type{2,3,4},dim_type{2},shape_type{2,3}}
    );
    auto pshape = std::get<0>(test_data);
    auto direction = std::get<1>(test_data);
    auto expected = std::get<2>(test_data);
    auto result = make_reduce_shape(pshape,direction);
    REQUIRE(result == expected);
}

TEST_CASE("test_check_slide_args","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;
    using gtensor::reduce_exception;
    using gtensor::detail::check_slide_args;

    REQUIRE_NOTHROW(check_slide_args(shape_type{0},dim_type{0},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{0},dim_type{0},index_type{2}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{1},dim_type{0},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{10},dim_type{0},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{10},dim_type{0},index_type{2}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{10},dim_type{0},index_type{5}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{10},dim_type{0},index_type{10}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{1,0},dim_type{0},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,0},dim_type{0},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,0},dim_type{0},index_type{2}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,0},dim_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,0},dim_type{1},index_type{2}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,0},dim_type{1},index_type{3}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,4},dim_type{0},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,4},dim_type{0},index_type{2}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,4},dim_type{1},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,4},dim_type{1},index_type{2}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,4},dim_type{1},index_type{3}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,4},dim_type{2},index_type{1}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,4},dim_type{2},index_type{2}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,4},dim_type{2},index_type{3}));
    REQUIRE_NOTHROW(check_slide_args(shape_type{2,3,4},dim_type{2},index_type{4}));

    REQUIRE_THROWS_AS(check_slide_args(shape_type{},dim_type{0},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(shape_type{0},dim_type{1},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(shape_type{10},dim_type{1},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(shape_type{10},dim_type{0},index_type{11}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(shape_type{2,3,4},dim_type{3},index_type{1}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(shape_type{2,3,4},dim_type{0},index_type{3}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(shape_type{2,3,4},dim_type{1},index_type{4}), reduce_exception);
    REQUIRE_THROWS_AS(check_slide_args(shape_type{2,3,4},dim_type{2},index_type{5}), reduce_exception);
}

TEST_CASE("test_make_slide_shape","[test_reduce]")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config,int>;
    using dim_type = config_type::dim_type;
    using index_type = config_type::index_type;
    using shape_type = config_type::shape_type;
    using gtensor::detail::make_slide_shape;
    //0pshape,1direction,2window_size,3window_step,4expected
    using test_type = std::tuple<shape_type,dim_type,index_type,index_type,shape_type>;
    auto test_data = GENERATE(
        test_type{shape_type{0},dim_type{0},index_type{1},index_type{1},shape_type{0}},
        test_type{shape_type{0},dim_type{0},index_type{2},index_type{1},shape_type{0}},
        test_type{shape_type{0},dim_type{0},index_type{1},index_type{2},shape_type{0}},
        test_type{shape_type{20,30,0},dim_type{0},index_type{5},index_type{2},shape_type{8,30,0}},
        test_type{shape_type{20,30,0},dim_type{1},index_type{5},index_type{2},shape_type{20,13,0}},
        test_type{shape_type{20,30,0},dim_type{2},index_type{5},index_type{2},shape_type{20,30,0}},
        test_type{shape_type{1},dim_type{0},index_type{1},index_type{1},shape_type{1}},
        test_type{shape_type{1},dim_type{0},index_type{1},index_type{2},shape_type{1}},
        test_type{shape_type{10},dim_type{0},index_type{1},index_type{1},shape_type{10}},
        test_type{shape_type{10},dim_type{0},index_type{1},index_type{2},shape_type{5}},
        test_type{shape_type{10},dim_type{0},index_type{1},index_type{5},shape_type{2}},
        test_type{shape_type{10},dim_type{0},index_type{2},index_type{1},shape_type{9}},
        test_type{shape_type{10},dim_type{0},index_type{2},index_type{2},shape_type{5}},
        test_type{shape_type{10},dim_type{0},index_type{2},index_type{5},shape_type{2}},
        test_type{shape_type{10},dim_type{0},index_type{5},index_type{1},shape_type{6}},
        test_type{shape_type{10},dim_type{0},index_type{5},index_type{2},shape_type{3}},
        test_type{shape_type{10},dim_type{0},index_type{5},index_type{5},shape_type{2}},
        test_type{shape_type{5,30,40},dim_type{0},index_type{3},index_type{2},shape_type{2,30,40}},
        test_type{shape_type{5,30,40},dim_type{1},index_type{5},index_type{1},shape_type{5,26,40}},
        test_type{shape_type{5,30,40},dim_type{2},index_type{10},index_type{3},shape_type{5,30,11}}
    );
    auto pshape = std::get<0>(test_data);
    auto direction = std::get<1>(test_data);
    auto window_size = std::get<2>(test_data);
    auto window_step = std::get<3>(test_data);
    auto expected = std::get<4>(test_data);
    auto result = make_slide_shape(pshape,direction,window_size,window_step);
    REQUIRE(result == expected);
}

namespace test_reduce_{

struct max
{
    template<typename It>
    auto operator()(It first, It last){
        const auto& init = *first;
        return std::accumulate(++first,last,init, [](const auto& u, const auto& v){return std::max(u,v);});
    }
};
struct min
{
    template<typename It>
    auto operator()(It first, It last){
        const auto& init = *first;
        return std::accumulate(++first,last,init, [](const auto& u, const auto& v){return std::min(u,v);});
    }
};
struct sum
{
    template<typename It>
    auto operator()(It first, It last){
        const auto& init = *first;
        return std::accumulate(++first,last,init,std::plus{});
    }
};
struct prod
{
    template<typename It>
    auto operator()(It first, It last){
        const auto& init = *first;
        return std::accumulate(++first,last,init,std::multiplies{});
    }
};

template<typename T>
struct cumsum{
    T cumsum_{0};
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, IdxT,IdxT){
        cumsum_ = T(0);
        for(;dfirst!=dlast; ++dfirst,++first){
            cumsum_+=*first;
            *dfirst = cumsum_;
        }
    }
};

struct moving_avarage{
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, const IdxT& window_size, const IdxT& window_step, const typename std::iterator_traits<It>::value_type& denom){
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
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, const IdxT&, const IdxT&){
        for (;dfirst!=dlast;++dfirst){
            auto prev = *first;
            *dfirst = *(++first) - prev;
        }
    }
};

struct diff_2{
    template<typename It, typename DstIt, typename IdxT>
    void operator()(It first, It last, DstIt dfirst, DstIt dlast, const IdxT&, const IdxT&){
        for (;dfirst!=dlast;++dfirst){
            auto v0 = *first;
            auto v1 = *(++first);
            auto v2 = *(++first);
            *dfirst = v2-v1-v1+v0;
            --first;
        }
    }
};

}   //end of namespace test_reduce_

TEST_CASE("test_reduce","[test_reduce]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using test_reduce_::sum;
    using test_reduce_::prod;
    using test_reduce_::max;
    using test_reduce_::min;
    using gtensor::reduce;
    using helpers_for_testing::apply_by_element;
    //0tensor,1direction,2functor,3expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, dim_type{0}, sum{}, tensor_type(value_type{})),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{0}, sum{}, tensor_type{}),
        std::make_tuple(tensor_type{}.reshape(1,0), dim_type{1}, sum{}, tensor_type{value_type{}}),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{0}, sum{}, tensor_type{}.reshape(3,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{1}, sum{}, tensor_type{}.reshape(2,0)),
        std::make_tuple(tensor_type{}.reshape(2,3,0), dim_type{2}, sum{}, tensor_type{{value_type{},value_type{},value_type{}},{value_type{},value_type{},value_type{}}}),
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{0}, sum{}, tensor_type(21)),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{0}, sum{}, tensor_type{21}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{1}, sum{}, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{0}, sum{}, tensor_type{1,2,3,4,5,6}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{1}, sum{}, tensor_type{21}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{0}, sum{}, tensor_type{5,7,9}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, sum{}, tensor_type{6,15}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{1}, prod{}, tensor_type{6,120}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{0}, max{}, tensor_type{4,6,7,9}),
        std::make_tuple(tensor_type{{1,6,7,9},{4,5,7,0}}, dim_type{1}, min{}, tensor_type{1,0}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{0}, sum{}, tensor_type{{4,6},{8,10}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{1}, sum{}, tensor_type{{2,4},{10,12}}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{2}, sum{}, tensor_type{{1,5},{9,13}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto direction = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = reduce(tensor, direction, functor);
        REQUIRE(result == expected);
        auto result1 = reduce(tensor, direction, functor);
        REQUIRE(result1 == expected);
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


    //0tensor,1direction,2functor
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{1,2,3,4,5,6}, dim_type{1}, sum{}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5},{6}}, dim_type{2}, sum{}),
        std::make_tuple(tensor_type{{1,2,3,4,5,6}}, dim_type{2}, sum{}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6}}, dim_type{4}, sum{}),
        std::make_tuple(tensor_type{{{0,1},{2,3}},{{4,5},{6,7}}}, dim_type{3}, sum{})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto direction = std::get<1>(t);
        auto functor = std::get<2>(t);
        REQUIRE_THROWS_AS(reduce(tensor, direction, functor), reduce_exception);
    };
    apply_by_element(test, test_data);
}

TEST_CASE("test_slide","[test_reduce]")
{
    using value_type = int;
    using tensor_type = gtensor::tensor<value_type>;
    using dim_type = typename tensor_type::dim_type;
    using index_type = typename tensor_type::index_type;
    using test_reduce_::cumsum;
    using test_reduce_::diff_1;
    using test_reduce_::diff_2;
    using gtensor::slide;
    using helpers_for_testing::apply_by_element;

    //0tensor,1direction,2functor,3window_size,4window_step,5expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, dim_type{0}, cumsum<value_type>{}, index_type{1}, index_type{1}, tensor_type{}),
        std::make_tuple(tensor_type{1}, dim_type{0}, cumsum<value_type>{}, index_type{1}, index_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, cumsum<value_type>{}, index_type{1}, index_type{1}, tensor_type{1,3,6,10,15}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, dim_type{0}, diff_1{}, index_type{2}, index_type{1}, tensor_type{2,-1,3,2,-3,2,1,1}),
        std::make_tuple(tensor_type{1,3,2,5,7,4,6,7,8}, dim_type{0}, diff_2{}, index_type{3}, index_type{1}, tensor_type{-3,4,-1,-5,5,-1,0}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, cumsum<value_type>{}, index_type{1}, index_type{1}, tensor_type{{1,2,3},{5,7,9},{12,15,18}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, cumsum<value_type>{}, index_type{1}, index_type{1}, tensor_type{{1,3,6},{4,9,15},{7,15,24}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto direction = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto window_size = std::get<3>(t);
        auto window_step = std::get<4>(t);
        auto expected = std::get<5>(t);
        auto result = slide(tensor, direction, functor, window_size, window_step);
        REQUIRE(result == expected);
        auto result1 = slide(tensor, direction, functor, window_size, window_step);
        REQUIRE(result1 == expected);
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

    //0tensor,1direction,2functor,3window_size,4window_step,5denom,6expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{}),
        std::make_tuple(tensor_type{1}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{1}),
        std::make_tuple(tensor_type{1,2,3,4,5}, dim_type{0}, moving_avarage{}, index_type{1}, index_type{1}, value_type{1}, tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10}, dim_type{0}, moving_avarage{}, index_type{3}, index_type{1}, value_type{3}, tensor_type{2,3,4,5,6,7,8,9}),
        std::make_tuple(tensor_type{1,2,3,4,5,6,7,8,9,10}, dim_type{0}, moving_avarage{}, index_type{3}, index_type{2}, value_type{3}, tensor_type{2,4,6,8}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{0}, moving_avarage{}, index_type{2}, index_type{1}, value_type{2}, tensor_type{{2,3,4},{5,6,7}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9}}, dim_type{1}, moving_avarage{}, index_type{2}, index_type{1}, value_type{2}, tensor_type{{1,2},{4,5},{7,8}})
    );
    auto test = [](const auto& t){
        auto tensor = std::get<0>(t);
        auto direction = std::get<1>(t);
        auto functor = std::get<2>(t);
        auto window_size = std::get<3>(t);
        auto window_step = std::get<4>(t);
        auto denom = std::get<5>(t);
        auto expected = std::get<6>(t);
        auto result = slide(tensor, direction, functor, window_size, window_step,denom);
        REQUIRE(result == expected);
        auto result1 = slide(tensor, direction, functor, window_size, window_step,denom);
        REQUIRE(result1 == expected);
    };
    apply_by_element(test, test_data);
}
