#include "catch.hpp"
#include "./catch2/trompeloeil.hpp"
#include <iostream>
#include <initializer_list>
#include <array>
#include <tuple>
#include "slice.hpp"

namespace test_slice_{

template<typename IdxT>
struct slice_item{
    using index_type = IdxT;
    slice_item():nop{1}{}
    slice_item(const index_type& i_):i{i_}{}
    const index_type i{0};
    const char nop{0};
};

using value_type = float;
using index_type = typename gtensor::config::default_config<value_type>::index_type;
using slice_item_type =  typename test_slice_::slice_item<index_type>;
using slice_init_type = typename std::initializer_list<slice_item_type>;

void f(slice_init_type l){}

template<typename...Args, std::enable_if_t<std::conjunction_v<std::is_convertible<Args,slice_init_type>...>,int> =0 >
void g(Args...args){}

template<typename...Args, std::enable_if_t<std::conjunction_v<std::is_same<Args,slice_init_type>...>,int> =0 >
void t(Args...args){}

template<typename...Ts, std::enable_if_t<std::conjunction_v<std::is_convertible<Ts,slice_item_type>...>,int> =0 >
void u(std::initializer_list<Ts>...args){}

void v(std::initializer_list<slice_init_type> l){}

template<typename...Args>
void h(Args...args){}

}



TEST_CASE("signed_unsigned","signed_unsigned"){
    std::size_t ui{10};
    std::ptrdiff_t i{-20};
    std::ptrdiff_t i1{-2};
    std::ptrdiff_t i2{-10};
    auto res = ui+i;
    auto res_ = ui+static_cast<std::size_t>(i);
    auto res1 = ui+i1;
    auto res1_ = ui+static_cast<std::size_t>(i1);
    auto res2 = ui+i2;

    //std::cout<<static_cast<std::size_t>(i)<<" "<<static_cast<std::size_t>(i1)<<std::endl;
    //std::cout<<res<<" "<<res_<<" "<<res1<<" "<<res1_<<" "<<res2;
}

TEST_CASE("slice_init_type","[slice_init_type]"){
    using test_slice_::slice_init_type;
    using test_slice_::f;
    using test_slice_::g;
    using test_slice_::h;
    using test_slice_::t;
    using test_slice_::u;
    using test_slice_::v;
    slice_init_type l{0,0,1};
    f(l);
    f({0,0,1});
    f({});
    //g({0,0,1});
    //h({0,0,1});
    //t({0,0,1});
    u({0,{},1});
    v({{1,2,3},{},{0,{},-1}});
    v({{},{},{1,2,3,4}});
    g(l);
    
}

TEST_CASE("slice_item","[slice_item]"){
    using value_type = float;
    using index_type = typename gtensor::config::default_config<value_type>::index_type;
    using slice_item_type =  typename test_slice_::slice_item<index_type>;
    SECTION("use_init_list"){
        using slice_init_type = typename std::initializer_list<slice_item_type>;
        slice_init_type l = {};
        
        slice_init_type l1 = {1};
        REQUIRE(l1.begin()[0].i == 1);
        REQUIRE(!l1.begin()[0].nop);
        
        slice_init_type l2 = {0,-1,1};
        REQUIRE(l2.begin()[0].i == 0);
        REQUIRE(!l2.begin()[0].nop);
        REQUIRE(l2.begin()[1].i == -1);
        REQUIRE(!l2.begin()[1].nop);
        REQUIRE(l2.begin()[2].i == 1);
        REQUIRE(!l2.begin()[2].nop);
        
        slice_init_type l3 = {{},{},-1};    
        REQUIRE(l3.begin()[0].nop);
        REQUIRE(l3.begin()[1].nop);
        REQUIRE(l3.begin()[2].i == -1);
        REQUIRE(!l3.begin()[2].nop);
    }
    SECTION("use_array"){
        using slice_init_type = typename slice_item_type[3];        
        /*
        slice_init_type l_{{},{},{},{}};
        */

        slice_init_type l = {};
        REQUIRE(l[0].nop);
        REQUIRE(l[1].nop);
        REQUIRE(l[2].nop);

        slice_init_type l1 = {1};
        REQUIRE(l1[0].i == 1);
        REQUIRE(!l1[0].nop);
        REQUIRE(l1[1].nop);
        REQUIRE(l1[2].nop);

        slice_init_type l2 = {0,-1,1};
        REQUIRE(l2[0].i == 0);
        REQUIRE(!l2[0].nop);
        REQUIRE(l2[1].i == -1);
        REQUIRE(!l2[1].nop);
        REQUIRE(l2[2].i == 1);
        REQUIRE(!l2[2].nop);
        
        slice_init_type l3 = {{},{},-1};    
        REQUIRE(l3[0].nop);
        REQUIRE(l3[1].nop);
        REQUIRE(l3[2].i == -1);
        REQUIRE(!l3[2].nop);
    }
    SECTION("use_std_array"){
        using slice_init_type = typename std::array<slice_item_type,3>;
        slice_init_type l = {};
        REQUIRE(l[0].nop);
        REQUIRE(l[1].nop);
        REQUIRE(l[2].nop);

        slice_init_type l1 = {1};
        REQUIRE(l1[0].i == 1);
        REQUIRE(!l1[0].nop);
        REQUIRE(l1[1].nop);
        REQUIRE(l1[2].nop);

        slice_init_type l2 = {0,-1,1};
        REQUIRE(l2[0].i == 0);
        REQUIRE(!l2[0].nop);
        REQUIRE(l2[1].i == -1);
        REQUIRE(!l2[1].nop);
        REQUIRE(l2[2].i == 1);
        REQUIRE(!l2[2].nop);

        /*
        slice_init_type l3 = {{},{},-1};    
        REQUIRE(l3[0].nop);
        REQUIRE(l3[1].nop);
        REQUIRE(l3[2].i == -1);
        REQUIRE(!l3[2].nop);
        */        
    }
    SECTION("use_tuple"){
        using slice_init_type = typename std::tuple<slice_item_type,slice_item_type,slice_item_type>;
        slice_init_type l = {};
        REQUIRE(std::get<0>(l).nop);
        REQUIRE(std::get<1>(l).nop);
        REQUIRE(std::get<2>(l).nop);
        
        /*
        slice_init_type l1 = {1};
        */

        slice_init_type l1 = {1,{},{}};
        REQUIRE(!std::get<0>(l1).nop);
        REQUIRE(std::get<0>(l1).i == 1);
        REQUIRE(std::get<1>(l1).nop);
        REQUIRE(std::get<2>(l1).nop);
    }     
}

TEST_CASE("test_slice","[test_slice]"){
    using slice_type = typename gtensor::slice<std::ptrdiff_t>;
    using nop_type = gtensor::config::NOP;
    nop_type nop{};
    SECTION("default_construction"){
        slice_type slice{};
        REQUIRE(!slice.is_start());
        REQUIRE(!slice.is_stop());
        REQUIRE(slice.is_step());
    }
    SECTION("construction_i__"){
        slice_type slice{-3};
        REQUIRE(slice.start == -3);        
        REQUIRE(slice.is_start());
        REQUIRE(!slice.is_stop());
        REQUIRE(slice.is_step());
        slice_type slice1{0,nop,nop};
        REQUIRE(slice1.start == 0);        
        REQUIRE(slice1.is_start());
        REQUIRE(!slice1.is_stop());
        REQUIRE(slice1.is_step());
        slice_type slice2 = {3,{},{}};
        REQUIRE(slice2.start == 3);        
        REQUIRE(slice2.is_start());
        REQUIRE(!slice2.is_stop());
        REQUIRE(slice2.is_step());
    }
    SECTION("construction_ij_"){
        slice_type slice{3,10};
        REQUIRE(slice.start == 3);
        REQUIRE(slice.stop == 10);
        REQUIRE(slice.is_start());
        REQUIRE(slice.is_stop());
        REQUIRE(slice.is_step());
        slice_type slice1{2,14,nop};
        REQUIRE(slice1.is_start());
        REQUIRE(slice1.is_stop());
        REQUIRE(slice1.is_step());
        REQUIRE(slice1.start == 2);
        REQUIRE(slice1.stop == 14);
    }
    SECTION("construction_ijk"){
        slice_type slice{3,10,-2};
        REQUIRE(slice.start == 3);
        REQUIRE(slice.stop == 10);
        REQUIRE(slice.step == -2);
        REQUIRE(slice.is_start());
        REQUIRE(slice.is_stop());
        REQUIRE(slice.is_step());        
    }
    SECTION("construction___k"){
        slice_type slice{{},{nop},-2};
        REQUIRE(slice.step == -2);
        REQUIRE(!slice.is_start());
        REQUIRE(!slice.is_stop());
        REQUIRE(slice.is_step());        
    }

}

TEST_CASE("test_fill_slice","[test_slice]"){
    using index_type = std::ptrdiff_t;
    using slice_type = typename gtensor::slice<index_type>;
    using nop_type = gtensor::config::NOP;
    nop_type nop{};
    using test_type = std::tuple<slice_type,slice_type,index_type>;
    auto test_data = GENERATE(
        test_type{slice_type(),slice_type(0,11,1),11},
        test_type{slice_type(3),slice_type(3,11,1),11},
        test_type{slice_type(2),slice_type(2,5,1),5},
        test_type{slice_type(0,3),slice_type(0,3,1),15},
        test_type{slice_type(1,9,2),slice_type(1,9,2),15},
        test_type{slice_type(-10,nop_type{},2),slice_type(5,15,2),15},
        test_type{slice_type(nop_type{},nop_type{},-1),slice_type(10,-1,-1),11},
        test_type{slice_type(-1,-4,-1),slice_type(10,7,-1),11},
        test_type{slice_type(7,nop_type{},-2),slice_type(7,-1,-2),11},
        test_type{slice_type(4,0,-1),slice_type(4,0,-1),11},
        test_type{slice_type(4,0,-1),slice_type(4,0,-1),11}
        );
    REQUIRE(gtensor::detail::fill_slice(std::get<0>(test_data),std::get<2>(test_data)) == std::get<1>(test_data));    
}

TEST_CASE("test_fill_slices","[test_fill_slices]"){
    using gtensor::detail::fill_slices;
    using value_type = float;
    using nop_type = gtensor::config::NOP;
    using difference_type = typename gtensor::config::default_config<value_type>::difference_type;
    using shape_type = typename gtensor::config::default_config<value_type>::shape_type;
    //using slice_item_type =  typename test_slice_::slice_item<difference_type>;
    using slice_item_type =  typename gtensor::detail::slice_item<difference_type, nop_type>;
    using slice_init_type = typename std::initializer_list<slice_item_type>;
    using slice_type = typename gtensor::slice<difference_type>;
    using vec_type = typename std::vector<slice_type>;
    nop_type nop{};

    SECTION("init_list"){
        REQUIRE(fill_slices<slice_type>(shape_type{2,3,4},{}) == std::vector<slice_type>{} );
        REQUIRE(fill_slices<slice_type>(shape_type{2,3,4},{{}}) == std::vector<slice_type>{{0,4,1}} );
        REQUIRE(fill_slices<slice_type>(shape_type{2,3,4},{{},{},{}}) == std::vector<slice_type>{{0,4,1},{0,3,1},{0,2,1}} );
        REQUIRE(fill_slices<slice_type>(shape_type{2,3,4},{{1,3,{nop}},{},{}}) == std::vector<slice_type>{{1,3,1},{0,3,1},{0,2,1}} );
        REQUIRE(fill_slices<slice_type>(shape_type{4,3},{{nop,nop,-1},{-4,nop,-1}}) == std::vector<slice_type>{{2,-1,-1},{0,-1,-1}});        
        REQUIRE(fill_slices<slice_type>(shape_type{4,3,2},{{},{1},{1,3}}) == std::vector<slice_type>{{0,2,1},{1,3,1},{1,3,1}});        
    }
    SECTION("variadic"){
        REQUIRE(fill_slices<slice_type>(shape_type{2,3,4}) == std::array<slice_type,0>{} );
        REQUIRE(fill_slices<slice_type>(shape_type{2,3,4},slice_type{}) == std::array<slice_type,1>{slice_type{0,4,1}} );
        REQUIRE(fill_slices<slice_type>(shape_type{2,3,4},slice_type{},slice_type{},slice_type{}) == std::array<slice_type,3>{slice_type{0,4,1},slice_type{0,3,1},slice_type{0,2,1}} );
        REQUIRE(fill_slices<slice_type>(shape_type{2,3,4},slice_type{1,3,{nop}},slice_type{},slice_type{}) == std::array<slice_type,3>{slice_type{1,3,1},slice_type{0,3,1},slice_type{0,2,1}} );
        REQUIRE(fill_slices<slice_type>(shape_type{4,3},slice_type{nop,nop,-1},slice_type{-4,nop,-1}) == std::array<slice_type,2>{slice_type{2,-1,-1},slice_type{0,-1,-1}});        
    }
}

TEST_CASE("test_check_slice","[test_check_slice]"){
    using value_type = float;
    using nop_type = gtensor::config::NOP;
    using difference_type = typename gtensor::config::default_config<value_type>::difference_type;
    using shape_type = typename gtensor::config::default_config<value_type>::shape_type;
    using slice_type = typename gtensor::slice<difference_type>;
    using gtensor::detail::check_slice;
    nop_type nop{};

    REQUIRE_NOTHROW(check_slice(slice_type{0,5,1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{0,5,1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{0,5,1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{0,5,1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{4,-1,-1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{0,-1,-1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{0,3,1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{2,5,1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{1,4,1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{1,4,1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{3,4,1}, difference_type(5)));
    REQUIRE_NOTHROW(check_slice(slice_type{0,-1,-1}, difference_type(5)));

    REQUIRE_THROWS_AS(check_slice(slice_type{5,5,1},difference_type(5)), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slice(slice_type{6,5,1},difference_type(5)), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slice(slice_type{-1,5,1},difference_type(5)), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slice(slice_type{1,-1,1},difference_type(5)), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slice(slice_type{0,0,1},difference_type(5)), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slice(slice_type{0,6,1},difference_type(5)), gtensor::subscript_exception);    
    REQUIRE_THROWS_AS(check_slice(slice_type{2,0,1},difference_type(5)), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slice(slice_type{5,5,-1},difference_type(5)), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slice(slice_type{-1,5,-1},difference_type(5)), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slice(slice_type{0,4,-1},difference_type(5)), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slice(slice_type{0,5,-1},difference_type(5)), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slice(slice_type{1,4,-1},difference_type(5)), gtensor::subscript_exception);
}

TEST_CASE("test_is_slices", "[test_is_slices]"){
    using value_type = float;
    using nop_type = gtensor::config::NOP;
    using difference_type = typename gtensor::config::default_config<value_type>::difference_type;
    using shape_type = typename gtensor::config::default_config<value_type>::shape_type;
    using slice_type = typename gtensor::slice<difference_type>;
    using gtensor::detail::is_slice;
    using gtensor::detail::is_slices;
    nop_type nop{};

    REQUIRE(is_slice<slice_type>);    
    REQUIRE(!is_slice<shape_type>);
    REQUIRE(!is_slice<difference_type>);

    REQUIRE(is_slices<>);
    REQUIRE(is_slices<slice_type>);
    REQUIRE(is_slices<slice_type,slice_type,slice_type>);
    REQUIRE(!is_slices<shape_type>);
    REQUIRE(!is_slices<difference_type>);
    REQUIRE(!is_slices<slice_type,difference_type,slice_type>);
}

TEST_CASE("test_check_slices","[test_check_slices]"){
    using gtensor::detail::check_slices;
    using value_type = float;
    using nop_type = gtensor::config::NOP;
    using difference_type = typename gtensor::config::default_config<value_type>::difference_type;
    using shape_type = typename gtensor::config::default_config<value_type>::shape_type;
    using slice_type = typename gtensor::slice<difference_type>;
    REQUIRE_NOTHROW(check_slices(shape_type{5},std::vector<slice_type>{}));
    REQUIRE_NOTHROW(check_slices(shape_type{5},std::vector{slice_type{0,5,1}}));
    REQUIRE_NOTHROW(check_slices(shape_type{3,4,5},std::vector{slice_type{0,5,1}}));
    REQUIRE_NOTHROW(check_slices(shape_type{3,4,5},std::vector{slice_type{0,5,1}, slice_type{0,4,1}}));
    REQUIRE_NOTHROW(check_slices(shape_type{3,4,5},std::vector{slice_type{0,5,1}, slice_type{0,4,1}, slice_type{2,-1,-1}}));
    
    REQUIRE_THROWS_AS(check_slices(shape_type{5},std::vector{slice_type{0,5,1}, slice_type{0,5,1}}), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slices(shape_type{5},std::vector{slice_type{0,-1,1}}), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slices(shape_type{3,4,5},std::vector{slice_type{0,5,1}, slice_type{0,5,1}}), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_slices(shape_type{3,4,5},std::vector{slice_type{0,5,1}, slice_type{1,1,1}, slice_type{0,3,1}}), gtensor::subscript_exception);
}

TEST_CASE("test_check_subdim_subs","[test_check_subdim_subs]"){
    using gtensor::detail::check_subdim_subs;
    using value_type = float;
    using nop_type = gtensor::config::NOP;
    using difference_type = typename gtensor::config::default_config<value_type>::difference_type;
    using shape_type = typename gtensor::config::default_config<value_type>::shape_type;    
    REQUIRE_NOTHROW(check_subdim_subs(shape_type{3,4,5},4));
    REQUIRE_NOTHROW(check_subdim_subs(shape_type{3,4,5},4,3));
    
    REQUIRE_THROWS_AS(check_subdim_subs(shape_type{5},0), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_subdim_subs(shape_type{5},0,0), gtensor::subscript_exception);    
    REQUIRE_THROWS_AS(check_subdim_subs(shape_type{3,4,5},1,2,3), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_subdim_subs(shape_type{3,4,5},5), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_subdim_subs(shape_type{3,4,5},0,4), gtensor::subscript_exception);
}

TEST_CASE("test_check_reshape_subs","[test_check_reshape_subs]"){
    using gtensor::detail::check_reshape_subs;
    using value_type = float;
    using nop_type = gtensor::config::NOP;
    using difference_type = typename gtensor::config::default_config<value_type>::difference_type;
    using shape_type = typename gtensor::config::default_config<value_type>::shape_type;    
    REQUIRE_NOTHROW(check_reshape_subs(5));
    REQUIRE_NOTHROW(check_reshape_subs(5, 5));
    REQUIRE_NOTHROW(check_reshape_subs(20, 4,5));
    REQUIRE_NOTHROW(check_reshape_subs(20, 20));
    REQUIRE_NOTHROW(check_reshape_subs(20, 1,20,1));
    REQUIRE_NOTHROW(check_reshape_subs(20, 2,10));
    REQUIRE_NOTHROW(check_reshape_subs(20, 2,5,2,1));
    REQUIRE_NOTHROW(check_reshape_subs(20, 4,5));
    
    REQUIRE_THROWS_AS(check_reshape_subs(5, 0), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_reshape_subs(5, 5,0), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_reshape_subs(5, 3,2), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_reshape_subs(60, 70), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_reshape_subs(60, 2,3,2,4), gtensor::subscript_exception);
}

TEST_CASE("test_check_transpose_subs","[test_check_transpose_subs]"){
    using gtensor::detail::check_transpose_subs;
    using value_type = float;
    using nop_type = gtensor::config::NOP;
    using difference_type = typename gtensor::config::default_config<value_type>::difference_type;
    using shape_type = typename gtensor::config::default_config<value_type>::shape_type;    
    REQUIRE_NOTHROW(check_transpose_subs(1));
    REQUIRE_NOTHROW(check_transpose_subs(1,0));
    REQUIRE_NOTHROW(check_transpose_subs(3,0,1,2));
    REQUIRE_NOTHROW(check_transpose_subs(3,0,2,1));
    REQUIRE_NOTHROW(check_transpose_subs(3,2,1,0));
    REQUIRE_NOTHROW(check_transpose_subs(3,2,0,1));
    REQUIRE_NOTHROW(check_transpose_subs(3,1,0,2));
    REQUIRE_NOTHROW(check_transpose_subs(3,1,0,2));
    
    REQUIRE_THROWS_AS(check_transpose_subs(1,0,0), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_transpose_subs(2,0,0), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_transpose_subs(3,0,1,2,3), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_transpose_subs(3,0,1,1), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_transpose_subs(3,0,2,2), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_transpose_subs(3,0,0,1), gtensor::subscript_exception);
    REQUIRE_THROWS_AS(check_transpose_subs(3,2,1,1), gtensor::subscript_exception);
}

