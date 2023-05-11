#include <iostream>
#include "catch.hpp"
#include "tensor.hpp"

TEST_CASE("test_tmp","[test_tmp]"){
    using tensor_type = gtensor::tensor<double>;
    using index_type = typename tensor_type::index_type;
    using slice_type = typename tensor_type::slice_type;
    using slice_item_type = typename slice_type::slice_item_type;
    using reduce_tag_type = typename slice_type::reduce_tag_type;

    tensor_type t{{1,2,3},{4,5,6}};

    //auto v = t(index_type{1});
    //std::cout<<std::endl<<t(0);
    // std::cout<<std::endl<<t(slice_type{},0);
    // std::cout<<std::endl<<t(slice_type{},1);
    //std::cout<<std::endl<<t(slice_type{});

    struct slice_maker{
        auto operator()(const index_type& start){return slice_type{start,reduce_tag_type{}};}
        //auto operator()(std::initializer_list<slice_item_type> slice_init_list){return slice_type{slice_init_list};}
        auto operator()(std::initializer_list<slice_item_type> slice_init_list){return slice_type(slice_init_list);}
        const auto& operator()(const slice_type& slice){return slice;}
    };

    auto maker = slice_maker{};
    auto some_slice = slice_type{1,2,3};
    std::cout<<std::endl<<maker(2);
    std::cout<<std::endl<<maker(some_slice);

}