#include "catch.hpp"
#include "descriptor.hpp"
#include "data_accessor.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"

namespace test_walker_traverser{

template<typename Order, typename Tr, typename Idx>
auto do_next(Tr& tr, Idx n){
    bool result{true};
    while(n!=0){
        result = tr.template next<Order>();
        --n;
    }
    return result;
};
template<typename Order, typename Tr, typename Idx>
auto do_prev(Tr& tr, Idx n){
    bool result{true};
    while(n!=0){
        result = tr.template prev<Order>();
        --n;
    }
    return result;
};

}   //end of namespace test_walker_traverser

TEST_CASE("test_make_axes","test_data_accessor")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config, int>;
    using dim_type = config_type::dim_type;
    using axes_type = typename config_type::template shape<dim_type>;
    using gtensor::detail::make_axes;
    using helpers_for_testing::apply_by_element;
    //0dim,1axes,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{1},0,dim_type{0}),
        std::make_tuple(dim_type{1},-1,dim_type{0}),
        std::make_tuple(dim_type{3},0,dim_type{0}),
        std::make_tuple(dim_type{3},2,dim_type{2}),
        std::make_tuple(dim_type{3},-1,dim_type{2}),
        std::make_tuple(dim_type{3},-2,dim_type{1}),
        std::make_tuple(dim_type{3},std::vector<int>{},axes_type{}),
        std::make_tuple(dim_type{3},std::vector<int>{2},axes_type{2}),
        std::make_tuple(dim_type{3},std::vector<int>{1,-1},axes_type{1,2}),
        std::make_tuple(dim_type{3},std::vector<int>{2,1,0},axes_type{2,1,0})
    );
    auto test = [](const auto& t){
        auto dim = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto result = make_axes<config_type>(dim,axes);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_range_traverser_axes_map_axes_scalar","test_data_accessor")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config, int>;
    using dim_type = config_type::dim_type;
    using axes_type = typename config_type::template shape<dim_type>;
    using gtensor::detail::make_range_traverser_axes_map;
    using helpers_for_testing::apply_by_element;
    //0dim,1axes,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{1},0,axes_type{0}),
        std::make_tuple(dim_type{2},0,axes_type{0,1}),
        std::make_tuple(dim_type{2},1,axes_type{1,0}),
        std::make_tuple(dim_type{4},0,axes_type{0,1,2,3}),
        std::make_tuple(dim_type{4},1,axes_type{1,0,2,3}),
        std::make_tuple(dim_type{4},2,axes_type{2,1,0,3}),
        std::make_tuple(dim_type{4},3,axes_type{3,1,2,0})
    );
    auto test = [](const auto& t){
        auto dim = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto expected = std::get<2>(t);

        auto result = make_range_traverser_axes_map<config_type>(dim,axes);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_range_traverser_axes_map_axes_container","test_data_accessor")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config, int>;
    using dim_type = config_type::dim_type;
    using axes_type = typename config_type::template shape<dim_type>;
    using gtensor::detail::make_range_traverser_axes_map;
    using helpers_for_testing::apply_by_element;
    //0dim,1axes,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(dim_type{1},axes_type{0},axes_type{0}),
        std::make_tuple(dim_type{2},axes_type{0},axes_type{0,1}),
        std::make_tuple(dim_type{2},axes_type{1},axes_type{1,0}),
        std::make_tuple(dim_type{2},axes_type{0,1},axes_type{0,1}),
        std::make_tuple(dim_type{2},axes_type{1,0},axes_type{1,0}),
        std::make_tuple(dim_type{4},axes_type{0},axes_type{0,1,2,3}),
        std::make_tuple(dim_type{4},axes_type{1},axes_type{1,0,2,3}),
        std::make_tuple(dim_type{4},axes_type{2},axes_type{2,1,0,3}),
        std::make_tuple(dim_type{4},axes_type{0,1},axes_type{0,1,2,3}),
        std::make_tuple(dim_type{4},axes_type{1,0},axes_type{0,1,2,3}),
        std::make_tuple(dim_type{4},axes_type{1,2},axes_type{1,2,0,3}),
        std::make_tuple(dim_type{4},axes_type{2,1},axes_type{1,2,0,3}),
        std::make_tuple(dim_type{4},axes_type{0,3},axes_type{0,3,2,1}),
        std::make_tuple(dim_type{4},axes_type{3,0,2},axes_type{0,3,2,1}),
        std::make_tuple(dim_type{4},axes_type{3,1,2},axes_type{1,3,2,0}),
        std::make_tuple(dim_type{4},axes_type{0,3,1,2},axes_type{0,1,2,3})
    );
    auto test = [](const auto& t){
        auto dim = std::get<0>(t);
        auto axes = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_range_traverser_axes_map<config_type>(dim,axes);
        auto is_in_axes = [&axes](const auto& axis){
            const auto last = axes.end();
            return std::find_if(axes.begin(),last,[axis](const auto& a){return axis == a;}) != last;
        };
        REQUIRE(std::is_partitioned(result.begin(),result.end(),is_in_axes));
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_range_traverser_shape","test_data_accessor")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config, int>;
    using dim_type = config_type::dim_type;
    using shape_type = config_type::shape_type;
    using axes_type = typename config_type::template shape<dim_type>;
    using gtensor::detail::make_range_traverser_shape;
    using helpers_for_testing::apply_by_element;
    //0shape,1axes_map,2expected
    auto test_data = std::make_tuple(
        std::make_tuple(shape_type{},axes_type{},shape_type{}),
        std::make_tuple(shape_type{1},axes_type{0},shape_type{1}),
        std::make_tuple(shape_type{5},axes_type{0},shape_type{5}),
        std::make_tuple(shape_type{4,5},axes_type{0,1},shape_type{4,5}),
        std::make_tuple(shape_type{4,5},axes_type{1,0},shape_type{5,4}),
        std::make_tuple(shape_type{1,2,3,4,5},axes_type{0,1,2,3,4},shape_type{1,2,3,4,5}),
        std::make_tuple(shape_type{1,2,3,4,5},axes_type{1,0,4,3,2},shape_type{2,1,5,4,3})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto axes_map = std::get<1>(t);
        auto expected = std::get<2>(t);
        auto result = make_range_traverser_shape(shape,axes_map);
        REQUIRE(result==expected);
    };
    apply_by_element(test,test_data);
}

TEST_CASE("test_make_range_traverser_strides_div","test_data_accessor")
{
    using config_type = gtensor::config::extend_config_t<gtensor::config::default_config, int>;
    using strides_div_type = gtensor::detail::strides_div_t<config_type>;
    using divider_type = typename strides_div_type::value_type;
    using shape_type = config_type::shape_type;
    using dim_type = config_type::dim_type;
    using gtensor::config::c_order;
    using gtensor::config::f_order;
    using gtensor::detail::make_range_traverser_strides_div;
    using helpers_for_testing::apply_by_element;
    //0shape,1axes_size,2order,3expected
    auto test_data = std::make_tuple(
        //c_order
        std::make_tuple(shape_type{},dim_type{0},c_order{},strides_div_type{}),
        std::make_tuple(shape_type{1},dim_type{0},c_order{},strides_div_type{divider_type{1}}),
        std::make_tuple(shape_type{5},dim_type{0},c_order{},strides_div_type{divider_type{1}}),
        std::make_tuple(shape_type{1},dim_type{1},c_order{},strides_div_type{divider_type{1}}),
        std::make_tuple(shape_type{5},dim_type{1},c_order{},strides_div_type{divider_type{1}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{0},c_order{},strides_div_type{divider_type{60},divider_type{12},divider_type{4},divider_type{4},divider_type{1}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{1},c_order{},strides_div_type{divider_type{1},divider_type{12},divider_type{4},divider_type{4},divider_type{1}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{2},c_order{},strides_div_type{divider_type{5},divider_type{1},divider_type{4},divider_type{4},divider_type{1}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{3},c_order{},strides_div_type{divider_type{15},divider_type{3},divider_type{1},divider_type{4},divider_type{1}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{4},c_order{},strides_div_type{divider_type{15},divider_type{3},divider_type{1},divider_type{1},divider_type{1}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{5},c_order{},strides_div_type{divider_type{60},divider_type{12},divider_type{4},divider_type{4},divider_type{1}}),
        //f_order
        std::make_tuple(shape_type{},dim_type{0},f_order{},strides_div_type{}),
        std::make_tuple(shape_type{1},dim_type{0},f_order{},strides_div_type{divider_type{1}}),
        std::make_tuple(shape_type{5},dim_type{0},f_order{},strides_div_type{divider_type{1}}),
        std::make_tuple(shape_type{1},dim_type{1},f_order{},strides_div_type{divider_type{1}}),
        std::make_tuple(shape_type{5},dim_type{1},f_order{},strides_div_type{divider_type{1}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{0},f_order{},strides_div_type{divider_type{1},divider_type{2},divider_type{10},divider_type{30},divider_type{30}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{1},f_order{},strides_div_type{divider_type{1},divider_type{1},divider_type{5},divider_type{15},divider_type{15}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{2},f_order{},strides_div_type{divider_type{1},divider_type{2},divider_type{1},divider_type{3},divider_type{3}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{3},f_order{},strides_div_type{divider_type{1},divider_type{2},divider_type{10},divider_type{1},divider_type{1}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{4},f_order{},strides_div_type{divider_type{1},divider_type{2},divider_type{10},divider_type{30},divider_type{1}}),
        std::make_tuple(shape_type{2,5,3,1,4},dim_type{5},f_order{},strides_div_type{divider_type{1},divider_type{2},divider_type{10},divider_type{30},divider_type{30}})
    );
    auto test = [](const auto& t){
        auto shape = std::get<0>(t);
        auto axes_size = std::get<1>(t);
        auto order = std::get<2>(t);
        auto expected = std::get<3>(t);
        auto result = make_range_traverser_strides_div<config_type>(shape,axes_size,order);
        REQUIRE(result==expected);
    };
    apply_by_element(test,test_data);
}



// TEST_CASE("test_make_strides_div_predicate","test_data_accessor")
// {
//     using config_type = gtensor::config::extend_config_t<gtensor::config::default_config, int>;
//     using shape_type = config_type::shape_type;
//     using strides_div_type = gtensor::detail::strides_div_t<config_type>;
//     using divider_type = typename strides_div_type::value_type;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using gtensor::detail::make_strides_div_predicate;
//     using helpers_for_testing::apply_by_element;
//     //0shape,1predicate,2order,3expected
//     auto test_data = std::make_tuple(
//         std::make_tuple(shape_type{},[](auto){return true;},c_order{},strides_div_type{}),
//         std::make_tuple(shape_type{1},[](auto){return true;},c_order{},strides_div_type{divider_type{1}}),
//         std::make_tuple(shape_type{1},[](auto){return false;},c_order{},strides_div_type{}),
//         std::make_tuple(shape_type{2,3,4,5},[](auto){return true;},c_order{},strides_div_type{divider_type{60},divider_type{20},divider_type{5},divider_type{1}}),
//         std::make_tuple(shape_type{2,3,4,5},[](auto){return false;},c_order{},strides_div_type{}),
//         std::make_tuple(shape_type{2,3,4,5},[](auto i){return i==0;},c_order{},strides_div_type{divider_type{1}}),
//         std::make_tuple(shape_type{2,3,4,5},[](auto i){return i==3;},c_order{},strides_div_type{divider_type{1}}),
//         std::make_tuple(shape_type{2,3,4,5},[](auto i){return i==0 || i==3;},c_order{},strides_div_type{divider_type{5},divider_type{1}}),
//         std::make_tuple(shape_type{2,3,4,5},[](auto i){return i!=0;},c_order{},strides_div_type{divider_type{20},divider_type{5},divider_type{1}}),
//         std::make_tuple(shape_type{2,3,4,5},[](auto i){return i!=2;},c_order{},strides_div_type{divider_type{15},divider_type{5},divider_type{1}}),
//         std::make_tuple(shape_type{2,3,4,5},[](auto i){return i!=1 && i!=3;},c_order{},strides_div_type{divider_type{4},divider_type{1}})
//     );
//     auto test = [](const auto& t){
//         auto shape = std::get<0>(t);
//         auto predicate = std::get<1>(t);
//         auto order = std::get<2>(t);
//         auto expected = std::get<3>(t);

//         auto result = make_strides_div_predicate<config_type>(shape,predicate,order);
//         REQUIRE(result == expected);
//     };
//     apply_by_element(test,test_data);
// }

// TEST_CASE("test_walker_forward_traverser","test_data_accessor")
// {
//     using value_type = int;
//     using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
//     using gtensor::walker_forward_traverser;
//     using shape_type = typename config_type::shape_type;
//     using index_type = typename config_type::index_type;
//     using storage_type = typename config_type::template storage<value_type>;
//     using indexer_type = gtensor::basic_indexer<storage_type&>;
//     using walker_type = gtensor::indexer_walker<config_type, indexer_type>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using gtensor::detail::make_strides;
//     using gtensor::detail::make_adapted_strides;
//     using gtensor::detail::make_reset_strides;
//     using test_walker_traverser::do_next;
//     using helpers_for_testing::apply_by_element;

//     const auto storage_1d = storage_type{1,2,3,4};
//     const auto shape_1d = shape_type{4};

//     const auto storage_c = storage_type{1,2,3,4,5,6,7,8,9,10,11,12};
//     const auto storage_f = storage_type{1,7,3,9,5,11,2,8,4,10,6,12};

//     //0elements_order,1storage,2shape,3command,4expected_index,5expected_element,6expected_is_next
//     auto test_data = std::make_tuple(
//         //elements in c_order, 1-d
//         //no command
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto&){return true;}, shape_type{0} ,value_type{1}, true),
//         //c_order traverse
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,1);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,2);}, shape_type{2}, value_type{3}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,3);}, shape_type{3}, value_type{4}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,4);}, shape_type{0}, value_type{1}, false),
//         //f_order traverse
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,1);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,2);}, shape_type{2}, value_type{3}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,3);}, shape_type{3}, value_type{4}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,4);}, shape_type{0}, value_type{1}, false),
//         //elements in f_order, 1-d
//         //no command
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto&){return true;}, shape_type{0} ,value_type{1}, true),
//         //c_order traverse
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,1);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,2);}, shape_type{2}, value_type{3}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,3);}, shape_type{3}, value_type{4}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,4);}, shape_type{0}, value_type{1}, false),
//         //f_order traverse
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,1);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,2);}, shape_type{2}, value_type{3}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,3);}, shape_type{3}, value_type{4}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,4);}, shape_type{0}, value_type{1}, false),
//         //elements in c_order, n-d
//         //no command
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto&){return true;}, shape_type{0,0,0} ,value_type{1}, true),
//         //c_order traverse
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto& tr){return do_next<c_order>(tr,1);}, shape_type{0,0,1}, value_type{2}, true),
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto& tr){return do_next<c_order>(tr,4);}, shape_type{0,2,0}, value_type{5}, true),
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto& tr){return do_next<c_order>(tr,7);}, shape_type{1,0,1}, value_type{8}, true),
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto& tr){return do_next<c_order>(tr,11);}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto& tr){return do_next<c_order>(tr,12);}, shape_type{0,0,0}, value_type{1}, false),
//         //f_order traverse
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto& tr){return do_next<f_order>(tr,1);}, shape_type{1,0,0}, value_type{7}, true),
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto& tr){return do_next<f_order>(tr,4);}, shape_type{0,2,0}, value_type{5}, true),
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto& tr){return do_next<f_order>(tr,6);}, shape_type{0,0,1}, value_type{2}, true),
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto& tr){return do_next<f_order>(tr,11);}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(c_order{}, storage_c, shape_type{2,3,2}, [](auto& tr){return do_next<f_order>(tr,12);}, shape_type{0,0,0}, value_type{1}, false),
//         //elements in f_order, n-d
//         //no command
//         std::make_tuple(f_order{}, storage_f, shape_type{2,3,2}, [](auto&){return true;}, shape_type{0,0,0} ,value_type{1}, true),
//         //c_order traverse
//         std::make_tuple(f_order{}, storage_f, shape_type{2,3,2}, [](auto& tr){return do_next<c_order>(tr,1);}, shape_type{0,0,1}, value_type{2}, true),
//         std::make_tuple(f_order{}, storage_f, shape_type{2,3,2}, [](auto& tr){return do_next<c_order>(tr,4);}, shape_type{0,2,0}, value_type{5}, true),
//         std::make_tuple(f_order{}, storage_f, shape_type{2,3,2}, [](auto& tr){return do_next<c_order>(tr,7);}, shape_type{1,0,1}, value_type{8}, true),
//         std::make_tuple(f_order{}, storage_f, shape_type{2,3,2}, [](auto& tr){return do_next<c_order>(tr,11);}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(f_order{}, storage_f, shape_type{2,3,2}, [](auto& tr){return do_next<c_order>(tr,12);}, shape_type{0,0,0}, value_type{1}, false),
//         //f_order traverse
//         std::make_tuple(f_order{}, storage_f, shape_type{1,2,3,2}, [](auto& tr){return do_next<f_order>(tr,1);}, shape_type{0,1,0,0}, value_type{7}, true),
//         std::make_tuple(f_order{}, storage_f, shape_type{2,3,2,1}, [](auto& tr){return do_next<f_order>(tr,4);}, shape_type{0,2,0,0}, value_type{5}, true),
//         std::make_tuple(f_order{}, storage_f, shape_type{2,1,3,2}, [](auto& tr){return do_next<f_order>(tr,6);}, shape_type{0,0,0,1}, value_type{2}, true),
//         std::make_tuple(f_order{}, storage_f, shape_type{2,1,3,2}, [](auto& tr){return do_next<f_order>(tr,11);}, shape_type{1,0,2,1}, value_type{12}, true),
//         std::make_tuple(f_order{}, storage_f, shape_type{2,1,3,2}, [](auto& tr){return do_next<f_order>(tr,12);}, shape_type{0,0,0,0}, value_type{1}, false)
//     );
//     auto test = [](const auto& t){
//         auto elements_order = std::get<0>(t);
//         auto storage = std::get<1>(t);
//         auto shape = std::get<2>(t);
//         auto command = std::get<3>(t);
//         auto expected_index = std::get<4>(t);
//         auto expected_element = std::get<5>(t);
//         auto expected_is_next = std::get<6>(t);
//         auto indexer = indexer_type{storage};
//         auto strides = make_strides(shape, elements_order);
//         auto adapted_strides = make_adapted_strides(shape,strides);
//         auto reset_strides = make_reset_strides(shape,strides);
//         index_type offset{0};
//         auto walker =  walker_type{adapted_strides, reset_strides, offset, indexer};
//         using traverser_type = walker_forward_traverser<config_type, walker_type>;
//         auto traverser = traverser_type{shape, walker};
//         auto result_is_next = command(traverser);
//         auto result_index = traverser.index();
//         auto result_element = *traverser.walker();
//         REQUIRE(result_is_next == expected_is_next);
//         REQUIRE(result_index == expected_index);
//         REQUIRE(result_element == expected_element);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_walker_bidirectional_traverser","test_data_accessor")
// {
//     using value_type = int;
//     using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
//     using gtensor::walker_forward_traverser;
//     using gtensor::walker_bidirectional_traverser;
//     using shape_type = typename config_type::shape_type;
//     using index_type = typename config_type::index_type;
//     using storage_type = typename config_type::template storage<value_type>;
//     using indexer_type = gtensor::basic_indexer<storage_type&>;
//     using walker_type = gtensor::indexer_walker<config_type, indexer_type>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using gtensor::detail::make_strides;
//     using gtensor::detail::make_adapted_strides;
//     using gtensor::detail::make_reset_strides;
//     using gtensor::detail::make_dividers;
//     using test_walker_traverser::do_next;
//     using test_walker_traverser::do_prev;
//     using helpers_for_testing::apply_by_element;

//     const auto storage_1d = storage_type{1,2,3,4};
//     const auto shape_1d = shape_type{4};

//     const auto storage_c = storage_type{1,2,3,4,5,6,7,8,9,10,11,12};
//     const auto storage_f = storage_type{1,7,3,9,5,11,2,8,4,10,6,12};
//     const auto shape = shape_type{2,3,2};

//     //0elements_order,1storage,2shape,3command,4expected_index,5expected_element,6expected_is_prev
//     auto test_data = std::make_tuple(
//         //elements in c_order, 1-d
//         //no command
//         std::make_tuple(c_order{}, storage_type{1,2,3,4}, shape_type{4}, [](auto&){return true;}, shape_type{0} ,value_type{1}, true),
//         //c_order traverse
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_prev<c_order>(tr,1);}, shape_type{3}, value_type{4}, false),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,4), do_prev<c_order>(tr,1);}, shape_type{3}, value_type{4}, false),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,4), do_prev<c_order>(tr,2);}, shape_type{2}, value_type{3}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,4), do_prev<c_order>(tr,3);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,4), do_prev<c_order>(tr,4);}, shape_type{0}, value_type{1}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), true;}, shape_type{3}, value_type{4}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), do_prev<c_order>(tr,2);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), do_prev<c_order>(tr,2), do_next<c_order>(tr,1);}, shape_type{2}, value_type{3}, true),
//         //f_order traverse
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_prev<f_order>(tr,1);}, shape_type{3}, value_type{4}, false),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,4), do_prev<f_order>(tr,1);}, shape_type{3}, value_type{4}, false),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,4), do_prev<f_order>(tr,2);}, shape_type{2}, value_type{3}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,4), do_prev<f_order>(tr,3);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,4), do_prev<f_order>(tr,4);}, shape_type{0}, value_type{1}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), true;}, shape_type{3}, value_type{4}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), do_prev<f_order>(tr,2);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), do_prev<f_order>(tr,2), do_next<f_order>(tr,1);}, shape_type{2}, value_type{3}, true),
//         //elements in f_order, 1-d
//         //no command
//         std::make_tuple(c_order{}, storage_1d, shape_1d, [](auto&){return true;}, shape_type{0} ,value_type{1}, true),
//         //c_order traverse
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_prev<c_order>(tr,1);}, shape_type{3}, value_type{4}, false),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,4), do_prev<c_order>(tr,1);}, shape_type{3}, value_type{4}, false),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,4), do_prev<c_order>(tr,2);}, shape_type{2}, value_type{3}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,4), do_prev<c_order>(tr,3);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<c_order>(tr,4), do_prev<c_order>(tr,4);}, shape_type{0}, value_type{1}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), true;}, shape_type{3}, value_type{4}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), do_prev<c_order>(tr,2);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), do_prev<c_order>(tr,2), do_next<c_order>(tr,1);}, shape_type{2}, value_type{3}, true),
//         //f_order traverse
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_prev<f_order>(tr,1);}, shape_type{3}, value_type{4}, false),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,4), do_prev<f_order>(tr,1);}, shape_type{3}, value_type{4}, false),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,4), do_prev<f_order>(tr,2);}, shape_type{2}, value_type{3}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,4), do_prev<f_order>(tr,3);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return do_next<f_order>(tr,4), do_prev<f_order>(tr,4);}, shape_type{0}, value_type{1}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), true;}, shape_type{3}, value_type{4}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), do_prev<f_order>(tr,2);}, shape_type{1}, value_type{2}, true),
//         std::make_tuple(f_order{}, storage_1d, shape_1d, [](auto& tr){return tr.to_last(), do_prev<f_order>(tr,2), do_next<f_order>(tr,1);}, shape_type{2}, value_type{3}, true),
//         //elements in c_order, n-d
//         //no command
//         std::make_tuple(c_order{}, storage_c, shape, [](auto&){return true;}, shape_type{0,0,0} ,value_type{1}, true),
//         //c_order traverse
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return do_prev<c_order>(tr,1);}, shape_type{1,2,1}, value_type{12}, false),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return do_next<c_order>(tr,12), do_prev<c_order>(tr,1);}, shape_type{1,2,1}, value_type{12}, false),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return do_next<c_order>(tr,12), do_prev<c_order>(tr,2);}, shape_type{1,2,0}, value_type{11}, true),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return do_next<c_order>(tr,12), do_prev<c_order>(tr,5);}, shape_type{1,0,1}, value_type{8}, true),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return do_next<c_order>(tr,12), do_prev<c_order>(tr,7);}, shape_type{0,2,1}, value_type{6}, true),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return tr.to_last(), true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return tr.to_last(), do_prev<c_order>(tr,2);}, shape_type{1,1,1}, value_type{10}, true),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return tr.to_last(), do_prev<c_order>(tr,4), do_next<c_order>(tr,1);}, shape_type{1,1,0}, value_type{9}, true),
//         //f_order traverse
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return do_prev<f_order>(tr,1);}, shape_type{1,2,1}, value_type{12}, false),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return do_next<f_order>(tr,12), do_prev<f_order>(tr,1);}, shape_type{1,2,1}, value_type{12}, false),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return do_next<f_order>(tr,12), do_prev<f_order>(tr,2);}, shape_type{0,2,1}, value_type{6}, true),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return do_next<f_order>(tr,12), do_prev<f_order>(tr,5);}, shape_type{1,0,1}, value_type{8}, true),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return do_next<f_order>(tr,12), do_prev<f_order>(tr,7);}, shape_type{1,2,0}, value_type{11}, true),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return tr.to_last(), true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return tr.to_last(), do_prev<f_order>(tr,2);}, shape_type{1,1,1}, value_type{10}, true),
//         std::make_tuple(c_order{}, storage_c, shape, [](auto& tr){return tr.to_last(), do_prev<f_order>(tr,4), do_next<f_order>(tr,1);}, shape_type{0,1,1}, value_type{4}, true),
//         //elements in f_order, n-d
//         //no command
//         std::make_tuple(f_order{}, storage_f, shape, [](auto&){return true;}, shape_type{0,0,0} ,value_type{1}, true),
//         //c_order traverse
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return do_prev<c_order>(tr,1);}, shape_type{1,2,1}, value_type{12}, false),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return do_next<c_order>(tr,12), do_prev<c_order>(tr,1);}, shape_type{1,2,1}, value_type{12}, false),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return do_next<c_order>(tr,12), do_prev<c_order>(tr,2);}, shape_type{1,2,0}, value_type{11}, true),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return do_next<c_order>(tr,12), do_prev<c_order>(tr,5);}, shape_type{1,0,1}, value_type{8}, true),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return do_next<c_order>(tr,12), do_prev<c_order>(tr,7);}, shape_type{0,2,1}, value_type{6}, true),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return tr.to_last(), true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return tr.to_last(), do_prev<c_order>(tr,2);}, shape_type{1,1,1}, value_type{10}, true),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return tr.to_last(), do_prev<c_order>(tr,4), do_next<c_order>(tr,1);}, shape_type{1,1,0}, value_type{9}, true),
//         //f_order traverse
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return do_prev<f_order>(tr,1);}, shape_type{1,2,1}, value_type{12}, false),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return do_next<f_order>(tr,12), do_prev<f_order>(tr,1);}, shape_type{1,2,1}, value_type{12}, false),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return do_next<f_order>(tr,12), do_prev<f_order>(tr,2);}, shape_type{0,2,1}, value_type{6}, true),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return do_next<f_order>(tr,12), do_prev<f_order>(tr,5);}, shape_type{1,0,1}, value_type{8}, true),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return do_next<f_order>(tr,12), do_prev<f_order>(tr,7);}, shape_type{1,2,0}, value_type{11}, true),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return tr.to_last(), true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return tr.to_last(), do_prev<f_order>(tr,2);}, shape_type{1,1,1}, value_type{10}, true),
//         std::make_tuple(f_order{}, storage_f, shape, [](auto& tr){return tr.to_last(), do_prev<f_order>(tr,4), do_next<f_order>(tr,1);}, shape_type{0,1,1}, value_type{4}, true)
//     );
//     auto test = [](const auto& t){
//         auto elements_order = std::get<0>(t);
//         auto storage = std::get<1>(t);
//         auto shape = std::get<2>(t);
//         auto command = std::get<3>(t);
//         auto expected_index = std::get<4>(t);
//         auto expected_element = std::get<5>(t);
//         auto expected_is_next = std::get<6>(t);
//         auto indexer = indexer_type{storage};
//         auto strides = make_strides(shape, elements_order);
//         auto adapted_strides = make_adapted_strides(shape,strides);
//         auto reset_strides = make_reset_strides(shape,strides);
//         index_type offset{0};
//         auto walker =  walker_type{adapted_strides, reset_strides, offset, indexer};
//         using traverser_type = walker_bidirectional_traverser<walker_forward_traverser<config_type, walker_type>>;
//         auto traverser = traverser_type{shape, walker};
//         auto result_is_next = command(traverser);
//         auto result_index = traverser.index();
//         auto result_element = *traverser.walker();
//         REQUIRE(result_is_next == expected_is_next);
//         REQUIRE(result_index == expected_index);
//         REQUIRE(result_element == expected_element);
//     };
//     apply_by_element(test, test_data);
// }

// TEST_CASE("test_walker_random_access_traverser","test_data_accessor")
// {
//     using value_type = int;
//     using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
//     using gtensor::walker_random_access_traverser;
//     using gtensor::walker_bidirectional_traverser;
//     using gtensor::walker_forward_traverser;
//     using shape_type = typename config_type::shape_type;
//     using index_type = typename config_type::index_type;
//     using storage_type = typename config_type::template storage<value_type>;
//     using indexer_type = gtensor::basic_indexer<storage_type&>;
//     using walker_type = gtensor::indexer_walker<config_type, indexer_type>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using gtensor::detail::make_strides;
//     using gtensor::detail::make_adapted_strides;
//     using gtensor::detail::make_reset_strides;
//     using gtensor::detail::make_strides_div;
//     using test_walker_traverser::do_next;
//     using test_walker_traverser::do_prev;
//     using helpers_for_testing::apply_by_element;

//     const auto storage_c = storage_type{1,2,3,4,5,6,7,8,9,10,11,12};
//     const auto storage_f = storage_type{1,7,3,9,5,11,2,8,4,10,6,12};
//     const auto shape = shape_type{2,3,2};

//     //0elements_order,1traverser_order,2storage,3shape,4command,5expected_index,6expected_element,7expected_is_next
//     auto test_data = std::make_tuple(
//         //elements in c_order, traverse c_order
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.move(0), true;}, shape_type{0,0,0}, value_type{1}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.move(1), true;}, shape_type{0,0,1}, value_type{2}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.move(5), true;}, shape_type{0,2,1}, value_type{6}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.move(8), true;}, shape_type{1,1,0}, value_type{9}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.move(11), true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.move(5),tr.next();}, shape_type{1,0,0}, value_type{7}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.move(11),tr.next();}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.move(5),tr.prev();}, shape_type{0,2,0}, value_type{5}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.to_last(),tr.move(7),true;}, shape_type{1,0,1}, value_type{8}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.move(7),tr.to_last(),true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.prev(),tr.move(3),true;}, shape_type{0,1,1}, value_type{4}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, [](auto& tr){return tr.next(),tr.move(3),true;}, shape_type{0,1,1}, value_type{4}, true),
//         //elements in c_order, traverse f_order
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.move(0), true;}, shape_type{0,0,0}, value_type{1}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.move(1), true;}, shape_type{1,0,0}, value_type{7}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.move(5), true;}, shape_type{1,2,0}, value_type{11}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.move(8), true;}, shape_type{0,1,1}, value_type{4}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.move(11), true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.move(5),tr.next();}, shape_type{0,0,1}, value_type{2}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.move(11),tr.next();}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.move(5),tr.prev();}, shape_type{0,2,0}, value_type{5}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.to_last(),tr.move(7),true;}, shape_type{1,0,1}, value_type{8}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.move(7),tr.to_last(),true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.prev(),tr.move(3),true;}, shape_type{1,1,0}, value_type{9}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, [](auto& tr){return tr.next(),tr.move(3),true;}, shape_type{1,1,0}, value_type{9}, true),
//         //elements in f_order, traverse c_order
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.move(0), true;}, shape_type{0,0,0}, value_type{1}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.move(1), true;}, shape_type{0,0,1}, value_type{2}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.move(5), true;}, shape_type{0,2,1}, value_type{6}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.move(8), true;}, shape_type{1,1,0}, value_type{9}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.move(11), true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.move(5),tr.next();}, shape_type{1,0,0}, value_type{7}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.move(11),tr.next();}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.move(5),tr.prev();}, shape_type{0,2,0}, value_type{5}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.to_last(),tr.move(7),true;}, shape_type{1,0,1}, value_type{8}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.move(7),tr.to_last(),true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.prev(),tr.move(3),true;}, shape_type{0,1,1}, value_type{4}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, [](auto& tr){return tr.next(),tr.move(3),true;}, shape_type{0,1,1}, value_type{4}, true),
//         //elements in f_order, traverse f_order
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.move(0), true;}, shape_type{0,0,0}, value_type{1}, true),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.move(1), true;}, shape_type{1,0,0}, value_type{7}, true),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.move(5), true;}, shape_type{1,2,0}, value_type{11}, true),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.move(8), true;}, shape_type{0,1,1}, value_type{4}, true),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.move(11), true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.move(5),tr.next();}, shape_type{0,0,1}, value_type{2}, true),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.move(11),tr.next();}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.move(5),tr.prev();}, shape_type{0,2,0}, value_type{5}, true),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.to_last(),tr.move(7),true;}, shape_type{1,0,1}, value_type{8}, true),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.move(7),tr.to_last(),true;}, shape_type{1,2,1}, value_type{12}, true),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.prev(),tr.move(3),true;}, shape_type{1,1,0}, value_type{9}, true),
//         std::make_tuple(f_order{}, f_order{}, storage_f, shape, [](auto& tr){return tr.next(),tr.move(3),true;}, shape_type{1,1,0}, value_type{9}, true)
//     );
//     auto test = [](const auto& t){
//         auto elements_order = std::get<0>(t);
//         auto traverser_order = std::get<1>(t);
//         auto storage = std::get<2>(t);
//         auto shape = std::get<3>(t);
//         auto command = std::get<4>(t);
//         auto expected_index = std::get<5>(t);
//         auto expected_element = std::get<6>(t);
//         auto expected_is_next = std::get<7>(t);
//         auto indexer = indexer_type{storage};
//         auto strides = make_strides(shape, elements_order);
//         auto adapted_strides = make_adapted_strides(shape,strides);
//         auto reset_strides = make_reset_strides(shape,strides);
//         index_type offset{0};
//         auto walker =  walker_type{adapted_strides, reset_strides, offset, indexer};
//         auto strides_div = make_strides_div<config_type>(shape, traverser_order);
//         using traverser_order_type = decltype(traverser_order);
//         using traverser_type = walker_random_access_traverser<walker_bidirectional_traverser<walker_forward_traverser<config_type,walker_type>>,traverser_order_type>;
//         auto traverser = traverser_type{shape, strides_div, walker};
//         auto result_is_next = command(traverser);
//         auto result_index = traverser.index();
//         auto result_element = *traverser.walker();
//         REQUIRE(result_is_next == expected_is_next);
//         REQUIRE(result_index == expected_index);
//         REQUIRE(result_element == expected_element);
//     };
//     apply_by_element(test, test_data);
// }


// TEST_CASE("test_walker_range_traverser","test_data_accessor")
// {
//     using value_type = int;
//     using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
//     using shape_type = typename config_type::shape_type;
//     using index_type = typename config_type::index_type;
//     using storage_type = typename config_type::template storage<value_type>;
//     using indexer_type = gtensor::basic_indexer<storage_type&>;
//     using walker_type = gtensor::indexer_walker<config_type, indexer_type>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using gtensor::detail::make_strides;
//     using gtensor::detail::make_adapted_strides;
//     using gtensor::detail::make_reset_strides;
//     using gtensor::detail::make_dividers;
//     using test_walker_traverser::do_next;
//     using test_walker_traverser::do_prev;
//     using helpers_for_testing::apply_by_element;

//     const auto storage_c = storage_type{1,5,9,2,6,10,3,7,11,4,8,12,13,17,21,14,18,22,15,19,23,16,20,24};
//     const auto storage_f = storage_type{1,13,2,14,3,15,4,16,5,17,6,18,7,19,8,20,9,21,10,22,11,23,12,24};
//     const auto shape = shape_type{2,4,3};
//     //traverse ranges: [0,2), [2,3)

//     //0elements_order,1storage,2shape,3axis_min,4axis_max,5command,6expected_index,7expected_element,8expected_is_next
//     auto test_data = std::make_tuple(
//         //elements in c_order
//         //traverse axes [2,3) c_order
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_prev<c_order>(tr,1);}, shape_type{0,0,2}, value_type{9}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_prev<c_order>(tr,1),do_next<c_order>(tr,1);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_next<c_order>(tr,1);}, shape_type{0,0,1}, value_type{5}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_next<c_order>(tr,2);}, shape_type{0,0,2}, value_type{9}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_next<c_order>(tr,3);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_next<c_order>(tr,3),do_prev<c_order>(tr,1);}, shape_type{0,0,2}, value_type{9}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return tr.to_last(),true;}, shape_type{0,0,2}, value_type{9}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return tr.to_last(),do_prev<c_order>(tr,1);}, shape_type{0,0,1}, value_type{5}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_next<c_order>(tr,3),tr.to_last(),true;}, shape_type{0,0,2}, value_type{9}, true),
//         // //traverse axes [0,2) c_order
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_prev<c_order>(tr,1);}, shape_type{1,3,0}, value_type{16}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_prev<c_order>(tr,1),do_next<c_order>(tr,1);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,1);}, shape_type{0,1,0}, value_type{2}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,3);}, shape_type{0,3,0}, value_type{4}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,7);}, shape_type{1,3,0}, value_type{16}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,8);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,8),do_prev<c_order>(tr,1);}, shape_type{1,3,0}, value_type{16}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,8),do_prev<c_order>(tr,5);}, shape_type{0,3,0}, value_type{4}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return tr.to_last(),true;}, shape_type{1,3,0}, value_type{16}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return tr.to_last(),do_prev<c_order>(tr,3),true;}, shape_type{1,0,0}, value_type{13}, true),
//         // //traverse axes [2,3) f_order
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_prev<f_order>(tr,1);}, shape_type{0,0,2}, value_type{9}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_prev<f_order>(tr,1),do_next<f_order>(tr,1);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_next<f_order>(tr,1);}, shape_type{0,0,1}, value_type{5}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_next<f_order>(tr,2);}, shape_type{0,0,2}, value_type{9}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_next<f_order>(tr,3);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_next<f_order>(tr,3),do_prev<f_order>(tr,1);}, shape_type{0,0,2}, value_type{9}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return tr.to_last(),true;}, shape_type{0,0,2}, value_type{9}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return tr.to_last(),do_prev<f_order>(tr,1);}, shape_type{0,0,1}, value_type{5}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 2, 3, [](auto& tr){return do_next<f_order>(tr,3),tr.to_last(),true;}, shape_type{0,0,2}, value_type{9}, true),
//         // //traverse axes [0,2) f_order
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_prev<f_order>(tr,1);}, shape_type{1,3,0}, value_type{16}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_prev<f_order>(tr,1),do_next<f_order>(tr,1);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,1);}, shape_type{1,0,0}, value_type{13}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,3);}, shape_type{1,1,0}, value_type{14}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,7);}, shape_type{1,3,0}, value_type{16}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,8);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,8),do_prev<f_order>(tr,1);}, shape_type{1,3,0}, value_type{16}, false),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,8),do_prev<f_order>(tr,5);}, shape_type{1,1,0}, value_type{14}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return tr.to_last(),true;}, shape_type{1,3,0}, value_type{16}, true),
//         std::make_tuple(c_order{}, storage_c, shape, 0, 2, [](auto& tr){return tr.to_last(),do_prev<f_order>(tr,3),true;}, shape_type{0,2,0}, value_type{3}, true),
//         //elements in f_order
//         //traverse axes [2,3) c_order
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_prev<c_order>(tr,1);}, shape_type{0,0,2}, value_type{9}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_prev<c_order>(tr,1),do_next<c_order>(tr,1);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_next<c_order>(tr,1);}, shape_type{0,0,1}, value_type{5}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_next<c_order>(tr,2);}, shape_type{0,0,2}, value_type{9}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_next<c_order>(tr,3);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_next<c_order>(tr,3),do_prev<c_order>(tr,1);}, shape_type{0,0,2}, value_type{9}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return tr.to_last(),true;}, shape_type{0,0,2}, value_type{9}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return tr.to_last(),do_prev<c_order>(tr,1);}, shape_type{0,0,1}, value_type{5}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_next<c_order>(tr,3),tr.to_last(),true;}, shape_type{0,0,2}, value_type{9}, true),
//         //traverse axes [0,2) c_order
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_prev<c_order>(tr,1);}, shape_type{1,3,0}, value_type{16}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_prev<c_order>(tr,1),do_next<c_order>(tr,1);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,1);}, shape_type{0,1,0}, value_type{2}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,3);}, shape_type{0,3,0}, value_type{4}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,7);}, shape_type{1,3,0}, value_type{16}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,8);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,8),do_prev<c_order>(tr,1);}, shape_type{1,3,0}, value_type{16}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<c_order>(tr,8),do_prev<c_order>(tr,5);}, shape_type{0,3,0}, value_type{4}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return tr.to_last(),true;}, shape_type{1,3,0}, value_type{16}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return tr.to_last(),do_prev<c_order>(tr,3),true;}, shape_type{1,0,0}, value_type{13}, true),
//         //traverse axes [2,3) f_order
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_prev<f_order>(tr,1);}, shape_type{0,0,2}, value_type{9}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_prev<f_order>(tr,1),do_next<f_order>(tr,1);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_next<f_order>(tr,1);}, shape_type{0,0,1}, value_type{5}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_next<f_order>(tr,2);}, shape_type{0,0,2}, value_type{9}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_next<f_order>(tr,3);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_next<f_order>(tr,3),do_prev<f_order>(tr,1);}, shape_type{0,0,2}, value_type{9}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return tr.to_last(),true;}, shape_type{0,0,2}, value_type{9}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return tr.to_last(),do_prev<f_order>(tr,1);}, shape_type{0,0,1}, value_type{5}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 2, 3, [](auto& tr){return do_next<f_order>(tr,3),tr.to_last(),true;}, shape_type{0,0,2}, value_type{9}, true),
//         //traverse axes [0,2) f_order
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_prev<f_order>(tr,1);}, shape_type{1,3,0}, value_type{16}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_prev<f_order>(tr,1),do_next<f_order>(tr,1);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,1);}, shape_type{1,0,0}, value_type{13}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,3);}, shape_type{1,1,0}, value_type{14}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,7);}, shape_type{1,3,0}, value_type{16}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,8);}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,8),do_prev<f_order>(tr,1);}, shape_type{1,3,0}, value_type{16}, false),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return do_next<f_order>(tr,8),do_prev<f_order>(tr,5);}, shape_type{1,1,0}, value_type{14}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return tr.to_last(),true;}, shape_type{1,3,0}, value_type{16}, true),
//         std::make_tuple(f_order{}, storage_f, shape, 0, 2, [](auto& tr){return tr.to_last(),do_prev<f_order>(tr,3),true;}, shape_type{0,2,0}, value_type{3}, true)
//     );
//     auto test = [](const auto& t){
//         auto elements_order = std::get<0>(t);
//         auto storage = std::get<1>(t);
//         auto shape = std::get<2>(t);
//         auto axis_min = std::get<3>(t);
//         auto axis_max = std::get<4>(t);
//         auto command = std::get<5>(t);
//         auto expected_index = std::get<6>(t);
//         auto expected_element = std::get<7>(t);
//         auto expected_is_next = std::get<8>(t);
//         using traverser_type = gtensor::walker_bidirectional_traverser<gtensor::walker_forward_range_traverser<config_type, walker_type>>;
//         auto indexer = indexer_type{storage};
//         auto strides = make_strides(shape, elements_order);
//         auto adapted_strides = make_adapted_strides(shape,strides);
//         auto reset_strides = make_reset_strides(shape,strides);
//         index_type offset{0};
//         auto walker =  walker_type{adapted_strides, reset_strides, offset, indexer};
//         auto traverser = traverser_type{shape, walker, axis_min, axis_max};
//         auto result_is_next = command(traverser);
//         auto result_index = traverser.index();
//         auto result_element = *traverser.walker();
//         REQUIRE(result_is_next == expected_is_next);
//         REQUIRE(result_index == expected_index);
//         REQUIRE(result_element == expected_element);
//     };
//     apply_by_element(test, test_data);
// }


// TEST_CASE("test_walker_random_access_traverser_predicate","test_data_accessor")
// {
//     using value_type = int;
//     using config_type = gtensor::config::extend_config_t<test_config::config_storage_selector_t<std::vector>,value_type>;
//     using shape_type = typename config_type::shape_type;
//     using index_type = typename config_type::index_type;
//     using storage_type = typename config_type::template storage<value_type>;
//     using indexer_type = gtensor::basic_indexer<storage_type&>;
//     using walker_type = gtensor::indexer_walker<config_type, indexer_type>;
//     using gtensor::config::c_order;
//     using gtensor::config::f_order;
//     using gtensor::detail::make_strides;
//     using gtensor::detail::make_adapted_strides;
//     using gtensor::detail::make_reset_strides;
//     using gtensor::detail::make_strides_div_predicate;
//     using test_walker_traverser::do_next;
//     using test_walker_traverser::do_prev;
//     using helpers_for_testing::apply_by_element;

//     const auto storage_c = storage_type{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24};
//     const auto storage_f = storage_type{1,13,5,17,9,21,2,14,6,18,10,22,3,15,7,19,11,23,4,16,8,20,12,24};
//     const auto shape = shape_type{2,3,4};
//     auto d_1 = [](const auto& d){if(d==0||d==2){return false;} return true;};
//     auto d_02 = [](const auto& d){if(d==1){return false;} return true;};

//     //0elements_order,1traverse_order,2storage,3shape,4predicate,5command,6expected_index,7expected_element
//     auto test_data = std::make_tuple(
//         //elements in c_order
//         //traverse direction 1 c_order,c_order
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, d_1, [](auto& tr){return tr.move(0),true;}, shape_type{0,0,0}, value_type{1}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, d_1, [](auto& tr){return tr.move(0),tr.next();}, shape_type{0,1,0}, value_type{5}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, d_1, [](auto& tr){return tr.move(0),tr.prev();}, shape_type{0,2,0}, value_type{9}, false),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, d_1, [](auto& tr){return tr.move(1),true;}, shape_type{0,1,0}, value_type{5}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, d_1, [](auto& tr){return tr.move(1),tr.next();}, shape_type{0,2,0}, value_type{9}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, d_1, [](auto& tr){return tr.move(1),tr.prev();}, shape_type{0,0,0}, value_type{1}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, d_1, [](auto& tr){return tr.move(2),true;}, shape_type{0,2,0}, value_type{9}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, d_1, [](auto& tr){return tr.move(2),tr.next();}, shape_type{0,0,0}, value_type{1}, false),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, d_1, [](auto& tr){return tr.move(2),tr.prev();}, shape_type{0,1,0}, value_type{5}, true),
//         std::make_tuple(c_order{}, c_order{}, storage_c, shape, d_1, [](auto& tr){return tr.move(2),tr.prev();}, shape_type{0,1,0}, value_type{5}, true),
//         //traverse direction 0,2 c_order,f_order
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, d_02, [](auto& tr){return tr.move(0),true;}, shape_type{0,0,0}, value_type{1}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, d_02, [](auto& tr){return tr.move(0),tr.next();}, shape_type{1,0,0}, value_type{13}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, d_02, [](auto& tr){return tr.move(0),tr.prev();}, shape_type{1,0,3}, value_type{16}, false),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, d_02, [](auto& tr){return tr.move(1),true;}, shape_type{1,0,0}, value_type{13}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, d_02, [](auto& tr){return tr.move(4),true;}, shape_type{0,0,2}, value_type{3}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, d_02, [](auto& tr){return tr.move(7),true;}, shape_type{1,0,3}, value_type{16}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, d_02, [](auto& tr){return tr.move(7),tr.prev();}, shape_type{0,0,3}, value_type{4}, true),
//         std::make_tuple(c_order{}, f_order{}, storage_c, shape, d_02, [](auto& tr){return tr.move(7),tr.next();}, shape_type{0,0,0}, value_type{1}, false),
//         //traverse direction 0,2 f_order,c_order
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, d_02, [](auto& tr){return tr.move(0),true;}, shape_type{0,0,0}, value_type{1}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, d_02, [](auto& tr){return tr.move(0),tr.next();}, shape_type{0,0,1}, value_type{2}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, d_02, [](auto& tr){return tr.move(0),tr.prev();}, shape_type{1,0,3}, value_type{16}, false),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, d_02, [](auto& tr){return tr.move(1),true;}, shape_type{0,0,1}, value_type{2}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, d_02, [](auto& tr){return tr.move(4),true;}, shape_type{1,0,0}, value_type{13}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, d_02, [](auto& tr){return tr.move(7),true;}, shape_type{1,0,3}, value_type{16}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, d_02, [](auto& tr){return tr.move(7),tr.prev();}, shape_type{1,0,2}, value_type{15}, true),
//         std::make_tuple(f_order{}, c_order{}, storage_f, shape, d_02, [](auto& tr){return tr.move(7),tr.next();}, shape_type{0,0,0}, value_type{1}, false)
//     );
//     auto test = [](const auto& t){
//         auto elements_order = std::get<0>(t);
//         auto traverse_order = std::get<1>(t);
//         auto storage = std::get<2>(t);
//         auto shape = std::get<3>(t);
//         auto predicate = std::get<4>(t);
//         auto command = std::get<5>(t);
//         auto expected_index = std::get<6>(t);
//         auto expected_element = std::get<7>(t);
//         auto expected_is_next = std::get<8>(t);
//         using traverser_type = gtensor::walker_random_access_traverser<config_type, walker_type, decltype(traverse_order), decltype(predicate)>;
//         auto indexer = indexer_type{storage};
//         auto strides = make_strides(shape, elements_order);
//         auto strides_div = make_strides_div_predicate<config_type>(shape,predicate,traverse_order);
//         auto adapted_strides = make_adapted_strides(shape,strides);
//         auto reset_strides = make_reset_strides(shape,strides);
//         index_type offset{0};
//         auto walker =  walker_type{adapted_strides, reset_strides, offset, indexer};
//         auto traverser = traverser_type{shape, strides_div, walker, predicate};
//         auto result_is_next = command(traverser);
//         auto result_index = traverser.index();
//         auto result_element = *traverser.walker();
//         REQUIRE(result_is_next == expected_is_next);
//         REQUIRE(result_index == expected_index);
//         REQUIRE(result_element == expected_element);
//     };
//     apply_by_element(test, test_data);
// }


