#include "catch.hpp"
#include "helpers_for_testing.hpp"
#include "test_config.hpp"
#include "sort_search.hpp"
#include "tensor.hpp"

//unique
TEMPLATE_TEST_CASE("test_sort_search_unique","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using order = typename tensor_type::order;
    using config_type = typename tensor_type::config_type;
    using index_type = typename tensor_type::index_type;
    using index_tensor_type = gtensor::tensor<index_type,order,config_type>;

    using gtensor::unique;
    using helpers_for_testing::apply_by_element;

    //0tensor,1return_index,2return_inverse,3return_counts,4axis,5expected
    auto test_data = std::make_tuple(
        //0d
        std::make_tuple(tensor_type(3),std::false_type{},std::false_type{},std::false_type{},0,tensor_type{3}),
        std::make_tuple(tensor_type(2),std::true_type{},std::true_type{},std::true_type{},0,
            std::make_tuple(tensor_type{2},index_tensor_type{0},index_tensor_type{0},index_tensor_type{1})
        ),
        //1d
        //only unique
        std::make_tuple(tensor_type{},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{}),
        std::make_tuple(tensor_type{4},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{4}),
        std::make_tuple(tensor_type{4,4,4,4},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{4}),
        std::make_tuple(tensor_type{1,2,3,4,5},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{4,3,1,2,1,3,3,4,4,5,2,5},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{1,2,3,4,5}),
        //return_index,return_inverse,return_count
        std::make_tuple(tensor_type{},std::true_type{},std::true_type{},std::true_type{},0,
            std::make_tuple(tensor_type{},index_tensor_type{},index_tensor_type{},index_tensor_type{})
        ),
        std::make_tuple(tensor_type{4},std::true_type{},std::true_type{},std::true_type{},0,
            std::make_tuple(tensor_type{4},index_tensor_type{0},index_tensor_type{0},index_tensor_type{1})
        ),
        std::make_tuple(tensor_type{4,4,4,4},std::true_type{},std::true_type{},std::true_type{},0,
            std::make_tuple(tensor_type{4},index_tensor_type{0},index_tensor_type{0,0,0,0},index_tensor_type{4})
        ),
        std::make_tuple(tensor_type{4,3,1,2,1,3,3,4,4,5,2,5},std::true_type{},std::true_type{},std::true_type{},0,
            std::make_tuple(tensor_type{1,2,3,4,5},index_tensor_type{2,3,1,0,9},index_tensor_type{3,2,0,1,0,2,2,3,3,4,1,4},index_tensor_type{2,2,3,3,2})
        ),
        std::make_tuple(tensor_type{4,1,2,3,1,4,3,5,1,5,4,5,1,4,4,5,3,3,1,1,3,3,3,1,2,1,3,2,4,1},std::true_type{},std::true_type{},std::true_type{},0,
            std::make_tuple(tensor_type{1,2,3,4,5},index_tensor_type{1,2,3,0,7},index_tensor_type{3,0,1,2,0,3,2,4,0,4,3,4,0,3,3,4,2,2,0,0,2,2,2,0,1,0,2,1,3,0},index_tensor_type{9,3,8,6,4})
        ),
        //return_index
        std::make_tuple(tensor_type{4,1,2,3,1,4,3,5,1,5,4,5,1,4,4,5,3,3,1,1,3,3,3,1,2,1,3,2,4,1},std::true_type{},std::false_type{},std::false_type{},0,
            std::make_tuple(tensor_type{1,2,3,4,5},index_tensor_type{1,2,3,0,7})
        ),
        //return_inverse
        std::make_tuple(tensor_type{4,1,2,3,1,4,3,5,1,5,4,5,1,4,4,5,3,3,1,1,3,3,3,1,2,1,3,2,4,1},std::false_type{},std::true_type{},std::false_type{},0,
            std::make_tuple(tensor_type{1,2,3,4,5},index_tensor_type{3,0,1,2,0,3,2,4,0,4,3,4,0,3,3,4,2,2,0,0,2,2,2,0,1,0,2,1,3,0})
        ),
        //return_count
        std::make_tuple(tensor_type{4,1,2,3,1,4,3,5,1,5,4,5,1,4,4,5,3,3,1,1,3,3,3,1,2,1,3,2,4,1},std::false_type{},std::false_type{},std::true_type{},0,
            std::make_tuple(tensor_type{1,2,3,4,5},index_tensor_type{9,3,8,6,4})
        ),
        //return_index,return_count
        std::make_tuple(tensor_type{4,1,2,3,1,4,3,5,1,5,4,5,1,4,4,5,3,3,1,1,3,3,3,1,2,1,3,2,4,1},std::true_type{},std::false_type{},std::true_type{},0,
            std::make_tuple(tensor_type{1,2,3,4,5},index_tensor_type{1,2,3,0,7},index_tensor_type{9,3,8,6,4})
        ),
        //return_inverse,return_count
        std::make_tuple(tensor_type{4,1,2,3,1,4,3,5,1,5,4,5,1,4,4,5,3,3,1,1,3,3,3,1,2,1,3,2,4,1},std::false_type{},std::true_type{},std::true_type{},0,
            std::make_tuple(tensor_type{1,2,3,4,5},index_tensor_type{3,0,1,2,0,3,2,4,0,4,3,4,0,3,3,4,2,2,0,0,2,2,2,0,1,0,2,1,3,0},index_tensor_type{9,3,8,6,4})
        ),
        //return_index,return_inverse
        std::make_tuple(tensor_type{4,1,2,3,1,4,3,5,1,5,4,5,1,4,4,5,3,3,1,1,3,3,3,1,2,1,3,2,4,1},std::true_type{},std::true_type{},std::false_type{},0,
            std::make_tuple(tensor_type{1,2,3,4,5},index_tensor_type{1,2,3,0,7},index_tensor_type{3,0,1,2,0,3,2,4,0,4,3,4,0,3,3,4,2,2,0,0,2,2,2,0,1,0,2,1,3,0})
        ),
        //nd only unique
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5}},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{2},{1},{3},{5},{4}},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{2},{1},{1},{5},{4},{3},{5},{2}},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{{1},{2},{3},{4},{5}}),
        std::make_tuple(tensor_type{{2},{1},{1},{5},{4},{3},{5},{2}},std::false_type{},std::false_type{},std::false_type{},1,tensor_type{{2},{1},{1},{5},{4},{3},{5},{2}}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}),
        std::make_tuple(tensor_type{{1,2,3},{1,2,3},{4,5,6},{7,8,9},{10,11,12}},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}),
        std::make_tuple(tensor_type{{7,8,9},{1,2,3},{1,2,3},{4,5,6},{7,8,9},{10,11,12},{4,5,6}},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}),
        std::make_tuple(tensor_type{{7,8,9},{1,2,3},{1,2,3},{4,5,6},{7,8,9},{10,11,12},{4,5,6}},std::false_type{},std::false_type{},std::false_type{},1,tensor_type{{7,8,9},{1,2,3},{1,2,3},{4,5,6},{7,8,9},{10,11,12},{4,5,6}}),
        std::make_tuple(tensor_type{{7,1,1,4,7,10,4},{8,2,2,5,8,11,5},{9,3,3,6,9,12,6}},std::false_type{},std::false_type{},std::false_type{},1,tensor_type{{1,4,7,10},{2,5,8,11},{3,6,9,12}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{1,5}},{{1,2},{1,2},{1,3}},{{1,2},{3,4},{1,5}}},std::false_type{},std::false_type{},std::false_type{},0,tensor_type{{{1,2},{1,2},{1,3}},{{1,2},{3,4},{1,5}}}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{1,2}},{{1,2},{1,5},{1,2}},{{1,2},{3,4},{1,2}}},std::false_type{},std::false_type{},std::false_type{},1,tensor_type{{{1,2},{3,4}},{{1,2},{1,5}},{{1,2},{3,4}}}),
        std::make_tuple(tensor_type{{{1,2,1},{3,4,3},{1,5,1}},{{1,2,1},{1,5,1},{1,2,1}},{{1,2,1},{3,4,3},{1,2,1}}},std::false_type{},std::false_type{},std::false_type{},2,tensor_type{{{1,2},{3,4},{1,5}},{{1,2},{1,5},{1,2}},{{1,2},{3,4},{1,2}}}),
        // //return_index,return_inverse,return_counts
        std::make_tuple(
            tensor_type{}.reshape(2,3,0),
            std::true_type{},
            std::true_type{},
            std::true_type{},
            0,
            std::make_tuple(tensor_type{}.reshape(0,3,0),index_tensor_type{},index_tensor_type{},index_tensor_type{})
        ),
        std::make_tuple(
            tensor_type{{1,2,3}},
            std::true_type{},
            std::true_type{},
            std::true_type{},
            0,
            std::make_tuple(tensor_type{{1,2,3}},index_tensor_type{0},index_tensor_type{0},index_tensor_type{1})
        ),
        std::make_tuple(
            tensor_type{{1,2,3},{1,2,3},{1,2,3},{1,2,3}},
            std::true_type{},
            std::true_type{},
            std::true_type{},
            0,
            std::make_tuple(tensor_type{{1,2,3}},index_tensor_type{0},index_tensor_type{0,0,0,0},index_tensor_type{4})
        ),
        std::make_tuple(
            tensor_type{{7,8,9},{1,2,3},{1,2,3},{4,5,6},{7,8,9},{1,2,3},{10,11,12},{4,5,6}},
            std::true_type{},
            std::true_type{},
            std::true_type{},
            0,
            std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},index_tensor_type{1,3,0,6},index_tensor_type{2,0,0,1,2,0,3,1},index_tensor_type{3,2,2,1})
        ),
        std::make_tuple(
            tensor_type{{7,1,1,4,7,10,4},{8,2,2,5,8,11,5},{9,3,3,6,9,12,6}},
            std::true_type{},
            std::true_type{},
            std::true_type{},
            1,
            std::make_tuple(tensor_type{{1,4,7,10},{2,5,8,11},{3,6,9,12}},index_tensor_type{1,3,0,5},index_tensor_type{2,0,0,1,2,3,1},index_tensor_type{2,2,2,1})
        ),
        std::make_tuple(
            tensor_type{{0,1,1,1,0,1,0,1,1,0,2,0,2,2,1,0,2,0,2,0,2,1,1,0,2,2,2,1,2,0},{2,0,2,2,0,0,0,1,2,0,2,0,1,2,1,2,0,0,0,1,0,0,0,1,1,2,2,0,0,0},{0,0,1,1,0,2,1,1,0,1,2,2,2,1,1,0,2,1,2,1,1,1,1,1,0,2,0,2,1,1}},
            std::true_type{},
            std::true_type{},
            std::true_type{},
            1,
            std::make_tuple(
                tensor_type{{0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2},{0,0,0,1,2,0,0,0,1,2,2,0,0,1,1,2,2,2},{0,1,2,1,0,0,1,2,1,0,1,1,2,0,2,0,1,2}},
                index_tensor_type{4,6,11,19,0,1,21,5,7,8,2,20,16,24,12,26,13,10},
                index_tensor_type{4,5,10,10,0,7,1,8,9,1,17,2,14,16,8,4,12,1,12,3,11,6,6,3,13,17,15,7,11,1},
                index_tensor_type{1,4,1,2,2,1,2,2,2,1,2,2,2,1,1,1,1,2}
            )
        ),
        std::make_tuple(
            tensor_type{
                {{1,0},{1,0},{0,1},{1,1},{0,1},{0,0},{0,1},{1,0},{1,1},{1,1},{0,0},{1,1},{0,1},{1,1},{0,1},{1,1},{0,0},{0,0},{0,0},{1,1},{0,0},{0,1},{0,0},{0,0},{0,0},{1,1},{1,1},{1,0},{1,0},{0,0}},
                {{0,0},{0,0},{1,1},{0,1},{0,1},{0,1},{1,1},{0,0},{1,1},{1,1},{1,0},{0,0},{1,0},{1,1},{1,0},{1,1},{1,1},{1,1},{1,0},{1,1},{1,0},{1,1},{0,0},{1,0},{0,1},{0,0},{0,0},{1,1},{1,1},{1,0}}
            },
            std::true_type{},
            std::true_type{},
            std::true_type{},
            1,
            std::make_tuple(
                tensor_type{{{0,0},{0,0},{0,0},{0,0},{0,1},{0,1},{0,1},{1,0},{1,0},{1,1},{1,1},{1,1}},{{0,0},{0,1},{1,0},{1,1},{0,1},{1,0},{1,1},{0,0},{1,1},{0,0},{0,1},{1,1}}},
                index_tensor_type{22,5,10,16,4,12,2,0,27,11,3,8},
                index_tensor_type{7,7,6,10,4,1,6,7,11,11,2,9,5,11,5,11,3,3,2,11,2,6,0,2,1,9,9,8,8,2},
                index_tensor_type{1,2,5,2,1,2,3,3,2,3,1,5}
            )
        ),
        //return_counts
        std::make_tuple(
            tensor_type{
                {{1,0},{1,0},{0,1},{1,1},{0,1},{0,0},{0,1},{1,0},{1,1},{1,1},{0,0},{1,1},{0,1},{1,1},{0,1},{1,1},{0,0},{0,0},{0,0},{1,1},{0,0},{0,1},{0,0},{0,0},{0,0},{1,1},{1,1},{1,0},{1,0},{0,0}},
                {{0,0},{0,0},{1,1},{0,1},{0,1},{0,1},{1,1},{0,0},{1,1},{1,1},{1,0},{0,0},{1,0},{1,1},{1,0},{1,1},{1,1},{1,1},{1,0},{1,1},{1,0},{1,1},{0,0},{1,0},{0,1},{0,0},{0,0},{1,1},{1,1},{1,0}}
            },
            std::false_type{},
            std::false_type{},
            std::true_type{},
            1,
            std::make_tuple(
                tensor_type{{{0,0},{0,0},{0,0},{0,0},{0,1},{0,1},{0,1},{1,0},{1,0},{1,1},{1,1},{1,1}},{{0,0},{0,1},{1,0},{1,1},{0,1},{1,0},{1,1},{0,0},{1,1},{0,0},{0,1},{1,1}}},
                index_tensor_type{1,2,5,2,1,2,3,3,2,3,1,5}
            )
        ),
        //return_index,return_counts
        std::make_tuple(
            tensor_type{
                {{1,0},{1,0},{0,1},{1,1},{0,1},{0,0},{0,1},{1,0},{1,1},{1,1},{0,0},{1,1},{0,1},{1,1},{0,1},{1,1},{0,0},{0,0},{0,0},{1,1},{0,0},{0,1},{0,0},{0,0},{0,0},{1,1},{1,1},{1,0},{1,0},{0,0}},
                {{0,0},{0,0},{1,1},{0,1},{0,1},{0,1},{1,1},{0,0},{1,1},{1,1},{1,0},{0,0},{1,0},{1,1},{1,0},{1,1},{1,1},{1,1},{1,0},{1,1},{1,0},{1,1},{0,0},{1,0},{0,1},{0,0},{0,0},{1,1},{1,1},{1,0}}
            },
            std::true_type{},
            std::false_type{},
            std::true_type{},
            1,
            std::make_tuple(
                tensor_type{{{0,0},{0,0},{0,0},{0,0},{0,1},{0,1},{0,1},{1,0},{1,0},{1,1},{1,1},{1,1}},{{0,0},{0,1},{1,0},{1,1},{0,1},{1,0},{1,1},{0,0},{1,1},{0,0},{0,1},{1,1}}},
                index_tensor_type{22,5,10,16,4,12,2,0,27,11,3,8},
                index_tensor_type{1,2,5,2,1,2,3,3,2,3,1,5}
            )
        ),
        //return_inverse
        std::make_tuple(
            tensor_type{
                {{1,0},{1,0},{0,1},{1,1},{0,1},{0,0},{0,1},{1,0},{1,1},{1,1},{0,0},{1,1},{0,1},{1,1},{0,1},{1,1},{0,0},{0,0},{0,0},{1,1},{0,0},{0,1},{0,0},{0,0},{0,0},{1,1},{1,1},{1,0},{1,0},{0,0}},
                {{0,0},{0,0},{1,1},{0,1},{0,1},{0,1},{1,1},{0,0},{1,1},{1,1},{1,0},{0,0},{1,0},{1,1},{1,0},{1,1},{1,1},{1,1},{1,0},{1,1},{1,0},{1,1},{0,0},{1,0},{0,1},{0,0},{0,0},{1,1},{1,1},{1,0}}
            },
            std::false_type{},
            std::true_type{},
            std::false_type{},
            1,
            std::make_tuple(
                tensor_type{{{0,0},{0,0},{0,0},{0,0},{0,1},{0,1},{0,1},{1,0},{1,0},{1,1},{1,1},{1,1}},{{0,0},{0,1},{1,0},{1,1},{0,1},{1,0},{1,1},{0,0},{1,1},{0,0},{0,1},{1,1}}},
                index_tensor_type{7,7,6,10,4,1,6,7,11,11,2,9,5,11,5,11,3,3,2,11,2,6,0,2,1,9,9,8,8,2}
            )
        ),
        //return_index
        std::make_tuple(
            tensor_type{
                {{1,0},{1,0},{0,1},{1,1},{0,1},{0,0},{0,1},{1,0},{1,1},{1,1},{0,0},{1,1},{0,1},{1,1},{0,1},{1,1},{0,0},{0,0},{0,0},{1,1},{0,0},{0,1},{0,0},{0,0},{0,0},{1,1},{1,1},{1,0},{1,0},{0,0}},
                {{0,0},{0,0},{1,1},{0,1},{0,1},{0,1},{1,1},{0,0},{1,1},{1,1},{1,0},{0,0},{1,0},{1,1},{1,0},{1,1},{1,1},{1,1},{1,0},{1,1},{1,0},{1,1},{0,0},{1,0},{0,1},{0,0},{0,0},{1,1},{1,1},{1,0}}
            },
            std::true_type{},
            std::false_type{},
            std::false_type{},
            1,
            std::make_tuple(
                tensor_type{{{0,0},{0,0},{0,0},{0,0},{0,1},{0,1},{0,1},{1,0},{1,0},{1,1},{1,1},{1,1}},{{0,0},{0,1},{1,0},{1,1},{0,1},{1,0},{1,1},{0,0},{1,1},{0,0},{0,1},{1,1}}},
                index_tensor_type{22,5,10,16,4,12,2,0,27,11,3,8}
            )
        )
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto return_index = std::get<1>(t);
        auto return_inverse = std::get<2>(t);
        auto return_counts = std::get<3>(t);
        auto axis = std::get<4>(t);
        auto expected = std::get<5>(t);

        auto result = unique(ten,return_index,return_inverse,return_counts,axis);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_sort_search_unique_default_args","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type,TestType>;

    using gtensor::unique;
    using helpers_for_testing::apply_by_element;

    //0tensor,1expected
    auto test_data = std::make_tuple(
        std::make_tuple(tensor_type{},tensor_type{}),
        std::make_tuple(tensor_type{1,2,3,4,5},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{4,3,1,2,1,3,3,4,4,5,2,5},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1},{2},{3},{4},{5}},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{2},{1},{3},{5},{4}},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{2},{1},{1},{5},{4},{3},{5},{2}},tensor_type{1,2,3,4,5}),
        std::make_tuple(tensor_type{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},tensor_type{1,2,3,4,5,6,7,8,9,10,11,12}),
        std::make_tuple(tensor_type{{7,8,9},{1,2,3},{1,2,3},{4,5,6},{7,8,9},{10,11,12},{4,5,6}},tensor_type{1,2,3,4,5,6,7,8,9,10,11,12}),
        std::make_tuple(tensor_type{{{1,2},{3,4},{1,5}},{{1,2},{1,2},{1,3}},{{1,2},{3,4},{1,5}}},tensor_type{1,2,3,4,5})
    );
    auto test = [](const auto& t){
        auto ten = std::get<0>(t);
        auto expected = std::get<1>(t);

        auto result = unique(ten);
        REQUIRE(result == expected);
    };
    apply_by_element(test,test_data);
}

TEMPLATE_TEST_CASE("test_sort_search_unique_no_axis","[test_sort_search]",
    gtensor::config::c_order,
    gtensor::config::f_order
)
{
    using value_type = double;
    using gtensor::tensor;
    using tensor_type = gtensor::tensor<value_type,TestType>;
    using gtensor::unique;
    using helpers_for_testing::apply_by_element;

    REQUIRE(unique(tensor_type{{{1,3,4,3,1},{5,1,5,2,2}},{{3,4,3,3,1},{5,1,3,2,1}},{{1,4,5,4,4},{1,4,1,3,4}}})==tensor_type{1,2,3,4,5});
    REQUIRE(unique(tensor_type{{{1,3,4,3,1},{5,1,5,2,2}},{{3,4,3,3,1},{5,1,3,2,1}},{{1,4,5,4,4},{1,4,1,3,4}}},std::true_type{})==std::make_tuple(tensor_type{1,2,3,4,5},tensor<int>{0,8,1,2,5}));
    REQUIRE(unique(tensor_type{{{1,3,4,3,1},{5,1,5,2,2}},{{3,4,3,3,1},{5,1,3,2,1}},{{1,4,5,4,4},{1,4,1,3,4}}},std::false_type{},std::false_type{},std::true_type{})==
        std::make_tuple(tensor_type{1,2,3,4,5},tensor<int>{9,3,7,7,4})
    );
    REQUIRE(unique(tensor_type{{{1,3,4,3,1},{5,1,5,2,2}},{{3,4,3,3,1},{5,1,3,2,1}},{{1,4,5,4,4},{1,4,1,3,4}}},std::true_type{},std::false_type{},std::true_type{})==
        std::make_tuple(tensor_type{1,2,3,4,5},tensor<int>{0,8,1,2,5},tensor<int>{9,3,7,7,4})
    );
    REQUIRE(unique(tensor_type{{{1,3,4,3,1},{5,1,5,2,2}},{{3,4,3,3,1},{5,1,3,2,1}},{{1,4,5,4,4},{1,4,1,3,4}}},std::false_type{},std::true_type{},std::false_type{})==
        std::make_tuple(tensor_type{1,2,3,4,5},tensor<int>{0,2,3,2,0,4,0,4,1,1,2,3,2,2,0,4,0,2,1,0,0,3,4,3,3,0,3,0,2,3})
    );
    REQUIRE(unique(tensor_type{{{1,3,4,3,1},{5,1,5,2,2}},{{3,4,3,3,1},{5,1,3,2,1}},{{1,4,5,4,4},{1,4,1,3,4}}},std::true_type{},std::true_type{},std::true_type{})==
        std::make_tuple(tensor_type{1,2,3,4,5},tensor<int>{0,8,1,2,5},tensor<int>{0,2,3,2,0,4,0,4,1,1,2,3,2,2,0,4,0,2,1,0,0,3,4,3,3,0,3,0,2,3},tensor<int>{9,3,7,7,4})
    );
}

TEST_CASE("test_sort_search_unique_exception","[test_sort_search]")
{
    using value_type = double;
    using tensor_type = gtensor::tensor<value_type>;
    using gtensor::value_error;
    using gtensor::unique;

    REQUIRE_THROWS_AS(unique(tensor_type(1),std::false_type{},std::false_type{},std::false_type{},1),value_error);
    REQUIRE_THROWS_AS(unique(tensor_type{1,2,3,4,5},std::false_type{},std::false_type{},std::false_type{},1),value_error);
    REQUIRE_THROWS_AS(unique(tensor_type{{1,2,3},{4,5,6}},std::false_type{},std::false_type{},std::false_type{},2),value_error);
}

