#include <tuple>
#include "catch.hpp"
#include "tensor.hpp"
#include "impl_walker_base.hpp"

namespace test_walker_{

using gtensor::tensor;
using gtensor::config::default_config;
using gtensor::stensor_impl_base;
using gtensor::walker;
using gtensor::walker_impl_base;
using gtensor::storage_walker_impl;

template<typename ValT, template<typename> typename Cfg>
struct storage_walker_test_tensor : public tensor<ValT,Cfg>{
    using base_type = tensor<ValT,Cfg>;
    using tensor::tensor;
    storage_walker_test_tensor(const base_type& base):
        base_type{base}
    {}
    storage_walker_impl<ValT,Cfg> create_native_walker()const{
        return static_cast<stensor_impl_base<ValT,Cfg>*>(get_impl().get())->create_walker();
    }
    walker<ValT,Cfg> create_walker()const{
        return std::make_unique<storage_walker_impl<ValT,Cfg>>(create_native_walker());
    }
};


//make 3d stensor with data {{{1,2,3},{4,5,6}}}
template<typename ValT>
struct stensor_maker{
    using value_type = ValT;
    using tensor_type = storage_walker_test_tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{{{1,2,3},{4,5,6}}};}
};
// //make trivial broadcast expression with data {{{1,2,3},{4,5,6}}}
// template<typename ValT>
// struct trivial_expression_maker{
//     using value_type = ValT;
//     using tensor_type = test_tensor<ValT,default_config>;
//     tensor_type operator()(){return tensor_type{{{-1,-1,-1},{-1,-1,-1}}} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{1,1,1},{4,4,4}}};}
// };
// //make expression with data {{{1,2,3},{4,5,6}}}
// template<typename ValT>
// struct not_trivial_expression_maker{
//     using value_type = ValT;
//     using tensor_type = test_tensor<ValT,default_config>;
//     tensor_type operator()(){return tensor_type{2} * tensor_type{-1,-1,-1} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{0,0,0},{3,3,3}}} + tensor_type{5,5,5} - tensor_type{3} ;}
// };
// //make expression with trivial subtree data {{{1,2,3},{4,5,6}}}
// template<typename ValT>
// struct trivial_subtree_expression_maker{
//     using value_type = ValT;
//     using tensor_type = test_tensor<ValT,default_config>;
//     tensor_type operator()(){return tensor_type{2} * tensor_type{-1,-1,-1} + (tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{0,0,0},{3,3,3}}}) + tensor_type{5,5,5} - tensor_type{3} ;}
// };
// //make view slice with data {{{1,2,3},{4,5,6}}}
// template<typename ValT>
// struct view_slice_maker{
//     using value_type = ValT;
//     using tensor_type = test_tensor<ValT,default_config>;
//     typename default_config<ValT>::nop_type nop;
//     tensor_type operator()(){return tensor_type{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{1,0,2,0,3,0},{4,0,5,0,6,0}}}({{1,2},{},{nop,nop,2}});}
// };
// //make view transpose with data {{{1,2,3},{4,5,6}}}
// template<typename ValT>
// struct view_transpose_maker{
//     using value_type = ValT;
//     using tensor_type = test_tensor<ValT,default_config>;
//     tensor_type operator()(){return tensor_type{{{1},{4}},{{2},{5}},{{3},{6}}}.transpose();}
// };
// //make view subdim with data {{{1,2,3},{4,5,6}}}
// template<typename ValT>
// struct view_subdim_maker{
//     using value_type = ValT;
//     using tensor_type = test_tensor<ValT,default_config>;
//     tensor_type operator()(){return tensor_type{{{{0,0,0},{0,0,0}}},{{{1,2,3},{4,5,6}}}}(1);}
// };
// //make view reshape with data {{{1,2,3},{4,5,6}}}
// template<typename ValT>
// struct view_reshape_maker{
//     using value_type = ValT;
//     using tensor_type = test_tensor<ValT,default_config>;
//     tensor_type operator()(){return tensor_type{{{1},{2},{3}},{{4},{5},{6}}}.reshape(1,2,3);}
// };


}   //end of namespace test_walker_


TEMPLATE_TEST_CASE("test_walker","test_walker",
                    test_walker_::stensor_maker<float>
                    // test_walker_::trivial_expression_maker<float>,
                    // test_walker_::trivial_subtree_expression_maker<float>,
                    // test_walker_::not_trivial_expression_maker<float>,
                    // test_walker_::view_slice_maker<float>,
                    // test_walker_::view_transpose_maker<float>,
                    // test_walker_::view_subdim_maker<float>,
                    // test_walker_::view_reshape_maker<float>
                    ){
    using value_type = typename TestType::value_type;
    using test_type = std::tuple<value_type,value_type>;
    using gtensor::config::default_config;
    using config_type = default_config<value_type>;
    using tensor_type = gtensor::tensor<value_type, default_config>;

    auto test_data = GENERATE(
        test_type{*TestType{}().create_walker(), value_type{1}},
        test_type{*TestType{}().create_walker().walk(0,1), value_type{1}},
        test_type{*TestType{}().create_walker().walk(1,1), value_type{4}},
        test_type{*TestType{}().create_walker().walk(2,1), value_type{2}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(2,2), value_type{6}},
        test_type{*TestType{}().create_walker().walk(3,1).walk(2,2), value_type{3}},
        test_type{*TestType{}().create_walker().walk(1,1).reset(1), value_type{1}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(2,2).reset(0), value_type{6}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(2,2).reset(3), value_type{6}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(2,2).reset(1), value_type{3}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(2,2).reset(2), value_type{4}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(2,2).reset(), value_type{1}}
    );
    auto deref = std::get<0>(test_data);
    auto expected_deref = std::get<1>(test_data);
    REQUIRE(deref == expected_deref);
}