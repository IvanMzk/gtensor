#include <tuple>
#include "catch.hpp"
#include "tensor.hpp"
#include "walker_base.hpp"

namespace test_walker_{

using gtensor::tensor;
using gtensor::config::default_config;
using gtensor::storing_base;
using gtensor::tensor_base;
using gtensor::walker;
using gtensor::walker_base;
using gtensor::storage_walker_polymorphic;
using gtensor::storage_walker;
using gtensor::evaluating_trivial_walker;
using gtensor::evaluating_indexer;
using gtensor::viewing_evaluating_walker;

template<typename ValT, template<typename> typename Cfg>
struct storage_walker_test_tensor : public tensor<ValT,Cfg>{
    using base_type = tensor<ValT,Cfg>;
    using tensor::tensor;
    storage_walker_test_tensor(const base_type& base):
        base_type{base}
    {}    
    storage_walker<ValT,Cfg> create_native_walker()const{
        return get_impl()->as_storing()->create_walker();
    }
    walker<ValT,Cfg> create_walker()const{
        return std::make_unique<storage_walker_polymorphic<ValT,Cfg>>(create_native_walker());
    }
};

template<typename ValT, template<typename> typename Cfg>
struct trivial_walker_test_tensor : public tensor<ValT,Cfg>{
    using base_type = tensor<ValT,Cfg>;
    using tensor::tensor;
    trivial_walker_test_tensor(const base_type& base):
        base_type{base}
    {}    
    evaluating_trivial_walker<ValT,Cfg> create_native_walker()const{
        return get_impl()->as_evaluating_trivial()->create_walker();
    }
    walker<ValT,Cfg> create_walker()const{
        return std::make_unique<evaluating_trivial_walker<ValT,Cfg>>(create_native_walker());
    }
};

template<typename ValT, template<typename> typename Cfg>
struct evaluating_walker_test_tensor : public tensor<ValT,Cfg>{
    using base_type = tensor<ValT,Cfg>;
    using tensor::tensor;
    evaluating_walker_test_tensor(const base_type& base):
        base_type{base}
    {}        
    walker<ValT,Cfg> create_walker()const{
        return get_impl()->as_evaluating()->create_walker();
    }
    evaluating_indexer<ValT,Cfg> create_storage()const{
        return get_impl()->as_evaluating()->create_storage();
    }
};

template<typename ValT, template<typename> typename Cfg>
struct view_expression_walker_test_tensor : public tensor<ValT,Cfg>{
    using base_type = tensor<ValT,Cfg>;
    using tensor::tensor;
    view_expression_walker_test_tensor(const base_type& base):
        base_type{base}
    {}
    viewing_evaluating_walker<ValT,Cfg> create_native_walker()const{
        std::cout<<std::endl<<"viewing_evaluating_walker<ValT,Cfg> create_native_walker()const{";
        std::cout<<std::endl<<get_impl()->as_viewing_evaluating();
        auto w = get_impl()->as_viewing_evaluating()->create_walker();
        std::cout<<std::endl<<"viewing_evaluating_walker<ValT,Cfg> create_native_walker()const{";
        return w;
    }
    // viewing_evaluating_walker<ValT,Cfg> create_native_walker()const{
    //     return get_impl()->as_viewing_evaluating()->create_walker();
    // }
    walker<ValT,Cfg> create_walker()const{
        return std::make_unique<viewing_evaluating_walker<ValT,Cfg>>(create_native_walker());
    }    
};


//make 3d stensor with data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct stensor_maker{
    using value_type = ValT;
    using tensor_type = TestTensorT;    
    tensor_type operator()(){return tensor_type{{{1,2,3},{4,5,6}}};}
};
//make expression with data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct not_trivial_expression_maker{
    using value_type = ValT;
    using tensor_type = TestTensorT;
    //evaluating_walker_test_tensor<ValT,default_config>;
    //tensor_type operator()(){return tensor_type{2} * tensor_type{-1,-1,-1} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{0,0,0},{3,3,3}}} + tensor_type{5,5,5} - tensor_type{3} ;}
    tensor_type operator()(){return tensor_type{{{0},{3}}} + tensor_type{1,2,3};}
};
//make expression with trivial subtree data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct trivial_subtree_expression_maker{
    using value_type = ValT;
    using tensor_type = TestTensorT;
    //evaluating_walker_test_tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{2} * tensor_type{-1,-1,-1} + (tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{0,0,0},{3,3,3}}}) + tensor_type{5,5,5} - tensor_type{3} ;}
};
//make trivial broadcast expression with data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct trivial_expression_maker_ewalker{
    using value_type = ValT;
    using tensor_type = TestTensorT;
    //evaluating_walker_test_tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{{{-1,-1,-1},{-1,-1,-1}}} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{1,1,1},{4,4,4}}};}
};
//make trivial broadcast expression with data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct trivial_expression_maker_trivial_walker{
    using value_type = ValT;
    using tensor_type = TestTensorT;
    //trivial_walker_test_tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{{{-1,-1,-1},{-1,-1,-1}}} + tensor_type{{{1,2,3},{1,2,3}}} + tensor_type{{{1,1,1},{4,4,4}}};}
};
//make view slice of stensor with data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct view_slice_of_stensor_maker{
    using value_type = ValT;
    using tensor_type = TestTensorT;
    //storage_walker_test_tensor<ValT,default_config>;
    typename default_config<ValT>::nop_type nop;
    tensor_type operator()(){return tensor_type{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{1,0,2,0,3,0},{4,0,5,0,6,0}}}({{1,2},{},{nop,nop,2}});}
};
//make view slice of non trivial expression with data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct view_slice_of_expression_maker{
    using value_type = ValT;
    using tensor_type = TestTensorT;
    //view_expression_walker_test_tensor<ValT,default_config>;
    typename default_config<ValT>::nop_type nop;    
    tensor_type operator()(){
        auto e = tensor_type{2} * tensor_type{{1,1,1,1,1,1}} + tensor_type{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{1,0,2,0,3,0},{4,0,5,0,6,0}}} - tensor_type{{{3,3,3,3,3,3}}} + tensor_type{1};
        return e({{1,2},{},{nop,nop,2}});        
    }
};
//make view slice of view of non trivial expression with data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct view_view_slice_of_expression_maker{
    using value_type = ValT;
    using tensor_type = TestTensorT;
    //view_expression_walker_test_tensor<ValT,default_config>;
    typename default_config<ValT>::nop_type nop;    
    tensor_type operator()(){
        auto e = tensor_type{2} * tensor_type{{1,1,1,1,1,1}} + tensor_type{{{0,0,0,0,0,0},{0,0,0,0,0,0}},{{3,0,2,0,1,0},{6,0,5,0,4,0}}} - tensor_type{{{3,3,3,3,3,3}}} + tensor_type{1};
        return e({{1,2},{},{nop,nop,2}})({{},{},{nop,nop,-1}});
    }
};
//make view transpose of stensor with data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct view_transpose_of_stensor_maker{
    using value_type = ValT;
    using tensor_type = TestTensorT;
    //storage_walker_test_tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{{{1},{4}},{{2},{5}},{{3},{6}}}.transpose();}
};
//make view subdim of stensor with data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct view_subdim_of_stensor_maker{
    using value_type = ValT;
    using tensor_type = TestTensorT;
    //storage_walker_test_tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{{{{0,0,0},{0,0,0}}},{{{1,2,3},{4,5,6}}}}(1);}
};
//make view reshape of stensor with data {{{1,2,3},{4,5,6}}}
template<typename ValT, typename TestTensorT>
struct view_reshape_of_stensor_maker{
    using value_type = ValT;
    using tensor_type = TestTensorT;
    //storage_walker_test_tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{{{1},{2},{3}},{{4},{5},{6}}}.reshape(1,2,3);}
};


}   //end of namespace test_walker_


TEMPLATE_TEST_CASE("test_walker","test_walker",
                    // (test_walker_::stensor_maker<float, test_walker_::storage_walker_test_tensor<float, test_walker_::default_config>>),
                    // (test_walker_::not_trivial_expression_maker<float, test_walker_::evaluating_walker_test_tensor<float, test_walker_::default_config>>),
                    // (test_walker_::trivial_subtree_expression_maker<float, test_walker_::evaluating_walker_test_tensor<float, test_walker_::default_config>>),
                    // (test_walker_::trivial_expression_maker_ewalker<float, test_walker_::evaluating_walker_test_tensor<float, test_walker_::default_config>>),
                    // (test_walker_::trivial_expression_maker_trivial_walker<float, test_walker_::trivial_walker_test_tensor<float, test_walker_::default_config>>),
                    // (test_walker_::view_slice_of_stensor_maker<float, test_walker_::storage_walker_test_tensor<float, test_walker_::default_config>>),
                    // (test_walker_::view_slice_of_expression_maker<float, test_walker_::view_expression_walker_test_tensor<float, test_walker_::default_config>>),
                    (test_walker_::view_view_slice_of_expression_maker<float, test_walker_::view_expression_walker_test_tensor<float, test_walker_::default_config>>)
                    // (test_walker_::view_transpose_of_stensor_maker<float, test_walker_::storage_walker_test_tensor<float, test_walker_::default_config>>),
                    // (test_walker_::view_subdim_of_stensor_maker<float, test_walker_::storage_walker_test_tensor<float, test_walker_::default_config>>),
                    // (test_walker_::view_reshape_of_stensor_maker<float, test_walker_::storage_walker_test_tensor<float, test_walker_::default_config>>)                    
                    ){
    using value_type = typename TestType::value_type;
    using test_type = std::tuple<value_type,value_type>;
    using gtensor::config::default_config;
    using config_type = default_config<value_type>;
    using tensor_type = gtensor::tensor<value_type, default_config>;    
    
    auto test_data = GENERATE(
        test_type{*TestType{}().create_walker(), value_type{1}},
        test_type{*TestType{}().create_walker().walk(2,1), value_type{1}},
        test_type{*TestType{}().create_walker().walk(1,1), value_type{4}},
        test_type{*TestType{}().create_walker().walk(0,1), value_type{2}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(0,2), value_type{6}},
        test_type{*TestType{}().create_walker().walk(3,1).walk(0,2), value_type{3}},
        test_type{*TestType{}().create_walker().walk(1,1).reset(1), value_type{1}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(0,2).reset(2), value_type{6}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(0,2).reset(3), value_type{6}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(0,2).reset(1), value_type{3}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(0,2).reset(0), value_type{4}},
        test_type{*TestType{}().create_walker().walk(1,1).walk(0,2).reset(), value_type{1}}
    );
    auto deref = std::get<0>(test_data);
    auto expected_deref = std::get<1>(test_data);
    REQUIRE(deref == expected_deref);
}

// TEMPLATE_TEST_CASE("test_evaluating_indexer","test_walker",
//                     (test_walker_::not_trivial_expression_maker<float, test_walker_::evaluating_walker_test_tensor<float, test_walker_::default_config>>),
//                     (test_walker_::trivial_subtree_expression_maker<float, test_walker_::evaluating_walker_test_tensor<float, test_walker_::default_config>>),
//                     (test_walker_::trivial_expression_maker_ewalker<float, test_walker_::evaluating_walker_test_tensor<float, test_walker_::default_config>>)
//                     ){
//     using value_type = typename TestType::value_type;
//     using test_type = std::tuple<value_type,value_type>;

//     //0result,1expected
//     auto test_data = GENERATE(
//         test_type{TestType{}().create_storage()[0],value_type{1}},
//         test_type{TestType{}().create_storage()[5],value_type{6}},
//         test_type{TestType{}().create_storage()[1],value_type{2}},
//         test_type{TestType{}().create_storage()[2],value_type{3}},
//         test_type{TestType{}().create_storage()[4],value_type{5}},
//         test_type{TestType{}().create_storage()[3],value_type{4}}
//     );
//     auto result = std::get<0>(test_data);
//     auto expected = std::get<1>(test_data);
//     REQUIRE(result == expected);
// }
