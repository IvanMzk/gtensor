#include "catch.hpp"
#include "impl_multiindex_iterator.hpp"
#include "tensor.hpp"

namespace test_multiindex_iterator_{

using gtensor::tensor;
using gtensor::config::default_config;
using gtensor::multiindex_iterator_impl;
using gtensor::walker;

template<typename ValT, template<typename> typename Cfg>
struct test_tensor : public tensor<ValT,Cfg>{
    using base_type = tensor<ValT,Cfg>;
    using iterator_type = multiindex_iterator_impl<ValT,default_config,walker<ValT,default_config>>;
    using tensor::tensor;
    test_tensor(const base_type& base):
        base_type{base}
    {}
    auto create_walker()const{return get_impl()->create_walker();}
    auto& shape()const{return get_impl()->shape();}
    auto& strides()const{return get_impl()->strides();}
    auto begin()const{return iterator_type{create_walker(),shape(),strides()};}
    auto end()const{return iterator_type{create_walker(),shape(),strides(),size()};}
};

//make 3d stensor with data {{{1,2,3},{4,5,6}}}
template<typename ValT>
struct stensor_maker{
    using value_type = ValT;
    using tensor_type = test_tensor<ValT,default_config>;
    tensor_type operator()(){return tensor_type{{{1,2,3},{4,5,6}}};}
};

}   //end of namespace test_multiindex_iterator_



TEST_CASE("test_multiindex_iterator","[test_multiindex_iterator]"){
    using value_type = float;
    using gtensor::config::default_config;
    using test_tensor_type = test_multiindex_iterator_::test_tensor<value_type, default_config>;
    SECTION("test_iter_deref"){
        using test_type = std::tuple<value_type, value_type>;
        //0deref,1expected_deref
        auto test_data = GENERATE(
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin(), value_type{1}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin().operator++(), value_type{2}},                
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin().operator++().operator++(), value_type{3}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin().operator++().operator++().operator++(), value_type{4}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin().operator++().operator++().operator++().operator++(), value_type{5}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin().operator++().operator++().operator++().operator++().operator++(), value_type{6}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin().operator++().operator++().operator++().operator++().operator++().operator--(), value_type{5}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin().operator++().operator++().operator++().operator++().operator++().operator--().operator--(), value_type{4}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin().operator++().operator++().operator++().operator++().operator++().operator--().operator--().operator--(), value_type{3}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin().operator++().operator++().operator++().operator++().operator++().operator--().operator--().operator--().operator--(), value_type{2}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.begin().operator++().operator++().operator++().operator++().operator++().operator--().operator--().operator--().operator--().operator--(), value_type{1}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--(), value_type{6}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--().operator--(), value_type{5}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--().operator--().operator--(), value_type{4}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--().operator--().operator--().operator--(), value_type{3}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--().operator--().operator--().operator--().operator--(), value_type{2}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--().operator--().operator--().operator--().operator--().operator--(), value_type{1}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--().operator--().operator--().operator--().operator--().operator--().operator++(), value_type{2}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--().operator--().operator--().operator--().operator--().operator--().operator++().operator++(), value_type{3}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--().operator--().operator--().operator--().operator--().operator--().operator++().operator++().operator++(), value_type{4}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--().operator--().operator--().operator--().operator--().operator--().operator++().operator++().operator++().operator++(), value_type{5}},
            test_type{*test_tensor_type{{{1,2,3},{4,5,6}}}.end().operator--().operator--().operator--().operator--().operator--().operator--().operator++().operator++().operator++().operator++().operator++(), value_type{6}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.begin()+0), value_type{1}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.begin()+1), value_type{2}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.begin()+2), value_type{3}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.begin()+3), value_type{4}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.begin()+4), value_type{5}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.begin()+5), value_type{6}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.end()-1), value_type{6}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.end()-2), value_type{5}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.end()-3), value_type{4}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.end()-4), value_type{3}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.end()-5), value_type{2}},
            test_type{*(test_tensor_type{{{1,2,3},{4,5,6}}}.end()-6), value_type{1}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin()[0], value_type{1}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin()[1], value_type{2}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin()[2], value_type{3}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin()[3], value_type{4}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin()[4], value_type{5}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin()[5], value_type{6}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end()[-1], value_type{6}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end()[-2], value_type{5}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end()[-3], value_type{4}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end()[-4], value_type{3}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end()[-5], value_type{2}},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end()[-6], value_type{1}}        
        );
        auto deref = std::get<0>(test_data);
        auto expected_deref = std::get<1>(test_data);
        REQUIRE(deref == expected_deref);
    }
    SECTION("test_iter_cmp"){
        using test_type = std::tuple<bool,bool>;
        //0cmp_result,1expected_cmp_result
        auto test_data = GENERATE(
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin() > test_tensor_type{{{1,2,3},{4,5,6}}}.end(), false},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end() > test_tensor_type{{{1,2,3},{4,5,6}}}.begin(), true},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin() < test_tensor_type{{{1,2,3},{4,5,6}}}.end(), true},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end() < test_tensor_type{{{1,2,3},{4,5,6}}}.begin(), false},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end() > test_tensor_type{{{1,2,3},{4,5,6}}}.end(), false},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin() > test_tensor_type{{{1,2,3},{4,5,6}}}.begin(), false},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end() < test_tensor_type{{{1,2,3},{4,5,6}}}.end(), false},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin() < test_tensor_type{{{1,2,3},{4,5,6}}}.begin(), false},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end() >= test_tensor_type{{{1,2,3},{4,5,6}}}.end(), true},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin() >= test_tensor_type{{{1,2,3},{4,5,6}}}.begin(), true},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.end() <= test_tensor_type{{{1,2,3},{4,5,6}}}.end(), true},
            test_type{test_tensor_type{{{1,2,3},{4,5,6}}}.begin() <= test_tensor_type{{{1,2,3},{4,5,6}}}.begin(), true}
        );
        auto cmp_result = std::get<0>(test_data);
        auto expected_cmp_result = std::get<1>(test_data);
        REQUIRE(cmp_result == expected_cmp_result);
    }
}

