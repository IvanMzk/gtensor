#include "catch.hpp"
#include "gtensor.hpp"


TEST_CASE("test_const_impl","[test_const_impl]"){

    using gtensor::tensor;
    using value_type = int;
    std::vector<int> expected{1,2,3,4,5};
    tensor<value_type> t({5}, expected.begin(), expected.end());

    std::vector<int> expected_e{2,4,6,8,10};
    auto e = t+t;
    auto it = e.begin();
    auto it_end = e.end();

    it+=4;
    REQUIRE(*it == 10);
    it-=4;
    REQUIRE(*it == 2);
    ++it;
    REQUIRE(*it == 4);

    // auto it_end1 = e.rend();
    // auto it_end2 = e.rend();

    // --it_end1;
    // it_end2+=-1;
    // std::cout<<std::endl<<*it_end1<<" "<<*it_end2;

    // it_end1+=-1;
    // --it_end2;
    // //it_end2+=-1;
    // std::cout<<std::endl<<*it_end1<<" "<<*it_end2;

    // --it_end1;
    // it_end2+=-1;
    // std::cout<<std::endl<<*it_end1<<" "<<*it_end2;

    // it_end1+=-1;
    // --it_end2;
    // std::cout<<std::endl<<*it_end1<<" "<<*it_end2;



    // REQUIRE(it_end - it == t.size());
    // REQUIRE(std::distance(it,it_end) == t.size());

    // REQUIRE(std::equal(e.rbegin(),e.rend(),expected_e.rbegin()));
    // REQUIRE(it == it);

    // REQUIRE(*it == 10);
    // --it_end;
    // REQUIRE(*it_end == 2);
    // ++it;
    // REQUIRE(*it == 8);
    // --it_end;
    // REQUIRE(*it_end == 4);

    // auto v = t({{{},{},2}});
    // std::vector<int> expected_v{1,3,5};
    // auto vit = v.rbegin();
    // auto vit_end = v.rend();
    // REQUIRE(*vit == *expected_v.rbegin());
    // REQUIRE(std::distance(vit,vit_end) == v.size());
    // REQUIRE(std::equal(v.rbegin(),v.rend(),expected_v.rbegin()));


    // auto ve = e({{{},{},2}});
    // std::vector<int> expected_ve{2,6,10};
    // auto veit = ve.rbegin();
    // auto veit_end = ve.rend();
    // REQUIRE(*veit == *expected_ve.rbegin());

    // auto ve1 = e.reshape(5,1);
    // std::vector<int> expected_ve1{2,4,6,8,10};
    // auto ve1it = ve1.rbegin();
    // auto ve1it_end = ve1.rend();
    // REQUIRE(std::equal(ve1.rbegin(),ve1.rend(),expected_ve1.rbegin()));
}